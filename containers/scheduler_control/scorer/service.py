from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from psycopg2.extras import execute_values
from sqlalchemy.orm import sessionmaker

from libs.scoring.golden_discovery_runtime import GoldenDiscoveryRuntimeScorer


logger = logging.getLogger("golden_discovery_ranker_v1")

# Ranker scores are clipped to [0, 1]; url_score > 1 is a manually assigned
# priority (golden_inject = 3.0, wiki_pageview_inject = 1 + views/max) that the
# ranker must not overwrite. Both inject scripts stamp url_score_updated_at, so
# without this the steering TTL refresh would rescore them after a few days.
MANUAL_PRIORITY_EXCLUDED = "(url_score IS NULL OR url_score <= 1.0)"


@dataclass(frozen=True)
class GoldenDiscoveryRankerConfig:
    total_shards: int
    num_workers: int
    worker_id: int
    batch_size: int
    scan_interval_sec: int
    max_batches_per_shard: int
    # Domain-priority steering: when enabled, each batch only picks URLs from
    # domains with `domain_state.domain_score > 0` (i.e. domains that have
    # appeared in a golden batch). Ordered by domain_score DESC so higher-tier
    # golden domains are scored first. When disabled, the scorer keeps the
    # legacy first_seen-ASC behavior.
    domain_priority_steering_enabled: bool = False
    # v2 continuous re-scoring (steering path only). The v2 ranker's features
    # (inlink_count_*, anchor_text) change over time, so url_score is no longer
    # a write-once value — it must be refreshed.
    #   rescore_ttl_days: re-pick a URL once its score is older than this many
    #     days (in addition to never-scored rows). Sized so one full pass over
    #     the steering-eligible pool finishes well inside the TTL; see deploy
    #     notes. 0 disables TTL refresh (write-once, never re-score).
    #   min_age_days: do NOT score a URL until it has been in the DB this long,
    #     letting inlinks/anchors accumulate first. Younger rows keep their v1
    #     inline score until they age in. 0 disables the gate.
    rescore_ttl_days: int = 5
    min_age_days: int = 3


class GoldenDiscoveryRankerService:
    def __init__(
        self,
        cfg: GoldenDiscoveryRankerConfig,
        Session: sessionmaker,
        scorer: GoldenDiscoveryRuntimeScorer,
    ):
        self.cfg = cfg
        self.Session = Session
        self.scorer = scorer
        # Steering TTL path: per-shard {domain_id: newest url_score_updated_at
        # this worker has rescored}.  Process-local and lost on restart; a
        # missing or stale watermark only costs extra tombstone reads.
        self._watermarks: dict[int, dict[int, datetime]] = {}

    @staticmethod
    def _table(shard_id: int) -> str:
        return f"url_state_current_{shard_id:03d}"

    def _shard_ids(self) -> list[int]:
        """Order in which this worker visits shards on each run_once.

        Every worker visits every shard. Workers stagger their starting
        offset by `total_shards // num_workers` so the four (or N) of
        them don't all queue up on shard 0 first — but the static
        partition is gone, so a worker that finishes its starting
        section keeps walking and picks up shards that another worker
        would previously have owned. URL-level FOR UPDATE SKIP LOCKED
        inside `_score_batch` handles the resulting concurrent claims
        on the same shard.

        Replaces the prior static partition (worker N saw shards
        [N, N + num_workers, ...]) which could not reassign work when
        one worker drew a disproportionate share of the very large
        (256M-row) shards while other workers sat idle after draining
        their own assignment.
        """
        if self.cfg.num_workers <= 0:
            return list(range(self.cfg.total_shards))
        step = max(1, self.cfg.total_shards // self.cfg.num_workers)
        offset = (self.cfg.worker_id * step) % self.cfg.total_shards
        return [
            (offset + i) % self.cfg.total_shards
            for i in range(self.cfg.total_shards)
        ]

    def _steering_unscored_sql(self, table: str) -> str:
        # Never-scored rows of golden-tier domains, as a LATERAL top-N per
        # domain.  WHY this shape and not a flat `JOIN domain_state ...
        # ORDER BY d.domain_score DESC, u.url_score_updated_at`: the flat
        # form forces a blocking Sort, because the lead ORDER BY key
        # (domain_score) lives on domain_state and has only ~3 distinct
        # values, so within a tier url_score_updated_at must be merged
        # ACROSS all domains — no index can produce that.  The Sort blocks
        # the LIMIT, so the planner materialises every matching row (in the
        # backlog regime ~all rows) via a Seq Scan.  Measured on a 64M-row
        # shard the flat plan cost ~11.5M and ignored the rescore index.
        #
        # The LATERAL drives from domain_state (golden domains for this
        # shard, via idx_domain_state_shard_score, ordered domain_score
        # DESC) and for each domain reads idx_..._golden_discovery_v2_rescore
        # (domain_id, url_score_updated_at ASC NULLS FIRST).  `IS NULL` is a
        # range on that index (the NULLS FIRST prefix), so the scan stops at
        # the first scored row.
        #
        # `d.shard_id = %(shard_id)s` is required AND correct: every domain
        # in url_state_current_{shard} has domain_state.shard_id=shard, and
        # the equality lets the shard_score index yield domain_score DESC
        # order (its lead column is shard_id).
        #
        # No FOR UPDATE here: a locking LATERAL locks up to LIMIT rows in
        # EVERY golden domain before the outer LIMIT discards most of them.
        # On YB each lock is a write, so that dominated the batch (lab: 11.8s
        # with the lock vs 0.3s without, 10 domains).  _lock_due_sql locks
        # only the chosen candidates.
        return f"""
            SELECT u.url, u.inlink_count_approx, u.inlink_count_external,
                   u.anchor_text, u.domain_id, u.url_score_updated_at
            FROM domain_state d
            CROSS JOIN LATERAL (
                SELECT url, inlink_count_approx, inlink_count_external,
                       anchor_text, domain_id, url_score_updated_at
                FROM {table} u
                WHERE u.domain_id = d.domain_id
                  AND u.should_crawl = TRUE
                  AND u.first_seen < NOW() - make_interval(days => %(min_age_days)s)
                  AND u.url_score_updated_at IS NULL
                  AND {MANUAL_PRIORITY_EXCLUDED}
                LIMIT %(limit)s
            ) u
            WHERE d.shard_id = %(shard_id)s
              AND d.domain_score > 0
            ORDER BY d.domain_score DESC
            LIMIT %(limit)s
            """

    def _steering_stale_sql(self, table: str) -> str:
        # TTL refresh: rows scored before %(ttl_cutoff)s.  Same LATERAL shape
        # as the unscored query, but the due condition is a closed range on
        # the rescore index, [watermark, ttl_cutoff), so YB seeks straight to
        # the domain's due rows and stops at the cutoff.  The previous
        # `IS NULL OR < cutoff` could not become an index bound; the TTL was
        # only a Filter, so every domain with fewer than LIMIT due rows was
        # scanned to the end of its index range (2-3M rows per batch on the
        # big shards, issue #108).
        #
        # The per-domain watermark (see _watermarks) skips the delete
        # tombstones that each rescore leaves at the old index position:
        # every UPDATE moves the entry to the tail of the domain range, so
        # without it each batch re-reads the previous batches' tombstones.
        #
        # ORDER BY must stay `ASC NULLS FIRST` to match the index order,
        # otherwise the planner cannot use the index for the LIMIT.
        return f"""
            SELECT u.url, u.inlink_count_approx, u.inlink_count_external,
                   u.anchor_text, u.domain_id, u.url_score_updated_at
            FROM domain_state d
            LEFT JOIN unnest(
                %(wm_domain_ids)s::bigint[], %(wm_values)s::timestamptz[]
            ) AS w(domain_id, watermark) ON w.domain_id = d.domain_id
            CROSS JOIN LATERAL (
                SELECT url, inlink_count_approx, inlink_count_external,
                       anchor_text, domain_id, url_score_updated_at
                FROM {table} u
                WHERE u.domain_id = d.domain_id
                  AND u.should_crawl = TRUE
                  AND u.first_seen < NOW() - make_interval(days => %(min_age_days)s)
                  AND u.url_score_updated_at >= COALESCE(w.watermark, '-infinity'::timestamptz)
                  AND u.url_score_updated_at < %(ttl_cutoff)s
                  AND {MANUAL_PRIORITY_EXCLUDED}
                ORDER BY u.url_score_updated_at ASC NULLS FIRST
                LIMIT %(limit)s
            ) u
            WHERE d.shard_id = %(shard_id)s
              AND d.domain_score > 0
            ORDER BY d.domain_score DESC,
                     u.url_score_updated_at ASC NULLS FIRST
            LIMIT %(limit)s
            """

    def _lock_due_sql(self, table: str) -> str:
        # Phase 2: lock the chosen candidates by primary key.  SKIP LOCKED
        # leaves rows another worker already claimed; the due condition is
        # re-checked so a row another worker rescored after our candidate
        # read is not scored twice.  ttl_cutoff = datetime.min means only
        # never-scored rows are due (write-once).
        return f"""
            SELECT url, inlink_count_approx, inlink_count_external, anchor_text
            FROM {table}
            WHERE url = ANY(%(urls)s)
              AND should_crawl = TRUE
              AND (url_score_updated_at IS NULL
                   OR url_score_updated_at < %(ttl_cutoff)s)
              AND {MANUAL_PRIORITY_EXCLUDED}
            FOR UPDATE SKIP LOCKED
            """

    def _pick_steering(self, cur, table: str, shard_id: int):
        """Claim up to batch_size due URLs from golden-tier domains.

        Gates: min_age skips URLs younger than min_age_days (they keep the
        v1 inline score; 0 = no-op).  Due rows are never-scored rows first,
        then (if rescore_ttl_days > 0) rows scored before the TTL cutoff.
        They are separate queries because the `IS NULL OR < cutoff`
        disjunction cannot be an index range.  rescore_ttl_days <= 0 =
        write-once.

        Returns (rows, unscored, rescored, new_watermarks, reset_watermarks).
        Watermarks are returned rather than stored so the caller applies
        them only after the transaction commits.
        """
        params = {
            "min_age_days": self.cfg.min_age_days,
            "shard_id": shard_id,
            "limit": self.cfg.batch_size,
            "ttl_cutoff": datetime.min.replace(tzinfo=timezone.utc),
        }
        cur.execute(self._steering_unscored_sql(table), params)
        candidates = cur.fetchall()
        unscored_urls = {row[0] for row in candidates}

        new_watermarks: dict[int, datetime] = {}
        reset_watermarks = False
        remaining = self.cfg.batch_size - len(candidates)
        if self.cfg.rescore_ttl_days > 0 and remaining > 0:
            shard_wm = self._watermarks.get(shard_id, {})
            params.update(
                limit=remaining,
                ttl_cutoff=datetime.now(timezone.utc)
                - timedelta(days=self.cfg.rescore_ttl_days),
                wm_domain_ids=list(shard_wm.keys()),
                wm_values=list(shard_wm.values()),
            )
            cur.execute(self._steering_stale_sql(table), params)
            stale = cur.fetchall()
            candidates.extend(stale)
            # Candidates are an ascending prefix of each domain's due range,
            # so the max timestamp per domain is a safe lower bound for its
            # next batch.  Rows SKIP LOCKED drops below are being scored by
            # the worker holding the lock.
            for row in stale:
                domain_id, updated_at = row[4], row[5]
                prev = new_watermarks.get(domain_id)
                if prev is None or updated_at > prev:
                    new_watermarks[domain_id] = updated_at
            # A short read means every golden domain in the shard is drained
            # above its watermark.  Start the next pass from scratch so rows
            # that fell below a watermark (a lock holder rolled back,
            # should_crawl flipped on, ...) are not lost.
            reset_watermarks = len(stale) < remaining

        if not candidates:
            return [], 0, 0, new_watermarks, reset_watermarks
        cur.execute(
            self._lock_due_sql(table),
            {
                "urls": [row[0] for row in candidates],
                "ttl_cutoff": params["ttl_cutoff"],
            },
        )
        rows = cur.fetchall()
        unscored = sum(1 for row in rows if row[0] in unscored_urls)
        return rows, unscored, len(rows) - unscored, new_watermarks, reset_watermarks

    def _score_batch(self, shard_id: int) -> int:
        table = self._table(shard_id)
        unscored = rescored = 0
        # Watermark changes are applied only after the transaction commits,
        # so a failed batch never advances past rows it did not score.
        new_watermarks: dict[int, datetime] = {}
        reset_watermarks = False

        with self.Session.begin() as sess:
            with sess.connection().connection.cursor() as cur:
                if self.cfg.domain_priority_steering_enabled:
                    fetched, unscored, rescored, new_watermarks, reset_watermarks = (
                        self._pick_steering(cur, table, shard_id)
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT url, inlink_count_approx, inlink_count_external, anchor_text
                        FROM {table}
                        WHERE should_crawl = TRUE
                          AND url_score_updated_at IS NULL
                          AND {MANUAL_PRIORITY_EXCLUDED}
                        ORDER BY first_seen ASC NULLS LAST
                        FOR UPDATE SKIP LOCKED
                        LIMIT %s
                        """,
                        (self.cfg.batch_size,),
                    )
                    fetched = cur.fetchall()
                    unscored = len(fetched)
                if not fetched:
                    if reset_watermarks:
                        self._watermarks.pop(shard_id, None)
                    return 0
                urls = [row[0] for row in fetched]

                # v2 ranker consumes prefetch features (inlink/anchor) via
                # score_many_rows; v1 only needs the URL string. Detect by
                # capability so this path stays drop-in for both.
                if hasattr(self.scorer, "score_many_rows"):
                    recs = [
                        {
                            "url": row[0],
                            "inlink_count_approx": row[1],
                            "inlink_count_external": row[2],
                            "anchor_text": row[3],
                        }
                        for row in fetched
                    ]
                    scores = self.scorer.score_many_rows(recs)
                else:
                    scores = self.scorer.score_many(urls)
                rows = list(zip(urls, scores))

                # Keep score history compact: the ranker refreshes
                # current.url_score in place and uses url_score_updated_at as
                # the only completion bit.
                execute_values(
                    cur,
                    f"""
                    UPDATE {table} AS u
                    SET
                        url_score = v.score::double precision,
                        url_score_updated_at = CURRENT_TIMESTAMP
                    FROM (VALUES %s) AS v(url, score)
                    WHERE u.url = v.url
                    """,
                    rows,
                    page_size=len(rows),
                )
                scored = len(rows)

        if reset_watermarks:
            self._watermarks.pop(shard_id, None)
        elif new_watermarks:
            self._watermarks.setdefault(shard_id, {}).update(new_watermarks)

        logger.info(
            "golden_discovery_ranker_v1.score_batch",
            extra={
                "event": "golden_discovery_ranker_v1.score_batch",
                "worker_id": self.cfg.worker_id,
                "shard_id": shard_id,
                "scored_urls": scored,
                "unscored_urls": unscored,
                "rescored_urls": rescored,
                "watermark_domains": len(self._watermarks.get(shard_id, {})),
                "watermark_reset": reset_watermarks,
            },
        )
        return scored

    def run_once(self) -> dict[str, int]:
        totals = {"scored_urls": 0, "scored_batches": 0}

        for shard_id in self._shard_ids():
            batches = 0
            while batches < self.cfg.max_batches_per_shard:
                count = self._score_batch(shard_id)
                if count == 0:
                    break
                batches += 1
                totals["scored_batches"] += 1
                totals["scored_urls"] += count

        logger.info(
            "golden_discovery_ranker_v1.run_once",
            extra={
                "event": "golden_discovery_ranker_v1.run_once",
                "worker_id": self.cfg.worker_id,
                **totals,
            },
        )
        return totals

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(
                    "golden_discovery_ranker_v1.error",
                    extra={
                        "event": "golden_discovery_ranker_v1.error",
                        "worker_id": self.cfg.worker_id,
                        "error": str(e),
                    },
                )
            time.sleep(self.cfg.scan_interval_sec)
