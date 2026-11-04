"""Domain health feedback for the offerer.

The offerer ranks domains by domain_state.domain_score only, so a golden-tier
domain keeps getting fresh 100-URL batches even when nearly every fetch on it
fails, or when the crawler produces no result for it at all. Measured on
production (2026-10-20..24): domains with an ok-rate below 50% took 4.5% of
all scheduled URLs and produced about a third of all fetch failures; another
0.6% went to domains whose scheduled URLs produced almost no results.

This module reads the trailing per-domain counters in domain_stats_daily and
marks such domains as demoted. The offerer excludes demoted domains from the
peek, except for one probe batch per probe interval so a domain that recovers
is picked up again. Nothing is written to the database.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, NamedTuple

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


logger = logging.getLogger("offerer")


@dataclass(frozen=True)
class DomainHealthConfig:
    enabled: bool = False
    # Trailing window, in days. Scheduled counts cover [today - window_days,
    # yesterday]; results cover [today - window_days, today], so a URL offered
    # late yesterday and fetched today still counts as a result.
    window_days: int = 3
    refresh_sec: int = 900
    # Low ok-rate: ok / (ok + fail) below min_ok_rate over at least min_results
    # results.
    min_results: int = 50
    min_ok_rate: float = 0.2
    # Black hole: (ok + fail) / scheduled below min_response_rate over at least
    # min_scheduled scheduled URLs.
    min_scheduled: int = 1000
    min_response_rate: float = 0.05
    # A demoted domain still gets one batch per probe interval.
    probe_interval_sec: int = 21600
    # If results / scheduled across all domains of this offerer is below this,
    # the pipeline (ingestor, stats) is behind or down, not the domains: no
    # domain is demoted for that refresh.
    pipeline_min_response_rate: float = 0.5


class DomainStats(NamedTuple):
    domain_id: int
    scheduled: int
    fetch_ok: int
    fetch_fail: int


@dataclass(frozen=True)
class HealthSnapshot:
    demoted: dict[int, str]
    pipeline_response_rate: float | None
    pipeline_ok: bool


def classify(stats: Iterable[DomainStats], cfg: DomainHealthConfig) -> HealthSnapshot:
    """Return the domains to demote, with the reason for each."""
    rows = list(stats)
    total_sched = sum(r.scheduled for r in rows)
    total_results = sum(r.fetch_ok + r.fetch_fail for r in rows)
    pipeline_rate = total_results / total_sched if total_sched > 0 else None
    if pipeline_rate is not None and pipeline_rate < cfg.pipeline_min_response_rate:
        return HealthSnapshot(demoted={}, pipeline_response_rate=pipeline_rate, pipeline_ok=False)

    demoted: dict[int, str] = {}
    for r in rows:
        results = r.fetch_ok + r.fetch_fail
        if results >= cfg.min_results and r.fetch_ok / results < cfg.min_ok_rate:
            demoted[r.domain_id] = "low_ok_rate"
        elif r.scheduled >= cfg.min_scheduled and results / r.scheduled < cfg.min_response_rate:
            demoted[r.domain_id] = "no_results"
    return HealthSnapshot(demoted=demoted, pipeline_response_rate=pipeline_rate, pipeline_ok=True)


def make_stats_loader(Session: sessionmaker) -> Callable[[int, int, int], list[DomainStats]]:
    """Read trailing per-domain counters for one offerer's shard range."""

    sql = text("""
    SELECT domain_id,
           COALESCE(SUM(num_scheduled) FILTER (WHERE event_date < CURRENT_DATE), 0) AS scheduled,
           COALESCE(SUM(num_fetch_ok), 0) AS fetch_ok,
           COALESCE(SUM(num_fetch_fail), 0) AS fetch_fail
    FROM domain_stats_daily
    WHERE event_date >= CURRENT_DATE - :window_days
      AND shard_id BETWEEN :shard_start AND :shard_end
    GROUP BY domain_id
    """)

    def load(shard_start: int, shard_end: int, window_days: int) -> list[DomainStats]:
        with Session() as sess:
            rows = sess.execute(sql, {
                "window_days": window_days,
                "shard_start": shard_start,
                "shard_end": shard_end,
            }).fetchall()
        return [
            DomainStats(int(r.domain_id), int(r.scheduled), int(r.fetch_ok), int(r.fetch_fail))
            for r in rows
        ]

    return load


class DomainHealthTracker:
    def __init__(
        self,
        cfg: DomainHealthConfig,
        loader: Callable[[int, int, int], list[DomainStats]],
        clock: Callable[[], float] = time.time,
    ):
        self.cfg = cfg
        self.loader = loader
        self.clock = clock
        self._demoted: dict[int, str] = {}
        self._last_probe: dict[int, float] = {}
        self._next_refresh_at = 0.0

    @property
    def demoted(self) -> dict[int, str]:
        return dict(self._demoted)

    def maybe_refresh(self, shard_start: int, shard_end: int) -> None:
        now = self.clock()
        if now < self._next_refresh_at:
            return
        # Schedule the next attempt first so a failing query is retried after
        # refresh_sec, not on every refill.
        self._next_refresh_at = now + self.cfg.refresh_sec
        try:
            stats = self.loader(shard_start, shard_end, self.cfg.window_days)
        except Exception as e:
            # Keep the previous snapshot; health is an optimisation, never a
            # reason to stop offering.
            logger.error(
                "offer.health_error",
                extra={"event": "offer.health_error", "error": str(e)},
            )
            return

        snap = classify(stats, self.cfg)
        self._demoted = snap.demoted
        self._last_probe = {d: t for d, t in self._last_probe.items() if d in self._demoted}
        reasons: dict[str, int] = {}
        for reason in snap.demoted.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        logger.info(
            "offer.health_refresh",
            extra={
                "event": "offer.health_refresh",
                "domains_seen": len(stats),
                "demoted": len(snap.demoted),
                "demoted_by_reason": reasons,
                "pipeline_ok": snap.pipeline_ok,
                "pipeline_response_rate": (
                    round(snap.pipeline_response_rate, 4)
                    if snap.pipeline_response_rate is not None else None
                ),
            },
        )

    def excluded_now(self) -> set[int]:
        """Demoted domains that are not due for a probe batch."""
        now = self.clock()
        interval = self.cfg.probe_interval_sec
        return {
            d for d in self._demoted
            if now - self._last_probe.get(d, float("-inf")) < interval
        }

    def note_offered(self, domain_ids: Iterable[int]) -> None:
        now = self.clock()
        for d in domain_ids:
            if d in self._demoted:
                self._last_probe[d] = now
