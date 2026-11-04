"""Tests for the steering-path due query and tombstone-skipping watermarks.

What we're protecting (issue #108):

  * The TTL condition is a closed index range
    `[watermark, ttl_cutoff)` on (domain_id, url_score_updated_at), not
    `IS NULL OR < cutoff`, which YB could only apply as a Filter after
    scanning the whole domain range.
  * Never-scored rows are picked by a separate query, before TTL rows.
  * Each domain's watermark advances to the newest timestamp it rescored,
    is passed to the next batch, and is dropped when the shard drains
    (short read) so rows that fell below it are picked up on the next pass.
  * A failed batch never advances a watermark.
  * Candidate reads take no locks; only the chosen URLs are locked, by
    primary key, with SKIP LOCKED and a re-check of the due condition.

The tests drive `_score_batch` against a scripted fake cursor; no DB or
scorer model is needed.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from containers.scheduler_control.scorer import service as service_mod
from containers.scheduler_control.scorer.service import (
    MANUAL_PRIORITY_EXCLUDED as MANUAL_PRIORITY_GUARD,
    GoldenDiscoveryRankerConfig,
    GoldenDiscoveryRankerService,
)

T0 = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _row(url, domain_id, updated_at):
    return (url, 0, 0, None, domain_id, updated_at)


class _ScriptedCursor:
    """Returns the queued result sets, one per candidate SELECT, in order.

    The phase-2 lock query (`url = ANY`) returns every requested candidate
    except those in `locked`, as if another worker held them.
    """

    def __init__(self, results, locked=()):
        self.results = list(results)
        self.locked = set(locked)
        self.calls = []
        self._last = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self.calls.append((sql, params))
        if "url = ANY" in sql:
            self._last = [
                (u, 0, 0, None) for u in params["urls"] if u not in self.locked
            ]
        else:
            self._last = self.results.pop(0) if self.results else []

    def selects(self, marker):
        return [c for c in self.calls if marker in c[0]]

    def fetchall(self):
        return list(self._last)


class _FakeSession:
    def __init__(self, cursor):
        self._cursor = cursor
        self.connection = lambda: type("C", (), {"connection": self})()

    def begin(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor


class _Scorer:
    def score_many(self, urls):
        return [0.5] * len(urls)


def _service(cursor, *, batch_size=4, ttl=5):
    cfg = GoldenDiscoveryRankerConfig(
        total_shards=256,
        num_workers=1,
        worker_id=0,
        batch_size=batch_size,
        scan_interval_sec=60,
        max_batches_per_shard=1,
        domain_priority_steering_enabled=True,
        rescore_ttl_days=ttl,
        min_age_days=3,
    )
    return GoldenDiscoveryRankerService(
        cfg=cfg, Session=_FakeSession(cursor), scorer=_Scorer()
    )


@mock.patch.object(service_mod, "execute_values")
class SteeringDueQueryTest(unittest.TestCase):
    def test_ttl_is_an_index_range_not_an_or(self, _ev):
        cur = _ScriptedCursor([[], []])
        _service(cur)._score_batch(0)
        unscored_sql, stale_sql = cur.calls[0][0], cur.calls[1][0]
        self.assertIn("u.url_score_updated_at IS NULL", unscored_sql)
        self.assertNotIn(" OR u.url_score_updated_at", unscored_sql)
        self.assertIn("u.url_score_updated_at < %(ttl_cutoff)s", stale_sql)
        self.assertIn(
            "u.url_score_updated_at >= COALESCE(w.watermark, '-infinity'::timestamptz)",
            stale_sql,
        )
        self.assertNotIn("IS NULL OR u.url_score_updated_at", stale_sql)

    def test_ttl_cutoff_is_now_minus_ttl(self, _ev):
        cur = _ScriptedCursor([[], []])
        before = datetime.now(timezone.utc)
        _service(cur, ttl=5)._score_batch(0)
        cutoff = cur.calls[1][1]["ttl_cutoff"]
        self.assertAlmostEqual(
            (before - timedelta(days=5) - cutoff).total_seconds(), 0, delta=5
        )

    def test_unscored_rows_fill_batch_first(self, _ev):
        cur = _ScriptedCursor([[_row(f"u{i}", 1, None) for i in range(4)]])
        self.assertEqual(_service(cur, batch_size=4)._score_batch(0), 4)
        self.assertEqual(cur.selects("ttl_cutoff)s ORDER BY"), [])

    def test_stale_query_gets_remaining_limit(self, _ev):
        cur = _ScriptedCursor([[_row("new", 1, None)], []])
        _service(cur, batch_size=4)._score_batch(0)
        self.assertEqual(cur.calls[1][1]["limit"], 3)

    def test_write_once_skips_stale_query(self, _ev):
        cur = _ScriptedCursor([[_row("new", 1, None)]])
        _service(cur, ttl=0)._score_batch(0)
        self.assertEqual(len(cur.calls), 2)
        self.assertNotIn("ttl_cutoff", cur.calls[0][0])
        # The lock re-check must then admit only never-scored rows.
        self.assertLess(cur.calls[1][1]["ttl_cutoff"], T0 - timedelta(days=3650))


@mock.patch.object(service_mod, "execute_values")
class LockOnlyChosenRowsTest(unittest.TestCase):
    def test_candidate_reads_take_no_locks(self, _ev):
        cur = _ScriptedCursor([[], [_row("a", 1, T0)]])
        _service(cur)._score_batch(0)
        self.assertNotIn("FOR UPDATE", cur.calls[0][0])
        self.assertNotIn("FOR UPDATE", cur.calls[1][0])
        lock_sql, lock_params = cur.calls[2]
        self.assertIn("FOR UPDATE SKIP LOCKED", lock_sql)
        self.assertIn("url_score_updated_at < %(ttl_cutoff)s", lock_sql)
        self.assertIn(MANUAL_PRIORITY_GUARD, lock_sql)
        self.assertEqual(lock_params["urls"], ["a"])

    def test_rows_locked_elsewhere_are_not_scored(self, ev):
        cur = _ScriptedCursor(
            [[_row("n", 1, None)], [_row("a", 1, T0), _row("b", 1, T0)]],
            locked={"a"},
        )
        self.assertEqual(_service(cur)._score_batch(0), 2)
        self.assertEqual(sorted(r[0] for r in ev.call_args[0][2]), ["b", "n"])

    def test_no_candidates_skips_lock_query(self, _ev):
        cur = _ScriptedCursor([[], []])
        _service(cur)._score_batch(0)
        self.assertEqual(cur.selects("url = ANY"), [])


@mock.patch.object(service_mod, "execute_values")
class WatermarkTest(unittest.TestCase):
    def test_full_batch_advances_watermark_per_domain(self, _ev):
        stale = [
            _row("a1", 1, T0),
            _row("b1", 2, T0 + timedelta(hours=1)),
            _row("a2", 1, T0 + timedelta(minutes=5)),
            _row("b2", 2, T0 + timedelta(hours=2)),
        ]
        cur = _ScriptedCursor([[], stale, [], []])
        svc = _service(cur, batch_size=4)
        svc._score_batch(7)
        self.assertEqual(
            svc._watermarks[7],
            {1: T0 + timedelta(minutes=5), 2: T0 + timedelta(hours=2)},
        )
        svc._score_batch(7)
        params = cur.calls[4][1]
        self.assertEqual(
            dict(zip(params["wm_domain_ids"], params["wm_values"])),
            {1: T0 + timedelta(minutes=5), 2: T0 + timedelta(hours=2)},
        )

    def test_short_read_resets_shard_watermarks(self, _ev):
        cur = _ScriptedCursor([[], [_row(f"u{i}", 1, T0) for i in range(4)], [], [_row("x", 1, T0)]])
        svc = _service(cur, batch_size=4)
        svc._score_batch(7)
        self.assertIn(7, svc._watermarks)
        svc._watermarks[8] = {9: T0}
        self.assertEqual(svc._score_batch(7), 1)
        self.assertNotIn(7, svc._watermarks)
        self.assertIn(8, svc._watermarks, "other shards keep their watermarks")

    def test_empty_read_resets_shard_watermarks(self, _ev):
        cur = _ScriptedCursor([[], []])
        svc = _service(cur)
        svc._watermarks[7] = {1: T0}
        self.assertEqual(svc._score_batch(7), 0)
        self.assertNotIn(7, svc._watermarks)

    def test_failed_update_does_not_advance_watermark(self, ev):
        ev.side_effect = RuntimeError("write conflict")
        cur = _ScriptedCursor([[], [_row(f"u{i}", 1, T0) for i in range(4)]])
        svc = _service(cur, batch_size=4)
        svc._watermarks[7] = {1: T0 - timedelta(days=1)}
        with self.assertRaises(RuntimeError):
            svc._score_batch(7)
        self.assertEqual(svc._watermarks[7], {1: T0 - timedelta(days=1)})


if __name__ == "__main__":
    unittest.main()
