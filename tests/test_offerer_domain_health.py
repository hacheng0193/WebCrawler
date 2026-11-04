from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from containers.scheduler_control.offerer.domain_health import (
    DomainHealthConfig,
    DomainHealthTracker,
    DomainStats,
    classify,
)
from containers.scheduler_control.offerer.selection.base import CandidateDomain
from containers.scheduler_control.offerer.service import (
    OffererConfig,
    OffererDerivation,
    OffererService,
)


CFG = DomainHealthConfig(enabled=True)


class _Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class ClassifyTest(unittest.TestCase):
    def test_healthy_domain_is_not_demoted(self):
        snap = classify([DomainStats(1, 3000, 2700, 300)], CFG)
        self.assertEqual(snap.demoted, {})
        self.assertTrue(snap.pipeline_ok)

    def test_low_ok_rate_is_demoted(self):
        rows = [DomainStats(1, 3000, 2900, 100), DomainStats(2, 1000, 50, 950)]
        snap = classify(rows, CFG)
        self.assertEqual(snap.demoted, {2: "low_ok_rate"})

    def test_low_ok_rate_needs_enough_results(self):
        rows = [DomainStats(1, 3000, 2900, 100), DomainStats(2, 40, 0, 40)]
        self.assertEqual(classify(rows, CFG).demoted, {})

    def test_domain_without_results_is_demoted(self):
        rows = [DomainStats(1, 40000, 39000, 1000), DomainStats(2, 40000, 30, 1)]
        snap = classify(rows, CFG)
        self.assertEqual(snap.demoted, {2: "no_results"})

    def test_no_results_needs_enough_scheduled(self):
        rows = [DomainStats(1, 40000, 39000, 1000), DomainStats(2, 999, 0, 0)]
        self.assertEqual(classify(rows, CFG).demoted, {})

    def test_pipeline_outage_demotes_nothing(self):
        # Ingest down: nothing has results, which says nothing about domains.
        rows = [DomainStats(1, 40000, 0, 0), DomainStats(2, 40000, 10, 0)]
        snap = classify(rows, CFG)
        self.assertEqual(snap.demoted, {})
        self.assertFalse(snap.pipeline_ok)

    def test_empty_stats(self):
        snap = classify([], CFG)
        self.assertEqual(snap.demoted, {})
        self.assertIsNone(snap.pipeline_response_rate)


class TrackerTest(unittest.TestCase):
    def _tracker(self, stats, clock, **cfg):
        calls = []

        def loader(shard_start, shard_end, window_days):
            calls.append((shard_start, shard_end, window_days))
            if isinstance(stats, Exception):
                raise stats
            return list(stats)

        cfg_obj = DomainHealthConfig(enabled=True, **cfg)
        return DomainHealthTracker(cfg_obj, loader, clock=clock), calls

    def test_refresh_is_rate_limited(self):
        clock = _Clock()
        tracker, calls = self._tracker([], clock, refresh_sec=900)
        tracker.maybe_refresh(0, 15)
        tracker.maybe_refresh(0, 15)
        clock.t += 899
        tracker.maybe_refresh(0, 15)
        self.assertEqual(len(calls), 1)
        clock.t += 1
        tracker.maybe_refresh(0, 15)
        self.assertEqual(calls, [(0, 15, 3), (0, 15, 3)])

    def test_demoted_domain_gets_one_probe_per_interval(self):
        clock = _Clock()
        rows = [DomainStats(1, 3000, 2900, 100), DomainStats(2, 1000, 0, 1000)]
        tracker, _ = self._tracker(rows, clock, probe_interval_sec=3600)
        tracker.maybe_refresh(0, 15)

        # Never probed yet: due now.
        self.assertEqual(tracker.excluded_now(), set())
        tracker.note_offered([1, 2])
        self.assertEqual(tracker.excluded_now(), {2})
        clock.t += 3599
        self.assertEqual(tracker.excluded_now(), {2})
        clock.t += 1
        self.assertEqual(tracker.excluded_now(), set())

    def test_loader_error_keeps_previous_snapshot(self):
        clock = _Clock()
        rows = [DomainStats(1, 3000, 2900, 100), DomainStats(2, 1000, 0, 1000)]
        state = {"stats": rows}

        def loader(shard_start, shard_end, window_days):
            if state["stats"] is None:
                raise RuntimeError("db down")
            return state["stats"]

        tracker = DomainHealthTracker(DomainHealthConfig(enabled=True), loader, clock=clock)
        tracker.maybe_refresh(0, 15)
        state["stats"] = None
        clock.t += 10_000
        tracker.maybe_refresh(0, 15)
        self.assertEqual(tracker.demoted, {2: "low_ok_rate"})

    def test_recovered_domain_is_released(self):
        clock = _Clock()
        state = {"stats": [DomainStats(1, 3000, 2900, 100), DomainStats(2, 1000, 0, 1000)]}
        tracker = DomainHealthTracker(
            DomainHealthConfig(enabled=True), lambda a, b, c: state["stats"], clock=clock,
        )
        tracker.maybe_refresh(0, 15)
        tracker.note_offered([2])
        self.assertEqual(tracker.excluded_now(), {2})

        state["stats"] = [DomainStats(1, 3000, 2900, 100), DomainStats(2, 1000, 900, 100)]
        clock.t += 900
        tracker.maybe_refresh(0, 15)
        self.assertEqual(tracker.excluded_now(), set())


class _StubGlobalStrategy:
    def __init__(self, candidates, per_domain_urls):
        self.candidates = candidates
        self.per_domain_urls = per_domain_urls
        self.peek_calls = []

    def peek_global_candidates(self, limit, exclude_domain_ids, shard_start, shard_end):
        self.peek_calls.append(set(exclude_domain_ids))
        return [c for c in self.candidates if c.domain_id not in exclude_domain_ids][:limit]

    def claim_domain_urls(self, shard_id, domain_id, per_domain_cap):
        return list(self.per_domain_urls.get(domain_id, []))


def _service(tmp, selector, health):
    stats_dir = os.path.join(tmp, "stats")
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    deriv = OffererDerivation(
        queue_dir_template=os.path.join(tmp, "queue_{id:02d}"),
        total_shards=16,
        shards_per_offerer=16,
    )
    cfg = OffererConfig(
        offerer_id=0,
        scan_interval_sec=60,
        max_domain_files=4,
        low_watermark_domains=1,
        per_domain_url_cap=10,
        stats_dir=stats_dir,
    )
    return OffererService(cfg, deriv, selector, health=health)


class ServiceHealthIntegrationTest(unittest.TestCase):
    def _health(self, clock):
        rows = [DomainStats(1, 3000, 2900, 100), DomainStats(2, 1000, 0, 1000)]
        return DomainHealthTracker(
            DomainHealthConfig(enabled=True), lambda a, b, c: rows, clock=clock,
        )

    def test_demoted_domain_is_excluded_after_its_probe(self):
        clock = _Clock()
        health = self._health(clock)
        cands = [
            CandidateDomain(domain_id=2, shard_id=0, domain_score=0.95),
            CandidateDomain(domain_id=1, shard_id=0, domain_score=0.8),
            CandidateDomain(domain_id=3, shard_id=1, domain_score=0.8),
        ]
        urls = {1: ["https://a/1"], 2: ["https://b/1"], 3: ["https://c/1"]}
        selector = _StubGlobalStrategy(cands, urls)
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, selector, health)
            # First refill: domain 2 is due for its probe.
            res = svc._refill_once_if_needed()
            self.assertEqual(res["new_domains"], 3)
            for p in Path(tmp, "queue_00").glob("*.json"):
                p.unlink()
            # Second refill: domain 2 was just probed, so it is excluded.
            res = svc._refill_once_if_needed()

        self.assertEqual(selector.peek_calls[1], {2})
        self.assertEqual(res["new_domains"], 2)

    def test_falls_back_to_demoted_domains_when_nothing_else(self):
        clock = _Clock()
        health = self._health(clock)
        health.maybe_refresh(0, 15)
        health.note_offered([2])
        selector = _StubGlobalStrategy(
            [CandidateDomain(domain_id=2, shard_id=0, domain_score=0.95)],
            {2: ["https://b/1"]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, selector, health)
            res = svc._refill_once_if_needed()

        self.assertEqual(selector.peek_calls, [{2}, set()])
        self.assertEqual(res["new_domains"], 1)

    def test_without_tracker_behaviour_is_unchanged(self):
        selector = _StubGlobalStrategy(
            [CandidateDomain(domain_id=2, shard_id=0, domain_score=0.95)],
            {2: ["https://b/1"]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, selector, None)
            res = svc._refill_once_if_needed()

        self.assertEqual(selector.peek_calls, [set()])
        self.assertEqual(res["new_domains"], 1)


if __name__ == "__main__":
    unittest.main()
