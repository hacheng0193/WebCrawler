"""Tests for the MGL P_self scorer (shadow and live modes).

What we're protecting:

  * The runtime feature layout is fixed (the artifact is trained offline on
    exactly these columns), deterministic, and never reads golden data.
  * The artifact loader rejects a feature-version or column-count mismatch.
  * The ranker's shadow hook only logs: it never changes what is written to
    url_score, and a failing shadow model never breaks a scoring batch.
  * Live mode writes the MGL score; a failing MGL batch falls back to the
    legacy score instead of failing.
  * A missing artifact disables shadow and live instead of failing startup,
    and the env kill switch turns live off.
"""
from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy import sparse

import containers.scheduler_control.scorer.service as service_mod
from containers.scheduler_control.scorer.main import load_mgl
from containers.scheduler_control.scorer.service import (
    GoldenDiscoveryRankerConfig,
    GoldenDiscoveryRankerService,
    shadow_stats,
)
from libs.scoring import mgl_pself_features as F
from libs.scoring.mgl_pself_runtime import MglPselfScorer


def _tiny_artifact(path: Path, feature_version: int = F.FEATURE_VERSION) -> None:
    """A real LightGBM model over the full column layout: inlink count drives the score."""
    rng = np.random.default_rng(0)
    n = 400
    inl = rng.integers(0, 50, n)
    X = sparse.lil_matrix((n, F.N_FEATURES), dtype=np.float32)
    X[:, F.NUM_COLS.index("log_inlink")] = np.log1p(inl)[:, None]
    X[:, F.NUM_COLS.index("url_len")] = rng.integers(10, 200, n)[:, None]
    y = (inl > 25).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4, "min_data_in_leaf": 5},
                        lgb.Dataset(X.tocsr(), y), 20)
    joblib.dump({"booster_model_str": booster.model_to_string(), "iso_x": [0.0, 1.0], "iso_y": [0.0, 1.0],
                 "feature_version": feature_version, "metadata": {"model_name": "tiny"}}, path)


class FeatureTest(unittest.TestCase):
    def test_layout_and_determinism(self):
        row = {"url": "https://www.example.com/news/2026/11/01/big-story-12345?x=1&y=2",
               "inlink_count_approx": 3, "inlink_count_external": 1, "anchor_text": "Big story today",
               "discovery_source_type": 1, "discovered_from": "https://www.example.com/"}
        num, toks = F.feature_row(row)
        self.assertEqual(len(num), len(F.NUM_COLS))
        self.assertEqual((num, toks), F.feature_row(dict(row)))
        col = dict(zip(F.NUM_COLS, num))
        self.assertEqual(col["has_date"], 1.0)
        self.assertEqual(col["has_longnum"], 1.0)
        self.assertEqual(col["n_params"], 2.0)
        self.assertEqual(col["parent_is_home"], 1.0)
        self.assertEqual(col["parent_same_host"], 1.0)
        self.assertEqual(col["anchor_words"], 3.0)
        self.assertTrue(all(0 <= t < F.NHASH for t in toks))

    def test_home_page_and_missing_link_fields(self):
        num, toks = F.feature_row({"url": "https://example.org/"})
        col = dict(zip(F.NUM_COLS, num))
        self.assertEqual(col["is_home"], 1.0)
        self.assertEqual(col["dst"], -1.0)
        self.assertEqual(col["has_parent"], 0.0)
        self.assertEqual(toks, set())

    def test_unparsable_url(self):
        self.assertIsNone(F.feature_row({"url": "http://[::1"}))


class RuntimeTest(unittest.TestCase):
    def test_scores_rank_by_inlinks_and_are_calibrated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.joblib"
            _tiny_artifact(p)
            sc = MglPselfScorer.load(p)
        rows = [{"url": "https://a.com/x", "inlink_count_approx": 40},
                {"url": "https://a.com/y", "inlink_count_approx": 0},
                {"url": "http://[::1"}]
        s = sc.score_many_rows(rows)
        self.assertEqual(len(s), 3)
        self.assertGreater(s[0], s[1])
        self.assertEqual(s[2], 0.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in s))
        self.assertEqual(sc.score_many_rows([]), [])

    def test_feature_version_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.joblib"
            _tiny_artifact(p, feature_version=F.FEATURE_VERSION + 1)
            with self.assertRaises(ValueError):
                MglPselfScorer.load(p)


class ShadowStatsTest(unittest.TestCase):
    def test_identical_ranking(self):
        x = np.arange(20, dtype=float)
        st = shadow_stats(x, x / 100)
        self.assertEqual(st["n"], 20)
        self.assertAlmostEqual(st["spearman"], 1.0)
        self.assertEqual(st["top10_overlap"], 1.0)
        self.assertAlmostEqual(st["expected_golden"], x.sum() / 100)

    def test_reversed_and_constant(self):
        x = np.arange(20, dtype=float)
        self.assertAlmostEqual(shadow_stats(x, -x)["spearman"], -1.0)
        self.assertEqual(shadow_stats(np.zeros(5), x[:5])["spearman"], 0.0)
        self.assertEqual(shadow_stats(np.zeros(0), np.zeros(0)), {"n": 0})


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchall(self):
        return self.rows


class _Session:
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


class _LiveScorer:
    def score_many_rows(self, recs):
        return [0.5 + 0.01 * i for i in range(len(recs))]


class _Shadow:
    def __init__(self, fail=False):
        self.fail = fail
        self.recs = None

    def score_many_rows(self, recs):
        if self.fail:
            raise RuntimeError("boom")
        self.recs = recs
        return [0.001 * i for i in range(len(recs))]


class ServiceShadowTest(unittest.TestCase):
    ROWS = [(f"https://a.com/{i}", i, 0, "anchor", 1, "https://a.com/") for i in range(5)]

    def _run(self, shadow, live=False):
        cur = _Cursor(list(self.ROWS))
        cfg = GoldenDiscoveryRankerConfig(total_shards=256, num_workers=1, worker_id=0, batch_size=10,
                                          scan_interval_sec=60, max_batches_per_shard=1,
                                          domain_priority_steering_enabled=True)
        svc = GoldenDiscoveryRankerService(cfg=cfg, Session=_Session(cur), scorer=_LiveScorer(),
                                           shadow_scorer=shadow, mgl_live=live)
        written = []
        orig = service_mod.execute_values
        service_mod.execute_values = lambda c, sql, rows, page_size=None: written.extend(rows)
        try:
            with self.assertLogs("golden_discovery_ranker_v1", level="INFO") as logs:
                n = svc._score_batch(7)
        finally:
            service_mod.execute_values = orig
        return n, written, logs.output, cur.sql

    def test_shadow_logs_and_does_not_change_written_scores(self):
        shadow = _Shadow()
        n, written, logs, sql = self._run(shadow)
        _, base_written, _, _ = self._run(None)
        self.assertEqual(n, 5)
        self.assertEqual(written, base_written)
        self.assertTrue(any("mgl_shadow" in line for line in logs))
        self.assertEqual(shadow.recs[0]["discovered_from"], "https://a.com/")
        self.assertEqual(shadow.recs[3]["discovery_source_type"], 1)
        self.assertIn("discovered_from", sql[0])

    def test_shadow_failure_does_not_break_batch(self):
        n, written, logs, _ = self._run(_Shadow(fail=True))
        self.assertEqual(n, 5)
        self.assertEqual(len(written), 5)
        self.assertTrue(any("mgl_shadow_error" in line for line in logs))


class ServiceLiveTest(unittest.TestCase):
    ROWS = ServiceShadowTest.ROWS
    _run = ServiceShadowTest._run

    def test_live_writes_mgl_scores(self):
        mgl = _Shadow()
        n, written, logs, _ = self._run(mgl, live=True)
        self.assertEqual(n, 5)
        self.assertEqual([s for _, s in written], [0.001 * i for i in range(5)])
        self.assertFalse(any("mgl_shadow" in line for line in logs))
        self.assertTrue(any("score_batch" in line for line in logs))
        self.assertEqual(mgl.recs[3]["discovery_source_type"], 1)

    def test_live_failure_falls_back_to_legacy_scores(self):
        n, written, logs, _ = self._run(_Shadow(fail=True), live=True)
        _, base_written, _, _ = self._run(None)
        self.assertEqual(n, 5)
        self.assertEqual(written, base_written)
        self.assertTrue(any("mgl_live_error" in line for line in logs))
        self.assertFalse(any("mgl_shadow_error" in line for line in logs))

    def test_live_without_scorer_is_off(self):
        _, written, _, _ = self._run(None, live=True)
        _, base_written, _, _ = self._run(None)
        self.assertEqual(written, base_written)


class LoaderTest(unittest.TestCase):
    def _events(self, logs):
        return {getattr(r, "event", None): r for r in logs.records}

    def test_disabled_by_default(self):
        with self.assertLogs("golden_discovery_ranker_v1", level="INFO") as logs:
            self.assertEqual(load_mgl({}, 0), (None, False))
        ev = self._events(logs)
        self.assertEqual(ev["golden_discovery_ranker_v1.mgl_shadow.disabled"].reason, "config")
        self.assertEqual(ev["golden_discovery_ranker_v1.mgl_live.disabled"].reason, "config")

    def test_missing_artifact_disables_instead_of_failing(self):
        with self.assertLogs("golden_discovery_ranker_v1", level="INFO") as logs:
            self.assertEqual(load_mgl({"mgl_shadow_enabled": True, "mgl_live_enabled": True,
                                       "mgl_shadow_artifact": "/nonexistent.joblib"}, 0), (None, False))
        ev = self._events(logs)
        self.assertEqual(ev["golden_discovery_ranker_v1.mgl_shadow.disabled"].reason, "artifact_load_failed")
        self.assertEqual(ev["golden_discovery_ranker_v1.mgl_live.disabled"].reason, "artifact_load_failed")

    def test_shadow_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.joblib"
            _tiny_artifact(p)
            with self.assertLogs("golden_discovery_ranker_v1", level="INFO") as logs:
                sc, live = load_mgl({"mgl_shadow_enabled": True, "mgl_shadow_artifact": str(p)}, 0)
        self.assertIsInstance(sc, MglPselfScorer)
        self.assertFalse(live)
        ev = self._events(logs)
        self.assertIn("golden_discovery_ranker_v1.mgl_shadow.enabled", ev)
        self.assertIn("golden_discovery_ranker_v1.mgl_live.disabled", ev)

    def test_live_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.joblib"
            _tiny_artifact(p)
            with self.assertLogs("golden_discovery_ranker_v1", level="INFO") as logs:
                sc, live = load_mgl({"mgl_live_enabled": True, "mgl_shadow_artifact": str(p)}, 0)
        self.assertIsInstance(sc, MglPselfScorer)
        self.assertTrue(live)
        self.assertIn("golden_discovery_ranker_v1.mgl_live.enabled", self._events(logs))

    def test_env_kill_switch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.joblib"
            _tiny_artifact(p)
            os.environ["GOLDEN_DISCOVERY_RANKER_V1_MGL_LIVE_ENABLED"] = "false"
            try:
                with self.assertLogs("golden_discovery_ranker_v1", level="INFO"):
                    sc, live = load_mgl({"mgl_live_enabled": True, "mgl_shadow_artifact": str(p)}, 0)
            finally:
                del os.environ["GOLDEN_DISCOVERY_RANKER_V1_MGL_LIVE_ENABLED"]
        self.assertIsNone(sc)
        self.assertFalse(live)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
