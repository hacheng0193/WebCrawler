"""MGL P_self runtime scorer: calibrated P(URL appears in the next golden batch).

Artifact (joblib dict), produced offline by fable-crawl-selection:
  booster_model_str  LightGBM text model over mgl_pself_features columns
  iso_x, iso_y       isotonic calibration knots (raw score -> probability)
  feature_version    must equal mgl_pself_features.FEATURE_VERSION
  metadata           free-form (model_name, trained_on, ...)

Golden batches are training labels only; no golden-derived value is a feature.
Inference is pinned to one thread (same reason as the v2 ranker).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from scipy import sparse

from .mgl_pself_features import FEATURE_VERSION, N_FEATURES, NUM_COLS, feature_row


class MglPselfScorer:
    def __init__(self, booster: lgb.Booster, iso_x, iso_y, metadata: dict[str, Any]):
        self.booster = booster
        self.iso_x = np.asarray(iso_x, dtype=np.float64)
        self.iso_y = np.asarray(iso_y, dtype=np.float64)
        self.metadata = metadata

    @classmethod
    def load(cls, path: str | Path) -> "MglPselfScorer":
        art = joblib.load(path)
        if art.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"mgl_pself artifact feature_version={art.get('feature_version')} "
                f"!= runtime {FEATURE_VERSION}"
            )
        booster = lgb.Booster(model_str=art["booster_model_str"])
        if booster.num_feature() != N_FEATURES:
            raise ValueError(f"mgl_pself artifact has {booster.num_feature()} features, expected {N_FEATURES}")
        return cls(booster, art["iso_x"], art["iso_y"], dict(art.get("metadata") or {}))

    def _matrix(self, rows: list[dict]) -> tuple[sparse.csr_matrix, list[int]]:
        data, ri, ci, ok = [], [], [], []
        n_num = len(NUM_COLS)
        for i, row in enumerate(rows):
            fr = feature_row(row)
            if fr is None:
                continue
            r = len(ok)
            ok.append(i)
            num, toks = fr
            for j, v in enumerate(num):
                if v:
                    data.append(v); ri.append(r); ci.append(j)
            for t in toks:
                data.append(1.0); ri.append(r); ci.append(n_num + t)
        X = sparse.csr_matrix((np.asarray(data, np.float32), (ri, ci)), shape=(len(ok), N_FEATURES))
        return X, ok

    def score_many_rows(self, rows: list[dict]) -> list[float]:
        """Calibrated probabilities, one per row; unparsable URLs score 0."""
        out = [0.0] * len(rows)
        if not rows:
            return out
        X, ok = self._matrix(rows)
        if ok:
            raw = self.booster.predict(X, num_threads=1)
            cal = np.interp(raw, self.iso_x, self.iso_y)
            for i, p in zip(ok, cal):
                out[i] = float(p)
        return out
