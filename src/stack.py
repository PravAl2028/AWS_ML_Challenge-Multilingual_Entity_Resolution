"""Stage 3 (v8): combine the pipeline's stage-2 score with the character cross-encoder score (``charce.py``).

Features per candidate pair (s, q): logits of p1, p2 and the cross-encoder score ce; the query-level competition of
ce and p2 (rank, margin over the query's best other S1); whether the query has no address, its source, and five cheap
record relations (house number equal / shifted / missing, canonical composite equal, legal form kept / added /
dropped / changed, core name equal, street equal) so the stack can weigh ce differently for twin-like pairs. A small, regularised LightGBM is trained out of fold
(the pipeline's 5 model folds, split by Source-1 entity) on the training pairs' out-of-fold p1 / p2 / ce, with the
same distractor-negative weight as the matchers. Test pairs get the mean of the 5 fold models.

Countries without training labels (an open set: any country label absent from the training records) keep the
stage-2 score and the stage-2 decoding rule, because the cross-encoder has never seen their text.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import _grp_max_other, _grp_rank

log = logging.getLogger("er")

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255, verbose=-1)
# Query-level competition only: S1-level views of ce (rank among the S1's candidates, best other candidate, number of
# other confident candidates) would reward multi-record support, which the test's distractor clusters inflate
# (brief F2) while the training OOF cannot show it; they are computed but not used.
FEATURES = ["lp2", "lp1", "lce", "ce_rank_q", "ce_alt_q", "p2_alt_q", "q_empty", "hn_rel", "hnc_eq", "legal_rel",
            "core_eq", "street_eq", "q_src"]
REC_COLS = ["a_empty", "a_hn", "a_hnc", "n_legal", "n_core", "a_street", "source"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def _codes(col: pd.Series) -> np.ndarray:
    """Integer codes of a string column, -1 for empty / missing (equal codes <=> equal non-empty strings)."""
    v = col.fillna("").astype(str).to_numpy(dtype=object)
    codes, uniq = pd.factorize(v)
    empty = np.nonzero(uniq == "")[0]
    if len(empty):
        codes = np.where(codes == empty[0], -1, codes)
    return codes.astype(np.int64)


def record_view(rec: pd.DataFrame) -> dict:
    """Per-record arrays used by the pair features (``rec`` = records.parquet with ``REC_COLS``)."""
    return {"empty": rec["a_empty"].to_numpy().astype(np.int8), "hn": rec["a_hn"].to_numpy().astype(np.int64),
            "hnc": _codes(rec["a_hnc"]), "legal": _codes(rec["n_legal"]), "core": _codes(rec["n_core"]),
            "street": _codes(rec["a_street"]), "src": rec["source"].to_numpy().astype(np.int8)}


def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 equal, 0 different, -1 one side missing."""
    return np.where((a < 0) | (b < 0), -1, (a == b).astype(np.int8)).astype(np.int8)


def features(s: np.ndarray, q: np.ndarray, p1: np.ndarray, p2: np.ndarray, ce: np.ndarray, rv: dict) -> pd.DataFrame:
    f = pd.DataFrame({"lp2": _logit(p2), "lp1": _logit(p1), "lce": _logit(ce)})
    ce = ce.astype(np.float32)
    f["ce_rank_q"] = _grp_rank(q, ce).astype(np.int16)
    f["ce_alt_q"] = ce - _grp_max_other(q, ce)
    f["p2_alt_q"] = p2.astype(np.float32) - _grp_max_other(q, p2.astype(np.float32))
    f["ce_rank_s"] = _grp_rank(s, ce).astype(np.int16)
    f["ce_max_other_s"] = _grp_max_other(s, ce)
    hi = (ce > 0.5).astype(np.int32)
    f["n_ce_hi_s"] = (pd.Series(hi).groupby(s).transform("sum").to_numpy() - hi).astype(np.int16)
    f["q_empty"] = rv["empty"][q]
    hq, hs = rv["hn"][q], rv["hn"][s]
    d = hq - hs
    # 0 both missing, 1 one missing, 2 equal, 3 shift +1..25, 4 shift -25..-1, 5 other
    f["hn_rel"] = np.select([(hq < 0) & (hs < 0), (hq < 0) | (hs < 0), d == 0, (d >= 1) & (d <= 25),
                             (d <= -1) & (d >= -25)], [0, 1, 2, 3, 4], 5).astype(np.int8)
    f["hnc_eq"] = _eq(rv["hnc"][q], rv["hnc"][s])
    lq, ls = rv["legal"][q], rv["legal"][s]
    # 0 same (incl. both none), 1 added by the query, 2 dropped by the query, 3 changed
    f["legal_rel"] = np.select([lq == ls, ls < 0, lq < 0], [0, 1, 2], 3).astype(np.int8)
    f["core_eq"] = _eq(rv["core"][q], rv["core"][s])
    f["street_eq"] = _eq(rv["street"][q], rv["street"][s])
    f["q_src"] = rv["src"][q]
    return f[FEATURES]


def _inner(groups: np.ndarray, frac: float = 0.08, seed: int = 7) -> np.ndarray:
    h = (groups.astype(np.int64) * 2654435761 + seed) % 1000
    return h < int(frac * 1000)


def train_oof(X: pd.DataFrame, y: np.ndarray, fold: np.ndarray, group: np.ndarray, w: np.ndarray | None,
              n_jobs: int, params: dict | None = None, max_rounds: int = 3000):
    params = dict(PARAMS, **(params or {}), num_threads=n_jobs)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    models = []
    for k in np.unique(fold):
        tr = fold != k
        va = _inner(group[tr])
        Xtr, ytr = X[tr], y[tr]
        wtr = None if w is None else w[tr]
        dtr = lgb.Dataset(Xtr[~va], label=ytr[~va], weight=None if wtr is None else wtr[~va])
        dva = lgb.Dataset(Xtr[va], label=ytr[va], weight=None if wtr is None else wtr[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        log.info("  stage-3 fold %d: %d rounds, logloss %.5f", k, m.best_iteration,
                 m.best_score["valid_0"]["binary_logloss"])
        oof[fold == k] = m.predict(X[fold == k], num_iteration=m.best_iteration)
        models.append(m)
    return oof, models


def predict_avg(models, X: pd.DataFrame) -> np.ndarray:
    p = np.zeros(len(X), dtype=np.float64)
    for m in models:
        p += m.predict(X, num_iteration=m.best_iteration)
    return (p / len(models)).astype(np.float32)


def importance(models) -> pd.Series:
    imp = sum(pd.Series(m.feature_importance("gain"), index=m.feature_name()) for m in models)
    return (imp / imp.sum()).sort_values(ascending=False)
