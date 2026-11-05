"""LightGBM models (MIT licence): prefilter, stage-1 matcher, stage-2 competition-aware matcher.

All models are trained with 2 folds split by Source-1 entity, so every training pair gets an
out-of-fold (OOF) score; decoding thresholds are tuned on OOF scores of the whole training set.
Test predictions average the fold models.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import _grp_max_other, _grp_rank

log = logging.getLogger("er")

PARAMS_MAIN = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   max_bin=255, verbose=-1)
PARAMS_PRE = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def _inner_split(groups: np.ndarray, frac: float = 0.08, seed: int = 7) -> np.ndarray:
    """Boolean mask selecting ~frac of the groups (by hash) for early stopping."""
    h = (groups.astype(np.int64) * 2654435761 + seed) % 1000
    return h < int(frac * 1000)


def fit_one(X: pd.DataFrame, y: np.ndarray, group: np.ndarray, params: dict, n_jobs: int,
            max_rounds: int = 3000, es: int = 100, weight: np.ndarray | None = None):
    """Fit one LightGBM model, early-stopping on ~8% of the groups held out."""
    va = _inner_split(group)
    dtr = lgb.Dataset(X[~va], label=y[~va], weight=None if weight is None else weight[~va], free_raw_data=True)
    dva = lgb.Dataset(X[va], label=y[va], weight=None if weight is None else weight[va], reference=dtr)
    m = lgb.train(dict(params, num_threads=n_jobs), dtr, num_boost_round=max_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
    log.info("  %d rounds, best logloss %.5f", m.best_iteration, m.best_score["valid_0"]["binary_logloss"])
    return m


def train_oof(X: pd.DataFrame, y: np.ndarray, fold: np.ndarray, group: np.ndarray, params: dict,
              n_jobs: int, max_rounds: int = 3000, es: int = 100, weight: np.ndarray | None = None):
    """Out-of-fold training over the folds in ``fold``. Returns (oof_pred, [models])."""
    oof = np.full(len(y), np.nan, dtype=np.float32)
    models = []
    for k in np.unique(fold):
        tr = fold != k
        m = fit_one(X[tr], y[tr], group[tr], params, n_jobs, max_rounds, es,
                    weight=None if weight is None else weight[tr])
        oof[fold == k] = m.predict(X[fold == k], num_iteration=m.best_iteration)
        models.append(m)
    return oof, models


def predict_oof(models, X: pd.DataFrame, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold prediction: rows of fold k are scored by the model trained without fold k."""
    out = np.empty(len(X), dtype=np.float32)
    for k, m in enumerate(models):
        idx = np.nonzero(fold == k)[0]
        if len(idx):
            out[idx] = m.predict(X.iloc[idx], num_iteration=m.best_iteration)
    return out


def copy_plan(q: np.ndarray, matched: np.ndarray, target: float, seed: int, n_rec: int) -> dict:
    """Copies of unmatched queries (new ids from ``n_rec`` on) until ``target`` of the queries are unmatched, as
    in the test-like validation. Returns rows of the candidate table to duplicate and the copies' query ids."""
    uq = np.unique(q)
    unm = uq[~matched[uq]]
    n_q, n_u = len(uq), len(unm)
    n_dup = int(np.ceil((target * n_q - n_u) / (1 - target)))
    pick = np.random.default_rng(seed).choice(unm, size=min(max(n_dup, 0), n_u), replace=False)
    new_id = pd.Series(n_rec + np.arange(len(pick)), index=pick)
    rows = np.nonzero(np.isin(q, pick))[0]
    return {"rows": rows, "new_q": new_id.reindex(q[rows]).to_numpy().astype(np.int64),
            "info": {"queries": int(n_q), "unmatched_before": n_u / max(n_q, 1), "copies": int(len(pick)),
                     "unmatched_after": (n_u + len(pick)) / max(n_q + len(pick), 1), "rows_added": int(len(rows))}}


def predict_avg(models, X: pd.DataFrame) -> np.ndarray:
    p = np.zeros(len(X), dtype=np.float64)
    for m in models:
        p += m.predict(X, num_iteration=m.best_iteration)
    return (p / len(models)).astype(np.float32)


def importance(models, top: int = 30) -> pd.Series:
    imp = sum(pd.Series(m.feature_importance("gain"), index=m.feature_name()) for m in models)
    return (imp / imp.sum()).sort_values(ascending=False).head(top)


# ------------------------------------------------------------------ stage 2 features
def collective_ctx(rec: pd.DataFrame, q_rows: np.ndarray) -> dict:
    """Per-row query attributes (integer codes) for the collective stage-2 features."""
    out = {}
    for name, col in (("name", "n_core"), ("hnc", "a_hnc"), ("street", "a_street")):
        if col not in rec:
            continue
        codes, uniq = pd.factorize(rec[col].fillna("").to_numpy(dtype=object))
        empty = np.nonzero(uniq == "")[0]
        if len(empty):
            codes = np.where(codes == empty[0], -1, codes)
        out[name] = codes[q_rows].astype(np.int64)
    out["empty"] = rec["a_empty"].to_numpy()[q_rows].astype(bool)
    return out


def _count_other(key: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """For each row: number of OTHER rows with the same key where ``flag`` holds."""
    tot = pd.Series(flag.astype(np.int32)).groupby(key).transform("sum").to_numpy()
    return (tot - flag.astype(np.int32)).astype(np.int16)


def _collective(s: np.ndarray, p1: np.ndarray, ctx: dict, f: pd.DataFrame):
    """S1-level consensus of the other candidates (their out-of-fold stage-1 scores share the S1's fold)."""
    # The matchers use only CONFIDENT siblings by default: counts of rejected same-name records (c_name_lo) or
    # of competing empty-address candidates (c_empty_hi) improved plain OOF but inflate with the test's extra
    # distractors and hurt the test-like score on dev1 (EXPERIMENTS.md, E6 vs E6b); they are computed here and
    # used only with --collective-counts (COUNT_FEATURES).
    conf, lo = p1 > 0.5, p1 < 0.3
    addr = ~ctx["empty"]
    big = np.int64(1) << 32
    if "name" in ctx:
        kn = s * big + ctx["name"]  # same S1, same query name
        f["c_name_hi"] = _count_other(kn, conf & addr)  # confident same-name siblings
        f["c_name_lo"] = _count_other(kn, lo & addr)  # rejected same-name records: a near-copy entity exists
    f["c_empty_hi"] = _count_other(s, (p1 > 0.3) & ~addr)  # an S1 rarely has 2+ empty-address matches
    for name in ("hnc", "street"):
        if name not in ctx:
            continue
        v = ctx[name]
        has = v >= 0
        tot = _count_other(s, conf & has)
        same = _count_other(s * big + np.where(has, v, -1), conf & has)
        f[f"c_{name}_sup"] = np.where(has, same, -1).astype(np.int16)
        f[f"c_{name}_con"] = np.where(has, tot - same, -1).astype(np.int16)


COUNT_FEATURES = ("c_name_lo", "c_empty_hi")


def stage2_features(cand: pd.DataFrame, src: np.ndarray, p1: np.ndarray, ctx: dict | None = None) -> pd.DataFrame:
    """Competition features from stage-1 probabilities (per query and per Source-1 entity), plus collective
    S1-level consensus features when ``ctx`` (see ``collective_ctx``) is given."""
    q = cand["q"].to_numpy()
    s = cand["s"].to_numpy().astype(np.int64)
    f = pd.DataFrame(index=cand.index)
    f["p1"] = p1
    f["p1_rank_q"] = _grp_rank(q, p1).astype(np.int16)
    f["p1_alt_q"] = p1 - _grp_max_other(q, p1)
    f["p1_sum_q"] = pd.Series(p1).groupby(q).transform("sum").to_numpy() - p1
    ks = s * 4 + src
    f["p1_rank_s"] = _grp_rank(ks, p1).astype(np.int16)
    f["p1_alt_s"] = p1 - _grp_max_other(ks, p1)
    hi = (p1 > 0.5).astype(np.int32)
    f["n_hi_s_same"] = pd.Series(hi).groupby(ks).transform("sum").to_numpy() - hi
    tot_hi = pd.Series(hi).groupby(s).transform("sum").to_numpy()
    f["n_hi_s_other"] = tot_hi - f["n_hi_s_same"].to_numpy() - hi
    f["p1_sum_s"] = pd.Series(p1).groupby(s).transform("sum").to_numpy() - p1
    f["p1_rank_s_all"] = _grp_rank(s, p1).astype(np.int16)
    if ctx is not None:
        _collective(s, p1, ctx, f)
    return f


# ------------------------------------------------------------------ prefilter threshold
def prefilter_threshold(p: np.ndarray, y: np.ndarray, rank: np.ndarray, keep_recall: float = 0.9995) -> float:
    """Largest threshold keeping >= keep_recall of the true pairs (rank-0 pairs are always kept)."""
    pos = (y == 1) & (rank != 0)
    n_pos_all = int((y == 1).sum())
    allowed = int(np.floor((1 - keep_recall) * n_pos_all))
    ps = np.sort(p[pos])
    if allowed <= 0 or len(ps) == 0:
        return 0.0
    return float(ps[min(allowed, len(ps)) - 1])
