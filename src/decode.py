"""Turn pairwise match probabilities into final per-Source-1 match lists.

Two structural facts from the training data drive the decoder:

1. Every Source-2/3 record belongs to at most ONE Source-1 entity, so each
   candidate record is first assigned to its single best-scoring Source-1
   entity (``assign``).
2. The metric is macro F0.5 per Source-1 entity with singletons included, so for
   every Source-1 entity we pick the subset of its assigned records that
   maximises the *expected* F0.5 under the model's probabilities
   (``select_expected_f05``); the empty set is one of the options, which is what
   protects singletons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25
NMAX = 24  # max candidates per Source-1 entity considered by the exact optimiser


def assign(pairs: pd.DataFrame, prob_col: str = "p", min_p: float = 0.0) -> pd.DataFrame:
    """Keep, for every query record ``q``, only its highest-probability Source-1 entity."""
    df = pairs[pairs[prob_col] >= min_p]
    df = df.sort_values(["q", prob_col], ascending=[True, False], kind="mergesort")
    return df.drop_duplicates("q", keep="first")


def _weights(n: int, k: int) -> np.ndarray:
    x = np.arange(n + 1)[:, None].astype(np.float64)
    y = np.arange(n + 1)[None, :].astype(np.float64)
    return (1 + BETA2) * x / (BETA2 * (x + y) + k)


def expected_f05_table(P: np.ndarray, miss_p: float = 0.0) -> np.ndarray:
    """Expected F0.5 of predicting the top-k (k = 0..n) for each row of ``P``.

    ``P`` is (m, n) with probabilities sorted descending per row. Match events are
    treated as independent Bernoullis; ``miss_p`` is the probability of one extra
    true match that is not among the candidates (blocking miss).
    Returns (m, n + 1).
    """
    m, n = P.shape
    out = np.zeros((m, n + 1))
    # suffix PMFs: suf[k] = distribution of #true among items k..n-1 (+ missed)
    suf = np.zeros((n + 1, m, n + 2))
    suf[n, :, 0] = 1.0 - miss_p
    suf[n, :, 1] = miss_p
    for k in range(n - 1, -1, -1):
        pk = P[:, k:k + 1]
        suf[k] = suf[k + 1] * (1 - pk)
        suf[k, :, 1:] += suf[k + 1, :, :-1] * pk
    out[:, 0] = suf[0, :, 0]
    pre = np.zeros((m, n + 2))
    pre[:, 0] = 1.0
    for k in range(1, n + 1):
        pk = P[:, k - 1:k]
        new = pre * (1 - pk)
        new[:, 1:] += pre[:, :-1] * pk
        pre = new
        W = _weights(n + 1, k)
        out[:, k] = np.einsum("mx,xy,my->m", pre, W, suf[k], optimize=True)
    return out


def select_expected_f05(assigned: pd.DataFrame, prob_col: str = "p", miss_p: float = 0.0,
                        prob_scale: float = 1.0, chunk: int = 200_000) -> pd.DataFrame:
    """Per Source-1 entity choose the prefix (by probability) maximising expected F0.5."""
    df = assigned[["s", "q", prob_col]].copy()
    df["pp"] = np.clip(df[prob_col].to_numpy() * prob_scale, 0.0, 1.0)
    df = df.sort_values(["s", "pp"], ascending=[True, False], kind="mergesort")
    df["rk"] = df.groupby("s").cumcount()
    df = df[df["rk"] < NMAX]
    sizes = df.groupby("s")["q"].transform("size").to_numpy()
    df["n"] = sizes
    keep_mask = np.zeros(len(df), dtype=bool)
    rk = df["rk"].to_numpy()
    pp = df["pp"].to_numpy()
    for n in np.unique(sizes):
        idx = np.nonzero(sizes == n)[0]
        # rows are contiguous per S1 entity and sorted by rank, so reshape works
        P_all = pp[idx].reshape(-1, n)
        best_k = np.empty(P_all.shape[0], dtype=np.int64)
        for st in range(0, P_all.shape[0], chunk):
            tab = expected_f05_table(P_all[st:st + chunk], miss_p=miss_p)
            best_k[st:st + chunk] = tab.argmax(axis=1)
        keep_mask[idx] = rk[idx] < np.repeat(best_k, n)
    return df.loc[keep_mask, ["s", "q", prob_col]]


def em_prior(p: np.ndarray, prior_train: float, n_iter: int = 200, tol: float = 1e-7) -> float:
    """Saerens, Latinne & Decaestecker (2002): EM estimate of the positive-class prior of an unlabeled set
    scored by a classifier calibrated for ``prior_train``."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    pi = prior_train
    for _ in range(n_iter):
        new = float(prior_shift(p, prior_train, pi).mean())
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def prior_shift(p: np.ndarray, prior_train: float, prior_new: float) -> np.ndarray:
    """Adjust calibrated probabilities from the training prior to a new prior (Bayes rule on the odds)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    a = prior_new / prior_train
    b = (1 - prior_new) / (1 - prior_train)
    return (a * p / (a * p + b * (1 - p))).astype(np.float32)


def select_threshold(assigned: pd.DataFrame, t: float, prob_col: str = "p") -> pd.DataFrame:
    return assigned.loc[assigned[prob_col] >= t, ["s", "q", prob_col]]


def to_lists(selected: pd.DataFrame, s1_ids: np.ndarray, q_ids: np.ndarray) -> dict:
    """Map integer codes back to string ids: {s1_id: [q_id, ...]} (every s1 present)."""
    out = {s: [] for s in s1_ids}
    if len(selected):
        s = s1_ids[selected["s"].to_numpy()]
        q = q_ids[selected["q"].to_numpy()]
        for a, b in zip(s, q):
            out[a].append(b)
    return out
