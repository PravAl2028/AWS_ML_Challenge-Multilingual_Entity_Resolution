"""Stage 2: candidate generation.

Every Source-2/3 record (query) retrieves its top-K Source-1 entities of the same country by
cosine of a combined char-3-gram TF-IDF vector  [sqrt(.5)*name | sqrt(.5)*address]. To keep the
exact sparse search affordable on CPU, retrieval is partitioned by (country, state): a query whose
state is recognised searches the S1 entities of that state (plus S1 entities without a state);
a query without a recognised state (empty address, France records without a region, ...) searches
its whole country. States that the training data shows to be interchangeable (e.g. S1 Telangana
vs S2/S3 Andhra Pradesh) are merged into one partition (``learn_state_merges``).
Candidates keep rank < K and score >= ratio * best score of that query.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from .io_utils import timer
from .par import G, pmap

log = logging.getLogger("er")
N_FEAT = 1 << 20


def _vec():
    return HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEAT, alternate_sign=False,
                             binary=True, norm=None, lowercase=False, dtype=np.float32)


def _hash_chunk(texts):
    return _vec().transform(texts).tocsr()


def tfidf(texts: np.ndarray, n_jobs: int, chunk: int = 200_000) -> sp.csr_matrix:
    """Binary hashed char-3-gram TF with IDF from the same corpus, rows L2-normalised."""
    tasks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    X = sp.vstack(pmap(_hash_chunk, tasks, n_jobs), format="csr")
    df = np.bincount(X.indices, minlength=N_FEAT).astype(np.float64)
    idf = (np.log((1.0 + X.shape[0]) / (1.0 + df)) + 1.0).astype(np.float32)
    X.data *= idf[X.indices]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    X = sp.diags((1.0 / norms).astype(np.float32)) @ X
    return X.tocsr().astype(np.float32)


def build_matrices(rec: pd.DataFrame, n_jobs: int, prune_df: float = 0.10, with_retrieval: bool = True):
    """Name / address TF-IDF and the combined retrieval matrix (None when ``with_retrieval`` is False).

    For retrieval only, 3-grams present in more than ``prune_df`` of the Source-1 records are
    dropped from the combined matrix: they carry little IDF weight but dominate the cost of the
    sparse product (measured on full-density dev data: ~2x faster, identical recall).
    """
    with timer("name TF-IDF"):
        Xn = tfidf(rec["n_full"].fillna("").to_numpy(dtype=object), n_jobs)
    with timer("address TF-IDF"):
        Xa = tfidf(rec["a_tok"].fillna("").to_numpy(dtype=object), n_jobs)
    if not with_retrieval:
        return Xn, Xa, None
    w = np.float32(np.sqrt(0.5))
    Xc = sp.hstack([Xn * w, Xa * w], format="csr")
    if prune_df and prune_df < 1:
        s1 = np.nonzero(rec["source"].to_numpy() == 1)[0]
        df = np.bincount(Xc[s1].indices, minlength=Xc.shape[1]) / max(1, len(s1))
        drop = df > prune_df
        keep = np.ones(Xc.shape[1], dtype=np.float32)
        keep[drop] = 0.0
        Xc = (Xc @ sp.diags(keep)).tocsr()
        Xc.eliminate_zeros()
        log.info("retrieval matrix: dropped %d very common 3-grams (df > %.2f)", int(drop.sum()), prune_df)
    return Xn, Xa, Xc


# ---------------------------------------------------------------- state partitions
def learn_state_merges(rec: pd.DataFrame, gt_pairs: pd.DataFrame, min_rate: float = 0.005) -> dict:
    """{country: {state: representative}} merging states that true pairs frequently cross.

    A (S1 state x, query state y) combination is merged when at least ``min_rate`` of the true
    pairs whose S1 is in state x carry state y on the query side.
    """
    st = rec["a_state"].fillna("").to_numpy(dtype=object)
    ctry = rec["country"].astype(str).to_numpy(dtype=object)
    s, q = gt_pairs["s"].to_numpy(), gt_pairs["q"].to_numpy()
    a, b, c = st[s], st[q], ctry[s]
    df = pd.DataFrame({"c": c, "a": a, "b": b})
    df = df[(df.a != "") & (df.b != "")]
    tot = df.groupby(["c", "a"]).size()
    cross = df[df.a != df.b].groupby(["c", "a", "b"]).size()
    merges = {}
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for (cc, x, y), n in cross.items():
        rate = n / max(1, tot.get((cc, x), 0))
        if rate >= min_rate:
            rx, ry = find((cc, x)), find((cc, y))
            if rx != ry:
                parent[ry] = rx
                log.info("merging state partitions %s: %s + %s (rate %.4f)", cc, x, y, rate)
    for key in list(parent):
        cc, x = key
        merges.setdefault(cc, {})[x] = find(key)[1]
    return merges


def partition_keys(rec: pd.DataFrame, merges: dict) -> np.ndarray:
    st = rec["a_state"].fillna("").to_numpy(dtype=object).copy()
    ctry = rec["country"].astype(str).to_numpy(dtype=object)
    for cc, m in merges.items():
        idx = np.nonzero(ctry == cc)[0]
        st[idx] = [m.get(x, x) for x in st[idx]]
    return st


# ---------------------------------------------------------------- retrieval
def _s1_block(g: dict, s_rows: np.ndarray):
    """Transposed S1 block of a search space; a worker keeps the last one, since consecutive query chunks of
    a large space (e.g. a whole country) share it."""
    key = (len(s_rows), int(s_rows[0]), int(s_rows[-1]), int(s_rows[len(s_rows) // 2]))
    hit = g.get("_s1_block")
    if hit is not None and hit[0] == key:
        return hit[1]
    g["_s1_block"] = None  # free the previous block before building the next one
    Xs = g["X"][s_rows]
    XsT = Xs.T.tocsr()
    del Xs
    g["_s1_block"] = (key, XsT)
    return XsT


def _search(task):
    s_rows, q_rows, K = task
    g = G()
    XsT = _s1_block(g, s_rows)
    nb = len(s_rows)
    k = min(K, nb)
    c = int(max(1, min(4096, 2e7 // max(nb, 1))))
    I_out = np.empty((len(q_rows), k), dtype=np.int32)
    S_out = np.empty((len(q_rows), k), dtype=np.float32)
    X = g["X"]
    for a in range(0, len(q_rows), c):
        Q = X[q_rows[a:a + c]]
        R = (Q @ XsT).toarray()
        if k < nb:
            part = np.argpartition(-R, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(nb), (R.shape[0], 1))
        sc = np.take_along_axis(R, part, axis=1)
        order = np.argsort(-sc, axis=1, kind="stable")
        I_out[a:a + c] = s_rows[np.take_along_axis(part, order, axis=1)]
        S_out[a:a + c] = np.take_along_axis(sc, order, axis=1)
    return q_rows, I_out, S_out


def retrieve(rec: pd.DataFrame, Xc: sp.csr_matrix, merges: dict, n_jobs: int, K: int = 50,
             ratio: float = 0.7, q_chunk: int = 20000, max_cost: float = 2e9) -> pd.DataFrame:
    """Top-K S1 candidates for every S2/S3 record; returns a candidate DataFrame.

    Query chunks are capped at ``q_chunk`` records and ``max_cost`` (queries x S1) row-pairs, so a large
    search space (e.g. a whole country for records without a state) is split over many workers instead of
    becoming one long task. The result does not depend on the chunking.
    """
    src = rec["source"].to_numpy()
    ctry = rec["country"].astype(str).to_numpy(dtype=object)
    key = partition_keys(rec, merges)
    tasks = []

    def add(rows, qs):
        step = int(max(256, min(q_chunk, max_cost // max(len(rows), 1))))
        for a in range(0, len(qs), step):
            tasks.append((rows, qs[a:a + step], K))

    for cc in pd.unique(ctry):
        s_idx = np.nonzero((ctry == cc) & (src == 1))[0].astype(np.int64)
        q_idx = np.nonzero((ctry == cc) & (src != 1))[0].astype(np.int64)
        if len(s_idx) == 0 or len(q_idx) == 0:
            continue
        s_key, q_key = key[s_idx], key[q_idx]
        s_states = set(pd.unique(s_key)) - {""}
        nostate = s_idx[s_key == ""]
        for b in sorted(s_states):
            add(np.concatenate([s_idx[s_key == b], nostate]), q_idx[q_key == b])
        rest = q_idx[~np.isin(q_key, list(s_states))]
        add(s_idx, rest)
        log.info("country %s: %d S1, %d queries, %d state partitions, %d country-wide queries",
                 cc, len(s_idx), len(q_idx), len(s_states), len(rest))
    # biggest tasks first for better load balance
    cost = np.array([len(t[0]) * len(t[1]) for t in tasks], dtype=np.float64)
    tasks = [tasks[i] for i in np.argsort(-cost)]
    with timer(f"sparse top-{K} retrieval ({len(tasks)} tasks, {cost.sum():.3g} row-pairs)"):
        res = pmap(_search, tasks, n_jobs, shared={"X": Xc})
    qs, ss, sc, rk, t1, t2 = [], [], [], [], [], []
    for q_rows, I, S in res:
        k = I.shape[1]
        top1 = S[:, 0]
        second = S[:, 1] if k > 1 else np.zeros_like(top1)
        keep = (S >= ratio * top1[:, None]) & (S > 0)
        keep[:, 0] = S[:, 0] > 0
        r_idx, c_idx = np.nonzero(keep)
        qs.append(q_rows[r_idx].astype(np.int32))
        ss.append(I[r_idx, c_idx])
        sc.append(S[r_idx, c_idx])
        rk.append(c_idx.astype(np.int16))
        t1.append(top1[r_idx])
        t2.append(second[r_idx])
    cand = pd.DataFrame({
        "q": np.concatenate(qs), "s": np.concatenate(ss).astype(np.int32), "score": np.concatenate(sc),
        "rank": np.concatenate(rk), "top1": np.concatenate(t1), "second": np.concatenate(t2)})
    cand = cand.sort_values(["q", "rank"], kind="mergesort").reset_index(drop=True)
    log.info("candidates: %d pairs (%.2f per query)", len(cand), len(cand) / max(1, cand.q.nunique()))
    return cand


# ---------------------------------------------------------------- house-number key channel
def key_candidates(rec: pd.DataFrame, cand: pd.DataFrame, Xn, Xa, K: int = 50,
                   per_million: float = 20.0) -> pd.DataFrame:
    """Add S1 entities of the same country that share the query's canonical primary house number
    (``a_hnc``, e.g. "3-6-57-13") when the number is rare -- carried by at most ``per_million`` S1 per million
    S1 of the country (at least 1) -- and TF-IDF retrieval missed them. The cap scales with the country's size
    so that the channel stays equally selective on a development subset and on the full data.

    Distinctive composite numbers recover true pairs whose name is a brand token, a website or only
    generic words (they rank beyond the top-K), or whose state label disagrees with the S1's. Added pairs get
    their name/address cosine as ``score``, rank K, the query's top1/second, and ``via_key`` = 1.
    """
    if "a_hnc" not in rec:
        return cand.assign(via_key=np.int8(0))
    src = rec["source"].to_numpy()
    ctry = rec["country"].astype(str).to_numpy(dtype=object)
    hnc = rec["a_hnc"].fillna("").to_numpy(dtype=object)
    idx = np.arange(len(rec))
    s1 = (src == 1) & (hnc != "")
    qm = (src != 1) & (hnc != "")
    S = pd.DataFrame({"c": ctry[s1], "k": hnc[s1], "s": idx[s1]})
    n_c = pd.Series(ctry[src == 1]).value_counts()
    cap = np.maximum(1, np.floor(per_million * S["c"].map(n_c).to_numpy() / 1e6))
    S = S[S.groupby(["c", "k"])["s"].transform("size").to_numpy() <= cap]
    Q = pd.DataFrame({"c": ctry[qm], "k": hnc[qm], "q": idx[qm]})
    new = Q.merge(S, on=["c", "k"])[["q", "s"]]
    have = pair_key_int(cand["q"].to_numpy(), cand["s"].to_numpy())
    new = new[~np.isin(pair_key_int(new["q"].to_numpy(), new["s"].to_numpy()), have)]
    if len(new) == 0:
        return cand.assign(via_key=np.int8(0))
    q, s_ = new["q"].to_numpy(), new["s"].to_numpy()
    sc = np.empty(len(q), dtype=np.float32)
    for a in range(0, len(q), 500_000):
        b = slice(a, a + 500_000)
        n = np.asarray(Xn[q[b]].multiply(Xn[s_[b]]).sum(axis=1)).ravel()
        d = np.asarray(Xa[q[b]].multiply(Xa[s_[b]]).sum(axis=1)).ravel()
        sc[b] = 0.5 * (n + d)
    qa = cand.groupby("q")[["top1", "second"]].first()
    add = pd.DataFrame({"q": q.astype(cand["q"].dtype), "s": s_.astype(cand["s"].dtype), "score": sc,
                        "rank": np.int16(K), "top1": qa["top1"].reindex(q).fillna(0).to_numpy(np.float32),
                        "second": qa["second"].reindex(q).fillna(0).to_numpy(np.float32)})
    out = pd.concat([cand.assign(via_key=np.int8(0)), add.assign(via_key=np.int8(1))], ignore_index=True)
    out = out.sort_values(["q", "rank"], kind="mergesort").reset_index(drop=True)
    log.info("key channel: +%d pairs (%.3f per query; keys held by <= %g S1 per million)", len(add),
             len(add) / max(1, cand["q"].nunique()), per_million)
    return out


def pair_key_int(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    return q.astype(np.int64) * (1 << 32) + s.astype(np.int64)
