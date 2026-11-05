"""Transductive (label-free) word-role statistics per country (v4).

Why: the matcher learns which name-token changes are harmless noise from the supervised token log-odds, which only
know the training countries' vocabularies. On an unseen country (France in the test) it cannot tell a noise filler
("groupe", "developpement") from a word swapped by a "different business at the same address" distractor
("club" -> "sport"), and it gets both wrong.

What: the data generator gives the two kinds of word different address behaviour. Noise words ride on true matches,
which keep the S1's house number. Distractor vocabulary rides on near-copies, whose house number is shifted by
+1..+25 (the twin signature), or on same-address look-alikes. So, for every (country, token), over best-retrieved
candidate pairs on the same street whose house numbers are equal or shifted by +1..+25, we count how often the
token takes part in the name difference while the address is the same. Four contexts are kept apart because the
same word can play different roles in them (US "partners" is a twin suffix when added, a noise filler when it
replaces a word):

    A  addition     query = S1 + extra tokens            (token is extra)
    D  deletion     query = S1 - missing tokens          (token is missing)
    E  replacement  query has extra AND missing tokens   (token is extra)
    M  replacement                                       (token is missing)

Rates are shrunk towards, and expressed relative to, the country's pooled rate of the same context (logit
difference), so train and test, whose distractor loads differ, stay comparable. Pairs that contribute to the
counts are scored leave-one-out, so a row never sees its own address relation through these features.

No labels are used anywhere: the statistics of each split come from that split's own candidate pairs, as for the
existing marker statistics (``features.marker_stats``).
"""
from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from . import features as FT

ROLE_FEATURES = ["ro_ctx", "ro_e_min", "ro_e_max", "ro_e_sup", "ro_m_min", "ro_m_max", "ro_m_sup"]
# cross-context scores (v6 experiment): every extra token scored in BOTH the addition and the replacement context,
# every missing token in both the deletion and the replacement context, whatever the pair's own context
CROSS_FEATURES = ["ro_ea_min", "ro_ee_min", "ro_ea_sup", "ro_ee_sup", "ro_md_min", "ro_mm_min"]
CTX = ("A", "D", "E", "M")
SHRINK = 2.0
# Reference rate a token is compared with. "typo": the country's same-address rate among one-token replacements that
# are plain typos (Jaro-Winkler >= 0.85), i.e. the rate of pure noise, which does not depend on how many near-copy
# distractors a split carries or on which vocabulary dominates a country's name changes. "base": the pooled rate of
# the context (all tokens), which mixes noise and distractor vocabulary in country-specific proportions.
REF = "typo"
TYPO_JW, TYPO_MINLEN, TYPO_MIN_N = 0.85, 4, 30


def _rel(hq, hs, cq, cs, sq, ss) -> int:
    """0 = same address (same street, equal canonical house number), 1 = near-copy shift (same street, house number
    or a single composite digit run shifted by +1..+25), -1 = anything else / unknown."""
    if not sq or sq != ss or not cq or not cs:
        return -1
    if cq == cs:
        return 0
    _, _, shape, nd, sd, _ = FT._comp_feats(cq, cs)
    if shape == 1.0 and nd == 1.0 and sd == sd and 1 <= sd <= 25:
        return 1
    if hq >= 0 and hs >= 0 and 1 <= hq - hs <= 25 and '-' not in cq and '-' not in cs:
        return 1
    return -1


def _diff(qn: str, sn: str):
    e, m = FT.extra_missing(qn, sn)
    return e, m


def _ctx_of(e, m) -> int:
    """0 no difference, 1 addition, 2 deletion, 3 replacement"""
    if e and m:
        return 3
    if e:
        return 1
    if m:
        return 2
    return 0


def role_tables(cand: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """Counts {(ctx, country, token): [eq, up]} over rank-0 candidate pairs, plus pooled base rates per (ctx, country).
    Also returns the set of contributing row indices with their (rel) so they can be scored leave-one-out."""
    rank = cand["rank"].to_numpy()
    idx = np.nonzero(rank == 0)[0]
    q, s = cand["q"].to_numpy()[idx], cand["s"].to_numpy()[idx]
    qn = rec["n_core"].fillna("").to_numpy(dtype=object)
    hn = rec["a_hn"].to_numpy()
    hc = rec["a_hnc"].fillna("").to_numpy(dtype=object)
    st = rec["a_street"].fillna("").to_numpy(dtype=object)
    ctry = rec["country"].astype(str).to_numpy(dtype=object)
    cnt = collections.defaultdict(lambda: [0, 0])
    base = collections.defaultdict(lambda: [0, 0])
    typo = collections.defaultdict(lambda: [0, 0])
    contrib = {}
    for i, qi, si in zip(idx, q, s):
        r = _rel(hn[qi], hn[si], hc[qi], hc[si], st[qi], st[si])
        if r < 0:
            continue
        e, m = _diff(qn[qi], qn[si])
        k = _ctx_of(e, m)
        if k == 0:
            continue
        c = ctry[si]
        ce, cm = ("A", "D") if k != 3 else ("E", "M")
        for w in e:
            cnt[(ce, c, w)][r] += 1
        for w in m:
            cnt[(cm, c, w)][r] += 1
        base[(ce if e else cm, c)][r] += 1
        if e and m:
            base[(cm, c)][r] += 1
            if len(e) == 1 and len(m) == 1:
                a, b = next(iter(e)), next(iter(m))
                if len(a) >= TYPO_MINLEN and len(b) >= TYPO_MINLEN and JaroWinkler.similarity(a, b) >= TYPO_JW:
                    typo[c][r] += 1
        contrib[int(i)] = r
    base_rate = {k: (v[0] + 1.0) / (v[0] + v[1] + 2.0) for k, v in base.items()}
    if REF == "typo":
        # one noise reference per country, used for every context (falls back to the pooled rate when too few typos)
        ref = {c: (v[0] + 1.0) / (v[0] + v[1] + 2.0) for c, v in typo.items() if v[0] + v[1] >= TYPO_MIN_N}
        base_rate = {k: ref.get(k[1], b) for k, b in base_rate.items()}
    return {"cnt": dict(cnt), "base": base_rate, "contrib": contrib,
            "typo": {c: tuple(v) for c, v in typo.items()}}


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _score(tab: dict, ctx: str, c: str, w: str, own: int):
    """Normalised role score of token w (logit(rate) - logit(base)); own = the row's own relation to leave out
    (-1 = none). Returns (score, support) or (nan, 0) when nothing is known."""
    b = tab["base"].get((ctx, c))
    v = tab["cnt"].get((ctx, c, w))
    if b is None or v is None:
        return math.nan, 0
    eq, up = v[0], v[1]
    if own == 0:
        eq -= 1
    elif own == 1:
        up -= 1
    n = eq + up
    if n <= 0:
        return math.nan, 0
    r = (eq + SHRINK * b) / (n + SHRINK)
    return _logit(r) - _logit(b), n


def cross_features(cand: pd.DataFrame, rec: pd.DataFrame, tab: dict | None = None) -> pd.DataFrame:
    """CROSS_FEATURES for every row of ``cand`` (see ``role_features``)."""
    if tab is None:
        tab = role_tables(cand, rec)
    contrib = tab["contrib"]
    q, s = cand["q"].to_numpy(), cand["s"].to_numpy()
    qn = rec["n_core"].fillna("").to_numpy(dtype=object)
    ctry = rec["country"].astype(str).to_numpy(dtype=object)
    n = len(cand)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in CROSS_FEATURES}
    out["ro_ea_sup"][:] = 0
    out["ro_ee_sup"][:] = 0
    for i in range(n):
        qi, si = q[i], s[i]
        e, m = _diff(qn[qi], qn[si])
        if not e and not m:
            continue
        own = contrib.get(i, -1)
        k = _ctx_of(e, m)
        c = ctry[si]
        for ctx, col, sup in (("A", "ro_ea_min", "ro_ea_sup"), ("E", "ro_ee_min", "ro_ee_sup")):
            if not e:
                continue
            own_c = own if (ctx == "A") == (k != 3) else -1  # leave-one-out only in the row's own context
            sc = [_score(tab, ctx, c, w, own_c) for w in e]
            vals = [(x, nn) for x, nn in sc if x == x]
            if vals:
                x, nn = min(vals)
                out[col][i] = x
                out[sup][i] = math.log1p(nn)
        for ctx, col in (("D", "ro_md_min"), ("M", "ro_mm_min")):
            if not m:
                continue
            own_c = own if (ctx == "D") == (k != 3) else -1
            vals = [x for x, _ in (_score(tab, ctx, c, w, own_c) for w in m) if x == x]
            if vals:
                out[col][i] = min(vals)
    return pd.DataFrame(out, index=cand.index)


def role_features(cand: pd.DataFrame, rec: pd.DataFrame, tab: dict | None = None) -> pd.DataFrame:
    """ROLE_FEATURES for every row of ``cand`` (columns q, s, rank). ``tab`` defaults to the tables of ``cand``
    itself (transductive, per split)."""
    if tab is None:
        tab = role_tables(cand, rec)
    contrib = tab["contrib"]
    q, s = cand["q"].to_numpy(), cand["s"].to_numpy()
    qn = rec["n_core"].fillna("").to_numpy(dtype=object)
    ctry = rec["country"].astype(str).to_numpy(dtype=object)
    n = len(cand)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ROLE_FEATURES}
    out["ro_ctx"][:] = 0
    out["ro_e_sup"][:] = 0
    out["ro_m_sup"][:] = 0
    cache = {}
    for i in range(n):
        qi, si = q[i], s[i]
        own = contrib.get(i, -1)
        key = (qi, si)
        e, m = _diff(qn[qi], qn[si])
        k = _ctx_of(e, m)
        out["ro_ctx"][i] = k
        if k == 0:
            continue
        c = ctry[si]
        ce, cm = ("A", "D") if k != 3 else ("E", "M")
        if e:
            sc = [_score(tab, ce, c, w, own) for w in e]
            vals = [x for x, _ in sc if x == x]
            if vals:
                j = int(np.argmin([x if x == x else np.inf for x, _ in sc]))
                out["ro_e_min"][i] = min(vals)
                out["ro_e_max"][i] = max(vals)
                out["ro_e_sup"][i] = math.log1p(sc[j][1])
        if m:
            sc = [_score(tab, cm, c, w, own) for w in m]
            vals = [x for x, _ in sc if x == x]
            if vals:
                j = int(np.argmin([x if x == x else np.inf for x, _ in sc]))
                out["ro_m_min"][i] = min(vals)
                out["ro_m_max"][i] = max(vals)
                out["ro_m_sup"][i] = math.log1p(sc[j][1])
        del key
    del cache
    return pd.DataFrame(out, index=cand.index)


ALL_FEATURES = ROLE_FEATURES + CROSS_FEATURES


def all_features(cand: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """ROLE_FEATURES + CROSS_FEATURES from one set of (transductive, per split) tables."""
    tab = role_tables(cand, rec)
    return pd.concat([role_features(cand, rec, tab), cross_features(cand, rec, tab)], axis=1)

