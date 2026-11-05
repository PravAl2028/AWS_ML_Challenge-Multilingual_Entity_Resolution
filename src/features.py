"""Pair features for (Source-1 entity, Source-2/3 record) candidates.

Two groups:
* ``context_features`` - cheap, vectorised: retrieval score / rank / margins, per-query and
  per-S1 competition statistics, TF-IDF cosines, state / house-number agreement. Used by the
  prefilter and by the matcher.
* ``pair_features``    - string and token features computed per pair in parallel chunks
  (rapidfuzz ratios, IDF-weighted token overlap, "twin" features that separate true matches
  from near-copy distractors: unexplained extra name tokens and their learned / unsupervised
  marker statistics, house-number deltas, numeric-token differences, legal-form conflicts, ...).
"""
from __future__ import annotations

import collections
import math
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from . import surface as SF
from . import text_names as TN
from .io_utils import timer
from .par import G, pmap

STOP = TN.STOP | {'and'}
LEGAL_VALS = TN.LEGAL_VALS


# ============================================================ helpers
def rowdot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(ia), dtype=np.float32)
    for a in range(0, len(ia), chunk):
        x = A[ia[a:a + chunk]]
        y = B[ib[a:a + chunk]]
        out[a:a + chunk] = np.asarray(x.multiply(y).sum(axis=1)).ravel()
    return out


def _grp_rank(key: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Dense 0-based rank of ``val`` (descending) within groups of ``key``."""
    order = np.lexsort((-val, key))
    k = key[order]
    start = np.r_[0, np.nonzero(k[1:] != k[:-1])[0] + 1]
    pos = np.arange(len(k)) - np.repeat(start, np.diff(np.r_[start, len(k)]))
    out = np.empty(len(k), dtype=np.int32)
    out[order] = pos
    return out


def _grp_max_other(key: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Max of ``val`` over the OTHER rows of the same group (-1 when alone)."""
    df = pd.DataFrame({"k": key, "v": val})
    g = df.groupby("k")["v"]
    mx = g.transform("max").to_numpy()
    cnt_max = df.assign(ism=(df.v.to_numpy() == mx)).groupby("k")["ism"].transform("sum").to_numpy()
    second = df.assign(v2=np.where(df.v.to_numpy() == mx, -np.inf, df.v.to_numpy())).groupby("k")["v2"] \
        .transform("max").to_numpy()
    other = np.where((val == mx) & (cnt_max == 1), second, mx)
    other[~np.isfinite(other)] = -1.0
    return other.astype(np.float32)


def count_map(values: np.ndarray, keys_country: np.ndarray) -> dict:
    """{(country, value): count} for non-empty values."""
    df = pd.DataFrame({"c": keys_country, "v": values})
    df = df[df.v != ""]
    return df.groupby(["c", "v"]).size().to_dict()


# ============================================================ context features (vectorised)
def context_features(cand: pd.DataFrame, rec: pd.DataFrame, Xn, Xa, part_key: np.ndarray) -> pd.DataFrame:
    q = cand["q"].to_numpy()
    s = cand["s"].to_numpy()
    src = rec["source"].to_numpy()[q]
    f = pd.DataFrame(index=cand.index)
    f["score"] = cand["score"].to_numpy()
    f["rank"] = cand["rank"].to_numpy().astype(np.int16)
    f["top1"] = cand["top1"].to_numpy()
    f["gap_top"] = f["top1"] - f["score"]
    f["gap_alt"] = np.where(f["rank"].to_numpy() == 0, f["score"] - cand["second"].to_numpy(), f["score"] - f["top1"])
    f["src"] = src.astype(np.int8)
    n_q = cand.groupby("q")["s"].transform("size").to_numpy()
    f["n_cand_q"] = n_q.astype(np.int16)
    ks = s.astype(np.int64) * 4 + src
    f["s_indeg"] = pd.Series(ks).map(pd.Series(ks).value_counts()).to_numpy().astype(np.int32)
    top = f["rank"].to_numpy() == 0
    top_cnt = pd.Series(ks[top]).value_counts()
    f["s_indeg_top1"] = pd.Series(ks).map(top_cnt).fillna(0).to_numpy().astype(np.int32)
    f["s_rrank"] = _grp_rank(ks, f["score"].to_numpy()).astype(np.int32)
    with timer("TF-IDF cosines"):
        ncos = rowdot(Xn, Xn, q, s)
        acos = rowdot(Xa, Xa, q, s)
    f["n_cos"] = ncos
    f["a_cos"] = acos
    f["n_cos_rank"] = _grp_rank(q, ncos).astype(np.int16)
    f["a_cos_rank"] = _grp_rank(q, acos).astype(np.int16)
    f["n_cos_alt"] = ncos - _grp_max_other(q, ncos)
    f["a_cos_alt"] = acos - _grp_max_other(q, acos)
    f["s_n_cos_rank"] = _grp_rank(ks, ncos).astype(np.int32)
    # state agreement (after partition merges); -1 = unknown on either side
    ks_, kq_ = part_key[s], part_key[q]
    f["state_eq"] = np.where((ks_ == "") | (kq_ == ""), -1, (ks_ == kq_).astype(np.int8)).astype(np.int8)
    hq, hs = rec["a_hn"].to_numpy()[q], rec["a_hn"].to_numpy()[s]
    both = (hq >= 0) & (hs >= 0)
    f["hn_eq"] = np.where(both, (hq == hs).astype(np.int8), -1).astype(np.int8)
    d = np.abs(hq - hs).astype(np.float64)
    f["hn_logdiff"] = np.where(both, np.log1p(d), -1).astype(np.float32)
    # signed shift (query - S1): near-copy distractors carry a small POSITIVE shift, true matches are symmetric
    sd = (hq - hs).astype(np.float64)
    f["hn_sdiff"] = np.where(both, sd, np.nan).astype(np.float32)
    f["hn_rdiff"] = np.where(both, sd / np.maximum(hs, 1), np.nan).astype(np.float32)
    f["hn_shift"] = np.select([~both, sd == 0, (sd >= 1) & (sd <= 25), (sd <= -1) & (sd >= -25), sd > 25],
                              [-1, 0, 1, 2, 3], 4).astype(np.int8)
    f["q_addr_empty"] = rec["a_empty"].to_numpy()[q].astype(np.int8)
    f["q_units"] = rec["a_units"].to_numpy()[q].astype(np.int8)
    f["q_flags"] = rec["n_flags"].to_numpy()[q].astype(np.int16)
    f["via_key"] = cand["via_key"].to_numpy().astype(np.int8) if "via_key" in cand else np.int8(0)
    return f


PREFILTER_FEATURES = ["score", "rank", "top1", "gap_top", "gap_alt", "src", "n_cand_q", "s_indeg", "s_indeg_top1",
                      "s_rrank", "n_cos", "a_cos", "n_cos_rank", "a_cos_rank", "n_cos_alt", "a_cos_alt",
                      "s_n_cos_rank", "state_eq", "hn_eq", "hn_logdiff", "q_addr_empty", "q_flags", "via_key"]


# ============================================================ token statistics
def _toks(s: str) -> list:
    return [t for t in s.split() if t not in STOP] if s else []


def extra_missing(qn: str, sn: str):
    a, b = set(_toks(sn)), set(_toks(qn))
    return b - a, a - b


def token_logodds(cand: pd.DataFrame, rec: pd.DataFrame, label: np.ndarray, mask: np.ndarray,
                  min_n: int = 3, shrink: float = 10.0) -> dict:
    """Supervised log-odds that an EXTRA (in query, not in S1) / MISSING name token indicates a match."""
    qn = rec["n_core"].to_numpy(dtype=object)
    idx = np.nonzero(mask)[0]
    q, s = cand["q"].to_numpy()[idx], cand["s"].to_numpy()[idx]
    y = label[idx]
    ep, en, mp_, mn = (collections.Counter() for _ in range(4))
    for qi, si, yy in zip(q, s, y):
        e, m = extra_missing(qn[qi], qn[si])
        if yy:
            ep.update(e)
            mp_.update(m)
        else:
            en.update(e)
            mn.update(m)
    out = {}
    for name, P, N in (("extra", ep, en), ("missing", mp_, mn)):
        tp, tn = sum(P.values()) + 1.0, sum(N.values()) + 1.0
        base = math.log(tp / tn)
        tab = {}
        for w in set(P) | set(N):
            n = P[w] + N[w]
            if n < min_n:
                continue
            lo = math.log((P[w] + 0.5) / (N[w] + 0.5)) - base
            tab[w] = lo * n / (n + shrink)
        out[name] = tab
    return out


def marker_stats(cand: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """Label-free (transductive) evidence that a name token is a distractor "marker".

    For best-ranked candidates whose query name is exactly the S1 name plus ONE extra token w,
    measure how often the house numbers agree. Noise fillers ride on true matches (same house
    number); marker words ride on near-copy distractors (shifted house number).
    Returns {w: (rate_hn_equal, support)}.
    """
    top = cand["rank"].to_numpy() == 0
    q, s = cand["q"].to_numpy()[top], cand["s"].to_numpy()[top]
    qn = rec["n_core"].to_numpy(dtype=object)
    hn = rec["a_hn"].to_numpy()
    eq, tot = collections.Counter(), collections.Counter()
    for qi, si in zip(q, s):
        if hn[qi] < 0 or hn[si] < 0:
            continue
        e, m = extra_missing(qn[qi], qn[si])
        if len(e) == 1 and not m:
            w = next(iter(e))
            tot[w] += 1
            eq[w] += int(hn[qi] == hn[si])
    return {w: ((eq[w] + 1.0) / (n + 4.0), n) for w, n in tot.items() if n >= 3}


def idf_table(texts: np.ndarray) -> tuple:
    df = collections.Counter()
    n = 0
    for t in texts:
        if t:
            df.update(set(t.split()))
            n += 1
    return {w: math.log((n + 1.0) / (c + 1.0)) + 1.0 for w, c in df.items()}, math.log(n + 1.0) + 1.0


# ============================================================ per-pair features (parallel)
def _soft(ta, tb_set, tb_list, idf, dflt):
    """IDF-weighted share of ``ta`` tokens with an exact or fuzzy (JW >= .85) partner in tb."""
    if not ta:
        return 0.0, 0
    tot = hit = 0.0
    unexpl = 0
    for t in ta:
        w = idf.get(t, dflt)
        tot += w
        if t in tb_set:
            hit += w
            continue
        best = 0.0
        if len(t) >= 4:
            for u in tb_list:
                v = JaroWinkler.similarity(t, u)
                if v > best:
                    best = v
        if best >= 0.85:
            hit += w * best
        else:
            unexpl += 1
    return hit / tot, unexpl


def _numset(s: str) -> set:
    return set(s.split()) if s else set()


def _intnums(ns: set) -> list:
    out = []
    for x in ns:
        d = ''.join(ch for ch in x if ch.isdigit())
        if d and len(d) <= 9:
            out.append(int(d))
    return out


_DRUN_RE = re.compile(r'\d+')


def _comp_feats(cq: str, cs: str):
    """Compare two canonical composite house numbers ("5-7-402-a"): (equal, digits-only equal, same shape,
    #differing runs, signed delta of the single differing digit run, prefix relation)."""
    if not cq or not cs:
        return -1.0, -1.0, -1.0, -1.0, np.nan, -1.0
    rq, rs = cq.split('-'), cs.split('-')
    dq, ds = ''.join(x for x in rq if x.isdigit()), ''.join(x for x in rs if x.isdigit())
    eq, deq = float(cq == cs), float(bool(dq) and dq == ds)
    sd = np.nan
    if [x.isdigit() for x in rq] == [x.isdigit() for x in rs]:
        diff = [(a, b) for a, b in zip(rq, rs) if a != b]
        shape, nd = 1.0, float(len(diff))
        if len(diff) == 1 and diff[0][0].isdigit():
            sd = int(diff[0][0][:9]) - int(diff[0][1][:9])
    else:
        shape, nd = 0.0, -1.0
    if len(rq) > len(rs) and rq[:len(rs)] == rs:
        pre = 1.0  # query extends the S1 number (8-7-90/1 -> 8-7-90/1/0 style)
    elif len(rs) > len(rq) and rs[:len(rq)] == rq:
        pre = 2.0  # query truncates it
    else:
        pre = 0.0
    return eq, deq, shape, nd, sd, pre


def _runs_feats(nq: str, ns: str):
    """Digit runs over all address numbers: (multiset Jaccard, signed delta when exactly one run differs)."""
    a = collections.Counter(_DRUN_RE.findall(nq)) if nq else collections.Counter()
    b = collections.Counter(_DRUN_RE.findall(ns)) if ns else collections.Counter()
    if not a and not b:
        return -1.0, np.nan
    inter, union = sum((a & b).values()), sum((a | b).values())
    x, y = a - b, b - a
    sd = np.nan
    if sum(x.values()) == 1 and sum(y.values()) == 1:
        u, v = next(iter(x)), next(iter(y))
        if len(u) <= 9 and len(v) <= 9:
            sd = int(u) - int(v)
    return inter / union, sd


def _unit_feats(uq: str, us: str):
    """Unit / floor / box numbers: state (-1 none, 2 only S1, 3 only query, 1 shared, 0 disjoint), signed delta
    of the closest pair when disjoint."""
    a, b = set(uq.split()) if uq else set(), set(us.split()) if us else set()
    if not a and not b:
        return -1.0, np.nan
    if not a:
        return 2.0, np.nan
    if not b:
        return 3.0, np.nan
    if a & b:
        return 1.0, np.nan
    da = [int(d[:9]) for d in (''.join(_DRUN_RE.findall(x)) for x in a) if d]
    db = [int(d[:9]) for d in (''.join(_DRUN_RE.findall(x)) for x in b) if d]
    sd = min(((abs(x - y), x - y) for x in da for y in db))[1] if da and db else np.nan
    return 0.0, sd


PAIR_FEATURES = [
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "nf_tset", "n_concat", "n_alias_tset", "n_web",
    "n_jacc", "n_idf_rec", "n_idf_prec", "n_max_shared", "n_first_eq", "n_extra", "n_missing",
    "n_extra_idf", "n_missing_idf", "lo_extra_min", "lo_extra_max", "lo_extra_sum", "lo_miss_min",
    "mk_rate_min", "mk_support", "legal_conflict", "legal_equal", "legal_q", "legal_s", "n_len_ratio",
    "q_name_cnt", "s_name_cnt",
    "a_ratio", "a_tsort", "a_tset", "a_jacc", "a_idf_jacc", "a_idf_cov_s", "a_max_shared", "a_extra_idf",
    "st_tset", "st_type_eq", "city_best", "city_eq", "num_jacc", "num_new_q", "num_miss_q", "num_min_absdiff",
    "num_digit_ed", "hn_ed", "hn_suf_eq", "s_addr_cnt", "q_addr_cnt", "a_len_ratio",
    "num_sdiff", "num_one_sdiff", "num_one_up", "legal_dir", "legal_exact",
    "hnc_eq", "hnc_digits_eq", "hnc_shape_eq", "hnc_ndiff", "hnc_sdiff", "hnc_prefix", "hnl_sdiff", "runs_jacc",
    "runs_one_sdiff", "runs_one_up", "unit_state", "unit_sdiff",
    "n_short_miss", "n_short_sub1", "n_short_extra", "lo_cov",
] + SF.SURFACE_FEATURES
LO_FEATURES = ("lo_extra_min", "lo_extra_max", "lo_extra_sum", "lo_miss_min", "lo_cov")
# signed features: missing = NaN (a sentinel would collide with a real value)
_NAN_INIT = {"num_sdiff", "num_one_sdiff", "hnc_sdiff", "hnl_sdiff", "runs_one_sdiff", "unit_sdiff"}


_STR_COLS = ("n_full", "n_core", "n_alias", "n_web", "n_legal", "a_tok", "a_street", "a_stype", "a_city",
             "a_nums", "a_hnsuf", "a_hnc", "a_unit", "name", "addr")  # name / addr: raw strings (surface features)


def _take(col, idx):
    return col.take(pa.array(idx)).to_pylist()


def _pair_chunk(task):
    q_all, s_all, lo_idx = task
    g = G()
    n = len(q_all)
    # gather the strings of this chunk from shared Arrow buffers (no per-object refcount traffic,
    # so forked workers do not copy the parent's memory pages)
    cols = {}
    for k in _STR_COLS:
        cols["q_" + k] = _take(g[k], q_all)
        cols["s_" + k] = _take(g[k], s_all)
    hq_all, hs_all = g["a_hn"][q_all], g["a_hn"][s_all]
    lq_all, ls_all = g["a_hnl"][q_all], g["a_hnl"][s_all]
    cc_all = g["country"][s_all]
    idf_n, dn = g["idf_n"]
    idf_a, da = g["idf_a"]
    lo_tabs, mk = g["lo_tabs"], g["mk"]
    name_cnt, addr_cnt = g["name_cnt"], g["addr_cnt"]
    F = {k: np.full(n, np.nan if k in _NAN_INIT else 0.0, dtype=np.float32) for k in PAIR_FEATURES}
    qnc, snc = cols["q_n_core"], cols["s_n_core"]
    qnf, snf = cols["q_n_full"], cols["s_n_full"]
    qa, sa = cols["q_a_tok"], cols["s_a_tok"]
    F["n_ratio"][:] = process.cpdist(qnf, snf, scorer=fuzz.ratio, workers=1)
    F["n_tsort"][:] = process.cpdist(qnc, snc, scorer=fuzz.token_sort_ratio, workers=1)
    F["n_tset"][:] = process.cpdist(qnc, snc, scorer=fuzz.token_set_ratio, workers=1)
    F["n_partial"][:] = process.cpdist(qnc, snc, scorer=fuzz.partial_ratio, workers=1)
    F["n_jw"][:] = process.cpdist(qnc, snc, scorer=JaroWinkler.normalized_similarity, workers=1)
    F["nf_tset"][:] = process.cpdist(qnf, snf, scorer=fuzz.token_set_ratio, workers=1)
    F["n_concat"][:] = process.cpdist([x.replace(' ', '') for x in qnc], [x.replace(' ', '') for x in snc],
                                      scorer=fuzz.ratio, workers=1)
    F["a_ratio"][:] = process.cpdist(qa, sa, scorer=fuzz.ratio, workers=1)
    F["a_tsort"][:] = process.cpdist(qa, sa, scorer=fuzz.token_sort_ratio, workers=1)
    F["a_tset"][:] = process.cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=1)
    qraw, sraw, qaraw, saraw = cols["q_name"], cols["s_name"], cols["q_addr"], cols["s_addr"]
    qst, sst = cols["q_a_street"], cols["s_a_street"]
    F["st_tset"][:] = process.cpdist(qst, sst, scorer=fuzz.token_set_ratio, workers=1)
    for r in range(n):
        # ---------------- name tokens
        ta, tb = _toks(snc[r]), _toks(qnc[r])
        sa_, sb_ = set(ta), set(tb)
        un = sa_ | sb_
        F["n_jacc"][r] = len(sa_ & sb_) / len(un) if un else 0.0
        rec_, miss = _soft(ta, sb_, tb, idf_n, dn)
        prec_, extra = _soft(tb, sa_, ta, idf_n, dn)
        F["n_idf_rec"][r], F["n_missing"][r] = rec_, miss
        F["n_idf_prec"][r], F["n_extra"][r] = prec_, extra
        shared = sa_ & sb_
        F["n_max_shared"][r] = max((idf_n.get(t, dn) for t in shared), default=0.0)
        F["n_first_eq"][r] = float(bool(ta) and bool(tb) and ta[0] == tb[0])
        # short tokens (acronyms, initials, titles, roman numerals): near-copy distractors change or drop them
        sm = [t for t in sa_ - sb_ if len(t) <= 4 and t.isalpha()]
        se = [t for t in sb_ - sa_ if len(t) <= 4 and t.isalpha()]
        F["n_short_miss"][r] = len(sm)
        F["n_short_extra"][r] = len(se)
        F["n_short_sub1"][r] = sum(1 for t in sm if any(Levenshtein.distance(t, u, score_cutoff=1) <= 1 for u in se))
        ex, ms = sb_ - sa_, sa_ - sb_
        F["n_extra_idf"][r] = sum(idf_n.get(t, dn) for t in ex)
        F["n_missing_idf"][r] = sum(idf_n.get(t, dn) for t in ms)
        lt = lo_tabs[lo_idx[r]]
        le, lm = lt["extra"], lt["missing"]
        vals = [le.get(t, 0.0) for t in ex]
        F["lo_extra_min"][r] = min(vals) if vals else 0.0
        F["lo_extra_max"][r] = max(vals) if vals else 0.0
        F["lo_extra_sum"][r] = sum(vals)
        vm = [lm.get(t, 0.0) for t in ms]
        F["lo_miss_min"][r] = min(vm) if vm else 0.0
        # share of the differing tokens that the supervised table knows (low for an unseen country's vocabulary)
        F["lo_cov"][r] = (sum(t in le for t in ex) + sum(t in lm for t in ms)) / (len(ex) + len(ms)) if (ex or ms)             else -1.0
        mr = [mk[t] for t in ex if t in mk]
        F["mk_rate_min"][r] = min(x[0] for x in mr) if mr else (-1.0 if not ex else 0.3)
        F["mk_support"][r] = max((x[1] for x in mr), default=0)
        # alias parts / website stems
        al = cols["q_n_alias"][r]
        if al:
            F["n_alias_tset"][r] = max(fuzz.token_set_ratio(p, snc[r]) for p in al.split('|'))
        else:
            F["n_alias_tset"][r] = -1.0
        wq = cols["q_n_web"][r]
        if wq:
            F["n_web"][r] = TN.web_match(wq, snf[r].split())
        else:
            F["n_web"][r] = -1.0
        lq, ls = cols["q_n_legal"][r], cols["s_n_legal"][r]
        fq = {TN.LEGAL_FAMILY.get(x, x) for x in lq.split()} if lq else set()
        fs = {TN.LEGAL_FAMILY.get(x, x) for x in ls.split()} if ls else set()
        F["legal_conflict"][r] = float(bool(fq) and bool(fs) and not (fq & fs))
        F["legal_equal"][r] = float(fq == fs)
        # direction of the legal-form change: 0 none, 1 same family, 2 added by q, 3 dropped by q, 4 changed
        F["legal_dir"][r] = 0 if not (fq or fs) else 1 if fq == fs else 2 if not fs else 3 if not fq else 4
        F["legal_exact"][r] = float(lq == ls)
        F["legal_q"][r] = float(bool(fq))
        F["legal_s"][r] = float(bool(fs))
        lq_, ls_ = len(qnc[r]), len(snc[r])
        F["n_len_ratio"][r] = min(lq_, ls_) / max(1, lq_, ls_)
        c = cc_all[r]
        F["q_name_cnt"][r] = name_cnt.get((c, qnc[r]), 0)
        F["s_name_cnt"][r] = name_cnt.get((c, snc[r]), 0)
        # ---------------- address tokens
        xa, xb = qa[r].split(), sa[r].split()
        A, B = set(xb), set(xa)  # A = S1 tokens, B = query tokens
        ua = A | B
        inter = A & B
        F["a_jacc"][r] = len(inter) / len(ua) if ua else -1.0
        wi = sum(idf_a.get(t, da) for t in inter)
        wu = sum(idf_a.get(t, da) for t in ua)
        wa = sum(idf_a.get(t, da) for t in A)
        F["a_idf_jacc"][r] = wi / wu if wu else -1.0
        F["a_idf_cov_s"][r] = wi / wa if wa else -1.0
        F["a_max_shared"][r] = max((idf_a.get(t, da) for t in inter), default=0.0)
        F["a_extra_idf"][r] = sum(idf_a.get(t, da) for t in B - A)
        tq, ts = cols["q_a_stype"][r], cols["s_a_stype"][r]
        F["st_type_eq"][r] = -1.0 if not (tq and ts) else float(tq == ts)
        cq, cs = cols["q_a_city"][r], cols["s_a_city"][r]
        if cq and cs:
            pq, ps = cq.split('|'), cs.split('|')
            F["city_best"][r] = max(fuzz.ratio(x, y) for x in pq for y in ps)
            F["city_eq"][r] = float(bool(set(pq) & set(ps)))
        else:
            F["city_best"][r] = -1.0
            F["city_eq"][r] = -1.0
        nq, ns = _numset(cols["q_a_nums"][r]), _numset(cols["s_a_nums"][r])
        if nq or ns:
            F["num_jacc"][r] = len(nq & ns) / len(nq | ns)
        else:
            F["num_jacc"][r] = -1.0
        F["num_new_q"][r] = len(nq - ns)
        F["num_miss_q"][r] = len(ns - nq)
        dq, ds = _intnums(nq - ns), _intnums(ns - nq)
        if dq and ds:
            F["num_min_absdiff"][r] = math.log1p(min(abs(x - y) for x in dq for y in ds))
            F["num_digit_ed"][r] = min(Levenshtein.distance(str(x), str(y)) for x in dq for y in ds)
            F["num_sdiff"][r] = min(((abs(x - y), x - y) for x in dq for y in ds))[1]
            if len(dq) == 1 and len(ds) == 1 and len(nq - ns) == 1 and len(ns - nq) == 1:
                F["num_one_sdiff"][r] = dq[0] - ds[0]
                F["num_one_up"][r] = float(1 <= dq[0] - ds[0] <= 25)
        else:
            F["num_min_absdiff"][r] = -1.0
            F["num_digit_ed"][r] = -1.0
        hq, hs = hq_all[r], hs_all[r]
        F["hn_ed"][r] = Levenshtein.distance(str(hq), str(hs)) if (hq >= 0 and hs >= 0) else -1.0
        F["hn_suf_eq"][r] = float(cols["q_a_hnsuf"][r] == cols["s_a_hnsuf"][r])
        (F["hnc_eq"][r], F["hnc_digits_eq"][r], F["hnc_shape_eq"][r], F["hnc_ndiff"][r], F["hnc_sdiff"][r],
         F["hnc_prefix"][r]) = _comp_feats(cols["q_a_hnc"][r], cols["s_a_hnc"][r])
        if lq_all[r] >= 0 and ls_all[r] >= 0:
            F["hnl_sdiff"][r] = lq_all[r] - ls_all[r]
        F["runs_jacc"][r], F["runs_one_sdiff"][r] = _runs_feats(cols["q_a_nums"][r], cols["s_a_nums"][r])
        F["runs_one_up"][r] = float(1 <= F["runs_one_sdiff"][r] <= 25)
        F["unit_state"][r], F["unit_sdiff"][r] = _unit_feats(cols["q_a_unit"][r], cols["s_a_unit"][r])
        F["s_addr_cnt"][r] = addr_cnt.get((c, sa[r]), 0)
        F["q_addr_cnt"][r] = addr_cnt.get((c, qa[r]), 0)
        la_, lb_ = len(qa[r]), len(sa[r])
        F["a_len_ratio"][r] = min(la_, lb_) / max(1, la_, lb_)
        # ---------------- raw-surface noise (before any folding): true matches carry more of it than near-copies
        for k, v in zip(SF.SURFACE_FEATURES, SF.pair_surface(qraw[r], sraw[r], qaraw[r], saraw[r])):
            F[k][r] = v
    return F


def pair_features(cand: pd.DataFrame, rec: pd.DataFrame, lo_tabs: list, lo_idx: np.ndarray, mk: dict,
                  n_jobs: int, chunk: int = 100_000) -> pd.DataFrame:
    """Compute PAIR_FEATURES for every candidate row (parallel over row chunks)."""
    s1 = rec["source"].to_numpy() == 1
    ctry = rec["country"].cat.codes.to_numpy().astype(np.int16)
    with timer("token IDF tables + S1 name/address counts"):
        idf_n = idf_table(rec["n_core"].to_numpy(dtype=object))
        idf_a = idf_table(rec["a_tok"].to_numpy(dtype=object))
        name_cnt = count_map(rec["n_core"].to_numpy(dtype=object)[s1], ctry[s1])
        addr_cnt = count_map(rec["a_tok"].to_numpy(dtype=object)[s1], ctry[s1])
    shared = {k: pa.array((rec[k].fillna("") if k in rec else pd.Series([""] * len(rec))).to_numpy(dtype=object),
                          type=pa.large_string()) for k in _STR_COLS}
    shared["a_hn"] = rec["a_hn"].to_numpy()
    shared["a_hnl"] = rec["a_hnl"].to_numpy() if "a_hnl" in rec else rec["a_hn"].to_numpy()
    shared.update(country=ctry, idf_n=idf_n, idf_a=idf_a, lo_tabs=lo_tabs, mk=mk, name_cnt=name_cnt,
                  addr_cnt=addr_cnt)
    q, s = cand["q"].to_numpy(), cand["s"].to_numpy()
    tasks = [(q[a:a + chunk], s[a:a + chunk], lo_idx[a:a + chunk]) for a in range(0, len(q), chunk)]
    with timer(f"pair features for {len(q)} pairs ({len(tasks)} chunks)"):
        parts = pmap(_pair_chunk, tasks, n_jobs, shared=shared)
    return pd.DataFrame({k: np.concatenate([p[k] for p in parts]) for k in PAIR_FEATURES}, index=cand.index)
