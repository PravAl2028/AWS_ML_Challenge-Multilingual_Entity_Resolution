"""Same-address noise-filler rule (v7): label-free, per split and per country.

Why: the data generator changes a name at the S1's own address in two ways. Noise on a true match replaces one name
word with a *filler* drawn from a short list (US/India: center, services, service, partners; the unseen country has
its own list) or appends one, and keeps the record's legal form. A "different business at the same address"
distractor is a separate entity whose name swaps a *vocabulary* word and whose legal form is drawn afresh. The
supervised token log-odds only know the training countries' fillers, and the transductive word-role features
(``roles.py``) mix a filler's noise role with its twin-suffix role (twins append / swap in the same words at a shifted
house number), so the matchers reject the unseen country's fillers when they double as twin suffixes.

What: for every (country, token) the rule counts, over the split's best-ranked candidate pairs at the same address
(same street, equal canonical house number) whose names differ by exactly one replaced token, how often the pair
keeps the legal form when the token is the one swapped in (same address = same street, equal canonical house number and
no address number the S1 lacks). Fillers keep it like pure noise does (<= 2% changes in
US, India and the test's third country); swapped-in vocabulary changes it in 14-30% of pairs (a different entity).
A token with enough support and a legal-form change rate <= ``R_MAX`` is a filler of that country. A candidate pair
at the same address whose query differs from the S1 name by one filler (replacing one word, or appended), with the
same legal form and a filler the supervised token table does not know, gets ``p = max(p, P_FLOOR)`` before decoding.

No labels are used: the table of each split comes from that split's own candidates (like ``marker_stats`` and the
word-role tables). The rule is generic (no word lists, no country names); on the training countries the fillers are
known to the token table, so it leaves the in-country scores unchanged.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from . import features as FT

N_MIN = 30        # support: same-address one-token replacements that swap the token in
R_MAX = 0.05      # max share of those pairs whose legal form differs
P_FLOOR = 0.95    # score given to a rule pair (the decoder still decides with the other candidates)


def _fields(rec: pd.DataFrame):
    return (rec["n_core"].fillna("").to_numpy(dtype=object), rec["n_legal"].fillna("").to_numpy(dtype=object),
            rec["a_hn"].to_numpy(), rec["a_hnc"].fillna("").to_numpy(dtype=object),
            rec["a_street"].fillna("").to_numpy(dtype=object),
            (rec["a_nums"].fillna("") + " " + rec["a_unit"].fillna("")).to_numpy(dtype=object),
            rec["country"].astype(str).to_numpy(dtype=object))


def _street_near(a: str, b: str) -> bool:
    """Same street up to noise: equal, or at most two character edits apart (both >= 6 characters), or one street's
    words contain the other's (a misspelt street type or a stray letter such as the "b" of "13 b rue ..." left in the
    street; the shorter side must keep a word of >= 4 characters)."""
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) >= 6 and Levenshtein.distance(a, b, score_cutoff=2) <= 2:
        return True
    ta, tb = set(a.split()), set(b.split())
    small = ta if len(ta) <= len(tb) else tb
    return (ta <= tb or tb <= ta) and any(len(w) >= 4 for w in small)


def _same_address(qi, si, hn, hc, st, nums) -> bool:
    """Same street (up to noise), equal canonical house number, and no address or unit number of the record that the
    S1 lacks (a planted near-copy that keeps the house number can still shift a flat / plot / unit / secondary
    composite number)."""
    if not hc[qi] or hc[qi] != hc[si] or not _street_near(st[qi], st[si]):
        return False
    return set(nums[qi].split()) <= set(nums[si].split())


def filler_table(cand: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """{(country, token): [keep, change]} over rank-0 same-address pairs with a one-token replacement."""
    rank = cand["rank"].to_numpy()
    idx = np.nonzero(rank == 0)[0]
    q, s = cand["q"].to_numpy()[idx], cand["s"].to_numpy()[idx]
    qn, lg, hn, hc, st, nums, ctry = _fields(rec)
    tab = collections.defaultdict(lambda: [0, 0])
    for qi, si in zip(q, s):
        if not _same_address(qi, si, hn, hc, st, nums):
            continue
        e, m = FT.extra_missing(qn[qi], qn[si])
        if len(e) != 1 or len(m) != 1:
            continue
        tab[(ctry[si], next(iter(e)))][0 if lg[qi] == lg[si] else 1] += 1
    return dict(tab)


def filler_set(tab: dict, n_min: int = N_MIN, r_max: float = R_MAX) -> set:
    """(country, token) pairs that behave like noise fillers."""
    out = set()
    for k, (keep, chg) in tab.items():
        n = keep + chg
        if n >= n_min and (chg + 0.5) / (n + 1.0) <= r_max:
            out.add(k)
    return out


def rule_mask(pairs: pd.DataFrame, rec: pd.DataFrame, fill: set, known: set | None = None) -> np.ndarray:
    """True for candidate pairs (columns q, s) the rule applies to (see module doc). ``known``: extra tokens of the
    supervised token table (``lo_full["extra"]``); a filler it knows is left to the matchers."""
    q, s = pairs["q"].to_numpy(), pairs["s"].to_numpy()
    qn, lg, hn, hc, st, nums, ctry = _fields(rec)
    words = {w for _, w in fill}
    out = np.zeros(len(pairs), bool)
    # cheap pre-selection: same legal form and equal canonical house number (the street is checked below)
    cand = np.nonzero((lg[q] == lg[s]) & (hc[q] == hc[s]) & (hc[q] != "") & (st[q] != "") & (st[s] != ""))[0]
    for i in cand:
        qi, si = q[i], s[i]
        e, m = FT.extra_missing(qn[qi], qn[si])
        if len(e) != 1 or len(m) > 1:
            continue
        w = next(iter(e))
        if w not in words or (ctry[si], w) not in fill:
            continue
        if known is not None and w in known:
            continue
        if not _same_address(qi, si, hn, hc, st, nums):
            continue
        out[i] = True
    # a query that qualifies with two S1 (several businesses at one address) is ambiguous: leave it to the matchers
    hit = np.nonzero(out)[0]
    if len(hit):
        vc = pd.Series(q[hit]).value_counts()
        out[hit[np.isin(q[hit], vc.index[vc.to_numpy() > 1])]] = False
    return out


def apply(scores: pd.DataFrame, cand: pd.DataFrame, rec: pd.DataFrame, known: set | None, col: str = "p2",
          p_floor: float = P_FLOOR, log=None) -> pd.DataFrame:
    """Return ``scores`` with ``col`` raised to ``p_floor`` on rule pairs (table from ``cand``'s rank-0 pairs)."""
    tab = filler_table(cand, rec)
    fill = filler_set(tab)
    mask = rule_mask(scores, rec, fill, known)
    out = scores.copy()
    p = out[col].to_numpy().astype(np.float64)
    changed = mask & (p < p_floor)
    p[mask] = np.maximum(p[mask], p_floor)
    out[col] = p.astype(scores[col].dtype)
    if log is not None:
        by_c = collections.Counter(c for c, _ in fill)
        log.info("filler rule: %d fillers per country %s; %d rule pairs, %d raised to %.2f", len(fill), dict(by_c),
                 int(mask.sum()), int(changed.sum()), p_floor)
        ctry = rec["country"].astype(str).to_numpy(dtype=object)
        for c in sorted(by_c):
            ws = sorted((w for cc, w in fill if cc == c), key=lambda w: -sum(tab[(c, w)]))
            log.info("  %s fillers: %s", c, ", ".join(f"{w} ({tab[(c, w)][0]}/{sum(tab[(c, w)])})" for w in ws[:25]))
        sc = ctry[out["s"].to_numpy()]
        for c in sorted(set(sc[changed])):
            log.info("  %s: %d pairs raised", c, int((changed & (sc == c)).sum()))
    return out


# ---------------------------------------------------------------------------------------------------------------
# v8b: the other side of the same statistic. A swapped-in word whose same-address one-token replacements change the
# legal form as often as a freshly drawn one (a "different business at the same address" distractor draws its legal
# form afresh; noise keeps it) is *vocabulary* of that country. Per country from the split's own candidates, so a
# word can be a filler in one country (US "service", "center") and vocabulary in another (French "service",
# "centre", "club", "comite"). Same-address pairs that swap in such a word get p = min(p, P_CAP).
V_MIN = 50        # support
V_RATE = 0.12     # min share of those pairs whose legal form differs (fillers: <= 0.02; vocabulary: 0.16-0.24)
V_LEN = 3         # min token length (single letters / legal-form remnants are left to the matchers)
P_CAP = 0.05


def vocab_set(tab: dict, n_min: int = V_MIN, rate: float = V_RATE, min_len: int = V_LEN) -> set:
    out = set()
    for (c, w), (keep, chg) in tab.items():
        n = keep + chg
        if n >= n_min and len(w) >= min_len and w.isalpha() and (chg + 0.5) / (n + 1.0) >= rate:
            out.add((c, w))
    return out


def cap_mask(pairs: pd.DataFrame, rec: pd.DataFrame, vocab: set) -> np.ndarray:
    """True for candidate pairs at the same address whose query replaces one S1 word by a vocabulary word."""
    q, s = pairs["q"].to_numpy(), pairs["s"].to_numpy()
    qn, lg, hn, hc, st, nums, ctry = _fields(rec)
    words = {w for _, w in vocab}
    out = np.zeros(len(pairs), bool)
    cand = np.nonzero((hc[q] == hc[s]) & (hc[q] != "") & (st[q] != "") & (st[s] != ""))[0]
    for i in cand:
        qi, si = q[i], s[i]
        e, m = FT.extra_missing(qn[qi], qn[si])
        if len(e) != 1 or len(m) != 1:
            continue
        w = next(iter(e))
        if w not in words or (ctry[si], w) not in vocab:
            continue
        if not _same_address(qi, si, hn, hc, st, nums):
            continue
        out[i] = True
    return out


def apply_cap(scores: pd.DataFrame, cand: pd.DataFrame, rec: pd.DataFrame, col: str = "p2", p_cap: float = P_CAP,
              log=None) -> pd.DataFrame:
    """Return ``scores`` with ``col`` capped at ``p_cap`` on same-address vocabulary swaps (table from ``cand``)."""
    tab = filler_table(cand, rec)
    vocab = vocab_set(tab)
    mask = cap_mask(scores, rec, vocab)
    out = scores.copy()
    p = out[col].to_numpy().astype(np.float64)
    changed = mask & (p > p_cap)
    p[mask] = np.minimum(p[mask], p_cap)
    out[col] = p.astype(scores[col].dtype)
    if log is not None:
        by_c = collections.Counter(c for c, _ in vocab)
        log.info("vocabulary cap: %d vocabulary words per country %s; %d cap pairs, %d lowered to %.2f", len(vocab),
                 dict(by_c), int(mask.sum()), int(changed.sum()), p_cap)
        ctry = rec["country"].astype(str).to_numpy(dtype=object)
        for c in sorted(by_c):
            ws = sorted((w for cc, w in vocab if cc == c), key=lambda w: -sum(tab[(c, w)]))
            log.info("  %s vocabulary: %s", c, ", ".join(f"{w} ({tab[(c, w)][1]}/{sum(tab[(c, w)])})" for w in ws[:30]))
        sc = ctry[out["s"].to_numpy()]
        for c in sorted(set(sc[changed])):
            log.info("  %s: %d pairs lowered", c, int((changed & (sc == c)).sum()))
    return out
