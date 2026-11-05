"""Raw-surface properties of a business name (before any case / accent / space folding).

Source-1 names are clean (title case, single spaces, ASCII). Queries carry random surface noise -- UPPER case,
accents, double spaces, punctuation -- and true matches carry more of it than planted near-copies ("twins"),
which tend to reproduce the S1's raw spelling apart from a changed detail (round-2 brief, fact F3). Every name
feature after ``text_names.parse_name`` is folded, so these properties are computed from the raw strings.
"""
from __future__ import annotations

import collections
import re
import unicodedata

from .text_names import INDIC_RX, LEGAL

# legal-form words in any raw spelling: "LLC", "L.L.C.", "Pvt.", "Private Limited", "S.A.R.L." ...
_LEGAL_RX = re.compile(r"(?i)(?<![\w])(?:" + "|".join(
    r"(?:\.\s?)?".join(re.escape(ch) for ch in w) for w in sorted(LEGAL, key=len, reverse=True)) + r")\.?(?![\w])")
_SEP_RX = re.compile(r"\s*[,;]\s*$|^\s*[,;]\s*")
_MULTI_SP = re.compile(r" {2,}")
CASE_NONE, CASE_LOWER, CASE_UPPER, CASE_TITLE, CASE_MIXED = -1, 0, 1, 2, 3


def case_class(raw: str) -> int:
    """-1 no cased letters, 0 all lower, 1 ALL UPPER, 2 every word capitalised, 3 other."""
    letters = [ch for ch in raw if ch.isalpha() and ch.lower() != ch.upper()]
    if not letters:
        return CASE_NONE
    if all(ch.islower() for ch in letters):
        return CASE_LOWER
    if all(ch.isupper() for ch in letters):
        return CASE_UPPER
    words = [w for w in raw.split() if any(ch.isalpha() and ch.lower() != ch.upper() for ch in w)]
    if all(next(ch for ch in w if ch.isalpha()).isupper() for w in words if any(ch.isalpha() for ch in w)):
        return CASE_TITLE
    return CASE_MIXED


def n_accents(raw: str) -> int:
    """Latin letters carrying a diacritic (Indic scripts are not counted)."""
    n = 0
    for ch in raw:
        if ch.isalpha() and ord(ch) > 127 and not INDIC_RX.match(ch):
            if any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", ch)) or ch in "øæœßđ":
                n += 1
    return n


def n_multi_space(raw: str) -> int:
    """Runs of two or more spaces."""
    return len(_MULTI_SP.findall(raw))


def punct_chars(raw: str) -> str:
    """Punctuation / symbol characters (not letters, digits, spaces or Indic marks)."""
    return "".join(ch for ch in raw if not ch.isalnum() and not ch.isspace() and not INDIC_RX.match(ch))


def strip_legal(raw: str) -> str:
    """The raw name without its legal-form words and the separators they leave behind (spacing kept)."""
    s = _LEGAL_RX.sub("", raw)
    prev = None
    while prev != s:
        prev = s
        s = _SEP_RX.sub("", s)
    return s.strip()


def raw_eq_mod_legal(q: str, s: str) -> tuple[float, float]:
    """(raw equality, case-folded equality) of two names after removing legal-form words."""
    a, b = strip_legal(q), strip_legal(s)
    return float(a == b), float(a.casefold() == b.casefold())


_DIGITS = re.compile(r"\d+")


def _accented(raw: str) -> collections.Counter:
    return collections.Counter(ch.lower() for ch in raw if ch.isalpha() and ord(ch) > 127 and not INDIC_RX.match(ch)
                               and (any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", ch))
                                    or ch in "øæœßđØÆŒĐ"))


def _addr_shape(raw: str, fold: bool) -> tuple:
    comps = [_DIGITS.sub("#", c.strip()) for c in raw.split(",") if c.strip()]
    return tuple(sorted(c.casefold() for c in comps) if fold else sorted(comps))


# Pair features, all measured on the raw strings. Accents, case and punctuation are taken RELATIVE to the S1
# (a French S1 name carries accents of its own; a query that reproduces them adds no noise).
SURFACE_FEATURES = ["sf_q_case", "sf_case_eq", "sf_acc_add", "sf_acc_rm", "sf_msp", "sf_punct_add", "sf_punct_rm",
                    "sf_eq_raw", "sf_eq_legal", "sf_eq_legal_cf", "sf_a_case", "sf_a_eq_nd", "sf_a_eq_nd_cf"]


def pair_surface(qn: str, sn: str, qa: str, sa: str) -> tuple:
    """SURFACE_FEATURES of one (query, S1) pair: q/S1 raw names ``qn``/``sn`` and raw addresses ``qa``/``sa``."""
    qc = case_class(qn)
    aq, as_ = _accented(qn), _accented(sn)
    pq, ps = collections.Counter(punct_chars(qn)), collections.Counter(punct_chars(sn))
    eq_l, eq_lcf = raw_eq_mod_legal(qn, sn)
    if qa.strip() and sa.strip():
        a_nd, a_ndcf = float(_addr_shape(qa, False) == _addr_shape(sa, False)), \
            float(_addr_shape(qa, True) == _addr_shape(sa, True))
    else:
        a_nd = a_ndcf = -1.0
    return (float(qc), float(qc == case_class(sn)), float(sum((aq - as_).values())), float(sum((as_ - aq).values())),
            float(n_multi_space(qn) - n_multi_space(sn)), float(sum((pq - ps).values())), float(sum((ps - pq).values())),
            float(qn == sn), eq_l, eq_lcf, float(case_class(qa)), a_nd, a_ndcf)
