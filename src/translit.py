# -*- coding: utf-8 -*-
"""Native-script (Indic) name handling learned from the training pairs.

In the data, a native-script S2/S3 name is a word-by-word phonetic rendering of the English
S1 name (same token count in 100% of pairs), so a token dictionary mined by positional
alignment of (S1 name, native S2/S3 name) training pairs translates ~96.5% of test native tokens
exactly. Out-of-vocabulary tokens are romanised with ``anyascii`` and snapped to the closest
Source-1 vocabulary word sharing the same consonant skeleton.
"""
from __future__ import annotations

import collections
import re
import unicodedata
from functools import lru_cache

from anyascii import anyascii
from rapidfuzz import fuzz

INDIC_RX = re.compile('[ऀ-෿]')
TOK_RX = re.compile(r"[\wऀ-෿]+")


def _clean(s: str) -> str:
    s = unicodedata.normalize('NFC', s or '')
    return s.replace('‌', '').replace('‍', '')


def split_tokens(s: str) -> list:
    return TOK_RX.findall(_clean(s).lower())


def is_native(tok: str) -> bool:
    return bool(INDIC_RX.search(tok))


def skeleton(s: str) -> str:
    s = s.lower().replace('ph', 'f')
    s = re.sub(r'[^a-z]', '', s)
    s = re.sub(r'[aeiouyhw]', '', s)
    return re.sub(r'(.)\1+', r'\1', s)


def mine_pairs(pairs) -> dict:
    """pairs: iterable of (latin_s1_name, native_name) -> {native_token: Counter(latin_token)}."""
    stats = collections.defaultdict(collections.Counter)
    for lat, nat in pairs:
        a, b = split_tokens(lat), split_tokens(nat)
        if len(a) != len(b) or not any(is_native(t) for t in b):
            continue
        for x, y in zip(a, b):
            if is_native(y) and not is_native(x):
                stats[y][x] += 1
    return stats


class NativeDict:
    def __init__(self, stats: dict, s1_vocab_counts: dict | None = None):
        self.map = {y: c.most_common(1)[0][0] for y, c in stats.items() if c}
        self.bucket = collections.defaultdict(list)
        for w, n in (s1_vocab_counts or {}).items():
            if w.isalpha() and len(w) >= 2:
                self.bucket[skeleton(w)].append((w, n))
        for k in self.bucket:
            self.bucket[k].sort(key=lambda x: -x[1])
            del self.bucket[k][20:]
        self._oov = {}

    def __len__(self):
        return len(self.map)

    def _oov_token(self, tok: str) -> str:
        r = self._oov.get(tok)
        if r is not None:
            return r
        a = re.sub(r'[^a-z0-9]', '', anyascii(tok).lower())
        best, bs = a, 0.0
        for w, _ in self.bucket.get(skeleton(a), ()):
            s = fuzz.ratio(a, w)
            if s > bs:
                best, bs = w, s
        r = best if bs >= 50 else a
        if len(self._oov) < 500_000:
            self._oov[tok] = r
        return r

    def token(self, tok: str) -> str:
        if not is_native(tok):
            return tok
        t = self.map.get(tok)
        return t if t is not None else self._oov_token(tok)

    def romanize(self, s: str) -> str:
        """Replace native tokens of ``s`` (keeps Latin tokens and punctuation between them)."""
        s = _clean(s)
        return TOK_RX.sub(lambda m: self.token(m.group(0).lower()) if is_native(m.group(0)) else m.group(0), s)
