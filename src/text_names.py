# -*- coding: utf-8 -*-
"""Business-name normalisation (country-agnostic, hand-written domain rules only).

Pipeline per name (see Documentation for the noise catalogue this was built from):
  1. Native-script (Indic) tokens -> English via a dictionary mined from training pairs
     (``translit.NativeDict``), falling back to the nearest Source-1 vocabulary word / anyascii.
  2. Website / handle forms (``sjdcement.com``, ``@pacificdigital``) -> stem, segmented into
     words with a unigram model of Source-1 tokens.
  3. Tags, phone numbers, ``(ID: 123)``, junk prefixes stripped.
  4. Alias connectors (d/b/a, f/k/a, formerly, née, trading as, ...) split; the part after the
     connector is the main name, other parts kept as alternates.
  5. Diacritics folded, dotted acronyms collapsed (s.a.r.l. -> sarl), '&'/'and'/'et' unified,
     OCR digit-for-letter fixes accepted only when they land in the Source-1 vocabulary.
  6. Leading honorifics / articles stripped; legal forms canonicalised (and removed for ``core``).
"""
from __future__ import annotations

import math
import re
import unicodedata

INDIC_RX = re.compile('[ऀ-෿]')
LATIN_EXT_RX = re.compile('[À-ɏ]')
ALIAS_RX = re.compile(
    r'\s+(?:formerly\s+known\s+as|doing\s+business\s+as|also\s+known\s+as|trading\s+as|'
    r'formerly:?|d\.b\.a\.?:?|d/b/a:?|dba:?|f/k/a:?|fka:?|a/k/a:?|aka:?|t/a:?|n[ée]e:?)\s+', re.I)
ID_RX = re.compile(r'[\(\[]*\s*\bID\s*:\s*[\[\(]*\s*\d+\s*[\]\)]*', re.I)
PHONE_RX = re.compile(r'(?:\s[-–]\s*)?\+?\d[\d\s\-\.]{6,}\d')
HASHNUM_RX = re.compile(r'#\s?\d{3,}\b')
TLD = r'(?:co\.in|com|net|org|in|co|biz|info|us|io|fr|eu)'
URL_RX = re.compile(r'(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.' + TLD + r'\b', re.I)
HANDLE_RX = re.compile(r'(?<![\w])[@#]([a-z][a-z0-9_]{1,})', re.I)
DOTTED_RX = re.compile(r'\b(?:[a-z]\.){1,}[a-z]\b\.?', re.I)
TOKEN_RX = re.compile(r"m/s|&|[\wऀ-෿‌‍]+")
MS_RX = re.compile(r'^\s*m/s\.?\s+', re.I)
JUNK_RX = re.compile(r'^\s*(?:\*{2,}|>>|<<|\.{3}|--)')

LEGAL = {
    'private': 'pvt', 'pvt': 'pvt', 'pvtltd': 'pvt ltd', 'limited': 'ltd', 'ltd': 'ltd',
    'incorporated': 'inc', 'inc': 'inc', 'corporation': 'corp', 'corp': 'corp',
    'company': 'co', 'co': 'co', 'cie': 'co', 'compagnie': 'co',
    'llc': 'llc', 'llp': 'llp', 'lp': 'lp', 'pc': 'pc', 'pllc': 'pllc', 'plc': 'plc',
    'lc': 'lc', 'pa': 'pa', 'opc': 'opc', 'ltda': 'ltd', 'gmbh': 'gmbh', 'ag': 'ag',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl', 'sa': 'sa', 'sci': 'sci',
    'snc': 'snc', 'ei': 'ei', 'selarl': 'selarl', 'scop': 'scop', 'scp': 'scp', 'scm': 'scm',
}
LEGAL_VALS = set(v for x in LEGAL.values() for v in x.split())
# legal "families" (a conflict between families is strong evidence of a different entity)
LEGAL_FAMILY = {'pvt': 'ltd', 'ltd': 'ltd', 'plc': 'ltd', 'inc': 'corp', 'corp': 'corp', 'co': 'co',
                'llc': 'llc', 'lc': 'llc', 'llp': 'llp', 'lp': 'lp', 'pc': 'pc', 'pllc': 'pc', 'pa': 'pc',
                'sarl': 'sarl', 'eurl': 'sarl', 'sas': 'sas', 'sasu': 'sas', 'sa': 'sa', 'sci': 'sci',
                'snc': 'snc', 'ei': 'ei'}
HONORIFIC = {'the', 'mr', 'mrs', 'ms', 'dr', 'shri', 'sri', 'shree', 'smt', 'sh', 'messrs', 'm/s',
             'la', 'le', 'les'}
CONNECT = {'and': '&', '&': '&', '+': '&', 'et': '&', 'und': '&', 'y': '&'}
STOP = {'&', 'of', 'the', 'de', 'du', 'des', 'la', 'le', 'd', 'l', 'a'}
OCR_DIGIT = str.maketrans({'0': 'o', '1': 'l', '5': 's', '8': 'b', '6': 'g', '3': 'e', '4': 'a', '7': 't'})

BRAND_SYLL = ['vantage', 'umbra', 'delta', 'novi', 'kelo', 'orbi', 'calo', 'brix', 'belo', 'halo',
              'cira', 'drex', 'xylo', 'lyra', 'pyra', 'mira', 'flux', 'aria', 'gild', 'tavo', 'faye',
              'riza', 'vera', 'ecto', 'dova', 'nyla', 'onyx', 'zeph', 'zeta', 'yuma', 'iri', 'avi',
              'vio', 'nex', 'jax', 'veo', 'quo', 'wex', 'kor', 'arc', 'evo', 'lum', 'sol', 'syn', 'io',
              'ari', 'ra', 'lo', 'a', 'o', 'x']

# name bit flags
F_NATIVE, F_WEB, F_ALIAS, F_HANDLE, F_BRAND, F_TAG, F_HONOR, F_BRACKET = 1, 2, 4, 8, 16, 32, 64, 128


def fold_latin(s: str) -> str:
    """Strip diacritics from Latin text; Indic scripts untouched."""
    s = unicodedata.normalize('NFKC', s)
    if not LATIN_EXT_RX.search(s):
        return s
    out = []
    for ch in unicodedata.normalize('NFKD', s):
        if unicodedata.combining(ch) and not INDIC_RX.match(ch):
            continue
        out.append(ch)
    s = unicodedata.normalize('NFC', ''.join(out))
    return s.replace('ø', 'o').replace('æ', 'ae').replace('œ', 'oe').replace('ß', 'ss').replace('đ', 'd')


def ocr_fix_token(t: str, vocab) -> str:
    if vocab is not None and t in vocab:
        return t
    mixed = any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
    if vocab is None:
        if mixed and sum(c.isalpha() for c in t) >= 3 and sum(c.isdigit() for c in t) <= 2:
            c = t.translate(OCR_DIGIT)
            return c if c.isalpha() else t
        return t
    cands = []
    if mixed:
        cands.append(t.translate(OCR_DIGIT))
    if t.startswith('l') and len(t) > 2 and t[1] not in 'aeiouyl':
        cands.append('i' + t[1:])
    if 'm' in t:
        cands.append(t.replace('m', 'rn'))
    for c in cands:
        if c in vocab:
            return c
        c2 = c.translate(OCR_DIGIT)
        if c2 in vocab:
            return c2
    return t


def basic_tokens(s: str, vocab=None) -> list:
    s = fold_latin(s).lower()
    s = MS_RX.sub(' m/s ', s)
    s = DOTTED_RX.sub(lambda m: m.group().replace('.', ''), s)
    s = re.sub(r"['’`´]", '', s)
    s = s.replace('&', ' & ').replace('+', ' & ')
    out = []
    for t in TOKEN_RX.findall(s):
        t = t.replace('‌', '').replace('‍', '')
        if not t:
            continue
        t = CONNECT.get(t, t)
        if not INDIC_RX.search(t):
            t = ocr_fix_token(t, vocab)
        out.append(t)
    return out


def _fix_stem(st: str) -> str:
    st = fold_latin(st).lower().replace('-', '').replace('_', '')
    if sum(c.isalpha() for c in st) >= 3 and sum(c.isdigit() for c in st) <= 2:
        st = st.translate(OCR_DIGIT)
    return re.sub(r'[^a-z0-9]', '', st)


def is_brand_token(t: str, min_len: int = 6) -> bool:
    if len(t) < min_len or not t.isalpha():
        return False
    n = len(t)
    best = [None] * (n + 1)
    best[0] = 0
    for i in range(n):
        if best[i] is None:
            continue
        for w in BRAND_SYLL:
            if t.startswith(w, i):
                j = i + len(w)
                k = best[i] + (1 if len(w) >= 3 else 0)
                if best[j] is None or k > best[j]:
                    best[j] = k
    return best[n] is not None and best[n] >= 2


class UnigramSegmenter:
    """Viterbi word segmentation with a unigram model of Source-1 name tokens (no external data)."""

    def __init__(self, counter: dict, max_len: int = 20):
        tot = float(sum(counter.values())) or 1.0
        self.logp = {w: math.log(c / tot) for w, c in counter.items() if 1 < len(w) <= max_len}
        self.max_len = max_len
        self.unk = math.log(1 / tot) - 5

    def segment(self, s: str) -> list:
        n = len(s)
        if n == 0:
            return []
        best = [(-1e18, 0)] * (n + 1)
        best[0] = (0.0, 0)
        for j in range(1, n + 1):
            for i in range(max(0, j - self.max_len), j):
                w = s[i:j]
                lp = self.logp.get(w)
                if lp is None:
                    lp = self.unk * len(w)
                v = best[i][0] + lp
                if v > best[j][0]:
                    best[j] = (v, i)
        out, j = [], n
        while j > 0:
            i = best[j][1]
            out.append(s[i:j])
            j = i
        return out[::-1]


def parse_name(raw: str, vocab=None, native=None, segmenter=None) -> dict:
    """Normalise one business name.

    vocab     : set of Source-1 name tokens (for OCR fixes), or None
    native    : translit.NativeDict (maps Indic tokens to English), or None
    segmenter : UnigramSegmenter for website / handle stems, or None
    """
    flags = 0
    s = unicodedata.normalize('NFKC', raw or '').strip()
    if INDIC_RX.search(s):
        flags |= F_NATIVE
        if native is not None:
            s = native.romanize(s)
    if LATIN_EXT_RX.search(s):
        s = fold_latin(s)
    web = ''
    m = URL_RX.search(s)
    if m:
        web = _fix_stem(m.group(1))
        flags |= F_WEB
        s = s[:m.start()] + ' ' + s[m.end():]
        s = re.sub(r'\|\s*(www\.?)?\s*$', ' ', s)
    elif re.fullmatch(r'\W*[a-z0-9]{3,}com\W*', s, re.I) and not s.lower().rstrip().endswith(
            ('telecom', 'infocom', 'sycom')):
        web = _fix_stem(re.sub(r'(?i)com\W*$', '', re.sub(r'^\W+', '', s)))
        flags |= F_WEB
        s = ''
    if ID_RX.search(s):
        flags |= F_TAG
        s = ID_RX.sub(' ', s)
    if HASHNUM_RX.search(s):
        flags |= F_TAG
        s = HASHNUM_RX.sub(' ', s)
    if PHONE_RX.search(s):
        flags |= F_TAG
        s = PHONE_RX.sub(' ', s)
    m = HANDLE_RX.search(s)
    if m:
        h = _fix_stem(m.group(1))
        web = web or h
        flags |= F_HANDLE
        s = s[:m.start()] + ' ' + s[m.end():]
    if re.search(r'[\(\)\[\]\{\}]', s):
        flags |= F_BRACKET
    parts = ALIAS_RX.split(s)
    if len(parts) > 1:
        flags |= F_ALIAS
        s_main, alt = parts[-1], parts[:-1]
    else:
        s_main, alt = s, []
    toks = basic_tokens(s_main, vocab)
    while toks and toks[0] in HONORIFIC:
        flags |= F_HONOR
        toks = toks[1:]
    toks = [t for t in toks if t != 'm/s']
    web_toks = []
    if web and not toks:
        web_toks = segmenter.segment(web) if segmenter is not None else [web]
        toks = list(web_toks)
    core, legal = [], []
    for t in toks:
        if t in LEGAL:
            v = LEGAL[t]
            legal.extend(v.split())
            core.extend(v.split())
        else:
            core.append(t)
    nolegal = [t for t in core if t not in LEGAL_VALS]
    alias_norm = []
    for pa in alt:
        tt = [LEGAL.get(t, t) for t in basic_tokens(pa, vocab) if t not in HONORIFIC]
        tt = [t for t in tt if t not in LEGAL_VALS]
        if tt:
            alias_norm.append(' '.join(tt))
    if nolegal and all(is_brand_token(t) for t in nolegal):
        flags |= F_BRAND
    return {
        'n_full': ' '.join(core),
        'n_core': ' '.join(nolegal) if nolegal else ' '.join(core),
        'n_alias': '|'.join(alias_norm),
        'n_web': web,
        'n_legal': ' '.join(sorted(set(legal))),
        'n_flags': flags,
    }


def concat_key(tokens) -> str:
    return ''.join(t for t in tokens if t not in STOP)


def initials(tokens) -> str:
    return ''.join(t[0] for t in tokens if t and t not in STOP)


def web_match(stem: str, ref_tokens) -> float:
    """How well a website/handle stem is explained by the reference tokens (1 = fully)."""
    stem = re.sub(r'[^a-z0-9]', '', stem.lower())
    toks = [re.sub(r'[^a-z0-9]', '', t.lower()) for t in ref_tokens]
    toks = [t for t in toks if t]
    if not stem or not toks:
        return 0.0
    full = ''.join(toks)
    nostop = ''.join(t for t in toks if t not in STOP and t != 'and')
    if stem in (full, nostop):
        return 1.0
    if len(stem) >= 5 and (full.startswith(stem) or nostop.startswith(stem)):
        return 1.0
    n = len(stem)
    best = [-1e9] * (n + 1)
    unexp = [0] * (n + 1)
    best[0] = 0.0
    tset = set(toks)
    for i in range(n):
        sc = best[i]
        if sc <= -1e9:
            continue
        if sc - 1.0 > best[i + 1]:
            best[i + 1] = sc - 1.0
            unexp[i + 1] = unexp[i] + 1
        for t in tset:
            if stem.startswith(t, i):
                j = i + len(t)
                if sc + len(t) > best[j]:
                    best[j] = sc + len(t)
                    unexp[j] = unexp[i]
            for L in range(min(len(t) - 1, n - i), 2, -1):
                if stem.startswith(t[:L], i):
                    j = i + L
                    if sc + L - 0.5 > best[j]:
                        best[j] = sc + L - 0.5
                        unexp[j] = unexp[i]
                    break
            if stem[i] == t[0] and sc + 0.8 > best[i + 1]:
                best[i + 1] = sc + 0.8
                unexp[i + 1] = unexp[i]
    return 1.0 - unexp[n] / n
