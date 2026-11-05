# -*- coding: utf-8 -*-
"""Country-agnostic address normaliser / light parser.

* Components (comma-separated parts) are a bag: their order is noise in every source.
* Surface forms are mapped to SHORT canonical tokens (street->st, avenue->av, rue->r, ...).
  Tables are keyed by surface form only (EN / FR / IN abbreviations are hand-written domain
  rules), so an unseen country simply gets fewer rewrites - nothing is filtered by country.
* State / region variants (2-letter codes, full names, native-script names, French
  departments) come from ``resources/state_map.json`` which is keyed by the country LABEL; an
  unknown country label just has no state map (the record is then searched country-wide).
* Units / PMB / PO boxes are split out and excluded from similarity (S2 drops 100% of units).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

NULL_COMPS = {'null', '<null>', 'n/a', 'na', 'none', 'nil', 'nan', '-', '--', 'n.a.', 'n.a', 'unknown', '.', ''}

_EN = {
    'street': 'st', 'str': 'st', 'st': 'st', 'saint': 'st', 'ste': 'ste', 'sainte': 'ste',
    'avenue': 'av', 'ave': 'av', 'av': 'av', 'avn': 'av', 'aveune': 'av', 'aveue': 'av', 'avnue': 'av', 'avneue': 'av',
    'road': 'rd', 'rd': 'rd', 'drive': 'dr', 'dr': 'dr', 'drv': 'dr', 'lane': 'ln', 'ln': 'ln',
    'court': 'ct', 'ct': 'ct', 'crt': 'ct', 'circle': 'cir', 'cir': 'cir', 'circ': 'cir', 'crcl': 'cir',
    'crossing': 'xing', 'xing': 'xing', 'crssng': 'xing', 'boulevard': 'bd', 'blvd': 'bd', 'bd': 'bd', 'boul': 'bd',
    'bld': 'bd', 'boulv': 'bd',
    'place': 'pl', 'pl': 'pl', 'parkway': 'pkwy', 'pkwy': 'pkwy', 'pky': 'pkwy', 'highway': 'hwy', 'hwy': 'hwy',
    'terrace': 'ter', 'ter': 'ter', 'terr': 'ter', 'trail': 'trl', 'trl': 'trl', 'cove': 'cv', 'cv': 'cv',
    'point': 'pt', 'pt': 'pt', 'square': 'sq', 'sq': 'sq', 'center': 'ctr', 'centre': 'ctr', 'ctr': 'ctr',
    'crescent': 'cres', 'cres': 'cres', 'expressway': 'expy', 'expy': 'expy', 'freeway': 'fwy', 'fwy': 'fwy',
    'manor': 'mnr', 'mnr': 'mnr', 'mount': 'mt', 'mt': 'mt', 'mountain': 'mtn', 'mtn': 'mtn', 'heights': 'hts',
    'hts': 'hts', 'junction': 'jct', 'jct': 'jct', 'loop': 'loop', 'lp': 'loop', 'path': 'path', 'pass': 'pass',
    'pike': 'pike', 'ridge': 'rdg', 'rdg': 'rdg', 'run': 'run', 'way': 'way', 'wy': 'way', 'alley': 'aly', 'aly': 'aly',
    'bend': 'bnd', 'bnd': 'bnd', 'bluff': 'blf', 'blf': 'blf', 'branch': 'br', 'brook': 'brk', 'brk': 'brk',
    'canyon': 'cyn', 'cyn': 'cyn', 'creek': 'crk', 'crk': 'crk', 'estates': 'est', 'est': 'est', 'falls': 'fls',
    'ferry': 'fry', 'field': 'fld', 'fields': 'flds', 'forest': 'frst', 'fort': 'ft', 'ft': 'ft', 'garden': 'gdn',
    'gardens': 'gdns', 'glen': 'gln', 'green': 'grn', 'grove': 'grv', 'grv': 'grv', 'harbor': 'hbr', 'hill': 'hl',
    'hills': 'hls', 'hollow': 'holw', 'island': 'is', 'lake': 'lk', 'lakes': 'lks', 'landing': 'lndg',
    'meadow': 'mdw', 'meadows': 'mdws', 'mill': 'ml', 'park': 'park', 'pk': 'park', 'plaza': 'plz', 'plz': 'plz',
    'port': 'prt', 'ranch': 'rnch', 'route': 'rte', 'rte': 'rte', 'rt': 'rte', 'row': 'row', 'spring': 'spg',
    'springs': 'spgs', 'station': 'sta', 'sta': 'sta', 'stn': 'sta', 'summit': 'smt', 'turnpike': 'tpke',
    'valley': 'vly', 'view': 'vw', 'village': 'vlg', 'vlg': 'vlg', 'vill': 'vlg', 'ville': 'vl', 'vista': 'vis',
    'walk': 'walk', 'woods': 'wds',
    'north': 'n', 'n': 'n', 'south': 's', 's': 's', 'east': 'e', 'e': 'e', 'west': 'w', 'w': 'w',
    'northeast': 'ne', 'ne': 'ne', 'northwest': 'nw', 'nw': 'nw', 'southeast': 'se', 'se': 'se',
    'southwest': 'sw', 'sw': 'sw',
    'apartment': 'unit', 'apt': 'unit', 'suite': 'unit', 'unit': 'unit', 'lot': 'unit', 'rm': 'unit',
    'room': 'unit', 'trlr': 'unit', 'spc': 'unit', 'space': 'unit',
    'floor': 'fl', 'fl': 'fl', 'flr': 'fl', 'building': 'bldg', 'bldg': 'bldg',
    'pmb': 'pobox', 'box': 'box', 'po': 'po', 'pob': 'pobox',
    'township': 'twp', 'twp': 'twp', 'townhsip': 'twp', 'tonwship': 'twp', 'twnshp': 'twp', 'borough': 'boro',
    'boro': 'boro', 'county': 'cnty', 'cnty': 'cnty', 'cdp': 'cdp', 'city': 'city', 'ctiy': 'city', 'cty': 'city',
    'town': 'town', 'of': 'of',
}
_IN = {
    'house': 'hno', 'hno': 'hno', 'hn': 'hno', 'h': 'h', 'door': 'hno', 'dno': 'hno',
    'plot': 'plot', 'pno': 'plot', 'plt': 'plot', 'flat': 'flat', 'flatno': 'flat', 'fno': 'flat',
    'khasra': 'kh', 'kh': 'kh', 'survey': 'sy', 'sy': 'sy', 'sno': 'sy', 'ward': 'wd', 'wd': 'wd', 'wno': 'wd',
    'shop': 'shop', 'office': 'off', 'off': 'off', 'no': 'no', 'number': 'no', 'nos': 'no', 'num': 'no',
    'near': 'nr', 'nr': 'nr', 'opp': 'opp', 'opposite': 'opp', 'behind': 'bh', 'b/h': 'bh', 'bh': 'bh',
    'beside': 'bes', 'besides': 'bes', 'c/o': 'co', 'co': 'co', 's/o': 'so', 'w/o': 'wo', 'd/o': 'do',
    'district': 'dist', 'dist': 'dist', 'distt': 'dist', 'taluka': 'tal', 'tal': 'tal', 'tehsil': 'teh',
    'teh': 'teh', 'mandal': 'mandal', 'post': 'po', 'gali': 'gali', 'marg': 'marg', 'nagar': 'nagar',
    'colony': 'colony', 'sector': 'sec', 'sec': 'sec', 'ground': 'gnd', 'gf': 'gf', 'ff': 'ff', 'sf': 'sf',
    'tf': 'tf', 'main': 'main', 'cross': 'crs', 'extn': 'ext', 'extension': 'ext', 'ext': 'ext',
    'apts': 'unit', 'appt': 'unit', 'apptt': 'unit', 'complex': 'cplx', 'cplx': 'cplx', 'chs': 'chs',
    'chsl': 'chs', 'soc': 'soc', 'society': 'soc', 'industrial': 'ind', 'indl': 'ind', 'ind': 'ind',
    'estate': 'est', 'area': 'area', 'phase': 'ph', 'ph': 'ph', 'block': 'blk', 'blk': 'blk',
    'bangalore': 'bengaluru', 'bengaluru': 'bengaluru', 'banglore': 'bengaluru', 'bombay': 'mumbai',
    'calcutta': 'kolkata', 'madras': 'chennai', 'gurugram': 'gurgaon', 'ahmadabad': 'ahmedabad', 'poona': 'pune',
    'mysore': 'mysuru', 'trivandrum': 'thiruvananthapuram', 'cochin': 'kochi', 'baroda': 'vadodara',
    'calicut': 'kozhikode',
}
_FR = {
    'rue': 'r', 'r': 'r', 'ruelle': 'rle', 'rle': 'rle', 'avenue': 'av', 'boulevard': 'bd', 'bvd': 'bd',
    'place': 'pl', 'allee': 'all', 'allees': 'all', 'all': 'all', 'al': 'all', 'impasse': 'imp', 'imp': 'imp',
    'chemin': 'ch', 'ch': 'ch', 'che': 'ch', 'chem': 'ch', 'chem1': 'ch', 'route': 'rte', 'rte': 'rte',
    'esplanade': 'espl', 'espl': 'espl', 'espl1': 'espl', 'quai': 'q', 'q': 'q', 'cours': 'crs', 'crs': 'crs',
    'cour2': 'crs', 'cour': 'cour', 'passage': 'pass', 'psg': 'pass', 'pas': 'pass', 'residence': 'res',
    'res': 'res', 'lotissement': 'lot', 'hameau': 'ham', 'ham': 'ham', 'faubourg': 'fg', 'fbg': 'fg', 'fg': 'fg',
    'cite': 'cite', 'chaussee': 'chau', 'chau1': 'chau', 'chau': 'chau', 'rond-point': 'rpt', 'rpt': 'rpt',
    'rondpoint': 'rpt', 'venelle': 'ven', 'ven': 'ven', 'sentier': 'sen', 'sente': 'sen', 'parvis': 'pvs',
    'pvs': 'pvs', 'promenade': 'prom', 'prom': 'prom', 'domaine': 'dom', 'dom': 'dom', 'zone': 'z', 'za': 'za',
    'zi': 'zi', 'zac': 'zac', 'batiment': 'bldg', 'bat': 'bldg', 'etage': 'fl', 'appartement': 'unit',
    'app': 'unit', 'escalier': 'esc', 'esc': 'esc', 'marechal': 'mal', 'mal': 'mal', 'general': 'gen',
    'gal': 'gen', 'gen': 'gen', 'docteur': 'dr', 'president': 'pdt', 'pdt': 'pdt', 'professeur': 'pr', 'pr': 'pr',
    'commandant': 'cdt', 'cdt': 'cdt', 'capitaine': 'cne', 'cne': 'cne', 'lieutenant': 'lt', 'lt': 'lt',
    'colonel': 'col', 'col': 'col', 'grande': 'gde', 'gde': 'gde', 'grand': 'gd', 'gd': 'gd', 'petite': 'pte',
    'pte': 'pte', 'petit': 'pt', 'bis': 'bis', 'ter': 'ter', 'quater': 'quater', 'boite': 'bp', 'bp': 'bp',
    'cs': 'bp', 'cedex': 'cedex',
}
STOP = {'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'et', 'the', 'and', 'of', 'a', 'au', 'aux', 'en', 'sur'}

CANON: dict = {}
for _tab in (_EN, _IN, _FR):
    for _k, _v in _tab.items():
        CANON.setdefault(_k, _v)
CANON.update({'bis': 'bis', 'ter': 'ter', 'b': 'b', 't': 't', 'lot': 'unit', 'lane': 'ln', 'village': 'vlg',
              'post': 'po'})

UNIT_TOK = {'unit', 'fl', 'bldg', 'pobox', 'box', 'po', 'bp', 'cedex', 'gf', 'ff', 'sf', 'tf', 'esc'}
STREET_TYPES = {v for k, v in _EN.items() if k in (
    'street', 'avenue', 'road', 'drive', 'lane', 'court', 'circle', 'crossing', 'boulevard', 'place', 'parkway',
    'highway', 'terrace', 'trail', 'cove', 'point', 'square', 'crescent', 'expressway', 'freeway', 'loop', 'path',
    'pike', 'ridge', 'run', 'way', 'alley', 'plaza', 'route', 'turnpike', 'bend')} | {
    'r', 'rle', 'all', 'imp', 'ch', 'espl', 'q', 'crs', 'cour', 'pass', 'fg', 'cite', 'chau', 'rpt', 'ven', 'sen',
    'pvs', 'prom', 'marg', 'gali', 'main'}
CITY_DECOR = {'cdp', 'city', 'ctiy', 'cty', 'icty', 'ciity', 'citty', 'ccdp', 'township', 'twp', 'townhsip',
              'tonwship', 'townshp', 'twnshp', 'borough', 'boro', 'town', 'village', 'of'}

_ORD_WORDS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
              'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14,
              'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19,
              'twentieth': 20, 'thirtieth': 30, 'fortieth': 40, 'fiftieth': 50, 'sixtieth': 60, 'seventieth': 70,
              'eightieth': 80, 'ninetieth': 90, 'premier': 1, 'premiere': 1, 'deuxieme': 2, 'troisieme': 3}
_TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
         'ninety': 90}
_ORD_RE = re.compile(r'\b(\d+)(st|nd|rd|th|er|eme)\b')
_TENS_ORD_RE = re.compile(r'\b(' + '|'.join(_TENS) + r')[\s-]+(' +
                          '|'.join(k for k, v in _ORD_WORDS.items() if v < 10) + r')\b')
_ORDW_RE = re.compile(r'\b(' + '|'.join(_ORD_WORDS) + r')\b')
_NUMPREFIX_RE = re.compile(r'(?:n°|nº|n˚|no\.|n\.|#+|\bno\b\s*[:.-]?)\s*(?=\d)')
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)*|[^\sa-z0-9,.;:()\[\]'\"`&+-]+")
_IN_PREFIX = {'h': 'hno', 'hn': 'hno', 'house': 'hno', 'd': 'hno', 'door': 'hno', 'p': 'plot', 'pl': 'plot',
              'plot': 'plot', 'f': 'flat', 'flat': 'flat', 'kh': 'kh', 'khasra': 'kh', 's': 'sy', 'sy': 'sy',
              'survey': 'sy', 'w': 'wd', 'wd': 'wd', 'ward': 'wd', 'shop': 'shop', 'office': 'off', 'off': 'off',
              'gali': 'gali', 'room': 'unit', 'r': 'unit', 'unit': 'unit'}
_IN_PREFIX_RE = re.compile(r'\b(' + '|'.join(sorted(_IN_PREFIX, key=len, reverse=True)) +
                           r')\s*[./]?\s*(?:no|n)\b\s*[.:#-]*\s*|\b(flat|plot|house|door|shop|khasra|ward)no\b\s*[.:#-]*\s*')
_HYPH_ALNUM_RE = re.compile(r'\b([a-z]{1,3})\s*-\s*(?=#*\d)')
# house number at the start of a (pre-processed) component: optional Indian prefix tokens (H No / D.No -> hno,
# Plot No -> plot, ...), an optional 1-2 letter prefix (B-352 -> b352, W6078, N93W17658) and a composite
# continuation (5-7-402/a, 34.12.12, 40- 49); ``n`` is the first digit run
_HN_PFX = ('hno', 'plot', 'flat', 'sy', 'kh', 'shop', 'off')
_HN_RE = re.compile(r'^\s*(?P<pfx>(?:(?:' + '|'.join(_HN_PFX) + r')\s+)*)(?P<lp>[a-z]{1,2}(?=\d))?(?P<n>\d+)'
                    r'(?P<rest>(?:\s*[-/.]\s*(?:\d+[a-z]?|[a-z]{1,2}(?![a-z]))|[nsew]\d+)*)'
                    r'(?:(?P<suf>[a-z])(?![a-z])|\s+(?P<bis>bis|ter|quater|1/2)\b)?')
# which numbered component is "the" house number when several exist (component order is shuffled between
# sources, so the choice must not depend on order): door/house no > bare number > plot > survey > flat/shop
_PFX_RANK = {'hno': 0, 'plot': 2, 'sy': 3, 'kh': 3, 'flat': 4, 'shop': 4, 'off': 4}
_SUF = {'bis': 'b', 'ter': 't', 'quater': 'q', '1/2': 'h'}
_RUN_RE = re.compile(r'\d+|[a-z]+')
_UNIT_HEAD = re.compile(
    r'^(fl|floor|unit|suite|ste|apt|apartment|po box|p ?o box|pmb|lot|bldg|building|room|rm|trlr|spc|box|dept|bp|'
    r'cs|cedex|appartement|appt|bat|batiment|escalier|etage|\d+(er|eme|e)? etage|rez de chaussee|'
    r'\d+(st|nd|rd|th)? ?(floor|flr|fl)s?|(ground|upper|lower|first|second|third|fourth|fifth) (floor|flr)s?|'
    r'#\s*(apt|unit|ste|suite|lot|condo|#|[a-z]?\d*[a-z]?$))(\W|$)')
_TYPE_VOCAB = sorted({k for k, v in CANON.items() if v in STREET_TYPES and len(k) >= 5})
_DECOR_CANON = {'cdp', 'city', 'twp', 'boro', 'town', 'cnty'}
_POST_RE = re.compile(r'(\d{5,6})(?:\s+(.*))?')
_DIGIT_RE = re.compile(r'\d')
_ALPHA_RE = re.compile(r'[a-z]+')


def fold(s: str) -> str:
    """NFKC + lower + strip Latin diacritics (non-Latin scripts are kept intact)."""
    s = unicodedata.normalize('NFKC', s).lower()
    out = []
    for ch in unicodedata.normalize('NFD', s):
        if unicodedata.category(ch) == 'Mn' and out and ord(out[-1]) < 0x250:
            continue
        out.append(ch)
    s = unicodedata.normalize('NFC', ''.join(out))
    return s.replace('œ', 'oe').replace('æ', 'ae').replace('ß', 'ss').replace('’', "'")


def pre(s: str) -> str:
    s = fold(s)
    s = re.sub(r'[\(\[]\s*(\d+[a-z]?)\s*[\)\]]', r' \1 ', s)  # '(25) rue' -> '25 rue'
    s = _IN_PREFIX_RE.sub(lambda m: ' ' + _IN_PREFIX[m.group(1) or m.group(2)] + ' ', s)
    s = _HYPH_ALNUM_RE.sub(r'\1', s)
    s = re.sub(r'([a-z])#+(?=\d)', r'\1', s)
    s = re.sub(r'#+(?=[a-z]?\d)', ' ', s)
    s = _NUMPREFIX_RE.sub(' ', s)
    s = re.sub(r'(?<=\d)\s*-\s*(?=[a-z]\b)', '', s)
    s = _TENS_ORD_RE.sub(lambda m: str(_TENS[m.group(1)] + _ORD_WORDS[m.group(2)]), s)
    s = _ORDW_RE.sub(lambda m: str(_ORD_WORDS[m.group(1)]), s)
    s = _ORD_RE.sub(r'\1', s)
    s = re.sub(r'(?<![0-9])0+(?=\d)', '', s)
    s = re.sub(r'(?<=\d)\.(?=\s|,|$)', '', s)
    return s


@lru_cache(maxsize=1 << 20)
def fix_type_typo(tok: str) -> str:
    """AVEUNE / CHEIMN / BOULEARD -> canonical street type (edit distance <= 1-2, same first letter)."""
    if tok in CANON:
        return CANON[tok]
    if len(tok) < 5 or tok.isdigit():
        return tok
    best, bd = None, 99
    for v in _TYPE_VOCAB:
        if v[0] != tok[0] or abs(len(v) - len(tok)) > 2:
            continue
        d = Levenshtein.distance(tok, v, score_cutoff=2)
        if d < bd:
            best, bd = v, d
    if best is not None and bd <= (1 if len(tok) < 7 else 2):
        return CANON[best]
    return tok


def canon_tokens(text: str, drop_stop: bool = True) -> list:
    t = pre(text).replace('.', ' ').replace("'", ' ')
    toks = []
    for tok in _TOKEN_RE.findall(t):
        c = CANON.get(tok, tok)
        if drop_stop and c in STOP:
            continue
        toks.append(c)
    return toks


def city_key(c: str) -> str:
    f = fold(c or '')
    f = re.sub(r'\bst\b\.?-?', 'saint ', f)
    f = re.sub(r'\bste\b\.?-?', 'sainte ', f)
    f = re.sub(r'\bft\b\.?', 'fort', f)
    f = re.sub(r'\bmt\b\.?', 'mount', f)
    toks = [t for t in re.findall(r'[a-z0-9]+', f) if t not in CITY_DECOR]
    toks = [_IN.get(t, t) if t in ('bangalore', 'banglore', 'bombay', 'calcutta', 'madras', 'gurugram', 'poona',
                                   'mysore', 'trivandrum', 'cochin', 'baroda', 'calicut', 'ahmadabad') else t
            for t in toks]
    return ' '.join(toks)


_STATE_MAPS = None


def state_maps() -> dict:
    """{country_label: {folded surface: (canonical_state, confidence)}} from resources/state_map.json."""
    global _STATE_MAPS
    if _STATE_MAPS is None:
        p = os.path.join(os.path.dirname(__file__), 'resources', 'state_map.json')
        with open(p, encoding='utf-8') as f:
            raw = json.load(f)
        _STATE_MAPS = {c: {k: (v[0], float(v[1])) for k, v in m.items()} for c, m in raw.items()}
    return _STATE_MAPS


def split_comps(addr: str) -> list:
    comps = []
    for p in addr.split(','):
        q = p.strip().strip('-').strip()
        if fold(q) in NULL_COMPS:
            continue
        comps.append(q)
    return comps


def _street_types(toks: list) -> list:
    """Street-type tokens; a leading 'st' followed by more words is 'Saint' (St Louis, Saint-Nazaire)."""
    return [t for i, t in enumerate(toks) if t in STREET_TYPES and not (i == 0 and t == 'st' and len(toks) > 1)]


def parse_address(addr: str, country: str) -> dict:
    """Parse one address. Returns a dict of compact fields (strings / ints) for storage."""
    smap = state_maps().get(country, {})
    comps = split_comps(addr or '')
    fc = [re.sub(r'\s+', ' ', fold(c)).strip() for c in comps]
    st = None
    for i, f in enumerate(fc):
        v = smap.get(f)
        if v is not None and (st is None or v[1] > st[1]):
            st = (v[0], v[1], i)
    hn, hn_last, hn_suf, hn_comp, street_type, street_body = -1, -1, '', '', '', ''
    tokens, cities, units, unit_nums, postcode = [], [], 0, set(), ''
    items = []
    for i, (raw, f) in enumerate(zip(comps, fc)):
        if st is not None and i == st[2]:
            continue
        if _UNIT_HEAD.match(f) and not re.match(r'^#\s*\d+\s+[a-z]', f):
            units += 1
            # unit / floor / box numbers stay out of the similarity bag but are kept for comparison
            unit_nums.update('-'.join(_RUN_RE.findall(t)) for t in canon_tokens(raw) if _DIGIT_RE.search(t))
            continue
        toks = [fix_type_typo(t) for t in canon_tokens(raw)]
        m = _HN_RE.match(pre(raw))
        types = _street_types(toks)
        m5 = _POST_RE.fullmatch(f)
        post_like = bool(m5) and (m5.group(2) is None or (not _DIGIT_RE.search(m5.group(2)) and not types))
        rank = min((_PFX_RANK.get(x, 1) for x in m.group('pfx').split()), default=1) if m else 9
        items.append((raw, toks, m, types, post_like, m5, rank))
    # the street component: the best-ranked numbered component (order-free: components are shuffled between
    # sources), else one with a street type, else a postcode-like "12345 Name" component (a 5-6 digit house
    # number on a street without a known type -- postcodes are practically absent in this data)
    numbered = [k for k, it in enumerate(items) if it[2] and not it[4]]
    si = min(numbered, key=lambda k: items[k][6]) if numbered else None
    if si is None:
        si = next((k for k, it in enumerate(items) if it[3] and not it[4]), None)
    if si is None:
        si = next((k for k, it in enumerate(items) if it[2] or it[3]), None)
    for k, (raw, toks, m, types, post_like, m5, rank) in enumerate(items):
        if k != si and post_like:
            postcode = m5.group(1)
            if m5.group(2):
                cities.append(city_key(m5.group(2)))
            continue
        tokens.extend(toks)
        if k != si:
            cities.append(city_key(raw))
            continue
        body = list(toks)
        if m:
            runs = _RUN_RE.findall((m.group('lp') or '') + m.group('n') + (m.group('rest') or ''))
            digits = [r for r in runs if r.isdigit()]
            hn = int(m.group('n')[:9])
            hn_last = int(digits[-1][:9])
            s = (m.group('suf') or m.group('bis') or '').strip()
            hn_suf = _SUF.get(s, s)
            hn_comp = '-'.join(runs + ([hn_suf] if hn_suf else []))
            # drop the prefix words and the number itself from the street body
            while body and body[0] in _HN_PFX:
                body = body[1:]
            num_toks = [fix_type_typo(t) for t in canon_tokens(m.group(0)[m.start('lp' if m.group('lp') else 'n'):])]
            if num_toks and body[:len(num_toks)] == num_toks:
                body = body[len(num_toks):]
            elif body and body[0].startswith(m.group('n')):
                body = body[1:]
                if hn_suf and body and body[0] in ('bis', 'ter', 'quater', 'b', 't', 'q', 'h', '1/2'):
                    body = body[1:]
        types = _street_types(body)
        street_type = types[0] if types else ''
        street_body = ' '.join(t for t in body if not (types and t == types[0]))
    # drop unit-ish tokens and city decorations (CDP, CITY, TOWNSHIP, COUNTY) from the similarity bag
    tokens = [t for t in tokens if t not in UNIT_TOK and t not in _DECOR_CANON]
    nums = sorted({t for t in tokens if _DIGIT_RE.search(t)})
    return {
        'a_tok': ' '.join(tokens),
        'a_state': st[0] if st else '',
        'a_hn': hn,
        'a_hnl': hn_last,
        'a_hnc': hn_comp,
        'a_hnsuf': hn_suf,
        'a_street': street_body,
        'a_stype': street_type,
        'a_city': '|'.join(c for c in cities if c),
        'a_post': postcode,
        'a_nums': ' '.join(nums),
        'a_unit': ' '.join(sorted(unit_nums)),
        'a_units': units,
        'a_empty': int(not comps),
    }
