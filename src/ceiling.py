"""Irreducible true pairs and the macro-F0.5 ceiling they imply.

A true pair (S1 s, query q) is *irreducible* when q's address field is empty and s's name also belongs to at
least one other S1 of the same country: nothing in q says which of those S1 it belongs to, and a coin-flip link
costs more F0.5 than it gains. Two name keys:

* ``exact``: ``re.sub(r'\\s+', ' ', NFKC(name).lower()).strip()``
* ``core``:  the exact name without ``.``, as ``[a-z0-9]+`` tokens, legal-form words removed (S1 names that
  differ only in legal form count as the same name)

Ceiling: leave every irreducible pair unlinked and get everything else right, i.e. per S1 with T true pairs of
which k are irreducible, TP = P = T - k (F = 1 when T = 0, F = 0 when k = T > 0); the macro mean over all S1.

    python -m src.ceiling --data-dir <dataset>     # the numbers from the raw training files
"""
from __future__ import annotations

import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

LEGAL_WORDS = frozenset("inc llc ltd pvt private limited corp corporation co company llp pllc pc plc lp "
                        "incorporated pa lc".split())
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z0-9]+")
KEYS = ("exact", "core")


def exact_name(raw) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", raw or "").lower()).strip()


def core_name(raw) -> str:
    return " ".join(t for t in _TOK.findall(exact_name(raw).replace(".", "")) if t not in LEGAL_WORDS)


NAME_KEY = {"exact": exact_name, "core": core_name}


def addr_empty(addr: np.ndarray) -> np.ndarray:
    """The raw address field is empty (or whitespace only)."""
    return np.fromiter((not (a or "").strip() for a in addr), dtype=bool, count=len(addr))


def shared_name(names: np.ndarray, country: np.ndarray, key: str) -> np.ndarray:
    """For each S1: its ``key`` name belongs to at least one other S1 of the same country (empty keys never do)."""
    k = pd.Series([NAME_KEY[key](x) for x in names], dtype=object)
    df = pd.DataFrame({"c": pd.Series(country, dtype=object).to_numpy(), "k": k.to_numpy()})
    dup = df.duplicated(["c", "k"], keep=False).to_numpy()
    return dup & (k.to_numpy() != "")


def irreducible_flags(rec: pd.DataFrame, gt_s: np.ndarray, gt_q: np.ndarray) -> dict:
    """{'exact': bool[n_pairs], 'core': bool[n_pairs]} for true pairs given as row indices into ``rec``
    (columns name, addr, source, country)."""
    src = rec["source"].to_numpy()
    s1 = np.nonzero(src == 1)[0]
    names = rec["name"].to_numpy(dtype=object)
    ctry = rec["country"].astype(str).to_numpy()
    q_empty = addr_empty(rec["addr"].to_numpy(dtype=object)[gt_q])
    out = {}
    for key in KEYS:
        sh = np.zeros(len(rec), bool)
        sh[s1] = shared_name(names[s1], ctry[s1], key)
        out[key] = q_empty & sh[gt_s]
    return out


def f05_ceiling(n_true_s: np.ndarray, k_s: np.ndarray) -> np.ndarray:
    """Per-row ceiling F0.5: TP = P = T - k."""
    T = n_true_s.astype(np.float64)
    tp = T - k_s
    return np.where(T == 0, 1.0, np.where(tp > 0, 1.25 * tp / (0.25 * T + np.maximum(tp, 1e-9)), 0.0))


def ceiling_stats(n_true_s: np.ndarray, s1: np.ndarray, gt_s: np.ndarray, irr: dict,
                  country_s1: np.ndarray | None = None) -> dict:
    """Ceiling (overall / per country), irreducible pair counts and S1 with only irreducible matches."""
    out = {"true_pairs": int(len(gt_s))}
    for key, flag in irr.items():
        k = np.bincount(gt_s[flag], minlength=len(n_true_s)).astype(np.float64)
        f = f05_ceiling(n_true_s, k)[s1]
        out[f"irr_{key}_pairs"] = int(flag.sum())
        out[f"irr_{key}_share"] = float(flag.mean()) if len(flag) else 0.0
        out[f"ceiling_{key}"] = float(f.mean())
        out[f"irr_{key}_only_s1"] = int(((k[s1] == n_true_s[s1]) & (n_true_s[s1] > 0)).sum())
        if country_s1 is not None:
            for c in pd.unique(country_s1):
                out[f"ceiling_{key}_{c}"] = float(f[country_s1 == c].mean())
    return out


def main():
    """Section-3 numbers straight from the raw training files (no pipeline caches needed)."""
    from .io_utils import load_ground_truth, read_tsv, setup_logging
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    setup_logging()
    parts = []
    for s in (1, 2, 3):
        df = read_tsv(f"{a.data_dir}/train/train_source{s}.tsv")
        df["source"] = np.int8(s)
        parts.append(df)
    rec = pd.concat(parts, ignore_index=True).rename(columns={"business_name": "name", "business_address": "addr"})
    del parts
    row = pd.Series(np.arange(len(rec)), index=rec["entity_id"].to_numpy())
    gt = load_ground_truth(a.data_dir)
    gs = row.reindex([x for x, lst in gt.items() for _ in lst]).to_numpy().astype(np.int64)
    gq = row.reindex([y for lst in gt.values() for y in lst]).to_numpy().astype(np.int64)
    irr = irreducible_flags(rec, gs, gq)
    n_true = np.bincount(gs, minlength=len(rec))
    s1 = np.nonzero(rec["source"].to_numpy() == 1)[0]
    st = ceiling_stats(n_true, s1, gs, irr, rec["country"].to_numpy()[s1])
    for k, v in st.items():
        print(f"{k:28s} {v}")
    # F1: empty-address queries and how many of them are matched
    emp = addr_empty(rec["addr"].to_numpy(dtype=object))
    matched = np.zeros(len(rec), bool)
    matched[gq] = True
    for s in (2, 3):
        m = (rec["source"].to_numpy() == s) & emp
        print(f"S{s} empty-address records {int(m.sum())}, matched {int((m & matched).sum())}")


if __name__ == "__main__":
    main()
