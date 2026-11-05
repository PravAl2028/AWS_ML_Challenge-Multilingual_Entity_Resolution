"""Stage 1: load a split, assign folds, learn native-script dictionaries, normalise every record."""
from __future__ import annotations

import collections
import logging
import os
import pickle
import zlib

import numpy as np
import pandas as pd

from . import text_address as TA
from . import text_names as TN
from .io_utils import load_ground_truth, load_split, timer
from .par import G, pmap
from .translit import NativeDict, mine_pairs

log = logging.getLogger("er")
N_FOLDS = 2


def fold_of(ids, n: int = N_FOLDS) -> np.ndarray:
    return np.fromiter((zlib.crc32(x.encode()) % n for x in ids), dtype=np.int8, count=len(ids))


def _s1_vocab(names) -> collections.Counter:
    c = collections.Counter()
    for n in names:
        c.update(t for t in TN.basic_tokens(n) if t.isalpha())
    return c


def _norm_chunk(task):
    names, addrs, countries, dfold = task
    g = G()
    vocab, seg, dicts = g['vocab'], g['seg'], g['dicts']
    out_n = [TN.parse_name(n, vocab, dicts.get(int(d)), seg) for n, d in zip(names, dfold)]
    out_a = [TA.parse_address(a, c) for a, c in zip(addrs, countries)]
    cols = {}
    for k in out_n[0]:
        cols[k] = [r[k] for r in out_n]
    for k in out_a[0]:
        cols[k] = [r[k] for r in out_a]
    return cols


def build_records(data_dir: str, split: str, art_dir: str, n_jobs: int, chunk: int = 20000) -> pd.DataFrame:
    """Return the normalised record table of a split (all three sources, one row per record)."""
    src = load_split(data_dir, split)
    rec = pd.concat([src[1], src[2], src[3]], ignore_index=True)
    del src
    rec = rec.rename(columns={"business_name": "name", "business_address": "addr"})
    rec["source"] = rec["source"].astype(np.int8)
    n1 = int((rec["source"] == 1).sum())
    log.info("%s: %d records (%d S1)", split, len(rec), n1)

    # ---- folds (train) and native-script dictionaries
    dicts_path = os.path.join(art_dir, "native_stats.pkl")
    rec["fold"] = np.int8(-1)
    rec["dfold"] = np.int8(N_FOLDS)  # which dictionary normalises this record (N_FOLDS = full dict)
    with timer("S1 vocabulary"):
        vocab_counts = _s1_vocab(rec["name"].to_numpy()[:n1])
    if split == "train":
        gt = load_ground_truth(data_dir)
        s1_ids = rec["entity_id"].to_numpy()[:n1]
        f1 = fold_of(s1_ids)
        s1_fold = dict(zip(s1_ids, f1))
        q_fold = {}
        for s1, lst in gt.items():
            f = s1_fold.get(s1)
            for q in lst:
                q_fold[q] = f
        ids = rec["entity_id"].to_numpy()
        qf = fold_of(ids[n1:])
        known = np.fromiter((q_fold.get(x, -1) for x in ids[n1:]), dtype=np.int8, count=len(ids) - n1)
        qf = np.where(known >= 0, known, qf)
        rec.loc[:n1 - 1, "fold"] = f1
        rec.loc[n1:, "fold"] = qf
        # mine (S1 name, native name) pairs per fold
        name_of = dict(zip(ids, rec["name"].to_numpy()))
        per_fold = [[] for _ in range(N_FOLDS)]
        for s1, lst in gt.items():
            f = s1_fold.get(s1)
            if f is None:
                continue
            a = name_of[s1]
            for q in lst:
                b = name_of.get(q)
                if b is not None and TN.INDIC_RX.search(b):
                    per_fold[f].append((a, b))
        stats = [mine_pairs(p) for p in per_fold]
        full = collections.defaultdict(collections.Counter)
        for st in stats:
            for k, c in st.items():
                full[k].update(c)
        with open(dicts_path, "wb") as f:
            pickle.dump(dict(full), f)
        log.info("native dictionary: %d tokens (fold sizes %s)", len(full), [len(s) for s in stats])
        # a query in fold k is normalised with the dictionary mined from the OTHER fold(s)
        dicts = {}
        for k in range(N_FOLDS):
            other = collections.defaultdict(collections.Counter)
            for j, st in enumerate(stats):
                if j != k:
                    for t, c in st.items():
                        other[t].update(c)
            dicts[k] = NativeDict(other, vocab_counts)
        rec.loc[n1:, "dfold"] = rec.loc[n1:, "fold"].to_numpy()
    else:
        with open(dicts_path, "rb") as f:
            full = pickle.load(f)
        dicts = {N_FOLDS: NativeDict(full, vocab_counts)}
    dicts.setdefault(N_FOLDS, NativeDict(full, vocab_counts))

    seg = TN.UnigramSegmenter(vocab_counts)
    vocab = set(vocab_counts)
    names = rec["name"].to_numpy(dtype=object)
    addrs = rec["addr"].to_numpy(dtype=object)
    ctry = rec["country"].to_numpy(dtype=object)
    dfold = rec["dfold"].to_numpy()
    tasks = [(names[i:i + chunk], addrs[i:i + chunk], ctry[i:i + chunk], dfold[i:i + chunk])
             for i in range(0, len(rec), chunk)]
    with timer(f"normalising {len(rec)} records on {n_jobs} workers"):
        parts = pmap(_norm_chunk, tasks, n_jobs, shared={"vocab": vocab, "seg": seg, "dicts": dicts})
    for k in parts[0]:
        vals = [v for p in parts for v in p[k]]
        if k in ("n_flags", "a_units", "a_empty"):
            rec[k] = np.asarray(vals, dtype=np.int16)
        elif k in ("a_hn", "a_hnl"):
            rec[k] = np.asarray(vals, dtype=np.int64)
        else:
            rec[k] = pd.array(vals, dtype="string")
    del parts
    rec["country"] = rec["country"].astype("category")
    rec["gid"] = np.arange(len(rec), dtype=np.int32)
    return rec
