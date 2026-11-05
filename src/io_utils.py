"""Reading the challenge TSVs and writing the two submission files."""
from __future__ import annotations

import csv
import logging
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

log = logging.getLogger("er")


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")


@contextmanager
def timer(msg: str):
    t = time.time()
    log.info("%s ...", msg)
    yield
    log.info("%s done in %.1fs", msg, time.time() - t)


def read_tsv(path: str, usecols=None) -> pd.DataFrame:
    # QUOTE_NONE: a handful of addresses contain stray double quotes; every line has
    # exactly four tab-separated fields, so quotes are kept as literal characters.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, usecols=usecols, engine="c")


def load_split(data_dir: str, split: str) -> dict:
    """Load ``{split}_source{1,2,3}.tsv`` from ``data_dir/split``."""
    base = os.path.join(data_dir, split)
    out = {}
    for s in (1, 2, 3):
        df = read_tsv(os.path.join(base, f"{split}_source{s}.tsv"))
        df["source"] = np.int8(s)
        out[s] = df
        log.info("loaded %s source%d: %d rows", split, s, len(df))
    return out


def load_ground_truth(data_dir: str) -> dict:
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    return {a: [x for x in b.split(",") if x] for a, b in
            zip(gt["source1_entity_id"].to_numpy(), gt["matched_entity_ids"].to_numpy())}


def write_id_lists(path: str, header: tuple, s1_ids, lists: dict):
    """Write a submission-style TSV: one row per S1 id, comma-joined unique ids."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for s1 in s1_ids:
            ids = lists.get(s1, ())
            seen, uniq = set(), []
            for x in ids:
                if x not in seen and (x.startswith("S2-") or x.startswith("S3-")):
                    seen.add(x)
                    uniq.append(x)
            f.write(f"{s1}\t{','.join(uniq)}\n")
