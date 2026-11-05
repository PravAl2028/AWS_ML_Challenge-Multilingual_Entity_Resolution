"""v10 step 1 (CPU, main box): choose the uncertain pairs and export a small, self-contained "pack" for the GPU step.

    python scripts/v10_export.py --data-dir <SR>/dataset --train-work ~/er_work_v6 --test-work ~/er_work_v7b \
        --v8-work ~/er_work_v8 --work ~/er_work_v10 [--lo 0.002 --hi 0.998 --max-train 2500000]

Band: a pair is "uncertain" when its v8 stage-3 score p3 lies in [lo, hi] (training: the v8 out-of-fold p3; test:
the v8 stack re-applied to the v6/v7b test scores, before the filler rule). When the training band holds more than
``--max-train`` pairs it is narrowed symmetrically in logit space; the same cut-off is applied to the test pairs.
Test pairs of countries without training labels are left out (they keep the v8 = v7 decisions).

Writes (only) to ``<work>/pack``: rec_train / rec_test (gid, name, addr of the records used), train_pairs (row, q, s,
y, w, fold), test_pairs (row, q, s), meta.json; and ``<work>/test_p3.parquet`` (v8 stage-3 test scores, aligned with
the v6 test scores table) for the stack step. ``row`` is the pair's row in the training OOF table / test scores table.
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import stack as SK  # noqa: E402
from src.evaluation import Truth  # noqa: E402
from src.io_utils import setup_logging  # noqa: E402
from src.prepare import fold_of  # noqa: E402

log = logging.getLogger("er")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--v8-work", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--lo", type=float, default=0.002)
    ap.add_argument("--hi", type=float, default=0.998)
    ap.add_argument("--max-train", type=int, default=2_500_000)
    ap.add_argument("--neg-weight", type=float, default=0.9)
    a = ap.parse_args()
    out = os.path.join(a.work, "pack")
    os.makedirs(out, exist_ok=True)
    setup_logging()
    T0 = time.time()

    # ---------------- training band (v8 OOF p3)
    truth = Truth(a.train_work, a.data_dir)
    rtr = pd.read_parquet(os.path.join(a.train_work, "train", "records.parquet"),
                          columns=["entity_id", "name", "addr", "country"])
    o8 = pd.read_parquet(os.path.join(a.v8_work, "oof_v8.parquet"), columns=["s", "q", "y", "p3"])
    s, q, y = o8["s"].to_numpy(), o8["q"].to_numpy(), o8["y"].to_numpy()
    z = np.abs(logit(o8["p3"].to_numpy()))
    cut = min(abs(logit(a.lo)), abs(logit(a.hi)))
    band = z <= cut
    if band.sum() > a.max_train:
        cut = float(np.partition(z, a.max_train - 1)[a.max_train - 1])
        band = z <= cut
        log.info("training band narrowed to |logit p3| <= %.3f (p in [%.4f, %.4f])", cut, 1 / (1 + np.exp(cut)),
                 1 / (1 + np.exp(-cut)))
    matched = np.zeros(len(rtr), bool)
    matched[truth.gtp["q"].to_numpy()] = True
    w = np.where((y == 0) & ~matched[q], 1.0 + a.neg_weight, 1.0).astype(np.float32)
    # folds of the pretrained model: every query record with ALL its candidate pairs in one fold -- the fold of its
    # true S1 when it has one (a true-match cluster stays together), else a hash of its own id. A fold model has
    # then never seen the query of a pair it scores (as on the test split), and an S1 of the scored fold appears in
    # the other fold only in no-match pairs.
    rf = fold_of(rtr["entity_id"].to_numpy(dtype=object), 2)
    qf = rf.copy()
    qf[truth.gtp["q"].to_numpy()] = rf[truth.gtp["s"].to_numpy()]
    fold = qf[q]
    rows = np.nonzero(band)[0]
    tp = pd.DataFrame({"row": rows.astype(np.int64), "q": q[rows], "s": s[rows], "y": y[rows].astype(np.int8),
                       "w": w[rows], "fold": fold[rows].astype(np.int8)})
    ctry_tr = truth.ctry[s[rows]]
    errs_in = float(np.where(y[rows] == 1, 1 - o8["p3"].to_numpy()[rows], o8["p3"].to_numpy()[rows]).sum())
    p3 = o8["p3"].to_numpy()
    errs_all = float(np.where(y == 1, 1 - p3, p3).sum())
    log.info("training pairs %d, band %d (%.2f%%), positives in band %.3f, fold sizes %s; expected errors in band "
             "%.0f of %.0f (%.1f%%)", len(o8), len(rows), 100 * len(rows) / len(o8), tp["y"].mean(),
             np.bincount(tp["fold"]).tolist(), errs_in, errs_all, 100 * errs_in / max(errs_all, 1e-9))
    pos = tp["y"].to_numpy() == 1
    log.info("fold check: positives in their S1's fold %.4f (must be ~1); queries split across folds %d",
             float((rf[tp["s"].to_numpy()[pos]] == tp["fold"].to_numpy()[pos]).mean()) if pos.any() else 1.0,
             int((tp.groupby("q")["fold"].nunique() > 1).sum()))

    # ---------------- test: v8 stage-3 scores before the filler rule
    rte = pd.read_parquet(os.path.join(a.test_work, "test", "records.parquet"),
                          columns=sorted(set(["entity_id", "name", "addr", "country"] + SK.REC_COLS)))
    sc = pd.read_parquet(os.path.join(a.test_work, "test", "scores.parquet"), columns=["s", "q", "p1", "p2"])
    cte = pd.read_parquet(os.path.join(a.v8_work, "ce", "ce_test.parquet"))
    if len(cte) != len(sc) or not (np.array_equal(cte["s"].to_numpy(), sc["s"].to_numpy())
                                   and np.array_equal(cte["q"].to_numpy(), sc["q"].to_numpy())):
        cte = sc[["s", "q"]].merge(cte, on=["s", "q"], how="left")
    sc["ce"] = cte["ce"].to_numpy()
    if sc["ce"].isna().any():
        raise SystemExit("test pairs without a cross-encoder score")
    with open(os.path.join(a.v8_work, "stack.pkl"), "rb") as f:
        S8 = pickle.load(f)
    st, qt = sc["s"].to_numpy(), sc["q"].to_numpy()
    Xt = SK.features(st, qt, sc["p1"].to_numpy(), sc["p2"].to_numpy(), sc["ce"].to_numpy(), SK.record_view(rte))
    p3t = SK.predict_avg(S8["models"], Xt)
    del Xt
    pd.DataFrame({"s": st, "q": qt, "p3": p3t}).to_parquet(os.path.join(a.work, "test_p3.parquet"), index=False)
    ctry = rte["country"].astype(str).to_numpy()
    known = np.isin(ctry[st], S8["train_countries"])
    bt = known & (np.abs(logit(p3t)) <= cut)
    trow = np.nonzero(bt)[0]
    te = pd.DataFrame({"row": trow.astype(np.int64), "q": qt[trow], "s": st[trow]})
    log.info("test pairs %d (%d of countries with training labels), band %d (%.2f%% of those)", len(sc),
             int(known.sum()), len(trow), 100 * len(trow) / max(int(known.sum()), 1))
    for c in sorted(set(ctry[st[trow]])):
        log.info("  test band %s: %d pairs; training band %s: %d pairs", c, int((ctry[st[trow]] == c).sum()), c,
                 int((ctry_tr == c).sum()))

    # ---------------- records used by the band pairs
    g_tr = np.unique(np.r_[tp["q"].to_numpy(), tp["s"].to_numpy()])
    g_te = np.unique(np.r_[te["q"].to_numpy(), te["s"].to_numpy()])
    pd.DataFrame({"gid": g_tr, "name": rtr["name"].to_numpy(dtype=object)[g_tr],
                  "addr": rtr["addr"].to_numpy(dtype=object)[g_tr]}).to_parquet(os.path.join(out, "rec_train.parquet"),
                                                                                  index=False)
    pd.DataFrame({"gid": g_te, "name": rte["name"].to_numpy(dtype=object)[g_te],
                  "addr": rte["addr"].to_numpy(dtype=object)[g_te]}).to_parquet(os.path.join(out, "rec_test.parquet"),
                                                                                 index=False)
    tp.to_parquet(os.path.join(out, "train_pairs.parquet"), index=False)
    te.to_parquet(os.path.join(out, "test_pairs.parquet"), index=False)
    meta = {"band_logit_cut": float(cut), "band_p": [float(1 / (1 + np.exp(cut))), float(1 / (1 + np.exp(-cut)))],
            "train_pairs_total": int(len(o8)), "train_band": int(len(rows)), "train_band_pos": float(tp["y"].mean()),
            "train_expected_errors_share_in_band": errs_in / max(errs_all, 1e-9),
            "test_pairs_total": int(len(sc)), "test_band": int(len(trow)), "train_countries": S8["train_countries"],
            "records_train": int(len(g_tr)), "records_test": int(len(g_te)), "neg_weight": a.neg_weight,
            "fold_split": "query record (true-match cluster)",
            "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    log.info("DONE: %s", meta)


if __name__ == "__main__":
    main()
