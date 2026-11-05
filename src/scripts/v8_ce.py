"""v8 step 1: train the character cross-encoder (``src/charce.py``) and score every candidate pair (GPU).

    python scripts/v8_ce.py --data-dir <SR>/dataset --train-work ~/er_work_v6 --test-work ~/er_work_v7b \
        --out ~/er_work_v8/ce [--epochs 1] [--max-minutes 25]

Inputs (read only): ``<train-work>/train/{records,oof}.parquet`` (the v6 run: training candidate pairs with labels
and out-of-fold p1/p2) and ``<test-work>/test/{records,scores}.parquet`` (the v6 test pairs and scores).
Two fold models, split by Source-1 entity: each is trained on the pairs of one half of the S1 and scores the other
half (out of fold) and all test pairs. Writes ``ce_train.parquet`` (s, q, ce), ``ce_test.parquet`` (s, q, ce, with
the mean of the two models), the models and the character table to ``--out``.
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import charce as CE  # noqa: E402
from src.evaluation import gt_pairs  # noqa: E402
from src.io_utils import setup_logging  # noqa: E402
from src.prepare import fold_of  # noqa: E402

log = logging.getLogger("er")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=0, help="feed-forward width (0 = 4 x d)")
    ap.add_argument("--no-ngrams", action="store_true", help="no hashed bigram / trigram embeddings")
    ap.add_argument("--neg-weight", type=float, default=0.9, help="extra weight of distractor negatives (as v3+)")
    ap.add_argument("--max-train-pairs", type=int, default=0, help="subsample each fold's training pairs (0 = all)")
    ap.add_argument("--max-minutes", type=float, default=24.0, help="training time budget per fold model (the schedule is fitted to it after 200 steps); 0 = no budget")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    setup_logging()
    fh = logging.FileHandler(os.path.join(a.out, "ce.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    import torch
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.device == "cuda" and dev.type != "cuda":
        raise SystemExit("CUDA is not available to torch: install a torch build that matches the driver (see "
                         "RUNBOOK_V8.md) or pass --device cpu")
    log.info("torch %s on %s%s", torch.__version__, dev,
             f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    T0 = time.time()

    # ---------------- records -> character codes
    cols = ["entity_id", "name", "addr", "source", "country", "gid"]
    rtr = pd.read_parquet(os.path.join(a.train_work, "train", "records.parquet"), columns=cols)
    rte = pd.read_parquet(os.path.join(a.test_work, "test", "records.parquet"), columns=cols[:-1])
    for r in (rtr, rte):
        r["name"] = r["name"].fillna("").astype(str)
        r["addr"] = r["addr"].fillna("").astype(str)
    samp = rtr.sample(min(len(rtr), 600_000), random_state=0)
    lut = CE.build_lut(list(samp["name"]) + list(samp["addr"]))
    np.save(os.path.join(a.out, "lut.npy"), lut)
    t = time.time()
    tr_n = CE.encode(rtr["name"].tolist(), CE.LN, lut)
    tr_a = CE.encode(rtr["addr"].tolist(), CE.LA, lut)
    te_n = CE.encode(rte["name"].tolist(), CE.LN, lut)
    te_a = CE.encode(rte["addr"].tolist(), CE.LA, lut)
    log.info("encoded %d train + %d test records in %.0f s", len(rtr), len(rte), time.time() - t)

    # ---------------- training pairs (v6 OOF table) and test pairs (v6 scores)
    oof = pd.read_parquet(os.path.join(a.train_work, "train", "oof.parquet"), columns=["s", "q", "y"])
    sc = pd.read_parquet(os.path.join(a.test_work, "test", "scores.parquet"), columns=["s", "q"])
    s_tr, q_tr, y_tr = oof["s"].to_numpy(), oof["q"].to_numpy(), oof["y"].to_numpy().astype(np.float32)
    matched = np.zeros(len(rtr), bool)
    matched[gt_pairs(rtr, a.data_dir)["q"].to_numpy()] = True
    w_tr = np.where((y_tr == 0) & ~matched[q_tr], 1.0 + a.neg_weight, 1.0).astype(np.float32)
    fold = fold_of(rtr["entity_id"].to_numpy(dtype=object), 2)[s_tr]
    log.info("training pairs %d (positives %.3f), test pairs %d; CE folds sizes %s", len(oof), y_tr.mean(), len(sc),
             np.bincount(fold).tolist())

    src_tr = CE.PairSource(tr_n, tr_a, dev)
    src_te = CE.PairSource(te_n, te_a, dev)
    ce_tr = np.full(len(oof), np.nan, dtype=np.float32)
    ce_te = np.zeros(len(sc), dtype=np.float64)
    rng = np.random.default_rng(0)
    for k in (0, 1):
        tr = np.nonzero(fold != k)[0]
        if a.max_train_pairs and len(tr) > a.max_train_pairs:
            tr = np.sort(rng.choice(tr, a.max_train_pairs, replace=False))
        log.info("CE fold %d: training on %d pairs", k, len(tr))
        model = CE.build_model(d=a.d, layers=a.layers, ff=a.ff or 4 * a.d, heads=max(1, a.d // 32),
                               ngrams=not a.no_ngrams)
        log.info("  parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
        model = CE.train(model, src_tr, q_tr[tr], s_tr[tr], y_tr[tr], w_tr[tr], epochs=a.epochs, bs=a.bs, lr=a.lr,
                         max_minutes=a.max_minutes or None, seed=k)
        torch.save(model.state_dict(), os.path.join(a.out, f"ce_fold{k}.pt"))
        va = np.nonzero(fold == k)[0]
        ce_tr[va] = CE.predict(model, src_tr, q_tr[va], s_tr[va])
        yv = y_tr[va]
        pv = np.clip(ce_tr[va], 1e-6, 1 - 1e-6)
        ll = float(-(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)).mean())
        acc = float(((pv > 0.5) == (yv > 0.5)).mean())
        log.info("CE fold %d out-of-fold: logloss %.5f, accuracy %.5f (%d pairs)", k, ll, acc, len(va))
        ce_te += CE.predict(model, src_te, sc["q"].to_numpy(), sc["s"].to_numpy())
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log.info("  fold %d done at %.1f min", k, (time.time() - T0) / 60)
    ce_te /= 2
    pd.DataFrame({"s": s_tr, "q": q_tr, "ce": ce_tr}).to_parquet(os.path.join(a.out, "ce_train.parquet"), index=False)
    pd.DataFrame({"s": sc["s"].to_numpy(), "q": sc["q"].to_numpy(), "ce": ce_te.astype(np.float32)}).to_parquet(
        os.path.join(a.out, "ce_test.parquet"), index=False)
    pv = np.clip(ce_tr, 1e-6, 1 - 1e-6)
    meta = {"pairs_train": int(len(oof)), "pairs_test": int(len(sc)), "d": a.d, "layers": a.layers,
            "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "neg_weight": a.neg_weight,
            "oof_logloss": float(-(y_tr * np.log(pv) + (1 - y_tr) * np.log(1 - pv)).mean()),
            "minutes": round((time.time() - T0) / 60, 1)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    log.info("DONE: %s", meta)


if __name__ == "__main__":
    main()
