"""v10 step 2 (GPU): fine-tune a pretrained cross-encoder (``src/ptce.py``) on the band pairs of the pack and score
them. Needs only the pack (``scripts/v10_export.py``), so it can run on any GPU machine.

    python scripts/v10_pt.py --pack ~/er_work_v10/pack --out ~/er_work_v10/pt --model xlmr [--folds 0,1]
        [--max-minutes 35] [--smoke]

Two fold models (the pack's ``fold``: each query record with all its candidate pairs and its true-match cluster in
one fold): the fold-k model is trained on the band pairs of the other fold, scores the band pairs of fold k (out of
fold) and all test band pairs. ``--folds 0`` and ``--folds 1`` can
run on two machines at once; copy both results into the same ``<out>/<tag>`` folder afterwards.
Writes ``<out>/<tag>/fold{k}_train.parquet`` (row, logit), ``fold{k}_test.parquet`` (row, logit), ``fold{k}.json``;
a finished fold is skipped when the script is started again. ``--smoke`` runs a 2-minute check into ``<tag>_smoke``.
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
from src import ptce as PT  # noqa: E402
from src.io_utils import setup_logging  # noqa: E402

log = logging.getLogger("er")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xlmr", help=f"preset {sorted(PT.MODELS)} (pinned revisions)")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=0, help="training batch (0 = 128 base / 64 large)")
    ap.add_argument("--lr", type=float, default=0.0, help="peak learning rate (0 = 3e-5 base / 1e-5 large)")
    ap.add_argument("--max-minutes", type=float, default=35.0, help="training time budget per fold model (0 = none)")
    ap.add_argument("--ln", type=int, default=32, help="name tokens kept per record")
    ap.add_argument("--la", type=int, default=48, help="address tokens kept per record")
    ap.add_argument("--pred-bs", type=int, default=0, help="scoring batch (0 = 1024 base / 512 large)")
    ap.add_argument("--no-save-model", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run (20k pairs, 1 min) to check the environment")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.model not in PT.MODELS:
        raise SystemExit(f"unknown model preset {a.model}; choose one of {sorted(PT.MODELS)}")
    name, rev, lic, n_par = PT.MODELS[a.model]
    large = n_par > 4e8
    bs = a.bs or (64 if large else 128)
    lr = a.lr or (1e-5 if large else 3e-5)
    pbs = a.pred_bs or (512 if large else 1024)
    tag = a.model + ("_smoke" if a.smoke else "")
    od = os.path.join(a.out, tag)
    os.makedirs(od, exist_ok=True)
    setup_logging()
    fh = logging.FileHandler(os.path.join(od, "pt.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    import torch
    import transformers
    import tokenizers
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.device == "cuda" and dev.type != "cuda":
        raise SystemExit("CUDA is not available to torch")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__}
    log.info("model %s @ %s (licence %s, %.0fM parameters); %s on %s%s; bs %d lr %.1e", name, rev, lic, n_par / 1e6,
             versions, dev, f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "", bs, lr)
    T0 = time.time()

    tok = PT.load_tokenizer(name, rev)
    fmt = PT.pair_format(tok)
    log.info("pair format: prefix %s middle %s suffix %s, pad %s", *fmt, tok.pad_token_id)
    tp = pd.read_parquet(os.path.join(a.pack, "train_pairs.parquet"))
    te = pd.read_parquet(os.path.join(a.pack, "test_pairs.parquet"))
    if a.smoke:
        tp = tp.sample(min(len(tp), 20_000), random_state=0).sort_values("row").reset_index(drop=True)
        te = te.sample(min(len(te), 5_000), random_state=0).sort_values("row").reset_index(drop=True)
    builders = {}
    for split, pairs in (("train", tp), ("test", te)):
        rec = pd.read_parquet(os.path.join(a.pack, f"rec_{split}.parquet"))
        need = np.unique(np.r_[pairs["q"].to_numpy(), pairs["s"].to_numpy()])
        rec = rec[np.isin(rec["gid"].to_numpy(), need)].reset_index(drop=True)
        lut = np.full(int(rec["gid"].max()) + 1, -1, dtype=np.int64)
        lut[rec["gid"].to_numpy()] = np.arange(len(rec))
        R, L = PT.tokenize_records(tok, rec["name"].tolist(), rec["addr"].tolist(), a.ln, a.la)
        log.info("%s: %d records tokenised, mean %.1f tokens (max %d)", split, len(rec), L.mean(), L.max())
        qi, si = lut[pairs["q"].to_numpy()], lut[pairs["s"].to_numpy()]
        assert (qi >= 0).all() and (si >= 0).all()
        builders[split] = (PT.PairBuilder(R, L, fmt, tok.pad_token_id, dev), qi, si)
    pb_tr, q_tr, s_tr = builders["train"]
    pb_te, q_te, s_te = builders["test"]
    y, w, fold = tp["y"].to_numpy().astype(np.float32), tp["w"].to_numpy(), tp["fold"].to_numpy()
    log.info("train band pairs %d (positives %.3f, folds %s), test band pairs %d", len(tp), y.mean(),
             np.bincount(fold, minlength=2).tolist(), len(te))

    for k in [int(x) for x in a.folds.split(",") if x != ""]:
        done = os.path.join(od, f"fold{k}.json")
        if os.path.exists(done):
            log.info("fold %d already done (%s): skipped", k, done)
            continue
        tr, va = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        log.info("fold %d: training on %d pairs, scoring %d out-of-fold + %d test pairs", k, len(tr), len(va), len(te))
        torch.manual_seed(k)
        model = PT.load_model(name, rev)
        model, info = PT.train(model, pb_tr, q_tr[tr], s_tr[tr], y[tr], w[tr], epochs=a.epochs, bs=bs, lr=lr,
                               max_minutes=(1.0 if a.smoke else a.max_minutes) or None, seed=k)
        if not a.no_save_model and not a.smoke:
            torch.save(model.state_dict(), os.path.join(od, f"fold{k}.pt"))
        lv = PT.predict(model, pb_tr, q_tr[va], s_tr[va], bs=pbs)
        pd.DataFrame({"row": tp["row"].to_numpy()[va], "logit": lv}).to_parquet(
            os.path.join(od, f"fold{k}_train.parquet"), index=False)
        pv = np.clip(1 / (1 + np.exp(-lv.astype(np.float64))), 1e-6, 1 - 1e-6)
        yv = y[va]
        ll = float(-(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)).mean())
        acc = float(((pv > 0.5) == (yv > 0.5)).mean())
        log.info("fold %d out-of-fold on the band: logloss %.5f accuracy %.5f (%d pairs)", k, ll, acc, len(va))
        lt = PT.predict(model, pb_te, q_te, s_te, bs=pbs)
        pd.DataFrame({"row": te["row"].to_numpy(), "logit": lt}).to_parquet(
            os.path.join(od, f"fold{k}_test.parquet"), index=False)
        meta = dict(info, fold=k, model=name, revision=rev, licence=lic, parameters=n_par, bs=bs, lr=lr,
                    epochs=a.epochs, ln=a.ln, la=a.la, oof_logloss=ll, oof_accuracy=acc, n_train=int(len(tr)),
                    n_oof=int(len(va)), n_test=int(len(te)), versions=versions,
                    minutes_total=round((time.time() - T0) / 60, 1))
        json.dump(meta, open(done, "w"), indent=1)
        log.info("fold %d DONE: %s", k, meta)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log.info("ALL DONE in %.1f min", (time.time() - T0) / 60)


if __name__ == "__main__":
    main()
