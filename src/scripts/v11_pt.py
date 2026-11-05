"""v11 GPU step: fine-tune a pretrained cross-encoder (``src/ptce.py``) on the v10 band pack and score it -- the same
job as ``scripts/v10_pt.py`` (same pack, same folds, same outputs, so ``scripts/v10_stack.py`` reads the result like
the v10 XLM-R scores), with three additions for the last hours before the deadline:

* ``--deadline-utc HH:MM``: every fold must be trained AND scored before that time. Before each fold the training
  budget is set to what is left after the estimated scoring time of the remaining folds (scoring speed measured on the
  previous fold, or ``--score-rate`` pairs/s for the first one); a fold that cannot get ``--min-train-minutes`` is not
  started, so a model is either complete (both folds) or skipped -- never half-done.
* when the pinned revision of a model cannot be fetched, the model's main revision is used and the resolved commit is
  recorded in ``fold{k}.json`` (``revision_used``);
* measured scoring time / pairs in ``fold{k}.json`` (used by ``sm_v11_gpu.sh`` to size the next model).

    python scripts/v11_pt.py --pack ~/er_work_v10/pack --out ~/er_work_v10/pt --model e5 [--folds 0,1]
        [--max-minutes 35] [--deadline-utc 16:00 --score-rate 1500] [--no-save-model] [--smoke]
"""
import argparse
import datetime as dt
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


def minutes_until(hhmm: str, now: dt.datetime | None = None) -> float:
    """Minutes from now (UTC) until today's HH:MM UTC (negative when it has passed)."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    h, m = (int(x) for x in hhmm.split(":"))
    return (now.replace(hour=h, minute=m, second=0, microsecond=0) - now).total_seconds() / 60.0


def fold_budget(mins_left: float, folds_left: int, score_min: float, cap: float, load_min: float = 1.5,
                margin: float = 3.0) -> float:
    """Training minutes for the next fold so that it and the remaining folds are trained and scored in time."""
    per_fold = (mins_left - margin) / max(folds_left, 1) - score_min - load_min
    return float(min(cap, per_fold))


def load_pretrained(name: str, rev: str):
    """Tokenizer + model at the pinned revision; the main revision when the pinned one cannot be fetched."""
    try:
        tok = PT.load_tokenizer(name, rev)
        model = PT.load_model(name, rev)
        return tok, model, rev
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - any hub / revision error
        log.warning("%s @ %s could not be loaded (%s: %s); using the main revision", name, rev, type(e).__name__,
                    str(e)[:200])
        tok = PT.load_tokenizer(name, None)
        model = PT.load_model(name, None)
        used = getattr(model.config, "_commit_hash", None) or "main"
        log.warning("%s: resolved revision %s", name, used)
        return tok, model, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="e5", help=f"preset {sorted(PT.MODELS)} (pinned revisions)")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=0, help="training batch (0 = 128 base / 64 large)")
    ap.add_argument("--lr", type=float, default=0.0, help="peak learning rate (0 = 3e-5 base / 1e-5 large)")
    ap.add_argument("--max-minutes", type=float, default=35.0, help="training time cap per fold model (0 = none)")
    ap.add_argument("--deadline-utc", default="", help="HH:MM UTC by which every fold must be trained and scored")
    ap.add_argument("--score-rate", type=float, default=0.0,
                    help="expected scoring speed (pairs/s) for the first fold's budget; later folds use the measured one")
    ap.add_argument("--min-train-minutes", type=float, default=10.0)
    ap.add_argument("--ln", type=int, default=32, help="name tokens kept per record")
    ap.add_argument("--la", type=int, default=48, help="address tokens kept per record")
    ap.add_argument("--pred-bs", type=int, default=0, help="scoring batch (0 = 1024 base / 512 large)")
    ap.add_argument("--tag", default="", help="output sub-folder (default: the model preset)")
    ap.add_argument("--seed", type=int, default=-1, help="seed base (default: fold number, as v10_pt.py)")
    ap.add_argument("--swap", action="store_true", help="read the S1 record first (diversity for a second run)")
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
    tag = (a.tag or a.model) + ("_smoke" if a.smoke else "")
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

    tok, _m, rev_used = load_pretrained(name, rev)
    del _m
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
        if a.swap:  # S1 record first, query record second
            qi, si = si, qi
        builders[split] = (PT.PairBuilder(R, L, fmt, tok.pad_token_id, dev), qi, si)
    pb_tr, q_tr, s_tr = builders["train"]
    pb_te, q_te, s_te = builders["test"]
    y, w, fold = tp["y"].to_numpy().astype(np.float32), tp["w"].to_numpy(), tp["fold"].to_numpy()
    log.info("train band pairs %d (positives %.3f, folds %s), test band pairs %d", len(tp), y.mean(),
             np.bincount(fold, minlength=2).tolist(), len(te))

    folds = [int(x) for x in a.folds.split(",") if x != ""]
    todo = [k for k in folds if not os.path.exists(os.path.join(od, f"fold{k}.json"))]
    for k in folds:
        if k not in todo:
            log.info("fold %d already done: skipped", k)
    rate = a.score_rate if a.score_rate > 0 else None
    for i, k in enumerate(todo):
        tr, va = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        budget = (1.0 if a.smoke else a.max_minutes) or None
        if a.deadline_utc and not a.smoke:
            left = minutes_until(a.deadline_utc)
            if rate:
                score_min = (len(va) + len(te)) / rate / 60.0
                b = fold_budget(left, len(todo) - i, score_min, a.max_minutes or 1e9)
                log.info("fold %d budget: %.0f min to the deadline %s UTC, %d fold(s) left, scoring ~%.1f min each at "
                         "%.0f pairs/s -> %.1f training minutes (cap %s)", k, left, a.deadline_utc, len(todo) - i,
                         score_min, rate, b, a.max_minutes)
                if b < a.min_train_minutes:
                    log.warning("fold %d NOT started: %.1f training minutes left (< %.0f); model %s stays incomplete "
                                "and is ignored by the stack", k, b, a.min_train_minutes, a.model)
                    break
                budget = b
            else:
                log.warning("no scoring-speed estimate: deadline not enforced for fold %d", k)
        log.info("fold %d: training on %d pairs (budget %s min), scoring %d out-of-fold + %d test pairs", k, len(tr),
                 None if budget is None else round(budget, 1), len(va), len(te))
        seed = k if a.seed < 0 else 100 * a.seed + k
        torch.manual_seed(seed)
        model = PT.load_model(name, rev_used if rev_used != "main" else None)
        model, info = PT.train(model, pb_tr, q_tr[tr], s_tr[tr], y[tr], w[tr], epochs=a.epochs, bs=bs, lr=lr,
                               max_minutes=budget, seed=seed)
        if not a.no_save_model and not a.smoke:
            torch.save(model.state_dict(), os.path.join(od, f"fold{k}.pt"))
        t_s = time.time()
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
        score_sec = time.time() - t_s
        rate = (len(va) + len(te)) / max(score_sec, 1e-9)
        meta = dict(info, fold=k, tag=tag, seed=seed, swap=bool(a.swap), model=name, revision=rev,
                    revision_used=rev_used, licence=lic, parameters=n_par,
                    bs=bs, lr=lr, epochs=a.epochs, ln=a.ln, la=a.la, oof_logloss=ll, oof_accuracy=acc,
                    n_train=int(len(tr)), n_oof=int(len(va)), n_test=int(len(te)), versions=versions,
                    score_minutes=round(score_sec / 60, 2), score_pairs=int(len(va) + len(te)),
                    score_rate=round(rate, 1), train_budget_minutes=budget,
                    minutes_total=round((time.time() - T0) / 60, 1))
        json.dump(meta, open(os.path.join(od, f"fold{k}.json"), "w"), indent=1)
        log.info("fold %d DONE: %s", k, meta)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    done = [k for k in folds if os.path.exists(os.path.join(od, f"fold{k}.json"))]
    log.info("ALL DONE in %.1f min: folds complete %s of %s", (time.time() - T0) / 60, done, folds)


if __name__ == "__main__":
    main()
