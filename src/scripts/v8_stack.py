"""v8 step 2: stage-3 stack of the v6 stage-2 score and the character cross-encoder score; evaluation; test output.

    python scripts/v8_stack.py --data-dir <SR>/dataset --train-work ~/er_work_v6 --test-work ~/er_work_v7b \
        --ce-dir ~/er_work_v8/ce --work ~/er_work_v8 --out-dir <SR>/output_v8 [--n-jobs 63]

1. Training (out of fold, the pipeline's 5 model folds): stage-3 LightGBM on p1 / p2 (v6 OOF) + ce (``v8_ce.py`` OOF).
2. Evaluation on the training OOF: the decoding rule is re-tuned for the stage-3 scores; macro F0.5 overall and per
   country vs the v6 stage-2 scores with their own best rule, paired per S1 (mean difference and its s.e.).
3. Test: stage-3 scores for pairs whose S1 country has training labels; other countries keep p2 and the v6 decoding
   rule. The v7 same-address filler rule is applied to all pairs (as in v7), then decoding, then the two TSVs
   (candidate_pairs.tsv = the same scored pairs as v6/v7) and the validator.
Writes only to --work and --out-dir.
"""
import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import decode as DC  # noqa: E402
from src import fillers as FI  # noqa: E402
from src import stack as SK  # noqa: E402
from src.evaluation import Truth, evaluate, decode as ev_decode  # noqa: E402
from src.io_utils import setup_logging, write_id_lists  # noqa: E402
from src.prepare import fold_of  # noqa: E402

log = logging.getLogger("er")


def per_s1_f(pairs: pd.DataFrame, cfg: dict, truth: Truth, with_metrics: bool = False):
    assigned = DC.assign(pairs, "p")
    sel = ev_decode(assigned, cfg)
    m, df = evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), truth.n_true_s, truth.s1, truth.ctry_s1,
                     return_f=True)
    return (df["f"].to_numpy(), m) if with_metrics else df["f"].to_numpy()


GRID = [(miss, scale) for miss in (0.0, 0.02) for scale in (0.8, 0.9, 1.0, 1.1)]


def tune_grid(pairs: pd.DataFrame, truth: Truth, grid=GRID) -> dict:
    """Best expected-F0.5 decoding rule on a small grid (miss_p x prob_scale), with its OOF metrics."""
    assigned = DC.assign(pairs, "p")
    best = None
    for miss, scale in grid:
        cfg = {"method": "expected_f05", "miss_p": miss, "scale": scale}
        sel = ev_decode(assigned, cfg)
        m = evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), truth.n_true_s, truth.s1, truth.ctry_s1)
        log.info("  decode %s -> f05 %.5f (%s)", cfg, m["f05"],
                 " ".join(f"{k}={v:.5f}" for k, v in m.items() if k.startswith("f05_")))
        if best is None or m["f05"] > best[1]["f05"]:
            best = (cfg, m)
    return dict(best[0], oof=best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--ce-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--neg-weight", type=float, default=0.9)
    ap.add_argument("--no-filler-rule", action="store_true")
    ap.add_argument("--vocab-cap", action="store_true", help="v8b: cap same-address distractor-vocabulary swaps")
    ap.add_argument("--test-only", action="store_true", help="reuse <work>/stack.pkl (no training / evaluation)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    setup_logging()
    fh = logging.FileHandler(os.path.join(a.work, "stack.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    T0 = time.time()

    if a.test_only:
        with open(os.path.join(a.work, "stack.pkl"), "rb") as f:
            S = pickle.load(f)
        return test_output(a, S["models"], S["decode"], S["train_countries"], T0)

    # ---------------- training table
    truth = Truth(a.train_work, a.data_dir)
    rtr = pd.read_parquet(os.path.join(a.train_work, "train", "records.parquet"),
                          columns=["entity_id", "country"] + SK.REC_COLS)
    oof = pd.read_parquet(os.path.join(a.train_work, "train", "oof.parquet"))
    ce = pd.read_parquet(os.path.join(a.ce_dir, "ce_train.parquet"))
    if len(ce) != len(oof) or not (np.array_equal(ce["s"].to_numpy(), oof["s"].to_numpy())
                                   and np.array_equal(ce["q"].to_numpy(), oof["q"].to_numpy())):
        ce = oof[["s", "q"]].merge(ce, on=["s", "q"], how="left")
    oof["ce"] = ce["ce"].to_numpy()
    if oof["ce"].isna().any():
        raise SystemExit(f"{int(oof['ce'].isna().sum())} training pairs have no cross-encoder score")
    s, q, y = oof["s"].to_numpy(), oof["q"].to_numpy(), oof["y"].to_numpy()
    X = SK.features(s, q, oof["p1"].to_numpy(), oof["p2"].to_numpy(), oof["ce"].to_numpy(), SK.record_view(rtr))
    fold = fold_of(rtr["entity_id"].to_numpy(dtype=object), 5)[s]
    matched = np.zeros(len(rtr), bool)
    matched[truth.gtp["q"].to_numpy()] = True
    w = np.where((y == 0) & ~matched[q], 1.0 + a.neg_weight, 1.0).astype(np.float32)
    log.info("stage-3 training: %d pairs, folds %s", len(oof), np.bincount(fold).tolist())
    p3, models = SK.train_oof(X, y, fold, s, w, a.n_jobs)
    log.info("stage-3 importance:\n%s", SK.importance(models).round(4).to_string())

    # ---------------- evaluation (training OOF): v6 stage 2 vs stage 3, each with its best decoding rule
    base_pairs = pd.DataFrame({"s": s, "q": q, "p": oof["p2"].to_numpy(), "y": y})
    new_pairs = base_pairs.assign(p=p3)
    ce_pairs = base_pairs.assign(p=oof["ce"].to_numpy())
    # the v6 run's own decoding rule (tuned on these OOF p2 scores) is the reference; the stage-3 rule is re-tuned
    # on a reduced grid (the expected-F0.5 decoder has always won on this data)
    with open(os.path.join(a.train_work, "artifacts", "models.pkl"), "rb") as fm:
        cfg_base = pickle.load(fm)["decode"]
    f_base, m_base = per_s1_f(base_pairs, cfg_base, truth, with_metrics=True)
    cfg_base = dict(cfg_base, oof=m_base)
    cfg_new = tune_grid(new_pairs, truth)
    cfg_ce = tune_grid(ce_pairs, truth, grid=[(0.0, 1.0)])
    f_new = per_s1_f(new_pairs, cfg_new, truth)
    d = f_new - f_base
    res = {"oof_v6_stage2": cfg_base["oof"], "oof_v8_stage3": cfg_new["oof"], "oof_ce_alone": cfg_ce["oof"],
           "decode_v6": {k: v for k, v in cfg_base.items() if k != "oof"},
           "decode_v8": {k: v for k, v in cfg_new.items() if k != "oof"},
           "paired_delta": float(d.mean()), "paired_se": float(d.std(ddof=1) / np.sqrt(len(d))),
           "s1_better": int((d > 1e-12).sum()), "s1_worse": int((d < -1e-12).sum())}
    for c in pd.unique(truth.ctry_s1):
        m = truth.ctry_s1 == c
        res[f"paired_delta_{c}"] = float(d[m].mean())
    ll = lambda p: float(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
    res["logloss"] = {"p2": ll(oof["p2"].to_numpy()), "ce": ll(oof["ce"].to_numpy()), "p3": ll(p3)}
    log.info("EVAL %s", json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.work, "eval_v8.json"), "w"), indent=1)
    pd.DataFrame({"s": s, "q": q, "y": y, "p2": oof["p2"].to_numpy(), "ce": oof["ce"].to_numpy(), "p3": p3}) \
        .to_parquet(os.path.join(a.work, "oof_v8.parquet"), index=False)
    with open(os.path.join(a.work, "stack.pkl"), "wb") as f:
        pickle.dump({"models": models, "features": SK.FEATURES, "decode": cfg_new, "decode_v6": cfg_base,
                     "train_countries": sorted(set(truth.ctry_s1))}, f)
    log.info("training + evaluation done in %.1f min", (time.time() - T0) / 60)
    test_output(a, models, cfg_new, sorted(set(truth.ctry_s1)), T0)


def test_output(a, models, cfg_new: dict, train_countries: list, T0: float):
    # ---------------- test
    need = sorted(set(["entity_id", "source", "country", "a_empty", "n_core", "n_legal", "a_hn", "a_hnc", "a_street",
                       "a_nums", "a_unit"] + SK.REC_COLS))
    rte = pd.read_parquet(os.path.join(a.test_work, "test", "records.parquet"), columns=need)
    sc = pd.read_parquet(os.path.join(a.test_work, "test", "scores.parquet"))
    cte = pd.read_parquet(os.path.join(a.ce_dir, "ce_test.parquet"))
    if len(cte) != len(sc) or not (np.array_equal(cte["s"].to_numpy(), sc["s"].to_numpy())
                                   and np.array_equal(cte["q"].to_numpy(), sc["q"].to_numpy())):
        cte = sc[["s", "q"]].merge(cte, on=["s", "q"], how="left")
    sc["ce"] = cte["ce"].to_numpy()
    if sc["ce"].isna().any():
        raise SystemExit(f"{int(sc['ce'].isna().sum())} test pairs have no cross-encoder score")
    st, qt = sc["s"].to_numpy(), sc["q"].to_numpy()
    Xt = SK.features(st, qt, sc["p1"].to_numpy(), sc["p2"].to_numpy(), sc["ce"].to_numpy(), SK.record_view(rte))
    p3t = SK.predict_avg(models, Xt)
    ctry = rte["country"].astype(str).to_numpy()
    known = np.isin(ctry[st], list(train_countries))
    sc["p3"] = np.where(known, p3t, sc["p2"].to_numpy()).astype(np.float32)
    log.info("test pairs %d; stage-3 score used for %d (countries with training labels), stage-2 kept for %d (%s)",
             len(sc), int(known.sum()), int((~known).sum()), sorted(set(ctry[st[~known]])))
    if not a.no_filler_rule:
        cc = pd.read_parquet(os.path.join(a.test_work, "test", "cand_ctx.parquet"), columns=["q", "s", "rank"])
        top = cc[cc["rank"].to_numpy() == 0].reset_index(drop=True)
        del cc
        sc = FI.apply(sc, top, rte, known=None, col="p3", log=log)
    if a.vocab_cap:
        if a.no_filler_rule:
            cc = pd.read_parquet(os.path.join(a.test_work, "test", "cand_ctx.parquet"), columns=["q", "s", "rank"])
            top = cc[cc["rank"].to_numpy() == 0].reset_index(drop=True)
            del cc
        sc = FI.apply_cap(sc, top, rte, col="p3", log=log)
    sc[["s", "q", "p1", "p2", "ce", "p3"]].to_parquet(os.path.join(a.work, "test_scores_v8.parquet"), index=False)

    # decode: known countries with the stage-3 rule, the others with the v6 rule (unchanged from v7)
    with open(os.path.join(a.test_work, "artifacts", "models.pkl"), "rb") as f:
        cfg_v6_run = pickle.load(f)["decode"]
    pairs = pd.DataFrame({"s": st, "q": qt, "p": sc["p3"].to_numpy()})
    parts = []
    for flag, cfg in ((True, cfg_new), (False, cfg_v6_run)):
        sub = pairs[known == flag]
        if not len(sub):
            continue
        assigned = DC.assign(sub, "p")
        if cfg["method"] == "threshold":
            parts.append(DC.select_threshold(assigned, cfg["t"]))
        else:
            parts.append(DC.select_expected_f05(assigned, "p", miss_p=cfg["miss_p"], prob_scale=cfg["scale"]))
        log.info("decoded %s countries with %s", "labelled" if flag else "unlabelled",
                 {k: v for k, v in cfg.items() if k != "oof"})
    sel = pd.concat(parts, ignore_index=True)
    ids = rte["entity_id"].to_numpy(dtype=object)
    src = rte["source"].to_numpy()
    s1_ids = ids[src == 1]
    matches = {x: [] for x in s1_ids}
    for x, z in zip(ids[sel["s"].to_numpy()], ids[sel["q"].to_numpy()]):
        matches[x].append(z)
    cands = {x: [] for x in s1_ids}
    for x, z in zip(ids[st], ids[qt]):
        cands[x].append(z)
    os.makedirs(a.out_dir, exist_ok=True)
    write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), ("source1_entity_id", "matched_entity_ids"),
                   s1_ids, matches)
    write_id_lists(os.path.join(a.out_dir, "candidate_pairs.tsv"), ("source1_entity_id", "candidate_entity_ids"),
                   s1_ids, cands)
    s1_rows = np.nonzero(src == 1)[0]
    n_sel = np.bincount(sel["s"].to_numpy(), minlength=len(rte))[s1_rows]
    c1 = ctry[s1_rows]
    log.info("wrote %d S1 rows (%d with matches, %d matched ids); %d candidate pairs", len(s1_ids),
             int((n_sel > 0).sum()), len(sel), len(sc))
    for c in pd.unique(c1):
        m = c1 == c
        log.info("  %s: %d S1, %.3f matches per S1, %.2f%% empty", c, m.sum(), n_sel[m].mean(),
                 100 * (n_sel[m] == 0).mean())
    v = os.path.abspath(os.path.join(a.data_dir, "..", "utils", "validate_submission.py"))
    if os.path.exists(v):
        subprocess.run([sys.executable, v, "--matching", os.path.join(a.out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(a.out_dir, "candidate_pairs.tsv"), "--test-dir",
                        os.path.join(a.data_dir, "test"), "--check-ids"], check=False)
    log.info("DONE in %.1f min", (time.time() - T0) / 60)


if __name__ == "__main__":
    main()
