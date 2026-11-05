"""v10 step 3 (CPU, main box): stage-4 stack of the v8 scores and the pretrained cross-encoder score(s) on the band
pairs; evaluation vs v8 (training OOF, paired per S1); test output.

    python scripts/v10_stack.py --data-dir <SR>/dataset --train-work ~/er_work_v6 --test-work ~/er_work_v7b \
        --v8-work ~/er_work_v8 --work ~/er_work_v10 --out-dir <SR>/output_v10 [--tags xlmr,e5] [--v9-work ~/er_work_v9]

Band pairs (``<work>/pack``) get a new score p4 from a LightGBM trained out of fold (the pipeline's 5 folds, split by
Source-1 entity) on: the v8 stack features (``src/stack.py``), the v8 score p3, each pretrained model's out-of-fold
logit and the query-level competition of their mean (rank, margin over the query's best other S1); with ``--v9-work``
also the v9 stage-2 score. Pairs outside the band keep p3. Countries without training labels keep the v8 (= v7)
scores and decoding rule. Then the v7 filler rule, the expected-F0.5 decoder (grid re-tuned on the training OOF),
the two TSVs (candidate_pairs.tsv = the same scored pairs as v6-v8) and the validator.
Writes only to --work and --out-dir.
"""
import argparse
import glob
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import v8_stack as V8  # noqa: E402
from src import decode as DC  # noqa: E402
from src import fillers as FI  # noqa: E402
from src import stack as SK  # noqa: E402
from src.evaluation import Truth  # noqa: E402
from src.features import _grp_max_other, _grp_rank  # noqa: E402
from src.io_utils import setup_logging, write_id_lists  # noqa: E402
from src.prepare import fold_of  # noqa: E402

log = logging.getLogger("er")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))).astype(np.float32)


def find_tags(pt_dir: str, tags: str) -> list:
    if tags:
        out = [t for t in tags.split(",") if t]
    else:
        out = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(pt_dir, "*", "fold0.json")))
        out = [t for t in out if not t.endswith("_smoke")]
    ok = []
    for t in out:
        miss = [k for k in (0, 1) if not os.path.exists(os.path.join(pt_dir, t, f"fold{k}.json"))]
        if miss:
            log.warning("model %s: folds %s missing - not used", t, miss)
        else:
            ok.append(t)
    if not ok:
        raise SystemExit(f"no complete pretrained-model scores under {pt_dir}")
    return ok


def load_pt(pt_dir: str, tag: str, n_train: int, n_test: int):
    """Out-of-fold training logits (NaN outside the band) and mean test logits of the fold models."""
    tr = np.full(n_train, np.nan, dtype=np.float32)
    te_sum = np.zeros(n_test, dtype=np.float64)
    te_n = np.zeros(n_test, dtype=np.int8)
    for k in (0, 1):
        a = pd.read_parquet(os.path.join(pt_dir, tag, f"fold{k}_train.parquet"))
        tr[a["row"].to_numpy()] = a["logit"].to_numpy()
        b = pd.read_parquet(os.path.join(pt_dir, tag, f"fold{k}_test.parquet"))
        te_sum[b["row"].to_numpy()] += b["logit"].to_numpy()
        te_n[b["row"].to_numpy()] += 1
    te = np.where(te_n > 0, te_sum / np.maximum(te_n, 1), np.nan).astype(np.float32)
    meta = [json.load(open(os.path.join(pt_dir, tag, f"fold{k}.json"))) for k in (0, 1)]
    return tr, te, meta


def build_X(s, q, p1, p2, ce, p3, rv, pts: dict, p2v9=None) -> pd.DataFrame:
    X = SK.features(s, q, p1, p2, ce, rv)
    X["lp3"] = logit(p3)
    L = np.stack([pts[t] for t in sorted(pts)], axis=1)
    for i, t in enumerate(sorted(pts)):
        X[f"lpt_{t}"] = L[:, i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # rows outside the band: all NaN
        mix = np.nanmean(L, axis=1)
    pm = np.where(np.isnan(mix), p3, sigmoid(mix)).astype(np.float32)
    X["lptm"] = logit(pm)
    X["pt_rank_q"] = _grp_rank(q, pm).astype(np.int16)
    X["pt_alt_q"] = pm - _grp_max_other(q, pm)
    if p2v9 is not None:
        X["lp2v9"] = logit(p2v9)
    return X


def aligned(path: str, s: np.ndarray, q: np.ndarray, col: str) -> np.ndarray:
    d = pd.read_parquet(path, columns=["s", "q", col])
    if len(d) == len(s) and np.array_equal(d["s"].to_numpy(), s) and np.array_equal(d["q"].to_numpy(), q):
        return d[col].to_numpy()
    m = pd.DataFrame({"s": s, "q": q}).merge(d, on=["s", "q"], how="left")[col].to_numpy()
    if np.isnan(m).any():
        raise SystemExit(f"{path}: {int(np.isnan(m).sum())} pairs missing")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--v8-work", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pt-dir", default="", help="default <work>/pt")
    ap.add_argument("--tags", default="", help="pretrained models to use (default: every complete one)")
    ap.add_argument("--v9-work", default="", help="optional: add the v9 stage-2 score (train/oof, test/scores)")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--neg-weight", type=float, default=0.9)
    ap.add_argument("--name", default="v10")
    a = ap.parse_args()
    pt_dir = a.pt_dir or os.path.join(a.work, "pt")
    os.makedirs(a.work, exist_ok=True)
    setup_logging()
    fh = logging.FileHandler(os.path.join(a.work, f"stack_{a.name}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    T0 = time.time()
    tags = find_tags(pt_dir, a.tags)
    log.info("pretrained models: %s", tags)

    # ---------------- training table: v6 OOF p1/p2, v8 ce + p3, pretrained logits on the band
    truth = Truth(a.train_work, a.data_dir)
    rtr = pd.read_parquet(os.path.join(a.train_work, "train", "records.parquet"),
                          columns=["entity_id", "country"] + SK.REC_COLS)
    oof = pd.read_parquet(os.path.join(a.train_work, "train", "oof.parquet"), columns=["s", "q", "y", "p1", "p2"])
    s, q, y = oof["s"].to_numpy(), oof["q"].to_numpy(), oof["y"].to_numpy()
    ce = aligned(os.path.join(a.v8_work, "ce", "ce_train.parquet"), s, q, "ce")
    p3 = aligned(os.path.join(a.v8_work, "oof_v8.parquet"), s, q, "p3")
    tp = pd.read_parquet(os.path.join(a.work, "pack", "train_pairs.parquet"), columns=["row", "q", "s"])
    assert np.array_equal(s[tp["row"].to_numpy()], tp["s"].to_numpy()) and \
        np.array_equal(q[tp["row"].to_numpy()], tp["q"].to_numpy()), "pack rows do not match the OOF table"
    band = np.zeros(len(oof), bool)
    band[tp["row"].to_numpy()] = True
    te_pairs = pd.read_parquet(os.path.join(a.work, "pack", "test_pairs.parquet"), columns=["row", "q", "s"])
    sc = pd.read_parquet(os.path.join(a.test_work, "test", "scores.parquet"), columns=["s", "q", "p1", "p2"])
    pts_tr, pts_te, metas = {}, {}, {}
    for t in tags:
        pts_tr[t], pts_te[t], metas[t] = load_pt(pt_dir, t, len(oof), len(sc))
        cov = np.isfinite(pts_tr[t][band]).mean()
        log.info("model %s: %s @ %s; training band coverage %.4f; fold OOF logloss %s", t, metas[t][0]["model"],
                 metas[t][0]["revision"][:10], cov, [round(m["oof_logloss"], 5) for m in metas[t]])
        if cov < 0.999:
            raise SystemExit(f"model {t} does not cover the training band ({cov:.4f})")
    # test-side inputs are checked here, before the stage-4 training, so a mismatch fails in minutes
    st, qt = sc["s"].to_numpy(), sc["q"].to_numpy()
    tb = np.zeros(len(sc), bool)
    tb[te_pairs["row"].to_numpy()] = True
    assert np.array_equal(st[te_pairs["row"].to_numpy()], te_pairs["s"].to_numpy()) and         np.array_equal(qt[te_pairs["row"].to_numpy()], te_pairs["q"].to_numpy()), "pack test rows do not match"
    for t in tags:
        cov = np.isfinite(pts_te[t][tb]).mean()
        if cov < 0.999:
            raise SystemExit(f"model {t} does not cover the test band ({cov:.4f})")
    p3t = aligned(os.path.join(a.work, "test_p3.parquet"), st, qt, "p3")
    p2v9, p2v9t = None, None
    if a.v9_work:
        p2v9 = aligned(os.path.join(a.v9_work, "train", "oof.parquet"), s, q, "p2")
        p2v9t = aligned(os.path.join(a.v9_work, "test", "scores.parquet"), st, qt, "p2")
    X = build_X(s, q, oof["p1"].to_numpy(), oof["p2"].to_numpy(), ce, p3, SK.record_view(rtr), pts_tr, p2v9)
    feats = list(X.columns)
    fold = fold_of(rtr["entity_id"].to_numpy(dtype=object), 5)[s]
    matched = np.zeros(len(rtr), bool)
    matched[truth.gtp["q"].to_numpy()] = True
    w = np.where((y == 0) & ~matched[q], 1.0 + a.neg_weight, 1.0).astype(np.float32)
    bi = np.nonzero(band)[0]
    log.info("stage-4 training: %d band pairs of %d (positives %.3f), features %s", len(bi), len(oof), y[bi].mean(),
             feats)
    p4b, models = SK.train_oof(X.iloc[bi].reset_index(drop=True), y[bi], fold[bi], s[bi], w[bi], a.n_jobs)
    log.info("stage-4 importance:\n%s", SK.importance(models).round(4).to_string())
    p4 = p3.astype(np.float32).copy()
    p4[bi] = p4b

    # ---------------- evaluation: v8 (p3, its own rule) vs v10 (p4, re-tuned rule), paired per S1
    with open(os.path.join(a.v8_work, "stack.pkl"), "rb") as f:
        S8 = pickle.load(f)
    base = pd.DataFrame({"s": s, "q": q, "p": p3, "y": y})
    f_base, m_base = V8.per_s1_f(base, S8["decode"], truth, with_metrics=True)
    cfg_new = V8.tune_grid(base.assign(p=p4), truth)
    f_new = V8.per_s1_f(base.assign(p=p4), cfg_new, truth)
    mix = sigmoid(X["lptm"].to_numpy())
    cfg_pt = V8.tune_grid(base.assign(p=np.where(band, mix, p3)), truth, grid=[(0.0, 1.0)])
    d = f_new - f_base
    yb = y[bi]
    ll = lambda p: float(-(yb * np.log(np.clip(p, 1e-6, 1)) + (1 - yb) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
    res = {"oof_v8": m_base, "oof_v10": cfg_new["oof"], "oof_v8_with_pt_mix_on_band": cfg_pt["oof"],
           "decode_v10": {k: v for k, v in cfg_new.items() if k != "oof"},
           "paired_delta_vs_v8": float(d.mean()), "paired_se": float(d.std(ddof=1) / np.sqrt(len(d))),
           "s1_better": int((d > 1e-12).sum()), "s1_worse": int((d < -1e-12).sum()),
           "band_logloss": {"p2": ll(oof["p2"].to_numpy()[bi]), "p3_v8": ll(p3[bi]), "pt_mix": ll(mix[bi]),
                            "p4_v10": ll(p4b)},
           "models": {t: {"model": metas[t][0]["model"], "revision": metas[t][0]["revision"],
                          "licence": metas[t][0]["licence"], "parameters": metas[t][0]["parameters"],
                          "versions": metas[t][0]["versions"], "steps": [m["steps"] for m in metas[t]],
                          "pairs_seen": [m["pairs_seen"] for m in metas[t]]} for t in tags},
           "v9_feature": bool(a.v9_work)}
    for c in pd.unique(truth.ctry_s1):
        res[f"paired_delta_{c}"] = float(d[truth.ctry_s1 == c].mean())
    log.info("EVAL %s", json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.work, f"eval_{a.name}.json"), "w"), indent=1)
    pd.DataFrame({"s": s, "q": q, "y": y, "p3": p3, "p4": p4}).to_parquet(
        os.path.join(a.work, f"oof_{a.name}.parquet"), index=False)
    with open(os.path.join(a.work, f"stack_{a.name}.pkl"), "wb") as f:
        pickle.dump({"models": models, "features": feats, "decode": cfg_new, "tags": tags,
                     "train_countries": S8["train_countries"]}, f)
    log.info("training + evaluation done in %.1f min", (time.time() - T0) / 60)
    del X

    # ---------------- test
    need = sorted(set(["entity_id", "source", "country", "a_empty", "n_core", "n_legal", "a_hn", "a_hnc", "a_street",
                       "a_nums", "a_unit"] + SK.REC_COLS))
    rte = pd.read_parquet(os.path.join(a.test_work, "test", "records.parquet"), columns=need)
    cte = aligned(os.path.join(a.v8_work, "ce", "ce_test.parquet"), st, qt, "ce")
    Xt = build_X(st, qt, sc["p1"].to_numpy(), sc["p2"].to_numpy(), cte, p3t, SK.record_view(rte), pts_te, p2v9t)
    ti = np.nonzero(tb)[0]
    p4t_b = SK.predict_avg(models, Xt.iloc[ti][feats].reset_index(drop=True))
    del Xt
    ctry = rte["country"].astype(str).to_numpy()
    known = np.isin(ctry[st], S8["train_countries"])
    p = np.where(known, p3t, sc["p2"].to_numpy()).astype(np.float32)
    p[ti] = p4t_b
    sc["p4"] = p
    log.info("test pairs %d: stage-4 score for %d band pairs, v8 stage-3 for %d other pairs of labelled countries, "
             "stage-2 kept for %d (%s)", len(sc), len(ti), int(known.sum()) - len(ti), int((~known).sum()),
             sorted(set(ctry[st[~known]])))
    cc = pd.read_parquet(os.path.join(a.test_work, "test", "cand_ctx.parquet"), columns=["q", "s", "rank"])
    top = cc[cc["rank"].to_numpy() == 0].reset_index(drop=True)
    del cc
    sc = FI.apply(sc, top, rte, known=None, col="p4", log=log)
    sc["p3v8"] = p3t
    sc[["s", "q", "p1", "p2", "p3v8", "p4"]].to_parquet(os.path.join(a.work, f"test_scores_{a.name}.parquet"),
                                                          index=False)
    with open(os.path.join(a.test_work, "artifacts", "models.pkl"), "rb") as f:
        cfg_v6_run = pickle.load(f)["decode"]
    pairs = pd.DataFrame({"s": st, "q": qt, "p": sc["p4"].to_numpy()})
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
    log.info("wrote %d S1 rows (%d with matches, %d matched ids); %d candidate pairs (%.2f per S1)", len(s1_ids),
             int((n_sel > 0).sum()), len(sel), len(sc), len(sc) / len(s1_ids))
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
