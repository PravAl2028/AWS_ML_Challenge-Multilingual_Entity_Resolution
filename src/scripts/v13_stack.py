"""v13 (CPU, ~15-20 min): the v11 stage-4 stack plus the query-level competition of the STRONGEST scores, seed
bagging, an optional stage-5 and an optional query-normalised decode -- each variant scored on the training OOF,
paired per S1 against the submitted-quality reference (v11 OOF p4, its own decoding rule); the best variant with a
significant gain writes the test output. Countries without training labels (France) are untouched: p2 + the v7
filler rule + the v6 decoding rule, exactly as v10 / v11 (checked row by row against --france-ref).

    python scripts/v13_stack.py --data-dir <SR>/dataset --train-work ~/er_work_v6 --test-work ~/er_work_v7b \
        --v8-work ~/er_work_v8 --v9-work ~/er_work_v9 --work ~/er_work_v10 --ref v11 \
        --out-dir <SR>/output_v13 --france-ref <SR>/output_v10 --n-jobs 63

Why these features: the v10 / v11 stage 4 sees the query-level competition of p2, of the char cross-encoder and of the
pretrained-model mean (``p2_alt_q``, ``ce_alt_q``, ``pt_alt_q``), but NOT of its strongest input, the v8 stage-3 score
p3 (importance 0.80), nor of the v9 stage-2 score. Every S2/S3 record belongs to at most one S1, so a pair whose query
has a second strong S1 candidate should be scored lower; the decoder's exclusive assignment only keeps the best S1
and does not lower its probability. Stage 5 repeats the same view on p4 (one more round of competition), and the
query-normalised decode divides a pair's score by the query's total score mass when it exceeds 1.
S1-level views are still left out (the test's distractor clusters inflate them; see src/stack.py).
Writes only to --work (files named <name>) and --out-dir.
"""
import argparse
import csv
import json
import logging
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import v8_stack as V8  # noqa: E402
import v10_stack as V10  # noqa: E402
from src import decode as DC  # noqa: E402
from src import fillers as FI  # noqa: E402
from src import stack as SK  # noqa: E402
from src.evaluation import Truth  # noqa: E402
from src.features import _grp_max_other, _grp_rank  # noqa: E402
from src.io_utils import setup_logging, write_id_lists  # noqa: E402
from src.prepare import fold_of  # noqa: E402

log = logging.getLogger("er")

GRID = [(0.0, 0.9), (0.0, 1.0), (0.0, 1.1)]
SEEDS = [dict(seed=11, bagging_seed=111, feature_fraction_seed=211),
         dict(seed=12, bagging_seed=112, feature_fraction_seed=212, feature_fraction=0.8),
         dict(seed=13, bagging_seed=113, feature_fraction_seed=213, num_leaves=63, min_data_in_leaf=300)]
S5_REL = ["q_empty", "hn_rel", "hnc_eq", "legal_rel", "core_eq", "street_eq", "q_src"]


def comp(q: np.ndarray, p: np.ndarray, name: str, full: bool = True) -> dict:
    """Query-level competition of score ``p``: margin over the query's best other S1, rank, and the other S1s' mass."""
    p = np.asarray(p, dtype=np.float32)
    out = {f"{name}_alt_q": (p - _grp_max_other(q, p)).astype(np.float32)}
    if full:
        tot = pd.Series(p).groupby(q).transform("sum").to_numpy().astype(np.float32)
        out[f"{name}_rank_q"] = _grp_rank(q, p).astype(np.int16)
        out[f"{name}_sumo_q"] = (tot - p).astype(np.float32)
    return out


def qnorm(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Divide by the query's total score mass when it exceeds 1 (a query belongs to at most one S1)."""
    p = np.asarray(p, dtype=np.float64)
    tot = pd.Series(p).groupby(q).transform("sum").to_numpy()
    return (p / np.maximum(tot, 1.0)).astype(np.float32)


def add_comp(X: pd.DataFrame, q, p3, p2v9):
    for k, v in comp(q, p3, "p3").items():
        X[k] = v
    if p2v9 is not None:
        for k, v in comp(q, p2v9, "p2v9", full=False).items():
            X[k] = v
    return X


def s5_frame(q, p4, X: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    c = comp(q, p4, "p4")
    f = pd.DataFrame({"lp4": V10.logit(p4[rows])})
    for k, v in c.items():
        f[k] = v[rows]
    for k in ["lp3", "p3_alt_q", "lptm"] + S5_REL:
        f[k] = X[k].to_numpy()[rows] if len(X) != len(rows) else X[k].to_numpy()
    return f


def paired(f_new, f_ref, truth) -> dict:
    d = f_new - f_ref
    out = {"delta": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
           "better": int((d > 1e-12).sum()), "worse": int((d < -1e-12).sum())}
    for c in pd.unique(truth.ctry_s1):
        m = truth.ctry_s1 == c
        out[f"delta_{c}"] = float(d[m].mean())
    return out


def read_sets(path: str) -> dict:
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    return {a: frozenset(x for x in b.split(",") if x) for a, b in zip(m.iloc[:, 0], m.iloc[:, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--v8-work", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pt-dir", default="")
    ap.add_argument("--tags", default="", help="pretrained models (default: every complete one)")
    ap.add_argument("--v9-work", default="")
    ap.add_argument("--ref", default="v11", help="reference stack run in --work (oof_<ref>.parquet, stack_<ref>.pkl)")
    ap.add_argument("--france-ref", default="", help="output folder whose unlabelled-country rows must be reproduced")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-stage5", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--neg-weight", type=float, default=0.9)
    ap.add_argument("--min-z", type=float, default=2.0, help="a variant must beat the reference by > min_z s.e.")
    ap.add_argument("--variant", default="", help="force a variant (s4, s5, s4_qn, s5_qn) instead of the best")
    ap.add_argument("--name", default="v13")
    a = ap.parse_args()
    pt_dir = a.pt_dir or os.path.join(a.work, "pt")
    setup_logging()
    fh = logging.FileHandler(os.path.join(a.work, f"stack_{a.name}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    T0 = time.time()
    tags = V10.find_tags(pt_dir, a.tags)
    log.info("pretrained models: %s", tags)

    # ------------------------------------------------ training table (as v10_stack) + new competition features
    truth = Truth(a.train_work, a.data_dir)
    rtr = pd.read_parquet(os.path.join(a.train_work, "train", "records.parquet"),
                          columns=["entity_id", "country"] + SK.REC_COLS)
    oof = pd.read_parquet(os.path.join(a.train_work, "train", "oof.parquet"), columns=["s", "q", "y", "p1", "p2"])
    s, q, y = oof["s"].to_numpy(), oof["q"].to_numpy(), oof["y"].to_numpy()
    ce = V10.aligned(os.path.join(a.v8_work, "ce", "ce_train.parquet"), s, q, "ce")
    p3 = V10.aligned(os.path.join(a.v8_work, "oof_v8.parquet"), s, q, "p3").astype(np.float32)
    tp = pd.read_parquet(os.path.join(a.work, "pack", "train_pairs.parquet"), columns=["row", "q", "s"])
    band = np.zeros(len(oof), bool)
    band[tp["row"].to_numpy()] = True
    sc = pd.read_parquet(os.path.join(a.test_work, "test", "scores.parquet"), columns=["s", "q", "p1", "p2"])
    pts_tr, pts_te = {}, {}
    for t in tags:
        pts_tr[t], pts_te[t], _ = V10.load_pt(pt_dir, t, len(oof), len(sc))
    p2v9 = p2v9t = None
    if a.v9_work:
        p2v9 = V10.aligned(os.path.join(a.v9_work, "train", "oof.parquet"), s, q, "p2")
        p2v9t = V10.aligned(os.path.join(a.v9_work, "test", "scores.parquet"), sc["s"].to_numpy(),
                            sc["q"].to_numpy(), "p2")
    X = V10.build_X(s, q, oof["p1"].to_numpy(), oof["p2"].to_numpy(), ce, p3, SK.record_view(rtr), pts_tr, p2v9)
    X = add_comp(X, q, p3, p2v9)
    feats = list(X.columns)
    fold = fold_of(rtr["entity_id"].to_numpy(dtype=object), 5)[s]
    matched = np.zeros(len(rtr), bool)
    matched[truth.gtp["q"].to_numpy()] = True
    w = np.where((y == 0) & ~matched[q], 1.0 + a.neg_weight, 1.0).astype(np.float32)
    bi = np.nonzero(band)[0]
    Xb = X.iloc[bi].reset_index(drop=True)
    log.info("stage-4 training: %d band pairs, %d features: %s", len(bi), len(feats), feats)
    bags, p4b = [], np.zeros(len(bi), dtype=np.float64)
    for k, prm in enumerate(SEEDS[:max(1, a.seeds)]):
        pb, models = SK.train_oof(Xb, y[bi], fold[bi], s[bi], w[bi], a.n_jobs, params=prm)
        bags.append(models)
        p4b += pb
        log.info("bag %d done (%.1f min)", k, (time.time() - T0) / 60)
    p4b /= len(bags)
    log.info("stage-4 importance (bag 0):\n%s", SK.importance(bags[0]).round(4).head(30).to_string())
    p4 = p3.copy()
    p4[bi] = p4b.astype(np.float32)
    models5, p5 = None, None
    if not a.no_stage5:
        X5 = s5_frame(q, p4, X, bi)
        p5b, models5 = SK.train_oof(X5, y[bi], fold[bi], s[bi], w[bi], a.n_jobs)
        log.info("stage-5 importance:\n%s", SK.importance(models5).round(4).to_string())
        p5 = p4.copy()
        p5[bi] = p5b
        del X5
    del Xb

    # ------------------------------------------------ evaluation, paired per S1 vs the reference and vs v8
    with open(os.path.join(a.v8_work, "stack.pkl"), "rb") as f:
        S8 = pickle.load(f)
    base = pd.DataFrame({"s": s, "q": q, "p": p3, "y": y})
    f_v8, m_v8 = V8.per_s1_f(base, S8["decode"], truth, with_metrics=True)
    ref = pd.read_parquet(os.path.join(a.work, f"oof_{a.ref}.parquet"), columns=["s", "q", "p4"])
    if not (np.array_equal(ref["s"].to_numpy(), s) and np.array_equal(ref["q"].to_numpy(), q)):
        ref = pd.DataFrame({"s": s, "q": q}).merge(ref, on=["s", "q"], how="left")
    p4_ref = ref["p4"].to_numpy()
    with open(os.path.join(a.work, f"stack_{a.ref}.pkl"), "rb") as f:
        cfg_ref = pickle.load(f)["decode"]
    f_ref, m_ref = V8.per_s1_f(base.assign(p=p4_ref), cfg_ref, truth, with_metrics=True)
    log.info("reference %s: %s", a.ref, json.dumps(m_ref))
    variants = {"s4": p4}
    if p5 is not None:
        variants["s5"] = p5
    for nm in list(variants):
        variants[nm + "_qn"] = qnorm(q, variants[nm])
    res = {"reference": a.ref, "oof_ref": m_ref, "oof_v8": m_v8, "variants": {}}
    yb = y[bi]
    ll = lambda p: float(-(yb * np.log(np.clip(p, 1e-6, 1)) + (1 - yb) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
    res["band_logloss"] = {"p3_v8": ll(p3[bi]), f"p4_{a.ref}": ll(p4_ref[bi])}
    fs = {}
    for nm, pv in variants.items():
        log.info("variant %s:", nm)
        cfg = V8.tune_grid(base.assign(p=pv), truth, grid=GRID)
        f_new = V8.per_s1_f(base.assign(p=pv), cfg, truth)
        fs[nm] = f_new
        res["band_logloss"][nm] = ll(pv[bi])
        res["variants"][nm] = {"decode": {k: v for k, v in cfg.items() if k != "oof"}, "oof": cfg["oof"],
                               "vs_ref": paired(f_new, f_ref, truth), "vs_v8": paired(f_new, f_v8, truth)}
        log.info("  %s vs %s: %s", nm, a.ref, json.dumps(res["variants"][nm]["vs_ref"]))
    best = a.variant or max(res["variants"], key=lambda k: res["variants"][k]["vs_ref"]["delta"])
    vb = res["variants"][best]
    ok = vb["vs_ref"]["delta"] > a.min_z * vb["vs_ref"]["se"]
    res["best"], res["significant"] = best, bool(ok)
    log.info("BEST variant %s: %+.5f vs %s (se %.5f) -> %s", best, vb["vs_ref"]["delta"], a.ref, vb["vs_ref"]["se"],
             "BUILD" if ok else "NOT significant: keep the reference")
    # v11_report-compatible keys (vs v8), for the LB projection
    res.update({"oof_v10": vb["oof"], "decode_v10": vb["decode"], "paired_delta_vs_v8": vb["vs_v8"]["delta"],
                "paired_se": vb["vs_v8"]["se"], "s1_better": vb["vs_v8"]["better"], "s1_worse": vb["vs_v8"]["worse"]})
    for c in pd.unique(truth.ctry_s1):
        res[f"paired_delta_{c}"] = vb["vs_v8"][f"delta_{c}"]
    json.dump(res, open(os.path.join(a.work, f"eval_{a.name}.json"), "w"), indent=1)
    pd.DataFrame({"s": s, "q": q, "y": y, "p3": p3, "p4": variants[best]}).to_parquet(
        os.path.join(a.work, f"oof_{a.name}.parquet"), index=False)
    with open(os.path.join(a.work, f"stack_{a.name}.pkl"), "wb") as f:
        pickle.dump({"bags": bags, "models5": models5, "features": feats, "decode": vb["decode"], "best": best,
                     "tags": tags, "train_countries": S8["train_countries"]}, f)
    log.info("training + evaluation done in %.1f min", (time.time() - T0) / 60)
    del X, oof, base
    if not ok and not os.environ.get("V13_FORCE"):
        log.info("DONE (no output written) in %.1f min", (time.time() - T0) / 60)
        return

    # ------------------------------------------------ test
    need = sorted(set(["entity_id", "source", "country", "a_empty", "n_core", "n_legal", "a_hn", "a_hnc", "a_street",
                       "a_nums", "a_unit"] + SK.REC_COLS))
    rte = pd.read_parquet(os.path.join(a.test_work, "test", "records.parquet"), columns=need)
    st, qt = sc["s"].to_numpy(), sc["q"].to_numpy()
    te_pairs = pd.read_parquet(os.path.join(a.work, "pack", "test_pairs.parquet"), columns=["row", "q", "s"])
    tb = np.zeros(len(sc), bool)
    tb[te_pairs["row"].to_numpy()] = True
    ti = np.nonzero(tb)[0]
    p3t = V10.aligned(os.path.join(a.work, "test_p3.parquet"), st, qt, "p3").astype(np.float32)
    cte = V10.aligned(os.path.join(a.v8_work, "ce", "ce_test.parquet"), st, qt, "ce")
    Xt = V10.build_X(st, qt, sc["p1"].to_numpy(), sc["p2"].to_numpy(), cte, p3t, SK.record_view(rte), pts_te, p2v9t)
    Xt = add_comp(Xt, qt, p3t, p2v9t)
    Xtb = Xt.iloc[ti][feats].reset_index(drop=True)
    p4t_b = np.mean([SK.predict_avg(m, Xtb) for m in bags], axis=0)
    ctry = rte["country"].astype(str).to_numpy()
    known = np.isin(ctry[st], S8["train_countries"])
    p = np.where(known, p3t, sc["p2"].to_numpy()).astype(np.float32)
    p[ti] = p4t_b
    if best.startswith("s5"):
        X5t = s5_frame(qt, p, Xt, ti)
        p[ti] = SK.predict_avg(models5, X5t)
    if best.endswith("_qn"):
        pk = qnorm(qt[known], p[known])
        p[known] = pk
    log.info("test pairs %d: %s score for %d band pairs, v8 stage-3 for %d other labelled pairs, stage-2 kept for %d (%s)",
             len(sc), best, len(ti), int(known.sum()) - len(ti), int((~known).sum()), sorted(set(ctry[st[~known]])))
    del Xt, Xtb
    cc = pd.read_parquet(os.path.join(a.test_work, "test", "cand_ctx.parquet"), columns=["q", "s", "rank"])
    top = cc[cc["rank"].to_numpy() == 0].reset_index(drop=True)
    del cc
    pairs = FI.apply(pd.DataFrame({"s": st, "q": qt, "p": p}), top, rte, known=None, col="p", log=log)
    pd.DataFrame({"s": st, "q": qt, "p": pairs["p"].to_numpy()}).to_parquet(
        os.path.join(a.work, f"test_scores_{a.name}.parquet"), index=False)
    with open(os.path.join(a.test_work, "artifacts", "models.pkl"), "rb") as f:
        cfg_v6_run = pickle.load(f)["decode"]
    parts = []
    for flag, cfg in ((True, vb["decode"]), (False, cfg_v6_run)):
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
    if a.france_ref:
        new = read_sets(os.path.join(a.out_dir, "matching_results.tsv"))
        old = read_sets(os.path.join(a.france_ref, "matching_results.tsv"))
        unk = [x for x, c in zip(s1_ids, c1) if c not in S8["train_countries"]]
        same = sum(new.get(x, frozenset()) == old.get(x, frozenset()) for x in unk)
        log.info("FRANCE CHECK: %d / %d unlabelled-country rows identical to %s%s", same, len(unk), a.france_ref,
                 "" if same == len(unk) else "  <-- NOT IDENTICAL")
    v = os.path.abspath(os.path.join(a.data_dir, "..", "utils", "validate_submission.py"))
    if os.path.exists(v):
        subprocess.run([sys.executable, v, "--matching", os.path.join(a.out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(a.out_dir, "candidate_pairs.tsv"), "--test-dir",
                        os.path.join(a.data_dir, "test"), "--check-ids"], check=False)
    log.info("DONE in %.1f min", (time.time() - T0) / 60)


if __name__ == "__main__":
    main()
