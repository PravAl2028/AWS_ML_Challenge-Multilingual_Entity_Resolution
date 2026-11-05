"""v13 final check (CPU, ~5-8 min): is v13's extra recall safe on the test's heavier distractor load?

v13 (``v13_stack.py``) beat v11 on the training OOF (+0.00005, z 5) but with a decoder scale of 1.1 instead of 1.0:
it links ~5.6k more test pairs than v11 (pair precision 0.99936 -> 0.99910, recall 0.97621 -> 0.97709 out of fold).
The test has ~40% unmatched S2/S3 records vs 26% in training (distractor clusters), so extra links are less precise on
the test than out of fold. This script scores every candidate on

* the plain training OOF, and
* a test-like OOF: 90% of the unmatched training queries get a second copy (same scores, label 0), i.e. the test's
  distractor rate (as ``src/testlike.py``, without re-predicting the matchers),

paired per S1 against v11, and ranks them by  OOF + 0.31 x (test-like - OOF)  -- the blend that predicted the public LB
from v2 -> v3 (VERSIONS.md / EXPERIMENTS.md). Candidates: v10, v11 (their own rules) and v13 at decoder scales
0.9 / 1.0 / 1.1. With ``--write`` it re-decodes v13's saved test scores at the chosen scale into --out-dir
(France: the v6 rule, checked identical to --france-ref), copies candidate_pairs.tsv and runs the validator.

    python scripts/v13_decide.py --data-dir <SR>/dataset --train-work ~/er_work_v6 --test-work ~/er_work_v7b \
        --work ~/er_work_v10 --v8-work ~/er_work_v8 [--write --out-dir <SR>/output_v13f --france-ref <SR>/output_v10]
"""
import argparse
import csv
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import v8_stack as V8  # noqa: E402
from src import decode as DC  # noqa: E402
from src.evaluation import Truth, evaluate, decode as ev_decode  # noqa: E402
from src.io_utils import setup_logging, write_id_lists  # noqa: E402

log = logging.getLogger("er")
LAMBDA = 0.31


def metrics(pairs: pd.DataFrame, cfg: dict, truth: Truth):
    assigned = DC.assign(pairs, "p")
    sel = ev_decode(assigned, cfg)
    m, df = evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), truth.n_true_s, truth.s1, truth.ctry_s1,
                     return_f=True)
    return df["f"].to_numpy(), m, len(sel)


def testlike(pairs: pd.DataFrame, matched_q: np.ndarray, frac: float, offset: int) -> pd.DataFrame:
    """Add a copy (new query id, label 0, same score) of ~frac of the unmatched queries' pairs."""
    q = pairs["q"].to_numpy()
    h = (q.astype(np.int64) * 2654435761 + 97) % 1000
    cp = (~matched_q[q]) & (h < int(frac * 1000))
    extra = pairs.loc[cp].copy()
    extra["q"] = extra["q"].to_numpy() + offset
    return pd.concat([pairs, extra], ignore_index=True)


def read_sets(path: str) -> dict:
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    return {a: frozenset(x for x in b.split(",") if x) for a, b in zip(m.iloc[:, 0], m.iloc[:, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--v8-work", required=True)
    ap.add_argument("--name", default="v13")
    ap.add_argument("--frac", type=float, default=0.9)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--scale", type=float, default=0.0, help="force the v13 decoder scale written (0 = best blend)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--france-ref", default="")
    ap.add_argument("--cand-from", default="", help="folder whose candidate_pairs.tsv is copied (default <SR>/output_<name>)")
    a = ap.parse_args()
    setup_logging()
    T0 = time.time()
    truth = Truth(a.train_work, a.data_dir)
    matched_q = np.zeros(truth.n, bool)
    matched_q[truth.gtp["q"].to_numpy()] = True
    offset = int(truth.n) + 1

    def load(name):
        d = pd.read_parquet(os.path.join(a.work, f"oof_{name}.parquet"), columns=["s", "q", "y", "p4"])
        with open(os.path.join(a.work, f"stack_{name}.pkl"), "rb") as f:
            cfg = pickle.load(f)["decode"]
        return d.rename(columns={"p4": "p"}), {k: v for k, v in cfg.items() if k != "oof"}

    cands = []
    for nm in ("v10", "v11"):
        if os.path.exists(os.path.join(a.work, f"oof_{nm}.parquet")):
            d, cfg = load(nm)
            cands.append((f"{nm}@{cfg.get('scale')}", nm, d, cfg))
    d13, cfg13 = load(a.name)
    for sc in (0.9, 1.0, 1.1):
        cands.append((f"{a.name}@{sc}", a.name, d13, dict(cfg13, scale=sc)))
    res, F = {}, {}
    tl_cache = {}
    for label, nm, d, cfg in cands:
        f_o, m_o, n_o = metrics(d, cfg, truth)
        if nm not in tl_cache:
            tl_cache[nm] = testlike(d, matched_q, a.frac, offset)
        f_t, m_t, n_t = metrics(tl_cache[nm], cfg, truth)
        F[label] = (f_o, f_t)
        res[label] = {"oof": m_o, "testlike": m_t, "links_oof": n_o, "links_tl": n_t}
        log.info("%-10s OOF f05 %.5f (prec %.5f rec %.5f single %.5f) | test-like f05 %.5f (prec %.5f rec %.5f single "
                 "%.5f)", label, m_o["f05"], m_o["pair_prec"], m_o["pair_rec"], m_o["singleton_acc"], m_t["f05"],
                 m_t["pair_prec"], m_t["pair_rec"], m_t["singleton_acc"])
    ref = next(k for k in F if k.startswith("v11@"))
    fo_r, ft_r = F[ref]
    print(f"\n=== paired per S1 vs {ref} (plain OOF / test-like, 90% of unmatched queries copied); blend = OOF + "
          f"{LAMBDA} x (test-like - OOF) ===")
    best, best_b = None, -1e9
    for label, (fo, ft) in F.items():
        do, dt = fo - fo_r, ft - ft_r
        b = do + LAMBDA * (dt - do)
        se = float(np.sqrt(((1 - LAMBDA) * do + LAMBDA * dt).var(ddof=1) / len(do)))
        res[label]["vs_ref"] = {"oof": float(do.mean()), "testlike": float(dt.mean()), "blend": float(b.mean()),
                                "blend_se": se}
        by_c = " ".join(f"{c} {float(b[truth.ctry_s1 == c].mean()):+.5f}" for c in pd.unique(truth.ctry_s1))
        print(f"  {label:10s} OOF {do.mean():+.5f}  test-like {dt.mean():+.5f}  blend {b.mean():+.5f} (se {se:.5f}; "
              f"{by_c})  links OOF {res[label]['links_oof']:,} / test-like {res[label]['links_tl']:,}")
        if label.startswith(a.name) and b.mean() > best_b:
            best, best_b = label, float(b.mean())
    se_b = res[best]["vs_ref"]["blend_se"]
    keep = best_b > 2 * se_b
    print(f"\nBEST {a.name} rule: {best}  blend vs {ref} {best_b:+.5f} (se {se_b:.5f}) -> "
          + ("USE IT" if keep else f"not significant: submit {ref.split('@')[0]}"))
    json.dump(res, open(os.path.join(a.work, f"decide_{a.name}.json"), "w"), indent=1, default=float)
    if not a.write:
        return
    scale = a.scale or float(best.split("@")[1])
    # ---------------- re-decode v13's saved test scores at the chosen scale (France: the v6 rule)
    rte = pd.read_parquet(os.path.join(a.test_work, "test", "records.parquet"), columns=["entity_id", "source",
                                                                                      "country"])
    ts = pd.read_parquet(os.path.join(a.work, f"test_scores_{a.name}.parquet"))
    with open(os.path.join(a.v8_work, "stack.pkl"), "rb") as f:
        train_countries = pickle.load(f)["train_countries"]
    with open(os.path.join(a.test_work, "artifacts", "models.pkl"), "rb") as f:
        cfg_v6_run = pickle.load(f)["decode"]
    ctry = rte["country"].astype(str).to_numpy()
    known = np.isin(ctry[ts["s"].to_numpy()], train_countries)
    parts = []
    for flag, cfg in ((True, dict(cfg13, scale=scale)), (False, cfg_v6_run)):
        sub = ts[known == flag]
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
    os.makedirs(a.out_dir, exist_ok=True)
    write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), ("source1_entity_id", "matched_entity_ids"),
                   s1_ids, matches)
    cand_from = a.cand_from or os.path.join(os.path.dirname(os.path.abspath(a.data_dir)), f"output_{a.name}")
    shutil.copyfile(os.path.join(cand_from, "candidate_pairs.tsv"), os.path.join(a.out_dir, "candidate_pairs.tsv"))
    c1 = ctry[src == 1]
    n_sel = np.bincount(sel["s"].to_numpy(), minlength=len(rte))[np.nonzero(src == 1)[0]]
    log.info("wrote %d S1 rows (%d with matches, %d matched ids) at labelled scale %.2f", len(s1_ids),
             int((n_sel > 0).sum()), len(sel), scale)
    if a.france_ref:
        new = read_sets(os.path.join(a.out_dir, "matching_results.tsv"))
        old = read_sets(os.path.join(a.france_ref, "matching_results.tsv"))
        unk = [x for x, c in zip(s1_ids, c1) if c not in train_countries]
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
