"""v14 (CPU, ~10-15 min): a CE-free, competition-aware decode for countries WITHOUT training labels (France), validated
on the labelled countries before anything is applied. US / India rows are copied unchanged from --base (v13 @ 1.0).

Why: France is decided by the v6 stage-2 score p2 alone (+ the v7 filler rule, decoder scale 1.1). p2 is scored pair
by pair; a record belongs to at most one S1, yet a French record with two plausible S1s (18% of French S1 share an
address, 34% share a name) can keep p2 = 0.8 on its best S1 next to 0.5 on another, and the decoder links it as if
0.8 were its probability. US / India get this competition from the stage-3/4 stacks; France never did. Two label-free
transforms of p2 use only p2 itself and the record's own candidate list (no text model: the v12a cross-encoder
failed exactly because it had never read French):

* T1 ``qn@scale``: p / max(1, the record's total p over its S1 candidates), then the expected-F0.5 decoder at scale s;
* T2 ``lr``: a 3-weight logistic recalibration of the uncertain pairs (p2 in [0.002, 0.998]) on logit p2, its margin over
  the record's best other S1 and logit(qn p2); fitted on US+India out-of-fold p2 and checked leave-one-country-out.

Evidence (training OOF p2 of the labelled countries, the France decoding rule, paired per S1, plain and test-like
with 90% of the unmatched queries duplicated): a transform is eligible only if it beats p2 by > 2 s.e. in EACH
labelled country on both views (T2: on the held-out country when fitted on the other). The best eligible one is
applied to the unlabelled countries; if none is eligible, nothing is written.

    python scripts/v14_france.py --data-dir <SR>/dataset --train-work ~/er_work_v6 --test-work ~/er_work_v7b \
        --v8-work ~/er_work_v8 --base <SR>/output_v13s10 --out-dir <SR>/output_v14 [--write]
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
from src import decode as DC  # noqa: E402
from src import fillers as FI  # noqa: E402
from src.evaluation import Truth, evaluate, decode as ev_decode  # noqa: E402
from src.features import _grp_max_other  # noqa: E402
from src.io_utils import setup_logging, write_id_lists  # noqa: E402

log = logging.getLogger("er")
LAMBDA = 0.31
BAND = 6.212606095751517  # |logit p| of the v10 band ([0.002, 0.998])
SCALES = (1.0, 1.1, 1.2, 1.3)


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def qnorm(q, p):
    p = np.asarray(p, dtype=np.float64)
    tot = pd.Series(p).groupby(q).transform("sum").to_numpy()
    return p / np.maximum(tot, 1.0)


def lr_feats(q, p):
    p = np.asarray(p, dtype=np.float64)
    other = _grp_max_other(q, p.astype(np.float32)).astype(np.float64)
    other = np.where(other < 0, 0.0, other)
    return np.column_stack([logit(p), p - other, logit(qnorm(q, p))])


def metrics(pairs, cfg, truth):
    assigned = DC.assign(pairs, "p")
    sel = ev_decode(assigned, cfg)
    m, df = evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), truth.n_true_s, truth.s1, truth.ctry_s1,
                     return_f=True)
    m["links"] = int(len(sel))
    return df["f"].to_numpy(), m


def testlike(pairs, matched_q, frac=0.9):
    q = pairs["q"].to_numpy()
    h = (q.astype(np.int64) * 2654435761 + 97) % 1000
    cp = (~matched_q[q]) & (h < int(frac * 1000))
    extra = pairs.loc[cp].copy()
    extra["q"] = extra["q"].to_numpy() + len(matched_q) + 1
    return pd.concat([pairs, extra], ignore_index=True)


def canon(v: str) -> frozenset:
    return frozenset(x for x in v.split(",") if x)


def read_rows(path):
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    return dict(zip(m.iloc[:, 0], m.iloc[:, 1]))


def decode_rows(sub, cfg, ids):
    assigned = DC.assign(sub, "p")
    if cfg["method"] == "threshold":
        sel = DC.select_threshold(assigned, cfg["t"])
    else:
        sel = DC.select_expected_f05(assigned, "p", miss_p=cfg["miss_p"], prob_scale=cfg["scale"])
    out = {}
    for x, z in zip(ids[sel["s"].to_numpy()], ids[sel["q"].to_numpy()]):
        out.setdefault(x, []).append(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--v8-work", required=True)
    ap.add_argument("--base", required=True, help="output folder whose labelled-country rows are kept (v13 @ 1.0)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", default="", help="apply this transform even if not eligible (e.g. qn@1.1, lr)")
    a = ap.parse_args()
    setup_logging()
    T0 = time.time()
    truth = Truth(a.train_work, a.data_dir)
    matched_q = np.zeros(truth.n, bool)
    matched_q[truth.gtp["q"].to_numpy()] = True
    oof = pd.read_parquet(os.path.join(a.train_work, "train", "oof.parquet"), columns=["s", "q", "y", "p2"])
    s, q, y, p2 = (oof[c].to_numpy() for c in ("s", "q", "y", "p2"))
    p2 = p2.astype(np.float64)
    with open(os.path.join(a.train_work, "artifacts", "models.pkl"), "rb") as fh:
        cfg6 = {k: v for k, v in pickle.load(fh)["decode"].items() if k != "oof"}
    log.info("France decoding rule (v6, in-country tuned): %s", cfg6)
    ctry_pair = truth.ctry[s]
    countries = sorted(set(truth.ctry_s1))
    base = pd.DataFrame({"s": s, "q": q, "y": y, "p": p2})
    tl_base = testlike(base, matched_q)
    f0, m0 = metrics(base, cfg6, truth)
    t0, mt0 = metrics(tl_base, cfg6, truth)
    log.info("p2 (France rule) OOF f05 %.5f prec %.5f rec %.5f single %.5f | test-like %.5f", m0["f05"],
             m0["pair_prec"], m0["pair_rec"], m0["singleton_acc"], mt0["f05"])
    res, cands = {"p2": {"oof": m0, "testlike": mt0}}, {}

    def score(label, pnew, cfg):
        f, m = metrics(base.assign(p=pnew), cfg, truth)
        ft, mt = metrics(testlike(base.assign(p=pnew), matched_q), cfg, truth)
        do, dt = f - f0, ft - t0
        r = {"oof": m, "testlike": mt, "cfg": cfg}
        ok = True
        for c in countries:
            k = truth.ctry_s1 == c
            for nm, d in (("oof", do), ("tl", dt)):
                mu, se = float(d[k].mean()), float(d[k].std(ddof=1) / np.sqrt(k.sum()))
                r[f"{nm}_{c}"] = [mu, se]
                ok &= mu > 2 * se
        b = do + LAMBDA * (dt - do)
        r["blend"] = float(b.mean())
        r["eligible"] = bool(ok)
        log.info("%-10s blend %+.5f | %s | prec %.5f rec %.5f single %.5f | %s", label, r["blend"],
                 " ".join(f"{c} oof {r['oof_' + c][0]:+.5f} tl {r['tl_' + c][0]:+.5f}" for c in countries),
                 m["pair_prec"], m["pair_rec"], m["singleton_acc"], "ELIGIBLE" if ok else "not eligible")
        res[label] = r
        return r

    # ---- T1: query-normalised p2 at several decoder scales
    pq = qnorm(q, p2)
    for sc in SCALES:
        cands[f"qn@{sc}"] = ("qn", sc)
        score(f"qn@{sc}", pq, dict(cfg6, scale=sc))
    # ---- T2: logistic recalibration of the uncertain pairs, leave-one-country-out
    from sklearn.linear_model import LogisticRegression
    band = np.abs(logit(p2, 1e-7)) <= BAND
    X = lr_feats(q, p2)
    w = np.where((y == 0) & ~matched_q[q], 1.9, 1.0)

    def fit(m):
        lr = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(X[m], y[m], sample_weight=w[m])
        return lr

    ploco = p2.copy()
    for c_ev in countries:
        lr = fit(band & (ctry_pair != c_ev))
        m = band & (ctry_pair == c_ev)
        ploco[m] = lr.predict_proba(X[m])[:, 1]
        log.info("LOCO lr fitted without %s: coef %s", c_ev, np.round(np.r_[lr.coef_[0], lr.intercept_], 3).tolist())
    score("lr(LOCO)", ploco, cfg6)
    lr_all = fit(band)
    res["lr_coef"] = np.r_[lr_all.coef_[0], lr_all.intercept_].tolist()
    cands["lr(LOCO)"] = ("lr", cfg6["scale"])
    elig = [k for k in cands if res[k]["eligible"]]
    best = a.force or (max(elig, key=lambda k: res[k]["blend"]) if elig else "")
    if a.force and a.force not in cands:
        raise SystemExit(f"--force must be one of {list(cands)}")
    print("\n=== France decode candidates, paired per S1 vs p2 on the labelled countries (France rule) ===")
    for k in cands:
        r = res[k]
        print(f"  {k:9s} blend {r['blend']:+.5f}  " + "  ".join(
            f"{c}: oof {r['oof_' + c][0]:+.5f} (se {r['oof_' + c][1]:.5f}) tl {r['tl_' + c][0]:+.5f}" for c in countries)
            + f"  prec {r['oof']['pair_prec']:.5f} rec {r['oof']['pair_rec']:.5f} single {r['oof']['singleton_acc']:.5f}"
            + ("  ELIGIBLE" if r["eligible"] else ""))
    print(f"CHOICE: {best or 'none eligible -> keep the base France rows'}")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir or ".", "v14_france_eval.json"), "w"), indent=1, default=float)
    if not a.write or not best:
        log.info("DONE (nothing written) in %.1f min", (time.time() - T0) / 60)
        return
    kind, sc = cands[best]

    # ---------------------------------------------------------------- test: unlabelled countries only
    need = ["entity_id", "source", "country", "a_empty", "n_core", "n_legal", "a_hn", "a_hnc", "a_street", "a_nums",
            "a_unit"]
    rte = pd.read_parquet(os.path.join(a.test_work, "test", "records.parquet"), columns=need)
    scs = pd.read_parquet(os.path.join(a.test_work, "test", "scores.parquet"), columns=["s", "q", "p2"])
    with open(os.path.join(a.v8_work, "stack.pkl"), "rb") as fh:
        train_countries = pickle.load(fh)["train_countries"]
    with open(os.path.join(a.test_work, "artifacts", "models.pkl"), "rb") as fh:
        cfg_run = {k: v for k, v in pickle.load(fh)["decode"].items() if k != "oof"}
    ids = rte["entity_id"].to_numpy(dtype=object)
    ctry = rte["country"].astype(str).to_numpy()
    src = rte["source"].to_numpy()
    unk = ~np.isin(ctry[scs["s"].to_numpy()], train_countries)
    ui = np.nonzero(unk)[0]
    sub = pd.DataFrame({"s": scs["s"].to_numpy()[ui], "q": scs["q"].to_numpy()[ui],
                        "p": scs["p2"].to_numpy()[ui].astype(np.float64)})
    cc = pd.read_parquet(os.path.join(a.test_work, "test", "cand_ctx.parquet"), columns=["q", "s", "rank"])
    top = cc[cc["rank"].to_numpy() == 0].reset_index(drop=True)
    del cc
    base_rows = read_rows(os.path.join(a.base, "matching_results.tsv"))
    s1_ids = ids[src == 1]
    c1 = ctry[src == 1]
    unk_s1 = [x for x, c in zip(s1_ids, c1) if c not in train_countries]
    chk = decode_rows(FI.apply(sub, top, rte, known=None, col="p"), cfg_run, ids)
    same = sum(canon(",".join(chk.get(x, []))) == canon(base_rows.get(x, "")) for x in unk_s1)
    log.info("identity check: %d / %d unlabelled-country rows reproduced from p2 (must be all)", same, len(unk_s1))
    if same != len(unk_s1):
        raise SystemExit("identity check failed: the base's unlabelled-country rows are not p2 + filler + rule")
    sq = sub["q"].to_numpy()
    if kind == "qn":
        pn = qnorm(sq, sub["p"].to_numpy())
        cfg_new = dict(cfg_run, scale=sc)
    else:
        pn = sub["p"].to_numpy().copy()
        m = np.abs(logit(pn, 1e-7)) <= BAND
        pn[m] = lr_all.predict_proba(lr_feats(sq, pn)[m])[:, 1]
        cfg_new = cfg_run
    nfr = int((pn != sub["p"].to_numpy()).sum())
    log.info("%s on %d unlabelled pairs (%d changed score); decoder %s", best, len(sub), nfr, cfg_new)
    new_rows = decode_rows(FI.apply(sub.assign(p=pn), top, rte, known=None, col="p"), cfg_new, ids)
    lists = {x: [z for z in base_rows.get(x, "").split(",") if z] for x in s1_ids}
    diff = {"rows_differ": 0, "only_base": 0, "only_new": 0, "empty_to_linked": 0, "linked_to_empty": 0}
    for x in unk_s1:
        old, new = canon(base_rows.get(x, "")), frozenset(new_rows.get(x, []))
        lists[x] = list(new_rows.get(x, []))
        if old != new:
            diff["rows_differ"] += 1
            diff["only_base"] += len(old - new)
            diff["only_new"] += len(new - old)
            diff["empty_to_linked"] += int(not old and bool(new))
            diff["linked_to_empty"] += int(bool(old) and not new)
    lb = float(np.mean([len(canon(base_rows.get(x, ""))) for x in unk_s1]))
    ln = float(np.mean([len(lists[x]) for x in unk_s1]))
    rel_fr = (ln - lb) / lb
    rel_in = (res[best]["oof"]["links"] - m0["links"]) / m0["links"]
    log.info("France vs base: %s; links per S1 %.4f -> %.4f (%+.2f%%); the same transform in-country: %+.2f%% links%s",
             diff, lb, ln, 100 * rel_fr, 100 * rel_in,
             "" if abs(rel_fr) <= 2 * abs(rel_in) + 0.002 else "  <-- WARNING: France reacts far more than the labelled "
             "countries (its p2 behaves differently): do not submit")
    os.makedirs(a.out_dir, exist_ok=True)
    write_id_lists(os.path.join(a.out_dir, "matching_results.tsv"), ("source1_entity_id", "matched_entity_ids"),
                   s1_ids, lists)
    shutil.copyfile(os.path.join(a.base, "candidate_pairs.tsv"), os.path.join(a.out_dir, "candidate_pairs.tsv"))
    chk_l = read_rows(os.path.join(a.out_dir, "matching_results.tsv"))
    unk_set = set(unk_s1)
    ident = all(canon(chk_l.get(x, "")) == canon(base_rows.get(x, "")) for x in s1_ids if x not in unk_set)
    log.info("labelled-country rows identical to the base: %s", ident)
    v = os.path.abspath(os.path.join(a.data_dir, "..", "utils", "validate_submission.py"))
    if os.path.exists(v):
        subprocess.run([sys.executable, v, "--matching", os.path.join(a.out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(a.out_dir, "candidate_pairs.tsv"), "--test-dir",
                        os.path.join(a.data_dir, "test"), "--check-ids"], check=False)
    log.info("DONE in %.1f min", (time.time() - T0) / 60)


if __name__ == "__main__":
    main()
