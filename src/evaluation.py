"""Evaluation of a cached training run: the single source of truth for experiment metrics.

    python -m src.evaluation --data-dir <dataset> --work-dir <work> --tag <name> [--compare <other_tag>]

Reads ``<work>/train/{records,cand_ctx,oof}.parquet`` and ``<work>/artifacts/decode.json`` and reports
overall / per-country macro F0.5, pair precision / recall, singleton accuracy, blocking and prefilter
recall, and the stage-1-only reference score. Every score comes with the ceiling set by the irreducible
pairs (``ceiling.py``) and the gap to it, plus a loss budget per error bucket restricted to reducible pairs.
Per-S1 scores are saved to ``<work>/eval/<tag>.parquet`` so two experiments on the same data can be compared
per S1 (paired difference with its standard error).
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from . import ceiling as CE
from . import decode as DC
from .io_utils import load_ground_truth, setup_logging

log = logging.getLogger("er")

DECODE_GRID_T = (0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
DECODE_GRID_E = [(miss, scale) for miss in (0.0, 0.02) for scale in (0.7, 0.8, 0.9, 1.0, 1.1)]


# ------------------------------------------------------------------ ground truth
def gt_pairs(rec: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """True (s, q) pairs as row indices into ``rec`` (pairs whose ids are not in ``rec`` are dropped)."""
    gt = load_ground_truth(data_dir)
    gid = pd.Series(rec["gid"].to_numpy(), index=rec["entity_id"].to_numpy())
    s, q = [], []
    for a, lst in gt.items():
        for b in lst:
            s.append(a)
            q.append(b)
    df = pd.DataFrame({"s": gid.reindex(s).to_numpy(), "q": gid.reindex(q).to_numpy()})
    return df.dropna().astype(np.int64)


def pair_key(q, s) -> np.ndarray:
    return np.asarray(q).astype(np.int64) * (1 << 32) + np.asarray(s).astype(np.int64)


def label_pairs(cand: pd.DataFrame, gtp: pd.DataFrame) -> np.ndarray:
    return np.isin(pair_key(cand["q"], cand["s"]), pair_key(gtp["q"], gtp["s"])).astype(np.int8)


# ------------------------------------------------------------------ metric
def per_s1(sel_s: np.ndarray, label_sel: np.ndarray, n_true_s: np.ndarray):
    """Per-row (indexed by gid) predicted count P, true positives TP and F0.5."""
    P = np.bincount(sel_s, minlength=len(n_true_s)).astype(np.float64)
    TP = np.bincount(sel_s, weights=label_sel.astype(np.float64), minlength=len(n_true_s))
    T = n_true_s.astype(np.float64)
    f = np.where((T == 0) & (P == 0), 1.0, np.where(TP > 0, 1.25 * TP / (0.25 * T + np.maximum(P, 1e-9)), 0.0))
    return P, TP, f


def evaluate(sel: pd.DataFrame, label_sel: np.ndarray, n_true_s: np.ndarray, s1_gids: np.ndarray,
             country_s1: np.ndarray | None = None, return_f: bool = False):
    """Exact macro F0.5 over all S1 (vectorised). ``sel``: selected pairs with column s."""
    P, TP, f = per_s1(sel["s"].to_numpy(), label_sel, n_true_s)
    T = n_true_s.astype(np.float64)
    fs = f[s1_gids]
    out = {"f05": float(fs.mean()), "pair_prec": float(TP.sum() / max(P.sum(), 1)),
           "pair_rec": float(TP.sum() / max(T.sum(), 1)),
           "singleton_acc": float(((P[s1_gids] == 0) & (T[s1_gids] == 0)).sum() / max((T[s1_gids] == 0).sum(), 1))}
    if country_s1 is not None:
        for c in pd.unique(country_s1):
            out[f"f05_{c}"] = float(fs[country_s1 == c].mean())
    if return_f:
        return out, pd.DataFrame({"gid": s1_gids, "f": fs, "P": P[s1_gids], "TP": TP[s1_gids], "T": T[s1_gids]})
    return out


def decode(assigned: pd.DataFrame, cfg: dict, prob_col: str = "p") -> pd.DataFrame:
    if cfg["method"] == "threshold":
        return DC.select_threshold(assigned, cfg["t"], prob_col)
    return DC.select_expected_f05(assigned, prob_col, miss_p=cfg["miss_p"], prob_scale=cfg["scale"])


def tune_decoding(pairs: pd.DataFrame, n_true_s, s1, ctry_s1, quiet=False) -> dict:
    """Grid over threshold / expected-F0.5 decoders on OOF scores; returns the best config (+ its metrics)."""
    assigned = DC.assign(pairs, "p")
    res = []
    for t in DECODE_GRID_T:
        cfg = {"method": "threshold", "t": t}
        sel = decode(assigned, cfg)
        res.append((cfg, evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), n_true_s, s1, ctry_s1)))
    for miss, scale in DECODE_GRID_E:
        cfg = {"method": "expected_f05", "miss_p": miss, "scale": scale}
        sel = decode(assigned, cfg)
        res.append((cfg, evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), n_true_s, s1, ctry_s1)))
    best = max(res, key=lambda r: r[1]["f05"])
    if not quiet:
        for cfg, m in res:
            log.info("decode %-60s f05=%.5f prec=%.4f rec=%.4f single=%.4f %s", json.dumps(cfg), m["f05"],
                     m["pair_prec"], m["pair_rec"], m["singleton_acc"],
                     " ".join(f"{k}={v:.4f}" for k, v in m.items() if k.startswith("f05_")))
        log.info("BEST decoding: %s -> %s", best[0], best[1])
    return dict(best[0], oof=best[1])


# ------------------------------------------------------------------ cached-run evaluation
class Truth:
    """Ground truth of a split, aligned to the row indices of its ``records.parquet``."""

    def __init__(self, work_dir: str, data_dir: str, split: str = "train"):
        self.path = os.path.join(work_dir, split, "records.parquet")
        rec = pd.read_parquet(self.path, columns=["entity_id", "source", "country", "gid"])
        self.n = len(rec)
        self.ids = rec["entity_id"].to_numpy(dtype=object)
        self.src = rec["source"].to_numpy()
        self.ctry = rec["country"].astype(str).to_numpy()
        self.s1 = np.nonzero(self.src == 1)[0]
        self.ctry_s1 = self.ctry[self.s1]
        self.gtp = gt_pairs(rec, data_dir)
        self.n_true_s = np.bincount(self.gtp["s"].to_numpy(), minlength=self.n)
        self.gt_key = pair_key(self.gtp["q"], self.gtp["s"])
        self.in_cand = None  # per true pair: retrieved by blocking (set by evaluate_run when cand_ctx is cached)
        self._irr = None

    def irreducible(self) -> dict:
        """Irreducible flags of the true pairs, {'exact': bool[], 'core': bool[]} (see ``ceiling``)."""
        if self._irr is None:
            rec = pd.read_parquet(self.path, columns=["name", "addr", "source", "country"])
            self._irr = CE.irreducible_flags(rec, self.gtp["s"].to_numpy(), self.gtp["q"].to_numpy())
        return self._irr

    def ceiling(self) -> dict:
        return CE.ceiling_stats(self.n_true_s, self.s1, self.gtp["s"].to_numpy(), self.irreducible(), self.ctry_s1)


def recall_stats(truth: Truth, q: np.ndarray, s: np.ndarray, prefix: str) -> dict:
    """Share of true pairs present in the (q, s) pair set, overall and per country of the S1."""
    hit = np.isin(truth.gt_key, pair_key(q, s))
    out = {f"{prefix}_recall": float(hit.mean()) if len(hit) else float("nan"),
           f"{prefix}_pairs": int(len(q)), f"{prefix}_per_query": float(len(q) / max(1, len(np.unique(q))))}
    c = truth.ctry[truth.gtp["s"].to_numpy()]
    for cc in pd.unique(c):
        out[f"{prefix}_recall_{cc}"] = float(hit[c == cc].mean())
    if prefix == "block":
        truth.in_cand = hit
    return out


# ------------------------------------------------------------------ ceiling and loss budget
def add_ceiling(m: dict, ce: dict, prefix: str = "") -> dict:
    """Add the ceiling and the gap to it (overall and per country) to a metrics dict."""
    for key in CE.KEYS:
        m[f"{prefix}ceiling_{key}"] = ce[f"ceiling_{key}"]
        m[f"{prefix}gap_{key}"] = ce[f"ceiling_{key}"] - m[f"{prefix}f05"]
    for k, v in list(m.items()):
        if k.startswith(f"{prefix}f05_") and f"ceiling_exact_{k[len(prefix) + 4:]}" in ce:
            m[f"{prefix}gap_exact_{k[len(prefix) + 4:]}"] = ce[f"ceiling_exact_{k[len(prefix) + 4:]}"] - v
    return m


class Budget:
    """Counterfactual macro-F0.5 gains of fixing sets of errors (every per-S1 F0.5 recomputed exactly)."""

    def __init__(self, n_true_s: np.ndarray, s1: np.ndarray, sel_s: np.ndarray, sel_y: np.ndarray):
        n = len(n_true_s)
        self.T = n_true_s.astype(np.float64)
        self.P0 = np.bincount(sel_s, minlength=n).astype(np.float64)
        self.TP0 = np.bincount(sel_s, weights=sel_y.astype(np.float64), minlength=n)
        self.s1 = s1
        self.base = float(self._f(self.P0, self.TP0).mean())

    def _f(self, P, TP):
        T = self.T
        f = np.where((T == 0) & (P == 0), 1.0, np.where(TP > 0, 1.25 * TP / (0.25 * T + np.maximum(P, 1e-9)), 0.0))
        return f[self.s1]

    def gain(self, add_s: np.ndarray, rm_s: np.ndarray) -> float:
        """Gain from adding the true pairs at ``add_s`` and removing false links at ``rm_s``."""
        n = len(self.T)
        dA = np.bincount(np.asarray(add_s, np.int64), minlength=n).astype(np.float64)
        dR = np.bincount(np.asarray(rm_s, np.int64), minlength=n).astype(np.float64)
        return float(self._f(self.P0 + dA - dR, self.TP0 + dA).mean()) - self.base


BUCKETS = ("block", "prefilter", "wrong_s1", "reject", "fp_nonsingle", "fp_single")


def bucket_pairs(truth: Truth, assigned: pd.DataFrame, sel: pd.DataFrame, scored_key: np.ndarray):
    """Bucket every true pair (block / prefilter / wrong_s1 / reject / tp) and every selected false link
    (wrong_s1_fp when its query belongs to another S1, else fp_nonsingle / fp_single).
    ``sel`` has columns s, q, y; ``scored_key`` = pair keys of every scored (post-prefilter) pair."""
    g = truth.gtp[["s", "q"]].copy()
    in_scored = np.isin(truth.gt_key, scored_key)
    in_cand = truth.in_cand if truth.in_cand is not None else in_scored
    asg = pd.Series(assigned["s"].to_numpy(), index=assigned["q"].to_numpy())
    g["asg_s"] = asg.reindex(g["q"].to_numpy()).fillna(-1).to_numpy().astype(np.int64)
    selected = np.isin(truth.gt_key, pair_key(sel["q"], sel["s"]))
    g["bucket"] = np.select([~in_cand, ~in_scored, g["asg_s"].to_numpy() != g["s"].to_numpy(), ~selected],
                            ["block", "prefilter", "wrong_s1", "reject"], "tp")
    fp = sel.loc[sel["y"].to_numpy() == 0, ["s", "q"]].copy()
    q_matched = np.isin(fp["q"].to_numpy(), truth.gtp["q"].to_numpy())
    fp["bucket"] = np.where(q_matched, "wrong_s1_fp",
                            np.where(truth.n_true_s[fp["s"].to_numpy()] == 0, "fp_single", "fp_nonsingle"))
    return g, fp


def loss_budget(truth: Truth, assigned: pd.DataFrame, sel: pd.DataFrame, scored_key: np.ndarray):
    """Loss budget restricted to reducible pairs. Returns (table, summary).

    ``gain`` = macro-F0.5 gain when only that bucket is fixed; ``gain_red`` = the same for its reducible pairs
    (exact-name irreducible pairs excluded; every false link is reducible). A wrong_s1 fix also removes the
    false link of that query. ``fix_all_reducible`` = gain when every reducible error is fixed at once."""
    irr = truth.irreducible()["exact"]
    g, fp = bucket_pairs(truth, assigned, sel, scored_key)
    B = Budget(truth.n_true_s, truth.s1, sel["s"].to_numpy(), sel["y"].to_numpy())
    wrong_fp = fp[fp["bucket"] == "wrong_s1_fp"]
    rows = []
    for b in BUCKETS:
        if b.startswith("fp"):
            s_ = fp.loc[fp["bucket"] == b, "s"].to_numpy()
            gain = B.gain(np.zeros(0, np.int64), s_)
            rows.append({"bucket": b, "pairs": len(s_), "irr_pairs": 0, "gain": gain, "gain_red": gain})
            continue
        m = (g["bucket"] == b).to_numpy()
        r = {"bucket": b, "pairs": int(m.sum()), "irr_pairs": int((m & irr).sum())}
        for tag, mm in (("gain", m), ("gain_red", m & ~irr)):
            rm = wrong_fp.loc[wrong_fp["q"].isin(g.loc[mm, "q"]), "s"].to_numpy() if b == "wrong_s1" \
                else np.zeros(0, np.int64)
            r[tag] = B.gain(g.loc[mm, "s"].to_numpy(), rm)
        rows.append(r)
    tab = pd.DataFrame(rows)
    ce = truth.ceiling()
    gap = ce["ceiling_exact"] - B.base
    tab["share_of_gap"] = tab["gain_red"] / gap if gap > 0 else np.nan
    red_err = (g["bucket"] != "tp").to_numpy() & ~irr
    summ = {"f05": B.base, "ceiling_exact": ce["ceiling_exact"], "gap_exact": gap,
            "fix_all_reducible": B.gain(g.loc[red_err, "s"].to_numpy(), fp["s"].to_numpy()),
            "irr_by_bucket": g.loc[irr, "bucket"].value_counts().to_dict(),
            "false_links": int(len(fp)), "wrong_s1_fp": int(len(wrong_fp))}
    return tab, summ


def log_budget(tab: pd.DataFrame, summ: dict, label: str):
    log.info("BUDGET[%s] f05=%.5f ceiling=%.5f gap=%.5f fix_all_reducible=+%.5f irreducible by bucket %s", label,
             summ["f05"], summ["ceiling_exact"], summ["gap_exact"], summ["fix_all_reducible"],
             json.dumps(summ["irr_by_bucket"]))
    for r in tab.itertuples():
        log.info("BUDGET[%s] %-13s pairs=%8d irr=%7d gain=%.5f gain_red=%.5f (%4.1f%% of gap)", label, r.bucket,
                 r.pairs, r.irr_pairs, r.gain, r.gain_red, 100 * r.share_of_gap)


def evaluate_run(work_dir: str, data_dir: str, art_dir: str | None = None, oof_name: str = "oof.parquet",
                 truth: Truth | None = None) -> tuple[dict, pd.DataFrame]:
    """Standard metrics of a cached training run. Returns (metrics, per-S1 frame)."""
    art_dir = art_dir or os.path.join(work_dir, "artifacts")
    truth = truth or Truth(work_dir, data_dir)
    out = {}
    cand_path = os.path.join(work_dir, "train", "cand_ctx.parquet")
    if os.path.exists(cand_path):
        cc = pd.read_parquet(cand_path, columns=["q", "s"])
        out.update(recall_stats(truth, cc["q"].to_numpy(), cc["s"].to_numpy(), "block"))
        del cc
    oof = pd.read_parquet(os.path.join(work_dir, "train", oof_name))
    out.update(recall_stats(truth, oof["q"].to_numpy(), oof["s"].to_numpy(), "prefilter"))
    with open(os.path.join(art_dir, "decode.json")) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if k != "oof"}
    pcol = "p2" if "p2" in oof else "p"
    pairs = pd.DataFrame({"s": oof["s"].to_numpy(), "q": oof["q"].to_numpy(), "p": oof[pcol].to_numpy(),
                          "y": oof["y"].to_numpy()})
    assigned = DC.assign(pairs, "p")
    sel = decode(assigned, cfg)
    m, per = evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), truth.n_true_s, truth.s1, truth.ctry_s1,
                      return_f=True)
    out.update(m)
    add_ceiling(out, truth.ceiling())
    tab, summ = loss_budget(truth, assigned, sel.assign(y=assigned.loc[sel.index, "y"].to_numpy()),
                            pair_key(pairs["q"], pairs["s"]))
    out["budget"], out["budget_summary"] = tab.to_dict("records"), summ
    irr = truth.irreducible()
    for key in CE.KEYS:
        k = np.bincount(truth.gtp["s"].to_numpy()[irr[key]], minlength=truth.n)
        per[f"k_irr_{key}"] = k[per["gid"].to_numpy()]
    out["decode"] = cfg
    out["n_pred_pairs"] = int(len(sel))
    out["pred_per_s1"] = float(len(sel) / len(truth.s1))
    out["pred_empty_rate"] = float((per["P"] == 0).mean())
    out["true_empty_rate"] = float((per["T"] == 0).mean())
    if "p1" in oof:
        best1 = tune_decoding(pairs.assign(p=oof["p1"].to_numpy()), truth.n_true_s, truth.s1, truth.ctry_s1,
                              quiet=True)
        out["stage1_f05"] = best1["oof"]["f05"]
    per["entity_id"] = truth.ids[per["gid"].to_numpy()]
    per["country"] = truth.ctry_s1
    return out, per


def fmt(m: dict) -> str:
    keys = ["f05"] + sorted(k for k in m if k.startswith("f05_")) + [
        "ceiling_exact", "gap_exact", "ceiling_core", "gap_core",
        "pair_prec", "pair_rec", "singleton_acc", "stage1_f05", "block_recall", "prefilter_recall",
        "prefilter_per_query", "pred_per_s1", "pred_empty_rate", "testlike_f05", "testlike_gap_exact"] + \
        sorted(k for k in m if k.startswith("testlike_f05_") or k.startswith("testlike_gap_exact_"))
    return "  ".join(f"{k}={m[k]:.5f}" for k in keys if k in m and isinstance(m[k], float))


def compare(per_a: pd.DataFrame, per_b: pd.DataFrame) -> dict:
    """Paired per-S1 comparison (b - a) on the S1 entities both runs share."""
    j = per_a[["entity_id", "f", "country"]].merge(per_b[["entity_id", "f"]], on="entity_id", suffixes=("_a", "_b"))
    d = j["f_b"].to_numpy() - j["f_a"].to_numpy()
    out = {"n": int(len(d)), "delta": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(max(len(d), 2))),
           "n_better": int((d > 1e-12).sum()), "n_worse": int((d < -1e-12).sum())}
    for c in pd.unique(j["country"]):
        dc = d[j["country"].to_numpy() == c]
        out[f"delta_{c}"] = float(dc.mean())
    return out


def save(work_dir: str, tag: str, metrics: dict, per: pd.DataFrame):
    d = os.path.join(work_dir, "eval")
    os.makedirs(d, exist_ok=True)
    cols = ["entity_id", "country", "f", "P", "TP", "T"] + [c for c in per.columns if c.startswith("k_irr_")]
    per[cols].to_parquet(os.path.join(d, f"{tag}.parquet"), index=False)
    with open(os.path.join(d, f"{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--art-dir", default=None)
    ap.add_argument("--tag", default=None, help="save per-S1 scores + metrics under <work>/eval/<tag>")
    ap.add_argument("--compare", default=None, help="tag of an earlier evaluation to compare against (paired)")
    ap.add_argument("--oof", default="oof.parquet")
    ap.add_argument("--testlike", action="store_true", help="also score the test-like validation (~40%% unmatched)")
    ap.add_argument("--testlike-target", type=float, default=0.40)
    args = ap.parse_args()
    setup_logging()
    truth = Truth(args.work_dir, args.data_dir)
    m, per = evaluate_run(args.work_dir, args.data_dir, args.art_dir, args.oof, truth=truth)
    if args.testlike:
        from . import testlike as TL
        os.makedirs(os.path.join(args.work_dir, "eval"), exist_ok=True)
        tl = TL.run(args.work_dir, args.data_dir, target=args.testlike_target, art_dir=args.art_dir, truth=truth)
        per_tl = tl["full"].pop("per_s1")
        for v in ("stage2", "full_prior_adj"):
            tl[v].pop("per_s1", None)
        if args.tag:
            per_tl[["entity_id", "country", "f", "P", "TP", "T"]].to_parquet(
                os.path.join(args.work_dir, "eval", f"{args.tag}_testlike.parquet"), index=False)
        if args.compare:
            ref = os.path.join(args.work_dir, "eval", f"{args.compare}_testlike.parquet")
            if os.path.exists(ref):
                c = compare(pd.read_parquet(ref), per_tl)
                log.info("PAIRED TESTLIKE vs %s: delta=%+.5f (se %.5f)  better=%d worse=%d  %s", args.compare,
                         c["delta"], c["se"], c["n_better"], c["n_worse"],
                         " ".join(f"{k}={v:+.5f}" for k, v in c.items() if k.startswith("delta_")))
        m["testlike_f05"] = tl["full"]["f05"]
        m["testlike_s2_f05"] = tl["stage2"]["f05"]
        ce = truth.ceiling()
        m["testlike_gap_exact"] = ce["ceiling_exact"] - m["testlike_f05"]
        for k, v in tl["full"].items():
            if k.startswith("f05_"):
                m[f"testlike_{k}"] = v
                m[f"testlike_gap_exact_{k[4:]}"] = ce.get(f"ceiling_exact_{k[4:]}", np.nan) - v
        tb = tl["full"].pop("budget")
        m["testlike_budget"], m["testlike_budget_summary"] = tb[0].to_dict("records"), tb[1]
        log_budget(tb[0], tb[1], "testlike")
        m["testlike"] = tl
        log.info("TESTLIKE %s", json.dumps(tl["info"]))
        log.info("TESTLIKE prior %s", json.dumps(tl["prior"]))
        m["testlike_prior_adj_f05"] = tl["full_prior_adj"]["f05"]
        for v in ("full", "stage2", "full_prior_adj"):
            r = tl[v]
            log.info("TESTLIKE[%s] f05=%.5f prec=%.4f rec=%.4f single=%.4f pred/s1=%.3f %s | tuned %s -> %.5f", v,
                     r["f05"], r["pair_prec"], r["pair_rec"], r["singleton_acc"], r["pred_per_s1"],
                     " ".join(f"{k}={x:.4f}" for k, x in r.items() if k.startswith("f05_")), r["tuned_cfg"],
                     r["tuned_f05"])
    log_budget(pd.DataFrame(m["budget"]), m["budget_summary"], "oof")
    ce = truth.ceiling()
    log.info("CEILING %s", " ".join(f"{k}={v}" for k, v in ce.items()))
    log.info("EVAL %s", fmt(m))
    for k, v in m.items():
        if k.startswith(("block_", "prefilter_")):
            log.info("  %s = %s", k, v)
    if args.tag:
        save(args.work_dir, args.tag, m, per)
    if args.compare:
        other = pd.read_parquet(os.path.join(args.work_dir, "eval", f"{args.compare}.parquet"))
        c = compare(other, per)
        log.info("PAIRED vs %s: delta=%+.5f (se %.5f)  better=%d worse=%d  %s", args.compare, c["delta"], c["se"],
                 c["n_better"], c["n_worse"],
                 " ".join(f"{k}={v:+.5f}" for k, v in c.items() if k.startswith("delta_")))


if __name__ == "__main__":
    main()
