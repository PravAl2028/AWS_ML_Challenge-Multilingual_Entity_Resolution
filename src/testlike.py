"""Test-like validation: OOF scoring under the test's distractor rate.

Test has ~5.75 queries per S1 vs ~4.68 in train, i.e. ~40% unmatched queries vs ~26%. To see how the
pipeline behaves under that shift, a random ~90% of the unmatched training queries get a second copy
(new query id, identical fields and candidate rows), bringing the unmatched share to ``target``. Then:

* ``full``   the S1-side competition counts of the context features (s_indeg, s_indeg_top1, s_rrank,
             s_n_cos_rank) are recomputed on the augmented pre-prefilter candidate set, stage 1 is re-predicted
             with the out-of-fold stage-1 models, stage-2 features are recomputed and stage 2 re-predicted;
* ``stage2`` stage-1 OOF scores are kept (copies inherit them) and only stage 2 is recomputed.

Assignment and decoding then run as usual; the copies are false candidates of every S1 they reach.
Every prediction uses the model of the row's own fold, so the score stays out-of-fold.
"""
from __future__ import annotations

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from . import decode as DC
from . import model as MD
from .evaluation import Truth, decode, evaluate, loss_budget, pair_key, tune_decoding
from .features import _grp_rank

log = logging.getLogger("er")

S1_COUNT_FEATURES = ("s_indeg", "s_indeg_top1", "s_rrank", "s_n_cos_rank")


def _s1_counts(cc: pd.DataFrame, src: np.ndarray) -> pd.DataFrame:
    """Recompute the per-S1 competition counts of ``features.context_features`` on a candidate set."""
    s = cc["s"].to_numpy().astype(np.int64)
    ks = s * 4 + src
    f = pd.DataFrame(index=cc.index)
    vc = pd.Series(ks).value_counts()
    f["s_indeg"] = pd.Series(ks).map(vc).to_numpy().astype(np.int32)
    top = cc["rank"].to_numpy() == 0
    f["s_indeg_top1"] = pd.Series(ks).map(pd.Series(ks[top]).value_counts()).fillna(0).to_numpy().astype(np.int32)
    f["s_rrank"] = _grp_rank(ks, cc["score"].to_numpy()).astype(np.int32)
    f["s_n_cos_rank"] = _grp_rank(ks, cc["n_cos"].to_numpy()).astype(np.int32)
    return f


def _rec_cols(work_dir: str) -> list:
    import pyarrow.parquet as pq
    return pq.ParquetFile(os.path.join(work_dir, "train", "records.parquet")).schema_arrow.names


def build(work_dir: str, truth: Truth, target: float = 0.40, seed: int = 0, art_dir: str | None = None) -> dict:
    """Return {'full': pairs DataFrame, 'stage2': pairs DataFrame, 'info': dict} for the test-like set."""
    art_dir = art_dir or os.path.join(work_dir, "artifacts")
    with open(os.path.join(art_dir, "models.pkl"), "rb") as f:
        M = pickle.load(f)
    X = pd.read_parquet(os.path.join(work_dir, "train", "pairs_feat.parquet"))
    oof = pd.read_parquet(os.path.join(work_dir, "train", "oof.parquet"))
    assert (oof["q"].to_numpy() == X["q"].to_numpy()).all() and (oof["s"].to_numpy() == X["s"].to_numpy()).all()
    q, s, fold = X["q"].to_numpy(), X["s"].to_numpy(), X["fold"].to_numpy()
    matched = np.zeros(truth.n, bool)
    matched[truth.gtp["q"].to_numpy()] = True
    uq = np.unique(q)
    unm = uq[~matched[uq]]
    n_q, n_u = len(uq), len(unm)
    n_dup = int(np.ceil((target * n_q - n_u) / (1 - target)))
    rng = np.random.default_rng(seed)
    pick = rng.choice(unm, size=min(max(n_dup, 0), n_u), replace=False)
    new_id = pd.Series(truth.n + np.arange(len(pick)), index=pick)
    info = {"queries": int(n_q), "unmatched_before": n_u / n_q, "copies": int(len(pick)),
            "unmatched_after": (n_u + len(pick)) / (n_q + len(pick))}
    src_ext = np.concatenate([truth.src, truth.src[pick]])

    # ---- augmented pre-prefilter candidate set -> recomputed S1-side counts for every post-prefilter row
    cc = pd.read_parquet(os.path.join(work_dir, "train", "cand_ctx.parquet"),
                         columns=["q", "s", "rank", "score", "n_cos"])
    dup = cc[cc["q"].isin(new_id.index)].copy()
    dup["q"] = new_id.reindex(dup["q"].to_numpy()).to_numpy()
    cc_aug = pd.concat([cc, dup], ignore_index=True)
    cnt = _s1_counts(cc_aug, src_ext[cc_aug["q"].to_numpy()])
    cnt_by_key = cnt.set_index(pair_key(cc_aug["q"], cc_aug["s"]))
    del cc, dup, cc_aug, cnt

    # ---- augmented post-prefilter rows (copies inherit features, label 0, fold of their S1)
    is_dup = X["q"].isin(new_id.index).to_numpy()
    Xd = X[is_dup].copy()
    Xd["q"] = new_id.reindex(Xd["q"].to_numpy()).to_numpy()
    Xd["y"] = 0
    p1d = oof["p1"].to_numpy()[is_dup]
    Xa = pd.concat([X, Xd], ignore_index=True)
    p1_old = np.concatenate([oof["p1"].to_numpy(), p1d])
    del X, Xd
    qa, sa, fa, ya = Xa["q"].to_numpy(), Xa["s"].to_numpy(), Xa["fold"].to_numpy(), Xa["y"].to_numpy()
    q_orig = np.where(qa >= truth.n, pd.Series(new_id.index, index=new_id.to_numpy()).reindex(qa).fillna(0)
                      .to_numpy().astype(np.int64), qa)
    ctx = None
    if any(c.startswith("c_") for c in M["feats2"]):  # collective stage-2 features: copies inherit attributes
        have = set(_rec_cols(work_dir))
        rec = pd.read_parquet(os.path.join(work_dir, "train", "records.parquet"),
                              columns=[c for c in ("n_core", "a_hnc", "a_street", "a_empty") if c in have])
        ctx = MD.collective_ctx(rec, q_orig)
        del rec
    cc_keys = pd.DataFrame({"q": qa, "s": sa, "rank": Xa["rank"].to_numpy()})
    out = {"info": info, "copy_of": pd.Series(new_id.index.to_numpy(), index=new_id.to_numpy())}  # copy -> original
    for variant in ("stage2", "full"):
        X1 = Xa[M["feats1"]].copy()
        if variant == "full":
            upd = cnt_by_key.reindex(pair_key(qa, sa))
            for c in S1_COUNT_FEATURES:
                if c in X1:
                    X1[c] = upd[c].to_numpy().astype(X1[c].dtype)
            p1 = MD.predict_oof(M["m1"], X1, fa)
        else:
            p1 = p1_old
        S2 = MD.stage2_features(cc_keys, src_ext[qa], p1, ctx)
        X2 = pd.concat([X1.reset_index(drop=True), S2.reset_index(drop=True)], axis=1)[M["feats2"]]
        p2 = MD.predict_oof(M["m2"], X2, fa)
        out[variant] = pd.DataFrame({"s": sa, "q": qa, "p": p2, "p1": p1, "y": ya})
    return out


def score(pairs: pd.DataFrame, truth: Truth, cfg: dict, tune: bool = True, budget: bool = False) -> dict:
    assigned = DC.assign(pairs, "p")
    sel = decode(assigned, cfg)
    m, per = evaluate(sel, assigned.loc[sel.index, "y"].to_numpy(), truth.n_true_s, truth.s1, truth.ctry_s1,
                      return_f=True)
    if budget:  # (table, summary) of the reducible loss budget; copies are distractors of every S1 they reach
        m["budget"] = loss_budget(truth, assigned, sel.assign(y=assigned.loc[sel.index, "y"].to_numpy()),
                                  pair_key(pairs["q"], pairs["s"]))
    per["entity_id"] = truth.ids[per["gid"].to_numpy()]
    per["country"] = truth.ctry_s1
    m["per_s1"] = per
    m["pred_per_s1"] = float(len(sel) / len(truth.s1))
    if tune:
        best = tune_decoding(pairs, truth.n_true_s, truth.s1, truth.ctry_s1, quiet=True)
        m["tuned_cfg"] = {k: v for k, v in best.items() if k != "oof"}
        m["tuned_f05"] = best["oof"]["f05"]
    return m


def run(work_dir: str, data_dir: str, target: float = 0.40, seed: int = 0, art_dir: str | None = None,
        truth: Truth | None = None) -> dict:
    truth = truth or Truth(work_dir, data_dir)
    art_dir = art_dir or os.path.join(work_dir, "artifacts")
    with open(os.path.join(art_dir, "decode.json")) as f:
        cfg = {k: v for k, v in json.load(f).items() if k != "oof"}
    T = build(work_dir, truth, target, seed, art_dir)
    res = {"info": T["info"]}
    for v in ("full", "stage2"):
        res[v] = score(T[v], truth, cfg, budget=(v == "full"))
    # prior-shift check: EM-estimate the pair prior on the (unlabeled) test-like scores, adjust, decode
    oof = pd.read_parquet(os.path.join(work_dir, "train", "oof.parquet"), columns=["y", "p2"])
    pi_tr = float(oof["y"].mean())
    pairs = T["full"]
    pi_new = DC.em_prior(pairs["p"].to_numpy(), pi_tr)
    adj = pairs.assign(p=DC.prior_shift(pairs["p"].to_numpy(), pi_tr, pi_new))
    res["prior"] = {"train": pi_tr, "em_estimate": pi_new, "true_testlike": float(pairs["y"].mean()),
                    "em_on_train_oof": DC.em_prior(oof["p2"].to_numpy(), pi_tr)}
    res["full_prior_adj"] = score(adj, truth, cfg)
    return res
