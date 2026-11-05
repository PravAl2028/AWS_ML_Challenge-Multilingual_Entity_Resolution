"""End-to-end pipeline: data -> normalisation -> blocking -> prefilter -> matcher -> decoding -> outputs.

    python -m src.run --data-dir <student_resource/dataset> --work-dir <work> --out-dir <output> all

Stages: ``train`` (fit everything on the training split and tune decoding on out-of-fold scores)
and ``predict`` (apply to the test split and write matching_results.tsv + candidate_pairs.tsv).
Intermediate artefacts are cached under --work-dir so a stage can be resumed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from . import blocking as BL
from . import decode as DC
from . import features as FT
from . import fillers as FI
from . import model as MD
from . import roles as RO
from . import surface as SF
from .evaluation import evaluate, gt_pairs, label_pairs, pair_key, tune_decoding  # noqa: F401 (re-exported)
from .io_utils import setup_logging, timer, write_id_lists
from .par import n_jobs_default
from .prepare import build_records, fold_of

log = logging.getLogger("er")


# ------------------------------------------------------------------ helpers
# Cache invalidation. ``--redo X`` recomputes stage X and everything after it; earlier stages come from
# the cache. Retrieval (the expensive sparse top-K search) is only redone with ``--force`` (or when its
# cache is missing), so ``--redo records`` re-parses the records but keeps the cached candidate set.
REDO_ORDER = ("records", "context", "features", "models", "decode")


def _redo(args, stage: str) -> bool:
    """True when ``stage`` must be recomputed rather than loaded from the cache."""
    if args.force:
        return True
    start = args.redo or "models"  # default: reuse cached data/features, retrain models, re-decode
    return REDO_ORDER.index(stage) >= REDO_ORDER.index(start)


def _cand_from_ctx(cc: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the raw retrieval output from a cached cand_ctx.parquet (caches written before cand.parquet)."""
    top = cc["rank"].to_numpy() == 0
    second = pd.Series((cc["score"] - cc["gap_alt"]).to_numpy()[top], index=cc["q"].to_numpy()[top])
    return pd.DataFrame({"q": cc["q"].to_numpy(), "s": cc["s"].to_numpy(), "score": cc["score"].to_numpy(),
                         "rank": cc["rank"].to_numpy(), "top1": cc["top1"].to_numpy(),
                         "second": second.reindex(cc["q"].to_numpy()).to_numpy().astype(np.float32)})


def _cached(path, build, force=False):
    if os.path.exists(path) and not force:
        log.info("loading cached %s", path)
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df


CE_FEATURES = ("ce", "ce_rank_q", "ce_alt_q")


def ce_features(q: np.ndarray, s: np.ndarray, path: str) -> pd.DataFrame:
    """v9: character cross-encoder features of the pairs (q, s) from a ``scripts/v8_ce.py`` score file (s, q, ce):
    the score and its query-level competition (rank among the query's S1, margin over the best other S1)."""
    from .features import _grp_max_other, _grp_rank
    ce = pd.read_parquet(path)
    key = pd.Series(ce["ce"].to_numpy(), index=pair_key(ce["q"], ce["s"]))
    v = key.reindex(pair_key(q, s)).to_numpy()
    if np.isnan(v).any():
        raise SystemExit(f"{int(np.isnan(v).sum())} of {len(v)} pairs have no cross-encoder score in {path}")
    v = v.astype(np.float32)
    return pd.DataFrame({"ce": v, "ce_rank_q": _grp_rank(q, v).astype(np.int16),
                         "ce_alt_q": v - _grp_max_other(q, v)})


def run_validator(out_dir: str, test_dir: str):
    v = os.path.join(os.path.dirname(os.path.abspath(test_dir)), "..", "utils", "validate_submission.py")
    v = os.path.abspath(v)
    if not os.path.exists(v):
        log.warning("validator not found at %s", v)
        return
    subprocess.run([sys.executable, v, "--matching", os.path.join(out_dir, "matching_results.tsv"),
                    "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"), "--test-dir", test_dir],
                   check=False)


# ------------------------------------------------------------------ shared split processing
def process_split(args, split: str, art: str, merges: dict | None, gtp: pd.DataFrame | None,
                  rec: pd.DataFrame | None = None):
    wd = os.path.join(args.work_dir, split)
    os.makedirs(wd, exist_ok=True)
    if rec is None:
        rec = _cached(os.path.join(wd, "records.parquet"),
                      lambda: build_records(args.data_dir, split, art, args.n_jobs), _redo(args, "records"))
    if rec["country"].dtype != "category":
        rec["country"] = rec["country"].astype("category")
    if split == "train":
        merges = BL.learn_state_merges(rec, gtp)
        with open(os.path.join(art, "state_merges.json"), "w") as f:
            json.dump(merges, f, indent=1)
    args.merges = merges
    part_key = BL.partition_keys(rec, merges)
    ctx_path = os.path.join(wd, "cand_ctx.parquet")
    cand_path = os.path.join(wd, "cand.parquet")
    if os.path.exists(ctx_path) and not _redo(args, "context"):
        log.info("loading cached %s", ctx_path)
        cc = pd.read_parquet(ctx_path)
        if "via_key" not in cc:  # caches written before the key channel existed
            cc["via_key"] = np.int8(0)
        return rec, cc, part_key
    have_cand = os.path.exists(cand_path) and not args.force
    old_ctx = os.path.exists(ctx_path) and not args.force
    Xn, Xa, Xc = BL.build_matrices(rec, args.n_jobs, with_retrieval=not (have_cand or old_ctx))
    if have_cand:
        log.info("loading cached %s", cand_path)
        cand = pd.read_parquet(cand_path)
    elif old_ctx:
        log.info("rebuilding %s from %s", cand_path, ctx_path)
        cand = _cand_from_ctx(pd.read_parquet(ctx_path))
        cand.to_parquet(cand_path, index=False)
    else:
        cand = BL.retrieve(rec, Xc, merges, args.n_jobs, K=args.K, ratio=args.ratio)
        cand.to_parquet(cand_path, index=False)
    del Xc
    if args.key_channel:
        cand = BL.key_candidates(rec, cand, Xn, Xa, K=args.K)
    with timer("context features"):
        ctx = FT.context_features(cand, rec, Xn, Xa, part_key)
    del Xn, Xa
    cc = pd.concat([cand[["q", "s"]], ctx], axis=1)
    del cand, ctx
    cc.to_parquet(ctx_path, index=False)
    return rec, cc, part_key


# ------------------------------------------------------------------ TRAIN
def stage_train(args):
    art = args.art_dir
    os.makedirs(art, exist_ok=True)
    rec_path = os.path.join(args.work_dir, "train", "records.parquet")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    if args.redo == "decode" and not args.force:
        return redecode_train(args)
    rec = _cached(rec_path, lambda: build_records(args.data_dir, "train", art, args.n_jobs), _redo(args, "records"))
    gtp = gt_pairs(rec, args.data_dir)
    rec, cc, part_key = process_split(args, "train", art, None, gtp, rec=rec)
    y = label_pairs(cc, gtp)
    src = rec["source"].to_numpy()
    s1 = np.nonzero(src == 1)[0]
    n_true_s = np.bincount(gtp["s"].to_numpy(), minlength=len(rec))
    ctry = rec["country"].astype(str).to_numpy()
    n_pos = len(gtp)
    log.info("blocking pair recall: %.5f  (%d / %d true pairs; %d candidates)", y.sum() / n_pos, y.sum(), n_pos,
             len(cc))
    for c in pd.unique(ctry[s1]):
        m = ctry[gtp["s"].to_numpy()] == c
        mc = ctry[cc["s"].to_numpy()] == c
        log.info("  %s: recall %.5f", c, y[mc].sum() / max(1, m.sum()))
    # model folds: split by Source-1 entity (hash of its id; with --folds 2 identical to the record folds used
    # for the native-script dictionaries). Every candidate pair takes the fold of its S1.
    fold = model_folds(rec, args.folds)[cc["s"].to_numpy()]
    grp = cc["s"].to_numpy()

    feat_path = os.path.join(args.work_dir, "train", "pairs_feat.parquet")
    folds_path = os.path.join(args.work_dir, "train", "pairs_feat.folds.json")
    cached_folds = json.load(open(folds_path)).get("folds") if os.path.exists(folds_path) else 2
    if os.path.exists(feat_path) and not _redo(args, "features") and cached_folds != args.folds:
        raise SystemExit(f"cached features were built with {cached_folds} folds; rerun with --redo features")
    if os.path.exists(feat_path) and not _redo(args, "features"):
        log.info("loading cached %s", feat_path)
        X1 = pd.read_parquet(feat_path)
        y, fold = X1.pop("y").to_numpy(), X1.pop("fold").to_numpy()
        cc = X1[["q", "s", "rank"]].copy()
        X1 = X1.drop(columns=["q", "s"])
        grp = cc["s"].to_numpy()
        with open(os.path.join(art, "prefilter.pkl"), "rb") as f:
            pre_models, thr, lo_full = pickle.load(f)
    else:
        X1, y, fold, grp, cc, pre_models, thr, lo_full = build_train_features(args, rec, cc, y, fold, grp, n_pos)
        X1.assign(q=cc["q"].to_numpy(), s=cc["s"].to_numpy(), y=y, fold=fold).to_parquet(feat_path, index=False)
        with open(folds_path, "w") as f:
            json.dump({"folds": args.folds}, f)
        with open(os.path.join(art, "prefilter.pkl"), "wb") as f:
            pickle.dump((pre_models, thr, lo_full), f)
    if args.roles and not set(RO.ALL_FEATURES) <= set(X1.columns):
        # v4+ transductive word-role features: label-free, computed from the split's own candidate pairs, so they
        # can be appended to cached v3 pair features without recomputing anything else
        with timer("word-role features (transductive)"):
            ro = RO.all_features(cc[["q", "s", "rank"]].reset_index(drop=True), rec)
        for c in RO.ALL_FEATURES:
            X1[c] = ro[c].to_numpy()
        # write a new file and swap it in: the cached file may be a hard link shared with another work dir
        tmp = feat_path + ".tmp"
        X1.assign(q=cc["q"].to_numpy(), s=cc["s"].to_numpy(), y=y, fold=fold).to_parquet(tmp, index=False)
        os.replace(tmp, feat_path)
        log.info("word-role features appended to %s (new file)", feat_path)
    if args.ce_train and not set(CE_FEATURES) <= set(X1.columns):
        # v9: character cross-encoder features (out-of-fold scores of scripts/v8_ce.py for exactly these pairs)
        with timer("cross-encoder features (v9)"):
            cef = ce_features(cc["q"].to_numpy(), cc["s"].to_numpy(), args.ce_train)
        for c in CE_FEATURES:
            X1[c] = cef[c].to_numpy()
        tmp = feat_path + ".tmp"
        X1.assign(q=cc["q"].to_numpy(), s=cc["s"].to_numpy(), y=y, fold=fold).to_parquet(tmp, index=False)
        os.replace(tmp, feat_path)
        log.info("cross-encoder features appended to %s (new file)", feat_path)
    if args.stop_after == "features":  # shared features for several model variants (forked work dirs)
        log.info("features cached in %s; stopping (--stop-after features)", feat_path)
        return
    drop = set(args.drop_features.split(",")) if args.drop_features else set()
    if not args.roles:
        drop |= set(RO.ALL_FEATURES)
    elif not args.cross_roles:
        drop |= set(RO.CROSS_FEATURES)
    if not args.surface:  # raw-surface name/address features (cached in pairs_feat) only with --surface
        drop |= set(SF.SURFACE_FEATURES)
    elif not set(SF.SURFACE_FEATURES) <= set(X1.columns):
        raise SystemExit("cached features have no surface columns; rerun with --redo features")
    if not args.collective_counts:  # E6 distractor-count features of stage 2 only with --collective-counts
        drop |= set(MD.COUNT_FEATURES)
    if drop:
        log.info("dropping features: %s", sorted(drop))
        X1 = X1.drop(columns=[c for c in X1.columns if c in drop])
    if "ce" in X1.columns and args.ce_dropout > 0:
        # v9: cross-encoder features blanked (missing) for a hash-chosen share of S1 groups, so the matchers also
        # learn to work without them: countries without training labels get them blanked at test time (the
        # cross-encoder has never seen their text), as the token log-odds dropout does for unseen vocabulary
        ce_drop = MD._inner_split(grp, frac=args.ce_dropout, seed=29)
        for c in CE_FEATURES:
            X1[c] = X1[c].astype(np.float32)
            X1.loc[ce_drop, c] = np.nan
        log.info("cross-encoder feature dropout on %d / %d rows", int(ce_drop.sum()), len(X1))
    feats1 = list(X1.columns)
    matched_q = np.zeros(len(rec), bool)
    matched_q[gt_pairs(rec, args.data_dir)["q"].to_numpy()] = True
    w = None
    if args.neg_weight > 0:  # emphasise distractors (negatives of queries that match no S1), as on test
        w = np.where((y == 0) & ~matched_q[cc["q"].to_numpy()], 1.0 + args.neg_weight, 1.0).astype(np.float32)
        log.info("distractor-negative weight %.2f on %d rows", 1 + args.neg_weight, int((w > 1).sum()))
    params = dict(MD.PARAMS_MAIN, **(json.loads(args.lgb) if args.lgb else {}))
    log.info("matcher params: %s", params)
    q_all, s_all = cc["q"].to_numpy(), cc["s"].to_numpy()
    aug = MD.copy_plan(q_all, matched_q, args.aug_target, args.aug_seed, len(rec)) if args.aug_target > 0 else None
    if aug is not None:  # train on folds with the test's distractor structure (copies of unmatched queries)
        log.info("training augmentation (stages %s): %s", args.aug_stages, aug["info"])
        r = aug["rows"]
        ya, folda, grpa = np.r_[y, np.zeros(len(r), y.dtype)], np.r_[fold, fold[r]], np.r_[grp, grp[r]]
        wa = None if w is None else np.r_[w, w[r]]
    with timer("stage-1 matcher"):
        if aug is not None and "1" in args.aug_stages:
            X1a = pd.concat([X1, X1.iloc[r]], ignore_index=True)
            p1a, m1 = MD.train_oof(X1a, ya, folda, grpa, params, args.n_jobs, max_rounds=args.max_rounds, weight=wa)
            del X1a
            p1 = p1a[:len(y)]
        else:
            p1, m1 = MD.train_oof(X1, y, fold, grp, params, args.n_jobs, max_rounds=args.max_rounds, weight=w)
    log.info("stage-1 importance:\n%s", MD.importance(m1).to_string())
    S2 = MD.stage2_features(cc, src[q_all], p1, MD.collective_ctx(rec, q_all))
    S2 = S2.drop(columns=[c for c in S2.columns if c in drop])
    X2 = pd.concat([X1, S2], axis=1)
    feats2 = list(X2.columns)
    with timer("stage-2 matcher"):
        if aug is not None:
            # stage-2 competition / collective features recomputed with the copies present (a copy inherits its
            # original's stage-1 score: same features, same fold model); the plain OOF scores are then
            # re-predicted with the features of the unaugmented set, so the OOF metric stays comparable
            qa = np.r_[q_all, aug["new_q"]]
            q_orig = np.r_[q_all, q_all[r]]
            keys = pd.DataFrame({"q": qa, "s": np.r_[s_all, s_all[r]], "rank": np.r_[cc["rank"].to_numpy(),
                                                                                         cc["rank"].to_numpy()[r]]})
            S2a = MD.stage2_features(keys, src[q_orig], np.r_[p1, p1[r]], MD.collective_ctx(rec, q_orig))
            X2a = pd.concat([pd.concat([X1, X1.iloc[r]], ignore_index=True), S2a[S2.columns]], axis=1)
            _, m2 = MD.train_oof(X2a, ya, folda, grpa, params, args.n_jobs, max_rounds=args.max_rounds, weight=wa)
            del X2a, S2a
            p2 = MD.predict_oof(m2, X2, fold)
        else:
            p2, m2 = MD.train_oof(X2, y, fold, grp, params, args.n_jobs, max_rounds=args.max_rounds, weight=w)
    log.info("stage-2 importance:\n%s", MD.importance(m2).to_string())

    # ---------------- decoding on OOF scores
    pairs = pd.DataFrame({"s": cc["s"].to_numpy(), "q": cc["q"].to_numpy(), "p": p2, "y": y})
    best = tune_decoding(pairs, n_true_s, s1, ctry[s1])
    pairs1 = pairs.assign(p=p1)
    log.info("stage-1 only (reference): %s", tune_decoding(pairs1, n_true_s, s1, ctry[s1], quiet=True))
    pd.DataFrame({"s": pairs.s, "q": pairs.q, "y": y, "p1": p1, "p2": p2}).to_parquet(
        os.path.join(args.work_dir, "train", "oof.parquet"), index=False)
    with open(os.path.join(art, "models.pkl"), "wb") as f:
        pickle.dump({"pre": pre_models, "pre_thr": thr, "pre_features": FT.PREFILTER_FEATURES,
                     "m1": m1, "feats1": feats1, "m2": m2, "feats2": feats2, "lo_full": lo_full,
                     "decode": best, "merges": args.merges, "K": args.K, "ratio": args.ratio,
                     "key_channel": bool(args.key_channel),
                     "train_countries": sorted(pd.unique(ctry[s1]).tolist())}, f)
    with open(os.path.join(art, "decode.json"), "w") as f:
        json.dump(best, f, indent=1)
    log.info("saved models to %s", art)
    if args.decode_on == "testlike":
        _save_decode(art, _tune_on_testlike(args))


def _tune_on_testlike(args) -> dict:
    """Tune the decoding rule on the test-like validation (training OOF at the test's distractor rate)."""
    from . import testlike as TL
    from .evaluation import Truth
    truth = Truth(args.work_dir, args.data_dir)
    T = TL.build(args.work_dir, truth, art_dir=args.art_dir)
    log.info("test-like set: %s", T["info"])
    best = tune_decoding(T["full"], truth.n_true_s, truth.s1, truth.ctry_s1)
    best["chosen_on"] = "testlike"
    return best


def _save_decode(art: str, best: dict):
    with open(os.path.join(art, "models.pkl"), "rb") as f:
        M = pickle.load(f)
    M["decode"] = best
    with open(os.path.join(art, "models.pkl"), "wb") as f:
        pickle.dump(M, f)
    with open(os.path.join(art, "decode.json"), "w") as f:
        json.dump(best, f, indent=1)
    log.info("decoding rule saved to %s: %s", art, {k: v for k, v in best.items() if k != "oof"})


def redecode_train(args):
    """``--redo decode``: re-tune the decoding rule on cached scores (OOF or test-like), keep the trained models."""
    art = args.art_dir
    rec = pd.read_parquet(os.path.join(args.work_dir, "train", "records.parquet"),
                          columns=["entity_id", "source", "country", "gid"])
    gtp = gt_pairs(rec, args.data_dir)
    s1 = np.nonzero(rec["source"].to_numpy() == 1)[0]
    ctry = rec["country"].astype(str).to_numpy()
    n_true_s = np.bincount(gtp["s"].to_numpy(), minlength=len(rec))
    oof = pd.read_parquet(os.path.join(args.work_dir, "train", "oof.parquet"))
    pairs = pd.DataFrame({"s": oof["s"].to_numpy(), "q": oof["q"].to_numpy(), "p": oof["p2"].to_numpy(),
                          "y": oof["y"].to_numpy()})
    best = tune_decoding(pairs, n_true_s, s1, ctry[s1])
    if args.decode_on == "testlike":
        best = _tune_on_testlike(args)
    _save_decode(art, best)


def model_folds(rec: pd.DataFrame, n: int) -> np.ndarray:
    """Model fold of every record row (only S1 rows are used: a pair takes the fold of its S1)."""
    if n == 2 and "fold" in rec:
        return rec["fold"].to_numpy()
    return fold_of(rec["entity_id"].to_numpy(dtype=object), n)


def build_train_features(args, rec, cc, y, fold, grp, n_pos):
    # ---------------- prefilter
    Xp = cc[FT.PREFILTER_FEATURES]
    pre_mask = MD._inner_split(grp, frac=args.pre_sample, seed=11) if args.pre_sample < 1 else np.ones(len(cc), bool)
    with timer("prefilter model"):
        oof_pre = np.full(len(cc), np.nan, dtype=np.float32)
        pre_models = []
        for k in range(args.folds):
            trm = pre_mask & (fold != k)
            m = MD.fit_one(Xp[trm], y[trm], grp[trm], MD.PARAMS_PRE, args.n_jobs, max_rounds=1000, es=50)
            oof_pre[fold == k] = m.predict(Xp[fold == k], num_iteration=m.best_iteration)
            pre_models.append(m)
    thr = MD.prefilter_threshold(oof_pre, y, cc["rank"].to_numpy(), args.keep_recall)
    keep = (oof_pre >= thr) | (cc["rank"].to_numpy() == 0)
    log.info("prefilter thr %.5f keeps %d / %d pairs (%.2f per query); recall %.5f -> %.5f", thr, keep.sum(),
             len(cc), keep.sum() / max(1, cc.q.nunique()), y.sum() / n_pos, y[keep].sum() / n_pos)
    cc = cc[keep].reset_index(drop=True)
    y, fold, grp = y[keep], fold[keep], grp[keep]
    cc["p_pre"] = oof_pre[keep]

    # ---------------- token statistics (OOF supervised, and unsupervised markers)
    with timer("token statistics"):
        lo_tabs = [FT.token_logodds(cc, rec, y, fold != k) for k in range(args.folds)]
        lo_full = FT.token_logodds(cc, rec, y, np.ones(len(cc), bool))
        mk = FT.marker_stats(cc, rec)
    lo_idx = fold.astype(np.int16)
    pf = FT.pair_features(cc, rec, lo_tabs, lo_idx, mk, args.n_jobs)
    if args.lo_dropout > 0:
        # token log-odds dropout: for a random share of S1 groups the supervised token tables are "unknown"
        # (as for a country never seen in training), so the matchers also learn to work without them
        drop_rows = MD._inner_split(grp, frac=args.lo_dropout, seed=23)
        pf.loc[drop_rows, [c for c in FT.LO_FEATURES if c in pf]] = 0.0  # includes lo_cov = 0 ("nothing known")
        log.info("token log-odds dropout on %d / %d rows", int(drop_rows.sum()), len(pf))
    X1 = pd.concat([cc.drop(columns=["q", "s"]), pf], axis=1)
    return X1, y, fold, grp, cc, pre_models, thr, lo_full


# ------------------------------------------------------------------ PREDICT
def stage_predict(args):
    art = args.art_dir
    with open(os.path.join(art, "models.pkl"), "rb") as f:
        M = pickle.load(f)
    args.K, args.ratio = M["K"], M["ratio"]
    args.key_channel = M.get("key_channel", False)  # candidate generation must match training
    scores_path = os.path.join(args.work_dir, "test", "scores.parquet")
    if args.redo == "decode" and not args.force and os.path.exists(scores_path):
        rec = pd.read_parquet(os.path.join(args.work_dir, "test", "records.parquet"),
                              columns=["entity_id", "source", "country"])
        return write_outputs(args, M, rec, pd.read_parquet(scores_path))
    rec, cc, part_key = process_split(args, "test", art, M["merges"], None)
    src = rec["source"].to_numpy()
    p_pre = MD.predict_avg(M["pre"], cc[M["pre_features"]])
    keep = (p_pre >= M["pre_thr"]) | (cc["rank"].to_numpy() == 0)
    log.info("test prefilter keeps %d / %d pairs", keep.sum(), len(cc))
    cc = cc[keep].reset_index(drop=True)
    cc["p_pre"] = p_pre[keep]
    mk = FT.marker_stats(cc, rec)
    pf = FT.pair_features(cc, rec, [M["lo_full"]], np.zeros(len(cc), np.int16), mk, args.n_jobs)
    X1 = pd.concat([cc.drop(columns=["q", "s"]), pf], axis=1)
    if any(c in M["feats1"] for c in RO.ALL_FEATURES):  # v4+: word-role statistics of the test split itself
        with timer("word-role features (transductive, test)"):
            ro = RO.all_features(cc[["q", "s", "rank"]], rec)
        for c in RO.ALL_FEATURES:
            X1[c] = ro[c].to_numpy()
    if "ce" in M["feats1"]:  # v9: cross-encoder scores of the test pairs (scripts/v8_ce.py)
        if not args.ce_test:
            raise SystemExit("these models use cross-encoder features: pass --ce-test <ce_test.parquet>")
        cef = ce_features(cc["q"].to_numpy(), cc["s"].to_numpy(), args.ce_test)
        seen = M.get("train_countries")
        unseen = (~np.isin(rec["country"].astype(str).to_numpy()[cc["s"].to_numpy()], seen)) if seen else \
            np.zeros(len(cc), bool)
        for c in CE_FEATURES:
            X1[c] = np.where(unseen, np.nan, cef[c].to_numpy().astype(np.float32)).astype(np.float32)
        log.info("cross-encoder features: %d test pairs, blanked for %d pairs of countries without training labels",
                 len(cc), int(unseen.sum()))
    X1 = X1[M["feats1"]]
    p1 = MD.predict_avg(M["m1"], X1)
    S2 = MD.stage2_features(cc, src[cc["q"].to_numpy()], p1, MD.collective_ctx(rec, cc["q"].to_numpy()))
    X2 = pd.concat([X1, S2], axis=1)[M["feats2"]]
    p2 = MD.predict_avg(M["m2"], X2)
    scores = pd.DataFrame({"s": cc["s"].to_numpy(), "q": cc["q"].to_numpy(), "p1": p1, "p2": p2})
    scores.to_parquet(scores_path, index=False)
    write_outputs(args, M, rec, scores, top=cc.loc[cc["rank"].to_numpy() == 0, ["q", "s", "rank"]])


def _filler_rule(args, rec: pd.DataFrame, scores: pd.DataFrame, top: pd.DataFrame | None, M: dict) -> pd.DataFrame:
    """v7: raise the scores of same-address noise-filler pairs (``fillers.py``) before decoding. Needs the test
    records' name / legal / address fields and the rank-0 candidate pairs (from ``cand_ctx`` when not given)."""
    need = ["country", "n_core", "n_legal", "a_hn", "a_hnc", "a_street", "a_nums", "a_unit"]
    if any(c not in rec for c in need):
        extra = pd.read_parquet(os.path.join(args.work_dir, "test", "records.parquet"),
                                columns=[c for c in need if c not in rec])
        rec = pd.concat([rec.reset_index(drop=True), extra], axis=1)
    if top is None:
        cc = pd.read_parquet(os.path.join(args.work_dir, "test", "cand_ctx.parquet"), columns=["q", "s", "rank"])
        top = cc[cc["rank"].to_numpy() == 0].reset_index(drop=True)
        del cc
    with timer("same-address filler rule"):
        out = FI.apply(scores, top, rec, known=None, col="p2", log=log)
    raised = out["p2"].to_numpy() != scores["p2"].to_numpy()
    scores.loc[raised, ["s", "q", "p2"]].assign(p2_rule=out["p2"].to_numpy()[raised]).to_parquet(
        os.path.join(args.work_dir, "test", "filler_rule_pairs.parquet"), index=False)
    return out


def _vocab_cap(args, rec: pd.DataFrame, scores: pd.DataFrame, top: pd.DataFrame | None) -> pd.DataFrame:
    """v8b: cap the scores of same-address swaps to a word that behaves like distractor vocabulary in this split
    (legal form redrawn in >= 12% of its same-address swaps; ``fillers.apply_cap``) at 0.05 before decoding."""
    need = ["country", "n_core", "n_legal", "a_hn", "a_hnc", "a_street", "a_nums", "a_unit"]
    if any(c not in rec for c in need):
        extra = pd.read_parquet(os.path.join(args.work_dir, "test", "records.parquet"),
                                columns=[c for c in need if c not in rec])
        rec = pd.concat([rec.reset_index(drop=True), extra], axis=1)
    if top is None:
        cc = pd.read_parquet(os.path.join(args.work_dir, "test", "cand_ctx.parquet"), columns=["q", "s", "rank"])
        top = cc[cc["rank"].to_numpy() == 0].reset_index(drop=True)
        del cc
    with timer("same-address vocabulary cap"):
        out = FI.apply_cap(scores, top, rec, col="p2", log=log)
    return out


def write_outputs(args, M: dict, rec: pd.DataFrame, scores: pd.DataFrame, top: pd.DataFrame | None = None):
    """Decode the test scores and write matching_results.tsv + candidate_pairs.tsv (then validate)."""
    src = rec["source"].to_numpy()
    if getattr(args, "filler_rule", False):
        scores = _filler_rule(args, rec, scores, top, M)
    if getattr(args, "vocab_cap", False):
        scores = _vocab_cap(args, rec, scores, top)
    pairs = pd.DataFrame({"s": scores["s"].to_numpy(), "q": scores["q"].to_numpy(), "p": scores["p2"].to_numpy()})
    assigned = DC.assign(pairs, "p")
    cfg = M["decode"]
    if cfg["method"] == "threshold":
        sel = DC.select_threshold(assigned, cfg["t"])
    else:
        sel = DC.select_expected_f05(assigned, "p", miss_p=cfg["miss_p"], prob_scale=cfg["scale"])
    ids = rec["entity_id"].to_numpy(dtype=object)
    s1_ids = ids[src == 1]
    matches = {x: [] for x in s1_ids}
    for a, b in zip(ids[sel["s"].to_numpy()], ids[sel["q"].to_numpy()]):
        matches[a].append(b)
    cands = {x: [] for x in s1_ids}
    for a, b in zip(ids[scores["s"].to_numpy()], ids[scores["q"].to_numpy()]):
        cands[a].append(b)
    os.makedirs(args.out_dir, exist_ok=True)
    write_id_lists(os.path.join(args.out_dir, "matching_results.tsv"), ("source1_entity_id", "matched_entity_ids"),
                   s1_ids, matches)
    write_id_lists(os.path.join(args.out_dir, "candidate_pairs.tsv"), ("source1_entity_id", "candidate_entity_ids"),
                   s1_ids, cands)
    n_nonempty = sum(1 for v in matches.values() if v)
    log.info("decoding rule: %s", {k: v for k, v in cfg.items() if k != "oof"})
    log.info("wrote %d S1 rows (%d with matches, %d matched ids); %d candidate pairs", len(s1_ids), n_nonempty,
             len(sel), len(scores))
    if "country" in rec:  # label-free sanity check per country
        s1_rows = np.nonzero(src == 1)[0]
        n_sel = np.bincount(sel["s"].to_numpy(), minlength=len(rec))[s1_rows]
        c1 = rec["country"].astype(str).to_numpy()[s1_rows]
        for c in pd.unique(c1):
            m = c1 == c
            log.info("  %s: %d S1, %.3f matches per S1, %.2f%% empty", c, m.sum(), n_sel[m].mean(),
                     100 * (n_sel[m] == 0).mean())
    oof_path = os.path.join(args.work_dir, "train", "oof.parquet")
    if os.path.exists(oof_path):  # label-free: EM estimate of the test pair prior vs the training prior
        pi_tr = float(pd.read_parquet(oof_path, columns=["y"])["y"].mean())
        log.info("pair prior: train %.4f, test EM estimate %.4f", pi_tr, DC.em_prior(pairs["p"].to_numpy(), pi_tr))
    run_validator(args.out_dir, os.path.join(args.data_dir, "test"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "predict", "all"])
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--work-dir", required=True, help="scratch folder for cached intermediate files")
    ap.add_argument("--out-dir", default="output", help="where matching_results.tsv / candidate_pairs.tsv go")
    ap.add_argument("--art-dir", default=None, help="model artefacts (default: <work-dir>/artifacts)")
    ap.add_argument("--n-jobs", type=int, default=n_jobs_default())
    ap.add_argument("--K", type=int, default=50, help="top-K S1 retrieved per S2/S3 record")
    ap.add_argument("--ratio", type=float, default=0.7, help="keep candidates with score >= ratio * best score")
    ap.add_argument("--keep-recall", type=float, default=0.9995, help="prefilter keeps this share of true pairs")
    ap.add_argument("--pre-sample", type=float, default=0.5, help="share of S1 groups used to fit the prefilter")
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5, help="model folds (split by S1); test averages the fold models")
    ap.add_argument("--drop-features", default="s_indeg,s_indeg_top1,s_rrank,s_n_cos_rank",
                    help="comma-separated feature names left out of both matchers (default: the per-S1 competition "
                         "counts, which drift with the test's extra distractors; they stay in the prefilter)")
    ap.add_argument("--lgb", default="", help="JSON overrides of the matcher LightGBM params")
    ap.add_argument("--decode-on", choices=["oof", "testlike"], default="oof",
                    help="tune the decoding rule on the OOF scores or on the test-like validation")
    ap.add_argument("--key-channel", action="store_true",
                    help="add candidates sharing a rare canonical house number (blocking recall experiment)")
    ap.add_argument("--lo-dropout", type=float, default=0.4,
                    help="share of S1 groups whose supervised token log-odds features are blanked in training")
    ap.add_argument("--surface", action=argparse.BooleanOptionalAction, default=True,
                    help="use the raw-surface name/address noise features (case, accents, spaces, punctuation, raw "
                         "equality modulo legal form) in both matchers (v3 default; --no-surface = v2)")
    ap.add_argument("--roles", action=argparse.BooleanOptionalAction, default=True,
                    help="v4: transductive per-country word-role features (share of same-address vs +1..25-shifted "
                         "pairs among name changes involving each token; label-free); --no-roles = v3")
    ap.add_argument("--cross-roles", action=argparse.BooleanOptionalAction, default=True,
                    help="v6: also score every changed token in the other contexts (addition vs replacement, "
                         "deletion vs replacement); --no-cross-roles = v5")
    ap.add_argument("--filler-rule", action=argparse.BooleanOptionalAction, default=True,
                    help="v7 (predict): pairs at the S1's same address whose query swaps in or appends one word that "
                         "behaves like a noise filler in this split (keeps the legal form like pure noise; "
                         "src/fillers.py) get p = max(p, 0.95) before decoding; --no-filler-rule = v6")
    ap.add_argument("--vocab-cap", action=argparse.BooleanOptionalAction, default=False,
                    help="v8b (predict): same-address swaps to a word whose same-address swaps redraw the legal form "
                         "(>= 12%%, distractor vocabulary of its country; src/fillers.py) get p = min(p, 0.05)")
    ap.add_argument("--collective-counts", action="store_true",
                    help="stage 2 also uses the E6 distractor counts (c_name_lo, c_empty_hi)")
    ap.add_argument("--aug-target", type=float, default=0.0,
                    help="train the matchers on folds augmented like the test-like validation: copies of unmatched "
                         "training queries until this share of queries is unmatched (e.g. 0.40); 0 = off")
    ap.add_argument("--aug-stages", choices=["2", "12"], default="2",
                    help="augment stage 2 only, or stage 1 and stage 2")
    ap.add_argument("--aug-seed", type=int, default=1, help="seed of the training copies (the test-like set uses 0)")
    ap.add_argument("--neg-weight", type=float, default=0.9,
                    help="extra training weight on negatives of queries that match no S1, which are more frequent on "
                         "test (v3 default 0.9; 0 = v2)")
    ap.add_argument("--ce-train", default=None,
                    help="v9 (train): out-of-fold character cross-encoder scores of the training pairs "
                         "(scripts/v8_ce.py ce_train.parquet) added as matcher features")
    ap.add_argument("--ce-dropout", type=float, default=0.3,
                    help="v9: share of S1 groups whose cross-encoder features are blanked in training")
    ap.add_argument("--ce-test", default=None,
                    help="v9 (predict): cross-encoder scores of the test pairs (scripts/v8_ce.py ce_test.parquet)")
    ap.add_argument("--stop-after", choices=["features"], default=None,
                    help="train: stop once the pair features (and prefilter) are cached")
    ap.add_argument("--force", action="store_true", help="recompute every cached file, including retrieval")
    ap.add_argument("--redo", choices=REDO_ORDER, default=None,
                    help="recompute this stage and everything after it (retrieval stays cached); "
                         "default: reuse records/candidates/features, retrain models, re-decode")
    args = ap.parse_args()
    args.art_dir = args.art_dir or os.path.join(args.work_dir, "artifacts")
    setup_logging()
    os.makedirs(args.work_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.work_dir, f"run_{args.stage}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info("args: %s", vars(args))
    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage in ("predict", "all"):
        stage_predict(args)


if __name__ == "__main__":
    main()
