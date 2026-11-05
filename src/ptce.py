"""Pretrained pair cross-encoder (v10): a multilingual transformer fine-tuned as a match / no-match classifier on the
uncertain candidate pairs.

What: the query record and the Source-1 record are read together as one sequence
``<cls> query name | query address <sep> S1 name | S1 address <sep>`` (the model's own pair format) and a one-logit
head is trained with a weighted binary cross-entropy. Only the pairs the v8 stack is unsure about (its score inside a
band, chosen on the training OOF and applied unchanged to the test pairs; ``scripts/v10_export.py``) are read, so the
model spends all of its capacity on the decisions that can still change. Pretraining gives it what the from-scratch
character model (``charce.py``) lacks: sub-word knowledge of names, places, spelling variants and scripts
(Latin / Devanagari / other Indic) in many languages.

How it is used (``scripts/v10_pt.py``, ``scripts/v10_stack.py``): two fold models give every band pair of the
training split an out-of-fold score. The folds keep each query record with all its candidate pairs (and each
true-match cluster) together, so a fold model never saw the query of a pair it scores, as on the test split. Test band
pairs get the mean logit of the fold models. A stage-4 LightGBM stack combines it with the v8 scores.

Licence / rules: the pretrained weights below are MIT-licensed (model cards checked), have <= 0.6B parameters, are
downloaded once from the Hugging Face hub at the pinned revision and run locally; fine-tuning uses the provided
training pairs only. Nothing is sent anywhere.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

log = logging.getLogger("er")

# preset -> (hub name, pinned revision, licence, parameters)
MODELS = {
    "xlmr": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", "mit", 278.9e6),
    "e5": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", "mit", 278.0e6),
    "xlmr-large": ("FacebookAI/xlm-roberta-large", "c23d21b0620b635a76227c604d44e43a9f0ee389", "mit", 561.2e6),
}
REC_SEP = " |"          # between a record's name and its address (plain text, not a special token)


def _torch():
    import torch
    return torch


def load_tokenizer(name: str, revision: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise SystemExit(f"{name}: no fast tokenizer available")
    return tok


def load_model(name: str, revision: str):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision, num_labels=1)
    return model.float()  # fp32 master weights; bf16 only inside autocast


def pair_format(tok):
    """(prefix, middle, suffix) special-token ids of the model's pair format, e.g. XLM-R ``<s> A </s></s> B </s>``."""
    try:
        ids = list(tok.build_inputs_with_special_tokens([-11], [-22]))
        ia, ib = ids.index(-11), ids.index(-22)
        return ids[:ia], ids[ia + 1:ib], ids[ib + 1:]
    except (AttributeError, NotImplementedError, ValueError, TypeError):
        pass
    xa = tok("alpha", add_special_tokens=False)["input_ids"]
    xb = tok("omega", add_special_tokens=False)["input_ids"]
    ids = list(tok("alpha", "omega")["input_ids"])
    ia = next(i for i in range(len(ids)) if ids[i:i + len(xa)] == xa)
    ib = next(i for i in range(ia + len(xa), len(ids)) if ids[i:i + len(xb)] == xb)
    return ids[:ia], ids[ia + len(xa):ib], ids[ib + len(xb):]


def tokenize_records(tok, names, addrs, ln: int = 32, la: int = 48, chunk: int = 200_000):
    """Token ids of ``name | address`` per record: (n, L) int32 matrix (pad 0) and lengths. The name and the address
    are truncated separately (``ln`` / ``la`` tokens), so a long name never pushes the address out."""
    sep = tok(REC_SEP, add_special_tokens=False)["input_ids"]
    n = len(names)
    L = ln + len(sep) + la
    out = np.zeros((n, L), dtype=np.int32)
    lens = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for st in range(0, n, chunk):
        nm = [x if isinstance(x, str) else "" for x in names[st:st + chunk]]
        ad = [x if isinstance(x, str) else "" for x in addrs[st:st + chunk]]
        en = tok(nm, add_special_tokens=False, truncation=True, max_length=ln)["input_ids"]
        ea = tok(ad, add_special_tokens=False, truncation=True, max_length=la)["input_ids"]
        for i, (x, y) in enumerate(zip(en, ea)):
            r = x + sep + y if y else x
            out[st + i, :len(r)] = r
            lens[st + i] = len(r)
        log.info("  tokenised %d / %d records (%.0f s)", min(st + chunk, n), n, time.time() - t0)
    return out, lens


class PairBuilder:
    """Assembles padded pair batches ``prefix + query + middle + S1 + suffix`` on the device from the per-record token
    matrix (query and S1 rows index the same matrix)."""

    def __init__(self, R: np.ndarray, lens: np.ndarray, fmt, pad_id: int, device):
        torch = _torch()
        self.torch = torch
        self.R = torch.as_tensor(R.astype(np.int32), device=device)
        self.lens = torch.as_tensor(lens.astype(np.int64), device=device)
        self.pre, self.mid, self.suf = [list(x) for x in fmt]
        self.pad = int(pad_id)
        self.device = device
        self.Lr = R.shape[1]

    def batch(self, q, s):
        torch = self.torch
        Rq, Rs = self.R[q], self.R[s]
        lq, ls = self.lens[q][:, None], self.lens[s][:, None]
        npre, nmid, nsuf = len(self.pre), len(self.mid), len(self.suf)
        T = npre + lq + nmid + ls + nsuf
        Tmax = int(T.max())
        B = q.shape[0]
        j = torch.arange(Tmax, device=self.device)[None, :].expand(B, Tmax)
        out = torch.full((B, Tmax), self.pad, dtype=torch.long, device=self.device)
        a0 = npre
        m = (j >= a0) & (j < a0 + lq)
        out = torch.where(m, torch.gather(Rq, 1, (j - a0).clamp(0, self.Lr - 1)).long(), out)
        b0 = a0 + lq + nmid
        m = (j >= b0) & (j < b0 + ls)
        out = torch.where(m, torch.gather(Rs, 1, (j - b0).clamp(0, self.Lr - 1)).long(), out)
        for t, tid in enumerate(self.pre):
            out[:, t] = tid
        for t, tid in enumerate(self.mid):
            out = torch.where(j == a0 + lq + t, torch.full_like(out, tid), out)
        for t, tid in enumerate(self.suf):
            out = torch.where(j == b0 + ls + t, torch.full_like(out, tid), out)
        return out, (j < T).long()


def _forward(model, ids, mask):
    return model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)


def train(model, pb: PairBuilder, q: np.ndarray, s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
          epochs: float = 1.0, bs: int = 128, lr: float = 2e-5, warmup_frac: float = 0.06, wd: float = 0.01,
          max_minutes: float | None = None, log_every: int = 200, seed: int = 0):
    """AdamW, linear warmup + linear decay, bf16 autocast. With ``max_minutes`` the number of steps is fitted to the
    measured throughput after 100 steps, so the schedule still ends at lr ~ 0 inside the budget."""
    torch = _torch()
    dev = pb.device
    torch.manual_seed(seed)
    model.to(dev).train()
    qt = torch.as_tensor(q.astype(np.int64), device=dev)
    st = torch.as_tensor(s.astype(np.int64), device=dev)
    yt = torch.as_tensor(y.astype(np.float32), device=dev)
    wt = None if w is None else torch.as_tensor(w.astype(np.float32), device=dev)
    n = len(q)
    plan = {"steps": max(1, int(epochs * n) // bs)}
    decay = [p for nme, p in model.named_parameters() if p.ndim >= 2 and "embeddings" not in nme]
    other = [p for nme, p in model.named_parameters() if not (p.ndim >= 2 and "embeddings" not in nme)]
    kw = {"fused": True} if dev.type == "cuda" else {}
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.98), eps=1e-6, **kw)

    def lr_at(k):
        tot = plan["steps"]
        warm = max(10, int(warmup_frac * tot))
        if k < warm:
            return (k + 1) / warm
        return max(0.0, (tot - k) / max(1, tot - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = dev.type == "cuda"
    t0 = time.time()
    k, pos = 0, 0
    run_loss, n_loss = 0.0, 0
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    perm = torch.randperm(n, device=dev, generator=g)
    while k < plan["steps"]:
        if pos + bs > n:
            perm = torch.randperm(n, device=dev, generator=g)
            pos = 0
        idx = perm[pos:pos + bs]
        pos += bs
        ids, mask = pb.batch(qt[idx], st[idx])
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            logit = _forward(model, ids, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.float(), yt[idx], weight=None if wt is None else wt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        k += 1
        if k % 10 == 0 or k == plan["steps"]:
            run_loss += float(loss.detach())
            n_loss += 1
        if max_minutes and not plan.get("fitted") and k == 100:
            plan["fitted"] = True
            rate = k / max(time.time() - t0, 1e-9)
            fit = int(rate * (max_minutes * 60 - (time.time() - t0))) + k
            log.info("  measured %.2f steps/s (%.0f pairs/s): %d steps planned, %d fit in %.0f min", rate, rate * bs,
                     plan["steps"], fit, max_minutes)
            if fit < plan["steps"]:
                plan["steps"] = max(fit, k + 1)
        if k % log_every == 0 or k == plan["steps"]:
            el = time.time() - t0
            log.info("  step %d/%d  loss %.4f  lr %.2e  %.0f pairs/s  %.1f min", k, plan["steps"],
                     run_loss / max(1, n_loss), sched.get_last_lr()[0], k * bs / max(el, 1e-9), el / 60)
            run_loss, n_loss = 0.0, 0
            if not math.isfinite(float(loss.detach())):
                raise SystemExit("training diverged (non-finite loss): lower --lr")
        if max_minutes and (time.time() - t0) / 60 > 1.1 * max_minutes:
            log.info("  stopped at 1.1 x the time budget after %d / %d steps", k, plan["steps"])
            break
    model.eval()
    return model, {"steps": k, "pairs_seen": k * bs, "minutes": round((time.time() - t0) / 60, 1)}


def predict(model, pb: PairBuilder, q: np.ndarray, s: np.ndarray, bs: int = 512) -> np.ndarray:
    """Logits of the pairs (q, s); pairs are processed sorted by length so batches carry little padding."""
    torch = _torch()
    dev = pb.device
    model.to(dev).eval()
    lens = pb.lens.cpu().numpy()
    order = np.argsort(lens[q] + lens[s], kind="stable")
    out = np.empty(len(q), dtype=np.float32)
    use_amp = dev.type == "cuda"
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(q), bs):
            o = order[i:i + bs]
            qb = torch.as_tensor(q[o].astype(np.int64), device=dev)
            sb = torch.as_tensor(s[o].astype(np.int64), device=dev)
            ids, mask = pb.batch(qb, sb)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                logit = _forward(model, ids, mask)
            out[o] = logit.float().cpu().numpy()
            if (i // bs) % 500 == 0 and i:
                log.info("  predict %d / %d  %.0f pairs/s", i, len(q), i / max(time.time() - t0, 1e-9))
    return out
