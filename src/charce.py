"""Character-level pair cross-encoder (v8): an extra, independent view of each candidate pair.

What: a small transformer (4 layers, d=256, ~3.3M parameters) trained FROM SCRATCH on the training candidate pairs
(no pretrained weights, no external data, no download). It reads the raw characters of both records -- query name,
query address, S1 name, S1 address, as four segments of one sequence with segment-relative positions -- and
predicts whether the pair is a match. Raw characters keep what the normalised pair features fold away: which letter
changed (a vowel swap vs a dropped letter vs an OCR confusion), duplicated letters, case / accent / spacing noise,
and the exact digits of house / plot / flat numbers.

How it is used (``scripts/v8_ce.py``, ``src/stack.py``): two fold models split by Source-1 entity give every training
pair an out-of-fold score; test pairs get the mean of the two models. A stage-3 LightGBM then combines the pipeline's
stage-2 score with this score (``stack.py``).

Licence / rules: PyTorch (BSD-3) is the only new dependency; the model is trained here on the provided training data.
"""
from __future__ import annotations

import collections
import logging
import math
import time

import numpy as np

log = logging.getLogger("er")

LN, LA = 40, 64                    # characters kept from the name / the address
PAD, CLS, SEP, UNK, UNK_INDIC = 0, 1, 2, 3, 4
N_SPECIAL = 5
VOCAB = 256                        # ids fit in uint8


def _is_indic(cp: int) -> bool:
    return 0x0900 <= cp <= 0x0DFF


def build_lut(texts, min_count: int = 20) -> np.ndarray:
    """Code point -> id table (uint8). The most frequent characters of ``texts`` get their own id; other Indic-script
    characters share one id, everything else another."""
    cnt = collections.Counter()
    for t in texts:
        if t:
            cnt.update(t)
    chars = [c for c, n in cnt.most_common() if n >= min_count and not _is_indic(ord(c))]
    chars = chars[:VOCAB - N_SPECIAL]
    lut = np.full(0x110000, UNK, dtype=np.uint8)
    lut[0x0900:0x0E00] = UNK_INDIC
    for i, c in enumerate(chars):
        lut[ord(c)] = N_SPECIAL + i
    log.info("char vocabulary: %d characters (+%d special ids)", len(chars), N_SPECIAL)
    return lut


def encode(texts, L: int, lut: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    """(n, L) uint8 matrix of character ids (first L characters, PAD after the end)."""
    n = len(texts)
    out = np.zeros((n, L), dtype=np.uint8)
    for st in range(0, n, chunk):
        part = [(t if isinstance(t, str) else "")[:L] for t in texts[st:st + chunk]]
        lens = np.fromiter(map(len, part), dtype=np.int64, count=len(part))
        tot = int(lens.sum())
        if tot == 0:
            continue
        cps = np.frombuffer("".join(part).encode("utf-32-le"), dtype=np.uint32)
        assert len(cps) == tot
        ids = lut[cps]
        rows = np.repeat(np.arange(len(part)), lens)
        starts = np.cumsum(lens) - lens
        cols = np.arange(tot) - np.repeat(starts, lens)
        out[st + rows, cols] = ids
    return out


def _torch():
    import torch  # imported lazily: the rest of the pipeline does not need torch
    return torch


NGRAM_BUCKETS = 1 << 17          # hashed character bigrams / trigrams (local context from the first layer on)


def build_model(d: int = 256, layers: int = 4, heads: int = 8, ff: int = 1024, drop: float = 0.1,
                ngrams: bool = True):
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class PairCE(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, d, padding_idx=PAD)
            self.use_ngrams = ngrams
            if ngrams:
                self.bi = nn.Embedding(NGRAM_BUCKETS, d)
                self.tri = nn.Embedding(NGRAM_BUCKETS, d)
                nn.init.normal_(self.bi.weight, std=0.02)
                nn.init.normal_(self.tri.weight, std=0.02)
            self.seg = nn.Embedding(5, d)
            self.pos = nn.Embedding(max(LN, LA) + 2, d)
            layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
            seg = [0] + [1] * (LN + 1) + [2] * (LA + 1) + [3] * (LN + 1) + [4] * (LA + 1)
            pos = ([0] + list(range(1, LN + 2)) + list(range(1, LA + 2)) + list(range(1, LN + 2))
                   + list(range(1, LA + 2)))
            self.register_buffer("seg_ids", torch.tensor(seg, dtype=torch.long)[None], persistent=False)
            self.register_buffer("pos_ids", torch.tensor(pos, dtype=torch.long)[None], persistent=False)

        def forward(self, x):
            h = self.tok(x) + self.seg(self.seg_ids) + self.pos(self.pos_ids)
            if self.use_ngrams:
                xl = F.pad(x[:, :-1], (1, 0))
                xr = F.pad(x[:, 1:], (0, 1))
                h = h + self.bi((xl * 1031 + x * 7) % NGRAM_BUCKETS) \
                      + self.tri((xl * 262147 + x * 1031 + xr * 7 + 3) % NGRAM_BUCKETS)
            h = self.enc(h, src_key_padding_mask=(x == PAD))
            return self.head(self.norm(h[:, 0])).squeeze(-1)

    return PairCE()


class PairSource:
    """Builds token batches on the device from per-record code matrices (query rows and S1 rows index the same
    record table)."""

    def __init__(self, names: np.ndarray, addrs: np.ndarray, device):
        torch = _torch()
        self.torch = torch
        self.n = torch.as_tensor(names, device=device)
        self.a = torch.as_tensor(addrs, device=device)
        self.device = device

    def batch(self, q, s):
        torch = self.torch
        B = q.shape[0]
        cls = torch.full((B, 1), CLS, dtype=torch.uint8, device=self.device)
        sep = torch.full((B, 1), SEP, dtype=torch.uint8, device=self.device)
        x = torch.cat([cls, self.n[q], sep, self.a[q], sep, self.n[s], sep, self.a[s], sep], dim=1)
        return x.long()


def train(model, src: PairSource, q: np.ndarray, s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
          epochs: float = 1.0, bs: int = 1024, lr: float = 5e-4, warmup: int = 1000, wd: float = 0.01,
          amp: bool = True, log_every: int = 500, max_minutes: float | None = None, seed: int = 0):
    """One (or a fraction of an) epoch of AdamW with warmup + cosine decay. Returns the trained model."""
    torch = _torch()
    dev = src.device
    torch.manual_seed(seed)
    model.to(dev).train()
    qt = torch.as_tensor(q.astype(np.int64), device=dev)
    st = torch.as_tensor(s.astype(np.int64), device=dev)
    yt = torch.as_tensor(y.astype(np.float32), device=dev)
    wt = None if w is None else torch.as_tensor(w.astype(np.float32), device=dev)
    n = len(q)
    plan = {"steps": max(1, int(epochs * n) // bs)}   # total steps; shortened once the throughput is known when a
    steps = plan["steps"]                              # time budget (max_minutes) is set, so the cosine still ends
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98), eps=1e-6)

    warmup = max(10, min(warmup, steps // 10))

    def lr_at(k):
        warm = min(1.0, (k + 1) / warmup)
        tot = plan["steps"]
        return warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(k, tot) / tot)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = amp and dev.type == "cuda"
    t0 = time.time()
    k = 0
    run_loss, n_loss = 0.0, 0
    perm = torch.randperm(n, device=dev)
    pos = 0
    while k < plan["steps"]:
        if pos + bs > n:
            perm = torch.randperm(n, device=dev)
            pos = 0
        idx = perm[pos:pos + bs]
        pos += bs
        xb = src.batch(qt[idx], st[idx])
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            logit = model(xb)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.float(), yt[idx], weight=None if wt is None else wt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        k += 1
        if k % 20 == 0:
            run_loss += float(loss.detach())
            n_loss += 1
        if max_minutes is not None and not plan.get("fitted") and (
                k == 200 or (k % 20 == 0 and time.time() - t0 > 6 * max_minutes)):
            plan["fitted"] = True
            rate = k / max(time.time() - t0, 1e-9)                  # steps per second
            fit = int(rate * max_minutes * 60)
            if fit < plan["steps"]:
                log.info("  ce time budget %.0f min at %.1f steps/s: %d of %d planned steps (%.2f epochs)",
                         max_minutes, rate, fit, plan["steps"], fit * bs / n)
                plan["steps"] = max(fit, k + 1)
        if k % log_every == 0 or k == plan["steps"]:
            el = time.time() - t0
            log.info("  ce step %d/%d  train loss %.4f  lr %.2e  %.0f pairs/s  %.1f min", k, plan["steps"],
                     run_loss / max(1, n_loss), sched.get_last_lr()[0], k * bs / max(el, 1e-9), el / 60)
            run_loss, n_loss = 0.0, 0
        if max_minutes is not None and (time.time() - t0) / 60 > 1.15 * max_minutes:
            log.info("  ce training stopped at 1.15 x the time budget after %d / %d steps", k, plan["steps"])
            break
    model.eval()
    return model


def predict(model, src: PairSource, q: np.ndarray, s: np.ndarray, bs: int = 4096, amp: bool = True) -> np.ndarray:
    torch = _torch()
    dev = src.device
    model.to(dev).eval()
    out = np.empty(len(q), dtype=np.float32)
    use_amp = amp and dev.type == "cuda"
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(q), bs):
            qb = torch.as_tensor(q[i:i + bs].astype(np.int64), device=dev)
            sb = torch.as_tensor(s[i:i + bs].astype(np.int64), device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                logit = model(src.batch(qb, sb))
            out[i:i + bs] = torch.sigmoid(logit.float()).cpu().numpy()
            if (i // bs) % 1000 == 0 and i:
                log.info("  ce predict %d / %d  %.0f pairs/s", i, len(q), i / max(time.time() - t0, 1e-9))
    return out
