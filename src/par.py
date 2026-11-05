"""Small multiprocessing helpers (fork on Linux so large read-only globals are shared)."""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import scipy.sparse as sp

_G: dict = {}
_CSR = "__csr_mmap__"
_ARROW = "__arrow_mmap__"


def n_jobs_default() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _init(shared):
    _G.clear()
    for k, v in shared.items():
        if isinstance(v, tuple) and len(v) == 3 and v[0] == _CSR:
            base, shape = v[1], v[2]
            v = sp.csr_matrix((np.load(base + ".data.npy", mmap_mode="r"),
                               np.load(base + ".indices.npy", mmap_mode="r"),
                               np.load(base + ".indptr.npy", mmap_mode="r")), shape=shape, copy=False)
        elif isinstance(v, tuple) and len(v) == 2 and v[0] == _ARROW:
            v = pa.ipc.open_file(pa.memory_map(v[1], "r")).get_batch(0).column(0)
        _G[k] = v


def _to_mmap(shared: dict, tmp: str) -> dict:
    """spawn start method (Windows): large CSR matrices (.npy) and large Arrow arrays (IPC files) go to
    memory-mapped files instead of being pickled into every worker, so all workers share one copy through the OS
    page cache (and the pickled initargs stay small enough for a Windows pipe)."""
    out = {}
    for k, v in shared.items():
        if sp.issparse(v) and v.format == "csr" and v.nnz > 5_000_000:
            base = os.path.join(tmp, k)
            np.save(base + ".data.npy", v.data)
            np.save(base + ".indices.npy", v.indices)
            np.save(base + ".indptr.npy", v.indptr)
            out[k] = (_CSR, base, v.shape)
        elif isinstance(v, pa.Array) and v.nbytes > 50_000_000:  # large string columns: Arrow IPC file, mmapped
            path = os.path.join(tmp, k + ".arrow")
            with pa.OSFile(path, "wb") as f, pa.ipc.new_file(f, pa.schema([("x", v.type)])) as w:
                w.write_batch(pa.record_batch([v], names=["x"]))
            out[k] = (_ARROW, path)
        else:
            out[k] = v
    return out


def G() -> dict:
    """Worker-side access to the shared read-only objects."""
    return _G


def pmap(fn, tasks, n_jobs: int, shared: dict | None = None, chunksize: int = 1):
    """Ordered parallel map of ``fn(task)``; ``shared`` is visible in workers via ``G()``."""
    shared = shared or {}
    tasks = list(tasks)
    if n_jobs <= 1 or len(tasks) <= 1:
        _init(shared)
        return [fn(t) for t in tasks]
    method = 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
    ctx = mp.get_context(method)
    if method == 'fork':
        _init(shared)  # inherited by forked workers without pickling
        with ctx.Pool(n_jobs) as pool:
            return pool.map(fn, tasks, chunksize=chunksize)
    tmp = tempfile.mkdtemp(prefix="er_shared_")
    try:
        with ctx.Pool(n_jobs, initializer=_init, initargs=(_to_mmap(shared, tmp),)) as pool:
            return pool.map(fn, tasks, chunksize=chunksize)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
