#!/usr/bin/env python3
"""cuVS baseline for LL's Day 11-12 comparison (IVF-Flat and IVF-PQ on the GPU).

Both libraries do the same job as our GPU stage: return C candidates per query. Recall is
computed after the same exact rerank (NumPy, not timed), so the numbers line up with bench_scan.

  ivf_flat  : FP32 vectors in the lists (4 * d bytes per vector), the full-precision reference
  ivf_pq    : pq_dim = d / 2 sub-vectors of 8 bits (d / 2 bytes per vector, same size as our 4-bit)
  ivf_pq_d4 : pq_dim = d / 4 (d / 4 bytes per vector, about our 1-bit index size)
  ivf_pq_d8 : pq_dim = d / 8 (d / 8 bytes per vector)

Timing follows docs/measurement.md: one warm-up, then the median of --reps searches of all
queries, each starting from queries in pinned host memory and ending with ids and distances back
on the host (--transfer 1, the default), in calls of --batch queries.
codes_mb is n x bytes per vector. Device memory follows docs/measurement.md (cudaMemGetInfo,
cupy pool emptied, above a baseline taken after a throwaway warm-up index is built and
searched): gpu_index_mb is what the index holds after build, gpu_peak_mb the most seen during one
extra, untimed search, sampled every 0.1 ms from another thread.

  --batch B      queries per search call (0 = all at once, the default)
  --transfer 0   queries already on the GPU and results left there (the timing before Nov 25)

Usage:
  cuvs_baseline.py --base B --query Q --gt G --nlist 1024 [--nprobe 16,32,64] [--C 100,1000]
                   [--algo ivf_pq,ivf_pq_d4,ivf_flat] [--reps 5] [--batch 0] [--transfer 1]
                   [--csv results/cuvs_sift.csv]
"""
import argparse
import csv
import ctypes
import gc
import os
import threading
import time

import cupy as cp
import cupyx
import numpy as np
from cuvs.neighbors import ivf_flat, ivf_pq


def load(path, dtype):
    """Reads .fvecs/.ivecs or .fbin/.ibin into a NumPy array."""
    if path.endswith("vecs"):
        raw = np.fromfile(path, dtype=np.int32)
        d = raw[0]
        return raw.reshape(-1, d + 1)[:, 1:].view(dtype).copy()
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.fromfile(path, dtype=dtype, offset=8, count=int(n) * int(d)).reshape(n, d)


def recall_at_10(ids, gt):
    return np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(ids, gt)])


def exact_rerank(base, queries, cand, k=10):
    """Exact distances for each query's candidates, then the k closest (CPU, not timed)."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for i, (q, c) in enumerate(zip(queries, cand)):
        c = c[c >= 0]
        dist = ((base[c] - q) ** 2).sum(1)
        out[i] = c[np.argsort(dist)[:k]]
    return out


def timed_search(search, reps):
    """One warm-up search, then the median wall time of reps searches (ms).
    The searches sit inside an NVTX range named "search", so Nsight can profile only them."""
    cp.cuda.nvtx.RangePush("search")
    result = search()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = search()
        cp.cuda.Device().synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    cp.cuda.nvtx.RangePop()
    return result, float(np.median(times))


def used_mb():
    """Device memory in use on this GPU (whole device, so one job per GPU)."""
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) / 2**20


_cudart = None


def peak_used_mb(fn):
    """Runs fn() while another thread samples device memory every 0.1 ms; returns the highest
    reading (MB) and the number of samples. The sampler calls cudaMemGetInfo through ctypes,
    which releases the GIL, and cuVS releases it during search."""
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so.12")
    def read():
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        _cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return (total.value - free.value) / 2**20
    peak, count, done = [read()], [0], threading.Event()

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], read())
            count[0] += 1
            time.sleep(1e-4)

    t = threading.Thread(target=sample)
    t.start()
    try:
        fn()
        cp.cuda.Device().synchronize()
    finally:
        done.set()
        t.join()
    return max(peak[0], read()), count[0]


def batched_search(module, params, index, queries, C, batch, transfer):
    """Searches `queries` in calls of `batch` queries (0 = one call). With transfer, `queries` is
    pinned host memory: each batch is uploaded, searched, and its results are downloaded."""
    nq = len(queries)
    step = nq if batch <= 0 else batch
    ids_out, dist_out = [], []
    for i in range(0, nq, step):
        q = cp.asarray(queries[i:i + step]) if transfer else queries[i:i + step]
        dist, ids = module.search(params, index, q, C)
        if transfer:
            ids_out.append(cp.asnumpy(cp.asarray(ids)))
            dist_out.append(cp.asnumpy(cp.asarray(dist)))
        else:
            ids_out.append(cp.asarray(ids))
            dist_out.append(cp.asarray(dist))
    if len(ids_out) == 1:
        return dist_out[0], ids_out[0]
    cat = np.concatenate if transfer else cp.concatenate
    return cat(dist_out), cat(ids_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", default="16,32,64")
    p.add_argument("--C", default="100,1000")
    p.add_argument("--algo", default="ivf_pq,ivf_flat")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--batch", type=int, default=0, help="queries per search call (0 = all)")
    p.add_argument("--transfer", type=int, default=1, help="0 = queries and results stay on the GPU")
    p.add_argument("--csv", default="")
    a = p.parse_args()

    base = load(a.base, np.float32)
    queries = load(a.query, np.float32)
    gt = load(a.gt, np.int32)[: len(queries)]
    n, d = base.shape
    print(f"base {n} x {d}, queries {len(queries)}, nlist {a.nlist}")
    base_gpu, queries_gpu = cp.asarray(base), cp.asarray(queries)
    queries_pinned = cupyx.empty_pinned(queries.shape, dtype=np.float32)
    queries_pinned[...] = queries
    search_queries = queries_pinned if a.transfer else queries_gpu
    print(f"batch {a.batch or 'all'}, transfer timed: {bool(a.transfer)}")

    pq_div = {"ivf_pq": 2, "ivf_pq_d4": 4, "ivf_pq_d8": 8}

    # Throwaway warm-up: a small index built and searched once, so cuVS's one-time setup is in
    # the memory baseline and not in the first index.
    wn = min(n, 100_000)
    warm = ivf_flat.build(ivf_flat.IndexParams(n_lists=max(1, min(a.nlist, wn // 32)), metric="sqeuclidean"),
                          base_gpu[:wn])
    ivf_flat.search(ivf_flat.SearchParams(n_probes=16), warm, queries_gpu[:128], 100)
    cp.cuda.Device().synchronize()
    del warm
    gc.collect()

    rows = []
    for algo in a.algo.split(","):
        if algo not in pq_div and algo != "ivf_flat":
            p.error(f"unknown algo {algo}; expected ivf_flat or one of {','.join(pq_div)}")
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
        before = used_mb()
        t0 = time.perf_counter()
        if algo in pq_div:
            module = ivf_pq
            index = ivf_pq.build(ivf_pq.IndexParams(n_lists=a.nlist, metric="sqeuclidean",
                                                    pq_dim=d // pq_div[algo], pq_bits=8), base_gpu)
            index_mb = n * (d // pq_div[algo]) / 2**20
        else:
            module = ivf_flat
            index = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.nlist, metric="sqeuclidean"), base_gpu)
            index_mb = n * d * 4 / 2**20
        cp.cuda.Device().synchronize()
        build_s = time.perf_counter() - t0
        cp.get_default_memory_pool().free_all_blocks()
        gpu_index_mb = used_mb() - before
        print(f"=== {algo}: build {build_s:.1f} s, codes {index_mb:.0f} MB, on GPU {gpu_index_mb:.0f} MB")

        for nprobe in map(int, a.nprobe.split(",")):
            for C in map(int, a.C.split(",")):
                params = module.SearchParams(n_probes=nprobe)
                (dist, ids), ms = timed_search(
                    lambda: batched_search(module, params, index, search_queries, C, a.batch, a.transfer), a.reps)
                cp.get_default_memory_pool().free_all_blocks()
                peak, samples = peak_used_mb(
                    lambda: batched_search(module, params, index, search_queries, C, a.batch, a.transfer))
                gpu_peak_mb = peak - before
                cand = cp.asnumpy(cp.asarray(ids)).astype(np.int64)
                no_rerank = recall_at_10(cand, gt)
                reranked = recall_at_10(exact_rerank(base, queries, cand), gt)
                qps = len(queries) / (ms / 1e3)
                print(f"nprobe={nprobe:<3d} C={C:<4d} search_ms={ms:7.1f} qps={qps:8.0f}  "
                      f"R@10 no_rerank={no_rerank:.4f} rerank={reranked:.4f}  "
                      f"peak {gpu_peak_mb:.0f} MB ({samples} samples)")
                rows.append(dict(algo=algo, nlist=a.nlist, nprobe=nprobe, C=C, nq=len(queries),
                                 search_ms=round(ms, 3), qps=round(qps, 1), index_mb=round(index_mb, 1),
                                 build_s=round(build_s, 2), recall10_no_rerank=round(no_rerank, 5),
                                 recall10_rerank=round(reranked, 5), batch=a.batch, transfer=a.transfer,
                                 codes_mb=round(index_mb, 1), gpu_index_mb=round(gpu_index_mb, 1),
                                 gpu_peak_mb=round(gpu_peak_mb, 1)))
        del index  # release it before the next index's memory baseline
        gc.collect()

    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
