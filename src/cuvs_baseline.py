#!/usr/bin/env python3
"""cuVS baseline for LL's Day 11-12 comparison (IVF-Flat and IVF-PQ on the GPU).

Both libraries do the same job as our GPU stage: return C candidates per query. Recall is
computed after the same exact rerank (NumPy, not timed), so the numbers line up with bench_scan.

  ivf_flat : FP32 vectors in the lists (4 * d bytes per vector), the full-precision reference
  ivf_pq   : pq_dim = d / 2 sub-vectors of 8 bits (d / 2 bytes per vector, same size as our 4-bit)

Usage:
  cuvs_baseline.py --base B --query Q --gt G --nlist 1024 [--nprobe 16,32,64] [--C 100,1000]
                   [--algo ivf_pq,ivf_flat] [--reps 5] [--csv results/cuvs_sift.csv]
"""
import argparse
import csv
import os
import time

import cupy as cp
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
    p.add_argument("--csv", default="")
    a = p.parse_args()

    base = load(a.base, np.float32)
    queries = load(a.query, np.float32)
    gt = load(a.gt, np.int32)[: len(queries)]
    n, d = base.shape
    print(f"base {n} x {d}, queries {len(queries)}, nlist {a.nlist}")
    base_gpu, queries_gpu = cp.asarray(base), cp.asarray(queries)

    rows = []
    for algo in a.algo.split(","):
        t0 = time.perf_counter()
        if algo == "ivf_pq":
            module = ivf_pq
            index = ivf_pq.build(ivf_pq.IndexParams(n_lists=a.nlist, metric="sqeuclidean",
                                                    pq_dim=d // 2, pq_bits=8), base_gpu)
            index_mb = n * (d // 2) / 2**20
        else:
            module = ivf_flat
            index = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.nlist, metric="sqeuclidean"), base_gpu)
            index_mb = n * d * 4 / 2**20
        cp.cuda.Device().synchronize()
        build_s = time.perf_counter() - t0
        print(f"=== {algo}: build {build_s:.1f} s, codes {index_mb:.0f} MB")

        for nprobe in map(int, a.nprobe.split(",")):
            for C in map(int, a.C.split(",")):
                params = module.SearchParams(n_probes=nprobe)
                (dist, ids), ms = timed_search(lambda: module.search(params, index, queries_gpu, C), a.reps)
                cand = cp.asnumpy(cp.asarray(ids)).astype(np.int64)
                no_rerank = recall_at_10(cand, gt)
                reranked = recall_at_10(exact_rerank(base, queries, cand), gt)
                qps = len(queries) / (ms / 1e3)
                print(f"nprobe={nprobe:<3d} C={C:<4d} search_ms={ms:7.1f} qps={qps:8.0f}  "
                      f"R@10 no_rerank={no_rerank:.4f} rerank={reranked:.4f}")
                rows.append(dict(algo=algo, nlist=a.nlist, nprobe=nprobe, C=C, nq=len(queries),
                                 search_ms=round(ms, 3), qps=round(qps, 1), index_mb=round(index_mb, 1),
                                 build_s=round(build_s, 2), recall10_no_rerank=round(no_rerank, 5),
                                 recall10_rerank=round(reranked, 5)))

    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
