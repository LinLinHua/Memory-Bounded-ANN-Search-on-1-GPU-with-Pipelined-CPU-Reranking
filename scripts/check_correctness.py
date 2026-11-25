#!/usr/bin/env python3
"""Ground-truth checks against Faiss (HS), following the proposal's correctness rules.

For one dataset, Faiss exact search (GpuIndexFlatL2, every query, top-K) is compared with
each ground-truth file given with --gt (the published SIFT1M answers, and the files our GPU
brute force writes with --out). Every query must have the same top-k set for k = 10 and
k = K, except where the lists differ only by ties: the k-th distances agree within a relative
1e-4, and every ID in one list but not the other lies within 1e-4 of that k-th distance.
Distances are recomputed in float64 on the CPU. Faiss's own FP32 distances must also agree
with them within a relative 1e-4.

Prints PASS or FAIL per check and exits non-zero if any check fails.

  python scripts/check_correctness.py --base B --query Q --gt name=path [--gt name=path ...]
"""
import argparse
import sys
import time

import faiss
import numpy as np

TOL = 1e-4


def load(path, dtype):
    """Reads .fvecs/.ivecs or .fbin/.ibin into a NumPy array."""
    if path.endswith("vecs"):
        raw = np.fromfile(path, dtype=np.int32)
        d = raw[0]
        return raw.reshape(-1, d + 1)[:, 1:].view(dtype).copy()
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.fromfile(path, dtype=dtype, offset=8, count=int(n) * int(d)).reshape(n, d)


def exact_dist(base, q, ids):
    diff = base[ids].astype(np.float64) - q.astype(np.float64)
    return np.einsum("ij,ij->i", diff, diff)


def compare(base, queries, a, b, k):
    """Counts queries whose top-k sets are equal, differ only by ties, or really differ."""
    same = ties = bad = 0
    worst = None
    for qi in range(len(queries)):
        sa, sb = a[qi, :k], b[qi, :k]
        if set(sa) == set(sb):
            same += 1
            continue
        da, db = exact_dist(base, queries[qi], sa), exact_dist(base, queries[qi], sb)
        kth_a, kth_b = da.max(), db.max()
        scale = max(kth_a, kth_b, 1e-30)
        only = np.array(sorted(set(sa) ^ set(sb)), dtype=np.int64)
        d_only = exact_dist(base, queries[qi], only)
        if abs(kth_a - kth_b) <= TOL * scale and np.all(np.abs(d_only - scale) <= TOL * scale):
            ties += 1
        else:
            bad += 1
            if worst is None:
                worst = (qi, sorted(set(sa) - set(sb))[:3], sorted(set(sb) - set(sa))[:3])
    return same, ties, bad, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--gt", action="append", required=True, help="name=path, may repeat")
    p.add_argument("--k", type=int, default=100, help="depth compared (also k = 10)")
    a = p.parse_args()

    base = load(a.base, np.float32)
    queries = load(a.query, np.float32)
    n, d = base.shape
    print(f"faiss {faiss.__version__}, base {n} x {d}, queries {len(queries)}")

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)
    index.add(base)
    t0 = time.perf_counter()
    dist, ids = index.search(queries, a.k)
    print(f"Faiss GpuIndexFlatL2 top-{a.k}: {time.perf_counter() - t0:.2f} s")
    del index

    ok_all = True

    def report(name, ok, detail):
        nonlocal ok_all
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    # Faiss's own distances against float64.
    rel = 0.0
    for qi in range(len(queries)):
        ref = exact_dist(base, queries[qi], ids[qi, :10])
        rel = max(rel, float(np.max(np.abs(dist[qi, :10] - ref) / np.maximum(ref, 1e-30))))
    report("Faiss distances vs float64", rel <= TOL, f"max relative error {rel:.2e}")

    for spec in a.gt:
        name, path = spec.split("=", 1)
        gt = load(path, np.int32).astype(np.int64)[: len(queries)]
        for k in sorted({10, min(a.k, gt.shape[1])}):
            same, ties, bad, worst = compare(base, queries, ids, gt, k)
            detail = f"top-{k}: {same} identical, {ties} differ only by ties, {bad} differ"
            if worst:
                detail += f" (first: query {worst[0]}, Faiss only {worst[1]}, {name} only {worst[2]})"
            report(f"Faiss vs {name}", bad == 0, detail)

    print("ALL CORRECTNESS CHECKS PASSED" if ok_all else "SOME CORRECTNESS CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
