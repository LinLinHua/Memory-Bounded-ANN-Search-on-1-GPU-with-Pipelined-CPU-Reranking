"""Check that a baseline environment works on the GPU node.

Runs exact search with the library on the first NQ SIFT1M queries and
compares it with the published ground truth (and, for Faiss, with its own
CPU search). Prints CHECK PASSED or exits non-zero.

    python scripts/smoke_baselines.py --lib faiss
    python scripts/smoke_baselines.py --lib cuvs
"""
import argparse
import importlib
import sys
import time

import numpy as np


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].copy().view(dtype)


def recall_at_10(ids, gt):
    return np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(ids, gt)])


def run_faiss(base, query, k):
    import faiss

    print(f"faiss {faiss.__version__}, GPUs visible: {faiss.get_num_gpus()}")
    assert faiss.get_num_gpus() >= 1, "Faiss sees no GPU"

    cpu = faiss.IndexFlatL2(base.shape[1])
    cpu.add(base)
    t = time.perf_counter()
    d_cpu, i_cpu = cpu.search(query, k)
    print(f"CPU IndexFlatL2: {time.perf_counter() - t:.3f} s")

    res = faiss.StandardGpuResources()
    res.noTempMemory()  # no preallocated scratch pool, as in the memory-budget runs
    gpu = faiss.GpuIndexFlatL2(res, base.shape[1])
    gpu.add(base)
    gpu.search(query[:10], k)  # warm-up
    t = time.perf_counter()
    d_gpu, i_gpu = gpu.search(query, k)
    print(f"GPU IndexFlatL2: {time.perf_counter() - t:.3f} s")

    rel = np.max(np.abs(d_gpu - d_cpu) / np.maximum(np.abs(d_cpu), 1e-12))
    print(f"max relative distance error, GPU vs CPU: {rel:.2e}")
    return i_gpu, rel <= 1e-4


def run_cuvs(base, query, k):
    import cupy as cp
    import cuvs
    import rmm
    from cuvs.neighbors import brute_force

    print(f"cuvs {cuvs.__version__}, rmm {rmm.__version__}, cupy {cp.__version__}")
    # Plain cudaMalloc instead of a pool, so measured device memory is real usage.
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

    available = []
    for name in ["brute_force", "ivf_flat", "ivf_pq", "cagra", "refine", "ivf_rabitq"]:
        try:
            importlib.import_module(f"cuvs.neighbors.{name}")
            available.append(name)
        except ImportError:
            pass
    print("cuvs.neighbors modules:", ", ".join(available))
    if "ivf_rabitq" not in available:
        print("NOTE: no cuvs.neighbors.ivf_rabitq in this version")

    base_d = cp.asarray(base)
    query_d = cp.asarray(query)
    index = brute_force.build(base_d, metric="sqeuclidean")
    brute_force.search(index, query_d[:10], k)  # warm-up
    cp.cuda.runtime.deviceSynchronize()
    t = time.perf_counter()
    _, ids = brute_force.search(index, query_d, k)
    cp.cuda.runtime.deviceSynchronize()
    print(f"GPU brute_force: {time.perf_counter() - t:.3f} s")
    return cp.asnumpy(cp.asarray(ids)), True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", choices=["faiss", "cuvs"], required=True)
    p.add_argument("--data", default="data/sift")
    p.add_argument("--nq", type=int, default=1000)
    args = p.parse_args()

    base = read_vecs(f"{args.data}/sift_base.fvecs", np.float32)
    query = read_vecs(f"{args.data}/sift_query.fvecs", np.float32)[: args.nq]
    gt = read_vecs(f"{args.data}/sift_groundtruth.ivecs", np.int32)[: args.nq]
    print(f"base {base.shape}, queries {query.shape}")

    run = run_faiss if args.lib == "faiss" else run_cuvs
    ids, ok = run(base, query, 10)
    r = recall_at_10(ids, gt)
    print(f"Recall@10 vs published ground truth: {r:.5f}")
    # Ties in the SIFT1M ground truth keep exact search slightly below 1.0
    # (our own GPU brute force gets 0.99944 on all 10,000 queries).
    ok = ok and r >= 0.999
    print("CHECK PASSED" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
