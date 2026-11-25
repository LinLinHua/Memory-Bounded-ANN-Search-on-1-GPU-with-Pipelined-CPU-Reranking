#!/usr/bin/env python3
"""Faiss baselines (HS): CPU and GPU IVF indexes on the same data and nlist as our scan.

Follows src/cuvs_baseline.py (LL) so the CSVs can be plotted together: the first eleven
columns are the same, the library returns C candidates per query, and recall is computed
after the same exact rerank (NumPy, not timed).

  cpu_ivf_sq8, cpu_ivf_sq4  multi-threaded CPU IVF with 8- or 4-bit scalar quantization
  gpu_ivf_flat              FP32 vectors in the lists (4 * d bytes per vector)
  gpu_ivf_sq8, gpu_ivf_sq4  8- or 4-bit scalar quantization (same code sizes as ours)
  gpu_ivf_pq                d / 2 sub-vectors of 8 bits (d / 2 bytes, same size as our 4-bit)
  gpu_ivf_pq_d4, _pq_d8     d / 4 or d / 8 sub-vectors (d / 4 or d / 8 bytes), so that PQ
                            also has indexes that fit the 1/8 and 1/16 budgets
  gpu_ivf_pq*_refine        PQ candidates, then exact distances on the CPU with
                            IndexRefineFlat: the closest existing equivalent of our design.
                            Its search time includes the CPU rerank.

GPU indexes store 32-bit IDs, like ours (Faiss defaults to 64-bit; --ids64 restores that).

Every index uses the same coarse quantizer: GPU k-means with a fixed seed, trained once per
process. Within one process the CPU and GPU versions of an encoding hold the same codes (the
GPU index is a copy of the CPU one).

GPU memory follows the proposal: no Faiss temporary-memory pool (--temp-mb 0), and memory is
reported above the CUDA context. gpu_index_mb is the device memory held after the index is
on the GPU (codes, IDs, centroids, Faiss's handles); gpu_peak_mb is the most seen during one
extra, untimed search, sampled with cudaMemGetInfo every 0.1 ms, so very short spikes can be
missed. Run with CUDA_MODULE_LOADING=EAGER (scripts/faiss_baseline_a40.sbatch does): then the
kernels' code is already on the GPU in the baseline, and only allocations are counted.

Rows are appended to --csv as they are measured (delete the file first for a fresh run).
GPU IVF-SQ is skipped for C > 512, where Faiss 1.14.1 aborts (see GPU_SQ_MAX_K).

Usage:
  faiss_baseline.py --base B --query Q --gt G --nlist 1024 [--nprobe 16,32,64] [--C 100,1000]
                    [--algo gpu_ivf_sq8,...] [--reps 5] [--batch 0] [--temp-mb 0]
                    [--csv results/faiss_sift.csv]
"""
import argparse
import csv
import ctypes
import gc
import os
import threading
import time

import faiss
import numpy as np

DEFAULT_ALGOS = ("cpu_ivf_sq8,cpu_ivf_sq4,gpu_ivf_flat,gpu_ivf_sq8,gpu_ivf_sq4,"
                 "gpu_ivf_pq,gpu_ivf_pq_refine,gpu_ivf_pq_d4,gpu_ivf_pq_d4_refine,"
                 "gpu_ivf_pq_d8,gpu_ivf_pq_d8_refine")
MB = 2**20


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


class DeviceMemory:
    """Device memory in use, from cudaMemGetInfo. It counts the whole GPU, so it is only
    meaningful when this process has the GPU to itself (one A40 per Slurm job)."""

    def __init__(self):
        self.cudart = ctypes.CDLL("libcudart.so.12")
        self.cudart.cudaFree(None)  # create the CUDA context before the first reading

    def used_mb(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return (total.value - free.value) / MB

    def peak_during(self, fn):
        """Runs fn() while another thread samples memory; returns the highest reading.
        Faiss releases the GIL during search, so the sampler keeps running."""
        peak, done = [self.used_mb()], threading.Event()

        def sample():
            while not done.is_set():
                peak[0] = max(peak[0], self.used_mb())
                time.sleep(1e-4)

        t = threading.Thread(target=sample)
        t.start()
        try:
            fn()
        finally:
            done.set()
            t.join()
        return max(peak[0], self.used_mb())


def search_batched(index, queries, k, batch):
    """One search call, or several of `batch` queries each (0 = all queries at once)."""
    if batch <= 0 or batch >= len(queries):
        return index.search(queries, k)
    parts = [index.search(queries[i:i + batch], k) for i in range(0, len(queries), batch)]
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def timed_search(search, reps):
    """One warm-up search, then the median wall time of reps searches (ms). Faiss GPU search
    copies results back to the host before returning, so no extra synchronize is needed."""
    result = search()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = search()
        times.append((time.perf_counter() - t0) * 1e3)
    return result, float(np.median(times))


def train_coarse(base, nlist, n_train, seed):
    """GPU k-means on a sample of the base vectors; returns a flat quantizer with the centroids."""
    d = base.shape[1]
    rng = np.random.default_rng(seed)
    sample = base if len(base) <= n_train else base[np.sort(rng.choice(len(base), n_train, replace=False))]
    km = faiss.Kmeans(d, nlist, niter=20, seed=seed, gpu=faiss.get_num_gpus() > 0,
                      max_points_per_centroid=max(256, -(-len(sample) // nlist)))
    km.train(np.ascontiguousarray(sample))
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(km.centroids)
    return quantizer, sample


def build_cpu(encoding, quantizer, base, sample):
    """CPU IVF index with the shared centroids. Training only fits the encoder, because the
    quantizer already holds nlist centroids."""
    d, nlist = base.shape[1], quantizer.ntotal
    if encoding == "flat":
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
    elif encoding in ("sq8", "sq4"):
        qtype = faiss.ScalarQuantizer.QT_8bit if encoding == "sq8" else faiss.ScalarQuantizer.QT_4bit
        index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, qtype)
    elif encoding in PQ_DIVISOR:
        index = faiss.IndexIVFPQ(quantizer, d, nlist, d // PQ_DIVISOR[encoding], 8)
    else:
        raise ValueError(encoding)
    index.train(sample)
    index.add(base)
    return index


def to_gpu(res, cpu_index, encoding, ids64):
    """Copies a CPU IVF index to GPU 0. IVF-PQ first tries FP32 lookup tables and falls back to
    FP16 when they do not fit in shared memory (d / 2 = 64 sub-vectors for SIFT1M)."""
    co = faiss.GpuClonerOptions()
    co.indicesOptions = faiss.INDICES_64_BIT if ids64 else faiss.INDICES_32_BIT
    try:
        return faiss.index_cpu_to_gpu(res, 0, cpu_index, co), False
    except RuntimeError as e:
        if encoding not in PQ_DIVISOR:
            raise
        print(f"  FP32 lookup tables rejected ({str(e).splitlines()[0][:80]}...), using FP16")
        co.useFloat16 = True
        return faiss.index_cpu_to_gpu(res, 0, cpu_index, co), True


def append_row(path, row):
    """Appends one result to the CSV right away, so a crash later in the run keeps it."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


# Faiss 1.14.1's GPU scalar-quantizer scan aborts the process (CUDA "invalid argument" in
# ivfInterleavedScanImpl with NUM_WARP_Q = 1024) when k > 512; IVF-Flat is fine at k = 1000.
GPU_SQ_MAX_K = 512

# PQ encodings: number of sub-vectors (bytes per vector) is d divided by this.
PQ_DIVISOR = {"pq": 2, "pq_d4": 4, "pq_d8": 8}
ENCODINGS = ["flat", "sq8", "sq4", *PQ_DIVISOR]
KNOWN_ALGOS = ([f"{dev}_ivf_{enc}" for dev in ("cpu", "gpu") for enc in ENCODINGS]
               + [f"gpu_ivf_{enc}_refine" for enc in PQ_DIVISOR])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", default="16,32,64")
    p.add_argument("--C", default="100,1000")
    p.add_argument("--algo", default=DEFAULT_ALGOS)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--nq", type=int, default=0, help="use only the first nq queries (0 = all)")
    p.add_argument("--batch", type=int, default=0, help="queries per search call (0 = all)")
    p.add_argument("--temp-mb", type=int, default=0,
                   help="Faiss temporary-memory pool in MB (0 = none, as in the budget runs)")
    p.add_argument("--n-train", type=int, default=1_000_000, help="k-means and encoder sample")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ids64", action="store_true", help="64-bit IDs on the GPU (Faiss's default)")
    p.add_argument("--csv", default="")
    a = p.parse_args()

    algos = a.algo.split(",")
    for algo in algos:
        if algo not in KNOWN_ALGOS:
            p.error(f"unknown algo {algo}; expected one of {','.join(KNOWN_ALGOS)}")
    use_gpu = any(x.startswith("gpu_") for x in algos)

    base = load(a.base, np.float32)
    queries = load(a.query, np.float32)
    if a.nq:
        queries = queries[: a.nq]
    gt = load(a.gt, np.int32)[: len(queries)]
    n, d = base.shape
    threads = faiss.omp_get_max_threads()
    print(f"faiss {faiss.__version__}, GPUs {faiss.get_num_gpus()}, OpenMP threads {threads}")
    print(f"base {n} x {d}, queries {len(queries)}, nlist {a.nlist}, batch {a.batch or 'all'}")

    t0 = time.perf_counter()
    quantizer, sample = train_coarse(base, a.nlist, a.n_train, a.seed)
    coarse_s = time.perf_counter() - t0
    gc.collect()  # releases the k-means GPU resources before the memory baseline
    print(f"coarse k-means: {coarse_s:.1f} s")

    mem = res = None
    if use_gpu:
        mem = DeviceMemory()
        res = faiss.StandardGpuResources()
        if a.temp_mb:
            res.setTempMemory(a.temp_mb * MB)
        else:
            res.noTempMemory()
        context_mb = mem.used_mb()
        print(f"device memory with the CUDA context only: {context_mb:.0f} MB (subtracted below)")

    cpu_indexes, build_times = {}, {}
    for algo in algos:
        device, _, rest = algo.partition("_ivf_")
        refine = rest.endswith("_refine")
        encoding = rest.removesuffix("_refine")
        if encoding not in cpu_indexes:
            t0 = time.perf_counter()
            cpu_indexes[encoding] = build_cpu(encoding, quantizer, base, sample)
            build_times[encoding] = time.perf_counter() - t0
        cpu_index = cpu_indexes[encoding]
        build_s = coarse_s + build_times[encoding]
        index_mb = n * cpu_index.code_size / MB

        lut_fp16 = False
        gpu_index_mb = ""
        if device == "gpu":
            t0 = time.perf_counter()
            gpu_index, lut_fp16 = to_gpu(res, cpu_index, encoding, a.ids64)
            build_s += time.perf_counter() - t0
            gpu_index_mb = round(mem.used_mb() - context_mb, 1)
            index = gpu_index
        else:
            index = cpu_index
        if refine:
            # Wraps the populated GPU index; the FP32 vectors for the exact rerank stay on the host.
            index = faiss.IndexRefineFlat(gpu_index, faiss.swig_ptr(base))
        print(f"=== {algo}: build {build_s:.1f} s, codes {index_mb:.0f} MB"
              + (f", on GPU {gpu_index_mb} MB" if device == "gpu" else ""))

        for nprobe in map(int, a.nprobe.split(",")):
            # ParameterSpace raises if the index has no nprobe, unlike a plain attribute.
            space = faiss.GpuParameterSpace() if device == "gpu" else faiss.ParameterSpace()
            space.set_index_parameter(gpu_index if refine else index, "nprobe", nprobe)
            for C in map(int, a.C.split(",")):
                if refine:
                    index.k_factor = C / 10
                    k = 10
                else:
                    k = C
                if device == "gpu" and encoding.startswith("sq") and C > GPU_SQ_MAX_K:
                    print(f"nprobe={nprobe:<3d} C={C:<4d} skipped: Faiss GPU SQ scan fails for "
                          f"k > {GPU_SQ_MAX_K}")
                    continue
                (_, ids), ms = timed_search(lambda: search_batched(index, queries, k, a.batch), a.reps)
                peak = ""
                if device == "gpu":
                    peak = round(mem.peak_during(lambda: search_batched(index, queries, k, a.batch))
                                 - context_mb, 1)
                ids = ids.astype(np.int64)
                if refine:
                    no_rerank, reranked = "", recall_at_10(ids, gt)
                else:
                    no_rerank = recall_at_10(ids, gt)
                    reranked = recall_at_10(exact_rerank(base, queries, ids), gt)
                qps = len(queries) / (ms / 1e3)
                print(f"nprobe={nprobe:<3d} C={C:<4d} search_ms={ms:8.1f} qps={qps:9.0f}  R@10 "
                      + (f"no_rerank={no_rerank:.4f} " if no_rerank != "" else "")
                      + f"rerank={reranked:.4f}" + (f"  peak {peak} MB" if peak != "" else ""))
                append_row(a.csv, dict(
                    algo=algo, nlist=a.nlist, nprobe=nprobe, C=C, nq=len(queries),
                    search_ms=round(ms, 3), qps=round(qps, 1), index_mb=round(index_mb, 1),
                    build_s=round(build_s, 2),
                    recall10_no_rerank=round(no_rerank, 5) if no_rerank != "" else "",
                    recall10_rerank=round(reranked, 5),
                    device=device, rerank_timed=int(refine), batch=a.batch, threads=threads,
                    temp_mb=a.temp_mb, gpu_index_mb=gpu_index_mb, gpu_peak_mb=peak,
                    pq_lut_fp16=int(lut_fp16), faiss=faiss.__version__,
                    pq_m=d // PQ_DIVISOR[encoding] if encoding in PQ_DIVISOR else "",
                    gpu_id_bits=(64 if a.ids64 else 32) if device == "gpu" else ""))

        if device == "gpu":
            del index, gpu_index
            gc.collect()
    if a.csv:
        print(f"appended to {a.csv}")


if __name__ == "__main__":
    main()
