# How we measure (all GPU benchmarks)

Every throughput number we compare (our scan, cuVS, Faiss, cuVS IVF-RaBitQ) is measured the same
way, so the numbers are apples to apples. `scripts/ll_compare_a40.sbatch` runs all four with
these settings.

## The timed region

| Rule | Value |
|---|---|
| Hardware | one NVIDIA A40 (CARC), one Slurm job per GPU |
| Queries | all 10,000 of the dataset's query file |
| Start of the timer | queries in host memory (pinned where the library allows it) |
| End of the timer | candidate ids and distances back in host memory |
| Inside the timer | query upload, coarse search, scan, top-C, result download |
| Outside the timer | index build, exact rerank, recall, correctness checks |
| Batch | queries per search call: 128, and all 10,000 in one call (`batch 0` in the baseline CSVs) |
| Repetitions | one warm-up, then the median of 5 runs |
| Parameters | nprobe 64; C 100 (1-bit and RaBitQ also C 1000) |
| Clock | host wall clock around the whole loop (`steady_clock` / `perf_counter`), after the GPU finishes |

The upload and download must be inside the timer because our pipeline reranks on the CPU: the
candidates have to reach host memory, whoever produces them.

## What we compare

Same method against same method, as in the proposal: our 8-bit and 4-bit scalar-quantized codes
against Faiss GPU IVF-SQ8 and IVF-SQ4, our 1-bit codes against cuVS IVF-RaBitQ, and FP32 IVF-Flat
(Faiss and cuVS) as the uncompressed reference. Product quantization (IVF-PQ) is a different
method and is not part of the comparison.

## Per benchmark

| Benchmark | Owner | Upload timed | Download timed | Batch option |
|---|---|---|---|---|
| `src/bench_scan.cu` (ours) | LL | yes (pinned staging buffer) | yes | `--batch` |
| `src/cuvs_baseline.py` | LL | yes with `--transfer 1` (default since Nov 25) | yes | `--batch` (0 = all) |
| `src/faiss_baseline.py` | HS | yes (Faiss GPU takes host arrays) | yes | `--batch` (0 = all) |
| `src/cuvs_rabitq_bench.cu` | HS | yes with `--transfer 1` (default since Nov 25) | yes | `--batch` (0 = all) |

`--transfer 0` in the two cuVS benchmarks reproduces the older numbers, where the queries were on
the GPU before timing started and (cuvs_baseline only) the results stayed there.

## Recall and memory

- Recall@10 after an exact rerank of the C candidates (FP32, on the CPU, not timed). Our C++
  rerank and the baselines' NumPy rerank return the same neighbors up to distance ties.
- `codes_mb`: n × bytes per vector of codes only (n·d·bits/8). The same definition in
  every CSV, for comparing index sizes.
- Device memory, measured the same way in all four benchmarks with `cudaMemGetInfo` (whole
  GPU, so one job per GPU):
  1. Baseline: device memory in use with the CUDA context, the library's handle and cuBLAS
     loaded (one tiny GEMM; Faiss's GPU k-means does this before its baseline) and, for cuVS
     IVF-Flat, the base vectors used only for building. cuBLAS alone is about 190 MB of kernels
     and library state, the same for every method, so it is left out. Everything the library
     allocates after that counts, its workspaces included.
  2. No warm-up index before the baseline: a throwaway index leaves cuVS's workspace allocated
     (about 640 MB for IVF-RaBitQ) and would hide it from every later number
     (`scripts/ll_rabitq_warmup_a40.sbatch` measured this; speed was the same either way).
  3. `gpu_index_mb`: in use once the index is ready, minus the baseline. Ours allocates its work
     buffers with the index, so for ours this includes them.
  4. `gpu_peak_mb`: the highest reading during one extra, untimed search of all queries at the
     benchmark's batch size, sampled every 0.1 ms from another thread, minus the baseline.
     This is the number to compare: everything the GPU must hold to answer the queries.
  Run with `CUDA_MODULE_LOADING=EAGER` so kernel code is in the baseline. Ours also reports
  `index_mb` and `work_mb` from `DeviceBudget`, which counts each of our allocations exactly.

## What still differs

- Each library trains its own coarse k-means (ours: `ivf_build`; Faiss: seed 1234, 1M sample;
  cuVS: its own), so the lists differ slightly. Compare at matching recall, not only matching
  nprobe.
- Faiss and cuVS allocate their own result buffers per call; ours reuses buffers sized for the
  largest batch. This is part of each library's cost and stays in the timer.
- The end-to-end pipeline (XJ, CPU rerank overlapped with the GPU) is not in the repo yet; when it
  is, it should time the same region plus the rerank.
