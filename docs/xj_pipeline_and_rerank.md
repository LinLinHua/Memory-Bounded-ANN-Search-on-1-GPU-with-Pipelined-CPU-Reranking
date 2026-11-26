# CPU Rerank and Pipeline (XJ)

This chapter covers the CPU side of the search: the exact rerank, where its time goes compared with
reranking on the GPU, how much the CPU–GPU pipeline gains, and what the GPU memory budget allows.
All numbers come from one NVIDIA A40 node (CARC b11-09, 2 × 16-core AMD EPYC 7313) with 16 CPU
cores, 10,000 queries and nprobe 64. Jobs 14870665 (rerank, pipeline, scaling, serial baseline) and
14870691 (budget), Nov 25, 2026. Tables come from `python3 scripts/xj_summary.py`.

## 1. Summary

- **Reranking belongs on the CPU.** For C ≥ 50 the CPU rerank beats the GPU reading the same
  vectors over PCIe: 134 ms against 230 ms for 10,000 DEEP10M queries at C = 1,000. The GPU spends
  99% of its time on PCIe; sending the candidate IDs to the CPU costs 1.5 ms.
- **The pipeline raises throughput by up to 1.78 times**, close to what the formula predicts
  (1–10% below it), when the rerank is a large share of the work (1-bit codes, C = 1,000). With 4-bit
  codes and C = 100 the rerank is small and the gain is at most 1.22 times.
- **It does not lower latency.** Each batch still passes through both stages in order; under the
  pipeline its latency stays the same or rises (up to 1.5 times on SIFT1M at batch 1,024).
- **Under a tight budget, 1-bit codes with CPU rerank lose almost nothing.** At 1/8 of the raw
  DEEP10M size only the 1-bit index fits, and it still reaches 45,700 queries/s, against 47,800 for
  the best setting at any budget. 8-bit codes never win.
- End to end, the pipelined search is **7,400 times faster** than the serial exact search on DEEP10M
  (recall@10 0.984) and 1,600 times on SIFT1M (0.995).

## 2. Rerank

The GPU returns C candidate IDs per query. The rerank computes their exact distances from the full
FP32 vectors and keeps the 10 nearest.

- **CPU** (`src/cpu_rerank.cpp`): OpenMP over queries with dynamic scheduling, an `omp simd`
  distance loop, and a sorted array of 10 for the top-k. Each thread prefetches the vector four
  candidates ahead, because candidate vectors are scattered across host memory.
- **GPU** (`src/gpu_rerank.cu`): one block per query, one warp per candidate, reading the vectors
  over PCIe from pinned host memory (zero-copy), since the budget leaves no room for them on the GPU.

Both return the same neighbors up to ties (relative distance error at most 1e-4) in every run.

## 3. Where the rerank time goes (result 3)

`src/bench_rerank_breakdown.cu` takes real 1-bit candidates from the scan and splits each path:

| Part | CPU path | GPU path |
|---|---|---|
| Candidate IDs | copied GPU → host (nq × C × 4 bytes) | none, they are already on the GPU |
| Fetch vectors | gather from DRAM | read over PCIe |
| Distances | SIMD on the CPU | warp per candidate |
| Results | none, already on the host | copied GPU → host (nq × 10 × 8 bytes) |

The fetch time is measured by difference: the same rerank runs again with every candidate remapped
to the first 256 vectors, which stay in cache (CPU) or in GPU memory (GPU).

DEEP10M, 10,000 queries (ms):

| C | Recall@10 | CPU: IDs | CPU: gather | CPU: compute | **CPU total** | GPU: PCIe | GPU: compute | **GPU total** |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.387 | 0.02 | 0.6 | 2.8 | 3.4 | 2.3 | 0.3 | **2.6** |
| 50 | 0.757 | 0.08 | 1.2 | 7.7 | **9.0** | 11.6 | 0.4 | 12.0 |
| 100 | 0.870 | 0.16 | 4.1 | 11.5 | **15.7** | 22.8 | 0.5 | 23.3 |
| 200 | 0.937 | 0.31 | 12.1 | 15.5 | **27.8** | 45.4 | 0.7 | 46.1 |
| 500 | 0.975 | 0.77 | 46.3 | 22.6 | **69.7** | 113.3 | 1.4 | 114.7 |
| 1,000 | 0.984 | 1.53 | 103.2 | 29.8 | **134.5** | 227.0 | 2.5 | 229.6 |

- The GPU is limited by PCIe: zero-copy reads reach 17 GB/s on DEEP10M (384-byte vectors) and
  12–17 GB/s on SIFT1M (512-byte vectors), below the 26 GB/s of a bulk pinned copy, because each
  read is a small random access.
- The CPU gathers at 37 GB/s at C = 1,000 (DEEP10M), about half the STREAM triad bandwidth with
  16 threads (72 GB/s), as expected for random 384-byte reads.
- Only for C ≤ 20, when there is almost nothing to fetch, does the GPU win, by under 1 ms.
- The rerank grows linearly with C, so the candidate count the scan needs for a target recall sets
  its cost: 1-bit codes need C ≈ 1,000 for recall 0.98 on DEEP10M.

## 4. Pipeline (result 4)

`src/bench_pipeline.cu` runs the same micro-batches two ways:

- **Sequential:** scan batch b, then rerank batch b.
- **Pipelined:** a scan thread runs the GPU scan of batch b+1 while the OpenMP team reranks batch b.
  Two candidate buffers form a queue of depth 1. The scan thread waits on the GPU with blocking
  synchronization, so it does not spin on a core the rerank needs; the rerank team gets 15 threads
  in both modes. (A scan thread was used instead of more CUDA streams because the scanner
  synchronizes at the end of each batch; a second set of work buffers would also cost GPU memory.)

**Correctness.** One extra pipelined run keeps the candidates it reranked; reranking them again
afterwards gives bit-identical results in every setting, so the pipeline has no races. Sequential and
pipelined runs differ in 0.05–1.3% of result entries, with the same recall, because the scan is not
deterministic: when estimates tie at the C-th place, which ones are kept depends on the order of
atomic updates (more ties with 4-bit codes).

The expected speedup is (T_scan + T_rerank) / max(T_scan, T_rerank), from the sequential stage
times. DEEP10M:

| Codes, C | Batch | Sequential QPS | Pipelined QPS | Speedup | Predicted | P50 latency seq → pipe (ms) |
|---|---|---|---|---|---|---|
| 1-bit, 1,000 | 1 | 1,864 | 2,186 | 1.17 | 1.19 | 0.53 → 0.57 |
| 1-bit, 1,000 | 16 | 14,237 | 20,824 | 1.46 | 1.49 | 1.11 → 1.15 |
| 1-bit, 1,000 | 64 | 21,992 | 37,904 | **1.72** | 1.79 | 2.89 → 2.43 |
| 1-bit, 1,000 | 256 | 29,618 | 45,453 | 1.53 | 1.68 | 8.12 → 8.23 |
| 1-bit, 1,000 | 1,024 | 31,910 | 47,698 | 1.49 | 1.65 | 31.3 → 31.1 |
| 4-bit, 100 | 1 | 2,852 | 2,819 | 0.99 | 1.04 | 0.35 → 0.35 |
| 4-bit, 100 | 64 | 33,271 | 39,850 | 1.20 | 1.25 | 1.92 → 1.98 |
| 4-bit, 100 | 256 | 37,654 | 45,870 | 1.22 | 1.32 | 6.77 → 6.93 |
| 4-bit, 100 | 1,024 | 45,156 | 47,823 | 1.06 | 1.16 | 21.0 → 21.9 |

- The gain is largest when the two stages take similar time (1-bit, batch 64: 254 ms scan,
  200 ms rerank). With 4-bit codes the rerank is 10–25% of the work, so there is little to hide.
- The measured speedup stays 1–10% below the prediction because the stages slow each other down: on
  SIFT1M the scan takes 18–30% longer when the rerank runs beside it (80 → 95 ms at batch 1,024,
  91 → 118 ms at batch 256), as both draw on host memory and the CPU. On DEEP10M it does not.
- At batch 1 the GPU is underused (one query per launch) and throughput is 2–3 thousand queries/s
  in either mode.
- On SIFT1M the pattern is the same (1-bit, C = 1,000: up to 1.78 times at batch 16), but there the
  rerank is the longer stage from batch 64 on, so the pipeline is limited by the CPU.

**Latency.** The pipeline overlaps different batches, so it helps throughput only. On SIFT1M with
1-bit codes, P50 latency at batch 1,024 rises from 32 ms to 49 ms: the rerank is the slower stage
(244 ms against 95 ms of scan), so each scanned batch waits in the queue until the rerank of the
previous batch is done, and the scan itself runs slower beside the rerank. To lower the latency
of one query, its own rerank would have to be split across threads, which this design does not do.

## 5. Pipeline under the GPU memory budget

`scripts/xj_budget_a40.sbatch` repeats the pipeline with the budget set to 1/16 … 1/2 of the raw FP32
data. The index and the scan's work buffers (which grow with the batch) must fit; settings that do not
are recorded as over budget. Best pipelined QPS that fits, DEEP10M (raw size 3,662 MiB):

| Budget | 1-bit, C = 1,000 | 4-bit, C = 100 | 8-bit, C = 100 |
|---|---|---|---|
| 1/16 (228 MiB) | over (index alone 232 MiB) | over | over |
| 1/8 (457 MiB) | **45,715** (batch 256) | over (index 498 MiB) | over |
| 1/4 (915 MiB) | 47,833 (batch 1,024) | **47,816** (batch 1,024) | over (index 955 MiB) |
| 1/2 (1,831 MiB) | 47,834 | **48,592** | 23,430 |

Recall@10 is 0.984 (1-bit), 0.985 (4-bit) and 0.986 (8-bit) throughout.

- **Below 1/4, 1-bit codes with CPU rerank are the only option, and they lose almost nothing**:
  45,700 queries/s at 1/8 against 48,600 for the best setting at any budget.
- From 1/4 up, 1-bit and 4-bit codes are tied within 2%. 8-bit codes are half as fast even when they
  fit, because the scan reads twice the bytes of 4-bit codes.
- The budget also caps the batch: at 1/8, the 1-bit scan's work buffers for batch 1,024 (525 MiB)
  do not fit, so the pipeline runs at batch 256.
- On SIFT1M (raw 488 MiB) nothing fits at 1/16 (30 MiB) even at batch 16, because the 1-bit index
  (28 MiB) leaves no room for the work buffers and cuBLAS; at 1/8 only 1-bit codes at batch 16 fit.

**Budget validation.** The budget counts every allocation of our scanner. `cudaMemGetInfo` measures
13–17 MiB more, from cuBLAS's workspace and allocator rounding, which do not go through the budget:
on SIFT1M at 1/4 (122 MiB), 4-bit at batch 1,024 counts 121 MiB but measures 134 MiB. Giving cuBLAS
a workspace from the budget (`cublasSetWorkspace`) would close this gap.

## 6. Scaling and end-to-end speedup (result 5)

OpenMP scaling of the CPU rerank, DEEP10M, 1-bit candidates (speedup over 1 thread):

| C | Prefetch | 2 threads | 4 | 8 | 16 | GB/s at the best |
|---|---|---|---|---|---|---|
| 100 | off | 1.87 | 2.62 | 5.15 | 6.41 | 21.2 |
| 100 | 4 ahead | 1.83 | 2.35 | 4.64 | 6.34 | 25.2 |
| 1,000 | off | 1.86 | 3.05 | 6.22 | 4.79 | 29.3 (8 threads) |
| 1,000 | 4 ahead | 2.04 | 2.95 | 6.13 | 6.30 | 27.9 |

- The rerank is memory-bound: scaling flattens past 8 threads at 20–30 GB/s of random gathers.
- Prefetch helps one thread by 17–29% at C = 100 and matters little once many threads keep the
  memory system busy.
- **These CPU numbers are noisy.** The debug node was shared with other jobs, which compete for
  memory bandwidth: the same SIFT1M rerank took 126 ms in an earlier run (job 14842004) and 297 ms in
  this one, and 16 threads were sometimes slower than 8. Only 16 of the 32 cores could be requested,
  so scaling across both sockets is untested.

End to end, against the serial exact search (`src/serial_bruteforce.cpp`, one thread; 6.4 queries/s
on DEEP10M over 1,000 queries, 49.7 on SIFT1M):

| Dataset | Setting | Recall@10 | Pipelined QPS | Speedup |
|---|---|---|---|---|
| DEEP10M | 4-bit, C = 100, batch 1,024 | 0.985 | 47,823 | 7,445 |
| DEEP10M | 1-bit, C = 1,000, batch 1,024 | 0.984 | 47,698 | 7,425 |
| SIFT1M | 4-bit, C = 100, batch 1,024 | 0.995 | 79,446 | 1,598 |
| SIFT1M | 1-bit, C = 1,000, batch 256 | 0.995 | 39,928 | 803 |

## 7. Limits

- CPU timings vary up to 2 times between runs on the shared debug node; a run on an otherwise idle
  node would make results 3 and 5 firmer.
- Scaling is measured up to 16 threads (one socket).
- Latency is the per-batch service time with all queries available at the start, not latency under
  an arrival process.
- The pipeline does not lower single-query latency; splitting one query's rerank across threads
  would.

## 8. Reproduce

    sbatch scripts/xj_pipeline_a40.sbatch   # rerank breakdown, pipeline, scaling, serial baseline
    sbatch scripts/xj_budget_a40.sbatch     # pipeline under the GPU memory budget
    python3 scripts/xj_summary.py           # tables for results 3, 4 and 5 and the budget table

On the debug partition: `sbatch --partition=debug --time=01:00:00 --cpus-per-task=16 ...`.
