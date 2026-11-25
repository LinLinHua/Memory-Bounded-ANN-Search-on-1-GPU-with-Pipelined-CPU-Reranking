# CPU Rerank and Pipeline (XJ)

This chapter covers the CPU side of the search: the exact rerank, where its time goes compared with
reranking on the GPU, and how much the CPU–GPU pipeline gains. Numbers are filled in from
`python3 scripts/xj_summary.py` after `scripts/xj_pipeline_a40.sbatch` runs on an A40 node.

## 1. Rerank

The GPU returns C candidate IDs per query. The rerank computes their exact distances from the full
FP32 vectors and keeps the 10 nearest.

- **CPU** (`src/cpu_rerank.cpp`): OpenMP over queries with dynamic scheduling, an `omp simd`
  distance loop, and a sorted array of 10 for the top-k. Each thread prefetches the vector
  `--prefetch` candidates ahead, because candidate vectors are scattered across host memory.
- **GPU** (`src/gpu_rerank.cu`): one block per query, one warp per candidate, reading the vectors
  over PCIe from pinned host memory (zero-copy), since the budget leaves no room for them on the GPU.

Both return the same neighbors up to ties (relative distance error at most 1e-4).

## 2. Where the rerank time goes (result 3)

`src/bench_rerank_breakdown.cu` takes real 1-bit candidates from the scan and splits each path:

| Part | CPU path | GPU path |
|---|---|---|
| Candidate IDs | copied GPU → host (nq × C × 4 bytes) | none, they are already on the GPU |
| Fetch vectors | gather from DRAM | read over PCIe |
| Distances | SIMD on the CPU | warp per candidate |
| Results | none, already on the host | copied GPU → host (nq × 10 × 8 bytes) |

The fetch time is measured by difference: the same rerank runs again with every candidate remapped
to the first 256 vectors, which stay in cache (CPU) or in GPU memory (GPU). What remains is the
distance work; the rest is fetching.

Results: _to fill_ (table from `xj_summary.py`, result 3).

## 3. Pipeline (result 4)

`src/bench_pipeline.cu` runs the same micro-batches two ways:

- **Sequential:** scan batch b, then rerank batch b.
- **Pipelined:** a scan thread runs the GPU scan of batch b+1 while the OpenMP team reranks batch b.
  Two candidate buffers form a queue of depth 1. The scan thread waits on the GPU with blocking
  synchronization, so it does not spin on a core the rerank needs; the rerank team gets one thread
  fewer than the node's cores in both modes.

Both modes must return bit-identical IDs and distances. The expected throughput speedup is
(T_scan + T_rerank) / max(T_scan, T_rerank), at most 2 when the two stages take equal time.

Latency is reported per batch, from the start of its scan to the end of its rerank. The pipeline
overlaps different batches, so it raises throughput, but each batch still passes through both
stages in order: its latency does not fall, and can rise when the two stages compete for memory
bandwidth or the scan of the next batch delays the current one.

Results: _to fill_ (result 4).

## 4. Scaling and end-to-end speedup (result 5)

- OpenMP scaling of the CPU rerank from 1 thread to all cores, with and without prefetch, at C = 100
  and 1,000. The rerank is memory-bound, so scaling should flatten near the STREAM bandwidth.
- End-to-end speedup of the pipelined search over the serial exact baseline
  (`src/serial_bruteforce.cpp`; DEEP10M on 1,000 queries).

Results: _to fill_ (result 5).

## 5. Reproduce

    sbatch scripts/xj_pipeline_a40.sbatch          # both datasets, all three steps
    python3 scripts/xj_summary.py                  # tables for results 3, 4 and 5
