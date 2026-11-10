# GPU Kernels and Analysis (LL)

This chapter covers the GPU side of the search: what each kernel does, how fast it runs, where the
time goes, and how it compares with NVIDIA cuVS. All numbers come from one NVIDIA A40 (CARC node
b11-09) with 10,000 queries. Job IDs are listed at the end.

## 1. Summary

- The GPU stage now runs at **84,600 queries/s on SIFT1M** and **47,100 on DEEP10M** with 4-bit
  codes (nprobe 64, C 100, recall@10 after rerank 0.995 and 0.985). On Day 7 the same settings ran
  at 33,300 and 14,100.
- Against cuVS IVF-PQ with codes of the same size, we are **1.5 times faster on SIFT1M** and
  **8% slower on DEEP10M** (was 1.8 times slower).
- Two changes did most of the work: picking the top C without a full sort (Day 11), and fusing that
  selection into the scan so most estimates never leave the chip (Day 12).
- A better 1-bit estimate (RaBitQ style) raised DEEP10M 1-bit recall from 0.929 to 0.984 at
  C = 1,000.

## 2. The GPU stage

For a batch of queries the GPU runs three steps and hands C candidate IDs to the CPU:

1. **Coarse search.** One matrix multiplication gives the distance from every query to every
   cluster center; a warp top-k keeps the nprobe closest clusters.
2. **Scan.** One thread block per (query, cluster) estimates the distance to every vector in that
   cluster from its compressed code.
3. **Top-C.** Keep each query's C smallest estimates.

Time per step for 10,000 DEEP10M queries, 4-bit codes, nprobe 64, C 100:

| Step | Day 7 (full sort) | Day 11 (radix select) | Day 12 (fused) |
|---|---|---|---|
| Coarse search | 19 ms | 18 ms | 16 ms |
| Scan | 246 ms | 247 ms | 184 ms (includes per-list top-C) |
| Top-C | 443 ms | 92 ms | 6 ms |
| **Total** | **711 ms** | **360 ms** | **213 ms** |
| GPU work memory | 1,066 MB | 357 MB | 66 MB |

## 3. Kernels

### 3.1 Distances with one GEMM, and a warp top-k

Squared distance is |q|^2 - 2 q.x + |x|^2, so the q.x terms for many queries and vectors come from
one cuBLAS SGEMM. Top-k then runs one warp per query: the warp reads 32 distances at a time and only
inserts a value into its sorted list (in shared memory) if it beats the current k-th best. After the
first few hundred values this is rare, so the loop runs at memory speed. Brute force, k-means and the
coarse search all use these two pieces.

**GPU brute force (SIFT1M):** 268 ms for 10,000 queries against 201 s for the serial CPU baseline,
about 750 times faster, with identical results. DEEP10M ground truth (top 100 for 10,000 queries
over 10 million vectors) takes 2.5 s.

### 3.2 Index build

k-means runs on the GPU: each iteration assigns points with the GEMM above plus an argmin, then adds
every point into its center with atomic adds. The SIFT1M index (1,024 lists) builds in 1.4 s; DEEP10M
(4,096 lists) in 9.3 s. Residuals (vector minus its center) are encoded on the CPU:

| Code | Bytes per SIFT vector | How a distance is estimated |
|---|---|---|
| 8-bit | 128 | decode each dimension, exact formula on the decoded residual |
| 4-bit | 64 | same, two dimensions per byte |
| 1-bit | 16 + 8 | sign bits after a random rotation, plus the residual norm and one correction factor |

### 3.3 Scan

A block first prepares the query for its cluster in shared memory. Each thread then takes one vector
at a time and reads its code in 16-byte loads (16 dimensions at 8 bits, 32 at 4 bits).

**1-bit estimate.** The first version compared the query's sign bits with the vector's and turned the
Hamming distance into an angle; it threw away almost everything about the query, and DEEP10M recall
stopped at 0.929 even with 1,000 candidates. The Day 12 version follows RaBitQ: the query keeps 4 bits
per dimension (stored as 4 bit planes, so each 32 dimensions still cost one AND and one popcount per
plane), and each vector stores a factor that corrects the bias of its sign code.
The query rotation is done once per query and once per centroid rather than once per block, since
R^T(q - c) = R^T q - R^T c.

| 1-bit recall@10 after rerank, nprobe 64 | First version | RaBitQ style |
|---|---|---|
| SIFT1M, C = 100 | 0.820 | 0.959 |
| DEEP10M, C = 100 | 0.643 | 0.870 |
| DEEP10M, C = 1,000 | 0.929 | 0.984 |

### 3.4 Top-C, three versions

- **Day 7, full sort.** CUB segmented radix sort of every estimate, although only C of up to several
  hundred thousand per query are needed.
- **Day 11, radix select.** One block per query finds the C-th smallest estimate 8 bits at a time
  (four histogram passes in shared memory), copies the values below it, and sorts only those C.
  Top-C time drops from 443 ms to 92 ms on DEEP10M.
- **Day 12, fused.** The scan block keeps its whole cluster's estimates in shared memory (the largest
  DEEP10M cluster needs 31 KB), runs the same radix select there, and writes only its C best.
  The query's C best are always among the clusters' C bests, so the final pass over nprobe x C values
  gives the same answer. Most estimates now never reach GPU memory.

## 4. Memory bandwidth (Nsight Compute)

The A40's peak is 696 GB/s; a plain read kernel reaches 591 GB/s. SIFT1M, nprobe 64, C 100:

| Kernel | 8-bit | 4-bit | 1-bit |
|---|---|---|---|
| Scan that writes every estimate (Day 11) | 584 GB/s (84%) | 538 GB/s (77%) | 580 GB/s (83%) |
| Fused scan + per-list top-C (Day 12) | 578 GB/s (83%) | 349 GB/s (50%) | 81 GB/s (12%) |
| Per-query top-C, Day 11 | 505 GB/s (73%) | | |

The Day 11 kernels run close to the memory limit, so the only way to go faster was to move fewer
bytes. Fusing does that: for 4-bit the scan reads 9.8 GB instead of 12.3 GB plus 1.7 GB of writes and
3.4 GB of top-C reads. The fused 4-bit kernel uses only half the bandwidth, but it moves so much less
data that it finishes sooner.

**Where fusing does not pay off.** For 1-bit codes the scan itself is cheap, so the per-cluster radix
select (four passes over shared memory) becomes the cost. On SIFT1M fused 1-bit runs at 67,300
queries/s against 133,200 for the Day 11 version. With C = 1,000 most clusters are smaller than C,
the selection is skipped, and fused 1-bit gets faster (74,600), which confirms where the time goes.
On DEEP10M, where the Day 11 top-C costs more, fused 1-bit wins (60,100 against 47,700).
The best setting is therefore: fused for 8-bit and 4-bit; for 1-bit, select on SIFT1M and fused on
DEEP10M.

## 5. Comparison with cuVS

cuVS 26.10 on the same data, number of clusters and nprobe (64), C = 100. Recall is after the same
exact rerank. IVF-PQ uses d / 2 bytes per vector, the same size as our 4-bit codes; IVF-Flat keeps the
full FP32 vectors.

| Dataset | Ours | cuVS |
|---|---|---|
| SIFT1M, ~65 MB codes | 4-bit fused: 84,600 queries/s, recall 0.9954 | IVF-PQ: 54,900, recall 0.9955 |
| DEEP10M, ~500 MB codes | 4-bit fused: 47,100 queries/s, recall 0.985 | IVF-PQ: 51,400, recall 0.988 |
| SIFT1M, precise codes | 8-bit fused: 37,100 | IVF-Flat (FP32): 18,400 |
| DEEP10M, precise codes | 8-bit fused: 22,900 | IVF-Flat (FP32): 9,800 |

Nsight shows the two libraries hitting different limits. cuVS IVF-Flat's scan streams at 97% of peak
bandwidth, better than our best (84%). cuVS IVF-PQ's main kernel uses under 1% of bandwidth: it is
limited by computing distances from lookup tables, not by memory. On DEEP10M that compute-bound design
is still slightly ahead of our memory-bound 4-bit scan.

## 6. What did not work, and limits

- **Fused top-C for 1-bit codes**, see section 4.
- **Merging counter updates within a warp** (to reduce atomic contention when many 1-bit estimates are
  equal) gave no gain and was removed.
- **cuVS IVF-PQ is still 8% faster on DEEP10M.** Closing it would need either the fused kernel to reach
  the bandwidth of the plain scan (it uses 50% now), or a lookup-table estimate like cuVS's.
- **C barely matters for 8-bit and 4-bit.** Raising C from 100 to 1,000 does not raise recall, because
  the missing neighbors sit in clusters that were never probed; nprobe sets the ceiling.
- Fused and select top-C support C up to 1,024; the fused version also needs the largest cluster to
  fit in shared memory (99 KB, about 24,000 vectors). Larger settings fall back to the full sort.
- The 1-bit index is larger than the 1-bit codes alone (230 MB on DEEP10M) because of the norm and
  factor stored per vector.

## 7. Reproduce

From the project root on CARC:

| Script | What it runs |
|---|---|
| ll_day1_a40.sbatch | bandwidth tests, GPU brute force |
| ll_day7_a40.sbatch | SIFT1M index and end-to-end checks |
| ll_deep10m_a40.sbatch | DEEP10M ground truth, index and checks |
| ll_day8_ncu_a40.sbatch | Nsight Compute per kernel |
| ll_day11_topc_a40.sbatch | full sort vs radix select |
| ll_day11_cuvs_a40.sbatch | cuVS baseline and its Nsight profile |
| ll_day12_fix_a40.sbatch | fused vs select, RaBitQ 1-bit, both datasets |

Jobs behind these numbers: 13745127, 13745470, 13757822, 13829346, 13909879, 13915798, 13915799.
