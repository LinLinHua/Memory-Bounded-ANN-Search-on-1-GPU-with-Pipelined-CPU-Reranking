# Memory-Bounded ANN Search on a Single GPU with Pipelined CPU Reranking

EE 451 course project. Approximate nearest-neighbor search where the GPU holds only a
compressed index under a fixed memory budget, and the CPU reranks the GPU's candidates
with the full-precision vectors kept in host memory.

Team: Lin Lin (LL, GPU kernels) · Xinyi (XJ, CPU rerank / pipeline) · He Sui (HS, data / baselines)

## How a query runs

1. **Coarse search (GPU):** find the nprobe closest IVF lists.
2. **Scan (GPU):** estimate distances from 8-, 4- or 1-bit codes; keep the best C candidates.
3. **Rerank (CPU):** exact distances for those C candidates; return the top 10.

## Layout

| Path | What it is |
|---|---|
| src/serial_bruteforce.cpp | Serial exact search, the correctness and speedup baseline |
| src/gpu_bruteforce.cu | GPU exact search (cuBLAS GEMM + top-k); also writes ground truth |
| src/ivf_build.cu | GPU k-means and IVF index with 8/4/1-bit codes |
| src/ivf_scan.cu | GPU coarse search, scan kernels and top-C |
| src/cpu_rerank.cpp, src/gpu_rerank.cu | CPU rerank (OpenMP + SIMD) and GPU zero-copy rerank |
| src/device_budget.h | Enforces the GPU memory budget |
| src/bench_scan.cu | End-to-end run with correctness checks |
| src/build_ivf.py, src/ref_scan.py | NumPy reference for the same index format |
| scripts/ | CARC job scripts (A40) and data preparation |
| docs/deep10m.md | How DEEP10M is prepared |
| TODO.md | Task list and owners |

## Run on CARC

Data lives in data/ (not in Git): SIFT1M in data/sift, DEEP10M in data/deep10m
(see docs/deep10m.md). From the project root:

    sbatch scripts/ll_day1_a40.sbatch     # bandwidth + GPU brute force on SIFT1M
    sbatch scripts/ll_day7_a40.sbatch     # build the SIFT1M index, end-to-end check
    sbatch scripts/ll_deep10m_a40.sbatch  # DEEP10M ground truth, index, end-to-end check
    sbatch scripts/rerank_a40.sbatch      # CPU vs GPU rerank

Logs go to logs/, result tables (CSV) to results/.
