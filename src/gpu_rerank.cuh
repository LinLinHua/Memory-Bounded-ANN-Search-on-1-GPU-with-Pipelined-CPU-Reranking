// Exact GPU rerank that reads candidate vectors straight from pinned host memory (zero-copy),
// the comparison point for the CPU rerank.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Same conventions as cpu_rerank.h: -1 candidates are padding, output rows are sorted ascending,
// short rows are filled with id -1 and distance +inf.
struct GpuRerankArgs {
    const float* base;      // n x d, device pointer to mapped pinned host memory
    const float* queries;   // nq x d, device memory
    const int32_t* cand;    // nq x C, device memory
    int64_t d;
    int64_t nq;
    int64_t C;
    int k;
};

// Requires d % 4 == 0 and k <= 32.
void gpu_rerank(const GpuRerankArgs& args, int32_t* out_ids, float* out_dists, cudaStream_t stream);

// Bytes of dynamic shared memory gpu_rerank needs for a given C.
size_t gpu_rerank_smem_bytes(int64_t C);
