// GPU building blocks shared by brute force, k-means and the IVF scan.
//
// Squared L2 distances come from one GEMM:  |q - x|^2 = |q|^2 - 2 q.x + |x|^2.
// Top-k selection keeps a sorted list per row and merges GEMM tiles into it.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_budget.h"  // CUDA_CHECK

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        const cublasStatus_t st_ = (call);                                    \
        if (st_ != CUBLAS_STATUS_SUCCESS)                                     \
            throw std::runtime_error(std::string(#call) + ": cuBLAS error " + \
                                     std::to_string(static_cast<int>(st_)));  \
    } while (0)

constexpr int kMaxTopK = 256;  // largest k that the top-k functions support

inline unsigned blocks_for(int64_t work, int64_t per_block) {
    return static_cast<unsigned>((work + per_block - 1) / per_block);
}

// out[i] = |x_i|^2 for the n rows of x (n x d).
void row_sq_norms(const float* x, int64_t n, int64_t d, float* out, cudaStream_t stream = 0);

// out[i * nb + j] = dot(a_i, b_j) for row-major a (na x d) and b (nb x d).
// Runs on the stream set on the cuBLAS handle.
void inner_products(cublasHandle_t handle, const float* a, int64_t na, const float* b, int64_t nb,
                    int64_t d, float* out);

// Empties nrows top-k lists: every slot becomes (distance +inf, id -1).
void topk_reset(float* best_d, int32_t* best_i, int64_t nrows, int k, cudaStream_t stream = 0);

// Merges one GEMM tile into each row's top-k list (kept sorted, smallest first).
// Candidate j of row r has
//   distance = col_norm[j] - 2 * dots[r * ncols + j] + row_norm[r]   (row_norm may be null)
//   id       = id_offset + j
void topk_update(const float* dots, int64_t nrows, int64_t ncols, const float* col_norm,
                 const float* row_norm, int64_t id_offset, int k, float* best_d, int32_t* best_i,
                 cudaStream_t stream = 0);

// Recomputes the distances in each top-k list directly from the vectors and re-sorts the list,
// so the final answer has no GEMM rounding error.
void topk_refine_exact(const float* base, const float* queries, int64_t nq, int64_t d, int k,
                       float* best_d, int32_t* best_i, cudaStream_t stream = 0);

// Times GPU work on one stream with a pair of events.
class GpuTimer {
public:
    explicit GpuTimer(cudaStream_t stream = 0) : stream_(stream) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

    // Waits for the GPU, then returns the elapsed time in milliseconds.
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaStream_t stream_;
    cudaEvent_t start_, stop_;
};
