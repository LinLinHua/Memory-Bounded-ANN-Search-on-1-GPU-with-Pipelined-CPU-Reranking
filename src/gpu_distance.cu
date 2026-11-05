#include "gpu_distance.cuh"

namespace {

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarpsPerBlock = 8;  // one row per warp
constexpr int kThreads = kWarpsPerBlock * 32;

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(kFullMask, v, offset);
    return v;  // complete in lane 0
}

__global__ void row_sq_norms_kernel(const float* x, int64_t n, int64_t d, float* out) {
    const int lane = threadIdx.x % 32;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / 32;
    if (row >= n) return;

    const float* v = x + row * d;
    float sum = 0.0f;
    for (int64_t i = lane; i < d; i += 32) sum += v[i] * v[i];
    sum = warp_sum(sum);
    if (lane == 0) out[row] = sum;
}

__global__ void topk_reset_kernel(float* best_d, int32_t* best_i, int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        best_d[i] = INFINITY;
        best_i[i] = -1;
    }
}

// Inserts (dist, id) into a warp's sorted list of length k in shared memory.
// All 32 lanes call it with the same arguments.
__device__ void warp_insert(float* list_d, int32_t* list_i, int k, float dist, int32_t id) {
    const int lane = threadIdx.x % 32;

    // New position = number of entries <= dist (on ties the older entry stays first).
    int pos = 0;
    for (int base = 0; base < k; base += 32) {
        const int j = base + lane;
        pos += __popc(__ballot_sync(kFullMask, j < k && list_d[j] <= dist));
    }
    if (pos >= k) return;

    // Shift entries pos .. k-2 one slot right: read them all first, then write.
    constexpr int kSlots = kMaxTopK / 32;
    float moved_d[kSlots];
    int32_t moved_i[kSlots];
#pragma unroll
    for (int s = 0; s < kSlots; s++) {
        const int j = s * 32 + lane;
        if (j > pos && j < k) {
            moved_d[s] = list_d[j - 1];
            moved_i[s] = list_i[j - 1];
        }
    }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < kSlots; s++) {
        const int j = s * 32 + lane;
        if (j > pos && j < k) {
            list_d[j] = moved_d[s];
            list_i[j] = moved_i[s];
        }
    }
    if (lane == 0) {
        list_d[pos] = dist;
        list_i[pos] = id;
    }
    __syncwarp();
}

// One warp per row. Lanes read 32 candidates at a time; a candidate is inserted only if it beats
// the current k-th best. After the list fills up that is rare, so the loop runs at memory speed.
__global__ void topk_update_kernel(const float* dots, int64_t nrows, int64_t ncols,
                                   const float* col_norm, const float* row_norm, int64_t id_offset,
                                   int k, float* best_d, int32_t* best_i) {
    __shared__ float list_d[kWarpsPerBlock][kMaxTopK];
    __shared__ int32_t list_i[kWarpsPerBlock][kMaxTopK];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= nrows) return;  // the whole warp leaves together

    float* my_d = list_d[warp];
    int32_t* my_i = list_i[warp];
    for (int j = lane; j < k; j += 32) {
        my_d[j] = best_d[row * k + j];
        my_i[j] = best_i[row * k + j];
    }
    __syncwarp();

    const float* dot_row = dots + row * ncols;
    const float rn = row_norm ? row_norm[row] : 0.0f;
    for (int64_t base = 0; base < ncols; base += 32) {
        const int64_t j = base + lane;
        const float dist = j < ncols ? col_norm[j] - 2.0f * dot_row[j] + rn : INFINITY;

        unsigned todo = __ballot_sync(kFullMask, dist < my_d[k - 1]);
        while (todo) {
            const int src = __ffs(todo) - 1;
            todo &= todo - 1;
            const float cand = __shfl_sync(kFullMask, dist, src);
            if (cand < my_d[k - 1])  // the threshold may have dropped since the ballot
                warp_insert(my_d, my_i, k, cand, static_cast<int32_t>(id_offset + base + src));
        }
    }

    for (int j = lane; j < k; j += 32) {
        best_d[row * k + j] = my_d[j];
        best_i[row * k + j] = my_i[j];
    }
}

// One warp per query: exact distance for each kept id, then lane 0 sorts the short list.
__global__ void topk_refine_kernel(const float* base, const float* queries, int64_t nq, int64_t d,
                                   int k, float* best_d, int32_t* best_i) {
    __shared__ float list_d[kWarpsPerBlock][kMaxTopK];
    __shared__ int32_t list_i[kWarpsPerBlock][kMaxTopK];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t q = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (q >= nq) return;

    float* my_d = list_d[warp];
    int32_t* my_i = list_i[warp];
    const float* query = queries + q * d;
    for (int j = 0; j < k; j++) {
        const int32_t id = best_i[q * k + j];
        float sum = 0.0f;
        if (id >= 0) {
            const float* v = base + static_cast<int64_t>(id) * d;
            for (int64_t i = lane; i < d; i += 32) {
                const float diff = query[i] - v[i];
                sum += diff * diff;
            }
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            my_d[j] = id >= 0 ? sum : INFINITY;
            my_i[j] = id;
        }
    }
    __syncwarp();

    if (lane == 0) {
        // Insertion sort by (distance, id); k is small.
        for (int j = 1; j < k; j++) {
            const float dv = my_d[j];
            const int32_t iv = my_i[j];
            int p = j;
            while (p > 0 && (my_d[p - 1] > dv || (my_d[p - 1] == dv && my_i[p - 1] > iv))) {
                my_d[p] = my_d[p - 1];
                my_i[p] = my_i[p - 1];
                p--;
            }
            my_d[p] = dv;
            my_i[p] = iv;
        }
        for (int j = 0; j < k; j++) {
            best_d[q * k + j] = my_d[j];
            best_i[q * k + j] = my_i[j];
        }
    }
}

void check_k(int k) {
    if (k < 1 || k > kMaxTopK)
        throw std::runtime_error("top-k needs 1 <= k <= " + std::to_string(kMaxTopK));
}

}  // namespace

void row_sq_norms(const float* x, int64_t n, int64_t d, float* out, cudaStream_t stream) {
    if (n == 0) return;
    row_sq_norms_kernel<<<blocks_for(n, kWarpsPerBlock), kThreads, 0, stream>>>(x, n, d, out);
    CUDA_CHECK(cudaGetLastError());
}

void inner_products(cublasHandle_t handle, const float* a, int64_t na, const float* b, int64_t nb,
                    int64_t d, float* out) {
    // cuBLAS is column-major, and a row-major (rows x d) matrix is a column-major (d x rows) one.
    // We ask for the column-major (nb x na) matrix b^T a; its element (j, i) sits at
    // out[i * nb + j], which is exactly the row-major (na x nb) result we want.
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(nb),
                             static_cast<int>(na), static_cast<int>(d), &one, b,
                             static_cast<int>(d), a, static_cast<int>(d), &zero, out,
                             static_cast<int>(nb)));
}

void topk_reset(float* best_d, int32_t* best_i, int64_t nrows, int k, cudaStream_t stream) {
    const int64_t count = nrows * k;
    if (count == 0) return;
    topk_reset_kernel<<<blocks_for(count, 256), 256, 0, stream>>>(best_d, best_i, count);
    CUDA_CHECK(cudaGetLastError());
}

void topk_update(const float* dots, int64_t nrows, int64_t ncols, const float* col_norm,
                 const float* row_norm, int64_t id_offset, int k, float* best_d, int32_t* best_i,
                 cudaStream_t stream) {
    check_k(k);
    if (nrows == 0) return;
    topk_update_kernel<<<blocks_for(nrows, kWarpsPerBlock), kThreads, 0, stream>>>(
        dots, nrows, ncols, col_norm, row_norm, id_offset, k, best_d, best_i);
    CUDA_CHECK(cudaGetLastError());
}

void topk_refine_exact(const float* base, const float* queries, int64_t nq, int64_t d, int k,
                       float* best_d, int32_t* best_i, cudaStream_t stream) {
    check_k(k);
    if (nq == 0) return;
    topk_refine_kernel<<<blocks_for(nq, kWarpsPerBlock), kThreads, 0, stream>>>(
        base, queries, nq, d, k, best_d, best_i);
    CUDA_CHECK(cudaGetLastError());
}
