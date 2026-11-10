#include "ivf_scan.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <functional>

#include "gpu_distance.cuh"

namespace {

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kScanThreads = 256;    // scan kernel that writes every estimate (Select, Sort)
constexpr int kFusedThreads = 512;   // fused scan + per-list top-C
constexpr int kSelectThreads = 512;  // per-query top-C
constexpr int kSelectItems = kMaxSelectC / kSelectThreads;
constexpr int kMaxSharedBytes = 99 * 1024;  // shared memory one block may use on the A40

struct ScanParams {
    const float* queries;    // batch x d
    const float* centroids;  // nlist x d
    const int64_t* offsets;  // nlist + 1
    const int32_t* probe_list;
    const int* probe_start;
    const int* seg_start;
    const void* codes;
    const float* sq_min;     // 8/4-bit
    const float* sq_step;    // 8/4-bit
    const float* norms;      // 1-bit: |residual|
    const float* factors;    // 1-bit: <sign code / sqrt(d), unit residual>
    const float* rot_queries;    // 1-bit: R^T q for the batch (batch x d)
    const float* rot_centroids;  // 1-bit: R^T c for every list (nlist x d)
    int64_t d;
    int nprobe;
    float* est;
    int32_t* est_row;
};

// ------------------------------------------------------------------------------------------------
// Distance estimation, shared by both scan kernels
//
// Per (query, list) the block first prepares the query in shared memory, laid out as
//   qv[d], aux[d], then for 1-bit: 4 bit planes of d/32 words and 4 scalars
//
//   8/4-bit : qv = q_r - sq_min and aux = step, where q_r = q - centroid. A residual decodes as
//             r = sq_min + code * step, so |q_r - r|^2 = sum_k (qv_k - code_k * aux_k)^2.
//   1-bit   : RaBitQ estimator. qv = R^T q_r = R^T q - R^T c (rotated query residual, from rotations
//             done once per query and once per centroid), kept at 4 bits as bit planes.
//             For a vector with rotated residual r', sign code b and norm |r|:
//               <r'/|r|, qv> ~ <b, qv> / sqrt(d) / factor,  factor = <b / sqrt(d), r'/|r|>
//               |q_r - r|^2 ~ |q_r|^2 + |r|^2 - 2 |r| <r'/|r|, qv>
//             <b, qv> needs one AND + popcount per bit plane and 32 dimensions.
// ------------------------------------------------------------------------------------------------

__host__ __device__ inline int64_t query_state_floats(int64_t d) { return 2 * d + 4 * (d / 32) + 4; }

__device__ inline int quantize4(float v, float lo, float step) {
    return min(15, max(0, __float2int_rn((v - lo) / step)));
}

template <int BITS>
__device__ void prepare_query(const ScanParams& p, int q, int32_t list, float* s) {
    const int64_t d = p.d;
    float* qv = s;
    float* aux = s + d;
    const float* query = p.queries + static_cast<int64_t>(q) * d;
    const float* cent = p.centroids + static_cast<int64_t>(list) * d;

    if constexpr (BITS == 1) {
        uint32_t* planes = reinterpret_cast<uint32_t*>(s + 2 * d);
        float* scalars = s + 2 * d + 4 * (d / 32);  // |q_r|^2, lowest value, step, sum of the 4-bit query

        const float* rq = p.rot_queries + static_cast<int64_t>(q) * d;
        const float* rc = p.rot_centroids + static_cast<int64_t>(list) * d;
        for (int64_t t = threadIdx.x; t < d; t += blockDim.x) {
            aux[t] = query[t] - cent[t];
            qv[t] = rq[t] - rc[t];
        }
        __syncthreads();
        if (threadIdx.x == 0) {  // serial loops keep the result deterministic
            float norm2 = 0.0f, lo = qv[0], hi = qv[0];
            for (int64_t i = 0; i < d; i++) {
                norm2 += aux[i] * aux[i];
                lo = fminf(lo, qv[i]);
                hi = fmaxf(hi, qv[i]);
            }
            const float step = hi > lo ? (hi - lo) / 15.0f : 1.0f;
            int sum_u = 0;
            for (int64_t i = 0; i < d; i++) sum_u += quantize4(qv[i], lo, step);
            scalars[0] = norm2;
            scalars[1] = lo;
            scalars[2] = step;
            scalars[3] = float(d) * lo + step * float(sum_u);
        }
        __syncthreads();
        // Bit plane b holds bit b of every 4-bit query value, 32 dimensions per word (d % 32 == 0).
        const int lane = threadIdx.x % 32;
        for (int64_t base = threadIdx.x - lane; base < d; base += blockDim.x) {
            const int u = quantize4(qv[base + lane], scalars[1], scalars[2]);
            for (int b = 0; b < 4; b++) {
                const unsigned word = __ballot_sync(kFullMask, (u >> b) & 1);
                if (lane == 0) planes[b * (d / 32) + base / 32] = word;
            }
        }
    } else {
        for (int64_t t = threadIdx.x; t < d; t += blockDim.x) {
            qv[t] = query[t] - cent[t] - p.sq_min[t];
            aux[t] = p.sq_step[t];
        }
    }
    __syncthreads();
}

template <int BITS>
__device__ inline float estimate(const ScanParams& p, int64_t row, const float* s) {
    const int64_t d = p.d;
    const float* qv = s;
    const float* aux = s + d;

    if constexpr (BITS == 1) {
        const int words = static_cast<int>(d / 32);
        const uint32_t* planes = reinterpret_cast<const uint32_t*>(s + 2 * d);
        const float* scalars = s + 2 * d + 4 * (d / 32);
        const uint32_t* code = static_cast<const uint32_t*>(p.codes) + row * words;
        int ones = 0, weighted = 0;  // set bits, and the 4-bit query summed over set bits
        for (int w = 0; w < words; w++) {
            const uint32_t c = code[w];
            ones += __popc(c);
            weighted += __popc(c & planes[w]) + 2 * __popc(c & planes[words + w]) +
                        4 * __popc(c & planes[2 * words + w]) + 8 * __popc(c & planes[3 * words + w]);
        }
        const float set_sum = scalars[1] * float(ones) + scalars[2] * float(weighted);
        const float dot_sign = 2.0f * set_sum - scalars[3];  // <b, qv> with b in {-1, +1}
        const float ip = dot_sign * (1.0f / sqrtf(float(d))) / p.factors[row];
        const float rn = p.norms[row];
        return scalars[0] + rn * rn - 2.0f * rn * ip;
    } else {
        constexpr int kDimsPerLoad = BITS == 8 ? 16 : 32;  // one 16-byte load
        const uint8_t* bytes = static_cast<const uint8_t*>(p.codes) + row * (d * BITS / 8);
        const uint4* code = reinterpret_cast<const uint4*>(bytes);
        float sum = 0.0f;
        for (int64_t w = 0; w < d / kDimsPerLoad; w++) {
            const uint4 chunk = code[w];
            const uint32_t word[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            const int64_t dim0 = w * kDimsPerLoad;
#pragma unroll
            for (int b = 0; b < 16; b++) {
                const uint32_t byte = (word[b / 4] >> (8 * (b % 4))) & 0xffu;
                if constexpr (BITS == 8) {
                    const float diff = qv[dim0 + b] - float(byte) * aux[dim0 + b];
                    sum += diff * diff;
                } else {
                    const int64_t k = dim0 + 2 * b;  // low nibble = even dimension
                    const float lo = qv[k] - float(byte & 0xfu) * aux[k];
                    const float hi = qv[k + 1] - float(byte >> 4) * aux[k + 1];
                    sum += lo * lo + hi * hi;
                }
            }
        }
        return sum;
    }
}

// ------------------------------------------------------------------------------------------------
// Block-wide selection of the k smallest values
// ------------------------------------------------------------------------------------------------

// Float -> unsigned key with the same order, so floats can be compared and bucketed as integers.
__device__ inline uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ inline float key_to_float(uint32_t key) {
    return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

struct SelectScratch {
    int hist[256];
    uint32_t prefix;
    int need, n_below, n_equal;
};

// Calls emit(slot, i) for the keep smallest of len values, giving each one a distinct slot in
// [0, keep) (in no particular order). key_at(i) returns the order key of value i.
// Radix select: find the keep-th smallest key 8 bits at a time (4 histogram passes), then take
// every key below it and enough keys equal to it. All threads of the block must call this.
template <typename KeyAt, typename Emit>
__device__ void keep_smallest(int len, int keep, KeyAt key_at, SelectScratch& s, Emit emit) {
    const bool take_all = keep >= len;
    uint32_t threshold = 0, mask = 0;
    int need = keep;
    if (!take_all) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int b = threadIdx.x; b < 256; b += blockDim.x) s.hist[b] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < len; i += blockDim.x) {
                const uint32_t key = key_at(i);
                if ((key & mask) == threshold) atomicAdd(&s.hist[(key >> shift) & 0xffu], 1);
            }
            __syncthreads();
            if (threadIdx.x == 0) {  // the bucket that holds the need-th smallest key
                int before = 0, digit = 0;
                while (before + s.hist[digit] < need) before += s.hist[digit++];
                s.prefix = threshold | (uint32_t(digit) << shift);
                s.need = need - before;
            }
            __syncthreads();
            threshold = s.prefix;
            need = s.need;
            mask |= 0xffu << shift;
        }
    }

    if (threadIdx.x == 0) s.n_below = s.n_equal = 0;
    __syncthreads();
    const int below_total = take_all ? len : keep - need;  // keys strictly below the threshold
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const uint32_t key = key_at(i);
        int slot = -1;
        if (take_all || key < threshold) {
            slot = atomicAdd(&s.n_below, 1);
        } else if (key == threshold) {
            const int j = atomicAdd(&s.n_equal, 1);
            if (j < need) slot = below_total + j;
        }
        if (slot >= 0) emit(slot, i);
    }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------------------------------------

// out[r][t] = sum_i x[r][i] * R[i][t], one thread per output, with an explicit fused multiply-add in a
// fixed order so the CPU check (std::fmaf, same order) gets bit-identical values.
__global__ void rotate_rows_kernel(const float* x, int64_t rows, int64_t d, const float* R, float* out) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * d) return;
    const int64_t r = idx / d, t = idx % d;
    float v = 0.0f;
    for (int64_t i = 0; i < d; i++) v = fmaf(x[r * d + i], R[i * d + t], v);
    out[idx] = v;
}

// Lays out each query's segment: the estimates of probe j start at probe_start[q][j].
__global__ void layout_kernel(const int32_t* probe_list, int nqb, int nprobe, const int64_t* offsets,
                              int* seg_len, int* probe_start) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q == 0) seg_len[nqb] = 0;  // so the exclusive sum ends with the total
    if (q >= nqb) return;
    int run = 0;
    for (int j = 0; j < nprobe; j++) {
        probe_start[q * nprobe + j] = run;
        const int32_t list = probe_list[q * nprobe + j];
        if (list >= 0) run += static_cast<int>(offsets[list + 1] - offsets[list]);
    }
    seg_len[q] = run;
}

// Select and Sort: one block per (probe j, query q) writes the estimate of every vector in list j.
template <int BITS>
__global__ void scan_kernel(ScanParams p) {
    extern __shared__ float smem[];
    const int q = blockIdx.y, j = blockIdx.x;
    const int32_t list = p.probe_list[q * p.nprobe + j];
    if (list < 0) return;  // fewer lists than nprobe
    prepare_query<BITS>(p, q, list, smem);

    const int64_t row0 = p.offsets[list];
    const int64_t size = p.offsets[list + 1] - row0;
    const int64_t out = p.seg_start[q] + p.probe_start[q * p.nprobe + j];
    for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
        p.est[out + i] = estimate<BITS>(p, row0 + i, smem);
        p.est_row[out + i] = static_cast<int32_t>(row0 + i);
    }
}

// Fused: one block per (probe j, query q) keeps the list's estimates in shared memory and writes
// only the list's C best to slots [(q * nprobe + j) * C, ... + C). The query's C best are always
// among the per-list C bests, so the final selection gives the same answer as over every estimate.
template <int BITS>
__global__ void scan_fused_kernel(ScanParams p, int C, float* cand_est, int32_t* cand_row) {
    extern __shared__ float smem[];
    __shared__ SelectScratch scratch;
    const int q = blockIdx.y, j = blockIdx.x;
    const int64_t base = (static_cast<int64_t>(q) * p.nprobe + j) * C;
    const int32_t list = p.probe_list[q * p.nprobe + j];

    int keep = 0;
    if (list >= 0) {  // the same for the whole block
        prepare_query<BITS>(p, q, list, smem);
        float* est = smem + query_state_floats(p.d);
        const int64_t row0 = p.offsets[list];
        const int size = static_cast<int>(p.offsets[list + 1] - row0);
        for (int i = threadIdx.x; i < size; i += blockDim.x) est[i] = estimate<BITS>(p, row0 + i, smem);
        __syncthreads();
        keep = min(C, size);
        keep_smallest(size, keep, [&](int i) { return order_key(est[i]); }, scratch,
                      [&](int slot, int i) {
                          cand_est[base + slot] = est[i];
                          cand_row[base + slot] = static_cast<int32_t>(row0 + i);
                      });
    }
    for (int c = keep + threadIdx.x; c < C; c += blockDim.x) {  // padding
        cand_est[base + c] = INFINITY;
        cand_row[base + c] = -1;
    }
}

// Per-query top-C, one block per query: radix select, then a sort of the C survivors in shared
// memory. Query q's estimates are est[begin, begin + len): from seg_start (Select), or a fixed
// stride of nprobe * C (Fused). Rows of -1 are padding.
__global__ void select_topc_kernel(const float* est, const int32_t* est_row, const int* seg_start,
                                   int stride, const int32_t* ids, int C, int32_t* out_ids, float* out_dist) {
    using BlockSort = cub::BlockRadixSort<uint32_t, kSelectThreads, kSelectItems, int32_t>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    __shared__ uint32_t kept_key[kMaxSelectC];
    __shared__ int32_t kept_row[kMaxSelectC];
    __shared__ SelectScratch scratch;

    const int q = blockIdx.x;
    const int64_t begin = seg_start ? seg_start[q] : static_cast<int64_t>(q) * stride;
    const int len = seg_start ? static_cast<int>(seg_start[q + 1] - begin) : stride;
    const float* e = est + begin;
    const int32_t* r = est_row + begin;
    const int keep = min(C, len);
    keep_smallest(len, keep, [&](int i) { return order_key(e[i]); }, scratch, [&](int slot, int i) {
        kept_key[slot] = order_key(e[i]);
        kept_row[slot] = r[i];
    });

    // Unused slots get the largest key so they sort last.
    uint32_t keys[kSelectItems];
    int32_t rows[kSelectItems];
    for (int k = 0; k < kSelectItems; k++) {
        const int slot = threadIdx.x * kSelectItems + k;
        keys[k] = slot < keep ? kept_key[slot] : 0xffffffffu;
        rows[k] = slot < keep ? kept_row[slot] : -1;
    }
    BlockSort(sort_storage).Sort(keys, rows);
    for (int k = 0; k < kSelectItems; k++) {
        const int c = threadIdx.x * kSelectItems + k;
        if (c >= C) continue;
        const bool valid = c < keep && rows[k] >= 0;
        out_ids[q * C + c] = valid ? ids[rows[k]] : -1;
        out_dist[q * C + c] = valid ? key_to_float(keys[k]) : INFINITY;
    }
}

// Sort: keeps the first C sorted estimates of each query and maps rows to original IDs.
__global__ void gather_kernel(const float* sorted_est, const int32_t* sorted_row, const int* seg_start,
                              const int32_t* ids, int C, int32_t* out_ids, float* out_dist) {
    const int q = blockIdx.x;
    const int begin = seg_start[q];
    const int len = seg_start[q + 1] - begin;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const bool valid = c < len;
        out_ids[q * C + c] = valid ? ids[sorted_row[begin + c]] : -1;
        out_dist[q * C + c] = valid ? sorted_est[begin + c] : INFINITY;
    }
}

template <int BITS>
void launch_scan(TopC topc, dim3 grid, size_t state_bytes, size_t fused_bytes, int C, const ScanParams& p,
                 cudaStream_t stream) {
    if (topc == TopC::Fused) {
        scan_fused_kernel<BITS><<<grid, kFusedThreads, fused_bytes, stream>>>(p, C, p.est, p.est_row);
    } else {
        scan_kernel<BITS><<<grid, kScanThreads, state_bytes, stream>>>(p);
    }
}

}  // namespace

template <typename T>
T* IvfScanner::alloc(int64_t count, const char* tag) {
    T* ptr = budget_.alloc<T>(static_cast<size_t>(count), tag);
    owned_.push_back(ptr);
    return ptr;
}

template <typename T>
T* IvfScanner::upload(const std::vector<T>& v, const char* tag) {
    T* ptr = alloc<T>(v.size(), tag);
    CUDA_CHECK(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

IvfScanner::IvfScanner(const IvfIndex& index, int bits, int max_batch, int max_nprobe, int max_C,
                       DeviceBudget& budget, TopC topc)
    : ix_(index), bits_(bits), max_batch_(max_batch), max_nprobe_(max_nprobe), max_C_(max_C),
      topc_(topc), budget_(budget) {
    const int64_t d = ix_.d, L = ix_.nlist;
    if (bits != 1 && bits != 4 && bits != 8) throw std::runtime_error("bits must be 1, 4 or 8");
    if (d % 32 != 0) throw std::runtime_error("the scan needs d to be a multiple of 32");
    if (max_nprobe < 1 || max_nprobe > std::min<int64_t>(L, kMaxTopK))
        throw std::runtime_error("nprobe must be between 1 and min(nlist, 256)");
    if (topc != TopC::Sort && max_C > kMaxSelectC)
        throw std::runtime_error("fused and select top-C support C <= 1024; use --topc sort for larger C");
    if (bits == 1 && ix_.factors1.empty())
        throw std::runtime_error("index has no factors1.f32 (1-bit factors); rebuild it with ivf_build");

    std::vector<int64_t> sizes(L);
    for (int64_t l = 0; l < L; l++) sizes[l] = ix_.list_size(l);
    std::sort(sizes.begin(), sizes.end(), std::greater<int64_t>());
    if (topc == TopC::Fused) {  // a whole list's estimates must fit in one block's shared memory
        fused_smem_ = (query_state_floats(d) + sizes[0]) * sizeof(float);
        if (fused_smem_ > kMaxSharedBytes)
            throw std::runtime_error("largest list does not fit in shared memory; use --topc select");
        auto* kernel = bits == 1 ? scan_fused_kernel<1> : bits == 4 ? scan_fused_kernel<4> : scan_fused_kernel<8>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(fused_smem_)));
    }

    CUDA_CHECK(cudaStreamCreate(&stream_));
    for (auto& e : events_) CUDA_CHECK(cudaEventCreate(&e));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));

    // ---- index: only the parts this code width reads
    const size_t used_before = budget_.used();
    centroids_ = upload(ix_.centroids, "centroids");
    centroid_norms_ = alloc<float>(L, "centroid norms");
    row_sq_norms(centroids_, L, d, centroid_norms_);
    offsets_ = upload(ix_.offsets, "list offsets");
    ids_ = upload(ix_.ids, "row ids");
    if (bits == 8) {
        codes_ = upload(ix_.codes8, "codes8");
        sq_min_ = upload(ix_.sq_min, "sq min");
        sq_step_ = upload(ix_.sq_step8, "sq step8");
    } else if (bits == 4) {
        codes_ = upload(ix_.codes4, "codes4");
        sq_min_ = upload(ix_.sq_min, "sq min");
        sq_step_ = upload(ix_.sq_step4, "sq step4");
    } else {
        codes_ = upload(ix_.codes1, "codes1");
        norms_ = upload(ix_.norms, "residual norms");
        factors_ = upload(ix_.factors1, "1-bit factors");
        rotation_ = upload(ix_.rotation, "rotation");
        rot_centroids_ = alloc<float>(L * d, "rotated centroids");
        rotate_rows_kernel<<<blocks_for(L * d, 256), 256>>>(centroids_, L, d, rotation_, rot_centroids_);
        CUDA_CHECK(cudaGetLastError());
    }
    index_bytes_ = budget_.used() - used_before;

    // ---- work buffers
    queries_ = alloc<float>(int64_t(max_batch) * d, "batch queries");
    if (bits == 1) rot_queries_ = alloc<float>(int64_t(max_batch) * d, "rotated batch queries");
    dots_ = alloc<float>(int64_t(max_batch) * L, "coarse dots");
    probe_dist_ = alloc<float>(int64_t(max_batch) * max_nprobe, "probe dists");
    probe_list_ = alloc<int32_t>(int64_t(max_batch) * max_nprobe, "probe lists");
    out_ids_ = alloc<int32_t>(int64_t(max_batch) * max_C, "candidate ids");
    out_dist_ = alloc<float>(int64_t(max_batch) * max_C, "candidate dists");

    if (topc == TopC::Fused) {  // only C candidates per (query, list) leave the scan
        capacity_ = int64_t(max_batch) * max_nprobe * max_C;
        est_ = alloc<float>(capacity_, "per-list candidates");
        est_row_ = alloc<int32_t>(capacity_, "per-list candidate rows");
    } else {  // every estimate of the batch is written; size for the largest lists
        int64_t max_segment = 0;
        for (int j = 0; j < max_nprobe; j++) max_segment += sizes[j];
        capacity_ = max_segment * max_batch;
        if (capacity_ > INT_MAX) throw std::runtime_error("batch too large: estimates exceed 2^31");
        probe_start_ = alloc<int>(int64_t(max_batch) * max_nprobe, "probe starts");
        seg_len_ = alloc<int>(max_batch + 1, "segment lengths");
        seg_start_ = alloc<int>(max_batch + 1, "segment starts");
        est_ = alloc<float>(capacity_, "estimates");
        est_row_ = alloc<int32_t>(capacity_, "estimate rows");

        size_t sort_bytes = 0, scan_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, seg_len_, seg_start_, max_batch + 1, stream_));
        if (topc == TopC::Sort) {  // the full sort needs a second copy of the estimates plus temp space
            sorted_est_ = alloc<float>(capacity_, "sorted estimates");
            sorted_row_ = alloc<int32_t>(capacity_, "sorted rows");
            CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, sort_bytes, est_, sorted_est_, est_row_, sorted_row_, static_cast<int>(capacity_),
                max_batch, seg_start_, seg_start_ + 1, 0, 32, stream_));
        }
        cub_temp_bytes_ = std::max(sort_bytes, scan_bytes);
        cub_temp_ = alloc<char>(cub_temp_bytes_, "cub temp");
    }
    work_bytes_ = budget_.used() - used_before - index_bytes_;

    CUDA_CHECK(cudaMallocHost(&h_queries_, size_t(max_batch) * d * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_ids_, size_t(max_batch) * max_C * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&h_dist_, size_t(max_batch) * max_C * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_total_, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    budget_.sample();
}

IvfScanner::~IvfScanner() {
    cudaFreeHost(h_queries_);
    cudaFreeHost(h_ids_);
    cudaFreeHost(h_dist_);
    cudaFreeHost(h_total_);
    for (void* ptr : owned_) budget_.free(ptr);
    for (auto& e : events_) cudaEventDestroy(e);
    cublasDestroy(cublas_);
    cudaStreamDestroy(stream_);
}

void IvfScanner::search(const float* queries, int64_t nq, int nprobe, int C, int32_t* cand_ids,
                        float* cand_dist) {
    if (nprobe < 1 || nprobe > max_nprobe_) throw std::runtime_error("nprobe above max_nprobe");
    if (C < 1 || C > max_C_) throw std::runtime_error("C above max_C");
    timing_ = ScanTiming{};
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t q0 = 0; q0 < nq; q0 += max_batch_) {
        const int nqb = static_cast<int>(std::min<int64_t>(max_batch_, nq - q0));
        search_batch(queries + q0 * ix_.d, nqb, nprobe, C, cand_ids + q0 * C, cand_dist + q0 * C);
    }
    timing_.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void IvfScanner::search_batch(const float* queries, int nqb, int nprobe, int C, int32_t* cand_ids,
                              float* cand_dist) {
    const int64_t d = ix_.d, L = ix_.nlist;

    // ---- 1. coarse search (and, for Select and Sort, where each query's estimates go)
    CUDA_CHECK(cudaEventRecord(events_[0], stream_));
    std::memcpy(h_queries_, queries, size_t(nqb) * d * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(queries_, h_queries_, size_t(nqb) * d * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    inner_products(cublas_, queries_, nqb, centroids_, L, d, dots_);
    topk_reset(probe_dist_, probe_list_, nqb, nprobe, stream_);
    topk_update(dots_, nqb, L, centroid_norms_, nullptr, 0, nprobe, probe_dist_, probe_list_, stream_);
    if (bits_ == 1) {
        rotate_rows_kernel<<<blocks_for(int64_t(nqb) * d, 256), 256, 0, stream_>>>(queries_, nqb, d, rotation_,
                                                                                  rot_queries_);
        CUDA_CHECK(cudaGetLastError());
    }
    int total = 0;
    if (topc_ != TopC::Fused) {
        layout_kernel<<<blocks_for(nqb, 128), 128, 0, stream_>>>(probe_list_, nqb, nprobe, offsets_,
                                                                 seg_len_, probe_start_);
        CUDA_CHECK(cudaGetLastError());
        size_t temp = cub_temp_bytes_;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(cub_temp_, temp, seg_len_, seg_start_, nqb + 1, stream_));
        if (topc_ == TopC::Sort) {  // the full sort needs the number of estimates on the host
            CUDA_CHECK(cudaMemcpyAsync(h_total_, seg_start_ + nqb, sizeof(int), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            total = *h_total_;
        }
    }
    CUDA_CHECK(cudaEventRecord(events_[1], stream_));

    // ---- 2. distance estimates
    const ScanParams p{queries_, centroids_, offsets_, probe_list_, probe_start_, seg_start_, codes_,
                       sq_min_, sq_step_, norms_, factors_, rot_queries_, rot_centroids_, d, nprobe,
                       est_, est_row_};
    const dim3 grid(nprobe, nqb);
    const size_t state_bytes = query_state_floats(d) * sizeof(float);
    if (bits_ == 1) launch_scan<1>(topc_, grid, state_bytes, fused_smem_, C, p, stream_);
    else if (bits_ == 4) launch_scan<4>(topc_, grid, state_bytes, fused_smem_, C, p, stream_);
    else launch_scan<8>(topc_, grid, state_bytes, fused_smem_, C, p, stream_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(events_[2], stream_));

    // ---- 3. top-C per query
    if (topc_ == TopC::Fused) {
        select_topc_kernel<<<nqb, kSelectThreads, 0, stream_>>>(est_, est_row_, nullptr, nprobe * C, ids_, C,
                                                                out_ids_, out_dist_);
    } else if (topc_ == TopC::Select) {
        select_topc_kernel<<<nqb, kSelectThreads, 0, stream_>>>(est_, est_row_, seg_start_, 0, ids_, C,
                                                                out_ids_, out_dist_);
    } else {
        size_t temp = 0;
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp, est_, sorted_est_, est_row_,
                                                            sorted_row_, total, nqb, seg_start_,
                                                            seg_start_ + 1, 0, 32, stream_));
        if (temp > cub_temp_bytes_) throw std::runtime_error("CUB needs more temp memory than reserved");
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(cub_temp_, temp, est_, sorted_est_, est_row_,
                                                            sorted_row_, total, nqb, seg_start_,
                                                            seg_start_ + 1, 0, 32, stream_));
        gather_kernel<<<nqb, 128, 0, stream_>>>(sorted_est_, sorted_row_, seg_start_, ids_, C, out_ids_,
                                                out_dist_);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_ids_, out_ids_, size_t(nqb) * C * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_dist_, out_dist_, size_t(nqb) * C * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(events_[3], stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    budget_.sample();

    std::memcpy(cand_ids, h_ids_, size_t(nqb) * C * sizeof(int32_t));
    std::memcpy(cand_dist, h_dist_, size_t(nqb) * C * sizeof(float));
    float ms[3];
    for (int i = 0; i < 3; i++) CUDA_CHECK(cudaEventElapsedTime(&ms[i], events_[i], events_[i + 1]));
    timing_.coarse += ms[0];
    timing_.scan += ms[1];
    timing_.select += ms[2];
}
