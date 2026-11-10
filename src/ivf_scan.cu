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
constexpr int kScanThreads = 256;
constexpr int kSelectThreads = 512;
constexpr int kSelectItems = kMaxSelectC / kSelectThreads;  // per thread in the final sort

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
    const float* norms;      // 1-bit
    const float* rotation;   // 1-bit
    int64_t d;
    int nprobe;
    float* est;
    int32_t* est_row;
};

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

// One block per (probe j, query q): estimates the distance from q to every vector in list j.
//
//   8/4-bit : r = sq_min + code * step, so  |q_r - r|^2 = sum_k ((q_r - sq_min)_k - code_k * step_k)^2
//             where q_r = q - centroid is the query residual. Each thread reads one 16-byte chunk
//             of codes at a time (16 dims for 8-bit, 32 dims for 4-bit).
//   1-bit   : estimated angle between the rotated residuals = pi * hamming / d, so
//             |q_r - r|^2 ~ |q_r|^2 + |r|^2 - 2 |q_r| |r| cos(pi * hamming / d)
template <int BITS>
__global__ void scan_kernel(ScanParams p) {
    extern __shared__ float smem[];
    const int64_t d = p.d;
    const int q = blockIdx.y;
    const int j = blockIdx.x;
    const int32_t list = p.probe_list[q * p.nprobe + j];
    if (list < 0) return;  // fewer lists than nprobe

    float* qr = smem;       // query residual
    float* aux = smem + d;  // 8/4-bit: step per dimension; 1-bit: rotated query residual
    const float* query = p.queries + static_cast<int64_t>(q) * d;
    const float* cent = p.centroids + static_cast<int64_t>(list) * d;
    for (int64_t t = threadIdx.x; t < d; t += blockDim.x) qr[t] = query[t] - cent[t];
    __syncthreads();

    const int64_t row0 = p.offsets[list];
    const int64_t size = p.offsets[list + 1] - row0;
    float* est = p.est + p.seg_start[q] + p.probe_start[q * p.nprobe + j];
    int32_t* est_row = p.est_row + p.seg_start[q] + p.probe_start[q * p.nprobe + j];

    if constexpr (BITS == 1) {
        uint32_t* qbits = reinterpret_cast<uint32_t*>(aux + d);
        float* qnorm2 = reinterpret_cast<float*>(qbits + d / 32);

        // Rotate the query residual (row vector times matrix), then take one sign bit per dim.
        for (int64_t t = threadIdx.x; t < d; t += blockDim.x) {
            float s = 0.0f;
            for (int64_t i = 0; i < d; i++) s += qr[i] * p.rotation[i * d + t];
            aux[t] = s;
        }
        __syncthreads();
        const int lane = threadIdx.x % 32;
        for (int64_t base = threadIdx.x - lane; base < d; base += blockDim.x) {  // d % 32 == 0
            const unsigned word = __ballot_sync(kFullMask, aux[base + lane] > 0.0f);
            if (lane == 0) qbits[base / 32] = word;
        }
        if (threadIdx.x == 0) {  // serial sum keeps the result deterministic
            float s = 0.0f;
            for (int64_t i = 0; i < d; i++) s += qr[i] * qr[i];
            *qnorm2 = s;
        }
        __syncthreads();

        const float qn2 = *qnorm2;
        const float qn = sqrtf(qn2);
        const int words = static_cast<int>(d / 32);
        const uint32_t* codes = static_cast<const uint32_t*>(p.codes);
        for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
            const int64_t row = row0 + i;
            const uint32_t* code = codes + row * words;
            int hamming = 0;
            for (int w = 0; w < words; w++) hamming += __popc(code[w] ^ qbits[w]);
            const float rn = p.norms[row];
            est[i] = qn2 + rn * rn - 2.0f * qn * rn * cospif(float(hamming) / float(d));
            est_row[i] = static_cast<int32_t>(row);
        }
    } else {
        for (int64_t t = threadIdx.x; t < d; t += blockDim.x) {
            qr[t] -= p.sq_min[t];
            aux[t] = p.sq_step[t];
        }
        __syncthreads();

        constexpr int kDimsPerLoad = BITS == 8 ? 16 : 32;
        const int64_t row_bytes = d * BITS / 8;
        const uint8_t* codes = static_cast<const uint8_t*>(p.codes);
        for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
            const int64_t row = row0 + i;
            const uint4* code = reinterpret_cast<const uint4*>(codes + row * row_bytes);
            float sum = 0.0f;
            for (int64_t w = 0; w < d / kDimsPerLoad; w++) {
                const uint4 chunk = code[w];
                const uint32_t word[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
                const int64_t dim0 = w * kDimsPerLoad;
#pragma unroll
                for (int b = 0; b < 16; b++) {
                    const uint32_t byte = (word[b / 4] >> (8 * (b % 4))) & 0xffu;
                    if constexpr (BITS == 8) {
                        const float diff = qr[dim0 + b] - float(byte) * aux[dim0 + b];
                        sum += diff * diff;
                    } else {
                        const int64_t k = dim0 + 2 * b;
                        const float lo = qr[k] - float(byte & 0xfu) * aux[k];
                        const float hi = qr[k + 1] - float(byte >> 4) * aux[k + 1];
                        sum += lo * lo + hi * hi;
                    }
                }
            }
            est[i] = sum;
            est_row[i] = static_cast<int32_t>(row);
        }
    }
}

// TopC::Sort: keeps the first C sorted estimates of each query and maps rows to original IDs.
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

// Float -> unsigned key with the same order, so floats can be compared and bucketed as integers.
__device__ inline uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ inline float key_to_float(uint32_t key) {
    return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

// TopC::Select, one block per query:
//   1. radix select: find the C-th smallest estimate, 8 bits at a time (4 histogram passes)
//   2. copy every estimate below it, plus enough that are equal to it, into shared memory
//   3. sort those C entries in shared memory and write them out
// Each pass only reads the query's estimates; nothing else is written to global memory.
__global__ void select_topc_kernel(const float* est, const int32_t* est_row, const int* seg_start,
                                   const int32_t* ids, int C, int32_t* out_ids, float* out_dist) {
    using BlockSort = cub::BlockRadixSort<uint32_t, kSelectThreads, kSelectItems, int32_t>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    __shared__ uint32_t kept_key[kMaxSelectC];
    __shared__ int32_t kept_row[kMaxSelectC];
    __shared__ int hist[256];
    __shared__ uint32_t shared_prefix;
    __shared__ int shared_need, n_below, n_equal;

    const int q = blockIdx.x;
    const int begin = seg_start[q];
    const int len = seg_start[q + 1] - begin;
    const float* e = est + begin;
    const int32_t* r = est_row + begin;
    const int keep = min(C, len);
    const bool take_all = keep == len;

    // ---- 1. after this loop, threshold = key of the keep-th smallest estimate, and
    //         need = how many estimates equal to the threshold must be kept
    uint32_t threshold = 0, mask = 0;
    int need = keep;
    if (!take_all) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < len; i += blockDim.x) {
                const uint32_t key = order_key(e[i]);
                if ((key & mask) == threshold) atomicAdd(&hist[(key >> shift) & 0xffu], 1);
            }
            __syncthreads();
            if (threadIdx.x == 0) {  // the bucket that holds the need-th smallest key
                int before = 0, digit = 0;
                while (before + hist[digit] < need) before += hist[digit++];
                shared_prefix = threshold | (uint32_t(digit) << shift);
                shared_need = need - before;
            }
            __syncthreads();
            threshold = shared_prefix;
            need = shared_need;
            mask |= 0xffu << shift;
        }
    }

    // ---- 2. copy the kept candidates to shared memory (in no particular order yet)
    if (threadIdx.x == 0) n_below = n_equal = 0;
    __syncthreads();
    const int below_total = take_all ? keep : keep - need;  // keys strictly below the threshold
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const uint32_t key = order_key(e[i]);
        int slot = -1;
        if (take_all || key < threshold) {
            slot = atomicAdd(&n_below, 1);
        } else if (key == threshold) {
            const int j = atomicAdd(&n_equal, 1);
            if (j < need) slot = below_total + j;
        }
        if (slot >= 0) {
            kept_key[slot] = key;
            kept_row[slot] = r[i];
        }
    }
    __syncthreads();

    // ---- 3. sort in shared memory; unused slots get the largest key so they sort last
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
        const bool valid = c < keep;
        out_ids[q * C + c] = valid ? ids[rows[k]] : -1;
        out_dist[q * C + c] = valid ? key_to_float(keys[k]) : INFINITY;
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
    if (topc == TopC::Select && max_C > kMaxSelectC)
        throw std::runtime_error("top-C select supports C <= 1024; use the sort version for larger C");

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
        rotation_ = upload(ix_.rotation, "rotation");
    }
    index_bytes_ = budget_.used() - used_before;

    // ---- work buffers, sized for the worst case: every query probes the largest lists
    std::vector<int64_t> sizes(L);
    for (int64_t l = 0; l < L; l++) sizes[l] = ix_.list_size(l);
    std::sort(sizes.begin(), sizes.end(), std::greater<int64_t>());
    int64_t max_segment = 0;
    for (int j = 0; j < max_nprobe; j++) max_segment += sizes[j];
    capacity_ = max_segment * max_batch;
    if (capacity_ > INT_MAX) throw std::runtime_error("batch too large: estimates exceed 2^31");

    queries_ = alloc<float>(int64_t(max_batch) * d, "batch queries");
    dots_ = alloc<float>(int64_t(max_batch) * L, "coarse dots");
    probe_dist_ = alloc<float>(int64_t(max_batch) * max_nprobe, "probe dists");
    probe_list_ = alloc<int32_t>(int64_t(max_batch) * max_nprobe, "probe lists");
    probe_start_ = alloc<int>(int64_t(max_batch) * max_nprobe, "probe starts");
    seg_len_ = alloc<int>(max_batch + 1, "segment lengths");
    seg_start_ = alloc<int>(max_batch + 1, "segment starts");
    est_ = alloc<float>(capacity_, "estimates");
    est_row_ = alloc<int32_t>(capacity_, "estimate rows");
    out_ids_ = alloc<int32_t>(int64_t(max_batch) * max_C, "candidate ids");
    out_dist_ = alloc<float>(int64_t(max_batch) * max_C, "candidate dists");

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

    // ---- 1. coarse search, then where each query's estimates go
    CUDA_CHECK(cudaEventRecord(events_[0], stream_));
    std::memcpy(h_queries_, queries, size_t(nqb) * d * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(queries_, h_queries_, size_t(nqb) * d * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    inner_products(cublas_, queries_, nqb, centroids_, L, d, dots_);
    topk_reset(probe_dist_, probe_list_, nqb, nprobe, stream_);
    topk_update(dots_, nqb, L, centroid_norms_, nullptr, 0, nprobe, probe_dist_, probe_list_, stream_);
    layout_kernel<<<blocks_for(nqb, 128), 128, 0, stream_>>>(probe_list_, nqb, nprobe, offsets_,
                                                             seg_len_, probe_start_);
    CUDA_CHECK(cudaGetLastError());
    size_t temp = cub_temp_bytes_;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(cub_temp_, temp, seg_len_, seg_start_, nqb + 1, stream_));
    CUDA_CHECK(cudaEventRecord(events_[1], stream_));

    // The full sort needs the total number of estimates on the host; the select version does not.
    int total = 0;
    if (topc_ == TopC::Sort) {
        CUDA_CHECK(cudaMemcpyAsync(h_total_, seg_start_ + nqb, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        total = *h_total_;
    }

    // ---- 2. distance estimates for every vector in the probed lists
    ScanParams p{queries_, centroids_, offsets_, probe_list_, probe_start_, seg_start_, codes_,
                 sq_min_, sq_step_, norms_, rotation_, d, nprobe, est_, est_row_};
    const dim3 grid(nprobe, nqb);
    if (bits_ == 1) {
        const size_t smem = (2 * d + d / 32 + 1) * sizeof(float);
        scan_kernel<1><<<grid, kScanThreads, smem, stream_>>>(p);
    } else if (bits_ == 4) {
        scan_kernel<4><<<grid, kScanThreads, 2 * d * sizeof(float), stream_>>>(p);
    } else {
        scan_kernel<8><<<grid, kScanThreads, 2 * d * sizeof(float), stream_>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(events_[2], stream_));

    // ---- 3. top-C per query
    if (topc_ == TopC::Select) {
        select_topc_kernel<<<nqb, kSelectThreads, 0, stream_>>>(est_, est_row_, seg_start_, ids_, C,
                                                                out_ids_, out_dist_);
    } else {
        temp = 0;
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
