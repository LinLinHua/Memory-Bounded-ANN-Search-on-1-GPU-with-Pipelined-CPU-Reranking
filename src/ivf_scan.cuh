// GPU IVF scan for a batch of queries (LL).
//
//   1. coarse search : GEMM against the centroids, keep the nprobe closest lists
//   2. scan          : estimate the distance to every vector in those lists from its 8/4/1-bit code
//   3. top-C         : keep the C smallest estimates per query, in one of three ways:
//        TopC::Fused  - the scan keeps each list's C best in shared memory and writes only those;
//                       a final pass picks the query's C best from nprobe * C (default)
//        TopC::Select - the scan writes every estimate; radix select per query (Day 11)
//        TopC::Sort   - the scan writes every estimate; CUB segmented sort per query (Day 7)
//
// Output follows the team's candidate format: original vector IDs, sorted by estimated distance,
// padded with id -1 / distance +inf when a query sees fewer than C vectors.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "device_budget.h"
#include "ivf_index.h"

enum class TopC { Fused, Select, Sort };

constexpr int kMaxSelectC = 1024;  // largest C that TopC::Fused and TopC::Select support

struct ScanTiming {  // milliseconds, summed over all batches of the last search()
    double coarse = 0;  // query upload + coarse search + segment layout
    double scan = 0;    // distance estimation (plus the per-list top-C for TopC::Fused)
    double select = 0;  // per-query top-C + candidate download
    double total = 0;   // wall clock of search()
};

class IvfScanner {
public:
    // Copies what the chosen code width needs to the GPU. All device memory comes from budget,
    // with work buffers sized for max_batch queries, max_nprobe lists and max_C candidates.
    IvfScanner(const IvfIndex& index, int bits, int max_batch, int max_nprobe, int max_C,
               DeviceBudget& budget, TopC topc = TopC::Fused);
    ~IvfScanner();
    IvfScanner(const IvfScanner&) = delete;
    IvfScanner& operator=(const IvfScanner&) = delete;

    // queries: nq x d on the host. Writes nq x C candidate IDs and estimated distances.
    void search(const float* queries, int64_t nq, int nprobe, int C, int32_t* cand_ids, float* cand_dist);

    const ScanTiming& timing() const { return timing_; }
    size_t index_bytes() const { return index_bytes_; }  // device bytes of the index itself
    size_t work_bytes() const { return work_bytes_; }    // device bytes of the work buffers

private:
    template <typename T>
    T* alloc(int64_t count, const char* tag);
    template <typename T>
    T* upload(const std::vector<T>& v, const char* tag);
    void search_batch(const float* queries, int nqb, int nprobe, int C, int32_t* cand_ids, float* cand_dist);

    const IvfIndex& ix_;
    const int bits_, max_batch_, max_nprobe_, max_C_;
    const TopC topc_;
    DeviceBudget& budget_;
    std::vector<void*> owned_;  // freed through the budget in the destructor
    size_t index_bytes_ = 0, work_bytes_ = 0;
    int64_t capacity_ = 0;      // estimates per batch in the worst case (Select and Sort)
    size_t fused_smem_ = 0;     // shared memory per block of the fused scan
    ScanTiming timing_;

    cublasHandle_t cublas_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t events_[4] = {};

    // index on the device
    float* centroids_ = nullptr;
    float* centroid_norms_ = nullptr;
    int64_t* offsets_ = nullptr;
    int32_t* ids_ = nullptr;
    void* codes_ = nullptr;
    float* sq_min_ = nullptr;      // 8/4-bit
    float* sq_step_ = nullptr;     // 8/4-bit
    float* norms_ = nullptr;       // 1-bit
    float* factors_ = nullptr;     // 1-bit
    float* rotation_ = nullptr;    // 1-bit
    float* rot_centroids_ = nullptr;  // 1-bit: R^T c per list

    // work buffers
    float* queries_ = nullptr;      // batch x d
    float* rot_queries_ = nullptr;  // 1-bit: R^T q, batch x d
    float* dots_ = nullptr;         // batch x nlist
    float* probe_dist_ = nullptr;   // batch x nprobe
    int32_t* probe_list_ = nullptr; // batch x nprobe
    int* probe_start_ = nullptr;    // batch x nprobe; where each list's estimates start (Select, Sort)
    int* seg_len_ = nullptr;        // batch + 1 (Select, Sort)
    int* seg_start_ = nullptr;      // batch + 1; exclusive prefix sum of seg_len (Select, Sort)
    float* est_ = nullptr;          // estimates: capacity (Select, Sort) or batch x nprobe x C (Fused)
    int32_t* est_row_ = nullptr;    // row of each estimate
    float* sorted_est_ = nullptr;   // Sort only
    int32_t* sorted_row_ = nullptr; // Sort only
    void* cub_temp_ = nullptr;
    size_t cub_temp_bytes_ = 0;
    int32_t* out_ids_ = nullptr;    // batch x C
    float* out_dist_ = nullptr;

    // pinned host staging
    float* h_queries_ = nullptr;
    int32_t* h_ids_ = nullptr;
    float* h_dist_ = nullptr;
    int* h_total_ = nullptr;
};
