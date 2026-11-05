// Exact CPU rerank of GPU candidates: OpenMP across queries, SIMD inside each distance.
#pragma once

#include <cstdint>

// Candidate IDs are original base-vector IDs; slots holding -1 are padding and are skipped.
// Results are written as nq rows of k entries, sorted by increasing distance.
// Rows with fewer than k valid candidates are filled with id -1 and distance +inf.
struct RerankArgs {
    const float* base;      // n x d, host memory
    const float* queries;   // nq x d
    const int32_t* cand;    // nq x C
    int64_t d;
    int64_t nq;
    int64_t C;
    int k;
    int prefetch_dist;      // 0 disables software prefetch
};

void cpu_rerank(const RerankArgs& args, int32_t* out_ids, float* out_dists);
