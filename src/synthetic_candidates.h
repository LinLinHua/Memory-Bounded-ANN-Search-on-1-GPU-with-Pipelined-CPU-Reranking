// Synthetic candidate lists for testing reranks before the GPU scan exists.
#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

// Each query's list holds its first min(C, kgt) ground-truth IDs plus random base IDs, shuffled,
// so a correct rerank must return the ground-truth top-k (Recall@k = 1 up to ties).
inline std::vector<int32_t> make_candidates(const int32_t* gt, int64_t kgt, int64_t nq, int64_t n,
                                            int64_t C, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> pick(0, static_cast<int32_t>(n - 1));
    std::vector<int32_t> cand(nq * C);
    for (int64_t q = 0; q < nq; q++) {
        int32_t* row = &cand[q * C];
        const int64_t from_gt = std::min(C, kgt);
        for (int64_t j = 0; j < from_gt; j++) row[j] = gt[q * kgt + j];
        // Random fillers may repeat ground-truth IDs; duplicates only make the test harder.
        for (int64_t j = from_gt; j < C; j++) row[j] = pick(rng);
        std::shuffle(row, row + C, rng);
    }
    return cand;
}
