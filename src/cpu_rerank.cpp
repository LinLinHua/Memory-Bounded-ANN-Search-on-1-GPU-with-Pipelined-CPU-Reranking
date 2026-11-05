#include "cpu_rerank.h"

#include <limits>

static inline float l2_sqr_simd(const float* a, const float* b, int64_t d) {
    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
    for (int64_t i = 0; i < d; i++) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Issues a prefetch for every 64-byte line of one base vector.
static inline void prefetch_vector(const float* v, int64_t d) {
    const char* p = reinterpret_cast<const char*>(v);
    for (int64_t off = 0; off < d * static_cast<int64_t>(sizeof(float)); off += 64) {
        __builtin_prefetch(p + off, 0, 0);
    }
}

// k is small (10 in our experiments), so a sorted array with insertion beats a heap.
static void rerank_one(const RerankArgs& a, int64_t q, int32_t* ids, float* dists) {
    const float* query = a.queries + q * a.d;
    const int32_t* cand = a.cand + q * a.C;
    for (int i = 0; i < a.k; i++) {
        ids[i] = -1;
        dists[i] = std::numeric_limits<float>::infinity();
    }
    for (int64_t j = 0; j < a.C; j++) {
        if (a.prefetch_dist > 0 && j + a.prefetch_dist < a.C) {
            const int32_t ahead = cand[j + a.prefetch_dist];
            if (ahead >= 0) prefetch_vector(a.base + ahead * a.d, a.d);
        }
        const int32_t id = cand[j];
        if (id < 0) continue;
        const float dist = l2_sqr_simd(query, a.base + static_cast<int64_t>(id) * a.d, a.d);
        if (dist >= dists[a.k - 1]) continue;
        int pos = a.k - 1;
        while (pos > 0 && dists[pos - 1] > dist) {
            dists[pos] = dists[pos - 1];
            ids[pos] = ids[pos - 1];
            pos--;
        }
        dists[pos] = dist;
        ids[pos] = id;
    }
}

void cpu_rerank(const RerankArgs& args, int32_t* out_ids, float* out_dists) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t q = 0; q < args.nq; q++) {
        rerank_one(args, q, out_ids + q * args.k, out_dists + q * args.k);
    }
}
