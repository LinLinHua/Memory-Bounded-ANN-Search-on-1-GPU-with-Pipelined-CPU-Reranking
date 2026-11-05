// GPU brute-force k-NN: cuBLAS GEMM for inner products + warp top-k (LL, Day 1-3).
// Also writes ground-truth files (used for DEEP10M).
//
// The base is processed in chunks and the queries in batches, so the GEMM output
// (qbatch x chunk floats) fits in one fixed buffer. The final top-k distances are recomputed
// exactly, so they can be compared with the serial baseline.
//
// Usage:
//   gpu_bruteforce --base B --query Q [--gt G] [--nq N] [--k 10] [--max-base N]
//                  [--qbatch 1024] [--chunk 262144] [--reps 3] [--check 100]
//                  [--out gt.ivecs] [--csv results/gpu_bruteforce.csv]
//
//   --check N   compares the first N queries with an exact CPU search
//   --out F     saves the top-k IDs of every query as .ivecs (ground truth)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

#include "cli_utils.h"
#include "dataset_io.h"
#include "device_budget.h"
#include "gpu_distance.cuh"

// Exact top-k for one query on the CPU, sorted by (distance, id).
static std::vector<std::pair<float, int32_t>> cpu_topk(const float* base, int64_t n, int64_t d,
                                                       const float* query, int k) {
    std::vector<std::pair<float, int32_t>> all(n);
    for (int64_t i = 0; i < n; i++) {
        const float* v = base + i * d;
        float sum = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            const float diff = query[j] - v[j];
            sum += diff * diff;
        }
        all[i] = {sum, static_cast<int32_t>(i)};
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    all.resize(k);
    return all;
}

int main(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        const std::string base_path = args.required("base");
        const std::string query_path = args.required("query");
        const std::string gt_path = args.str("gt");
        const std::string out_path = args.str("out");
        const std::string csv_path = args.str("csv");
        const int64_t nq_arg = args.num("nq", -1);
        const int64_t max_base = args.num("max-base", -1);
        const int k = static_cast<int>(args.num("k", 10));
        const int64_t qbatch = args.num("qbatch", 1024);
        int64_t chunk = args.num("chunk", 262144);
        const int reps = static_cast<int>(args.num("reps", 3));
        const int64_t check = args.num("check", 100);
        args.reject_unknown();

        // ---- load data
        int64_t n, d, nq_file, dq;
        const auto base = load_vectors<float>(base_path, n, d, max_base);
        const auto queries = load_vectors<float>(query_path, nq_file, dq);
        if (d != dq) throw std::runtime_error("base and query dimensions differ");
        const int64_t nq = nq_arg < 0 ? nq_file : std::min(nq_arg, nq_file);
        if (k < 1 || k > kMaxTopK || k > n) throw std::runtime_error("k is out of range");
        if (reps < 1) throw std::runtime_error("reps must be positive");
        chunk = std::min(chunk, n);
        std::printf("n=%lld d=%lld nq=%lld k=%d qbatch=%lld chunk=%lld\n", (long long)n, (long long)d,
                    (long long)nq, k, (long long)qbatch, (long long)chunk);

        // ---- device memory (brute force is a baseline, so the budget only records the peak)
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));
        DeviceBudget budget(size_t(44) << 30);
        float* d_base = budget.alloc<float>(n * d, "base");
        float* d_base_norm = budget.alloc<float>(n, "base norms");
        float* d_queries = budget.alloc<float>(nq * d, "queries");
        float* d_query_norm = budget.alloc<float>(nq, "query norms");
        float* d_dots = budget.alloc<float>(std::min(qbatch, nq) * chunk, "gemm tile");
        float* d_best_d = budget.alloc<float>(nq * k, "top-k dists");
        int32_t* d_best_i = budget.alloc<int32_t>(nq * k, "top-k ids");

        GpuTimer timer;
        timer.start();
        CUDA_CHECK(cudaMemcpy(d_base, base.data(), n * d * sizeof(float), cudaMemcpyHostToDevice));
        row_sq_norms(d_base, n, d, d_base_norm);
        const float upload_ms = timer.stop_ms();
        CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), nq * d * sizeof(float),
                              cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));  // plain FP32, no TF32

        // ---- search: for every (query batch, base chunk) tile, GEMM then merge into the top-k
        auto search = [&]() {
            row_sq_norms(d_queries, nq, d, d_query_norm);
            topk_reset(d_best_d, d_best_i, nq, k);
            for (int64_t q0 = 0; q0 < nq; q0 += qbatch) {
                const int64_t nqb = std::min(qbatch, nq - q0);
                for (int64_t c0 = 0; c0 < n; c0 += chunk) {
                    const int64_t ncb = std::min(chunk, n - c0);
                    inner_products(handle, d_queries + q0 * d, nqb, d_base + c0 * d, ncb, d, d_dots);
                    topk_update(d_dots, nqb, ncb, d_base_norm + c0, d_query_norm + q0, c0, k,
                                d_best_d + q0 * k, d_best_i + q0 * k);
                }
            }
            topk_refine_exact(d_base, d_queries, nq, d, k, d_best_d, d_best_i);
        };

        search();  // warm-up
        std::vector<double> times;
        for (int r = 0; r < reps; r++) {
            timer.start();
            search();
            times.push_back(timer.stop_ms());
        }
        budget.sample();
        const double search_ms = median(times);
        const double qps = nq / (search_ms / 1000.0);

        std::vector<int32_t> ids(nq * k);
        std::vector<float> dists(nq * k);
        CUDA_CHECK(cudaMemcpy(ids.data(), d_best_i, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dists.data(), d_best_d, dists.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // ---- recall against the official ground truth
        double recall = -1.0;
        if (!gt_path.empty()) {
            int64_t ngt, kgt;
            const auto gt = load_vectors<int32_t>(gt_path, ngt, kgt);
            if (ngt < nq || kgt < k) throw std::runtime_error("ground truth is too small");
            int64_t hits = 0;
            for (int64_t q = 0; q < nq; q++) {
                const int32_t* truth = &gt[q * kgt];
                for (int j = 0; j < k; j++)
                    if (std::find(truth, truth + k, ids[q * k + j]) != truth + k) hits++;
            }
            recall = double(hits) / double(nq * k);
        }

        // ---- exact CPU check: distances must agree rank by rank, IDs may differ only on ties
        const int64_t nc = std::min(check, nq);
        double max_rel = 0.0;
        int64_t rank_mismatch = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(max : max_rel) reduction(+ : rank_mismatch)
        for (int64_t q = 0; q < nc; q++) {
            const auto ref = cpu_topk(base.data(), n, d, &queries[q * d], k);
            for (int j = 0; j < k; j++) {
                const double g = dists[q * k + j], c = ref[j].first;
                const double rel = std::fabs(g - c) / std::max(std::fabs(c), 1e-6);
                max_rel = std::max(max_rel, rel);
                if (ids[q * k + j] != ref[j].second && rel > 1e-4) rank_mismatch++;
            }
        }

        std::printf("upload_ms=%.1f search_ms=%.1f qps=%.1f recall=%.6f\n", upload_ms, search_ms, qps, recall);
        std::printf("check_queries=%lld max_rel_err=%.2e rank_mismatch=%lld\n", (long long)nc, max_rel,
                    (long long)rank_mismatch);
        budget.report(stdout);

        if (!out_path.empty()) {
            write_ivecs(out_path, ids.data(), nq, k);
            std::printf("wrote %s (%lld x %d)\n", out_path.c_str(), (long long)nq, k);
        }
        if (FILE* csv = open_csv(csv_path, "base,n,d,nq,k,qbatch,chunk,search_ms,qps,recall,"
                                           "check_queries,max_rel_err,rank_mismatch")) {
            std::fprintf(csv, "%s,%lld,%lld,%lld,%d,%lld,%lld,%.3f,%.1f,%.6f,%lld,%.3e,%lld\n",
                         base_path.c_str(), (long long)n, (long long)d, (long long)nq, k,
                         (long long)qbatch, (long long)chunk, search_ms, qps, recall, (long long)nc,
                         max_rel, (long long)rank_mismatch);
            std::fclose(csv);
        }
        cublasDestroy(handle);
        return rank_mismatch == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
