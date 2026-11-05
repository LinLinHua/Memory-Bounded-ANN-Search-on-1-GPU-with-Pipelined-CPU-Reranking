// Tests and times the CPU rerank on synthetic candidates, before the GPU scan exists.
//
// Usage:
//   bench_cpu_rerank --base B.fvecs --query Q.fvecs --gt G.ivecs
//                    [--nq N] [--k K] [--C 10,100,1000] [--prefetch P] [--reps R] [--csv out.csv]

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_rerank.h"
#include "synthetic_candidates.h"
#include "vecs_io.h"

static const char* require_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

// SIFT values are small integers, so squared distances are exact in float in any summation order.
// For float datasets such as DEEP, ties are detected only approximately.
static float l2_sqr_ref(const float* a, const float* b, int64_t d) {
    float sum = 0.0f;
    for (int64_t i = 0; i < d; i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sum;
}

static std::vector<int64_t> parse_list(const std::string& s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::atoll(item.c_str()));
    return out;
}

int main(int argc, char** argv) {
    try {
        std::string base_path, query_path, gt_path, csv_path, c_list = "10,20,50,100,200,500,1000";
        int64_t nq_arg = -1;
        int k = 10, prefetch = 4, reps = 5;
        for (int i = 1; i < argc; i++) {
            if (!std::strcmp(argv[i], "--base")) base_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--query")) query_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--gt")) gt_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--csv")) csv_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--C")) c_list = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--nq")) nq_arg = std::atoll(require_value(argc, argv, i));
            else if (!std::strcmp(argv[i], "--k")) k = std::atoi(require_value(argc, argv, i));
            else if (!std::strcmp(argv[i], "--prefetch")) prefetch = std::atoi(require_value(argc, argv, i));
            else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(require_value(argc, argv, i));
            else throw std::runtime_error(std::string("unknown argument ") + argv[i]);
        }
        if (base_path.empty() || query_path.empty() || gt_path.empty())
            throw std::runtime_error("--base, --query and --gt are required");
        if (reps < 1) throw std::runtime_error("reps must be positive");

        int64_t n, d, nq_file, dq, ngt, kgt;
        const auto base = read_vecs<float>(base_path, n, d);
        const auto queries = read_vecs<float>(query_path, nq_file, dq);
        const auto gt = read_vecs<int32_t>(gt_path, ngt, kgt);
        if (d != dq) throw std::runtime_error("base and query dimensions differ");
        if (ngt < nq_file) throw std::runtime_error("ground truth has fewer rows than queries");
        if (k < 1 || k > kgt) throw std::runtime_error("k is out of range");
        const int64_t nq = nq_arg < 0 ? nq_file : nq_arg;
        if (nq < 1 || nq > nq_file) throw std::runtime_error("nq is out of range");

        const int threads = omp_get_max_threads();
        std::printf("n=%lld d=%lld nq=%lld k=%d threads=%d prefetch=%d\n",
                    static_cast<long long>(n), static_cast<long long>(d),
                    static_cast<long long>(nq), k, threads, prefetch);

        FILE* csv = nullptr;
        if (!csv_path.empty()) {
            std::ifstream existing(csv_path);
            const bool write_header = !existing.good() || existing.peek() == EOF;
            existing.close();
            csv = std::fopen(csv_path.c_str(), "a");
            if (!csv) throw std::runtime_error("cannot open " + csv_path);
            if (write_header)
                std::fprintf(
                    csv, "nq,k,C,threads,prefetch,median_s,qps,us_per_query,gbs,recall,recall_ties\n");
        }

        std::vector<int32_t> ids(nq * k);
        std::vector<float> dists(nq * k);
        for (const int64_t C : parse_list(c_list)) {
            if (C < k) throw std::runtime_error("each C must be at least k");
            const auto cand = make_candidates(gt.data(), kgt, nq, n, C, 12345);
            const RerankArgs args{base.data(), queries.data(), cand.data(), d, nq, C, k, prefetch};

            cpu_rerank(args, ids.data(), dists.data());  // Warm-up.
            std::vector<double> times;
            for (int r = 0; r < reps; r++) {
                const auto start = std::chrono::steady_clock::now();
                cpu_rerank(args, ids.data(), dists.data());
                const auto stop = std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double>(stop - start).count());
            }
            std::sort(times.begin(), times.end());
            const double t = times[times.size() / 2];

            // A miss whose distance equals the k-th true distance is a tie, not an error.
            int64_t hits = 0, tie_hits = 0;
            for (int64_t q = 0; q < nq; q++) {
                const int32_t* truth = &gt[q * kgt];
                const float kth = l2_sqr_ref(&queries[q * d], &base[truth[k - 1] * d], d);
                for (int i = 0; i < k; i++) {
                    if (i > 0 && dists[q * k + i] < dists[q * k + i - 1])
                        throw std::runtime_error("results are not sorted");
                    if (std::find(truth, truth + k, ids[q * k + i]) != truth + k) hits++;
                    else if (ids[q * k + i] >= 0 && dists[q * k + i] <= kth) tie_hits++;
                }
            }
            const double recall = static_cast<double>(hits) / static_cast<double>(nq * k);
            const double recall_ties =
                static_cast<double>(hits + tie_hits) / static_cast<double>(nq * k);
            const double gbs = static_cast<double>(nq) * C * d * sizeof(float) / t / 1e9;
            std::printf(
                "C=%lld median_s=%.6f qps=%.1f us_per_query=%.2f gbs=%.2f recall=%.6f "
                "recall_ties=%.6f\n",
                static_cast<long long>(C), t, nq / t, t / nq * 1e6, gbs, recall, recall_ties);
            if (csv)
                std::fprintf(csv, "%lld,%d,%lld,%d,%d,%.6f,%.1f,%.2f,%.2f,%.6f,%.6f\n",
                             static_cast<long long>(nq), k, static_cast<long long>(C), threads,
                             prefetch, t, nq / t, t / nq * 1e6, gbs, recall, recall_ties);
        }
        if (csv) std::fclose(csv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
