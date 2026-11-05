// Serial brute-force nearest neighbor search, used as the correctness and speedup baseline.
//
// Usage:
//   serial_bruteforce --base B.fvecs --query Q.fvecs --gt G.ivecs [--nq N] [--k K] [--csv out.csv]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vecs_io.h"

static inline float l2_sqr(const float* a, const float* b, int64_t d) {
    float sum = 0.0f;
    for (int64_t i = 0; i < d; i++) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Returns the k nearest (distance, id) pairs in increasing order of distance.
static std::vector<std::pair<float, int32_t>> search_one(
    const float* query, const float* base, int64_t n, int64_t d, int64_t k) {
    std::priority_queue<std::pair<float, int32_t>> heap;  // Largest kept distance on top.
    for (int64_t i = 0; i < n; i++) {
        const float dist = l2_sqr(query, base + i * d, d);
        if (static_cast<int64_t>(heap.size()) < k) {
            heap.emplace(dist, static_cast<int32_t>(i));
        } else if (dist < heap.top().first) {
            heap.pop();
            heap.emplace(dist, static_cast<int32_t>(i));
        }
    }
    std::vector<std::pair<float, int32_t>> result(heap.size());
    for (int64_t i = static_cast<int64_t>(heap.size()) - 1; i >= 0; i--) {
        result[i] = heap.top();
        heap.pop();
    }
    return result;
}

static const char* require_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

int main(int argc, char** argv) {
    try {
        std::string base_path, query_path, gt_path, csv_path;
        int64_t nq_arg = -1, k = 10;
        for (int i = 1; i < argc; i++) {
            if (!std::strcmp(argv[i], "--base")) base_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--query")) query_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--gt")) gt_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--csv")) csv_path = require_value(argc, argv, i);
            else if (!std::strcmp(argv[i], "--nq")) nq_arg = std::atoll(require_value(argc, argv, i));
            else if (!std::strcmp(argv[i], "--k")) k = std::atoll(require_value(argc, argv, i));
            else throw std::runtime_error(std::string("unknown argument ") + argv[i]);
        }
        if (base_path.empty() || query_path.empty() || gt_path.empty())
            throw std::runtime_error("--base, --query and --gt are required");

        int64_t n, d, nq_file, dq, ngt, kgt;
        const auto base = read_vecs<float>(base_path, n, d);
        const auto queries = read_vecs<float>(query_path, nq_file, dq);
        const auto gt = read_vecs<int32_t>(gt_path, ngt, kgt);

        if (d != dq) throw std::runtime_error("base and query dimensions differ");
        if (ngt < nq_file) throw std::runtime_error("ground truth has fewer rows than queries");
        if (k < 1 || k > n || k > kgt) throw std::runtime_error("k is out of range");
        const int64_t nq = nq_arg < 0 ? nq_file : nq_arg;
        if (nq < 1 || nq > nq_file) throw std::runtime_error("nq is out of range");

        std::vector<std::vector<std::pair<float, int32_t>>> results(nq);
        const auto start = std::chrono::steady_clock::now();
        for (int64_t q = 0; q < nq; q++) {
            results[q] = search_one(&queries[q * d], base.data(), n, d, k);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();

        // Recall is computed outside the timed region.
        int64_t hits = 0;
        for (int64_t q = 0; q < nq; q++) {
            const int32_t* truth = &gt[q * kgt];
            for (const auto& item : results[q]) {
                if (std::find(truth, truth + k, item.second) != truth + k) hits++;
            }
        }
        const double recall = static_cast<double>(hits) / static_cast<double>(nq * k);
        const double qps = static_cast<double>(nq) / seconds;

        std::printf("n=%lld d=%lld nq=%lld k=%lld\n", static_cast<long long>(n),
                    static_cast<long long>(d), static_cast<long long>(nq),
                    static_cast<long long>(k));
        std::printf("time_s=%.3f qps=%.3f recall=%.6f\n", seconds, qps, recall);

        if (!csv_path.empty()) {
            std::ifstream existing(csv_path);
            const bool write_header = !existing.good() || existing.peek() == EOF;
            existing.close();
            std::ofstream csv(csv_path, std::ios::app);
            if (!csv) throw std::runtime_error("cannot open " + csv_path);
            if (write_header) csv << "n,d,nq,k,time_s,qps,recall\n";
            csv << n << ',' << d << ',' << nq << ',' << k << ',' << seconds << ',' << qps << ','
                << recall << '\n';
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
