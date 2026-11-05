// Compares GPU zero-copy rerank with CPU rerank on the same synthetic candidates (result 3),
// and checks both return the same neighbors. All device memory goes through DeviceBudget.
//
// Usage:
//   bench_gpu_rerank --base B.fvecs --query Q.fvecs --gt G.ivecs
//                    [--nq N] [--k K] [--C 10,100,1000] [--prefetch P] [--reps R]
//                    [--budget-mb M] [--csv out.csv]

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_rerank.h"
#include "device_budget.h"
#include "gpu_rerank.cuh"
#include "synthetic_candidates.h"
#include "vecs_io.h"

static const char* require_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

static std::vector<int64_t> parse_list(const std::string& s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::atoll(item.c_str()));
    return out;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    try {
        std::string base_path, query_path, gt_path, csv_path, c_list = "10,20,50,100,200,500,1000";
        int64_t nq_arg = -1, budget_mb = 4096;
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
            else if (!std::strcmp(argv[i], "--budget-mb")) budget_mb = std::atoll(require_value(argc, argv, i));
            else throw std::runtime_error(std::string("unknown argument ") + argv[i]);
        }
        if (base_path.empty() || query_path.empty() || gt_path.empty())
            throw std::runtime_error("--base, --query and --gt are required");
        if (reps < 1) throw std::runtime_error("reps must be positive");

        int64_t n, d, nq_file, dq, ngt, kgt;
        auto base = read_vecs<float>(base_path, n, d);
        const auto queries = read_vecs<float>(query_path, nq_file, dq);
        const auto gt = read_vecs<int32_t>(gt_path, ngt, kgt);
        if (d != dq) throw std::runtime_error("base and query dimensions differ");
        if (ngt < nq_file) throw std::runtime_error("ground truth has fewer rows than queries");
        if (k < 1 || k > kgt) throw std::runtime_error("k is out of range");
        const int64_t nq = nq_arg < 0 ? nq_file : nq_arg;
        if (nq < 1 || nq > nq_file) throw std::runtime_error("nq is out of range");

        const auto c_values = parse_list(c_list);
        const int64_t c_max = *std::max_element(c_values.begin(), c_values.end());

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));  // Create the context before the budget takes its baseline.
        DeviceBudget budget(static_cast<size_t>(budget_mb) << 20);

        // Base vectors stay in host DRAM; the GPU reads them over PCIe through a mapped pointer.
        const size_t base_bytes = base.size() * sizeof(float);
        CUDA_CHECK(cudaHostRegister(base.data(), base_bytes,
                                    cudaHostRegisterMapped | cudaHostRegisterReadOnly));
        float* base_mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&base_mapped), base.data(), 0));

        float* d_queries = budget.alloc<float>(nq * d, "queries");
        CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), nq * d * sizeof(float), cudaMemcpyHostToDevice));
        int32_t* d_cand = budget.alloc<int32_t>(nq * c_max, "candidates");
        int32_t* d_ids = budget.alloc<int32_t>(nq * k, "result ids");
        float* d_dists = budget.alloc<float>(nq * k, "result dists");

        int32_t* h_cand = nullptr;
        int32_t* h_ids = nullptr;
        float* h_dists = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_cand, nq * c_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_ids, nq * k * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_dists, nq * k * sizeof(float)));
        std::vector<int32_t> cpu_ids(nq * k);
        std::vector<float> cpu_dists(nq * k);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t ev[4];
        for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));

        std::printf("n=%lld d=%lld nq=%lld k=%d cpu_threads=%d prefetch=%d\n",
                    static_cast<long long>(n), static_cast<long long>(d),
                    static_cast<long long>(nq), k, omp_get_max_threads(), prefetch);

        FILE* csv = nullptr;
        if (!csv_path.empty()) {
            std::ifstream existing(csv_path);
            const bool write_header = !existing.good() || existing.peek() == EOF;
            existing.close();
            csv = std::fopen(csv_path.c_str(), "a");
            if (!csv) throw std::runtime_error("cannot open " + csv_path);
            if (write_header)
                std::fprintf(csv,
                             "nq,k,C,cpu_threads,prefetch,gpu_h2d_ms,gpu_kernel_ms,gpu_d2h_ms,"
                             "gpu_total_ms,cpu_ms,max_rel_err,rank_mismatch\n");
        }

        for (const int64_t C : c_values) {
            if (C < k) throw std::runtime_error("each C must be at least k");
            const auto cand = make_candidates(gt.data(), kgt, nq, n, C, 12345);
            std::copy(cand.begin(), cand.end(), h_cand);
            const GpuRerankArgs gargs{base_mapped, d_queries, d_cand, d, nq, C, k};
            const RerankArgs cargs{base.data(), queries.data(), cand.data(), d, nq, C, k, prefetch};

            std::vector<double> h2d, kern, d2h, total, cpu;
            for (int r = 0; r <= reps; r++) {  // Rep 0 is warm-up.
                CUDA_CHECK(cudaEventRecord(ev[0], stream));
                CUDA_CHECK(cudaMemcpyAsync(d_cand, h_cand, nq * C * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaEventRecord(ev[1], stream));
                gpu_rerank(gargs, d_ids, d_dists, stream);
                CUDA_CHECK(cudaEventRecord(ev[2], stream));
                CUDA_CHECK(cudaMemcpyAsync(h_ids, d_ids, nq * k * sizeof(int32_t),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(h_dists, d_dists, nq * k * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaEventRecord(ev[3], stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                budget.sample();

                const auto start = std::chrono::steady_clock::now();
                cpu_rerank(cargs, cpu_ids.data(), cpu_dists.data());
                const auto stop = std::chrono::steady_clock::now();
                if (r == 0) continue;

                float ms[3];
                for (int i = 0; i < 3; i++) CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
                h2d.push_back(ms[0]);
                kern.push_back(ms[1]);
                d2h.push_back(ms[2]);
                total.push_back(ms[0] + ms[1] + ms[2]);
                cpu.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            }

            // Rank-by-rank distances must agree; IDs may differ only between tied distances.
            double max_rel = 0.0;
            int64_t rank_mismatch = 0;
            for (int64_t i = 0; i < nq * k; i++) {
                const double a = h_dists[i], b = cpu_dists[i];
                const double rel = std::fabs(a - b) / std::max(1e-12, std::fabs(b));
                max_rel = std::max(max_rel, rel);
                if (h_ids[i] != cpu_ids[i] && rel > 1e-4) rank_mismatch++;
            }

            std::printf(
                "C=%lld gpu_h2d_ms=%.3f gpu_kernel_ms=%.3f gpu_d2h_ms=%.3f gpu_total_ms=%.3f "
                "cpu_ms=%.3f max_rel_err=%.2e rank_mismatch=%lld\n",
                static_cast<long long>(C), median(h2d), median(kern), median(d2h), median(total),
                median(cpu), max_rel, static_cast<long long>(rank_mismatch));
            if (csv)
                std::fprintf(csv, "%lld,%d,%lld,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%lld\n",
                             static_cast<long long>(nq), k, static_cast<long long>(C),
                             omp_get_max_threads(), prefetch, median(h2d), median(kern), median(d2h),
                             median(total), median(cpu), max_rel, static_cast<long long>(rank_mismatch));
        }
        if (csv) std::fclose(csv);

        budget.report(stdout);
        for (auto& e : ev) cudaEventDestroy(e);
        cudaStreamDestroy(stream);
        cudaFreeHost(h_cand);
        cudaFreeHost(h_ids);
        cudaFreeHost(h_dists);
        cudaHostUnregister(base.data());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
