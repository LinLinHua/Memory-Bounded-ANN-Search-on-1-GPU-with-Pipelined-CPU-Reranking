// CPU vs GPU rerank on real scan candidates, split into its parts (XJ, result 3), plus OpenMP
// scaling of the CPU rerank (result 5).
//
// The scan runs once with the largest C; the candidates for a smaller C are the first C of each
// row (the scan returns them sorted by estimated distance).
//
// Where each part comes from, per C:
//   CPU path  id transfer   : copy nq x C candidate IDs from the GPU to pinned host memory
//             gather+compute: cpu_rerank on the real candidates
//             compute       : cpu_rerank on the same candidates remapped to the first --hot vectors,
//                             which stay in cache, so the time left is the distance work
//             gather        : the difference, i.e. the time spent fetching vectors from DRAM
//             result return : none, the results are already on the host
//   GPU path  id transfer   : none, the scan leaves the candidates on the GPU
//             PCIe+compute  : gpu_rerank reading the base vectors from pinned host memory (zero-copy)
//             compute       : gpu_rerank on the remapped candidates, with the hot vectors in GPU memory
//             PCIe          : the difference
//             result return : copy the nq x 10 IDs and distances to the host
//
// Usage:
//   bench_rerank_breakdown --base B --query Q --gt G --index DIR [--label sift]
//                          [--bits 1] [--topc fused] [--nprobe 64] [--batch 256] [--nq 10000]
//                          [--C 10,20,50,100,200,500,1000] [--prefetch 4] [--hot 256] [--reps 5]
//                          [--scaling-C 100,1000] [--threads-list 1,2,4,8,16] [--prefetch-list 0,4]
//                          [--budget-mb 16384] [--csv results/rerank_breakdown.csv]
//                          [--scaling-csv results/rerank_scaling.csv]

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cli_utils.h"
#include "cpu_rerank.h"
#include "dataset_io.h"
#include "device_budget.h"
#include "gpu_rerank.cuh"
#include "ivf_index.h"
#include "ivf_scan.cuh"

namespace {

constexpr int kRecallK = 10;
constexpr double kTolerance = 1e-4;

// Median wall time of fn over reps runs, after one warm-up run.
template <typename Fn>
double cpu_ms(Fn fn, int reps) {
    fn();
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        const auto a = std::chrono::steady_clock::now();
        fn();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    return median(t);
}

// Median GPU time of the work fn enqueues on stream, after one warm-up run.
template <typename Fn>
double gpu_ms(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop, Fn fn, int reps) {
    fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        t.push_back(ms);
    }
    return median(t);
}

double recall_at_10(const int32_t* ids, const int32_t* gt, int64_t kgt, int64_t nq) {
    int64_t hits = 0;
    for (int64_t q = 0; q < nq; q++) {
        const int32_t* truth = gt + q * kgt;
        for (int j = 0; j < kRecallK; j++)
            if (std::find(truth, truth + kRecallK, ids[q * kRecallK + j]) != truth + kRecallK) hits++;
    }
    return double(hits) / double(nq * kRecallK);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        const std::string base_path = args.required("base");
        const std::string query_path = args.required("query");
        const std::string gt_path = args.required("gt");
        const std::string index_dir = args.required("index");
        const std::string label = args.str("label", "");
        const int bits = static_cast<int>(args.num("bits", 1));
        const std::string topc_name = args.str("topc", "fused");
        const int nprobe = static_cast<int>(args.num("nprobe", 64));
        const int batch = static_cast<int>(args.num("batch", 256));
        const int64_t nq_arg = args.num("nq", -1);
        const auto c_list = args.list("C", "10,20,50,100,200,500,1000");
        const int prefetch = static_cast<int>(args.num("prefetch", 4));
        int64_t hot = args.num("hot", 256);
        const int reps = static_cast<int>(args.num("reps", 5));
        const auto scaling_c = args.list("scaling-C", "100,1000");
        const auto threads_list = args.list("threads-list", "1,2,4,8,16");
        const auto prefetch_list = args.list("prefetch-list", "0,4");
        const int64_t budget_mb = args.num("budget-mb", 16384);
        const std::string csv_path = args.str("csv");
        const std::string scaling_csv_path = args.str("scaling-csv");
        args.reject_unknown();
        if (topc_name != "fused" && topc_name != "select" && topc_name != "sort")
            throw std::runtime_error("--topc must be fused, select or sort");
        const TopC topc = topc_name == "sort" ? TopC::Sort : topc_name == "select" ? TopC::Select : TopC::Fused;
        if (reps < 1) throw std::runtime_error("reps must be positive");

        // ---- data
        int64_t n, d, nq_file, dq, ngt, kgt;
        auto base = load_vectors<float>(base_path, n, d);
        const auto queries = load_vectors<float>(query_path, nq_file, dq);
        const auto gt = load_vectors<int32_t>(gt_path, ngt, kgt);
        const IvfIndex ix = load_ivf(index_dir);
        if (d != dq || d != ix.d || n != ix.n) throw std::runtime_error("data and index do not match");
        if (d % 4 != 0) throw std::runtime_error("gpu_rerank needs d divisible by 4");
        const int64_t nq = nq_arg < 0 ? nq_file : std::min(nq_arg, nq_file);
        if (ngt < nq || kgt < kRecallK) throw std::runtime_error("ground truth is too small");
        hot = std::min(hot, n);
        const int64_t c_max = *std::max_element(c_list.begin(), c_list.end());
        if (*std::min_element(c_list.begin(), c_list.end()) < kRecallK)
            throw std::runtime_error("each C must be at least 10");
        for (const int64_t c : scaling_c)
            if (std::find(c_list.begin(), c_list.end(), c) == c_list.end())
                throw std::runtime_error("every --scaling-C value must also be in --C");
        const int max_threads = omp_get_max_threads();
        std::printf("n=%lld d=%lld nq=%lld bits=%d topc=%s nprobe=%d threads=%d prefetch=%d hot=%lld\n",
                    (long long)n, (long long)d, (long long)nq, bits, topc_name.c_str(), nprobe, max_threads,
                    prefetch, (long long)hot);

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));
        DeviceBudget budget(static_cast<size_t>(budget_mb) << 20);

        // ---- real candidates from the scan, at the largest C
        std::vector<int32_t> cand_max(nq * c_max);
        {
            std::vector<float> est(nq * c_max);
            IvfScanner scanner(ix, bits, batch, nprobe, static_cast<int>(c_max), budget, topc);
            scanner.search(queries.data(), nq, nprobe, static_cast<int>(c_max), cand_max.data(), est.data());
        }

        // ---- device and pinned buffers (the scanner's memory is freed by now)
        CUDA_CHECK(cudaHostRegister(base.data(), base.size() * sizeof(float),
                                    cudaHostRegisterMapped | cudaHostRegisterReadOnly));
        float* base_mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&base_mapped), base.data(), 0));
        float* d_queries = budget.alloc<float>(nq * d, "queries");
        float* d_hot_base = budget.alloc<float>(hot * d, "hot base vectors");
        int32_t* d_cand = budget.alloc<int32_t>(nq * c_max, "candidates");
        int32_t* d_cand_hot = budget.alloc<int32_t>(nq * c_max, "hot candidates");
        int32_t* d_ids = budget.alloc<int32_t>(nq * kRecallK, "result ids");
        float* d_dists = budget.alloc<float>(nq * kRecallK, "result dists");
        CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), nq * d * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_hot_base, base.data(), hot * d * sizeof(float), cudaMemcpyHostToDevice));

        int32_t* h_cand = nullptr;
        int32_t* h_ids = nullptr;
        float* h_dists = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_cand, nq * c_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_ids, nq * kRecallK * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&h_dists, nq * kRecallK * sizeof(float)));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        FILE* csv = open_csv(csv_path,
                             "label,bits,topc,nprobe,nq,C,threads,prefetch,recall10,"
                             "cpu_id_d2h_ms,cpu_rerank_ms,cpu_compute_ms,cpu_gather_ms,cpu_path_ms,"
                             "gpu_kernel_ms,gpu_compute_ms,gpu_pcie_ms,gpu_result_d2h_ms,gpu_path_ms,"
                             "vector_mb,cpu_gather_gbs,gpu_pcie_gbs,max_rel_err,rank_mismatch");
        FILE* scsv = open_csv(scaling_csv_path,
                              "label,bits,nq,C,prefetch,threads,ms,qps,speedup,efficiency,gbs");

        std::vector<int32_t> cand, cand_hot;
        std::vector<int32_t> cpu_ids(nq * kRecallK), hot_ids(nq * kRecallK);
        std::vector<float> cpu_dists(nq * kRecallK), hot_dists(nq * kRecallK);
        bool all_ok = true;

        for (const int64_t C : c_list) {
            cand.resize(nq * C);
            cand_hot.resize(nq * C);
            for (int64_t q = 0; q < nq; q++)
                for (int64_t j = 0; j < C; j++) {
                    const int32_t id = cand_max[q * c_max + j];
                    cand[q * C + j] = id;
                    cand_hot[q * C + j] = id < 0 ? -1 : static_cast<int32_t>(id % hot);
                }
            const double vector_mb = double(nq) * C * d * sizeof(float) / 1e6;

            // ---- CPU path
            omp_set_num_threads(max_threads);
            CUDA_CHECK(cudaMemcpy(d_cand, cand.data(), nq * C * sizeof(int32_t), cudaMemcpyHostToDevice));
            const double id_d2h = gpu_ms(stream, ev_start, ev_stop, [&] {
                CUDA_CHECK(cudaMemcpyAsync(h_cand, d_cand, nq * C * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            }, reps);
            const RerankArgs real{base.data(), queries.data(), cand.data(), d, nq, C, kRecallK, prefetch};
            const RerankArgs cached{base.data(), queries.data(), cand_hot.data(), d, nq, C, kRecallK, prefetch};
            const double cpu_total = cpu_ms([&] { cpu_rerank(real, cpu_ids.data(), cpu_dists.data()); }, reps);
            const double cpu_compute = cpu_ms([&] { cpu_rerank(cached, hot_ids.data(), hot_dists.data()); }, reps);
            const double cpu_gather = std::max(0.0, cpu_total - cpu_compute);

            // ---- GPU path
            CUDA_CHECK(cudaMemcpy(d_cand_hot, cand_hot.data(), nq * C * sizeof(int32_t), cudaMemcpyHostToDevice));
            const GpuRerankArgs zero_copy{base_mapped, d_queries, d_cand, d, nq, C, kRecallK};
            const GpuRerankArgs on_device{d_hot_base, d_queries, d_cand_hot, d, nq, C, kRecallK};
            const double gpu_compute = gpu_ms(stream, ev_start, ev_stop,
                                              [&] { gpu_rerank(on_device, d_ids, d_dists, stream); }, reps);
            const double gpu_kernel = gpu_ms(stream, ev_start, ev_stop,
                                             [&] { gpu_rerank(zero_copy, d_ids, d_dists, stream); }, reps);
            const double gpu_pcie = std::max(0.0, gpu_kernel - gpu_compute);
            const double result_d2h = gpu_ms(stream, ev_start, ev_stop, [&] {
                CUDA_CHECK(cudaMemcpyAsync(h_ids, d_ids, nq * kRecallK * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(h_dists, d_dists, nq * kRecallK * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }, reps);
            budget.sample();

            // The last GPU run was the zero-copy one: same neighbors as the CPU, up to ties.
            double max_rel = 0.0;
            int64_t rank_mismatch = 0;
            for (int64_t i = 0; i < nq * kRecallK; i++) {
                const double a = h_dists[i], b = cpu_dists[i];
                if (h_ids[i] < 0 && cpu_ids[i] < 0) continue;  // padding on both sides
                const double rel = std::fabs(a - b) / std::max(1e-12, std::fabs(b));
                max_rel = std::max(max_rel, rel);
                if (h_ids[i] != cpu_ids[i] && rel > kTolerance) rank_mismatch++;
            }
            all_ok = all_ok && rank_mismatch == 0;
            const double recall = recall_at_10(cpu_ids.data(), gt.data(), kgt, nq);
            const double cpu_path = id_d2h + cpu_total;
            const double gpu_path = gpu_kernel + result_d2h;

            std::printf("C=%-4lld recall %.4f | CPU: ids %.3f + rerank %.2f (compute %.2f, gather %.2f) = %.2f ms"
                        " | GPU: kernel %.2f (compute %.2f, PCIe %.2f) + results %.3f = %.2f ms"
                        " | gather %.1f GB/s, PCIe %.1f GB/s, mismatch %lld\n",
                        (long long)C, recall, id_d2h, cpu_total, cpu_compute, cpu_gather, cpu_path, gpu_kernel,
                        gpu_compute, gpu_pcie, result_d2h, gpu_path, vector_mb / std::max(cpu_gather, 1e-9),
                        vector_mb / std::max(gpu_pcie, 1e-9), (long long)rank_mismatch);
            if (csv)
                std::fprintf(csv,
                             "%s,%d,%s,%d,%lld,%lld,%d,%d,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                             "%.4f,%.2f,%.2f,%.2f,%.3e,%lld\n",
                             label.c_str(), bits, topc_name.c_str(), nprobe, (long long)nq, (long long)C,
                             max_threads, prefetch, recall, id_d2h, cpu_total, cpu_compute, cpu_gather, cpu_path,
                             gpu_kernel, gpu_compute, gpu_pcie, result_d2h, gpu_path, vector_mb,
                             vector_mb / std::max(cpu_gather, 1e-9), vector_mb / std::max(gpu_pcie, 1e-9),
                             max_rel, (long long)rank_mismatch);

            // ---- OpenMP scaling of the CPU rerank
            if (std::find(scaling_c.begin(), scaling_c.end(), C) == scaling_c.end()) continue;
            for (const int64_t pf : prefetch_list) {
                const RerankArgs a{base.data(), queries.data(), cand.data(), d, nq, C, kRecallK, static_cast<int>(pf)};
                double t1 = 0.0;
                for (const int64_t t : threads_list) {
                    omp_set_num_threads(static_cast<int>(t));
                    const double ms = cpu_ms([&] { cpu_rerank(a, cpu_ids.data(), cpu_dists.data()); }, reps);
                    if (t1 == 0.0) t1 = ms * double(t);  // 1-thread time, from the first entry (normally 1 thread)
                    const double speedup = t1 / ms;
                    std::printf("  scaling C=%lld prefetch=%lld threads=%-3lld %9.2f ms  speedup %.2f  %.1f GB/s\n",
                                (long long)C, (long long)pf, (long long)t, ms, speedup, vector_mb / ms);
                    if (scsv)
                        std::fprintf(scsv, "%s,%d,%lld,%lld,%lld,%lld,%.4f,%.1f,%.3f,%.3f,%.2f\n", label.c_str(),
                                     bits, (long long)nq, (long long)C, (long long)pf, (long long)t, ms,
                                     nq / (ms / 1e3), speedup, speedup / double(t), vector_mb / ms);
                }
            }
            omp_set_num_threads(max_threads);
        }

        std::printf("\n");
        budget.report(stdout);
        std::printf("%s\n", all_ok ? "CPU AND GPU RERANK AGREE" : "CPU AND GPU RERANK DIFFER");
        if (csv) std::fclose(csv);
        if (scsv) std::fclose(scsv);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaStreamDestroy(stream);
        cudaFreeHost(h_cand);
        cudaFreeHost(h_ids);
        cudaFreeHost(h_dists);
        cudaHostUnregister(base.data());
        return all_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
