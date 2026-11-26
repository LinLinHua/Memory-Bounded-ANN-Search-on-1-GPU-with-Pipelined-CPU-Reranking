// cuVS IVF-RaBitQ baseline (HS). cuVS has no Python binding for it yet, so this is C++.
//
// Same job and CSV columns as src/cuvs_baseline.py: return C candidates per query, then recall
// after an exact rerank on the CPU (not timed). Timing follows docs/measurement.md: each timed
// search starts from the queries in pinned host memory, uploads each batch, and copies the
// candidates back to the host (--transfer 1, the default). --transfer 0 keeps the queries on
// the GPU before timing starts (the timing before Nov 25).
//
//   bits_per_dim 1      1 bit per dimension (binary RaBitQ, like our 1-bit codes)
//   bits_per_dim 2..9   extended RaBitQ with more bits per dimension (4 = our 4-bit size)
//
// Device memory comes from cudaMemGetInfo: gpu_index_mb is what the index holds after build,
// gpu_peak_mb the most seen during one extra, untimed search (sampled every 0.1 ms), both
// above a baseline taken after a throwaway warm-up (a small index built and searched once), so
// cuVS's one-time setup is not counted in the first index. Run it with
// CUDA_MODULE_LOADING=EAGER so the kernels' code is in that baseline, as for Faiss.
//
// Usage:
//   cuvs_rabitq_bench --base B --query Q --gt G --nlist 1024 [--bits 1,2,3,4]
//                     [--nprobe 16,32,64] [--C 100,1000] [--batch 0] [--reps 5] [--transfer 1]
//                     [--mode quant4|quant8|lut16|lut32] [--csv results/cuvs_rabitq_sift.csv]

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <vector>

#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <cuvs/version_config.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include "cli_utils.h"
#include "dataset_io.h"

namespace rq = cuvs::neighbors::ivf_rabitq;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kMB = 1 << 20;

#define CUDA_OK(x)                                                                         \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e_));           \
    } while (0)

double used_mb() {
    size_t free = 0, total = 0;
    CUDA_OK(cudaMemGetInfo(&free, &total));
    return (total - free) / kMB;
}

// Runs fn() while another thread samples device memory; returns the highest reading (MB).
template <typename F>
double peak_during(F fn) {
    std::atomic<bool> done{false};
    std::atomic<double> peak{used_mb()};
    std::thread sampler([&] {
        cudaSetDevice(0);
        while (!done) {
            double u = used_mb(), p = peak.load();
            while (u > p && !peak.compare_exchange_weak(p, u)) {}
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    fn();
    done = true;
    sampler.join();
    return std::max(peak.load(), used_mb());
}

double recall_at_10(const std::vector<int64_t>& ids, int64_t width, const std::vector<int32_t>& gt,
                    int64_t kgt, int64_t nq) {
    int64_t hits = 0;
    for (int64_t q = 0; q < nq; q++) {
        std::unordered_set<int64_t> truth(gt.begin() + q * kgt, gt.begin() + q * kgt + 10);
        for (int64_t j = 0; j < std::min<int64_t>(10, width); j++) hits += truth.count(ids[q * width + j]);
    }
    return double(hits) / (10.0 * nq);
}

// Exact distances for each query's C candidates on the CPU; keeps the 10 closest.
std::vector<int64_t> exact_rerank(const std::vector<float>& base, const std::vector<float>& queries,
                                  int64_t d, const std::vector<int64_t>& cand, int64_t C, int64_t nq) {
    std::vector<int64_t> out(nq * 10, -1);
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(C);
        const float* qv = &queries[q * d];
        for (int64_t j = 0; j < C; j++) {
            const int64_t id = cand[q * C + j];
            if (id < 0) continue;
            const float* x = &base[id * d];
            float s = 0;
            for (int64_t t = 0; t < d; t++) s += (qv[t] - x[t]) * (qv[t] - x[t]);
            scored.emplace_back(s, id);
        }
        const size_t keep = std::min<size_t>(10, scored.size());
        std::partial_sort(scored.begin(), scored.begin() + keep, scored.end());
        for (size_t j = 0; j < keep; j++) out[q * 10 + j] = scored[j].second;
    }
    return out;
}

rq::search_mode parse_mode(const std::string& s) {
    if (s == "quant4") return rq::search_mode::QUANT4;
    if (s == "quant8") return rq::search_mode::QUANT8;
    if (s == "lut16") return rq::search_mode::LUT16;
    if (s == "lut32") return rq::search_mode::LUT32;
    throw std::runtime_error("unknown --mode " + s);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        const std::string base_path = args.required("base"), query_path = args.required("query");
        const std::string gt_path = args.required("gt"), csv_path = args.str("csv");
        const int64_t nlist = args.num("nlist", 1024), batch_arg = args.num("batch", 0);
        const int reps = static_cast<int>(args.num("reps", 5));
        const bool transfer = args.num("transfer", 1) != 0;
        const auto bits_list = args.list("bits", "1,2,3,4");
        const auto nprobe_list = args.list("nprobe", "16,32,64");
        const auto c_list = args.list("C", "100,1000");
        const std::string mode_name = args.str("mode", "quant4");
        const rq::search_mode mode = parse_mode(mode_name);
        args.reject_unknown();

        int64_t n, d, nq, dq, ngt, kgt;
        auto base = load_vectors<float>(base_path, n, d);
        auto queries = load_vectors<float>(query_path, nq, dq);
        auto gt = load_vectors<int32_t>(gt_path, ngt, kgt);
        if (dq != d || ngt < nq) throw std::runtime_error("query / ground-truth shapes do not match");
        const int64_t batch = (batch_arg <= 0 || batch_arg > nq) ? nq : batch_arg;
        const std::string cuvs_version = CUVS_VERSION;
        std::printf("cuvs %s, base %lld x %lld, queries %lld, nlist %lld, batch %lld, mode %s\n",
                    cuvs_version.c_str(), (long long)n, (long long)d, (long long)nq, (long long)nlist,
                    (long long)batch, mode_name.c_str());

        CUDA_OK(cudaFree(nullptr));
        raft::device_resources handle;
        auto stream = raft::resource::get_cuda_stream(handle);
        const double context_mb = used_mb();
        std::printf("device memory with the CUDA context and handle: %.0f MB\n", context_mb);

        FILE* csv = open_csv(csv_path,
                             "algo,nlist,nprobe,C,nq,search_ms,qps,index_mb,build_s,recall10_no_rerank,"
                             "recall10_rerank,batch,gpu_index_mb,gpu_peak_mb,rerank_timed,bits_per_dim,mode,cuvs");

        auto queries_d = raft::make_device_matrix<float, int64_t>(handle, nq, d);
        float* queries_pinned = nullptr;  // where each timed search starts when transfer is on
        CUDA_OK(cudaMallocHost(&queries_pinned, nq * d * sizeof(float)));
        std::copy(queries.begin(), queries.end(), queries_pinned);
        if (!transfer) raft::copy(queries_d.data_handle(), queries_pinned, nq * d, stream);
        raft::resource::sync_stream(handle);

        {  // throwaway warm-up on the first 100,000 vectors: cuVS's one-time setup goes into the baseline
            const int64_t wn = std::min<int64_t>(n, 100000), wq = std::min<int64_t>(nq, 128);
            rq::index_params wp;
            wp.n_lists = static_cast<uint32_t>(std::min<int64_t>(nlist, wn / 32));
            wp.bits_per_dim = static_cast<uint32_t>(bits_list[0]);
            auto warm = rq::build(handle, wp, raft::make_host_matrix_view<const float, int64_t>(base.data(), wn, d));
            rq::search_params sp;
            sp.n_probes = 16;
            sp.mode = mode;
            auto wn_d = raft::make_device_matrix<int64_t, int64_t>(handle, wq, 100);
            auto wd_d = raft::make_device_matrix<float, int64_t>(handle, wq, 100);
            raft::copy(queries_d.data_handle(), queries_pinned, wq * d, stream);
            rq::search(handle, sp, warm,
                       raft::make_device_matrix_view<const float, int64_t>(queries_d.data_handle(), wq, d),
                       wn_d.view(), wd_d.view());
            raft::resource::sync_stream(handle);
        }
        CUDA_OK(cudaDeviceSynchronize());
        const double base_mb = used_mb();
        std::printf("device memory after the warm-up: %.0f MB (the baseline subtracted below)\n", base_mb);

        for (int64_t bits : bits_list) {
            rq::index_params ip;
            ip.n_lists = static_cast<uint32_t>(nlist);
            ip.bits_per_dim = static_cast<uint32_t>(bits);
            const auto t0 = Clock::now();
            // From host memory: no full-precision copy of the dataset stays on the GPU.
            auto index = rq::build(handle, ip,
                                   raft::make_host_matrix_view<const float, int64_t>(base.data(), n, d));
            raft::resource::sync_stream(handle);
            const double build_s = std::chrono::duration<double>(Clock::now() - t0).count();
            const double gpu_index_mb = used_mb() - base_mb;
            const double index_mb = n * d * bits / 8.0 / kMB;
            const std::string algo = "ivf_rabitq_b" + std::to_string(bits);
            std::printf("=== %s: build %.1f s, codes %.0f MB, on GPU %.1f MB\n", algo.c_str(), build_s,
                        index_mb, gpu_index_mb);

            for (int64_t nprobe : nprobe_list) {
                for (int64_t C : c_list) {
                    rq::search_params sp;
                    sp.n_probes = static_cast<uint32_t>(nprobe);
                    sp.mode = mode;
                    std::vector<int64_t> cand(nq * C);
                    auto nbr_d = raft::make_device_matrix<int64_t, int64_t>(handle, batch, C);
                    auto dist_d = raft::make_device_matrix<float, int64_t>(handle, batch, C);
                    auto search_all = [&] {
                        for (int64_t off = 0; off < nq; off += batch) {
                            const int64_t b = std::min(batch, nq - off);
                            if (transfer)
                                raft::copy(queries_d.data_handle() + off * d, queries_pinned + off * d, b * d, stream);
                            rq::search(handle, sp, index,
                                       raft::make_device_matrix_view<const float, int64_t>(
                                           queries_d.data_handle() + off * d, b, d),
                                       raft::make_device_matrix_view<int64_t, int64_t>(nbr_d.data_handle(), b, C),
                                       raft::make_device_matrix_view<float, int64_t>(dist_d.data_handle(), b, C));
                            raft::copy(cand.data() + off * C, nbr_d.data_handle(), b * C, stream);
                            raft::resource::sync_stream(handle);
                        }
                    };
                    search_all();  // warm-up
                    std::vector<double> times;
                    for (int r = 0; r < reps; r++) {
                        const auto s0 = Clock::now();
                        search_all();
                        times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s0).count());
                    }
                    const double ms = median(times);
                    // Index, queries and output buffers are all on the device during the search.
                    const double peak_mb = peak_during(search_all) - base_mb;
                    const double no_rerank = recall_at_10(cand, C, gt, kgt, nq);
                    const double reranked = recall_at_10(exact_rerank(base, queries, d, cand, C, nq), 10, gt, kgt, nq);
                    const double qps = nq / (ms / 1e3);
                    std::printf("nprobe=%-3lld C=%-4lld search_ms=%8.1f qps=%9.0f  R@10 no_rerank=%.4f "
                                "rerank=%.4f  peak %.1f MB\n", (long long)nprobe, (long long)C, ms, qps,
                                no_rerank, reranked, peak_mb);
                    if (csv) {
                        std::fprintf(csv, "%s,%lld,%lld,%lld,%lld,%.3f,%.1f,%.1f,%.2f,%.5f,%.5f,%lld,%.1f,%.1f,0,%lld,%s,%s\n",
                                     algo.c_str(), (long long)nlist, (long long)nprobe, (long long)C, (long long)nq,
                                     ms, qps, index_mb, build_s, no_rerank, reranked,
                                     (long long)(batch_arg <= 0 ? 0 : batch), gpu_index_mb, peak_mb,
                                     (long long)bits, mode_name.c_str(), cuvs_version.c_str());
                        std::fflush(csv);
                    }
                }
            }
        }
        if (csv) std::fclose(csv);
        cudaFreeHost(queries_pinned);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
