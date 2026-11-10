// Day 7 checkpoint: end to end with LL's GPU scan (LL, Day 4-7).
//
//   GPU scan (8/4/1-bit) -> candidate IDs -> CPU rerank (XJ, cpu_rerank.h) -> Recall@10
//
// For each (bits, nprobe, C) it also checks:
//   - the scan against a CPU reference on the first --check queries: same estimated distances
//     (relative error <= 1e-4) in the same top-C order
//   - the GPU rerank (gpu_rerank.cuh) returns the same neighbors as the CPU rerank
//
// Usage:
//   bench_scan --base B --query Q --gt G --index DIR
//              [--bits 8,1] [--nprobe 16,32,64] [--C 100,1000] [--nq 10000] [--batch 256]
//              [--check 100] [--reps 5] [--budget-mb 16384] [--csv results/scan_sift.csv]
//              [--topc fused|select|sort]   (see ivf_scan.cuh; fused is the default)

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>
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

// Fraction of the true top-10 that appears in the first width entries of each row.
double recall_in(const int32_t* rows, int64_t stride, int64_t width, const int32_t* gt, int64_t kgt, int64_t nq) {
    int64_t hits = 0;
    for (int64_t q = 0; q < nq; q++) {
        const int32_t* row = rows + q * stride;
        for (int j = 0; j < kRecallK; j++)
            if (std::find(row, row + width, gt[q * kgt + j]) != row + width) hits++;
    }
    return double(hits) / double(nq * kRecallK);
}

// CPU version of the scan for one query: same probes, same estimator, sorted.
// Returns (estimate, original id) pairs, at most C of them.
std::vector<std::pair<float, int32_t>> reference_scan(const IvfIndex& ix, const float* q, int bits,
                                                      int nprobe, int C) {
    const int64_t d = ix.d;
    std::vector<std::pair<float, int32_t>> lists(ix.nlist);
    for (int64_t l = 0; l < ix.nlist; l++) {
        float s = 0.0f;
        for (int64_t k = 0; k < d; k++) {
            const float diff = q[k] - ix.centroids[l * d + k];
            s += diff * diff;
        }
        lists[l] = {s, static_cast<int32_t>(l)};
    }
    std::partial_sort(lists.begin(), lists.begin() + nprobe, lists.end());

    std::vector<std::pair<float, int32_t>> all;
    const int64_t words = d / 32;
    std::vector<float> qr(d), rotated(d);
    std::vector<uint32_t> planes(4 * words);  // bit planes of the 4-bit rotated query (1-bit codes)
    auto quantize4 = [](float v, float lo, float step) {
        return std::min(15, std::max(0, static_cast<int>(std::nearbyint((v - lo) / step))));
    };
    for (int j = 0; j < nprobe; j++) {
        const int64_t l = lists[j].second;
        for (int64_t k = 0; k < d; k++) qr[k] = q[k] - ix.centroids[l * d + k];

        // Same steps as prepare_query<1> in ivf_scan.cu (fmaf matches the GPU's fused multiply-add).
        float qn2 = 0.0f, lo = 0.0f, step = 1.0f, total = 0.0f;
        if (bits == 1) {
            const float* cent = &ix.centroids[l * d];
            for (int64_t t = 0; t < d; t++) {  // R^T q - R^T c, like rotate_rows_kernel
                float rq = 0.0f, rc = 0.0f;
                for (int64_t i = 0; i < d; i++) {
                    rq = std::fmaf(q[i], ix.rotation[i * d + t], rq);
                    rc = std::fmaf(cent[i], ix.rotation[i * d + t], rc);
                }
                rotated[t] = rq - rc;
            }
            lo = rotated[0];
            float hi = rotated[0];
            for (int64_t i = 0; i < d; i++) {
                qn2 = std::fmaf(qr[i], qr[i], qn2);
                lo = std::min(lo, rotated[i]);
                hi = std::max(hi, rotated[i]);
            }
            step = hi > lo ? (hi - lo) / 15.0f : 1.0f;
            int sum_u = 0;
            std::fill(planes.begin(), planes.end(), 0u);
            for (int64_t t = 0; t < d; t++) {
                const int u = quantize4(rotated[t], lo, step);
                sum_u += u;
                for (int b = 0; b < 4; b++)
                    if ((u >> b) & 1) planes[b * words + t / 32] |= 1u << (t % 32);
            }
            total = float(d) * lo + step * float(sum_u);
        }

        for (int64_t row = ix.offsets[l]; row < ix.offsets[l + 1]; row++) {
            float est = 0.0f;
            if (bits == 1) {
                int ones = 0, weighted = 0;
                for (int64_t w = 0; w < words; w++) {
                    const uint32_t c = ix.codes1[row * words + w];
                    ones += __builtin_popcount(c);
                    for (int b = 0; b < 4; b++) weighted += (1 << b) * __builtin_popcount(c & planes[b * words + w]);
                }
                const float set_sum = lo * float(ones) + step * float(weighted);
                const float ip = (2.0f * set_sum - total) * (1.0f / std::sqrt(float(d))) / ix.factors1[row];
                const float rn = ix.norms[row];
                est = qn2 + rn * rn - 2.0f * rn * ip;
            } else {
                for (int64_t k = 0; k < d; k++) {
                    const uint8_t code = bits == 8 ? ix.codes8[row * d + k]
                                                   : (ix.codes4[row * d / 2 + k / 2] >> (4 * (k % 2))) & 0xf;
                    const float step = bits == 8 ? ix.sq_step8[k] : ix.sq_step4[k];
                    const float diff = qr[k] - (ix.sq_min[k] + code * step);
                    est += diff * diff;
                }
            }
            all.push_back({est, ix.ids[row]});
        }
    }
    const int64_t keep = std::min<int64_t>(C, all.size());
    std::partial_sort(all.begin(), all.begin() + keep, all.end());
    all.resize(keep);
    return all;
}

struct CheckResult {
    double max_rel = 0.0;
    int64_t mismatch = 0;  // ranks whose estimate differs by more than the tolerance
};

// Rank-by-rank comparison with the CPU reference. IDs may differ only between equal estimates.
CheckResult check_scan(const IvfIndex& ix, const float* queries, int64_t nc, int bits, int nprobe, int C,
                       const int32_t* ids, const float* est) {
    double max_rel = 0.0;
    int64_t mismatch = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(max : max_rel) reduction(+ : mismatch)
    for (int64_t q = 0; q < nc; q++) {
        const auto ref = reference_scan(ix, queries + q * ix.d, bits, nprobe, C);
        for (int64_t c = 0; c < C; c++) {
            if (c >= static_cast<int64_t>(ref.size())) {  // padding
                if (ids[q * C + c] != -1) mismatch++;
                continue;
            }
            const double g = est[q * C + c], want = ref[c].first;
            const double rel = std::fabs(g - want) / std::max(std::fabs(want), 1e-6);
            max_rel = std::max(max_rel, rel);
            if (rel > kTolerance) mismatch++;
        }
    }
    return CheckResult{max_rel, mismatch};
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        const std::string base_path = args.required("base");
        const std::string query_path = args.required("query");
        const std::string gt_path = args.required("gt");
        const std::string index_dir = args.required("index");
        const auto bits_list = args.list("bits", "8,1");
        const auto nprobe_list = args.list("nprobe", "16,32,64");
        const auto c_list = args.list("C", "100,1000");
        const int64_t nq_arg = args.num("nq", -1);
        const int batch = static_cast<int>(args.num("batch", 256));
        const int64_t check = args.num("check", 100);
        const int reps = static_cast<int>(args.num("reps", 5));
        const int64_t budget_mb = args.num("budget-mb", 16384);
        const std::string csv_path = args.str("csv");
        const std::string topc_name = args.str("topc", "fused");
        if (topc_name != "fused" && topc_name != "select" && topc_name != "sort")
            throw std::runtime_error("--topc must be fused, select or sort");
        const TopC topc = topc_name == "sort" ? TopC::Sort : topc_name == "select" ? TopC::Select : TopC::Fused;
        args.reject_unknown();

        // ---- data
        int64_t n, d, nq_file, dq, ngt, kgt;
        auto base = load_vectors<float>(base_path, n, d);
        const auto queries = load_vectors<float>(query_path, nq_file, dq);
        const auto gt = load_vectors<int32_t>(gt_path, ngt, kgt);
        const IvfIndex ix = load_ivf(index_dir);
        if (d != dq || d != ix.d || n != ix.n) throw std::runtime_error("data and index do not match");
        const int64_t nq = nq_arg < 0 ? nq_file : std::min(nq_arg, nq_file);
        if (ngt < nq || kgt < kRecallK) throw std::runtime_error("ground truth is too small");
        const int max_nprobe = static_cast<int>(*std::max_element(nprobe_list.begin(), nprobe_list.end()));
        const int max_C = static_cast<int>(*std::max_element(c_list.begin(), c_list.end()));
        const int64_t nc = std::min(check, nq);
        std::printf("n=%lld d=%lld nlist=%lld nq=%lld batch=%d threads=%d\n", (long long)n, (long long)d,
                    (long long)ix.nlist, (long long)nq, batch, omp_get_max_threads());

        // ---- GPU setup: the scan's memory goes through the budget; the rerank check is separate
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));
        DeviceBudget budget(static_cast<size_t>(budget_mb) << 20);
        CUDA_CHECK(cudaHostRegister(base.data(), base.size() * sizeof(float),
                                    cudaHostRegisterMapped | cudaHostRegisterReadOnly));
        float* base_mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&base_mapped), base.data(), 0));
        float* rr_queries = nullptr;
        int32_t* rr_cand = nullptr;
        int32_t* rr_ids = nullptr;
        float* rr_dists = nullptr;
        CUDA_CHECK(cudaMalloc(&rr_queries, nq * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rr_cand, nq * max_C * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&rr_ids, nq * kRecallK * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&rr_dists, nq * kRecallK * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(rr_queries, queries.data(), nq * d * sizeof(float), cudaMemcpyHostToDevice));

        FILE* csv = open_csv(csv_path,
                             "bits,nprobe,C,nq,batch,index_mb,work_mb,scan_ms,qps,coarse_ms,scan_kernel_ms,"
                             "select_ms,cand_recall,recall10_no_rerank,recall10_rerank,cpu_rerank_ms,"
                             "check_queries,max_rel_err,scan_mismatch,rerank_mismatch,topc");
        bool all_ok = true;

        for (const int64_t bits : bits_list) {
            IvfScanner scanner(ix, static_cast<int>(bits), batch, max_nprobe, max_C, budget, topc);
            const double index_mb = scanner.index_bytes() / double(1 << 20);
            const double work_mb = scanner.work_bytes() / double(1 << 20);
            std::printf("\n=== %lld-bit  top-C %s  index %.1f MB  work buffers %.1f MB\n", (long long)bits, topc_name.c_str(), index_mb, work_mb);

            for (const int64_t nprobe : nprobe_list) {
                for (const int64_t C : c_list) {
                    std::vector<int32_t> cand(nq * C);
                    std::vector<float> est(nq * C);

                    // GPU scan: one warm-up, then the median of reps runs
                    std::vector<double> times;
                    ScanTiming parts;
                    for (int r = 0; r <= reps; r++) {
                        scanner.search(queries.data(), nq, int(nprobe), int(C), cand.data(), est.data());
                        if (r == 0) continue;
                        times.push_back(scanner.timing().total);
                        parts = scanner.timing();
                    }
                    const double scan_ms = median(times);

                    // CPU rerank of the candidates (XJ's code)
                    std::vector<int32_t> cpu_ids(nq * kRecallK);
                    std::vector<float> cpu_dists(nq * kRecallK);
                    const RerankArgs cargs{base.data(), queries.data(), cand.data(), d, nq, C, kRecallK, 4};
                    std::vector<double> rerank_times;
                    for (int r = 0; r < 3; r++) {
                        const auto t0 = std::chrono::steady_clock::now();
                        cpu_rerank(cargs, cpu_ids.data(), cpu_dists.data());
                        rerank_times.push_back(ms_since(t0));
                    }

                    // GPU rerank of the same candidates must return the same neighbors
                    std::vector<int32_t> gpu_ids(nq * kRecallK);
                    std::vector<float> gpu_dists(nq * kRecallK);
                    CUDA_CHECK(cudaMemcpy(rr_cand, cand.data(), nq * C * sizeof(int32_t), cudaMemcpyHostToDevice));
                    gpu_rerank(GpuRerankArgs{base_mapped, rr_queries, rr_cand, d, nq, C, kRecallK}, rr_ids, rr_dists, 0);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(gpu_ids.data(), rr_ids, gpu_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(gpu_dists.data(), rr_dists, gpu_dists.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    int64_t rerank_mismatch = 0;
                    for (int64_t i = 0; i < nq * kRecallK; i++) {
                        const double a = gpu_dists[i], b = cpu_dists[i];
                        const bool same_dist = (std::isinf(a) && std::isinf(b)) ||
                                               std::fabs(a - b) <= kTolerance * std::max(std::fabs(b), 1e-6);
                        if (!same_dist) rerank_mismatch++;  // IDs may differ only between tied distances
                    }

                    // recall and scan correctness
                    const double cand_recall = recall_in(cand.data(), C, C, gt.data(), kgt, nq);
                    const double no_rerank = recall_in(cand.data(), C, kRecallK, gt.data(), kgt, nq);
                    const double reranked = recall_in(cpu_ids.data(), kRecallK, kRecallK, gt.data(), kgt, nq);
                    const auto chk = check_scan(ix, queries.data(), nc, int(bits), int(nprobe), int(C),
                                                cand.data(), est.data());
                    all_ok = all_ok && chk.mismatch == 0 && rerank_mismatch == 0;

                    std::printf("nprobe=%-3lld C=%-4lld scan_ms=%7.1f qps=%8.0f (coarse %.1f, scan %.1f, select %.1f)  "
                                "cand_recall=%.4f R@10 no_rerank=%.4f rerank=%.4f cpu_rerank_ms=%.1f  "
                                "check: max_rel=%.1e scan_mismatch=%lld rerank_mismatch=%lld\n",
                                (long long)nprobe, (long long)C, scan_ms, nq / (scan_ms / 1e3), parts.coarse,
                                parts.scan, parts.select, cand_recall, no_rerank, reranked, median(rerank_times),
                                chk.max_rel, (long long)chk.mismatch, (long long)rerank_mismatch);
                    if (csv)
                        std::fprintf(csv, "%lld,%lld,%lld,%lld,%d,%.2f,%.2f,%.3f,%.1f,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.3f,%lld,%.3e,%lld,%lld,%s\n",
                                     (long long)bits, (long long)nprobe, (long long)C, (long long)nq, batch,
                                     index_mb, work_mb, scan_ms, nq / (scan_ms / 1e3), parts.coarse, parts.scan,
                                     parts.select, cand_recall, no_rerank, reranked, median(rerank_times),
                                     (long long)nc, chk.max_rel, (long long)chk.mismatch, (long long)rerank_mismatch,
                                     topc_name.c_str());
                }
            }
        }

        std::printf("\n");
        budget.report(stdout);
        std::printf("%s\n", all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (csv) std::fclose(csv);
        cudaFree(rr_queries);
        cudaFree(rr_cand);
        cudaFree(rr_ids);
        cudaFree(rr_dists);
        cudaHostUnregister(base.data());
        return all_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
