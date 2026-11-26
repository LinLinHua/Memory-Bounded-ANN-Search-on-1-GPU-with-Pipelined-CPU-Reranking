// Sequential vs pipelined search (XJ, results 4 and 5).
//
//   sequential : for each micro-batch, GPU scan, then CPU rerank, one after the other
//   pipelined  : a scan thread runs the GPU scan of batch b+1 while the OpenMP team reranks batch b.
//                Two candidate buffers form a queue of depth 1, so the scan never runs more than
//                one batch ahead of the rerank.
//
// For every micro-batch size it reports throughput, the per-batch latency (P50, P99), the measured
// pipeline speedup and the one the formula predicts from the sequential stage times:
//   predicted = (T_scan + T_rerank) / max(T_scan, T_rerank)
// and checks the pipeline for races: one extra pipelined run keeps the candidates it reranked, and a
// plain rerank of those candidates afterwards must give bit-identical results.
//
// Sequential and pipelined runs can still differ slightly, because the scan itself is not
// deterministic: when several estimates tie at the C-th place, which of them are kept depends on the
// order of atomic updates. That difference is reported (seq_pipe_diff) but is not an error.
//
// Latency is the time from the start of a batch's scan to the end of its rerank, with every query
// available at the start. It is the per-batch service time, not a latency under an arrival process.
//
// Usage:
//   bench_pipeline --base B --query Q --gt G --index DIR [--label sift]
//                  [--bits 4] [--topc fused] [--nprobe 64] [--C 100]
//                  [--batch 1,16,64,256,1024] [--nq 10000] [--threads T] [--prefetch 4]
//                  [--reps 5] [--budget-mb 16384] [--csv results/pipeline.csv]
//   --threads is the rerank team size in both modes (default: all OpenMP threads minus one, which the
//   scan thread uses).

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include "cli_utils.h"
#include "cpu_rerank.h"
#include "dataset_io.h"
#include "device_budget.h"
#include "ivf_index.h"
#include "ivf_scan.cuh"

namespace {

constexpr int kRecallK = 10;

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Nearest-rank percentile, p in (0, 1].
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const auto rank = static_cast<int64_t>(std::ceil(p * double(v.size())));
    return v[std::clamp<int64_t>(rank - 1, 0, static_cast<int64_t>(v.size()) - 1)];
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

struct Workload {
    const float* base;
    const float* queries;
    int64_t d, nq;
    int nprobe, C, batch, prefetch;
};

struct RunResult {
    double total_ms = 0;            // wall clock of the whole query set
    double scan_ms = 0;             // sum of per-batch scan times
    double rerank_ms = 0;           // sum of per-batch rerank times
    std::vector<double> latency;    // per batch: scan start -> rerank end
    std::vector<int32_t> ids;       // nq x 10
    std::vector<float> dists;       // nq x 10
};

int64_t num_batches(const Workload& w) { return (w.nq + w.batch - 1) / w.batch; }

RerankArgs rerank_args(const Workload& w, int64_t q0, int64_t nqb, const int32_t* cand) {
    return RerankArgs{w.base, w.queries + q0 * w.d, cand, w.d, nqb, w.C, kRecallK, w.prefetch};
}

RunResult run_sequential(IvfScanner& scanner, const Workload& w) {
    RunResult r;
    r.ids.resize(w.nq * kRecallK);
    r.dists.resize(w.nq * kRecallK);
    std::vector<int32_t> cand(int64_t(w.batch) * w.C);
    std::vector<float> est(int64_t(w.batch) * w.C);

    const auto t0 = Clock::now();
    for (int64_t q0 = 0; q0 < w.nq; q0 += w.batch) {
        const int64_t nqb = std::min<int64_t>(w.batch, w.nq - q0);
        const auto a = Clock::now();
        scanner.search(w.queries + q0 * w.d, nqb, w.nprobe, w.C, cand.data(), est.data());
        const auto b = Clock::now();
        cpu_rerank(rerank_args(w, q0, nqb, cand.data()), r.ids.data() + q0 * kRecallK,
                   r.dists.data() + q0 * kRecallK);
        const auto c = Clock::now();
        r.scan_ms += ms_between(a, b);
        r.rerank_ms += ms_between(b, c);
        r.latency.push_back(ms_between(a, c));
    }
    r.total_ms = ms_between(t0, Clock::now());
    return r;
}

// The scan thread fills slot b % 2 with the candidates of batch b; the calling thread (and its
// OpenMP team) reranks it and hands the slot back.
// keep, when given, receives the candidates of every batch as the rerank used them (nq x C).
RunResult run_pipelined(IvfScanner& scanner, const Workload& w, std::vector<int32_t>* keep = nullptr) {
    RunResult r;
    r.ids.resize(w.nq * kRecallK);
    r.dists.resize(w.nq * kRecallK);
    const int64_t nb = num_batches(w);
    std::vector<int32_t> cand[2];
    std::vector<float> est[2];
    for (int s = 0; s < 2; s++) {
        cand[s].resize(int64_t(w.batch) * w.C);
        est[s].resize(int64_t(w.batch) * w.C);
    }
    std::vector<Clock::time_point> scan_start(nb), scan_end(nb), rerank_end(nb);

    std::mutex mu;
    std::condition_variable cv;
    bool full[2] = {false, false};
    bool failed = false;
    std::exception_ptr scan_error;

    const auto t0 = Clock::now();
    std::thread scan_thread([&] {
        try {
            CUDA_CHECK(cudaSetDevice(0));
            for (int64_t b = 0; b < nb; b++) {
                const int s = static_cast<int>(b % 2);
                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return !full[s] || failed; });
                    if (failed) return;
                }
                const int64_t q0 = b * w.batch;
                const int64_t nqb = std::min<int64_t>(w.batch, w.nq - q0);
                scan_start[b] = Clock::now();
                scanner.search(w.queries + q0 * w.d, nqb, w.nprobe, w.C, cand[s].data(), est[s].data());
                scan_end[b] = Clock::now();
                {
                    std::lock_guard<std::mutex> lock(mu);
                    full[s] = true;
                }
                cv.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu);
            scan_error = std::current_exception();
            failed = true;
            cv.notify_all();
        }
    });

    for (int64_t b = 0; b < nb; b++) {
        const int s = static_cast<int>(b % 2);
        {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&] { return full[s] || failed; });
            if (failed) break;
        }
        const int64_t q0 = b * w.batch;
        const int64_t nqb = std::min<int64_t>(w.batch, w.nq - q0);
        const auto a = Clock::now();
        cpu_rerank(rerank_args(w, q0, nqb, cand[s].data()), r.ids.data() + q0 * kRecallK,
                   r.dists.data() + q0 * kRecallK);
        rerank_end[b] = Clock::now();
        r.rerank_ms += ms_between(a, rerank_end[b]);
        if (keep) std::copy_n(cand[s].data(), nqb * w.C, keep->data() + q0 * w.C);
        {
            std::lock_guard<std::mutex> lock(mu);
            full[s] = false;
        }
        cv.notify_all();
    }
    scan_thread.join();
    if (scan_error) std::rethrow_exception(scan_error);
    r.total_ms = ms_between(t0, Clock::now());

    for (int64_t b = 0; b < nb; b++) {
        r.scan_ms += ms_between(scan_start[b], scan_end[b]);
        r.latency.push_back(ms_between(scan_start[b], rerank_end[b]));
    }
    return r;
}

// Entries whose ID or distance bits differ.
int64_t count_diff(const std::vector<int32_t>& ids_a, const std::vector<float>& dists_a,
                   const std::vector<int32_t>& ids_b, const std::vector<float>& dists_b) {
    int64_t bad = 0;
    for (size_t i = 0; i < ids_a.size(); i++)
        if (ids_a[i] != ids_b[i] || std::memcmp(&dists_a[i], &dists_b[i], sizeof(float)) != 0) bad++;
    return bad;
}

// Race check: rerank the candidates a pipelined run used, after the fact, and compare.
int64_t pipeline_race_check(IvfScanner& scanner, const Workload& w) {
    std::vector<int32_t> cand(w.nq * w.C);
    const RunResult pipe = run_pipelined(scanner, w, &cand);
    std::vector<int32_t> ids(w.nq * kRecallK);
    std::vector<float> dists(w.nq * kRecallK);
    cpu_rerank(rerank_args(w, 0, w.nq, cand.data()), ids.data(), dists.data());
    return count_diff(pipe.ids, pipe.dists, ids, dists);
}

// Runs one mode reps + 1 times (the first is warm-up) and keeps the run with the median total.
template <typename Fn>
RunResult median_run(Fn run, int reps) {
    run();
    std::vector<RunResult> runs;
    for (int i = 0; i < reps; i++) runs.push_back(run());
    std::sort(runs.begin(), runs.end(),
              [](const RunResult& x, const RunResult& y) { return x.total_ms < y.total_ms; });
    return runs[runs.size() / 2];
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
        const int bits = static_cast<int>(args.num("bits", 4));
        const std::string topc_name = args.str("topc", "fused");
        const int nprobe = static_cast<int>(args.num("nprobe", 64));
        const int C = static_cast<int>(args.num("C", 100));
        const auto batch_list = args.list("batch", "1,16,64,256,1024");
        const int64_t nq_arg = args.num("nq", -1);
        const int threads_arg = static_cast<int>(args.num("threads", -1));
        const int prefetch = static_cast<int>(args.num("prefetch", 4));
        const int reps = static_cast<int>(args.num("reps", 5));
        const int64_t budget_mb = args.num("budget-mb", 16384);
        const std::string csv_path = args.str("csv");
        args.reject_unknown();
        if (topc_name != "fused" && topc_name != "select" && topc_name != "sort")
            throw std::runtime_error("--topc must be fused, select or sort");
        const TopC topc = topc_name == "sort" ? TopC::Sort : topc_name == "select" ? TopC::Select : TopC::Fused;
        if (reps < 1) throw std::runtime_error("reps must be positive");

        // ---- data
        int64_t n, d, nq_file, dq, ngt, kgt;
        const auto base = load_vectors<float>(base_path, n, d);
        const auto queries = load_vectors<float>(query_path, nq_file, dq);
        const auto gt = load_vectors<int32_t>(gt_path, ngt, kgt);
        const IvfIndex ix = load_ivf(index_dir);
        if (d != dq || d != ix.d || n != ix.n) throw std::runtime_error("data and index do not match");
        const int64_t nq = nq_arg < 0 ? nq_file : std::min(nq_arg, nq_file);
        if (ngt < nq || kgt < kRecallK) throw std::runtime_error("ground truth is too small");

        const int threads = threads_arg > 0 ? threads_arg : std::max(1, omp_get_max_threads() - 1);
        omp_set_num_threads(threads);
        std::printf("n=%lld d=%lld nq=%lld bits=%d topc=%s nprobe=%d C=%d rerank_threads=%d prefetch=%d\n",
                    (long long)n, (long long)d, (long long)nq, bits, topc_name.c_str(), nprobe, C, threads,
                    prefetch);

        // ---- GPU: the scan thread sleeps while it waits for the GPU instead of spinning on a core
        // the rerank could use. The flag must be set before the context exists.
        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));
        DeviceBudget budget(static_cast<size_t>(budget_mb) << 20);

        FILE* csv = open_csv(csv_path,
                             "label,bits,topc,nprobe,C,batch,nq,threads,prefetch,mode,total_ms,qps,scan_ms,"
                             "rerank_ms,predicted_speedup,speedup,lat_p50_ms,lat_p99_ms,lat_mean_ms,recall10,"
                             "mismatch,seq_pipe_diff,index_mb,work_mb,budget_mb,peak_mb,measured_peak_mb");
        bool all_ok = true;

        for (const int64_t batch : batch_list) {
            IvfScanner scanner(ix, bits, static_cast<int>(batch), nprobe, C, budget, topc);
            const double index_mb = scanner.index_bytes() / double(1 << 20);
            const double work_mb = scanner.work_bytes() / double(1 << 20);
            const Workload w{base.data(), queries.data(), d, nq, nprobe, C, static_cast<int>(batch), prefetch};

            const RunResult seq = median_run([&] { return run_sequential(scanner, w); }, reps);
            const RunResult pipe = median_run([&] { return run_pipelined(scanner, w); }, reps);
            const int64_t mismatch = pipeline_race_check(scanner, w);
            const int64_t seq_pipe_diff = count_diff(seq.ids, seq.dists, pipe.ids, pipe.dists);
            all_ok = all_ok && mismatch == 0;

            const double predicted = (seq.scan_ms + seq.rerank_ms) / std::max(seq.scan_ms, seq.rerank_ms);
            const double speedup = seq.total_ms / pipe.total_ms;
            const double recall = recall_at_10(seq.ids.data(), gt.data(), kgt, nq);
            const double recall_pipe = recall_at_10(pipe.ids.data(), gt.data(), kgt, nq);

            std::printf("\n=== batch %lld  index %.1f MB  work %.1f MB  recall@10 %.4f / %.4f  "
                        "pipeline mismatch %lld  seq-vs-pipe diff %lld (scan ties)\n",
                        (long long)batch, index_mb, work_mb, recall, recall_pipe, (long long)mismatch,
                        (long long)seq_pipe_diff);
            for (const auto* run : {&seq, &pipe}) {
                const bool is_seq = run == &seq;
                const double mean =
                    std::accumulate(run->latency.begin(), run->latency.end(), 0.0) / double(run->latency.size());
                std::printf("%-10s total %9.1f ms  qps %9.0f  scan %8.1f ms  rerank %8.1f ms  "
                            "latency p50 %.3f p99 %.3f mean %.3f ms\n",
                            is_seq ? "sequential" : "pipelined", run->total_ms, nq / (run->total_ms / 1e3),
                            run->scan_ms, run->rerank_ms, percentile(run->latency, 0.5),
                            percentile(run->latency, 0.99), mean);
                if (csv)
                    std::fprintf(csv,
                                 "%s,%d,%s,%d,%d,%lld,%lld,%d,%d,%s,%.3f,%.1f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,"
                                 "%.4f,%.5f,%lld,%lld,%.2f,%.2f,%lld,%.2f,%.2f\n",
                                 label.c_str(), bits, topc_name.c_str(), nprobe, C, (long long)batch,
                                 (long long)nq, threads, prefetch, is_seq ? "sequential" : "pipelined",
                                 run->total_ms, nq / (run->total_ms / 1e3), run->scan_ms, run->rerank_ms,
                                 predicted, is_seq ? 1.0 : speedup, percentile(run->latency, 0.5),
                                 percentile(run->latency, 0.99), mean, is_seq ? recall : recall_pipe,
                                 (long long)mismatch, (long long)seq_pipe_diff, index_mb,
                                 work_mb, (long long)budget_mb, budget.peak() / double(1 << 20),
                                 budget.measured_peak() / double(1 << 20));
            }
            std::printf("speedup measured %.3f  predicted %.3f\n", speedup, predicted);
            if (csv) std::fflush(csv);
        }

        std::printf("\n");
        budget.report(stdout);
        std::printf("%s\n", all_ok ? "PIPELINE RACE CHECK PASSED" : "PIPELINE RACE CHECK FAILED");
        if (csv) std::fclose(csv);
        return all_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
