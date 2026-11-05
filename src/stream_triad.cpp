// STREAM-style host memory bandwidth test (copy and triad), used to bound CPU rerank throughput.
//
// Usage:
//   stream_triad [--mb M] [--reps R]   (M = MiB per array, default 1024)

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    long long mb = 1024;
    int reps = 10;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--mb")) mb = std::atoll(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
    }
    const long long n = mb * 1024 * 1024 / sizeof(double);
    std::vector<double> a(n), b(n), c(n);

    // First touch in parallel so pages land on the NUMA node of the thread that uses them.
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    double best_copy = 1e30, best_triad = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < n; i++) c[i] = a[i];
        auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
        auto t2 = std::chrono::steady_clock::now();
        best_copy = std::min(best_copy, std::chrono::duration<double>(t1 - t0).count());
        best_triad = std::min(best_triad, std::chrono::duration<double>(t2 - t1).count());
    }

    const double bytes = static_cast<double>(n) * sizeof(double);
    std::printf("threads=%d mib_per_array=%lld\n", omp_get_max_threads(), mb);
    std::printf("copy_gbs=%.1f triad_gbs=%.1f\n", 2 * bytes / best_copy / 1e9,
                3 * bytes / best_triad / 1e9);
    std::printf("check=%.1f\n", a[n / 2]);
    return 0;
}
