// Measures the bandwidths our cost model needs (LL, Day 1-3):
//   device memory : read kernel, copy kernel, cudaMemcpy device-to-device
//   PCIe          : cudaMemcpy host <-> device, from pinned and from pageable host memory
//   zero-copy     : a kernel reading mapped pinned host memory (how the GPU rerank reads vectors)
// Every number is the median of the timed runs after one warm-up run.
//
// Usage: bandwidth_bench [--mb 1024] [--reps 10] [--csv results/bandwidth_a40.csv]

#include <cstdio>
#include <cstring>
#include <vector>

#include "cli_utils.h"
#include "gpu_distance.cuh"

namespace {

// Sums the buffer so the loads are not optimized away. The data is zero, so the sink is never written.
__global__ void read_kernel(const float4* src, int64_t n4, float* sink) {
    float sum = 0.0f;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = src[i];
        sum += v.x + v.y + v.z + v.w;
    }
    if (sum == -1.0f) *sink = sum;
}

__global__ void copy_kernel(const float4* src, float4* dst, int64_t n4) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = src[i];
}

// Runs fn once to warm up, then the timed runs; returns the median in milliseconds.
template <typename Fn>
double median_ms(int runs, Fn fn) {
    GpuTimer timer;
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> times;
    for (int r = 0; r < runs; r++) {
        timer.start();
        fn();
        times.push_back(timer.stop_ms());
    }
    return median(times);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        const size_t mb = args.num("mb", 1024);
        const int runs = static_cast<int>(args.num("reps", 10));
        const std::string csv_path = args.str("csv");
        args.reject_unknown();

        const size_t bytes = mb << 20;
        const int64_t n4 = bytes / sizeof(float4);

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        const unsigned grid = prop.multiProcessorCount * 8;
        const unsigned threads = 256;
        std::printf("gpu=%s sms=%d buffer_mb=%zu runs=%d\n", prop.name, prop.multiProcessorCount, mb, runs);

        FILE* csv = open_csv(csv_path, "test,mb,ms,gbs");
        auto report = [&](const char* test, size_t mbytes, double bytes_moved, double ms) {
            const double gbs = bytes_moved / 1e9 / (ms / 1e3);
            std::printf("%-15s mb=%-5zu ms=%8.3f gbs=%7.1f\n", test, mbytes, ms, gbs);
            if (csv) std::fprintf(csv, "%s,%zu,%.4f,%.2f\n", test, mbytes, ms, gbs);
        };

        // ---- device memory
        float4* d_a = nullptr;
        float4* d_b = nullptr;
        float* d_sink = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_a, 0, bytes));

        auto device_read = [&] { read_kernel<<<grid, threads>>>(d_a, n4, d_sink); };
        auto device_copy = [&] { copy_kernel<<<grid, threads>>>(d_a, d_b, n4); };
        auto memcpy_d2d = [&] { CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice)); };
        report("device_read", mb, bytes, median_ms(runs, device_read));
        report("device_copy", mb, 2.0 * bytes, median_ms(runs, device_copy));  // read + write
        report("memcpy_d2d", mb, 2.0 * bytes, median_ms(runs, memcpy_d2d));

        // ---- PCIe, several transfer sizes
        char* pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&pinned, bytes));
        std::memset(pinned, 0, bytes);
        std::vector<char> pageable(bytes, 0);  // zero-filled, so every page is already mapped

        const std::vector<size_t> sizes_mb = {1, 16, 256, mb};
        for (const size_t size_mb : sizes_mb) {
            if (size_mb > mb) continue;
            const size_t sz = size_mb << 20;
            auto h2d_pinned = [&] { CUDA_CHECK(cudaMemcpy(d_a, pinned, sz, cudaMemcpyHostToDevice)); };
            auto d2h_pinned = [&] { CUDA_CHECK(cudaMemcpy(pinned, d_a, sz, cudaMemcpyDeviceToHost)); };
            auto h2d_pageable = [&] { CUDA_CHECK(cudaMemcpy(d_a, pageable.data(), sz, cudaMemcpyHostToDevice)); };
            auto d2h_pageable = [&] { CUDA_CHECK(cudaMemcpy(pageable.data(), d_a, sz, cudaMemcpyDeviceToHost)); };
            report("h2d_pinned", size_mb, sz, median_ms(runs, h2d_pinned));
            report("d2h_pinned", size_mb, sz, median_ms(runs, d2h_pinned));
            report("h2d_pageable", size_mb, sz, median_ms(runs, h2d_pageable));
            report("d2h_pageable", size_mb, sz, median_ms(runs, d2h_pageable));
        }

        // ---- zero-copy: the kernel reads pinned host memory over PCIe
        CUDA_CHECK(cudaFreeHost(pinned));
        CUDA_CHECK(cudaHostAlloc(&pinned, bytes, cudaHostAllocMapped));
        std::memset(pinned, 0, bytes);
        float4* mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), pinned, 0));
        auto zerocopy_read = [&] { read_kernel<<<grid, threads>>>(mapped, n4, d_sink); };
        report("zerocopy_read", mb, bytes, median_ms(runs, zerocopy_read));

        CUDA_CHECK(cudaFreeHost(pinned));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_sink);
        if (csv) std::fclose(csv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
