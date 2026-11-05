// Builds the IVF index on the GPU and encodes the residuals as 8-, 4- and 1-bit codes (LL, Day 4-7).
//
//   1. k-means on a random training sample (Lloyd iterations on the GPU):
//        assignment = GEMM + argmin, update = atomic per-centroid sums
//   2. assign every base vector to its nearest centroid, group vectors by list (CSR layout)
//   3. encode residuals r = x - centroid on the CPU:
//        8/4-bit : per-dimension scalar quantization over the residual range
//        1-bit   : sign bits of r after a random orthogonal rotation, plus |r|
//
// The output matches src/build_ivf.py, so src/ref_scan.py can check this index too:
//   centroids.f32 (L x d)   list_offsets.i64 (L+1)   ids.i32 (n, original ID of each row)
//   codes8.u8 (n x d)   codes4.u8 (n x d/2, low nibble = even dim)   codes1.u32 (n x d/32)
//   norms.f32 (n)   sq_min.f32, sq_step8.f32, sq_step4.f32 (d each)   rotation.f32 (d x d)
//   meta.json
//
// Usage:
//   ivf_build --base B --out DIR [--nlist 1024] [--train 0] [--iters 20] [--seed 1234] [--max-base N]
//   --train 0 uses min(n, 256 * nlist) training points.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include "cli_utils.h"
#include "dataset_io.h"
#include "device_budget.h"
#include "gpu_distance.cuh"

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int64_t kAssignChunk = 65536;  // points per GEMM in the assignment step

// Nearest centroid of each point, one warp per point.
// Score = |c|^2 - 2 x.c, which ranks centroids like |x - c|^2 (|x|^2 is the same for all).
__global__ void argmin_kernel(const float* dots, int64_t npts, int64_t L, const float* cnorm,
                              int32_t* assign, float* score) {
    const int lane = threadIdx.x % 32;
    const int64_t p = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / 32;
    if (p >= npts) return;

    float best = INFINITY;
    int32_t arg = INT32_MAX;
    for (int64_t l = lane; l < L; l += 32) {
        const float s = cnorm[l] - 2.0f * dots[p * L + l];
        if (s < best) {
            best = s;
            arg = static_cast<int32_t>(l);
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {  // smallest score, then smallest centroid id
        const float other = __shfl_down_sync(0xffffffffu, best, offset);
        const int32_t other_arg = __shfl_down_sync(0xffffffffu, arg, offset);
        if (other < best || (other == best && other_arg < arg)) {
            best = other;
            arg = other_arg;
        }
    }
    if (lane == 0) {
        assign[p] = arg;
        score[p] = best;
    }
}

// Adds every point into its centroid's sum. One thread per (point, dimension).
__global__ void accumulate_kernel(const float* x, int64_t npts, int64_t d, const int32_t* assign,
                                  float* sums, int* counts) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= npts * d) return;
    const int64_t p = i / d, j = i % d;
    const int32_t c = assign[p];
    atomicAdd(&sums[c * d + j], x[i]);
    if (j == 0) atomicAdd(&counts[c], 1);
}

// New centroid = mean of its points. Empty clusters keep their old value (reseeded on the host).
__global__ void divide_kernel(const float* sums, const int* counts, int64_t L, int64_t d, float* cent) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= L * d) return;
    const int count = counts[i / d];
    if (count > 0) cent[i] = sums[i] / count;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Assigns n device-resident points to their nearest centroid.
// Returns the k-means objective (sum of squared distances).
double assign_points(cublasHandle_t handle, const float* d_x, const float* d_xnorm, int64_t n,
                     int64_t d, const float* d_cent, const float* d_cnorm, int64_t L, float* d_dots,
                     int32_t* d_assign, float* d_score) {
    for (int64_t p0 = 0; p0 < n; p0 += kAssignChunk) {
        const int64_t m = std::min(kAssignChunk, n - p0);
        inner_products(handle, d_x + p0 * d, m, d_cent, L, d, d_dots);
        argmin_kernel<<<blocks_for(m, kWarpsPerBlock), kWarpsPerBlock * 32>>>(d_dots, m, L, d_cnorm,
                                                                              d_assign + p0, d_score + p0);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<float> score(n), xnorm(n);
    CUDA_CHECK(cudaMemcpy(score.data(), d_score, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xnorm.data(), d_xnorm, n * sizeof(float), cudaMemcpyDeviceToHost));
    double objective = 0.0;
    for (int64_t i = 0; i < n; i++) objective += double(xnorm[i]) + double(score[i]);
    return objective;
}

// Lloyd's k-means on the GPU. Starts from L distinct random training points.
std::vector<float> train_kmeans(cublasHandle_t handle, const std::vector<float>& train, int64_t nt,
                                int64_t d, int64_t L, int iters, std::mt19937& rng) {
    std::vector<int64_t> order(nt);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<float> cent(L * d);
    for (int64_t l = 0; l < L; l++)
        std::copy_n(&train[order[l] * d], d, &cent[l * d]);

    DeviceBudget mem(size_t(44) << 30);
    float* d_x = mem.alloc<float>(nt * d, "train points");
    float* d_xnorm = mem.alloc<float>(nt, "train norms");
    float* d_cent = mem.alloc<float>(L * d, "centroids");
    float* d_cnorm = mem.alloc<float>(L, "centroid norms");
    float* d_sums = mem.alloc<float>(L * d, "centroid sums");
    int* d_counts = mem.alloc<int>(L, "centroid counts");
    int32_t* d_assign = mem.alloc<int32_t>(nt, "assignment");
    float* d_score = mem.alloc<float>(nt, "scores");
    float* d_dots = mem.alloc<float>(std::min(kAssignChunk, nt) * L, "gemm tile");

    CUDA_CHECK(cudaMemcpy(d_x, train.data(), nt * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent, cent.data(), L * d * sizeof(float), cudaMemcpyHostToDevice));
    row_sq_norms(d_x, nt, d, d_xnorm);

    std::uniform_int_distribution<int64_t> pick(0, nt - 1);
    std::vector<int> counts(L);
    for (int it = 0; it < iters; it++) {
        row_sq_norms(d_cent, L, d, d_cnorm);
        const double objective =
            assign_points(handle, d_x, d_xnorm, nt, d, d_cent, d_cnorm, L, d_dots, d_assign, d_score);

        CUDA_CHECK(cudaMemset(d_sums, 0, L * d * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, L * sizeof(int)));
        accumulate_kernel<<<blocks_for(nt * d, 256), 256>>>(d_x, nt, d, d_assign, d_sums, d_counts);
        divide_kernel<<<blocks_for(L * d, 256), 256>>>(d_sums, d_counts, L, d, d_cent);
        CUDA_CHECK(cudaGetLastError());

        // Restart empty clusters at a random training point.
        CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, L * sizeof(int), cudaMemcpyDeviceToHost));
        int empty = 0;
        for (int64_t l = 0; l < L; l++) {
            if (counts[l] > 0) continue;
            empty++;
            CUDA_CHECK(cudaMemcpy(d_cent + l * d, &train[pick(rng) * d], d * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        std::printf("  iter %2d  objective %.6e  empty %d\n", it, objective, empty);
    }
    CUDA_CHECK(cudaMemcpy(cent.data(), d_cent, L * d * sizeof(float), cudaMemcpyDeviceToHost));
    return cent;
}

// List id of every base vector.
std::vector<int32_t> assign_all(cublasHandle_t handle, const std::vector<float>& base, int64_t n,
                                int64_t d, const std::vector<float>& cent, int64_t L) {
    DeviceBudget mem(size_t(44) << 30);
    float* d_x = mem.alloc<float>(n * d, "base");
    float* d_xnorm = mem.alloc<float>(n, "base norms");
    float* d_cent = mem.alloc<float>(L * d, "centroids");
    float* d_cnorm = mem.alloc<float>(L, "centroid norms");
    int32_t* d_assign = mem.alloc<int32_t>(n, "assignment");
    float* d_score = mem.alloc<float>(n, "scores");
    float* d_dots = mem.alloc<float>(std::min(kAssignChunk, n) * L, "gemm tile");

    CUDA_CHECK(cudaMemcpy(d_x, base.data(), n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent, cent.data(), L * d * sizeof(float), cudaMemcpyHostToDevice));
    row_sq_norms(d_x, n, d, d_xnorm);
    row_sq_norms(d_cent, L, d, d_cnorm);
    const double objective =
        assign_points(handle, d_x, d_xnorm, n, d, d_cent, d_cnorm, L, d_dots, d_assign, d_score);
    std::printf("  base objective %.6e (mean %.4f per vector)\n", objective, objective / n);

    std::vector<int32_t> assign(n);
    CUDA_CHECK(cudaMemcpy(assign.data(), d_assign, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    return assign;
}

// Random orthogonal d x d matrix: Gram-Schmidt on Gaussian rows (in double).
std::vector<float> random_rotation(int64_t d, std::mt19937& rng) {
    std::normal_distribution<double> gauss;
    std::vector<double> m(d * d);
    for (auto& v : m) v = gauss(rng);
    for (int64_t i = 0; i < d; i++) {
        double* row = &m[i * d];
        for (int64_t j = 0; j < i; j++) {
            const double* prev = &m[j * d];
            double dot = 0.0;
            for (int64_t t = 0; t < d; t++) dot += row[t] * prev[t];
            for (int64_t t = 0; t < d; t++) row[t] -= dot * prev[t];
        }
        double norm = 0.0;
        for (int64_t t = 0; t < d; t++) norm += row[t] * row[t];
        norm = std::sqrt(norm);
        for (int64_t t = 0; t < d; t++) row[t] /= norm;
    }
    return std::vector<float>(m.begin(), m.end());
}

template <typename T>
void write_file(const std::string& dir, const std::string& name, const std::vector<T>& v) {
    std::ofstream f(dir + "/" + name, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
    if (!f) throw std::runtime_error("cannot write " + dir + "/" + name);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        const std::string base_path = args.required("base");
        const std::string out_dir = args.required("out");
        const int64_t L = args.num("nlist", 1024);
        int64_t nt = args.num("train", 0);
        const int iters = static_cast<int>(args.num("iters", 20));
        const uint32_t seed = static_cast<uint32_t>(args.num("seed", 1234));
        const int64_t max_base = args.num("max-base", -1);
        args.reject_unknown();

        const auto t_start = std::chrono::steady_clock::now();
        int64_t n, d;
        const auto base = load_vectors<float>(base_path, n, d, max_base);
        if (d % 32 != 0) throw std::runtime_error("d must be a multiple of 32 (1-bit and 4-bit codes)");
        if (nt <= 0) nt = std::min(n, 256 * L);
        nt = std::min(nt, n);
        if (L > nt) throw std::runtime_error("need at least nlist training points");
        std::printf("base %lld x %lld, nlist %lld, train %lld, iters %d, seed %u\n", (long long)n,
                    (long long)d, (long long)L, (long long)nt, iters, seed);
        if (std::system(("mkdir -p '" + out_dir + "'").c_str()) != 0)
            throw std::runtime_error("cannot create " + out_dir);

        std::mt19937 rng(seed);
        CUDA_CHECK(cudaSetDevice(0));
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

        // ---- 1. k-means on a random sample
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int64_t> sample(n);
        std::iota(sample.begin(), sample.end(), 0);
        std::shuffle(sample.begin(), sample.end(), rng);
        std::vector<float> train(nt * d);
        for (int64_t i = 0; i < nt; i++) std::copy_n(&base[sample[i] * d], d, &train[i * d]);
        const auto cent = train_kmeans(handle, train, nt, d, L, iters, rng);
        std::printf("k-means: %.1f s\n", seconds_since(t0));

        // ---- 2. assign all vectors, group them by list
        t0 = std::chrono::steady_clock::now();
        const auto assign = assign_all(handle, base, n, d, cent, L);
        std::vector<int64_t> offsets(L + 1, 0);
        for (int64_t i = 0; i < n; i++) offsets[assign[i] + 1]++;
        for (int64_t l = 0; l < L; l++) offsets[l + 1] += offsets[l];
        std::vector<int32_t> ids(n);  // row -> original vector id, rows grouped by list
        std::vector<int64_t> next(offsets.begin(), offsets.end() - 1);
        for (int64_t i = 0; i < n; i++) ids[next[assign[i]]++] = static_cast<int32_t>(i);
        std::vector<int64_t> sizes(L);
        for (int64_t l = 0; l < L; l++) sizes[l] = offsets[l + 1] - offsets[l];
        std::sort(sizes.begin(), sizes.end());
        std::printf("assignment: %.1f s; list size min %lld, median %lld, max %lld\n", seconds_since(t0),
                    (long long)sizes.front(), (long long)sizes[L / 2], (long long)sizes.back());

        // ---- 3. encode residuals
        t0 = std::chrono::steady_clock::now();
        auto residual = [&](int64_t row, float* r) {
            const float* x = &base[static_cast<int64_t>(ids[row]) * d];
            const float* c = &cent[static_cast<int64_t>(assign[ids[row]]) * d];
            for (int64_t j = 0; j < d; j++) r[j] = x[j] - c[j];
        };

        // Per-dimension residual range, shared by all lists.
        std::vector<float> vmin(d, INFINITY), vmax(d, -INFINITY);
#pragma omp parallel
        {
            std::vector<float> r(d), lo(d, INFINITY), hi(d, -INFINITY);
#pragma omp for schedule(static)
            for (int64_t row = 0; row < n; row++) {
                residual(row, r.data());
                for (int64_t j = 0; j < d; j++) {
                    lo[j] = std::min(lo[j], r[j]);
                    hi[j] = std::max(hi[j], r[j]);
                }
            }
#pragma omp critical
            for (int64_t j = 0; j < d; j++) {
                vmin[j] = std::min(vmin[j], lo[j]);
                vmax[j] = std::max(vmax[j], hi[j]);
            }
        }
        std::vector<float> step8(d), step4(d);
        for (int64_t j = 0; j < d; j++) {
            const float range = std::max(vmax[j] - vmin[j], 1e-12f);
            step8[j] = range / 255.0f;
            step4[j] = range / 15.0f;
        }
        const auto rotation = random_rotation(d, rng);

        std::vector<uint8_t> codes8(n * d), codes4(n * d / 2);
        std::vector<uint32_t> codes1(n * d / 32, 0);
        std::vector<float> norms(n);
        auto quantize = [](float r, float lo, float step, int levels) {
            return static_cast<uint8_t>(std::clamp(std::nearbyint((r - lo) / step), 0.0f, float(levels)));
        };
#pragma omp parallel
        {
            std::vector<float> r(d), rotated(d);
#pragma omp for schedule(static)
            for (int64_t row = 0; row < n; row++) {
                residual(row, r.data());
                float norm2 = 0.0f;
                for (int64_t j = 0; j < d; j++) {
                    norm2 += r[j] * r[j];
                    codes8[row * d + j] = quantize(r[j], vmin[j], step8[j], 255);
                }
                norms[row] = std::sqrt(norm2);
                for (int64_t j = 0; j < d; j += 2) {
                    const uint8_t lo = quantize(r[j], vmin[j], step4[j], 15);
                    const uint8_t hi = quantize(r[j + 1], vmin[j + 1], step4[j + 1], 15);
                    codes4[row * d / 2 + j / 2] = static_cast<uint8_t>(lo | (hi << 4));
                }
                // rotated = r * R (row vector times matrix), then keep only the signs
                std::fill(rotated.begin(), rotated.end(), 0.0f);
                for (int64_t i = 0; i < d; i++)
                    for (int64_t t = 0; t < d; t++) rotated[t] += r[i] * rotation[i * d + t];
                for (int64_t t = 0; t < d; t++)
                    if (rotated[t] > 0.0f) codes1[row * (d / 32) + t / 32] |= 1u << (t % 32);
            }
        }
        std::printf("encoding: %.1f s\n", seconds_since(t0));

        // ---- write the index
        write_file(out_dir, "centroids.f32", cent);
        write_file(out_dir, "list_offsets.i64", offsets);
        write_file(out_dir, "ids.i32", ids);
        write_file(out_dir, "codes8.u8", codes8);
        write_file(out_dir, "codes4.u8", codes4);
        write_file(out_dir, "codes1.u32", codes1);
        write_file(out_dir, "norms.f32", norms);
        write_file(out_dir, "sq_min.f32", vmin);
        write_file(out_dir, "sq_step8.f32", step8);
        write_file(out_dir, "sq_step4.f32", step4);
        write_file(out_dir, "rotation.f32", rotation);
        std::ofstream meta(out_dir + "/meta.json");
        meta << "{\n  \"n\": " << n << ",\n  \"d\": " << d << ",\n  \"nlist\": " << L
             << ",\n  \"seed\": " << seed << ",\n  \"train\": " << nt << ",\n  \"iters\": " << iters
             << ",\n  \"builder\": \"ivf_build\"\n}\n";

        auto mb = [](size_t bytes) { return bytes / double(1 << 20); };
        std::printf("index sizes (MB): 8-bit %.0f, 4-bit %.0f, 1-bit %.0f (+norms %.0f), raw FP32 %.0f\n",
                    mb(codes8.size()), mb(codes4.size()), mb(codes1.size() * 4), mb(norms.size() * 4),
                    mb(base.size() * 4));
        std::printf("total: %.1f s, wrote %s\n", seconds_since(t_start), out_dir.c_str());
        cublasDestroy(handle);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
