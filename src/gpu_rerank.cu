#include "gpu_rerank.cuh"

#include <cfloat>
#include <stdexcept>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// One block per query. Each warp computes whole candidate distances (lanes split the dimensions,
// so one candidate is a contiguous 512-byte read over PCIe for SIFT), then the block picks the
// k smallest with k rounds of argmin over shared memory. k is small, so this is cheap next to the
// PCIe reads.
__global__ void rerank_kernel(GpuRerankArgs a, int32_t* out_ids, float* out_dists) {
    extern __shared__ float dist[];  // C entries
    __shared__ float warp_best[kWarps];
    __shared__ int warp_arg[kWarps];

    const int64_t q = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int32_t* cand = a.cand + q * a.C;
    const float4* query = reinterpret_cast<const float4*>(a.queries + q * a.d);
    const int64_t d4 = a.d / 4;

    for (int64_t j = warp; j < a.C; j += kWarps) {
        const int32_t id = cand[j];
        float sum = 0.0f;
        if (id >= 0) {
            const float4* v = reinterpret_cast<const float4*>(a.base + static_cast<int64_t>(id) * a.d);
            for (int64_t i = lane; i < d4; i += 32) {
                const float4 x = v[i];
                const float4 y = query[i];
                const float dx = x.x - y.x, dy = x.y - y.y, dz = x.z - y.z, dw = x.w - y.w;
                sum += dx * dx + dy * dy + dz * dz + dw * dw;
            }
        }
        for (int off = 16; off > 0; off /= 2) sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) dist[j] = id >= 0 ? sum : FLT_MAX;
    }
    __syncthreads();

    for (int r = 0; r < a.k; r++) {
        float best = FLT_MAX;
        int arg = -1;
        for (int64_t j = threadIdx.x; j < a.C; j += kThreads) {
            if (dist[j] < best) {
                best = dist[j];
                arg = static_cast<int>(j);
            }
        }
        for (int off = 16; off > 0; off /= 2) {
            const float ob = __shfl_down_sync(0xffffffff, best, off);
            const int oa = __shfl_down_sync(0xffffffff, arg, off);
            if (ob < best || (ob == best && oa >= 0 && (arg < 0 || oa < arg))) {
                best = ob;
                arg = oa;
            }
        }
        if (lane == 0) {
            warp_best[warp] = best;
            warp_arg[warp] = arg;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < kWarps; w++) {
                if (warp_best[w] < best ||
                    (warp_best[w] == best && warp_arg[w] >= 0 && (arg < 0 || warp_arg[w] < arg))) {
                    best = warp_best[w];
                    arg = warp_arg[w];
                }
            }
            const bool valid = arg >= 0 && best < FLT_MAX;
            out_ids[q * a.k + r] = valid ? cand[arg] : -1;
            out_dists[q * a.k + r] = valid ? best : __int_as_float(0x7f800000);
            if (valid) dist[arg] = FLT_MAX;
        }
        __syncthreads();
    }
}

}  // namespace

size_t gpu_rerank_smem_bytes(int64_t C) { return static_cast<size_t>(C) * sizeof(float); }

void gpu_rerank(const GpuRerankArgs& args, int32_t* out_ids, float* out_dists, cudaStream_t stream) {
    if (args.d % 4 != 0) throw std::runtime_error("gpu_rerank needs d divisible by 4");
    if (args.k < 1 || args.k > 32) throw std::runtime_error("gpu_rerank needs 1 <= k <= 32");
    const size_t smem = gpu_rerank_smem_bytes(args.C);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(rerank_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
    }
    rerank_kernel<<<static_cast<unsigned>(args.nq), kThreads, smem, stream>>>(args, out_ids, out_dists);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
