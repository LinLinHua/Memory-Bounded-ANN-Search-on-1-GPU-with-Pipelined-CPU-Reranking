// Software-enforced GPU memory budget B: every device allocation goes through DeviceBudget,
// and an allocation that would push the total above B throws instead of succeeding.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess)                                                          \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err_)); \
    } while (0)

class DeviceBudget {
public:
    explicit DeviceBudget(size_t budget_bytes) : budget_(budget_bytes) {
        size_t total = 0;
        CUDA_CHECK(cudaMemGetInfo(&baseline_free_, &total));
        min_free_ = baseline_free_;
    }

    ~DeviceBudget() {
        for (auto& entry : blocks_) cudaFree(entry.first);
    }

    DeviceBudget(const DeviceBudget&) = delete;
    DeviceBudget& operator=(const DeviceBudget&) = delete;

    template <typename T>
    T* alloc(size_t count, const char* tag) {
        const size_t bytes = count * sizeof(T);
        if (used_ + bytes > budget_) {
            throw std::runtime_error(std::string("budget exceeded by ") + tag + ": need " +
                                     std::to_string(used_ + bytes) + " bytes, budget " +
                                     std::to_string(budget_));
        }
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        blocks_[ptr] = {bytes, tag};
        used_ += bytes;
        if (used_ > peak_) peak_ = used_;
        sample();
        return static_cast<T*>(ptr);
    }

    void free(void* ptr) {
        auto it = blocks_.find(ptr);
        if (it == blocks_.end()) throw std::runtime_error("free of a pointer not owned by budget");
        used_ -= it->second.bytes;
        blocks_.erase(it);
        CUDA_CHECK(cudaFree(ptr));
    }

    // Records the lowest free memory seen, so the peak can be checked independently of our counter.
    void sample() {
        size_t free_now = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_now, &total));
        if (free_now < min_free_) min_free_ = free_now;
    }

    size_t budget() const { return budget_; }
    size_t used() const { return used_; }
    size_t peak() const { return peak_; }
    // Device memory consumed since construction, as seen by cudaMemGetInfo (includes allocator rounding).
    size_t measured_peak() const { return baseline_free_ - min_free_; }

    void report(FILE* out) const {
        std::fprintf(out, "budget_bytes=%zu peak_bytes=%zu measured_peak_bytes=%zu\n", budget_,
                     peak_, measured_peak());
        for (const auto& entry : blocks_)
            std::fprintf(out, "  live %s: %zu bytes\n", entry.second.tag.c_str(), entry.second.bytes);
    }

private:
    struct Block {
        size_t bytes;
        std::string tag;
    };

    size_t budget_;
    size_t used_ = 0;
    size_t peak_ = 0;
    size_t baseline_free_ = 0;
    size_t min_free_ = 0;
    std::map<void*, Block> blocks_;
};
