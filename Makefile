# Build on a compute node (source scripts/env.sh first): -march=native depends on the CPU.
# A40 is sm_86.

CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -O3 -march=native -ffast-math -fopenmp -Wall -Wextra
NVCCFLAGS := -std=c++17 -O3 -arch=sm_86 -Xcompiler "-march=native -ffast-math -fopenmp -Wall"
BUILD := build
LIBS := -lcublas

CPU_BINS := $(BUILD)/serial_bruteforce $(BUILD)/stream_triad $(BUILD)/bench_cpu_rerank
GPU_BINS := $(BUILD)/bench_gpu_rerank $(BUILD)/bench_pipeline $(BUILD)/bench_rerank_breakdown

.PHONY: all cpu gpu clean
all: cpu gpu
cpu: $(CPU_BINS)
gpu: $(GPU_BINS)

$(BUILD):
	mkdir -p $@

$(BUILD)/serial_bruteforce: src/serial_bruteforce.cpp src/dataset_io.h src/vecs_io.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/stream_triad: src/stream_triad.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/bench_cpu_rerank: src/bench_cpu_rerank.cpp src/cpu_rerank.cpp src/cpu_rerank.h \
		src/synthetic_candidates.h src/vecs_io.h | $(BUILD)
	$(CXX) $(CXXFLAGS) src/bench_cpu_rerank.cpp src/cpu_rerank.cpp -o $@

$(BUILD)/bench_gpu_rerank: src/bench_gpu_rerank.cu src/gpu_rerank.cu src/gpu_rerank.cuh \
		src/cpu_rerank.cpp src/cpu_rerank.h src/device_budget.h src/synthetic_candidates.h \
		src/vecs_io.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/bench_gpu_rerank.cu src/gpu_rerank.cu src/cpu_rerank.cpp -o $@

# Scan + rerank programs (XJ). They link LL's scanner, so they need cuBLAS.
SCAN_DEPS := src/ivf_scan.cu src/ivf_scan.cuh src/ivf_index.h src/gpu_distance.cu src/gpu_distance.cuh \
	src/cpu_rerank.cpp src/cpu_rerank.h src/device_budget.h src/cli_utils.h src/dataset_io.h

$(BUILD)/bench_pipeline: src/bench_pipeline.cu $(SCAN_DEPS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/bench_pipeline.cu src/ivf_scan.cu src/gpu_distance.cu src/cpu_rerank.cpp \
		$(LIBS) -o $@

$(BUILD)/bench_rerank_breakdown: src/bench_rerank_breakdown.cu src/gpu_rerank.cu src/gpu_rerank.cuh \
		$(SCAN_DEPS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/bench_rerank_breakdown.cu src/ivf_scan.cu src/gpu_distance.cu \
		src/cpu_rerank.cpp src/gpu_rerank.cu $(LIBS) -o $@

clean:
	rm -rf $(BUILD)
