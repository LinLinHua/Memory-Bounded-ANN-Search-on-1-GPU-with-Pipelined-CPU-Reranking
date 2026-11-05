# LL's GPU programs. Kept apart from the main Makefile while XJ is editing it; merge later.
# Build on a GPU compute node after "source scripts/env.sh":
#   make -f ll.mk BUILD=build/JOBID
# A40 is sm_86.

NVCC := nvcc
NVCCFLAGS := -std=c++17 -O3 -arch=sm_86 -Xcompiler "-march=native -fopenmp -Wall"
LIBS := -lcublas
BUILD := build

BINS := $(BUILD)/gpu_bruteforce $(BUILD)/bandwidth_bench $(BUILD)/ivf_build $(BUILD)/bench_scan
COMMON := src/gpu_distance.cu src/gpu_distance.cuh src/device_budget.h src/cli_utils.h

.PHONY: all
all: $(BINS)

$(BUILD):
	mkdir -p $@

$(BUILD)/gpu_bruteforce: src/gpu_bruteforce.cu src/dataset_io.h $(COMMON) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/gpu_bruteforce.cu src/gpu_distance.cu $(LIBS) -o $@

$(BUILD)/bandwidth_bench: src/bandwidth_bench.cu $(COMMON) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/bandwidth_bench.cu src/gpu_distance.cu $(LIBS) -o $@

$(BUILD)/ivf_build: src/ivf_build.cu src/dataset_io.h $(COMMON) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/ivf_build.cu src/gpu_distance.cu $(LIBS) -o $@

$(BUILD)/bench_scan: src/bench_scan.cu src/ivf_scan.cu src/ivf_scan.cuh src/ivf_index.h \
		src/cpu_rerank.cpp src/cpu_rerank.h src/gpu_rerank.cu src/gpu_rerank.cuh $(COMMON) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/bench_scan.cu src/ivf_scan.cu src/gpu_distance.cu \
		src/cpu_rerank.cpp src/gpu_rerank.cu $(LIBS) -o $@
