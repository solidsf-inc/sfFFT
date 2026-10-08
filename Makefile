# Default target is the NVIDIA GB10 (DGX Spark). Override with e.g. `make ARCH=sm_90`.
NVCC ?= nvcc
ARCH ?= sm_121
NVCCFLAGS ?= -O3 -std=c++17

sffft: src/sffft.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ $< -lcufft

bench: sffft
	./scripts/run_bench.sh

clean:
	rm -f sffft

.PHONY: bench clean
