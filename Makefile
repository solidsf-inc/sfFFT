# Default target is the NVIDIA GB10 (DGX Spark). Override with e.g. `make ARCH=sm_90`.
NVCC ?= nvcc
ARCH ?= sm_121
NVCCFLAGS ?= -O3 -std=c++17
CXX ?= c++
TEST_BIN ?= tests/selection_test

sffft: src/sffft.cu src/selection.h
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ $< -lcufft

bench: sffft
	./scripts/run_bench.sh

clean:
	rm -f sffft $(TEST_BIN)

test:
	$(CXX) -O2 -std=c++17 -o $(TEST_BIN) tests/selection_test.cpp
	$(TEST_BIN)

test-gpu: sffft
	bash tests/precision_smoke.sh

.PHONY: bench clean test test-gpu
