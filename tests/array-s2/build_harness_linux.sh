#!/bin/sh
# Builds the S2 array harness's CPU side on Linux (no GPU backend: array_s2_gpu_none.cpp), against the
# library `cmake --build build --target superslm` writes. Usage: tests/array-s2/build_harness_linux.sh [out]
set -e
cd "$(dirname "$0")/../.."
out="${1:-build/array_s2}"
cmake --build build --target superslm
g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Itests/array-s2 \
	tests/array-s2/array_s2_main.cpp tests/array-s2/array_s2_cpu.cpp tests/array-s2/array_s2_gpu_none.cpp \
	build/libsuperslm.a -lpthread -o "$out"
echo "built $out"
