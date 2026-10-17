#!/usr/bin/env bash
# wine-gpu-setup.sh -- run SuperSLM's D3D12 compute path in a GPU-less Linux container:
#   mingw-w64 build  ->  Wine 9 (wine64)  ->  vkd3d-proton (DXIL -> SPIR-V)  ->  Mesa lavapipe (CPU Vulkan)
#
# Usage:
#   wine-gpu-setup.sh setup                 # apt packages, vkd3d-proton download, Xvfb, Wine prefix (idempotent)
#   wine-gpu-setup.sh build <repo>          # mingw-build <repo>/out/superslm_tests.exe + shaders (build.bat's list)
#   wine-gpu-setup.sh run <repo> [exe args] # run <repo>/out/superslm_tests.exe under wine from the repo root
#   wine-gpu-setup.sh exec <exe> [args]     # run any mingw D3D12 exe under the same environment
#
# Why each piece (see wine-gpu-notes.md for the evidence):
#  - Wine's builtin d3d12 (vkd3d 1.10) refuses lavapipe (FL 11_0 check: viewportSubPixelBits) and
#    does not take DXIL anyway; vkd3d-proton does both. Its DLLs sit beside the exe and are forced
#    native with WINEDLLOVERRIDES.
#  - vkd3d-proton reports lavapipe below FL 12_0 (the harness asks for 12_0); VKD3D_FEATURE_LEVEL=12_1
#    overrides the report. Everything SuperSLM's shaders use (int64, 64-bit groupshared, atomics,
#    barriers) works on lavapipe; the override only skips caps the compute path never touches.
#  - winevulkan needs a Wine graphics driver, i.e. an X display: Xvfb on :99 (no real GPU or screen).
#
# Prerequisites not installed by `setup`: mingw-w64 (apt g++-mingw-w64-x86-64-posix mingw-w64-x86-64-dev)
# and the Linux DXC release (github.com/microsoft/DirectXShaderCompiler releases, linux_dxc_*.tar.gz) unpacked
# so that $DXC (default /opt/dxc/bin/dxc) exists. A software-GPU run is a smoke run: it proves byte-identity on
# tiny fixtures; the certified runs on both real GPUs, timings and the debug layer stay on the release machine.
set -euo pipefail

VKP_VER="${VKP_VER:-2.14.1}"
CACHE="${WINE_GPU_CACHE:-/tmp/wine-gpu}"
export WINEPREFIX="${WINEPREFIX:-$CACHE/prefix}"
export DISPLAY="${WINE_GPU_DISPLAY:-:99}"
WINE=/usr/lib/wine/wine64          # Ubuntu's wine64 package installs no /usr/bin/wine64
VKP_DIR="$CACHE/vkd3d-proton-$VKP_VER/x64"
DXC="${DXC:-/opt/dxc/bin/dxc}"
CXX="${CXX:-x86_64-w64-mingw32-g++-posix}"

wine_env() {
	export WINEDLLOVERRIDES="d3d12,d3d12core=n"
	export VKD3D_FEATURE_LEVEL=12_1
	export WINEDEBUG="${WINEDEBUG:--all}"
	export VKD3D_DEBUG="${VKD3D_DEBUG:-err}"
	export VKD3D_SHADER_CACHE_PATH="${VKD3D_SHADER_CACHE_PATH:-$CACHE}"   # keep vkd3d-proton.cache out of the repo
}

start_xvfb() {
	if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && ! pgrep -f "Xvfb $DISPLAY" >/dev/null; then
		(Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp >"$CACHE/xvfb.log" 2>&1 &)
		for _ in $(seq 50); do pgrep -f "Xvfb $DISPLAY" >/dev/null && break; sleep 0.1; done
		sleep 1
	fi
}

cmd_setup() {
	mkdir -p "$CACHE"
	DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		wine64 mesa-vulkan-drivers vulkan-tools xvfb xauth zstd >/dev/null
	vulkaninfo --summary 2>/dev/null | grep -q llvmpipe || { echo "lavapipe (llvmpipe) Vulkan device not visible" >&2; exit 1; }
	if [ ! -f "$VKP_DIR/d3d12.dll" ]; then
		local dl="$CACHE/dl-vkd3d-proton-$VKP_VER"
		mkdir -p "$dl"
		curl -fsSL -o "$dl/vkp.tar.zst" \
			"https://github.com/HansKristian-Work/vkd3d-proton/releases/download/v$VKP_VER/vkd3d-proton-$VKP_VER.tar.zst"
		tar --zstd -xf "$dl/vkp.tar.zst" -C "$CACHE"
	fi
	start_xvfb
	if [ ! -f "$WINEPREFIX/system.reg" ]; then
		WINEDEBUG=-all "$WINE" wineboot -i >/dev/null 2>&1 || true
	fi
	echo "setup ok: wine=$($WINE --version) vkd3d-proton=$VKP_VER prefix=$WINEPREFIX display=$DISPLAY"
}

# build.bat's own superslm_tests source list and defines (cl line after the dxc loop).
SRCS="src/artifact.cpp src/sha256.cpp src/tokenizer.cpp src/model.cpp src/intmath.cpp src/silu_lut.cpp
src/matmul.cpp src/proof_manifest.cpp src/trace_hook.cpp src/forward/checked_chain_funnel.cpp
src/forward/forward_sites.cpp src/forward/parallel_split.cpp src/decode_digest.cpp src/kv_pages.cpp src/spx1_format.cpp
src/sslm_abi.cpp src/damped_greedy_antilm.cpp src/damped_greedy_topk.cpp src/damped_greedy_phaseD.cpp
src/damped_greedy_phaseD_loop.cpp src/gpu/superslm_gpu.cpp src/gpu/gpu_1p0.cpp tests/test_main.cpp
tests/test_slm18x_saturation_census.cpp tests/test_slm19x_schema_damped_greedy.cpp tests/test_tiled_gemm.cpp
tests/test_attn_rowsites.cpp tests/test_decode_threading.cpp tests/test_decode_threading_d1.cpp
src/gpu/gpu_inflight_registry.cpp tests/gpu-g3-core/g3_superslm_tests_adapter.cpp tests/gpu-g3-core/g3_cells_ledger.cpp"
CXXFLAGS="-std=c++20 -O2 -Iinclude -Itests -Isrc/gpu -Itests/gpu-g3-core -DSUPERSLM_LEDGER_YIELD_HOOK -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h
-DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION -DSUPERSLM_ENABLE_MATVEC_TEST_SEAMS -DSUPERSLM_O11_ALLOC_INJECTION
-DSUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION"

cmd_build() {
	local repo; repo="$(cd "$1" && pwd)"
	local obj="$CACHE/obj-$(echo "$repo" | tr / _)"
	mkdir -p "$repo/out/shaders" "$obj"
	cd "$repo"
	# The exe must live under the repo (out/) so ShaderSourceDirOrEmpty finds src/gpu/shaders.
	for f in src/gpu/shaders/*.hlsl; do
		"$DXC" -T cs_6_2 -E main -Fo "out/shaders/$(basename "$f" .hlsl).cso" "$f" -O3 -HV 2018 -WX >/dev/null
	done
	local jobs="${JOBS:-2}"
	# shellcheck disable=SC2086
	printf '%s\n' $SRCS | xargs -P "$jobs" -I{} sh -c \
		'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX"' '"$(echo $CXXFLAGS)"' -c "$2" -o "$o"' _ "$obj" {}
	"$CXX" -O2 "$obj"/*.o -o out/superslm_tests.exe -ld3d12 -ldxgi -ldxguid -lshlwapi \
		-static -static-libgcc -static-libstdc++
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" out/
	echo "built $repo/out/superslm_tests.exe (+ $(ls out/shaders/*.cso | wc -l) shaders, vkd3d-proton DLLs)"
}

cmd_run() {
	local repo; repo="$(cd "$1" && pwd)"; shift
	start_xvfb; wine_env
	cd "$repo"   # fixtures are searched under CWD/tests/fixtures
	exec "$WINE" out/superslm_tests.exe "$@"
}

cmd_exec() {
	local exe="$1"; shift
	start_xvfb; wine_env
	local dir; dir="$(dirname "$exe")"
	[ -f "$dir/d3d12.dll" ] || cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$dir/"
	exec "$WINE" "$exe" "$@"
}

# Sourced by another script (tests/paged-kv-gpu/build_paged_kv_gpu_mingw.sh) for its variables and functions
# (CACHE, VKP_DIR, DXC, CXX, SRCS, wine_env, start_xvfb): stop before dispatching a command.
if [ "${WINE_GPU_SOURCED:-0}" = 1 ]; then return 0 2>/dev/null || exit 0; fi

case "${1:-}" in
	setup) cmd_setup ;;
	build) shift; cmd_build "$@" ;;
	run) shift; cmd_run "$@" ;;
	exec) shift; cmd_exec "$@" ;;
	*) sed -n '2,12p' "$0"; exit 2 ;;
esac
