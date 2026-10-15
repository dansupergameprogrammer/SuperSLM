#!/usr/bin/env bash
# The t2112 suite's cloud build: mingw-w64 builds of chosen cells against the engine, for the software GPU
# (Wine + vkd3d-proton + lavapipe; tools/cloud_gpu/wine_gpu.sh). build_link_red.bat's engine list and defines.
#
# Usage: tests/t2112-gpu-1p0-red-suite/build_cell_mingw.sh [--out=DIR] CELL.cpp ...
#   --out  default <repo>/out/t2112 (under the repo, so the shader staleness check finds src/gpu/shaders)
# Run:   tools/cloud_gpu/wine_gpu.sh exec <repo>/out/t2112/dim3_concurrency_red.exe --model1p5b=<fdef.sslm>
#        (fdef.sslm: python3 tools/gen_decode_threading_fixture.py <dir>)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
OUT="$REPO/out/t2112"
CELLS=()
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		*) CELLS+=("$(cd "$(dirname "$a")" && pwd)/$(basename "$a")") ;;
	esac
done
[ ${#CELLS[@]} -gt 0 ] || { sed -n '2,8p' "$0"; exit 2; }
FLAGS="-std=c++20 -O2 -I$REPO/include -I$REPO/tests -I$REPO/src/gpu -I$HERE -DNOMINMAX -DWIN32_LEAN_AND_MEAN
-include d3d12.h -include d3d12sdklayers.h -DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION -DSUPERSLM_O11_ALLOC_INJECTION"
ENGINE="src/artifact.cpp src/sha256.cpp src/tokenizer.cpp src/model.cpp src/intmath.cpp src/silu_lut.cpp src/matmul.cpp
src/proof_manifest.cpp src/trace_hook.cpp src/forward/checked_chain_funnel.cpp src/forward/forward_sites.cpp
src/forward/parallel_split.cpp src/gpu/gpu_inflight_registry.cpp src/decode_digest.cpp src/kv_pages.cpp
src/gpu/superslm_gpu.cpp src/gpu/gpu_1p0.cpp"
OBJ="$OUT/obj"
mkdir -p "$OBJ" "$OUT/shaders"
cd "$REPO"
for f in src/gpu/shaders/*.hlsl; do
	o="$OUT/shaders/$(basename "$f" .hlsl).cso"
	[ "$o" -nt "$f" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX >/dev/null
done
# shellcheck disable=SC2086
printf '%s\n' $ENGINE | xargs -P "${JOBS:-2}" -I{} sh -c \
	'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX $(echo $FLAGS)"' -c "$2" -o "$o"' _ "$OBJ" {}
for c in "${CELLS[@]}"; do
	n="$(basename "$c" .cpp)"
	# shellcheck disable=SC2086
	"$CXX" $FLAGS -c "$c" -o "$OBJ/cell_$n.obj"
	"$CXX" -O2 "$OBJ"/*.o "$OBJ/cell_$n.obj" -o "$OUT/$n.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi \
		-static -static-libgcc -static-libstdc++
	echo "built $OUT/$n.exe"
done
cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$OUT/"
