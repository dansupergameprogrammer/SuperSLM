#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), MR1 gate: "Wine one-layer smoke equal on pkv_def and pkv_qk", software GPU
# (Wine + vkd3d-proton + lavapipe, tools/cloud_gpu/wine_gpu.sh).
#
# Usage: tests/multirow-gpu/mr1_wine_smoke.sh [--out=DIR] [--jobs=N] [--build-only]
#   --out  default <repo>/out/mr1-smoke (under the repo: the engine's shader staleness check walks up to
#          src/gpu/shaders). Each run uses its own WINEPREFIX (<out>/wineprefix-<pid>) unless WINEPREFIX is set.
#
# Legs:
#   twin   mr1_twin_capture.exe on pkv_def.sslm and pkv_qk.sslm at p0 = 0 and p0 = 3: the one-row reference,
#          one SLM5 blob per (row, layer), deterministic across two fresh contexts. Green before the port.
#   wide   mr1_wide_smoke.exe (engine and cell built with -DSUPERSLM_MULTIROW_WIDE) on pkv_def.sslm and
#          pkv_qk.sslm: ONE widened layer (rulings-mr1.md ruling 1) through the builder's test-only entry
#          sslm_gpu_test_only_mr1_wide_one_layer_smoke, at H/M in {2/1, 2/2, 4/3, 4/4, 8/5, 8/8} x p0 in {0, 3, 7},
#          graded by this suite (see mr1_wide_smoke.cpp), tied to the twin capture, plus one must-reject.
# Exit 0 only when every leg passes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
OUT="$REPO/out/mr1-smoke"
JOBS=2
RUN=1
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--build-only) RUN=0 ;;
		*) sed -n '2,18p' "$0"; exit 2 ;;
	esac
done
export WINEPREFIX="${WINEPREFIX_MR1:-$OUT/wineprefix-$$}"
cd "$REPO"
mkdir -p "$OUT/shaders" "$OUT/obj"

for f in src/gpu/shaders/*.hlsl; do
	o="$OUT/shaders/$(basename "$f" .hlsl).cso"
	if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ -n "$(find src/gpu/shaders -name '*.hlsli' -newer "$o")" ]; then
		"$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX >/dev/null
	fi
done
cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$OUT/"

# The widened variants, named as build.bat compiles them (<site>_wide_h<H>, <g>_gemm_wide_h<H>).
for h in 2 4 8; do
	for f in src/gpu/shaders_wide/*_site_wide.hlsl; do
		o="$OUT/shaders/$(basename "$f" .hlsl)_h$h.cso"
		[ -f "$o" ] && [ "$o" -nt "$f" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX -D SSLM_WIDE_H=$h -I src/gpu/shaders >/dev/null
	done
	gi=1
	for g in q_proj kv_proj o_proj gate_proj up_proj down_proj; do
		o="$OUT/shaders/${g}_gemm_wide_h$h.cso"
		[ -f "$o" ] && [ "$o" -nt src/gpu/shaders_wide/proj_gemm_wide.hlsl ] || "$DXC" -T cs_6_2 -E main -Fo "$o" \
			src/gpu/shaders_wide/proj_gemm_wide.hlsl -O3 -HV 2018 -WX -D SSLM_WIDE_H=$h -D SSLM_GEMM_SITE=$gi -I src/gpu/shaders >/dev/null
		gi=$((gi + 1))
	done
done

FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
ENGINE=()
# shellcheck disable=SC2086
for s in $SRCS; do case "$s" in src/*) ENGINE+=("$s") ;; esac; done
build() {  # build OBJDIR "EXTRA FLAGS" EXE CELL
	mkdir -p "$1"
	printf '%s\n' "${ENGINE[@]}" "$4" | xargs -P "$JOBS" -I{} bash -c \
		'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX $FLAGS $2"' -w -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$1" {}
	"$CXX" -O2 "$1"/*.o -o "$OUT/$3" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc -static-libstdc++
}
build "$OUT/obj" "" mr1_twin_capture.exe tests/multirow-gpu/mr1_twin_capture.cpp
build "$OUT/obj-wide" "-DSUPERSLM_MULTIROW_WIDE" mr1_wide_smoke.exe tests/multirow-gpu/mr1_wide_smoke.cpp
echo "built $OUT/mr1_twin_capture.exe $OUT/mr1_wide_smoke.exe"
[ "$RUN" = 1 ] || exit 0

FIX="$OUT/fixtures"; mkdir -p "$FIX"
[ -f "$FIX/pkv_def.sslm" ] && [ -f "$FIX/pkv_qk.sslm" ] || python3 tools/gen_paged_kv_fixture.py "$FIX" >/dev/null
start_xvfb; wine_env
FAILED=()
for m in pkv_def pkv_qk; do
	for s in 0 3; do
		set +e
		"$WINE" "$OUT/mr1_twin_capture.exe" --model="$FIX/$m.sslm" --rows=8 --start=$s --out="$OUT/twin-$m-s$s" \
			>"$OUT/twin-$m-s$s.log" 2>&1
		rc=$?
		set -e
		tail -2 "$OUT/twin-$m-s$s.log" | sed 's/^/    /'
		if [ $rc = 0 ]; then echo "PASS twin $m p0=$s"; else echo "FAIL twin $m p0=$s (exit $rc)"; FAILED+=("twin-$m-s$s"); fi
	done
done
for m in pkv_def pkv_qk; do
	set +e
	"$WINE" "$OUT/mr1_wide_smoke.exe" --model="$FIX/$m.sslm" >"$OUT/wide-$m.log" 2>&1
	rc=$?
	set -e
	grep -E "^FAIL|smoke:" "$OUT/wide-$m.log" | head -12 | sed 's/^/    /'
	if [ $rc = 0 ]; then echo "PASS wide $m"; else echo "FAIL wide $m (exit $rc, log $OUT/wide-$m.log)"; FAILED+=("wide-$m"); fi
done
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR1 Wine smoke: PASS"; exit 0; fi
echo "MR1 Wine smoke: FAIL (${FAILED[*]})"
exit 1
