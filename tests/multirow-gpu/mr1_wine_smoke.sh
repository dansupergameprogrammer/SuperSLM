#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), MR1 gate: "Wine one-layer smoke equal on pkv_def and pkv_qk", software GPU
# (Wine + vkd3d-proton + lavapipe, tools/cloud_gpu/wine_gpu.sh).
#
# Usage: tests/multirow-gpu/mr1_wine_smoke.sh [--out=DIR] [--jobs=N] [--build-only] [--with-mutants]
#   --out  default <repo>/out/mr1-smoke (under the repo: the engine's shader staleness check walks up to
#          src/gpu/shaders). Each run uses its own WINEPREFIX (<out>/wineprefix-<pid>) unless WINEPREFIX is set.
#
# Legs:
#   twin   mr1_twin_capture.exe on pkv_def.sslm and pkv_qk.sslm at p0 = 0 and p0 = 3: the one-row reference,
#          one SLM5 blob per (row, layer), deterministic across two fresh contexts. Green before the port.
#   wide   mr1_wide_smoke.exe (engine and cell built with -DSUPERSLM_MULTIROW_WIDE) on pkv_def.sslm and
#          pkv_qk.sslm: ONE widened layer (rulings-mr1.md ruling 1) through the builder's test-only entry
#          sslm_gpu_test_only_mr1_wide_one_layer_smoke, at H/M in {2/1, 2/2, 4/3, 4/4, 8/5, 8/8} x p0 in {0, 3, 7},
#          graded by this suite (see mr1_wide_smoke.cpp), tied to the twin capture, plus one must-reject. Every
#          case runs on a zero and on a canary K/V background, plus MX1-host cases at p0 = 256 / NQH - 2.
#   mutants (opt-in, --with-mutants; rebuilds the engine once): the code reviewer's planted mutants must turn the
#          wide leg red, each through the check it targets, on both fixtures:
#            S6  kv_proj_site_wide.hlsl without its sticky return (padded rows reach the K/V store): red only in
#                canary cases, by stray bytes (the zero background cannot see it);
#            H2  multirow_wide.inl sizes attention groups for row 0 (`max_width = width0`): red in the MX1-host
#                cases.
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
MUTANTS=0
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--build-only) RUN=0 ;;
		--with-mutants) MUTANTS=1 ;;
		*) sed -n '2,26p' "$0"; exit 2 ;;
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

# must_reject NAME EXE-DIR PATTERN: the wide smoke from EXE-DIR must fail on both fixtures, with at least one FAIL
# line matching PATTERN (the check the mutant targets) and none outside it.
must_reject() {
	local name="$1" dir="$2" pat="$3" m rc
	for m in pkv_def pkv_qk; do
		set +e
		"$WINE" "$dir/mr1_wide_smoke.exe" --model="$FIX/$m.sslm" >"$dir/wide-$m.log" 2>&1
		rc=$?
		set -e
		local hit miss
		hit=$(grep -E '^FAIL' "$dir/wide-$m.log" | grep -cE "$pat" || true)
		miss=$(grep -E '^FAIL' "$dir/wide-$m.log" | grep -vE "$pat" | grep -vE '^FAIL widened one-layer smoke' | head -3 || true)
		grep -E "smoke:" "$dir/wide-$m.log" | sed 's/^/    /'
		if [ $rc != 0 ] && [ "$hit" -gt 0 ] && [ -z "$miss" ]; then
			echo "PASS must-reject $name $m ($hit failing cases, all through /$pat/)"
		else
			echo "FAIL must-reject $name $m (exit $rc, $hit matching; other: ${miss:-none}; log $dir/wide-$m.log)"
			FAILED+=("mutant-$name-$m")
		fi
	done
}
if [ "$MUTANTS" = 1 ]; then
	# S6: shader-only, so the built exe runs against a shader directory with the three mutated variants.
	S6="$OUT/mutant-s6"; rm -rf "$S6"; mkdir -p "$S6/src"
	cp "$OUT/mr1_wide_smoke.exe" "$OUT/d3d12.dll" "$OUT/d3d12core.dll" "$S6/"
	cp -r "$OUT/shaders" "$S6/shaders"
	sed 's/^\(\s*\)if (sticky != kTagOk) return;/\1\/\/ S6: sticky return removed/' \
		src/gpu/shaders_wide/kv_proj_site_wide.hlsl >"$S6/src/kv_proj_site_wide.hlsl"
	cmp -s src/gpu/shaders_wide/kv_proj_site_wide.hlsl "$S6/src/kv_proj_site_wide.hlsl" && { echo "FAIL mutant S6 changed nothing"; exit 1; }
	for h in 2 4 8; do
		"$DXC" -T cs_6_2 -E main -Fo "$S6/shaders/kv_proj_site_wide_h$h.cso" "$S6/src/kv_proj_site_wide.hlsl" -O3 -HV 2018 -WX \
			-D SSLM_WIDE_H=$h -I src/gpu/shaders >/dev/null
	done
	must_reject S6 "$S6" 'canary: E4 K/V differing 0 stray [1-9]'
	# H2: host code, so a copy of the tree (mtimes kept: the engine's shader staleness check reads them) is
	# mutated and the engine rebuilt there; the shaders are the unmutated ones.
	H2="$OUT/mutant-h2"; rm -rf "$H2"; mkdir -p "$H2/tree/tests"
	cp -rp src include "$H2/tree/"; cp -rp tests/multirow-gpu "$H2/tree/tests/"
	for hdr in tests/*.h; do [ -e "$hdr" ] && cp -p "$hdr" "$H2/tree/tests/"; done
	sed -i 's/const uint32_t max_width = width0 + (variant_h - 1u);/const uint32_t max_width = width0;  \/\/ H2/' \
		"$H2/tree/src/gpu/multirow_wide.inl"
	grep -q 'max_width = width0;  // H2' "$H2/tree/src/gpu/multirow_wide.inl" || { echo "FAIL mutant H2 changed nothing"; exit 1; }
	( cd "$H2/tree" && OUT="$H2" && build "$H2/obj-wide" "-DSUPERSLM_MULTIROW_WIDE" mr1_wide_smoke.exe tests/multirow-gpu/mr1_wide_smoke.cpp )
	cp "$OUT/d3d12.dll" "$OUT/d3d12core.dll" "$H2/"; cp -r "$OUT/shaders" "$H2/shaders"
	must_reject H2 "$H2" 'p0=[0-9]{2,} '
fi
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR1 Wine smoke: PASS"; exit 0; fi
echo "MR1 Wine smoke: FAIL (${FAILED[*]})"
exit 1
