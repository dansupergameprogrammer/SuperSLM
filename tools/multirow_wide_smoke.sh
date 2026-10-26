#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2): the builder's Wine smokes on the software GPU (Wine + vkd3d-proton + lavapipe,
# tools/cloud_gpu/wine_gpu.sh; run its `setup` once first). MR1 gate: "Wine one-layer smoke equal on pkv_def and
# pkv_qk" (multirow_wide_smoke.cpp). MR2 and MR3: the recorder through the published verbs against the one-row
# twin, the batch with open chunks and the provisional statuses (multirow_rows_smoke.cpp; a development aid, the
# test author's suite grades the cells); in --suite=mr2, MRX's refusals of the removed schema-forced kind, its flag and Canonical (the jump) on F-SFD
# (tools/gen_schema_fast_decode_fixture.py). MR4: --suite=bench runs cell 10.2's box bench (multirow_cost_bench.cpp) in a
# reduced form, to show it closes every M and writes a table header that compiles and mr4_box.py's CSV; the numbers are lavapipe's.
#
# Usage: tools/multirow_wide_smoke.sh [--out=DIR] [--jobs=N] [--build-only] [--quick] [--suite=mr1|mr2|bench|all]
#   --out    default <repo>/out/mr1-wide-smoke. Each run gets its own WINEPREFIX (<out>/wineprefix-<pid>)
#            unless WINEPREFIX_MR1 names one.
#   --quick  the tools' reduced grids
#   --suite  which smoke's legs run (default mr1)
#
# Builds the engine with -DSUPERSLM_MULTIROW_WIDE (the multi-row test seams and MR1's smoke one-shot) and
# tools/multirow_wide_smoke.cpp, compiles the one-row and widened shaders as build.bat does, writes the
# fixtures (tools/gen_paged_kv_fixture.py; tools/gen_multirow_adapter_fixture.py for pkv_def_r4.adp) and runs
# three legs: pkv_def, pkv_def with pkv_def_r4.adp, and pkv_qk. Exit 0 only when every leg passes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# shellcheck source=cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
OUT="$REPO/out/mr1-wide-smoke"
JOBS=2
RUN=1
QUICK=()
SUITE=mr1
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--build-only) RUN=0 ;;
		--quick) QUICK=(--quick) ;;
		--suite=*) SUITE="${a#--suite=}" ;;
		*) sed -n '2,19p' "$0"; exit 2 ;;
	esac
done
mkdir -p "$OUT/shaders" "$OUT/obj"
OUT="$(cd "$OUT" && pwd)"
export WINEPREFIX="${WINEPREFIX_MR1:-$OUT/wineprefix-$$}"
cd "$REPO"

for f in src/gpu/shaders/*.hlsl; do
	"$DXC" -T cs_6_2 -E main -Fo "$OUT/shaders/$(basename "$f" .hlsl).cso" "$f" -O3 -HV 2018 -WX >/dev/null
done
for h in 2 4 8; do
	for f in src/gpu/shaders_wide/*_site_wide.hlsl; do
		"$DXC" -T cs_6_2 -E main -Fo "$OUT/shaders/$(basename "$f" .hlsl)_h$h.cso" "$f" -O3 -HV 2018 -WX \
			-D SSLM_WIDE_H=$h -I src/gpu/shaders >/dev/null
	done
	gi=1
	for g in q_proj kv_proj o_proj gate_proj up_proj down_proj; do
		"$DXC" -T cs_6_2 -E main -Fo "$OUT/shaders/${g}_gemm_wide_h$h.cso" src/gpu/shaders_wide/proj_gemm_wide.hlsl \
			-O3 -HV 2018 -WX -D SSLM_WIDE_H=$h -D SSLM_GEMM_SITE=$gi -I src/gpu/shaders >/dev/null
		gi=$((gi + 1))
	done
done
cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$OUT/"

# mingw-w64 before 12 lacks one D3D12 enumerator the harness names; supply the Windows SDK's value.
COMPAT="$OUT/mingw_d3d12_compat.h"
cat >"$COMPAT" <<'EOF'
#pragma once
#if defined(__MINGW64_VERSION_MAJOR) && __MINGW64_VERSION_MAJOR < 12
static constexpr D3D12_RESOURCE_STATES D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE =
    static_cast<D3D12_RESOURCE_STATES>(0x400000);
#endif
EOF
FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h -include $COMPAT -DSUPERSLM_MULTIROW_WIDE"
ENGINE=()
# shellcheck disable=SC2086
for s in $SRCS; do case "$s" in src/*) ENGINE+=("$s") ;; esac; done
mkdir -p "$OUT/obj/tools"
printf '%s\n' "${ENGINE[@]}" tools/multirow_wide_smoke.cpp tools/multirow_rows_smoke.cpp tools/multirow_cost_bench.cpp | xargs -P "$JOBS" -I{} bash -c \
	'd="$1"; case "$2" in tools/*) d="$1/tools" ;; esac; o="$d/$(echo "$2" | tr / _).o"; '"$CXX $FLAGS"' -w -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$OUT/obj" {}
for t in multirow_wide_smoke multirow_rows_smoke multirow_cost_bench; do
	"$CXX" -O2 "$OUT/obj"/*.o "$OUT/obj/tools/tools_$t.cpp.o" -o "$OUT/$t.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi \
		-static -static-libgcc -static-libstdc++
done
echo "built $OUT/multirow_wide_smoke.exe, multirow_rows_smoke.exe and multirow_cost_bench.exe (+ $(ls "$OUT"/shaders/*.cso | wc -l) shaders)"
[ "$RUN" = 1 ] || exit 0

FIX="$OUT/fixtures"
mkdir -p "$FIX"
# The fixture writer is slow (it writes every variant); reuse MR_FIXTURES' or an earlier run's when present.
for src in "${MR_FIXTURES:-}" "$REPO/out/mr-p41/fixtures" "$REPO/out/mr2/fixtures"; do
	[ -n "$src" ] && [ -f "$src/pkv_def.sslm" ] && [ -f "$src/pkv_qk.sslm" ] && [ ! -f "$FIX/pkv_qk.sslm" ] && cp "$src"/pkv_def.sslm "$src"/pkv_qk.sslm "$FIX/"
done
[ -f "$FIX/pkv_def.sslm" ] && [ -f "$FIX/pkv_qk.sslm" ] || python3 tools/gen_paged_kv_fixture.py "$FIX" >/dev/null
python3 tools/gen_multirow_adapter_fixture.py "$FIX/pkv_def.sslm" "$FIX/pkv_def_r4.adp" >/dev/null
start_xvfb
wine_env
FAILED=()
leg() {
	local exe="$1" name="$2"; shift 2
	set +e
	"$WINE" "$OUT/$exe.exe" "${QUICK[@]}" "$@" >"$OUT/$name.log" 2>&1
	local rc=$?
	set -e
	{ grep -E '^(PASS|FAIL)' "$OUT/$name.log" || true; } | sed 's/^/    /'  # a log with no verdict lines is not an exit
	tail -1 "$OUT/$name.log"
	[ $rc = 0 ] || FAILED+=("$name")
}
for s in mr1 mr2; do
	case "$SUITE" in "$s"|all) ;; *) continue ;; esac
	exe=multirow_wide_smoke
	[ "$s" = mr2 ] && exe=multirow_rows_smoke
	leg "$exe" "$s-pkv_def" --model="$FIX/pkv_def.sslm"
	leg "$exe" "$s-pkv_def_r4" --model="$FIX/pkv_def.sslm" --adapter="$FIX/pkv_def_r4.adp"
	leg "$exe" "$s-pkv_qk" --model="$FIX/pkv_qk.sslm"
	if [ "$s" = mr2 ]; then
		# The close's refusal path: RoPE tables truncated to 12 positions, so every row at 12 or later refuses.
		leg "$exe" mr2-refusal-pkv_def --model="$FIX/pkv_def.sslm" --refusal-only --rope-positions=12
		leg "$exe" mr2-refusal-pkv_qk --model="$FIX/pkv_qk.sslm" --refusal-only --rope-positions=12
		# MRX: the removed schema-forced kind and flag, and the jump's modes, refused on F-SFD (a forced-next table).
		for src in "${MR_FIXTURES:-}" "$REPO/out/mr-p41/fixtures"; do
			[ -n "$src" ] && [ -f "$src/sfd/fsfd.sslm" ] && [ ! -f "$FIX/sfd/fsfd.sslm" ] && mkdir -p "$FIX/sfd" &&
				cp "$src/sfd/fsfd.sslm" "$src/sfd/fsfd_ref.txt" "$FIX/sfd/"
		done
		[ -f "$FIX/sfd/fsfd.sslm" ] || python3 tools/gen_schema_fast_decode_fixture.py "$FIX/sfd" >/dev/null
		leg "$exe" mrx-forced-fsfd --model="$FIX/sfd/fsfd.sslm" --forced-schema=potion_shop_order
	fi
done
case "$SUITE" in
	bench|all)
		leg multirow_cost_bench bench-pkv_def --model="$FIX/pkv_def.sslm" --runs=3 --p0=3 --out="$OUT/gpu_rows_cost_table.h" \
			--json="$OUT/multirow_cost_bench.json" --csv="$OUT/multirow_cost_bench.csv" --model-name=pkv_def
		# The written header is the shipped one's layout: the suite's table cell compiles against either.
		for h in "$OUT" include; do
			rm -rf "$OUT/tablecheck"; mkdir -p "$OUT/tablecheck/superslm"
			cp "$h/$( [ "$h" = include ] && echo superslm/ )gpu_rows_cost_table.h" "$OUT/tablecheck/superslm/gpu_rows_cost_table.h" 2>/dev/null || true
			if printf '#include "superslm/gpu_rows_cost_table.h"\nint main(){return superslm_gpu::kGpuRowsCostRatioPerMille[0] == 1000 && superslm_gpu::kGpuRowCostUs == 6200 ? 0 : 1;}\n' |
				g++ -x c++ -std=c++17 -Wall -Werror -I"$OUT/tablecheck" - -o "$OUT/tablecheck/t" && "$OUT/tablecheck/t"; then
				echo "    PASS $h table header compiles, r(1) = 1000"
			else
				echo "    FAIL $h table header"; FAILED+=(bench-table-$h)
			fi
		done
		[ "$(head -1 "$OUT/multirow_cost_bench.csv")" = "model,M,logits,runs,warmup,median_us,p95_us,max_us,clock,wall_median_us,wall_p95_us,wall_max_us,logits_stage_gpu_us,host_finish_us" ] &&
			[ "$(grep -c ',0,[0-9]*,[0-9]*,[0-9]*,[0-9]*,[0-9]*,gpu,[0-9]*,[0-9]*,[0-9]*,,$' "$OUT/multirow_cost_bench.csv")" = 8 ] &&
			[ "$(grep -c ',1,[0-9]*,[0-9]*,[0-9]*,[0-9]*,[0-9]*,gpu,[0-9]*,[0-9]*,[0-9]*,[1-9][0-9]*,[1-9][0-9]*$' "$OUT/multirow_cost_bench.csv")" = 8 ] &&
			[ "$(wc -l <"$OUT/multirow_cost_bench.csv")" = 17 ] && echo "    PASS CSV: mr4_box.py's columns plus clock and wall, 16 rows on the GPU clock, logits 1 with positive logits_stage_gpu_us and host_finish_us" ||
			{ echo "    FAIL CSV"; FAILED+=(bench-csv); }
		# MB5's grader on this table (ruling 33 S7): g(M) from logits_stage_gpu_us, host_finish_us reported.
		if python3 tests/multirow-gpu/mr4_box.py report-exact "$OUT/multirow_cost_bench.csv" pkv_def; then echo "    PASS mr4_box report-exact"
		else echo "    FAIL mr4_box report-exact"; FAILED+=(bench-box); fi ;;
esac
if [ ${#FAILED[@]} = 0 ]; then echo "Multi-row Wine smoke ($SUITE): PASS"; else echo "Multi-row Wine smoke ($SUITE): FAIL (${FAILED[*]})"; exit 1; fi
