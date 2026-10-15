#!/usr/bin/env bash
# Paged-KV plan (rev 16.2) step G1: the GPU red suite's Linux build, mingw-w64 against a mingw-built engine,
# for the software GPU (Wine + vkd3d-proton + lavapipe; tools/cloud_gpu/wine_gpu.sh). It builds the SAME
# executables as tests/paged-kv-gpu/build_paged_kv_gpu.bat (the box build), from the same table.
#
# Usage: tests/paged-kv-gpu/build_paged_kv_gpu_mingw.sh [--out=DIR] [--jobs=N] [--state-tracking] [--list] [TARGET ...]
#   TARGET   g2 | g3 | g4 | g5 | g6   every executable whose bracket is that step (default: all five)
#            <exe name>          one executable from the table below (e.g. pkvg_g2, g1d_cell_3_6f)
#   --out    default <repo>/out/pkvg (the exes must sit under the repo: the shader staleness check walks up
#            from the exe to src/gpu/shaders)
#   --jobs   parallel compiles (default 2; the container is shared)
#   --list   print the table and exit
#   --state-tracking  build engine and cells with SUPERSLM_GPU_STATE_TRACKING (src/gpu/gpu_resource_state_tracker.h):
#            the host-side resource-state tracker checks every barrier, copy and dispatch and prints
#            "# GPU STATE VIOLATION" lines and an exit summary. Default --out becomes <repo>/out/pkvg-track and the
#            objects are cached apart, so tracking and plain builds never mix. The software GPU has no D3D12
#            validation; this is its stand-in for the resource-state class.
#
# Each executable is reported LINKED, or RED BY LINK with its undefined symbols (also written to
# <out>/<exe>.undefined), or RED BY COMPILE. RED BY LINK is the expected state of a step's executables until
# that step lands; the script exits 0 unless something failed to COMPILE (an engine or a cell), which is never
# expected.
#
# Engine objects: the source list is wine_gpu.sh's SRCS (build.bat's list) restricted to src/, so it follows
# the engine as G3-G5 add files. The CPU sources compile once; the GPU sources (src/gpu/*.cpp in that list)
# compile once per configuration:
#   plain  no definitions                                         (the step runners, g1d 3.12, g1e, G6 timing)
#   alloc  SUPERSLM_GPU_ALLOC_FAULT_INJECTION SUPERSLM_GPU_TEST_SEAMS  (11.4, 3.6d first touch, g1d 3.6e/f/g stranded)
#   t2169  SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION          (g1d 3.6g recovered)
#   bench  SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING                  (3.6d bench prefill)
# A cell compiles with its executable's configuration definitions, so cells and engine agree.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT=""
JOBS=2
LIST=0
TRACK=0
TARGETS=()
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--list) LIST=1 ;;
		--state-tracking) TRACK=1 ;;
		-*) echo "unknown flag $a" >&2; exit 2 ;;
		*) TARGETS+=("$a") ;;
	esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(g2 g3 g4 g5 g6)
if [ -z "$OUT" ]; then
	if [ "$TRACK" = 1 ]; then OUT="$REPO/out/pkvg-track"; else OUT="$REPO/out/pkvg"; fi
fi

# ---- the suite's executables (keep in step with build_paged_kv_gpu.bat) ------------------------------------
# name | config | step | sources (tests/paged-kv-gpu/*.cpp; NAME@DEF,DEF compiles NAME with -DDEF ...) | extra
#   extra: hooks = link te425_cells.cpp (main renamed) and te425_getdevice.cpp, the te425 InstallHooks family
TABLE='
pkvg_g2|plain|G2|pkvg_main pkvg_group_g1a g1a_g2_legacy_view|
pkvg_g3|plain|G3|pkvg_main pkvg_group_g1a g1a_g2_legacy_view g1a_g3_pool pkvg_group_g1b g1b_g3_cells pkvg_group_g1c@PKVG_STEP=3 g1c_cell_3_6@PKVG_STEP=3 g1c_cell_3_6b@PKVG_STEP=3 g1c_cell_3_6c@PKVG_STEP=3 g1c_cell_3_6d@PKVG_STEP=3 g1c_debug_layer@PKVG_STEP=3|
pkvg_alloc_g3|alloc|G3|pkvg_main pkvg_group_g1b g1b_g3_pool_ceiling|
pkvg_bench_g3|bench|G3|pkvg_main pkvg_group_g1c@PKVG_STEP=3 g1c_cell_3_6d_bench@PKVG_STEP=3 g1c_debug_layer@PKVG_STEP=3|
pkvg_firsttouch_g3|alloc|G3|pkvg_main pkvg_group_g1c@PKVG_STEP=3 g1c_cell_3_6d_first_touch@PKVG_STEP=3 g1c_debug_layer@PKVG_STEP=3|hooks
g1d_cell_3_6e_g3|alloc|G3|g1d_cell_3_6e@G1D_3_6E_G3_ONLY|
g1d_cell_3_6f|alloc|G3|g1d_cell_3_6f|hooks
g1d_cell_3_6g_recovered|t2169|G3|g1d_cell_3_6g_recovered|
g1d_cell_3_6g_stranded|alloc|G3|g1d_cell_3_6g_stranded|hooks
pkvg_g4|plain|G4|pkvg_main pkvg_group_g1a g1a_g2_legacy_view g1a_g3_pool g1a_g4_prefix pkvg_group_g1b g1b_g3_cells g1b_g4_cells pkvg_group_g1c@PKVG_STEP=4 g1c_cell_3_6@PKVG_STEP=4 g1c_cell_3_6b@PKVG_STEP=4 g1c_cell_3_6c@PKVG_STEP=4 g1c_cell_3_6d@PKVG_STEP=4 g1c_debug_layer@PKVG_STEP=4|
g1d_cell_3_6e|alloc|G4|g1d_cell_3_6e|
g1d_cell_3_12_g4|plain|G4|g1d_cell_3_12@G1D_3_12_G4_ONLY|
pkvg_g5|plain|G5|pkvg_main pkvg_group_g1a g1a_g2_legacy_view g1a_g3_pool g1a_g4_prefix g1a_g5_slm6 pkvg_group_g1b g1b_g3_cells g1b_g4_cells g1b_g5_cells pkvg_group_g1c@PKVG_STEP=5 g1c_cell_3_6@PKVG_STEP=5 g1c_cell_3_6b@PKVG_STEP=5 g1c_cell_3_6c@PKVG_STEP=5 g1c_cell_3_6d@PKVG_STEP=5 g1c_debug_layer@PKVG_STEP=5|
g1d_cell_3_12|plain|G5|g1d_cell_3_12|
g1e_cell_9_6|plain|G5|g1e_cell_9_6|
g1e_cell_9_8|plain|G5|g1e_cell_9_8|
g1e_cell_9_12|plain|G5|g1e_cell_9_12|
g1e_cell_9_13|plain|G5|g1e_cell_9_13|
g1e_cell_8_17|plain|G5|g1e_cell_8_17|
g1e_cell_slm6_cap|plain|G5|g1e_cell_slm6_cap|
pkvg_same_handle_race|alloc|G5|pkvg_same_handle_race|
g1e_cell_10_3|plain|G6|g1e_cell_10_3|
g1e_cell_10_5|plain|G6|g1e_cell_10_5|
g1e_cell_10_7|plain|G6|g1e_cell_10_7|
pkvg_g6_decode_timing|plain|G6|pkvg_g6_decode_timing|
'
# Host-only programs (no engine): the suite's process runner.
HOST_TABLE="pkvg_suite"

if [ "$LIST" = 1 ]; then
	echo "$TABLE" | sed '/^$/d' | awk -F'|' '{printf "%-24s %-6s %-3s %s %s\n", $1, $2, $3, $5, $4}'
	echo "host-only: $HOST_TABLE"
	exit 0
fi

config_defs() {
	case "$1" in
		plain) echo "" ;;
		alloc) echo "-DSUPERSLM_GPU_ALLOC_FAULT_INJECTION -DSUPERSLM_GPU_TEST_SEAMS" ;;
		t2169) echo "-DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION" ;;
		bench) echo "-DSUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING" ;;
		*) echo "unknown configuration $1" >&2; exit 2 ;;
	esac
}

BASEFLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/paged-kv -Itests/paged-kv-gpu
-Itests/t2791-gpu-prefill-read-red-suite -Itests/te425-gpu-alloc-status-red-suite -DNOMINMAX -DWIN32_LEAN_AND_MEAN
-include d3d12.h -include d3d12sdklayers.h"
CELLWARN="-Wall -Wextra -Wno-unused-parameter"
OBJ="$CACHE/pkvg-obj-$(echo "$REPO" | tr / _)"
if [ "$TRACK" = 1 ]; then
	BASEFLAGS="$BASEFLAGS -DSUPERSLM_GPU_STATE_TRACKING"
	OBJ="$OBJ-track"
fi
mkdir -p "$OBJ" "$OUT/shaders"
cd "$REPO"

# compile OUTDIR "FLAGS" SRC... : one object per source, rebuilt when any dependency (-MMD) is newer.
compile() {
	local outdir="$1" flags="$2"; shift 2
	mkdir -p "$outdir"
	# shellcheck disable=SC2016
	printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c '
		src="$1"; o="$2/$(basename "$src" .cpp).o"
		if [ -f "$o" ] && [ -f "$o.d" ]; then
			stale=0
			for d in $(sed -e "s/^[^:]*://" -e "s/\\\\$//" "$o.d"); do [ "$d" -nt "$o" ] && { stale=1; break; }; done
			[ $stale = 0 ] && exit 0
		fi
		if ! '"$CXX"' $3 -MMD -MF "$o.d" -c "$src" -o "$o" 2>"$o.log"; then
			echo "RED BY COMPILE: $src ($2)"; sed -n "1,40p" "$o.log"; exit 1
		fi
		w="$(grep -E "warning|error" "$o.log" | grep -v "WIN32_LEAN_AND_MEAN\|NOMINMAX" | head -20 || true)"
		[ -n "$w" ] && printf "warnings: %s\n%s\n" "$src" "$w"
		exit 0' _ {} "$outdir" "$flags"
}

ENGINE_CPU=() ENGINE_GPU=()
# shellcheck disable=SC2086
for s in $SRCS; do
	case "$s" in
		src/gpu/*) ENGINE_GPU+=("$s") ;;
		src/*) ENGINE_CPU+=("$s") ;;
	esac
done

engine_cpu_done=0
build_engine() {  # build_engine CONFIG
	local cfg="$1" flags
	if [ $engine_cpu_done = 0 ]; then
		echo "== engine: ${#ENGINE_CPU[@]} CPU sources"
		compile "$OBJ/engine-cpu" "$(echo $BASEFLAGS)" "${ENGINE_CPU[@]}"
		engine_cpu_done=1
	fi
	[ -f "$OBJ/engine-gpu-$cfg/.done-$$" ] && return 0
	flags="$(echo $BASEFLAGS) $(config_defs "$cfg")"
	echo "== engine: ${#ENGINE_GPU[@]} GPU sources, configuration $cfg"
	compile "$OBJ/engine-gpu-$cfg" "$flags" "${ENGINE_GPU[@]}"
	touch "$OBJ/engine-gpu-$cfg/.done-$$"
}

hooks_objs() {  # te425's InstallHooks family, compiled in the alloc configuration (its hooks need the seams)
	local d="$OBJ/te425-alloc" flags
	flags="$(echo $BASEFLAGS) $(config_defs alloc) -Dmain=te425_cells_main"
	compile "$d" "$flags" tests/te425-gpu-alloc-status-red-suite/te425_cells.cpp \
		tests/te425-gpu-alloc-status-red-suite/te425_getdevice.cpp
	echo "$d/te425_cells.o $d/te425_getdevice.o"
}

shaders() {
	local n=0
	for f in src/gpu/shaders/*.hlsl; do
		local o="$OUT/shaders/$(basename "$f" .hlsl).cso"
		if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ -n "$(find src/gpu/shaders -name '*.hlsli' -newer "$o" 2>/dev/null)" ]; then
			"$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX >/dev/null
		fi
		n=$((n + 1))
	done
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$OUT/"
	echo "== shaders: $n under $OUT/shaders (+ vkd3d-proton DLLs)"
}

LINKED=() REDLINK=() REDCOMPILE=()
build_exe() {  # build_exe "name|config|step|sources|extra"
	local name cfg step srcs extra
	IFS='|' read -r name cfg step srcs extra <<<"$1"
	build_engine "$cfg"
	local objs=() entry src defs tag flags
	for entry in $srcs; do
		src="${entry%%@*}"; defs=""; tag="none"
		if [ "$entry" != "$src" ]; then
			defs="$(echo "${entry#*@}" | tr ',' '\n' | sed 's/^/-D/' | tr '\n' ' ')"
			tag="$(echo "${entry#*@}" | tr ',=' '__')"
		fi
		flags="$(echo $BASEFLAGS) $(config_defs "$cfg") $defs $CELLWARN"
		if ! compile "$OBJ/cells-$cfg-$tag" "$flags" "tests/paged-kv-gpu/$src.cpp"; then
			REDCOMPILE+=("$name"); return 0
		fi
		objs+=("$OBJ/cells-$cfg-$tag/$src.o")
	done
	local hook=""
	[ "$extra" = hooks ] && hook="$(hooks_objs | tail -1)"
	local log="$OUT/$name.link.log"
	# shellcheck disable=SC2086
	if "$CXX" -O2 "${objs[@]}" $hook "$OBJ"/engine-cpu/*.o "$OBJ/engine-gpu-$cfg"/*.o -o "$OUT/$name.exe" \
		-ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc -static-libstdc++ >"$log" 2>&1; then
		rm -f "$OUT/$name.undefined"
		echo "[$step] $name: LINKED ($cfg)"
		LINKED+=("$name")
	else
		grep -o "undefined reference to \`[^']*'" "$log" | sed "s/undefined reference to \`//; s/'\$//" | sort -u >"$OUT/$name.undefined" || true
		if [ -s "$OUT/$name.undefined" ]; then
			echo "[$step] $name: RED BY LINK ($cfg) -- undefined:"
			sed 's/^/    /' "$OUT/$name.undefined"
			REDLINK+=("$name")
		else
			echo "[$step] $name: UNEXPECTED LINK FAILURE -- see $log"; sed -n '1,30p' "$log"
			REDCOMPILE+=("$name")
		fi
		rm -f "$OUT/$name.exe"
	fi
}

build_host() {  # programs with no engine
	local name="$1"
	"$CXX" -std=c++20 -O2 -Wall -Wextra "tests/paged-kv-gpu/$name.cpp" -o "$OUT/$name.exe" -static -static-libgcc -static-libstdc++ \
		&& echo "[host] $name: LINKED" || { echo "[host] $name: RED BY COMPILE"; REDCOMPILE+=("$name"); }
}

shaders
for h in $HOST_TABLE; do build_host "$h"; done
while IFS= read -r row; do
	[ -z "$row" ] && continue
	IFS='|' read -r name _ step _ _ <<<"$row"
	want=0
	for t in "${TARGETS[@]}"; do
		[ "$t" = "$name" ] && want=1
		[ "$(echo "$t" | tr a-z A-Z)" = "$step" ] && want=1
	done
	[ $want = 1 ] && build_exe "$row"
done <<<"$TABLE"
rm -f "$OBJ"/engine-gpu-*/.done-$$

echo "== ${#LINKED[@]} linked, ${#REDLINK[@]} red by link, ${#REDCOMPILE[@]} red by compile"
[ ${#REDCOMPILE[@]} -eq 0 ] || { echo "RED BY COMPILE: ${REDCOMPILE[*]}"; exit 1; }
exit 0
