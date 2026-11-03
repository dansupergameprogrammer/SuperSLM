#!/usr/bin/env bash
# Schema fast-decode plan rev 1.4.4, slices G1-F, G1-M and G1-J: the GPU cells' cloud build, mingw-w64 against a
# mingw-built engine. It builds the SAME executables as build_sfd_gpu.bat (the box build), from the same table.
# The cloud has no D3D12 device: an executable here is graded LINKED (it compiles and every verb and seam it
# calls is defined), RED BY LINK (with its undefined symbols, written to <out>/<exe>.undefined) or RED BY
# COMPILE. Running the cells is box-only (§10 GB-F, GB-M, GB-J and GB1).
#
# Usage: tests/schema-fast-decode-gpu/build_sfd_gpu_mingw.sh [--out=DIR] [--jobs=N] [--list] [TARGET ...]
#   TARGET   g1f | g1m | g1j    every executable of that slice (default: all three)
#            <exe name>         one executable from the table below
#   --out    default <repo>/out/sfdg
#   --jobs   parallel compiles (default 2; the container is shared)
#
# Engine: wine_gpu.sh's SRCS (build.bat's list) restricted to src/, compiled in two configurations:
#   plain  no definitions: the production library. Compiled only, to show the seams compile out of it.
#   seams  SUPERSLM_GPU_TEST_SEAMS SUPERSLM_GPU_ALLOC_FAULT_INJECTION SUPERSLM_ENABLE_BAD_ALLOC_INJECTION
#          SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION (G8.1): every
#          cell links against this one (S-GSUB, S-GROWS, S-GBOUND, S-GCHUNKFAULT, the table-build bad-alloc).
# Shaders are not built here (dxc is optional in the cloud; no cell runs here). Exit 0 unless something
# failed to COMPILE.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT="$REPO/out/sfdg"
JOBS=2
LIST=0
TARGETS=()
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--list) LIST=1 ;;
		-*) echo "unknown flag $a" >&2; exit 2 ;;
		*) TARGETS+=("$a") ;;
	esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(g1f g1m g1j)

# ---- the executables (keep in step with build_sfd_gpu.bat) -------------------------------------------------
# name | slice | sources (tests/schema-fast-decode-gpu/*.cpp; a source with a slash is repo-relative)
TABLE='
sfdg_smoke|G1F|sfdg_smoke
sfdg_cell_g0_1|G1F|sfdg_cell_g0_1
sfdg_cell_g2_1|G1F|sfdg_cell_g2_1
sfdg_cell_g12_4|G1F|sfdg_cell_g12_4
sfdg_cell_g12_5|G1F|sfdg_cell_g12_5
sfdg_cell_g5_1|G1M|sfdg_cell_g5_1
sfdg_cell_g6|G1M|sfdg_cell_g6
sfdg_cell_g8_1|G1M|sfdg_cell_g8_1
sfdg_cell_g7_1|G1J|sfdg_cell_g7_1
sfdg_cell_g9_1|G1J|sfdg_cell_g9_1
sfdg_cell_g11_1|G1J|sfdg_cell_g11_1
sfdg_cell_g12_verb|G1J|sfdg_cell_g12_verb
sfdg_cell_g12_6|G1J|sfdg_cell_g12_6
sfdg_cell_g12_7|G1J|sfdg_cell_g12_7
sfdg_cell_g12_10|G1J|sfdg_cell_g12_10
schema_fast_decode_gpu_p2_ref|G1J|tools/schema_fast_decode_gpu_p2_ref
sslm_schema_fast_decode_gpu_jf_bench|G1J|tools/schema_fast_decode_gpu_jf_bench
'

if [ "$LIST" = 1 ]; then
	echo "$TABLE" | sed '/^$/d' | awk -F'|' '{printf "%-22s %-4s %s\n", $1, $2, $3}'
	exit 0
fi

BASEFLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/paged-kv -Itests/paged-kv-gpu
-Itests/t2791-gpu-prefill-read-red-suite -Itests/schema-fast-decode-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN
-include d3d12.h -include d3d12sdklayers.h -DD3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE=D3D12_RESOURCE_STATES(0x400000)"
# mingw-w64 13's d3d12.h lacks D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE (0x400000 in the Windows SDK),
# which d3d12_harness.h names; the definition above stands in for it in this cloud build only.
SEAMDEFS="-DSUPERSLM_GPU_TEST_SEAMS -DSUPERSLM_GPU_ALLOC_FAULT_INJECTION -DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION -DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION"
CELLWARN="-Wall -Wextra -Wno-unused-parameter"
OBJ="$CACHE/sfdg-obj-$(echo "$REPO" | tr / _)"
mkdir -p "$OBJ" "$OUT"
cd "$REPO"

# compile OUTDIR "FLAGS" SRC... : one object per source, rebuilt when any dependency (-MMD) is newer.
compile() {
	local outdir="$1" flags="$2"; shift 2
	mkdir -p "$outdir"
	# shellcheck disable=SC2016
	printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c '
		src="$1"; o="$2/$(echo "$src" | tr / _ | sed "s/\.cpp$//").o"
		if [ -f "$o" ] && [ -f "$o.d" ]; then
			stale=0
			for d in $(sed -e "s/^[^:]*://" -e "s/\\\\$//" "$o.d"); do [ "$d" -nt "$o" ] && { stale=1; break; }; done
			[ $stale = 0 ] && exit 0
		fi
		if ! '"$CXX"' $3 -MMD -MF "$o.d" -c "$src" -o "$o" 2>"$o.log"; then
			echo "RED BY COMPILE: $src"; sed -n "1,40p" "$o.log"; rm -f "$o"; exit 1
		fi
		exit 0' _ {} "$outdir" "$flags"
}

ENGINE=()
# shellcheck disable=SC2086
for s in $SRCS; do
	case "$s" in src/*) ENGINE+=("$s") ;; esac
done

echo "== engine: ${#ENGINE[@]} sources, configuration plain (the production library; compile only)"
compile "$OBJ/engine-plain" "$(echo $BASEFLAGS)" "${ENGINE[@]}" || { echo "ENGINE (plain) FAILED TO COMPILE"; exit 1; }
echo "== engine: ${#ENGINE[@]} sources, configuration seams"
compile "$OBJ/engine-seams" "$(echo $BASEFLAGS) $SEAMDEFS" "${ENGINE[@]}" || { echo "ENGINE (seams) FAILED TO COMPILE"; exit 1; }

LINKED=() REDLINK=() REDCOMPILE=() NOTWRITTEN=()
build_exe() {  # build_exe "name|slice|sources"
	local name slice srcs
	IFS='|' read -r name slice srcs <<<"$1"
	local objs=() src
	for src in $srcs; do
		local path="tests/schema-fast-decode-gpu/$src.cpp"
		case "$src" in */*) path="$src.cpp" ;; esac
		if [ ! -f "$path" ]; then
			echo "[$slice] $name: NOT WRITTEN ($path)"; NOTWRITTEN+=("$name"); return 0
		fi
		if ! compile "$OBJ/cells" "$(echo $BASEFLAGS) $SEAMDEFS $CELLWARN" "$path"; then
			REDCOMPILE+=("$name"); return 0
		fi
		objs+=("$OBJ/cells/$(echo "${path%.cpp}" | tr / _).o")
	done
	local log="$OUT/$name.link.log"
	if "$CXX" -O2 "${objs[@]}" "$OBJ"/engine-seams/*.o -o "$OUT/$name.exe" \
		-ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc -static-libstdc++ >"$log" 2>&1; then
		rm -f "$OUT/$name.undefined"
		echo "[$slice] $name: LINKED"
		LINKED+=("$name")
	else
		grep -o "undefined reference to \`[^']*'" "$log" | sed "s/undefined reference to \`//; s/'\$//" | sort -u >"$OUT/$name.undefined" || true
		if [ -s "$OUT/$name.undefined" ]; then
			echo "[$slice] $name: RED BY LINK -- undefined:"
			sed 's/^/    /' "$OUT/$name.undefined"
			REDLINK+=("$name")
		else
			echo "[$slice] $name: UNEXPECTED LINK FAILURE -- see $log"; sed -n '1,30p' "$log"
			REDCOMPILE+=("$name")
		fi
		rm -f "$OUT/$name.exe"
	fi
}

while IFS= read -r row; do
	[ -z "$row" ] && continue
	IFS='|' read -r name slice _ <<<"$row"
	want=0
	for t in "${TARGETS[@]}"; do
		[ "$t" = "$name" ] && want=1
		[ "$(echo "$t" | tr a-z A-Z | tr -d -)" = "$slice" ] && want=1
	done
	[ $want = 1 ] && build_exe "$row"
done <<<"$TABLE"

echo "== ${#LINKED[@]} linked, ${#REDLINK[@]} red by link, ${#REDCOMPILE[@]} red by compile, ${#NOTWRITTEN[@]} not written"
[ ${#REDCOMPILE[@]} -eq 0 ] || { echo "RED BY COMPILE: ${REDCOMPILE[*]}"; exit 1; }
exit 0
