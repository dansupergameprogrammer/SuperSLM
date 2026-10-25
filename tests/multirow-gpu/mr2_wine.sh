#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR2 (the recorder): the cloud legs of the MR2 red suite, on the software
# GPU (Wine + vkd3d-proton + lavapipe, tools/cloud_gpu/wine_gpu.sh).
#
# Usage: tests/multirow-gpu/mr2_wine.sh [--out=DIR] [--jobs=N] [--base=TAG] [--only=LEG[,LEG...]] [--build-only]
#   --out   default <repo>/out/mr2 (under the repo: the engine's shader staleness check walks up to src/gpu/shaders)
#   --base  the base tag P4.2 anchors to (default MR_BASE_TAG, else the newest v* tag that is an ancestor of HEAD)
#   --only  p42 | wave | cells | mutants   (default: p42,wave,cells)
# Each run uses its own WINEPREFIX (<out>/wineprefix-<pid>) unless WINEPREFIX_MR2 is set.
#
# Legs (each printed PASS or FAIL; exit 0 only when every leg run passes):
#   p42      P4.2: mr2_p42_capture.exe built from the BASE TAG's tree (a detached worktree) and from the branch, on
#            pkv_def and pkv_qk; the branch's lines must equal p42_base_<tag>.txt, which the tag's binary wrote
#            (the run also re-derives it from the tag binary and requires it unchanged).
#   wave     4.9: mr2_wave_probe.exe records D3D12_FEATURE_DATA_D3D12_OPTIONS1.WaveLaneCountMin/Max and the lane
#            count a throwaway probe shader sees at the widened sites' group shape (diagnostic: always PASS when
#            the probe ran, the numbers go into the record).
#   cells    mr2_cells.exe, every cloud cell (see mr2_cells.cpp), on pkv_def and pkv_qk, with the refusal fixtures
#            (mr2_fixtures.py) and pkv_def_r4.adp (tools/gen_multirow_adapter_fixture.py). When src/ defines no
#            recorder entry, the cells are built against mr2_stub.cpp and are red-unimplemented. MR2_CELLS=6.1,6.6
#            runs a subset; MR2_PARTITIONS sets 6.2's count (default 200, the plan's). Before the cells, the
#            references they grade against are checked with no recorder involved (--instrument: twin determinism,
#            the refusal fixtures, the GPU-vs-CPU anchor), so they are green before the recorder exists. 4.10: 6.1's
#            hashes equal mr2_410_hashes.txt, which the first green run writes (commit it).
#   P4.1 is tests/multirow-gpu/p41_cloud.sh (MR1's runner), run unchanged at every step.
#   mutants  the MR2 mutants that can be built today (see mr2_mutants.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT="$REPO/out/mr2"
JOBS=2
RUN=1
ONLY="p42,wave,cells"
BASE="${MR_BASE_TAG:-}"
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--base=*) BASE="${a#--base=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--build-only) RUN=0 ;;
		*) sed -n '2,25p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
[ -n "$BASE" ] || BASE="$(git describe --tags --abbrev=0 --match 'v*' HEAD)"
export WINEPREFIX="${WINEPREFIX_MR2:-$OUT/wineprefix-$$}"
mkdir -p "$OUT"
want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"

# shaders TREE DEST [wide]: the tree's one-row shaders (and with `wide`, its widened variants as build.bat names
# them) compiled into DEST.
shaders() {
	local tree="$1" dest="$2" f o h g gi
	mkdir -p "$dest"
	for f in "$tree"/src/gpu/shaders/*.hlsl; do
		o="$dest/$(basename "$f" .hlsl).cso"
		[ -f "$o" ] && [ "$o" -nt "$f" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX >/dev/null
	done
	[ "${3:-}" = wide ] || return 0
	for h in 2 4 8; do
		for f in "$tree"/src/gpu/shaders_wide/*_site_wide.hlsl; do
			o="$dest/$(basename "$f" .hlsl)_h$h.cso"
			[ -f "$o" ] && [ "$o" -nt "$f" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX -D SSLM_WIDE_H=$h \
				-I "$tree/src/gpu/shaders" >/dev/null
		done
		gi=1
		for g in q_proj kv_proj o_proj gate_proj up_proj down_proj; do
			o="$dest/${g}_gemm_wide_h$h.cso"
			[ -f "$o" ] && [ "$o" -nt "$tree/src/gpu/shaders_wide/proj_gemm_wide.hlsl" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" \
				"$tree/src/gpu/shaders_wide/proj_gemm_wide.hlsl" -O3 -HV 2018 -WX -D SSLM_WIDE_H=$h -D SSLM_GEMM_SITE=$gi \
				-I "$tree/src/gpu/shaders" >/dev/null
			gi=$((gi + 1))
		done
	done
}

# build TREE OBJDIR "EXTRA" EXE CELL...: the tree's engine sources (wine_gpu.sh's SRCS, src/ entries) plus the
# cell sources, compiled in TREE (cell paths relative to it) and linked to EXE.
build() {
	local tree="$1" obj="$2" extra="$3" exe="$4"
	shift 4
	local engine
	engine="$(cd "$tree" && WINE_GPU_SOURCED=1 bash -c '. tools/cloud_gpu/wine_gpu.sh; for s in $SRCS; do case "$s" in src/*) echo "$s" ;; esac; done')"
	mkdir -p "$obj" "$(dirname "$exe")"
	( cd "$tree" && printf '%s\n' $engine "$@" | xargs -P "$JOBS" -I{} bash -c \
		'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX $FLAGS $extra"' -w -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$obj" {} )
	"$CXX" -O2 "$obj"/*.o -o "$exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc -static-libstdc++
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$(dirname "$exe")/"
}

# Fixtures: tools/gen_paged_kv_fixture.py's (slow: it writes every variant, ~30 min of CPU), reused from MR_FIXTURES
# or the MR1 smoke's directory when they are already there.
FIX="$OUT/fixtures"
fixtures() {
	mkdir -p "$FIX"
	local src
	for src in "${MR_FIXTURES:-}" "$REPO/out/mr1-smoke/fixtures"; do
		[ -n "$src" ] && [ -f "$src/pkv_def.sslm" ] && [ -f "$src/pkv_qk.sslm" ] && [ ! -f "$FIX/pkv_def.sslm" ] && cp "$src"/pkv_*.sslm "$FIX/"
	done
	[ -f "$FIX/pkv_def.sslm" ] && [ -f "$FIX/pkv_qk.sslm" ] || python3 tools/gen_paged_kv_fixture.py "$FIX" >/dev/null
}
WINE_UP=0
wine_up() { [ "$WINE_UP" = 1 ] || { start_xvfb; wine_env; WINE_UP=1; }; }

# ------------------------------------------------------------------------------------------------ P4.2
if want p42; then
	BT="$OUT/base-tree"
	if [ ! -d "$BT/src" ]; then
		git worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
		git worktree add --detach "$BT" "$BASE" >/dev/null
	fi
	[ "$(git -C "$BT" rev-parse HEAD)" = "$(git rev-parse "$BASE^{commit}")" ] || { echo "FAIL p42: $BT is not $BASE"; exit 1; }
	mkdir -p "$BT/tests/multirow-gpu"
	cp "$HERE/mr2_p42_capture.cpp" "$HERE/mingw_d3d12_compat.h" "$BT/tests/multirow-gpu/"
	shaders "$BT" "$BT/out/p42/shaders"
	build "$BT" "$BT/out/p42/obj" "" "$BT/out/p42/mr2_p42_capture.exe" tests/multirow-gpu/mr2_p42_capture.cpp
	shaders "$REPO" "$OUT/p42/shaders"
	build "$REPO" "$OUT/p42/obj" "" "$OUT/p42/mr2_p42_capture.exe" tests/multirow-gpu/mr2_p42_capture.cpp
	if [ "$RUN" = 1 ]; then
		fixtures; wine_up
		PIN="$HERE/p42_base_$BASE.txt"
		set +e
		"$WINE" "$BT/out/p42/mr2_p42_capture.exe" --model="$FIX/pkv_def.sslm" --model="$FIX/pkv_qk.sslm" >"$OUT/p42-base.txt" 2>"$OUT/p42-base.err"
		rb=$?
		"$WINE" "$OUT/p42/mr2_p42_capture.exe" --model="$FIX/pkv_def.sslm" --model="$FIX/pkv_qk.sslm" >"$OUT/p42-branch.txt" 2>"$OUT/p42-branch.err"
		rr=$?
		set -e
		tr -d '\r' <"$OUT/p42-base.txt" >"$OUT/p42-base.lf"; tr -d '\r' <"$OUT/p42-branch.txt" >"$OUT/p42-branch.lf"
		if [ ! -f "$PIN" ]; then
			echo "p42: no pin $PIN; writing it from the base tag's binary (commit it)"
			[ $rb = 0 ] && cp "$OUT/p42-base.lf" "$PIN"
		fi
		if [ $rb != 0 ]; then echo "FAIL p42: the base tag's binary failed (exit $rb, $OUT/p42-base.txt)"; FAILED+=(p42)
		elif ! diff -q "$PIN" "$OUT/p42-base.lf" >/dev/null; then echo "FAIL p42: the base tag's binary no longer reproduces $PIN"; FAILED+=(p42)
		elif [ $rr != 0 ] || ! diff "$PIN" "$OUT/p42-branch.lf"; then echo "FAIL p42: branch differs from $BASE (exit $rr)"; FAILED+=(p42)
		else echo "PASS p42: $(wc -l <"$PIN") hashes equal to $BASE's binary"; fi
	fi
fi

# ------------------------------------------------------------------------------------------------ 4.9
if want wave; then
	mkdir -p "$OUT/wave"
	"$DXC" -T cs_6_2 -E main -Fo "$OUT/wave/mr2_wave_probe.cso" "$HERE/mr2_wave_probe.hlsl" -O3 -HV 2018 >/dev/null
	"$CXX" -std=c++20 -O2 "$HERE/mr2_wave_probe.cpp" -o "$OUT/wave/mr2_wave_probe.exe" -ld3d12 -ldxgi -ldxguid \
		-static -static-libgcc -static-libstdc++
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$OUT/wave/"
	if [ "$RUN" = 1 ]; then
		wine_up
		set +e
		"$WINE" "$OUT/wave/mr2_wave_probe.exe" --cso="$OUT/wave/mr2_wave_probe.cso" >"$OUT/wave.txt" 2>&1
		rc=$?
		set -e
		sed 's/^/    /' "$OUT/wave.txt" | tr -d '\r'
		if [ $rc = 0 ]; then echo "PASS wave (4.9, recorded)"; else echo "FAIL wave: probe exit $rc"; FAILED+=(wave); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ cells
if want cells; then
	STUB=()
	EXTRA="-DSUPERSLM_MULTIROW_WIDE"
	if ! grep -rqs "sslm_gpu_seq_embed_rows" src/gpu/*.cpp src/gpu/*.inl; then
		STUB=(tests/multirow-gpu/mr2_stub.cpp)
		EXTRA="$EXTRA -DMR2_STUB"
		echo "cells: src/ defines no recorder entry; building against mr2_stub.cpp (red-unimplemented)"
	fi
	shaders "$REPO" "$OUT/cells/shaders" wide
	build "$REPO" "$OUT/cells/obj$([ ${#STUB[@]} -gt 0 ] && echo -stub)" "$EXTRA" "$OUT/cells/mr2_cells.exe" \
		tests/multirow-gpu/mr2_cells.cpp "${STUB[@]}"
	if [ "$RUN" = 1 ]; then
		fixtures; wine_up
		python3 "$HERE/mr2_fixtures.py" "$FIX" >/dev/null
		[ -f "$FIX/pkv_def_r4.adp" ] || python3 tools/gen_multirow_adapter_fixture.py "$FIX/pkv_def.sslm" "$FIX/pkv_def_r4.adp" >/dev/null
		# The references first (no recorder involved; green before the recorder exists).
		for m in pkv_def pkv_qk; do
			set +e
			"$WINE" "$OUT/cells/mr2_cells.exe" --model="$FIX/$m.sslm" --fixtures="$FIX" --instrument >"$OUT/instruments-$m.log" 2>&1
			rc=$?
			set -e
			grep -E "^(FAIL|RED|PASS)" "$OUT/instruments-$m.log" | tr -d '\r' | head -20 | sed 's/^/    /'
			if [ $rc = 0 ]; then echo "PASS instruments $m"; else echo "FAIL instruments $m"; FAILED+=("instruments-$m"); fi
		done
		for m in pkv_def pkv_qk; do
			set +e
			"$WINE" "$OUT/cells/mr2_cells.exe" --model="$FIX/$m.sslm" --fixtures="$FIX" --adapter="$FIX/pkv_def_r4.adp" \
				--partitions="${MR2_PARTITIONS:-200}" ${MR2_CELLS:+--cells=$MR2_CELLS} >"$OUT/cells-$m.log" 2>&1
			rc=$?
			set -e
			grep -E "^(FAIL|RED|cell)|cells:" "$OUT/cells-$m.log" | tr -d '\r' | head -60 | sed 's/^/    /'
			if [ $rc = 0 ]; then echo "PASS cells $m"; else echo "FAIL cells $m (exit $rc, $OUT/cells-$m.log)"; FAILED+=("cells-$m"); cells_red=1; fi
		done
		# 4.10: 6.1's blob hashes are pinned from the first green run (mr2_410_hashes.txt, committed); every later run
		# must reproduce them byte for byte. A red run never writes the pin.
		if [ -z "${MR2_CELLS:-}" ] || [[ ",$MR2_CELLS," == *",6.1,"* ]]; then
			PIN410="$HERE/mr2_410_hashes.txt"
			cat "$OUT/cells-pkv_def.log" "$OUT/cells-pkv_qk.log" | tr -d '\r' | grep "^hash 4.10 " >"$OUT/hash410.txt" || true
			if [ ! -s "$OUT/hash410.txt" ]; then echo "FAIL 4.10: the cells printed no hashes"; FAILED+=(4.10)
			elif [ ! -f "$PIN410" ]; then
				if [ "${cells_red:-0}" = 0 ]; then cp "$OUT/hash410.txt" "$PIN410"; echo "4.10: pinned $(wc -l <"$PIN410") hashes from this green run into $PIN410 (commit it)"
				else echo "FAIL 4.10: no pin, and a red run does not write one"; FAILED+=(4.10); fi
			elif diff "$PIN410" "$OUT/hash410.txt" >"$OUT/hash410.diff"; then echo "PASS 4.10 ($(wc -l <"$PIN410") hashes = the pin)"
			else echo "FAIL 4.10: hashes differ from $PIN410 ($OUT/hash410.diff)"; FAILED+=(4.10); fi
		fi
	fi
fi

if want mutants; then
	if bash "$HERE/mr2_mutants.sh" --out="$OUT/mutants" --jobs="$JOBS"; then echo "PASS mutants"; else FAILED+=(mutants); fi
fi

[ "$RUN" = 1 ] || { echo "built"; exit 0; }
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR2 Wine: PASS"; exit 0; fi
echo "MR2 Wine: FAIL (${FAILED[*]})"
exit 1
