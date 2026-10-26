#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR5 (G1-X: Exact on the GPU): the cloud legs of the MR5 red suite, on the
# software GPU (Wine + vkd3d-proton + lavapipe). Record: build/red-suite-mr5.md. The box parts (10.5 Exact's frames,
# 10.2's g(M) from MR5's logits bracket) are MB5's; mr4_box.py grades them (ruling 18a).
#
# Usage: tests/multirow-gpu/mr5_wine.sh [--out=DIR] [--jobs=N] [--base=REV] [--only=LEG[,LEG...]] [--build-only] [--clean]
#   --out    default <repo>/out/mr5
#   --base   1.17's revision the c83 leg compares Off against (passed to mr4_wine.sh; its default otherwise)
#   --only   census | cells | green | box | mutants   (default: census,cells,green,box; mutants is mr5_mutants.sh)
#   --clean  on exit, remove the build trees and the Wine prefixes (the logs stay)
# Fixtures: F-SFD and its independent reference fsfd_ref.txt (tools/gen_schema_fast_decode_fixture.py) in <out>/fixtures
# (MR5_FIXTURES overrides), made when absent.
#
# Legs (each printed PASS or FAIL; exit 0 only when every leg run passes):
#   census   tests/ci/check_gpu_status_site_census.py: every status and catch site in src/gpu is in te425's list.
#   cells    mr5_cells.exe: 2.2, 2.3x, 7.10, 7.11e, 7.17, 7.18, 7.19, 8.4, gx.1, gx.2, gx.4-gx.13 (MR5_CELLS=... runs a
#            subset). Every name mr5_entry.h lists that src/ and include/ do not define yet is linked from
#            mr5_red_stubs.cpp, so the suite builds and runs red today; the leg passes -DMR5_HAVE_<NAME> for each one
#            the library defines (grep below), so a stub never sits next to the real definition.
#   green    the cells MR5 turns green that are red today only because the GPU refuses Exact (status 11; ruling 25's
#            list): mrx_gpu g.1, g.2 (non-vacuous at MR5: SSLM_OK, the chain kept, bytes equal the Off twin), g.3 (kind
#            2 admitted), g.4b (mrx_wine.sh --only=gpu,legacy); mr4_cells 7.9 and n1, and c83 (mr4_wine.sh
#            --only=cells,c83), all re-pointed to Exact (red-suite-mr5.md).
#   box      mr4_box.py --self-test: ruling 18a's flipped clock rule and g(M)'s derivation on synthetic tables (host).
#   mutants  mr5_mutants.sh (MX19, MX23, MX24, MXK6 on a green n1, and the XG mutants that return under MR5).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT="$REPO/out/mr5"
JOBS=2
RUN=1
CLEAN=0
ONLY="census,cells,green,box"
BASE="${MR_BASE_TAG:-}"
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--base=*) BASE="${a#--base=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--build-only) RUN=0 ;;
		--clean) CLEAN=1 ;;
		*) sed -n '2,29p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
mkdir -p "$OUT"
export WINEPREFIX="${WINEPREFIX_MR5:-$OUT/wineprefix-$$}"
cleanup() {
	if [ -z "${WINEPREFIX_MR5:-}" ]; then
		for p in "$WINEPREFIX" "$WINEPREFIX"-*; do [ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }; done
	fi
	if [ "$CLEAN" = 1 ]; then
		rm -rf "$OUT/cells/obj" "$OUT/cells/shaders"
		[ -d "$OUT/mr4/base-tree" ] && { git worktree remove --force "$OUT/mr4/base-tree" 2>/dev/null || rm -rf "$OUT/mr4/base-tree"; }
		rm -rf "$OUT/mr4/cells" "$OUT/mr4/c83"
		git worktree prune
	fi
	true
}
trap cleanup EXIT
want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
# shellcheck source=mr_lib.sh
. "$HERE/mr_lib.sh"
F5="${MR5_FIXTURES:-$OUT/fixtures}"
fixtures5() {
	mkdir -p "$F5"
	[ -f "$F5/fsfd.sslm" ] && [ -f "$F5/fsfd_ref.txt" ] || python3 tools/gen_schema_fast_decode_fixture.py "$F5" >/dev/null
	[ -f "$F5/fsfd.sslm" ] && [ -f "$F5/fsfd_ref.txt" ] || { echo "FAIL fixtures: no fsfd.sslm / fsfd_ref.txt in $F5"; return 1; }
}

# The MR5 names src/ already defines: -DMR5_HAVE_<NAME> for each (mr5_entry.h, mr5_red_stubs.cpp).
mr5_have() {
	local d="" pair name sym
	for pair in ROWS_PRODUCED:sslm_gpu_seq_rows_produced SATROW:ArmGpuSatRowForTest NARROW:ArmGpuFinishNarrowFaultForTest \
		READBACK:ArmGpuExactReadbackFaultForTest VERIFY:SetGpuExactVerifyAlwaysEqualForTest PASSLOG:GpuExactPassLogResetForTest; do
		name="${pair%%:*}" sym="${pair#*:}"
		# A definition: the name followed by its parameter list and a body, in src/ (a declaration in a header is not enough).
		if grep -rqE "\b$sym\s*\([^;]*\)\s*(noexcept\s*)?\{" src/ 2>/dev/null; then d="$d -DMR5_HAVE_$name"; fi
	done
	echo "$d"
}

# ------------------------------------------------------------------------------------------------ census
if want census; then
	if python3 tests/ci/check_gpu_status_site_census.py >"$OUT/census.log" 2>&1; then echo "PASS census ($(tail -1 "$OUT/census.log"))"
	else { sed 's/^/    /' "$OUT/census.log" | head -20; } || true; echo "FAIL census"; FAILED+=(census); fi
fi

# ------------------------------------------------------------------------------------------------ cells
if want cells; then
	HAVE="$(mr5_have)"
	EXTRA="-DSUPERSLM_MULTIROW_WIDE -DSUPERSLM_GPU_TEST_SEAMS -DSUPERSLM_GPU_ALLOC_FAULT_INJECTION -DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION -DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION$HAVE"
	echo "    cells: the library defines:${HAVE:- none of the mr5_entry.h names} (the rest are mr5_red_stubs.cpp's)"
	shaders "$REPO" "$OUT/cells/shaders" wide
	OBJD="$OUT/cells/obj"
	if [ "$(cat "$OBJD.flags" 2>/dev/null)" != "$FLAGS $EXTRA" ]; then rm -rf "$OBJD"; mkdir -p "$OUT/cells"; echo "$FLAGS $EXTRA" >"$OBJD.flags"; fi
	build "$REPO" "$OBJD" "$EXTRA" "$OUT/cells/mr5_cells.exe" tests/multirow-gpu/mr5_cells.cpp tests/multirow-gpu/mr5_red_stubs.cpp
	if [ "$RUN" = 1 ]; then
		fixtures5
		wine_up
		set +e
		(cd "$OUT/cells" && WINEPREFIX="$WINEPREFIX-cells" timeout 5400 "$WINE" ./mr5_cells.exe --fixtures="$F5" ${MR5_CELLS:+--cells=$MR5_CELLS}) >"$OUT/cells.log" 2>&1
		rc=$?
		set -e
		{ grep -E "^(FAIL|RED)|cell |cells:" "$OUT/cells.log" | tr -d '\r' | head -80 | sed 's/^/    /'; } || true
		if [ $rc = 0 ]; then echo "PASS cells"; else echo "FAIL cells (exit $rc, $OUT/cells.log)"; FAILED+=(cells); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ green (ruling 25's list)
if want green; then
	[ "$RUN" = 1 ] && fixtures5
	ok=1
	b=()
	[ "$RUN" = 1 ] || b=(--build-only)
	if ! MRX_FIXTURES="${MRX_FIXTURES:-$F5}" bash "$HERE/mrx_wine.sh" --out="$OUT/mrx" --jobs="$JOBS" --only=gpu,legacy --clean "${b[@]}" >"$OUT/green-mrx.log" 2>&1; then ok=0; fi
	{ grep -E "^(PASS|FAIL|RED)|cell " "$OUT/green-mrx.log" | grep -vE "^MRX" | head -30 | sed 's/^/    mrx: /'; } || true
	if ! MR4_CELLS=7.9,n1 bash "$HERE/mr4_wine.sh" --out="$OUT/mr4" --jobs="$JOBS" --only=cells,c83 ${BASE:+--base=$BASE} "${b[@]}" \
		>"$OUT/green-mr4.log" 2>&1; then ok=0; fi
	{ grep -E "^(PASS|FAIL|RED)|cell |    c83:" "$OUT/green-mr4.log" | grep -vE "^MR4" | head -30 | sed 's/^/    mr4: /'; } || true
	if [ $ok = 1 ]; then echo "PASS green (g.1, g.2, g.3, g.4b, 7.9, n1, c83)"; else echo "FAIL green ($OUT/green-*.log)"; FAILED+=(green); fi
fi

# ------------------------------------------------------------------------------------------------ box (host)
if want box; then
	if python3 "$HERE/mr4_box.py" --self-test >"$OUT/box.log" 2>&1; then echo "PASS box (mr4_box.py self-test: ruling 18a's clock rule and g(M))"
	else sed 's/^/    /' "$OUT/box.log"; echo "FAIL box"; FAILED+=(box); fi
fi

if want mutants; then
	if bash "$HERE/mr5_mutants.sh" --out="$OUT/mutants" --jobs="$JOBS"; then echo "PASS mutants"; else FAILED+=(mutants); fi
fi

[ "$RUN" = 1 ] || { echo "built"; exit 0; }
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR5 Wine: PASS"; exit 0; fi
echo "MR5 Wine: FAIL (${FAILED[*]})"
exit 1
