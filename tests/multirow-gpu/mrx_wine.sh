#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MRX (plan §16 erratum E-11, ruling 20): Canonical removal. The red suite's
# runner: the CPU-only legs (mrx_cpu.sh, native Linux, no Wine) and the GPU legs on the software GPU (Wine +
# vkd3d-proton + lavapipe). Record: build/red-suite-mrx.md.
#
# Usage: tests/multirow-gpu/mrx_wine.sh [--out=DIR] [--jobs=N] [--only=LEG[,LEG...]] [--build-only] [--clean]
#   --out    default <repo>/out/mrx
#   --only   cpu | census | gpu | legacy   (default: all four)
#   --clean  on exit, remove the build trees, the v1.15-1.17 worktrees and the Wine prefixes (the logs stay)
# Fixtures: F-SFD (tools/gen_schema_fast_decode_fixture.py) in out/mrx/fixtures (or MRX_FIXTURES), made when absent.
#
# Legs (each printed PASS or FAIL; every leg runs whatever an earlier one did; exit 0 only when every leg run passes):
#   cpu      mrx_cpu.sh: header (no SSLM_JUMP_CANONICAL / SSLM_GPU_ROWS_SCHEMA_FORCED in the public headers, 0 not
#            reused), changelog (1.18.0 "Removed"), cells (x.1 mode 0 refused, x.3 NULL = explicit Exact), legacy (x.5:
#            1.15-1.17 CPU Canonical saves restore and continue).
#   census   tests/ci/check_gpu_status_site_census.py (every status and catch site in src/gpu is in te425's list).
#   gpu      mrx_gpu.exe g.1 (mode 0 refused on sslm_gpu_seq_jump_forward), g.2 (NULL = explicit Exact; red until
#            MR5), g.3 (embed_rows kind 1 refused as a bad spec).
#   legacy   the recorder (mrx_gpu.cpp built from the v1.17.0 tree with -DMRX_V117) saves after 1.17's GPU Canonical
#            jump; the branch's mrx_gpu.exe g.4a (restore, same bytes, decode-only continuation) and g.4b (continues
#            under explicit Exact and NULL; red until MR5).
# MR5 (red-suite-mr5.md): g.1, g.2 and g.4b are non-vacuous (SSLM_OK, the chain kept, bytes equal the Off twin; the
# model maps with the device logits head), and g.3's kind 2 is admitted. mr5_wine.sh's green leg runs gpu and legacy.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
OUT="$REPO/out/mrx"
JOBS=2
RUN=1
CLEAN=0
ONLY="cpu,census,gpu,legacy"
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--build-only) RUN=0 ;;
		--clean) CLEAN=1 ;;
		*) sed -n '2,24p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
mkdir -p "$OUT"
want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
TAG117=v1.17.0
T117="$OUT/tree-$TAG117"
export WINEPREFIX="${WINEPREFIX_MRX:-$OUT/wineprefix-$$}"
cleanup() {
	if [ -z "${WINEPREFIX_MRX:-}" ]; then
		for p in "$WINEPREFIX" "$WINEPREFIX"-*; do [ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }; done
	fi
	if [ "$CLEAN" = 1 ]; then
		rm -rf "$OUT/gpu" "$OUT/cpu"
		for t in v1.15.0 v1.16.0 v1.17.0; do [ -d "$OUT/tree-$t" ] && { git worktree remove --force "$OUT/tree-$t" 2>/dev/null || rm -rf "$OUT/tree-$t"; }; done
		git worktree prune
	fi
	true
}
trap cleanup EXIT

# ------------------------------------------------------------------------------------------------ cpu
if want cpu; then
	if [ "$RUN" = 1 ]; then
		if bash "$HERE/mrx_cpu.sh" --out="$OUT" --jobs="$JOBS" 2>&1 | tee "$OUT/cpu.log" | sed 's/^/  /'; [ "${PIPESTATUS[0]}" = 0 ]; then
			echo "PASS cpu"
		else echo "FAIL cpu ($OUT/cpu.log)"; FAILED+=(cpu); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ census
if want census; then
	if python3 tests/ci/check_gpu_status_site_census.py >"$OUT/census.log" 2>&1; then echo "PASS census ($(tail -1 "$OUT/census.log"))"
	else { sed 's/^/    /' "$OUT/census.log" | head -20; } || true; echo "FAIL census"; FAILED+=(census); fi
fi

# ------------------------------------------------------------------------------------------------ GPU builds
if want gpu || want legacy; then
	# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
	WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
	FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
	# shellcheck source=mr_lib.sh
	. "$HERE/mr_lib.sh"
	FX="${MRX_FIXTURES:-$OUT/fixtures}"
	need_fixtures() { [ -f "$FX/fsfd.sslm" ] || { mkdir -p "$FX"; python3 tools/gen_schema_fast_decode_fixture.py "$FX" >/dev/null; }; }
	GB="$OUT/gpu/branch"
	if shaders "$REPO" "$GB/shaders" wide >"$OUT/gpu-build.log" 2>&1 &&
		build "$REPO" "$GB/obj" "" "$GB/mrx_gpu.exe" tests/multirow-gpu/mrx_gpu.cpp >>"$OUT/gpu-build.log" 2>&1; then GB_OK=1
	else GB_OK=0; { tail -5 "$OUT/gpu-build.log" | sed 's/^/    /'; } || true; echo "FAIL build: the branch's mrx_gpu.exe ($OUT/gpu-build.log)"; FAILED+=(build); fi
fi

# ------------------------------------------------------------------------------------------------ gpu
if want gpu && [ "${GB_OK:-0}" = 1 ] && [ "$RUN" = 1 ]; then
	need_fixtures
	wine_up
	set +e
	(cd "$GB" && WINEPREFIX="$WINEPREFIX-gpu" "$WINE" ./mrx_gpu.exe --fixtures="$FX" --cells=g.1,g.2,g.3) 2>/dev/null | tr -d '\r' >"$OUT/gpu.log"
	rc=${PIPESTATUS[0]}
	set -e
	{ grep -E "^(FAIL|RED)|cell |cells:" "$OUT/gpu.log" | head -60 | sed 's/^/    /'; } || true
	if [ "$rc" = 0 ]; then echo "PASS gpu (g.1, g.2, g.3)"; else echo "FAIL gpu (exit $rc, $OUT/gpu.log)"; FAILED+=(gpu); fi
fi

# ------------------------------------------------------------------------------------------------ legacy
if want legacy && [ "${GB_OK:-0}" = 1 ]; then
	ok=1
	REC="$OUT/legacy-gpu/$TAG117"
	G7="$OUT/gpu/$TAG117"
	if [ ! -f "$REC/longkey.next" ]; then
		if [ ! -d "$T117/src" ]; then git worktree add --detach "$T117" "$TAG117" >/dev/null 2>&1 || { ok=0; echo "    legacy: no tree for $TAG117"; }; fi
		if [ $ok = 1 ]; then
			mkdir -p "$T117/tests/multirow-gpu"
			cp "$HERE/mrx_gpu.cpp" "$HERE/mingw_d3d12_compat.h" "$T117/tests/multirow-gpu/"
			if shaders "$T117" "$G7/shaders" >"$OUT/gpu-$TAG117-build.log" 2>&1 &&
				build "$T117" "$G7/obj" "-DMRX_V117" "$G7/mrx_gpu.exe" tests/multirow-gpu/mrx_gpu.cpp >>"$OUT/gpu-$TAG117-build.log" 2>&1; then :
			else ok=0; { tail -5 "$OUT/gpu-$TAG117-build.log" | sed 's/^/    /'; } || true; echo "    legacy: the $TAG117 recorder did not build"; fi
		fi
		if [ $ok = 1 ] && [ "$RUN" = 1 ]; then
			need_fixtures
			wine_up
			mkdir -p "$REC"
			set +e
			(cd "$G7" && WINEPREFIX="$WINEPREFIX-legacy" "$WINE" ./mrx_gpu.exe --fixtures="$FX" --record="$REC") 2>/dev/null |
				tr -d '\r' >"$OUT/gpu-$TAG117-record.log"
			rc=${PIPESTATUS[0]}
			set -e
			sed 's/^/    /' "$OUT/gpu-$TAG117-record.log" | head -6 || true
			[ "$rc" = 0 ] || { ok=0; echo "    legacy: the $TAG117 recorder failed (exit $rc)"; }
		fi
	fi
	if [ "$RUN" = 1 ]; then
		need_fixtures
		wine_up
		set +e
		(cd "$GB" && WINEPREFIX="$WINEPREFIX-gpu" "$WINE" ./mrx_gpu.exe --fixtures="$FX" --legacy="$REC" --cells=g.4) 2>/dev/null | tr -d '\r' >"$OUT/gpu-legacy.log"
		rc=${PIPESTATUS[0]}
		set -e
		{ grep -E "^(FAIL|RED)|cell |cells:" "$OUT/gpu-legacy.log" | head -30 | sed 's/^/    /'; } || true
		[ "$rc" = 0 ] || ok=0
		if [ $ok = 1 ]; then echo "PASS legacy (g.4: 1.17's GPU Canonical saves restore and continue)"; else echo "FAIL legacy ($OUT/gpu-legacy.log)"; FAILED+=(legacy); fi
	fi
fi

[ "$RUN" = 1 ] || { echo "built"; exit 0; }
if [ ${#FAILED[@]} -eq 0 ]; then echo "MRX: PASS"; exit 0; fi
echo "MRX: FAIL (${FAILED[*]})"
exit 1
