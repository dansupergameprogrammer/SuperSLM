#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2) cell P4.1 [cloud]: the base tag's existing GPU suite runs unmodified and
# green on the branch, on the software GPU (Wine + vkd3d-proton + lavapipe, tools/cloud_gpu/wine_gpu.sh).
#
# Usage: tests/multirow-gpu/p41_cloud.sh [--base=TAG] [--out=DIR] [--jobs=N] [--skip-run] [--only=LEG]
#   --base     the base tag the branch is anchored to (default: MR_BASE_TAG, else the newest v* tag that is
#              an ancestor of HEAD)
#   --out      default <repo>/out/mr-p41 (under the repo: the engine's shader staleness check walks up to
#              src/gpu/shaders)
#   --skip-run build only
#   --only     unmodified | tests | pkvg   run one leg
#   --with-pkvg  also run the pkvg leg (off by default: under lavapipe its lifetime cells, e.g. 1.9's 1,000
#              prefix cycles, run for hours)
# Each run uses its own WINEPREFIX (<out>/wineprefix-<pid>) unless WINEPREFIX is set.
#
# Legs, each printed PASS or FAIL, exit 0 only when every leg run passes:
#   unmodified  `git diff <base> HEAD` is empty over tests/ (except tests/multirow-gpu/), and build.bat's
#               superslm_tests compile command (sources, /D, /I) equals the tag's: the suite that runs is the
#               tag's. build.bat may gain shader steps (the widened variants).
#   tests       superslm_tests.exe, built from build.bat's own superslm_tests command (sources, /D and /I
#               parsed from the file, so the cloud build follows the box build), run under Wine; exit 0.
#   pkvg        (opt-in) the paged-KV GPU suite's step-G5 runner (tests/paged-kv-gpu, pkvg_g5) over
#               tools/gen_paged_kv_fixture.py's fixtures (pkv_def, pkv_qk, ...), real-artifact cells skipped; exit 0.
#
# Build notes: mingw-w64 11's d3d12.h lacks an enumerator the engine uses, so compiles go through
# tests/multirow-gpu/mingw_cxx.sh (force-includes mingw_d3d12_compat.h). wine_gpu.sh's own `build` source
# list predates tests/test_schema_fast_decode.cpp and no longer links; this script does not use it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

BASE="${MR_BASE_TAG:-}"
OUT="$REPO/out/mr-p41"
JOBS=2
RUN=1
ONLY=""
WITH_PKVG=0
for a in "$@"; do
	case "$a" in
		--base=*) BASE="${a#--base=}" ;;
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--skip-run) RUN=0 ;;
		--only=*) ONLY="${a#--only=}" ;;
		--with-pkvg) WITH_PKVG=1 ;;
		*) sed -n '2,25p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
if [ -z "$BASE" ]; then
	BASE="$(git describe --tags --abbrev=0 --match 'v*' HEAD)"
fi
export WINEPREFIX="${WINEPREFIX_P41:-$OUT/wineprefix-$$}"
mkdir -p "$OUT/shaders" "$OUT/obj"
FAILED=()
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

if want unmodified; then
	echo "== P4.1 unmodified: the suite's sources and build.bat's superslm_tests command against $BASE"
	ok=1
	if ! git diff --quiet "$BASE" HEAD -- tests ':(exclude)tests/multirow-gpu'; then
		echo "    tests/ outside tests/multirow-gpu differs from $BASE:"
		git diff --stat "$BASE" HEAD -- tests ':(exclude)tests/multirow-gpu' | sed 's/^/        /'
		ok=0
	fi
	# build.bat may gain shader steps (the widened variants); the superslm_tests compile line may not change.
	if ! diff <(git show "$BASE:build.bat" | python3 -I "$HERE/p41_parse_build_bat.py" -) \
		<(python3 -I "$HERE/p41_parse_build_bat.py" build.bat) >/dev/null; then
		echo "    build.bat's superslm_tests command (sources, /D, /I) differs from $BASE"
		ok=0
	fi
	if [ $ok = 1 ]; then echo "PASS unmodified"; else echo "FAIL unmodified"; FAILED+=(unmodified); fi
fi

shaders() {
	for f in src/gpu/shaders/*.hlsl; do
		local o="$OUT/shaders/$(basename "$f" .hlsl).cso"
		if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ -n "$(find src/gpu/shaders -name '*.hlsli' -newer "$o")" ]; then
			"$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX >/dev/null
		fi
	done
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$OUT/"
}

if want tests; then
	echo "== P4.1 tests: superslm_tests.exe from build.bat's command"
	PARSED="$(python3 -I "$HERE/p41_parse_build_bat.py" build.bat)"
	SRCS_BAT="$(sed -n 1p <<<"$PARSED")"; DEFS_BAT="$(sed -n 2p <<<"$PARSED")"; INCS_BAT="$(sed -n 3p <<<"$PARSED")"
	FLAGS="-std=c++20 -O2 $INCS_BAT $DEFS_BAT -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
	shaders
	OBJ="$OUT/obj/tests"; mkdir -p "$OBJ"
	# shellcheck disable=SC2086
	if ! printf '%s\n' $SRCS_BAT | xargs -P "$JOBS" -I{} bash -c \
		'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX $FLAGS"' -w -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$OBJ" {}; then
		FAILED+=(tests-compile)
	elif ! "$CXX" -O2 "$OBJ"/*.o -o "$OUT/superslm_tests.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi \
		-static -static-libgcc -static-libstdc++ >"$OUT/superslm_tests.link.log" 2>&1; then
		echo "FAIL tests: link (see $OUT/superslm_tests.link.log)"; sed -n '1,20p' "$OUT/superslm_tests.link.log"
		FAILED+=(tests-link)
	elif [ "$RUN" = 1 ]; then
		# The fixtures CI points the suite at (.github/workflows/tests.yml): generated, never committed.
		FIX="$OUT/fixtures"; mkdir -p "$FIX/dt" "$FIX/sfd"
		[ -f "$FIX/pkv_def.sslm" ] || python3 tools/gen_paged_kv_fixture.py "$FIX" >/dev/null
		[ -n "$(ls "$FIX/dt")" ] || python3 tools/gen_decode_threading_fixture.py "$FIX/dt" >/dev/null
		[ -n "$(ls "$FIX/sfd")" ] || python3 tools/gen_schema_fast_decode_fixture.py "$FIX/sfd" >/dev/null
		export SUPERSLM_PAGED_KV_FIXTURE_DIR="$FIX" SUPERSLM_DECODE_THREADING_FIXTURE_DIR="$FIX/dt" \
			SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR="$FIX/sfd"
		start_xvfb; wine_env
		set +e
		"$WINE" "$OUT/superslm_tests.exe" >"$OUT/superslm_tests.log" 2>&1
		rc=$?
		set -e
		tail -5 "$OUT/superslm_tests.log" | sed 's/^/    /'
		if [ $rc = 0 ]; then echo "PASS tests (superslm_tests.exe exit 0)"; else echo "FAIL tests: exit $rc (log $OUT/superslm_tests.log)"; FAILED+=(tests); fi
	fi
fi

if { [ -z "$ONLY" ] && [ "$WITH_PKVG" = 1 ]; } || [ "$ONLY" = pkvg ]; then
	echo "== P4.1 pkvg: the paged-KV GPU suite's G5 runner on pkv_def and pkv_qk"
	FIX="$OUT/fixtures"; mkdir -p "$FIX"
	[ -f "$FIX/pkv_def.sslm" ] && [ -f "$FIX/pkv_qk.sslm" ] || python3 tools/gen_paged_kv_fixture.py "$FIX" >/dev/null
	if ! bash tests/paged-kv-gpu/build_paged_kv_gpu_mingw.sh --out="$OUT/pkvg" --jobs="$JOBS" pkvg_g5 >"$OUT/pkvg.build.log" 2>&1 \
		|| [ ! -f "$OUT/pkvg/pkvg_g5.exe" ]; then
		echo "FAIL pkvg: build (see $OUT/pkvg.build.log)"; tail -20 "$OUT/pkvg.build.log"
		FAILED+=(pkvg-build)
	elif [ "$RUN" = 1 ]; then
		start_xvfb; wine_env
		# From the repo root: --pkv-reference defaults to tests/paged-kv/reference under the working directory.
		# --allow-skip: the cells that need the real 0.5B/1.5B artifacts are box-only and skip here.
		set +e
		"$WINE" "$OUT/pkvg/pkvg_g5.exe" --pkv-fixtures="$FIX" --allow-skip >"$OUT/pkvg_g5.log" 2>&1
		rc=$?
		set -e
		tail -3 "$OUT/pkvg_g5.log" | sed 's/^/    /'
		if [ $rc = 0 ]; then echo "PASS pkvg (pkvg_g5.exe exit 0 on $FIX)"; else echo "FAIL pkvg: exit $rc (log $OUT/pkvg_g5.log)"; FAILED+=(pkvg); fi
	fi
fi

if [ ${#FAILED[@]} -eq 0 ]; then echo "P4.1 cloud: PASS"; exit 0; fi
echo "P4.1 cloud: FAIL (${FAILED[*]})"; exit 1
