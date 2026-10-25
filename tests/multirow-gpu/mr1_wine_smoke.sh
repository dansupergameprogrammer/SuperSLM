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
#   wide   the widened leg: NOT WRITTEN. The plan names the gate but not what "one-layer" runs or which entry
#          the MR1 port exposes for it (there is no recorder until MR2). Reported as RED: the smoke cannot be
#          graded until that entry is named; see the MR1 red-suite record.
# Exit 0 only when every leg passes, so the gate stays red until the widened leg exists and agrees.
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

FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
ENGINE=()
# shellcheck disable=SC2086
for s in $SRCS; do case "$s" in src/*) ENGINE+=("$s") ;; esac; done
printf '%s\n' "${ENGINE[@]}" tests/multirow-gpu/mr1_twin_capture.cpp | xargs -P "$JOBS" -I{} bash -c \
	'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX $FLAGS"' -w -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$OUT/obj" {}
"$CXX" -O2 "$OUT/obj"/*.o -o "$OUT/mr1_twin_capture.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi \
	-static -static-libgcc -static-libstdc++
echo "built $OUT/mr1_twin_capture.exe"
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
echo "RED wide: the widened one-layer leg is not written -- the MR1 entry it calls is not named by the plan"
FAILED+=(wide-unwritten)
echo "MR1 Wine smoke: FAIL (${FAILED[*]})"
exit 1
