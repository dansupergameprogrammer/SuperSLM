#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR3 (the public prompt verb): the cloud legs of the MR3 red suite, on the
# software GPU (Wine + vkd3d-proton + lavapipe).
#
# Usage: tests/multirow-gpu/mr3_wine.sh [--out=DIR] [--jobs=N] [--base=TAG] [--only=LEG[,LEG...]] [--build-only]
#   --out   default <repo>/out/mr3
#   --base  the base tag 9.2 restores against (default MR_BASE_TAG, else the newest v* tag that is an ancestor of HEAD)
#   --only  contract | cells | b92 | tsan | mutants   (default: contract,cells,b92,tsan)
# Fixtures are MR2's (out/mr2/fixtures, or MR3_FIXTURES), made by mr2_wine.sh's helpers when absent.
#
# Legs (each printed PASS or FAIL; exit 0 only when every leg run passes):
#   contract  include/superslm/gpu_1p0.h publishes plan §4.1's verbs and types, and the two branch-provisional statuses
#             (§3.7, mr3_entry.h) appended after SSLM_RESTORE_PRODUCER_MISMATCH (30), each marked provisional on its line.
#   cells     mr3_cells.exe on pkv_def and pkv_qk (MR3_CELLS=2.1,7.6 runs a subset; MR3_CYCLES sets 1.1's count, default
#             1000; MR3_RACES sets 3.1's free-running count, default 10000). Built with -DMR3_UNPUBLISHED until the
#             contract leg passes (see mr3_entry.h).
#   b92       9.2: mr3_92_blobs.exe built from the base tag's tree and from the branch; the tag's blobs restore on the
#             branch and the branch's (written through embed_rows) restore on the tag, every reader printing the lines the
#             tag prints reading its own blobs, and the branch's blobs byte-equal to the tag's.
#   tsan      3.3's TSan leg: mr3_tsan.cpp, native g++ -fsanitize=thread, against src/gpu/multirow_rows_host.h.
#   mutants   MX9, MX10, MX17, MX18 (mr3_mutants.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT="$REPO/out/mr3"
JOBS=2
RUN=1
ONLY="contract,cells,b92,tsan"
BASE="${MR_BASE_TAG:-}"
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--base=*) BASE="${a#--base=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--build-only) RUN=0 ;;
		*) sed -n '2,24p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
[ -n "$BASE" ] || BASE="$(git describe --tags --abbrev=0 --match 'v*' HEAD)"
export WINEPREFIX="${WINEPREFIX_MR3:-$OUT/wineprefix-$$}"
# A prefix this run made (not a caller's) is removed at exit, with its per-model children: each is a full Wine tree.
if [ -z "${WINEPREFIX_MR3:-}" ]; then
	trap 'for p in "$WINEPREFIX" "$WINEPREFIX"-*; do [ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }; done; true' EXIT
fi
mkdir -p "$OUT"
want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
# shellcheck source=mr_lib.sh
. "$HERE/mr_lib.sh"
FIX="${MR3_FIXTURES:-$REPO/out/mr2/fixtures}"

# ------------------------------------------------------------------------------------------------ contract
# Exit 0 when gpu_1p0.h publishes MR3's surface; prints what is missing otherwise.
contract() {
	python3 -I - "$REPO/include/superslm/gpu_1p0.h" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
miss = []
for sym in ["#define SSLM_GPU_MAX_ROWS 8", "SSLM_GPU_ROWS_FIRST_PREWALKED", "SSLM_GPU_ROWS_PROMPT", "SSLM_GPU_ROWS_SCHEMA_FORCED",
            "SSLM_GPU_ROWS_EXACT_DRAFT", "SslmGpuRowsSpec", "sslm_gpu_seq_embed_rows(", "sslm_gpu_seq_rows_abandon(",
            "sslm_gpu_seq_rows_open(", "sslm_gpu_seq_last_rows_committed("]:
    if sym not in src: miss.append("declares no " + sym)
m = re.search(r"enum class SslmGpuStatus[^{]*\{(.*?)\};", src, re.S)
if not m: miss.append("no SslmGpuStatus enum")
else:
    body = m.group(1)
    lines = [l for l in body.split("\n")]
    names = [re.match(r"\s*(SSLM_[A-Z0-9_]+)", l).group(1) for l in lines if re.match(r"\s*SSLM_[A-Z0-9_]+", l)]
    if "SSLM_RESTORE_PRODUCER_MISMATCH" not in names: miss.append("the enum has no SSLM_RESTORE_PRODUCER_MISMATCH (30)")
    else:
        after = names[names.index("SSLM_RESTORE_PRODUCER_MISMATCH") + 1:]
        for want in ["SSLM_GPU_ROWS_CAP_EXCEEDED", "SSLM_GPU_ROWS_CHUNK_OPEN"]:
            if want not in after: miss.append(want + " is not appended after SSLM_RESTORE_PRODUCER_MISMATCH")
            else:
                line = next(l for l in lines if re.match(r"\s*" + want + r"\b", l))
                # the marker may sit on the enumerator's line or in the comment block directly above it
                i = lines.index(line)
                ctx = " ".join(lines[max(0, i - 3):i + 1]).lower()
                if "provisional" not in ctx: miss.append(want + " is not marked provisional")
for x in miss: print("    contract: " + x)
sys.exit(1 if miss else 0)
PY
}
PUBLISHED=0
if contract >"$OUT/contract.txt"; then PUBLISHED=1; fi
if want contract; then
	if [ $PUBLISHED = 1 ]; then echo "PASS contract (gpu_1p0.h publishes §4.1 and both provisional statuses)"
	else cat "$OUT/contract.txt"; echo "FAIL contract"; FAILED+=(contract); fi
fi

# ------------------------------------------------------------------------------------------------ cells
if want cells; then
	EXTRA="-DSUPERSLM_MULTIROW_WIDE -DSUPERSLM_GPU_ALLOC_FAULT_INJECTION -DSUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING"
	[ $PUBLISHED = 1 ] || { EXTRA="$EXTRA -DMR3_UNPUBLISHED"; echo "cells: building with -DMR3_UNPUBLISHED (MR2's internal entry; provisional statuses are sentinels)"; }
	grep -rqs "sslm_gpu_seq_embed_rows" src/gpu/*.cpp || { echo "FAIL cells: src/ defines no sslm_gpu_seq_embed_rows"; exit 1; }
	shaders "$REPO" "$OUT/cells/shaders" wide
	OBJD="$OUT/cells/obj"
	if [ "$(cat "$OBJD.flags" 2>/dev/null)" != "$FLAGS $EXTRA" ]; then rm -rf "$OBJD"; mkdir -p "$OUT/cells"; echo "$FLAGS $EXTRA" >"$OBJD.flags"; fi
	build "$REPO" "$OBJD" "$EXTRA" "$OUT/cells/mr3_cells.exe" tests/multirow-gpu/mr3_cells.cpp
	if [ "$RUN" = 1 ]; then
		OUT_SAVE="$OUT"; OUT="$REPO/out/mr2"; FIX="${MR3_FIXTURES:-$REPO/out/mr2/fixtures}"; fixtures; OUT="$OUT_SAVE"
		wine_up
		python3 "$HERE/mr2_fixtures.py" "$FIX" >/dev/null
		[ -f "$FIX/pkv_def_r4.adp" ] || python3 tools/gen_multirow_adapter_fixture.py "$FIX/pkv_def.sslm" "$FIX/pkv_def_r4.adp" >/dev/null
		for m in pkv_def pkv_qk; do
			set +e
			WINEPREFIX="$WINEPREFIX-$m" "$WINE" "$OUT/cells/mr3_cells.exe" --model="$FIX/$m.sslm" --fixtures="$FIX" --adapter="$FIX/pkv_def_r4.adp" \
				--cycles="${MR3_CYCLES:-1000}" --race-iterations="${MR3_RACES:-10000}" \
				${MR3_CELLS:+--cells=$MR3_CELLS} >"$OUT/cells-$m.log" 2>&1
			rc=$?
			set -e
			grep -E "^(FAIL|RED|cell)|cells:" "$OUT/cells-$m.log" | tr -d '\r' | head -80 | sed 's/^/    /'
			if [ $rc = 0 ]; then echo "PASS cells $m"; else echo "FAIL cells $m (exit $rc, $OUT/cells-$m.log)"; FAILED+=("cells-$m"); fi
		done
	fi
fi

# ------------------------------------------------------------------------------------------------ 9.2
if want b92; then
	BT="$REPO/out/mr2/base-tree"
	if [ ! -d "$BT/src" ]; then
		git worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
		git worktree add --detach "$BT" "$BASE" >/dev/null
	fi
	[ "$(git -C "$BT" rev-parse HEAD)" = "$(git rev-parse "$BASE^{commit}")" ] || { echo "FAIL b92: $BT is not $BASE"; exit 1; }
	mkdir -p "$BT/tests/multirow-gpu"
	cp "$HERE/mr3_92_blobs.cpp" "$HERE/mingw_d3d12_compat.h" "$BT/tests/multirow-gpu/"
	shaders "$BT" "$BT/out/b92/shaders"
	build "$BT" "$BT/out/b92/obj" "" "$BT/out/b92/mr3_92_blobs.exe" tests/multirow-gpu/mr3_92_blobs.cpp
	shaders "$REPO" "$OUT/b92/shaders" wide
	B92X="-DSUPERSLM_MULTIROW_WIDE -DMR3_BRANCH"
	if [ "$(cat "$OUT/b92/obj.flags" 2>/dev/null)" != "$FLAGS $B92X" ]; then rm -rf "$OUT/b92/obj"; mkdir -p "$OUT/b92"; echo "$FLAGS $B92X" >"$OUT/b92/obj.flags"; fi
	build "$REPO" "$OUT/b92/obj" "$B92X" "$OUT/b92/mr3_92_blobs.exe" tests/multirow-gpu/mr3_92_blobs.cpp
	if [ "$RUN" = 1 ]; then
		OUT_SAVE="$OUT"; OUT="$REPO/out/mr2"; fixtures; OUT="$OUT_SAVE"
		wine_up
		D="$OUT/b92/blobs"; rm -rf "$D"; mkdir -p "$D/tag" "$D/branch"
		MODELS=(--model="$FIX/pkv_def.sslm" --model="$FIX/pkv_qk.sslm")
		run92() { (cd "$1" && WINEPREFIX="$WINEPREFIX-b92" "$WINE" ./mr3_92_blobs.exe "${@:2}") | tr -d '\r'; }
		ok=1
		run92 "$BT/out/b92" "${MODELS[@]}" --write="$D/tag" >"$D/w-tag.txt" || ok=0
		run92 "$OUT/b92" "${MODELS[@]}" --write="$D/branch" >"$D/w-branch.txt" || ok=0
		run92 "$BT/out/b92" "${MODELS[@]}" --read="$D/tag" >"$D/tag-reads-tag.txt" || ok=0
		run92 "$OUT/b92" "${MODELS[@]}" --read="$D/tag" >"$D/branch-reads-tag.txt" || ok=0
		run92 "$BT/out/b92" "${MODELS[@]}" --read="$D/branch" >"$D/tag-reads-branch.txt" || ok=0
		run92 "$OUT/b92" "${MODELS[@]}" --read="$D/branch" >"$D/branch-reads-branch.txt" || ok=0
		[ "$(grep -c '^blob ' "$D/tag-reads-tag.txt")" = 4 ] || { ok=0; echo "    b92: the tag read $(grep -c '^blob ' "$D/tag-reads-tag.txt") of 4 blobs"; }
		for f in branch-reads-tag tag-reads-branch branch-reads-branch; do
			diff -q "$D/tag-reads-tag.txt" "$D/$f.txt" >/dev/null || { ok=0; echo "    b92: $f differs from the tag reading its own blobs"; diff "$D/tag-reads-tag.txt" "$D/$f.txt" | head -6 | sed 's/^/      /'; }
		done
		for b in "$D"/tag/*.blob; do cmp -s "$b" "$D/branch/$(basename "$b")" || { ok=0; echo "    b92: $(basename "$b") written through embed_rows differs from the tag's"; }; done
		if [ $ok = 1 ]; then echo "PASS b92 (9.2 against $BASE: 4 blobs each way)"; else echo "FAIL b92 ($D)"; FAILED+=(b92); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ 3.3 TSan
if want tsan; then
	mkdir -p "$OUT/tsan"
	if g++ -std=c++20 -O1 -g -fsanitize=thread -Isrc/gpu -Iinclude "$HERE/mr3_tsan.cpp" -o "$OUT/tsan/mr3_tsan" 2>"$OUT/tsan/build.log"; then
		if [ "$RUN" = 1 ]; then
			set +e
			TSAN_OPTIONS="halt_on_error=1 exitcode=66" "$OUT/tsan/mr3_tsan" >"$OUT/tsan/run.log" 2>&1
			rc=$?
			set -e
			sed 's/^/    /' "$OUT/tsan/run.log" | grep -E "PASS|RED|WARNING: ThreadSanitizer" | head -5
			if [ $rc = 0 ] && ! grep -q "ThreadSanitizer" "$OUT/tsan/run.log"; then echo "PASS tsan"; else echo "FAIL tsan (exit $rc, $OUT/tsan/run.log)"; FAILED+=(tsan); fi
		fi
	else
		echo "FAIL tsan: build ($OUT/tsan/build.log)"; FAILED+=(tsan)
	fi
fi

if want mutants; then
	if bash "$HERE/mr3_mutants.sh" --out="$OUT/mutants" --jobs="$JOBS"; then echo "PASS mutants"; else FAILED+=(mutants); fi
fi

[ "$RUN" = 1 ] || { echo "built"; exit 0; }
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR3 Wine: PASS"; exit 0; fi
echo "MR3 Wine: FAIL (${FAILED[*]})"
exit 1
