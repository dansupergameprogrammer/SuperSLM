#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR4 (on 1.17): the cloud legs of the MR4 red suite, on the software GPU (Wine +
# vkd3d-proton + lavapipe). The box parts (10.2's measured table, 10.5 Canonical's frames, 8.3 on B2's spans at the
# 0.5B) are MB4's; see red-suite-mr4.md.
#
# Usage: tests/multirow-gpu/mr4_wine.sh [--out=DIR] [--jobs=N] [--base=REV] [--only=LEG[,LEG...]] [--build-only]
#   --out   default <repo>/out/mr4
#   --base  1.17's revision 8.3 compares against (default MR_BASE_TAG, else the merge base with origin/claude/sfd-gpu-1p17)
#   --only  contract | cells | c83 | table | mutants   (default: contract,cells,c83,table)
# Fixtures: F-SFD (tools/gen_schema_fast_decode_fixture.py) and fsfd_refuse (mr4_fixtures.py) in out/mr4/fixtures (or
# MR4_FIXTURES), made when absent; pkv_def.sslm (the no-table fixture of 2.3) from MR2's fixtures.
#
# Legs (each printed PASS or FAIL; exit 0 only when every leg run passes):
#   contract  gpu_1p0.h admits SCHEMA_FORCED and FIRST_PREWALKED (neither still marked a reserved bad spec).
#   cells     mr4_cells.exe: 2.3, 7.9, 8.10 (MR4_CELLS=2.3 runs a subset).
#   c83       8.3: mr4_83_canon.exe built from 1.17's tree (--base) and from the branch; the canonical loop's lines (tokens,
#             blob, walk, ready flag, context, at chunk budgets 1..8 and 64, dedicated and pooled, four schemas) are equal,
#             every budget gives the same line within a binary, and potion_shop_order's stream is the CPU's pinned one.
#   table     C18's table as data (rulings item 12): mr4_table.cpp compiled against the header the record names.
#   mutants   MX16 (mr4_mutants.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT="$REPO/out/mr4"
JOBS=2
RUN=1
ONLY="contract,cells,c83,table"
BASE="${MR_BASE_TAG:-}"
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--base=*) BASE="${a#--base=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--build-only) RUN=0 ;;
		*) sed -n '2,26p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
[ -n "$BASE" ] || BASE="$(git merge-base HEAD origin/claude/sfd-gpu-1p17)"
export WINEPREFIX="${WINEPREFIX_MR4:-$OUT/wineprefix-$$}"
if [ -z "${WINEPREFIX_MR4:-}" ]; then
	trap 'for p in "$WINEPREFIX" "$WINEPREFIX"-*; do [ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }; done; true' EXIT
fi
mkdir -p "$OUT"
want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
# shellcheck source=mr_lib.sh
. "$HERE/mr_lib.sh"
F4="${MR4_FIXTURES:-$OUT/fixtures}"
fixtures4() {
	mkdir -p "$F4"
	[ -f "$F4/fsfd.sslm" ] || python3 tools/gen_schema_fast_decode_fixture.py "$F4" >/dev/null
	[ -f "$F4/fsfd_refuse.sslm" ] && [ "$F4/fsfd_refuse.sslm" -nt "$HERE/mr4_fixtures.py" ] || python3 "$HERE/mr4_fixtures.py" "$F4" >/dev/null
	if [ ! -f "$F4/pkv_def.sslm" ]; then
		local OUT_SAVE="$OUT"; OUT="$REPO/out/mr2"; fixtures; OUT="$OUT_SAVE"
		cp "$REPO/out/mr2/fixtures/pkv_def.sslm" "$F4/"
	fi
}

# ------------------------------------------------------------------------------------------------ contract
if want contract; then
	if python3 -I - "$REPO/include/superslm/gpu_1p0.h" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
miss = []
for sym in ("SSLM_GPU_ROWS_SCHEMA_FORCED", "SSLM_GPU_ROWS_FIRST_PREWALKED"):
    lines = [l for l in src.split("\n") if sym in l and ("= 1" in l or "#define" in l)]
    if not lines: miss.append("declares no " + sym)
    elif any("bad spec" in l.lower() or "reserved" in l.lower() for l in lines): miss.append(sym + " is still marked reserved (a bad spec)")
for x in miss: print("    contract: " + x)
sys.exit(1 if miss else 0)
PY
	then echo "PASS contract (SCHEMA_FORCED and FIRST_PREWALKED admitted in gpu_1p0.h)"; else echo "FAIL contract"; FAILED+=(contract); fi
fi

# ------------------------------------------------------------------------------------------------ cells
if want cells; then
	EXTRA="-DSUPERSLM_MULTIROW_WIDE"
	shaders "$REPO" "$OUT/cells/shaders" wide
	OBJD="$OUT/cells/obj"
	if [ "$(cat "$OBJD.flags" 2>/dev/null)" != "$FLAGS $EXTRA" ]; then rm -rf "$OBJD"; mkdir -p "$OUT/cells"; echo "$FLAGS $EXTRA" >"$OBJD.flags"; fi
	build "$REPO" "$OBJD" "$EXTRA" "$OUT/cells/mr4_cells.exe" tests/multirow-gpu/mr4_cells.cpp
	if [ "$RUN" = 1 ]; then
		fixtures4
		wine_up
		set +e
		(cd "$OUT/cells" && WINEPREFIX="$WINEPREFIX-cells" "$WINE" ./mr4_cells.exe --fixtures="$F4" ${MR4_CELLS:+--cells=$MR4_CELLS}) >"$OUT/cells.log" 2>&1
		rc=$?
		set -e
		grep -E "^(FAIL|RED|7\.9:)|cell |cells:" "$OUT/cells.log" | tr -d '\r' | head -80 | sed 's/^/    /'
		if [ $rc = 0 ]; then echo "PASS cells"; else echo "FAIL cells (exit $rc, $OUT/cells.log)"; FAILED+=(cells); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ 8.3
if want c83; then
	BT="$OUT/base-tree"
	if [ ! -d "$BT/src" ] || [ "$(git -C "$BT" rev-parse HEAD 2>/dev/null)" != "$(git rev-parse "$BASE^{commit}")" ]; then
		git worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
		git worktree add --detach "$BT" "$BASE" >/dev/null
	fi
	mkdir -p "$BT/tests/multirow-gpu"
	cp "$HERE/mr4_83_canon.cpp" "$HERE/mingw_d3d12_compat.h" "$BT/tests/multirow-gpu/"
	shaders "$BT" "$BT/out/c83/shaders"
	build "$BT" "$BT/out/c83/obj" "" "$BT/out/c83/mr4_83_canon.exe" tests/multirow-gpu/mr4_83_canon.cpp
	shaders "$REPO" "$OUT/c83/shaders" wide
	build "$REPO" "$OUT/c83/obj" "" "$OUT/c83/mr4_83_canon.exe" tests/multirow-gpu/mr4_83_canon.cpp
	if [ "$RUN" = 1 ]; then
		fixtures4
		wine_up
		ok=1
		(cd "$BT/out/c83" && WINEPREFIX="$WINEPREFIX-c83" "$WINE" ./mr4_83_canon.exe --fixtures="$F4") | tr -d '\r' >"$OUT/c83-base.txt" || ok=0
		(cd "$OUT/c83" && WINEPREFIX="$WINEPREFIX-c83" "$WINE" ./mr4_83_canon.exe --fixtures="$F4") | tr -d '\r' >"$OUT/c83-branch.txt" || ok=0
		n="$(grep -c '^canon ' "$OUT/c83-base.txt" || true)"
		[ "$n" = 56 ] || { ok=0; echo "    c83: 1.17's binary printed $n of 56 lines"; }
		if ! diff -q "$OUT/c83-base.txt" "$OUT/c83-branch.txt" >/dev/null; then
			ok=0; echo "    c83: the branch's canonical loop differs from 1.17's ($(diff "$OUT/c83-base.txt" "$OUT/c83-branch.txt" | grep -c '^>') lines):"
			diff "$OUT/c83-base.txt" "$OUT/c83-branch.txt" | head -8 | sed 's/^/      /'
		fi
		# Chunk invariance within the branch (SFD §5.6): every budget gives the same line per schema and kind.
		inv="$(sed -E 's/ budget [0-9]+ / /' "$OUT/c83-branch.txt" | sort -u | grep -c '^canon ' || true)"
		[ "$inv" = 8 ] || { ok=0; echo "    c83: the branch's lines vary with the chunk budget ($inv distinct of 8 schema x kind)"; }
		# The CPU's pinned POTION stream (1.17's G7.1 digest, tests/schema-fast-decode-gpu/sfdg_helpers.h).
		grep '^canon potion_shop_order ' "$OUT/c83-branch.txt" | grep -qv 'tokens 24 5498cef23900bfbfad790d7174b6badc04bddc58afdca629daf8e2d0b48488bd ' &&
			{ ok=0; echo "    c83: potion_shop_order's stream is not the CPU's pinned one"; }
		if [ $ok = 1 ]; then echo "PASS c83 (8.3: 56 canonical-loop lines equal 1.17's ($(git rev-parse --short=8 "$BASE")), chunk-invariant, POTION = the CPU's pin)"
		else echo "FAIL c83 ($OUT/c83-*.txt)"; FAILED+=(c83); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ C18's table
if want table; then
	mkdir -p "$OUT/table"
	if g++ -std=c++20 -Iinclude -Isrc "$HERE/mr4_table.cpp" -o "$OUT/table/mr4_table" 2>"$OUT/table/build.log"; then
		if "$OUT/table/mr4_table" | sed 's/^/    /'; then echo "PASS table"; else echo "FAIL table"; FAILED+=(table); fi
	else
		grep -m3 "error" "$OUT/table/build.log" | sed 's/^/    /'
		echo "FAIL table: red-unimplemented (the C18 table header mr4_table.cpp names does not compile; $OUT/table/build.log)"; FAILED+=(table)
	fi
fi

if want mutants; then
	if bash "$HERE/mr4_mutants.sh" --out="$OUT/mutants" --jobs="$JOBS"; then echo "PASS mutants"; else FAILED+=(mutants); fi
fi

[ "$RUN" = 1 ] || { echo "built"; exit 0; }
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR4 Wine: PASS"; exit 0; fi
echo "MR4 Wine: FAIL (${FAILED[*]})"
exit 1
