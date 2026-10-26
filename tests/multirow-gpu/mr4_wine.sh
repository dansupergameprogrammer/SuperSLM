#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR4 (on 1.17): the cloud legs of the MR4 red suite, on the software GPU (Wine +
# vkd3d-proton + lavapipe). The box parts (10.2's measured table, 10.5 Canonical's frames, 8.3 on B2's spans at the
# 0.5B) are MB4's; see red-suite-mr4.md.
#
# Usage: tests/multirow-gpu/mr4_wine.sh [--out=DIR] [--jobs=N] [--base=REV] [--only=LEG[,LEG...]] [--build-only] [--fixtures-only]
#   --out   default <repo>/out/mr4
#   --base  1.17's revision 8.3 compares against (default MR_BASE_TAG, else the merge base with origin/claude/sfd-gpu-1p17,
#           else v1.17.0)
#   --only  census | contract | cells | c83 | table | sfdg | mutants   (default: census,cells,c83,table,sfdg; contract RETIRED)
#   --fixtures-only  make the fixtures below and exit (mr4_mutants.sh uses it)
# Fixtures: all three in <out>/fixtures (MR4_FIXTURES overrides), made when absent, so a fresh clone with no out/ needs
# nothing: F-SFD (tools/gen_schema_fast_decode_fixture.py), fsfd_refuse (mr4_fixtures.py), and pkv_def.sslm (2.3's
# no-table fixture: copied from MR_FIXTURES or out/mr1-smoke/fixtures when there, else tools/gen_paged_kv_fixture.py,
# ~30 min of CPU once).
#
# Legs (each printed PASS or FAIL; exit 0 only when every leg run passes):
#   census    tests/ci/check_gpu_status_site_census.py: every status and catch site in src/gpu is in te425's list.
#   contract  gpu_1p0.h admits SCHEMA_FORCED and FIRST_PREWALKED (neither still marked a reserved bad spec).
#   cells     mr4_cells.exe: 7.9, 4.6j, 4.6e, 4.6f, n1 (MR4_CELLS=7.9 runs a subset; 2.3, 8.10, s1 retired at MRX).
#   c83       8.3: mr4_83_canon.exe built from 1.17's tree (--base) and from the branch; the canonical loop's lines (tokens,
#             blob, walk, ready flag, context, at chunk budgets 1..8 and 64, dedicated and pooled, four schemas) are equal,
#             every budget gives the same line within a binary, and potion_shop_order's stream is the CPU's pinned one.
#   table     C18's table as data (rulings item 12): mr4_table.cpp compiled against the header the record names.
#   sfdg      MR4 review S4, ruling 18: 1.17's 15 SFD-GPU cells (tests/schema-fast-decode-gpu: G0.1 ... G12.10, the
#             jump verb's G12.x among them), built with build_sfd_gpu_mingw.sh from 1.17's tree (--base) and from the
#             branch, each run under Wine on F-SFD. Graded: the branch's output equals the base binary's line for line
#             (Wine noise and the per-run durations removed). Not a pass/fail claim on the cells, which are box-only:
#             some fail on the base itself under Wine (review N3), and must fail the same way on the branch.
#             MRX re-baseline: the jump-driving cells' divergence must equal mr4_sfdg_mrx_<base>.diff exactly.
#   mutants   MX16 (mr4_mutants.sh).
# MRX (E-11, ruling 20), marked, not deleted (red-suite-mrx.md): the contract leg RETIRES (it requires
# SSLM_GPU_ROWS_SCHEMA_FORCED, which MRX removes: mrx_cpu.sh's header leg requires the opposite); c83 RE-POINTS to an
# Exact loop at MR5 (opts NULL means Exact, refused on the GPU until then); cells: see mr4_cells.cpp's MRX block; the
# 10.5 Canonical frames (mr4_box.py canon105) retire or re-point to Exact with MR6.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT="$REPO/out/mr4"
JOBS=2
RUN=1
ONLY="census,cells,c83,table,sfdg"  # MRX (ruling 22): contract RETIRED from the defaults
BASE="${MR_BASE_TAG:-}"
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--base=*) BASE="${a#--base=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--build-only) RUN=0 ;;
		--fixtures-only) ONLY=fixtures ;;
		*) sed -n '2,34p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
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
	# pkv_def (2.3's no-table fixture): mr_lib's fixtures() into F4 itself, named as its argument -- copied from
	# MR_FIXTURES or the MR1 smoke's directory when there, else generated (tools/gen_paged_kv_fixture.py). It used to
	# copy from out/mr2, which nothing had built on a fresh clone (MR4 re-check N6), and then leaned on mr_lib's FIX,
	# which mr_lib computed from OUT when it was sourced (the builder's runner nit).
	[ -f "$F4/pkv_def.sslm" ] || fixtures "$F4"
	[ -f "$F4/pkv_def.sslm" ] || { echo "FAIL fixtures: no pkv_def.sslm in $F4"; return 1; }
}

if [ "$ONLY" = fixtures ]; then
	if fixtures4; then echo "fixtures: $F4 ($(cd "$F4" && ls fsfd.sslm fsfd_refuse.sslm pkv_def.sslm | tr '\n' ' '))"; exit 0; fi
	exit 1
fi

# ------------------------------------------------------------------------------------------------ census
if want census; then
	if python3 tests/ci/check_gpu_status_site_census.py >"$OUT/census.log" 2>&1; then echo "PASS census ($(tail -1 "$OUT/census.log"))"
	else { sed 's/^/    /' "$OUT/census.log" | head -20; } || true; echo "FAIL census"; FAILED+=(census); fi
fi

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
		{ grep -E "^(FAIL|RED|7\.9:)|cell |cells:" "$OUT/cells.log" | tr -d '\r' | head -80 | sed 's/^/    /'; } || true
		if [ $rc = 0 ]; then echo "PASS cells"; else echo "FAIL cells (exit $rc, $OUT/cells.log)"; FAILED+=(cells); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ 8.3
BT="$OUT/base-tree"
base_tree() {
	# The base, resolved only by the legs that build from it (c83, sfdg), so census, cells, table and --fixtures-only run
	# on a clone without that branch. Without origin/claude/sfd-gpu-1p17 (a single-branch clone) it is the v1.17.0 tag
	# (ruling 20's base).
	if [ -z "$BASE" ]; then
		BASE="$(git merge-base HEAD origin/claude/sfd-gpu-1p17 2>/dev/null || true)"
		[ -n "$BASE" ] || { BASE=v1.17.0; echo "    base: no origin/claude/sfd-gpu-1p17; using $BASE"; }
	fi
	if [ ! -d "$BT/src" ] || [ "$(git -C "$BT" rev-parse HEAD 2>/dev/null)" != "$(git rev-parse "$BASE^{commit}")" ]; then
		git worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
		git worktree add --detach "$BT" "$BASE" >/dev/null
	fi
}
if want c83; then
	base_tree
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
			{ diff "$OUT/c83-base.txt" "$OUT/c83-branch.txt" | head -8 | sed 's/^/      /'; } || true
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
		{ grep -m3 "error" "$OUT/table/build.log" | sed 's/^/    /'; } || true
		echo "FAIL table: red-unimplemented (the C18 table header mr4_table.cpp names does not compile; $OUT/table/build.log)"; FAILED+=(table)
	fi
fi

# ------------------------------------------------------------------------------------------------ 1.17's SFD-GPU cells
SFDG_CELLS="sfdg_smoke sfdg_cell_g0_1 sfdg_cell_g2_1 sfdg_cell_g12_4 sfdg_cell_g12_5 sfdg_cell_g5_1 sfdg_cell_g6 sfdg_cell_g8_1
sfdg_cell_g7_1 sfdg_cell_g9_1 sfdg_cell_g11_1 sfdg_cell_g12_verb sfdg_cell_g12_6 sfdg_cell_g12_7 sfdg_cell_g12_10"
# sfdg_run TREE DIR WIDE LOG: build the 15 cells from TREE into DIR (their own mingw script; it supplies its own D3D12
# stand-in, so not through mingw_cxx.sh), stage the shaders and DLLs, and run each on F-SFD, all output into LOG.
sfdg_run() {
	local tree="$1" dir="$2" wide="$3" log="$4" c rc
	# shellcheck disable=SC2086
	if ! (cd "$tree" && CXX= bash tests/schema-fast-decode-gpu/build_sfd_gpu_mingw.sh --out="$dir" --jobs="$JOBS" $SFDG_CELLS) \
		>"$dir.build.log" 2>&1; then
		echo "    sfdg: the build failed in $tree ($dir.build.log)"; return 1
	fi
	shaders "$tree" "$dir/shaders" $wide
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$dir/"
	: >"$log"
	for c in $SFDG_CELLS; do
		echo "== $c" >>"$log"
		if [ ! -f "$dir/$c.exe" ]; then echo "NOT BUILT" >>"$log"; continue; fi
		set +e
		(cd "$dir" && WINEPREFIX="$WINEPREFIX-sfdg" timeout 2400 "$WINE" "./$c.exe" --fixture-dir="$F4") 2>/dev/null |
			tr -d '\r' | grep -vE '^(wine|fixme|err:|warn:)' |
			sed -E 's/[(]?[0-9]+([.][0-9]+)? ?(ms|s)[)]?$/<t>/; s/ [0-9]+([.][0-9]+)? ?ms\b/ <t>/g' >>"$log"
		rc=${PIPESTATUS[0]}
		set -e
		echo "exit $rc" >>"$log"
	done
}
if want sfdg; then
	base_tree
	if [ "$RUN" = 1 ]; then
		fixtures4
		wine_up
		mkdir -p "$OUT/sfdg"
		ok=1
		sfdg_run "$BT" "$OUT/sfdg/base" "" "$OUT/sfdg/base.txt" || ok=0
		sfdg_run "$REPO" "$OUT/sfdg/branch" wide "$OUT/sfdg/branch.txt" || ok=0
		# MRX (E-11, ruling 20): re-baselined. NULL options now mean Exact, which the GPU refuses until MR5, so the
		# cells that drive the jump verb (G2.1, G7.1, G9.1, G11.1, G12.1-G12.3/G12.8, G12.6, G12.7, G12.10) diverge from
		# 1.17's by exactly the lines in mr4_sfdg_mrx_<base>.diff (`diff base branch`, recorded at MRX against ff55ae99).
		# The leg grades that the divergence is exactly that file, line for line, and that every other line is equal.
		SFDG_EXPECT="$HERE/mr4_sfdg_mrx_$(git rev-parse --short=8 "$BASE").diff"
		if [ $ok = 1 ]; then
			diff "$OUT/sfdg/base.txt" "$OUT/sfdg/branch.txt" >"$OUT/sfdg/diff.txt" || true
			if [ -f "$SFDG_EXPECT" ] && cmp -s "$OUT/sfdg/diff.txt" "$SFDG_EXPECT"; then
				echo "    sfdg: the branch differs from 1.17's by exactly MRX's recorded $(grep -c '^>' "$SFDG_EXPECT") lines ($(basename "$SFDG_EXPECT"))"
			elif [ -s "$OUT/sfdg/diff.txt" ]; then
				ok=0; echo "    sfdg: the branch's divergence from 1.17's is not MRX's recorded one ($(grep -c '^>' "$OUT/sfdg/diff.txt") lines; $SFDG_EXPECT):"
				{ diff "$SFDG_EXPECT" "$OUT/sfdg/diff.txt" 2>&1 | head -12 | sed 's/^/      /'; } || true
			fi
		fi
		awk '/^== /{c=$2} /^exit /{printf "    %-22s exit %s\n", c, $2}' "$OUT/sfdg/branch.txt"  # both binaries: the same
		if [ $ok = 1 ]; then echo "PASS sfdg (1.17's 15 SFD-GPU cells: the branch's output equals $(git rev-parse --short=8 "$BASE")'s line for line, but for MRX's recorded divergence)"
		else echo "FAIL sfdg ($OUT/sfdg/*.txt)"; FAILED+=(sfdg); fi
	fi
fi

if want mutants; then
	if bash "$HERE/mr4_mutants.sh" --out="$OUT/mutants" --jobs="$JOBS"; then echo "PASS mutants"; else FAILED+=(mutants); fi
fi

[ "$RUN" = 1 ] || { echo "built"; exit 0; }
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR4 Wine: PASS"; exit 0; fi
echo "MR4 Wine: FAIL (${FAILED[*]})"
exit 1
