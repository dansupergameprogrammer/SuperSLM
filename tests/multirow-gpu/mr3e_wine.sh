#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR3e (plan §16 erratum E-7, the embedder prompt path): the cloud legs of the
# MR3e red suite on the software GPU (Wine + vkd3d-proton + lavapipe). Record: build/red-suite-mr3e.md.
#
# Usage: tests/multirow-gpu/mr3e_wine.sh [--out=DIR] [--jobs=N] [--base=REV] [--only=LEG[,LEG...]] [--build-only] [--clean]
#   --out    default <repo>/out/mr3e
#   --base   the head before MR3e, which "unchanged" is graded against (default MR3E_BASE, else ae6debb3)
#   --only   cells | base | abi | mutants | optout   (default: all five)
#   --clean  delete the build trees, the base worktree and the Wine prefixes afterwards (keeps the logs)
# Fixtures (MR3E_FIXTURES, default out/mr3e/fixtures): pkv_def and pkv_qk (MR_FIXTURES or the MR1 smoke's, else
# tools/gen_paged_kv_fixture.py), and pkv_def_refuse from mr2_fixtures.py.
#
# Legs (each printed PASS or FAIL; exit 0 only when every leg run passes):
#   cells    mr3e_cells.exe: sweep (E-7 a) and refusal (E-7 b), each graded in-process against the one-row path.
#   base     the same executable built from --base's tree. Every `obs` line (status, blob, final hidden, context,
#            ready flag for every sweep and refusal case) must equal the branch's.
#   abi      E-7 (d): gpu_1p0.h declares the same functions and the same struct bodies as --base's.
#   mutants  MXE2, the tail row dropped: in a pass with fewer rows than its variant (k < H), the last real row is
#            treated as padding. The sweep must go red. It proves the bridge runs multi-row passes and that the sweep
#            sees a lost tail row.
#            MXE1, the control: the bridge's pass height pinned at 1. The sweep must stay green, which proves the
#            sweep compares bytes, not speed.
#   optout   E-7 (c): with the opt-out set (SSLM_GPU_MULTIROW_PROMPT=0), the sweep is green on the branch AND on
#            MXE2's binary. The opt-out bypasses the multi-row code, so a broken tail cannot reach it. MXE2 without
#            the opt-out must be red, or this leg proves nothing.
#
# CONTRACT FOR THE BUILDER (the test author's proposal; each is one edit here):
#   - the opt-out is the process environment variable SSLM_GPU_MULTIROW_PROMPT=0 (no verb, no ABI change);
#   - gpu_1p0.cpp defines, with internal linkage, `uint32_t PromptRowsCap(...)` returning the bridge's pass height H
#     in 1..SSLM_GPU_MAX_ROWS (1 under the opt-out), and the bridge's passes go through the row-chunk slices
#     (`slice.real_rows = c.rows;`, the site MXE2 mutates).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
# shellcheck source=../../tools/cloud_gpu/wine_gpu.sh
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"

OUT="$REPO/out/mr3e"
JOBS=2
RUN=1
CLEAN=0
ONLY="cells,base,abi,mutants,optout"
BASE="${MR3E_BASE:-ae6debb3}"
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--base=*) BASE="${a#--base=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--build-only) RUN=0 ;;
		--clean) CLEAN=1 ;;
		*) sed -n '2,31p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
mkdir -p "$OUT"
export WINEPREFIX="${WINEPREFIX_MR3E:-$OUT/wineprefix-$$}"
BT="$OUT/base-tree"
cleanup() {
	if [ -z "${WINEPREFIX_MR3E:-}" ]; then
		for p in "$WINEPREFIX" "$WINEPREFIX"-*; do
			[ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }
		done
	fi
	if [ "$CLEAN" = 1 ]; then
		rm -rf "$OUT/cells/obj" "$OUT/cells/shaders" "$OUT/mutants"
		[ -d "$BT" ] && { git worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"; git worktree prune; }
	fi
	true
}
trap cleanup EXIT
want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
FLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
EXTRA="-DSUPERSLM_MULTIROW_WIDE -DSUPERSLM_GPU_ALLOC_FAULT_INJECTION"
# shellcheck source=mr_lib.sh
. "$HERE/mr_lib.sh"
F3E="${MR3E_FIXTURES:-$OUT/fixtures}"
fixtures3e() {
	FIX="$F3E"; fixtures
	[ -f "$F3E/pkv_def_refuse.sslm" ] || python3 "$HERE/mr2_fixtures.py" "$F3E" >/dev/null
}
CELLS="$OUT/cells"
run_cells() {  # run_cells DIR LOG [ENV=VAL] -- ARGS...: the executable in DIR, its output in LOG (CR stripped); its exit code
	local dir="$1" log="$2" envs=()
	shift 2
	while [ "$1" != "--" ]; do envs+=("$1"); shift; done
	shift
	set +e
	(cd "$dir" && env "${envs[@]}" WINEPREFIX="$WINEPREFIX-run" "$WINE" ./mr3e_cells.exe --fixtures="$F3E" "$@") 2>&1 | tr -d '\r' >"$log"
	local rc=${PIPESTATUS[0]}
	set -e
	grep -q '^cells: ' "$log" || rc=99  # no verdict line: the run crashed
	return $rc
}

# ------------------------------------------------------------------------------------------------ build
need_cells=0
for l in cells mutants optout; do want $l && need_cells=1; done
if [ $need_cells = 1 ]; then
	shaders "$REPO" "$CELLS/shaders" wide
	if [ "$(cat "$CELLS/obj.flags" 2>/dev/null)" != "$FLAGS $EXTRA" ]; then rm -rf "$CELLS/obj"; mkdir -p "$CELLS"; echo "$FLAGS $EXTRA" >"$CELLS/obj.flags"; fi
	build "$REPO" "$CELLS/obj" "$EXTRA" "$CELLS/mr3e_cells.exe" tests/multirow-gpu/mr3e_cells.cpp
fi
if [ "$RUN" = 1 ]; then fixtures3e; wine_up; fi

# ------------------------------------------------------------------------------------------------ cells
if want cells && [ "$RUN" = 1 ]; then
	rc=0; run_cells "$CELLS" "$OUT/cells.log" -- --print-obs || rc=$?
	grep -E "^(FAIL|RED)|cell |cells:|refusal:" "$OUT/cells.log" | head -60 | sed 's/^/    /'
	grep '^obs ' "$OUT/cells.log" >"$OUT/obs-branch.txt" || true
	if [ $rc = 0 ]; then echo "PASS cells"; else echo "FAIL cells (exit $rc, $OUT/cells.log)"; FAILED+=(cells); fi
fi

# ------------------------------------------------------------------------------------------------ base
if want base; then
	if [ ! -d "$BT/src" ] || [ "$(git -C "$BT" rev-parse HEAD 2>/dev/null)" != "$(git rev-parse "$BASE^{commit}")" ]; then
		git worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
		git worktree add --detach "$BT" "$BASE" >/dev/null
	fi
	mkdir -p "$BT/tests/multirow-gpu"
	cp "$HERE/mr3e_cells.cpp" "$HERE/mr2_common.h" "$HERE/mr2_entry.h" "$HERE/mingw_d3d12_compat.h" "$BT/tests/multirow-gpu/"
	shaders "$BT" "$BT/out/m3e/shaders" wide
	build "$BT" "$BT/out/m3e/obj" "$EXTRA" "$BT/out/m3e/mr3e_cells.exe" tests/multirow-gpu/mr3e_cells.cpp
	if [ "$RUN" = 1 ]; then
		[ -f "$OUT/obs-branch.txt" ] || { run_cells "$CELLS" "$OUT/cells.log" -- --print-obs || true; grep '^obs ' "$OUT/cells.log" >"$OUT/obs-branch.txt" || true; }
		ok=1
		run_cells "$BT/out/m3e" "$OUT/base.log" -- --print-obs || true
		grep '^obs ' "$OUT/base.log" >"$OUT/obs-base.txt" || true
		nb="$(wc -l <"$OUT/obs-base.txt")"; nr="$(wc -l <"$OUT/obs-branch.txt")"
		[ "$nb" -gt 300 ] || { ok=0; echo "    base: the base binary printed only $nb obs lines ($OUT/base.log)"; }
		if ! diff -q "$OUT/obs-base.txt" "$OUT/obs-branch.txt" >/dev/null; then
			ok=0; echo "    base: $(diff "$OUT/obs-base.txt" "$OUT/obs-branch.txt" | grep -c '^>') of $nr obs lines differ from $(git rev-parse --short=8 "$BASE")'s:"
			diff "$OUT/obs-base.txt" "$OUT/obs-branch.txt" | head -8 | cut -c1-240 | sed 's/^/      /'
		fi
		if [ $ok = 1 ]; then echo "PASS base ($nr obs lines equal to $(git rev-parse --short=8 "$BASE")'s binary)"; else echo "FAIL base"; FAILED+=(base); fi
	fi
fi

# ------------------------------------------------------------------------------------------------ abi
if want abi; then
	if git show "$BASE:include/superslm/gpu_1p0.h" >"$OUT/gpu_1p0.base.h" 2>/dev/null &&
		python3 -I - "$OUT/gpu_1p0.base.h" include/superslm/gpu_1p0.h <<'PY'
import re, sys
def surface(path):
    src = re.sub(r"/\*.*?\*/", "", open(path).read(), flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    flat = re.sub(r"\s+", " ", src)
    funcs = {}
    for m in re.finditer(r"([A-Za-z_][\w\s\*]*?)\b((?:sslm|Sslm)\w+)\s*\(([^;{)]*)\)\s*(noexcept)?\s*;", flat):
        funcs[m.group(2)] = re.sub(r"\s+", " ", (m.group(1) + "(" + m.group(3) + ")").strip())
    structs = {m.group(1): m.group(2).strip() for m in re.finditer(r"typedef struct (\w+) \{(.*?)\}", flat)}
    return funcs, structs
bf, bs = surface(sys.argv[1]); nf, ns = surface(sys.argv[2])
bad = ["new public function " + f for f in sorted(set(nf) - set(bf))]
bad += ["removed public function " + f for f in sorted(set(bf) - set(nf))]
bad += ["signature changed: " + f for f in sorted(set(bf) & set(nf)) if bf[f] != nf[f]]
bad += ["struct changed: " + s for s in sorted(set(bs) | set(ns)) if bs.get(s) != ns.get(s)]
for b in bad: print("    abi: " + b)
print("    abi: %d functions, %d structs compared" % (len(nf), len(ns)))
sys.exit(1 if bad else 0)
PY
	then echo "PASS abi (gpu_1p0.h: no new verb, no signature or struct change against $(git rev-parse --short=8 "$BASE"))"; else echo "FAIL abi"; FAILED+=(abi); fi
fi

# ------------------------------------------------------------------------------------------------ mutants
G=src/gpu/gpu_1p0.cpp
MUT="$OUT/mutants"
# host_mutant NAME FILE COUNT PATTERN REPLACEMENT: FILE in a copy of the tree with every match of the Python regex
# PATTERN (exactly COUNT of them) replaced by REPLACEMENT; its translation unit rebuilt with the cells' flags; relinked.
host_mutant() {
	local name="$1" file="$2" count="$3" d="$MUT/$1"
	rm -rf "$d"; mkdir -p "$d/tree"
	cp "$CELLS/d3d12.dll" "$CELLS/d3d12core.dll" "$d/"; cp -r "$CELLS/shaders" "$d/shaders"; cp -a "$CELLS/obj" "$d/obj"
	cp -a include src tests "$d/tree/"
	if ! MX_PAT="$4" MX_NEW="$5" MX_N="$count" python3 -I -c "import os,re,sys; p=sys.argv[1]; s=open(p).read()
s2,n=re.subn(os.environ['MX_PAT'], os.environ['MX_NEW'], s, flags=re.M)
if n!=int(os.environ['MX_N']): sys.exit('%d matches of the site in %s, want %s' % (n, p, os.environ['MX_N']))
open(p,'w').write(s2)" "$d/tree/$file"; then
		echo "    $name: its site is not in $file as the contract states (red-unimplemented until the builder writes it)"
		rm -rf "$d"; return 1
	fi
	# shellcheck disable=SC2086
	if ! (cd "$d/tree" && $CXX $FLAGS $EXTRA -w -c "$file" -o "$d/obj/$(echo "$file" | tr / _).o") >"$d/build.log" 2>&1 ||
		! "$CXX" -O2 "$d/obj"/*.o -o "$d/mr3e_cells.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc \
			-static-libstdc++ >>"$d/build.log" 2>&1; then
		echo "    $name: build failed ($d/build.log)"; return 1
	fi
	rm -rf "$d/tree" "$d/obj"
}
MXE2_RED=0
if want mutants || want optout; then
	mkdir -p "$MUT"
	have_mxe2=0
	host_mutant MXE2 $G 1 '^(\s*)slice\.real_rows = c\.rows;' \
		'\1slice.real_rows = (c.rows > 1 && c.rows < c.variant_h) ? c.rows - 1 : c.rows;  // MXE2: the tail row dropped' && have_mxe2=1
	if [ $have_mxe2 = 1 ] && [ "$RUN" = 1 ]; then
		rc=0; run_cells "$MUT/MXE2" "$MUT/MXE2.log" -- --cells=sweep --quick || rc=$?
		[ $rc != 0 ] && MXE2_RED=1
	fi
fi
if want mutants && [ "$RUN" = 1 ]; then
	if [ $MXE2_RED = 1 ]; then echo "PASS mutant MXE2 -> red in sweep ($(grep -m1 '^RED cell' "$MUT/MXE2.log" || echo crashed))"
	else [ "${have_mxe2:-0}" = 1 ] && echo "    MXE2: the sweep stayed green: the bridge runs no multi-row pass with a tail"; echo "FAIL mutant MXE2 survived"; FAILED+=(MXE2); fi
	if host_mutant MXE1 $G 1 '^((?:static\s+)?(?:inline\s+)?uint32_t\s+PromptRowsCap\s*\([^)]*\)\s*(?:noexcept\s*)?\{)' \
		'\1\n\treturn 1u;  // MXE1: the bridge pinned at H = 1 (the control)'; then
		rc=0; run_cells "$MUT/MXE1" "$MUT/MXE1.log" -- --cells=sweep --quick || rc=$?
		if [ $rc = 0 ]; then echo "PASS control MXE1 (H = 1): the sweep stays green"; else echo "FAIL control MXE1: the sweep went red at H = 1 ($MUT/MXE1.log)"; FAILED+=(MXE1); fi
	else echo "FAIL control MXE1 (no site)"; FAILED+=(MXE1); fi
fi

# ------------------------------------------------------------------------------------------------ opt-out
if want optout && [ "$RUN" = 1 ]; then
	ok=1
	rc=0; run_cells "$CELLS" "$OUT/optout.log" SSLM_GPU_MULTIROW_PROMPT=0 -- --cells=sweep --quick || rc=$?
	[ $rc = 0 ] || { ok=0; echo "    optout: the sweep with the opt-out set is red on the branch ($OUT/optout.log)"; }
	if [ -x "$MUT/MXE2/mr3e_cells.exe" ]; then
		rc=0; run_cells "$MUT/MXE2" "$OUT/optout-mxe2.log" SSLM_GPU_MULTIROW_PROMPT=0 -- --cells=sweep --quick || rc=$?
		[ $rc = 0 ] || { ok=0; echo "    optout: with the opt-out set, MXE2's broken tail still reached the bridge ($OUT/optout-mxe2.log)"; }
		[ $MXE2_RED = 1 ] || { ok=0; echo "    optout: MXE2 without the opt-out is green too, so the opt-out is not shown to change the path"; }
	else
		ok=0; echo "    optout: no MXE2 binary"
	fi
	if [ $ok = 1 ]; then echo "PASS optout (SSLM_GPU_MULTIROW_PROMPT=0 forces the one-row path: MXE2 red without it, green with it)"; else echo "FAIL optout"; FAILED+=(optout); fi
fi

[ "$RUN" = 1 ] || { echo "built"; exit 0; }
if [ ${#FAILED[@]} -eq 0 ]; then echo "MR3e Wine: PASS"; exit 0; fi
echo "MR3e Wine: FAIL (${FAILED[*]})"
exit 1
