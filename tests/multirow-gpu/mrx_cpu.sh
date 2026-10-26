#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MRX (plan §16 erratum E-11, ruling 20): Canonical removal, the CPU-only legs.
# Native Linux, no Wine: g++ builds the CPU library from each tree's CMakeLists core sources. mrx_wine.sh runs this
# as its `cpu` leg; it also runs alone. Record: build/red-suite-mrx.md.
#
# Usage: tests/multirow-gpu/mrx_cpu.sh [--out=DIR] [--jobs=N] [--only=LEG[,LEG...]] [--clean]
#   --only  header | changelog | cells | legacy | kh1   (default: all five)
# Fixtures: F-SFD (MRX_FIXTURES, default out/mrx/fixtures; tools/gen_schema_fast_decode_fixture.py when absent).
#
# Legs (each PASS or FAIL; every leg runs whatever an earlier one did; exit 0 only when every leg run passes):
#   header     E-11 (a), (c) as ruling 21 amends them: no public header (include/superslm/*.h, *.inc) names
#              SSLM_JUMP_CANONICAL, SSLM_GPU_ROWS_SCHEMA_FORCED or (ruling 22) SSLM_GPU_ROWS_FIRST_PREWALKED; sslm_jump_mode has SSLM_JUMP_DEFAULT = 0 and
#              SSLM_JUMP_EXACT = 1 (ruling 21a, pinned by ruling 29), every enumerator with an explicit value; and the GPU side has its DEFAULT
#              too: an enum in gpu_1p0.h with a *_JUMP_DEFAULT = 0 enumerator, or, where SslmGpuJumpOptions' mode takes
#              the CPU enum's values (no GPU enum), gpu_1p0.h names SSLM_JUMP_DEFAULT.
#   changelog  E-11 (f), ruling 21: CHANGELOG.md's 1.18.0 section has a "Removed" heading whose text says "Canonical is
#              removed; mode 0 is now SSLM_JUMP_DEFAULT, which selects Exact, as do NULL options." (case, whitespace,
#              backticks and asterisks ignored).
#   cells      mrx_cpu x.1 (mode 0 = explicit Exact; a mode outside the enum refused, nothing changed) and x.3 (NULL
#              equals explicit Exact, saves included).
#   legacy     x.5: the recorder, built from v1.15.0, v1.16.0 and v1.17.0, saves under each release's Canonical jump;
#              the branch restores every blob and continues under the mode asked for.
#   kh1        ruling 29 (S1): mutant KH1, SSLM_JUMP_EXACT = 2 in a copy of the tree; the header leg and x.1 must go red.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/out/mrx"
JOBS=4
ONLY="header,changelog,cells,legacy,kh1"
CLEAN=0
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		--clean) CLEAN=1 ;;
		*) sed -n '2,22p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
mkdir -p "$OUT"
want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
FX="${MRX_FIXTURES:-$OUT/fixtures}"
TAGS="v1.15.0 v1.16.0 v1.17.0"
cleanup() {
	[ "$CLEAN" = 1 ] || return 0
	rm -rf "$OUT/cpu"
	for t in $TAGS; do [ -d "$OUT/tree-$t" ] && { git worktree remove --force "$OUT/tree-$t" 2>/dev/null || rm -rf "$OUT/tree-$t"; }; done
	git worktree prune
}
trap cleanup EXIT

# cpu_exe TREE OBJ EXE: the tree's CMakeLists core sources plus mrx_cpu.cpp, compiled natively and linked.
cpu_exe() {
	local tree="$1" obj="$2" exe="$3" srcs
	mkdir -p "$obj"
	srcs="$(sed -n '/^set(SUPERSLM_CORE_SOURCES/,/^)/p' "$tree/CMakeLists.txt" | grep -o 'src/[^ )]*\.cpp')"
	[ -n "$srcs" ] || { echo "    no core sources in $tree/CMakeLists.txt"; return 1; }
	cp "$HERE/mrx_cpu.cpp" "$obj/mrx_cpu.cpp"
	# shellcheck disable=SC2086
	(cd "$tree" && printf '%s\n' $srcs "$obj/mrx_cpu.cpp" | xargs -P "$JOBS" -I{} bash -c \
		'o="$1/$(basename "$2").o"; [ "$o" -nt "$2" ] && exit 0; g++ -std=c++20 -O2 -ffp-contract=off -w -Iinclude -Isrc -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$obj" {}) &&
		g++ -O2 "$obj"/*.o -o "$exe" -lpthread
}
need_fixtures() {
	[ -f "$FX/fsfd.sslm" ] || { mkdir -p "$FX"; python3 tools/gen_schema_fast_decode_fixture.py "$FX" >/dev/null; }
}

# ------------------------------------------------------------------------------------------------ header
# header_check DIR: the header leg's predicate on DIR (include/superslm, or the KH1 mutant's copy).
header_check() {
	python3 -I - "$1" <<'PY'
import glob, os, re, sys
bad = []
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.h")) + glob.glob(os.path.join(sys.argv[1], "*.inc")))
for p in files:
    src = open(p, encoding="utf-8", errors="replace").read()
    for name in ("SSLM_JUMP_CANONICAL", "SSLM_GPU_ROWS_SCHEMA_FORCED", "SSLM_GPU_ROWS_FIRST_PREWALKED"):
        for i, line in enumerate(src.split("\n"), 1):
            if name in line: bad.append("%s:%d names %s" % (os.path.relpath(p), i, name))
abi = open(os.path.join(sys.argv[1], "sslm_abi.h")).read()
m = re.search(r"typedef enum sslm_jump_mode\s*\{(.*?)\}", abi, re.S)
if not m: bad.append("sslm_abi.h: no sslm_jump_mode enum")
else:
    vals = dict((k, int(v, 0)) for k, v in re.findall(r"(\w+)\s*=\s*(-?\w+)", m.group(1)))
    if vals.get("SSLM_JUMP_DEFAULT") != 0: bad.append("sslm_jump_mode: SSLM_JUMP_DEFAULT is not 0 (%r)" % vals.get("SSLM_JUMP_DEFAULT"))
    # Ruling 21a, pinned by ruling 29 (MRX review S1): Exact is and stays 1, so a 1.17 binary that passes 1 keeps Exact.
    if vals.get("SSLM_JUMP_EXACT") != 1: bad.append("sslm_jump_mode: SSLM_JUMP_EXACT is not 1 (%r; ruling 21a)" % vals.get("SSLM_JUMP_EXACT"))
    names = re.findall(r"\b(SSLM_JUMP_\w+)\b", m.group(1))
    implicit = [n for n in names if n not in vals]
    if implicit: bad.append("sslm_jump_mode has enumerators without an explicit value: %s" % implicit)
gpu = open(os.path.join(sys.argv[1], "gpu_1p0.h")).read()
genums = [b for b in re.findall(r"typedef enum\s*\w*\s*\{(.*?)\}", gpu, re.S) if "JUMP" in b]
if genums:
    if not any(re.search(r"\b\w*_JUMP_DEFAULT\s*=\s*0\b", b) for b in genums): bad.append("gpu_1p0.h: its jump-mode enum has no *_JUMP_DEFAULT = 0")
elif "SSLM_JUMP_DEFAULT" not in gpu:
    bad.append("gpu_1p0.h: SslmGpuJumpOptions' mode is not given SSLM_JUMP_DEFAULT (no GPU enum, and the header never names it)")
for b in bad[:20]: print("    header: " + b)
if len(bad) > 20: print("    header: ... %d more" % (len(bad) - 20))
sys.exit(1 if bad else 0)
PY
}
if want header; then
	if header_check include/superslm
	then echo "PASS header (no SSLM_JUMP_CANONICAL, SSLM_GPU_ROWS_SCHEMA_FORCED or SSLM_GPU_ROWS_FIRST_PREWALKED; SSLM_JUMP_DEFAULT = 0, SSLM_JUMP_EXACT = 1, CPU and GPU)"; else echo "FAIL header"; FAILED+=(header); fi
fi

# ------------------------------------------------------------------------------------------------ changelog
if want changelog; then
	if python3 -I - CHANGELOG.md <<'PY'
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"^##\s*\[?1\.18\.0\]?.*?$(.*?)(?=^##\s*\[?\d|\Z)", src, re.M | re.S)
if not m: print("    changelog: no 1.18.0 section"); sys.exit(1)
sec = m.group(1)
r = re.search(r"^###\s*Removed\s*$(.*?)(?=^###|\Z)", sec, re.M | re.S)
if not r: print("    changelog: the 1.18.0 section has no \"### Removed\""); sys.exit(1)
text = re.sub(r"\s+", " ", re.sub(r"[`*]", "", r.group(1))).lower()
want = "Canonical is removed; mode 0 is now SSLM_JUMP_DEFAULT, which selects Exact, as do NULL options."
if want.lower() not in text:
    print("    changelog: 1.18.0's Removed does not say \"%s\"" % want); sys.exit(1)
PY
	then echo "PASS changelog (1.18.0 Removed: Canonical removed, mode 0 is SSLM_JUMP_DEFAULT = Exact, as NULL)"; else echo "FAIL changelog"; FAILED+=(changelog); fi
fi

# ------------------------------------------------------------------------------------------------ cells
BR="$OUT/cpu/branch"
if want cells || want legacy; then
	need_fixtures
	if cpu_exe "$REPO" "$BR/obj" "$BR/mrx_cpu" >"$OUT/cpu-build.log" 2>&1; then BR_OK=1; else BR_OK=0; tail -5 "$OUT/cpu-build.log" | sed 's/^/    /' || true; echo "FAIL build: the branch's mrx_cpu ($OUT/cpu-build.log)"; FAILED+=(build); fi
fi
if want cells && [ "${BR_OK:-0}" = 1 ]; then
	rc=0; "$BR/mrx_cpu" --fixtures="$FX" --cells=x.1,x.3 >"$OUT/cells.log" 2>&1 || rc=$?
	{ grep -E "^(FAIL|RED)|cell |cells:" "$OUT/cells.log" | head -40 | sed 's/^/    /'; } || true
	if [ $rc = 0 ]; then echo "PASS cells (x.1, x.3)"; else echo "FAIL cells (exit $rc, $OUT/cells.log)"; FAILED+=(cells); fi
fi

# ------------------------------------------------------------------------------------------------ legacy
if want legacy && [ "${BR_OK:-0}" = 1 ]; then
	ok=1
	for t in $TAGS; do
		tt="$OUT/tree-$t" rec="$OUT/legacy/$t"
		if [ -f "$rec/longkey.next" ]; then continue; fi
		if [ ! -d "$tt/src" ]; then git worktree add --detach "$tt" "$t" >/dev/null 2>&1 || { ok=0; echo "    legacy: no tree for $t"; continue; }; fi
		mkdir -p "$rec"
		if ! cpu_exe "$tt" "$OUT/cpu/$t/obj" "$OUT/cpu/$t/mrx_cpu" >"$OUT/cpu/$t.build.log" 2>&1; then ok=0; echo "    legacy: the $t recorder did not build ($OUT/cpu/$t.build.log)"; continue; fi
		"$OUT/cpu/$t/mrx_cpu" --fixtures="$FX" --record="$rec" >"$OUT/cpu/$t.record.log" 2>&1 || { ok=0; echo "    legacy: the $t recorder failed:"; sed 's/^/      /' "$OUT/cpu/$t.record.log" | head -5 || true; }
	done
	rc=0; "$BR/mrx_cpu" --fixtures="$FX" --legacy="$OUT/legacy" --cells=x.5 >"$OUT/legacy.log" 2>&1 || rc=$?
	{ grep -E "^(FAIL|RED)|cell |cells:" "$OUT/legacy.log" | head -30 | sed 's/^/    /'; } || true
	[ $rc = 0 ] || ok=0
	if [ $ok = 1 ]; then echo "PASS legacy (x.5: 1.15-1.17 Canonical saves restore and continue)"; else echo "FAIL legacy ($OUT/legacy.log)"; FAILED+=(legacy); fi
fi

# ------------------------------------------------------------------------------------------------ kh1
# Ruling 29 (MRX review S1): mutant KH1 renumbers SSLM_JUMP_EXACT to 2 in a copy of the tree. The header leg and x.1
# (the literal 1 against the symbol) must both go red on it.
if want kh1; then
	need_fixtures
	K="$OUT/kh1"
	rm -rf "$K"; mkdir -p "$K/tree"
	cp -a include src CMakeLists.txt "$K/tree/"
	ok=1
	if ! python3 -I - "$K/tree/include/superslm/sslm_abi.h" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s2, n = re.subn(r"(\bSSLM_JUMP_EXACT\s*=\s*)1\b", r"\g<1>2", s)
if n != 1: sys.exit("KH1: %d sites of SSLM_JUMP_EXACT = 1, want 1" % n)
open(p, "w").write(s2)
PY
	then echo "FAIL kh1: the site SSLM_JUMP_EXACT = 1 is not in sslm_abi.h"; FAILED+=(kh1); ok=2; fi
	if [ $ok = 1 ]; then
		if header_check "$K/tree/include/superslm" >"$K/header.log" 2>&1; then ok=0; echo "    kh1: the header leg stayed green"; fi
		if cpu_exe "$K/tree" "$K/obj" "$K/mrx_cpu" >"$K/build.log" 2>&1; then
			"$K/mrx_cpu" --fixtures="$FX" --cells=x.1 >"$K/x1.log" 2>&1 && { ok=0; echo "    kh1: x.1 stayed green"; }
			grep -q "^RED cell x.1" "$K/x1.log" || { grep -q "^cells: " "$K/x1.log" && { ok=0; echo "    kh1: x.1 printed no RED line"; }; }
		else
			ok=0; echo "    kh1: the mutant did not build ($K/build.log)"
		fi
		if [ $ok = 1 ]; then echo "PASS kh1 (SSLM_JUMP_EXACT = 2: the header leg and x.1 go red)"; else echo "FAIL kh1 survived"; FAILED+=(kh1); fi
	fi
	rm -rf "$K/tree" "$K/obj"
fi

if [ ${#FAILED[@]} -eq 0 ]; then echo "MRX CPU: PASS"; exit 0; fi
echo "MRX CPU: FAIL (${FAILED[*]})"
exit 1
