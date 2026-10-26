#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR5: the MR5 red suite's mutants (plan §11: MX19, MX23, MX24; G1-X's XG
# mutants that return under MR5; MXK6 per ruling 25), each a one-site edit of a copy of the engine, rebuilt with the
# cells' objects and flags. Each must turn its cell red, AND that cell must be green on the unmutated build first
# (ruling 25: a mutant is graded only against a green cell; against a cell already red it is VACUOUS, which fails).
# Today every MR5 cell is red (the GPU refuses Exact), so every mutant here is VACUOUS until MR5 builds Exact.
# Usage: tests/multirow-gpu/mr5_mutants.sh [--out=DIR] [--jobs=N] [--only=MX19,...]
#   MX19  the Exact commit stores row a - 1's delta as the counter total                                  -> 7.10
#   MX23  the Exact commit sums the deltas over all M rows                                                 -> 7.10
#   MX24  the Exact dead end leaves ready_for_logits clear                                                 -> 7.17   PENDING site
#   MXK6  the jump no longer restores last_rows_committed (mr4_cells; ruling 25: on a green n1)            -> n1
#   XG4a  the rejected rows' dedicated mirror rows are not restored (the mirror copy runs over all M rows) -> gx.2
#   XG4b  the rejected rows' device K/V is not restored: RECORDED, not run (needs S-GDEVROW, or the builder's
#         equivalence finding on G1-X's first task, SFD plan XG4b)
#   XG5   the GPU reads back the final codes of the wrong row (row M - 1 instead of a - 1)                 -> gx.1
#   XG6   the GPU keeps the pass's counter totals (MX23's edit, graded on the jump's cell)                 -> gx.2
#   XG7   the GPU bridge embeds a hostile draft token                                                      -> gx.5   PENDING site
#   XG12  Exact mode sets or clears the prefill snapshot                                                   -> gx.12  PENDING site
#   XG13  the check runs after every sub-chunk is recorded (rows past a mismatch computed)                 -> gx.13  PENDING site
#   XG17  the GPU leaves rejected rows' pages mapped (the close's unmap skipped)                            -> gx.2
#   XG18  the GPU rollback clears `dirty`                                                                  -> gx.2   PENDING site
#   XG14  RETIRED (ruling 26 Q3: Canonical is removed, so no draft table learns from a canonical generation)
#   XG21  DEFERRED with the GPU draft table (ruling 26 Q3)
# The sites: MX19, MX23, XG4a, XG5, XG6 and XG17 are lines of RowsCommitPrefix (src/gpu/gpu_1p0.cpp), the close the plan
# says Exact's CommitPrefix(a, a - 1) reuses (§3.5, §4.7); one edit here if MR5's Exact close is a function of its own.
# MX24, XG7, XG12, XG13 and XG18 have no site before MR5 writes the Exact close and the jump's Exact pass (the MR4 MX16
# precedent): each prints PENDING and fails until the builder names its site and this file gains the applicator.
# A mutant that crashes the cells (no verdict line) counts as red.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
OUT="$REPO/out/mr5/mutants"
JOBS=2
ONLY=""
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		*) sed -n '2,32p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
TOP="$(dirname "$OUT")"
F5="${MR5_FIXTURES:-$TOP/fixtures}"
F4="${MR4_FIXTURES:-$TOP/mr4/fixtures}"
bash "$HERE/mr5_wine.sh" --out="$TOP" --only=cells --build-only --jobs="$JOBS" >/dev/null
bash "$HERE/mr4_wine.sh" --out="$TOP/mr4" --only=cells --build-only --jobs="$JOBS" >/dev/null
MR4_FIXTURES="$F4" bash "$HERE/mr4_wine.sh" --out="$TOP/mr4" --fixtures-only >/dev/null || { echo "mutants: FAIL (fixtures in $F4)"; exit 1; }
[ -f "$F5/fsfd.sslm" ] && [ -f "$F5/fsfd_ref.txt" ] || python3 tools/gen_schema_fast_decode_fixture.py "$F5" >/dev/null
RETIRED=",XG14,XG21,XG4b,"
want() {
	case "$RETIRED" in *",$1,"*) return 1 ;; esac
	[ -z "$ONLY" ] && return 0
	case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}
FAILED=()
mkdir -p "$OUT"
export WINEPREFIX="${WINEPREFIX_MR5:-$OUT/wineprefix-$$}"
if [ -z "${WINEPREFIX_MR5:-}" ]; then
	trap 'for p in "$WINEPREFIX" "$WINEPREFIX"-*; do [ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }; done; true' EXIT
fi
start_xvfb; wine_env

# The suites: SUITE -> (cell build dir, exe, fixtures).
suite_dir() { case "$1" in mr5) echo "$TOP/cells" ;; mr4) echo "$TOP/mr4/cells" ;; esac; }
suite_exe() { case "$1" in mr5) echo mr5_cells.exe ;; mr4) echo mr4_cells.exe ;; esac; }
suite_fx() { case "$1" in mr5) echo "$F5" ;; mr4) echo "$F4" ;; esac; }

# run_cells SUITE DIR CELLS LOG: the suite's exe in DIR on CELLS; the exit status.
run_cells() {
	local suite="$1" dir="$2" cells="$3" log="$4" rc
	set +e
	(cd "$dir" && WINEPREFIX="$WINEPREFIX-m" timeout 3600 "$WINE" "./$(suite_exe "$suite")" --fixtures="$(suite_fx "$suite")" --cells="$cells") >"$log" 2>&1
	rc=$?
	set -e
	return $rc
}
# baseline SUITE CELL: 0 when CELL is green on the unmutated build (memoized per suite and cell).
declare -A BASE_GREEN=()
baseline() {
	local suite="$1" cell="$2" key="$1:$2"
	if [ -z "${BASE_GREEN[$key]:-}" ]; then
		mkdir -p "$OUT/baseline"
		run_cells "$suite" "$(suite_dir "$suite")" "$cell" "$OUT/baseline/$suite-$cell.log" || true
		if tr -d '\r' <"$OUT/baseline/$suite-$cell.log" | grep -q "^    cell $cell: PASS"; then BASE_GREEN[$key]=1; else BASE_GREEN[$key]=0; fi
	fi
	[ "${BASE_GREEN[$key]}" = 1 ]
}

# host_mutant NAME SUITE FILE COUNT PATTERN REPLACEMENT: FILE in a copy of the tree with every match of the Python
# regex PATTERN (exactly COUNT of them) replaced by REPLACEMENT; its translation unit rebuilt with the suite's flags;
# relinked.
host_mutant() {
	local name="$1" suite="$2" file="$3" count="$4" d="$OUT/$1" cells
	cells="$(suite_dir "$suite")"
	rm -rf "$d"; mkdir -p "$d/tree"
	cp "$cells/d3d12.dll" "$cells/d3d12core.dll" "$d/"; cp -r "$cells/shaders" "$d/shaders"; cp -a "$cells/obj" "$d/obj"
	cp -a include src tests "$d/tree/"
	if ! MX_PAT="$5" MX_NEW="$6" MX_N="$count" python3 -I -c "import os,re,sys; p=sys.argv[1]; s=open(p).read()
s2,n=re.subn(os.environ['MX_PAT'], os.environ['MX_NEW'], s, flags=re.M)
if n!=int(os.environ['MX_N']): sys.exit('%d matches of the site in %s, want %s' % (n, p, os.environ['MX_N']))
open(p,'w').write(s2)" "$d/tree/$file"; then
		echo "FAIL mutant $name: its site is not in $file as stated (the engine changed; update the applicator)"; FAILED+=("$name"); rm -rf "$d/tree" "$d/obj"; return 1
	fi
	# shellcheck disable=SC2086
	if ! (cd "$d/tree" && "$CXX" $(cat "$cells/obj.flags") -w -c "$file" -o "$d/obj/$(echo "$file" | tr / _).o") >"$d/build.log" 2>&1 ||
		! "$CXX" -O2 "$d/obj"/*.o -o "$d/$(suite_exe "$suite")" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc \
			-static-libstdc++ >>"$d/build.log" 2>&1; then
		echo "FAIL mutant $name: build ($d/build.log)"; FAILED+=("$name"); rm -rf "$d/tree" "$d/obj"; return 1
	fi
	rm -rf "$d/tree" "$d/obj"
}
# must_red NAME SUITE CELLS: every cell green on the unmutated build (else VACUOUS), and red under the mutant.
must_red() {
	local name="$1" suite="$2" cells="$3" c rc ok=1 vac=""
	for c in ${cells//,/ }; do baseline "$suite" "$c" || vac="$vac $c"; done
	if [ -n "$vac" ]; then
		if [ "$name" = MXK6 ]; then
			# Ruling 29 N2: NOT GRADED, not PASS. It still fails this leg: MR5's done criteria need MXK6 red on a green n1.
			echo "FAIL mutant MXK6: NOT GRADED (n1 red on the unmutated build; ruling 25: graded once n1 is green)"
		else
			echo "FAIL mutant $name: VACUOUS (cell$vac red on the unmutated build; ruling 25: graded only against a green cell)"
		fi
		FAILED+=("$name"); rm -rf "$OUT/$name/shaders"; return 0
	fi
	run_cells "$suite" "$OUT/$name" "$cells" "$OUT/$name/run.log" && rc=0 || rc=$?
	if tr -d '\r' <"$OUT/$name/run.log" | grep -q '^cells: '; then
		for c in ${cells//,/ }; do
			if ! tr -d '\r' <"$OUT/$name/run.log" | grep -q "^RED cell $c:"; then ok=0; echo "    $name: cell $c stayed green"; fi
		done
	else
		echo "    $name: the cells crashed (no verdict line; exit $rc): red"
	fi
	[ "$rc" != 0 ] || ok=0
	if [ $ok = 1 ]; then echo "PASS mutant $name -> red in $cells"; else echo "FAIL mutant $name survived ($cells)"; FAILED+=("$name"); fi
	rm -rf "$OUT/$name/shaders"
}
pending() { echo "FAIL mutant $1: PENDING (no site before MR5: $2) -> $3"; FAILED+=("$1"); }

G=src/gpu/gpu_1p0.cpp
SUM='^(\s*)for \(uint32_t r = 0; r <= f; \+\+r\) counters\[i\] \+= rows\[r\]\.counters\[i\];'
if want MX19; then host_mutant MX19 mr5 $G 1 "$SUM" '\1counters[i] = rows[f].counters[i];  // MX19: row a - 1 delta as the total' &&
	must_red MX19 mr5 7.10; fi
if want MX23; then host_mutant MX23 mr5 $G 1 "$SUM" '\1for (uint32_t r = 0; r < M; ++r) counters[i] += rows[r].counters[i];  // MX23: all M rows' &&
	must_red MX23 mr5 7.10; fi
if want MX24; then pending MX24 "the Exact close's dead-end branch, which re-arms ready_for_logits" 7.17; fi
if want MXK6; then host_mutant MXK6 mr4 $G 1 '~LastRowsKeep\(\) \{ s->last_rows_committed = v; \}' \
	'~LastRowsKeep() { (void)s; (void)v; }  // MXK6: no restore' && must_red MXK6 mr4 n1; fi
if want XG4a; then host_mutant XG4a mr5 $G 1 '^(\s*)for \(uint32_t r = 0; r < k; \+\+r\) copy_row\(r, L\);' \
	'\1for (uint32_t r = 0; r < M; ++r) copy_row(r, L);  // XG4a: the rejected rows mirrored too' && must_red XG4a mr5 gx.2; fi
if want XG5; then host_mutant XG5 mr5 $G 1 '^(\s*)const GpuRowsRowState& rf = rows\[f\];' \
	'\1const GpuRowsRowState& rf = rows[M - 1];  // XG5: the wrong row read back' && must_red XG5 mr5 gx.1; fi
if want XG6; then host_mutant XG6 mr5 $G 1 "$SUM" '\1for (uint32_t r = 0; r < M; ++r) counters[i] += rows[r].counters[i];  // XG6: the pass totals' &&
	must_red XG6 mr5 gx.2; fi
if want XG7; then pending XG7 "the jump's Exact pass, which cuts the draft at its first hostile token" gx.5; fi
if want XG12; then pending XG12 "the jump's Exact pass, which leaves the prefill snapshot alone" gx.12; fi
if want XG13; then pending XG13 "the jump's per-pass check, which stops at the first mismatch" gx.13; fi
if want XG17; then host_mutant XG17 mr5 $G 1 '^(\s*)status = RowsUnmapTailTo\(seq, std::max\(cover, c\.entry\.holder_mapped\)\);' \
	'\1status = SSLM_OK; (void)cover;  // XG17: rejected rows stay mapped' && must_red XG17 mr5 gx.2; fi
if want XG18; then pending XG18 "the Exact rollback's page release, which keeps a freed page dirty" gx.2; fi
echo "    mutant XG4b: RECORDED, not run (S-GDEVROW or the builder's equivalence finding); XG14: RETIRED (ruling 26 Q3); XG21: DEFERRED (ruling 26 Q3)"
if [ ${#FAILED[@]} -eq 0 ]; then echo "mutants: PASS"; exit 0; fi
echo "mutants: FAIL (${FAILED[*]})"
exit 1
