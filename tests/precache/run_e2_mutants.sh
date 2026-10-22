#!/usr/bin/env bash
# Runs apply_e2_mutant.py's mutants (T-3060 F3, F6, F7, F14; plan §9 E2 receipt (3)) against the E2 restore cells.
#
# usage: run_e2_mutants.sh <source tree, a clean checkout> <configured build dir> <output dir> [mutant ...]
#
# The caller sets SUPERSLM_PAGED_KV_FIXTURE_DIR and SUPERSLM_PRECACHE_SYNTHETIC_DIR. For each mutant (default: all) it
# applies the mutant, rebuilds superslm_precache_e1, runs the cells below, restores src/ with git checkout, and writes
# <out>/<mutant>.log. Each cell runs in a process of its own, so a crash in one cell cannot hide another cell's named
# assertions. <out>/summary.txt lists, per run, the cells red beyond the unmutated baseline's and, separately, the
# cells that crashed. A mutant is killed when a cell green unmutated turns red by assertion; a mutant that only
# crashes cells is reported as killed-by-crash, which is not a named-status kill.
set -u
src=$(cd "$1" && pwd); build=$2; out=$3; shift 3
mkdir -p "$out"
cells=(2.1/E2 2.2/E2 2.3/E2 2.3b/E2 2.4/E2 2.5/E2 2.13/E2 2.14/E2 5.1/E2 5.2/E2 5.6/E2 5.7/E2 5.11/E2 6.4/E2 6.7/E2
       7.1/E2 7.2/E2 7.3/E2 7.11/E2 7.12/E2:independent-parser 7.12/E2:format 8.4/E2 9.4/E2 11.1/E2 11.2/E2 11.6/E2
       1.1/E2 1.3/E2)
if [ -n "$(git -C "$src" status --porcelain -- src)" ]; then echo "src/ is not clean" >&2; exit 2; fi
mutants=("$@")
if [ ${#mutants[@]} -eq 0 ]; then mapfile -t mutants < <(python3 "$src/tests/precache/apply_e2_mutant.py" --list); fi

red_cells() { grep -oE '^\[E[0-9]\] [^ ]+ RED' "$1" | awk '{print $2}' | sort; }

run_one() {  # name -> "crashed: <cells>" (empty when none), or BUILD-FAIL
	cmake --build "$build" --target superslm_precache_e1 > "$out/$1.build.log" 2>&1 || { echo "BUILD-FAIL"; return; }
	: > "$out/$1.log"; local crashed="" c rc
	for c in "${cells[@]}"; do
		(cd "$build" && timeout 3600 ./superslm_precache_e1 "=$c" >> "$out/$1.log" 2>&1); rc=$?
		[ $rc -eq 0 ] || [ $rc -eq 1 ] || crashed+="$c(exit $rc) "
	done
	echo "crashed: ${crashed:-none}"
}
totals() { awk '/ (green|RED) \(/{n++; if($3=="RED") r++; sub(/\(/,"",$4); k+=$4; f+=$6} END{printf "%d cells, %d red; %d checks, %d failures", n, r, k, f}' "$1"; }

: > "$out/summary.txt"
r=$(run_one baseline); echo "baseline $r $(totals "$out/baseline.log")" | tee -a "$out/summary.txt"
red_cells "$out/baseline.log" > "$out/baseline.red"
for m in "${mutants[@]}"; do
	python3 "$src/tests/precache/apply_e2_mutant.py" "$src" "$m" > /dev/null || { echo "$m APPLY-FAIL" | tee -a "$out/summary.txt"; continue; }
	result=$(run_one "$m")
	git -C "$src" checkout -q -- src
	extra=$(comm -13 "$out/baseline.red" <(red_cells "$out/$m.log") | tr '\n' ' ')
	verdict=survived
	if [ "$result" = BUILD-FAIL ]; then verdict=build-fail
	elif [ -n "$extra" ]; then verdict=killed
	elif [ "$result" != "crashed: none" ]; then verdict=killed-by-crash; fi
	echo "$m $verdict $(totals "$out/$m.log") newly-red: ${extra:-none} $result" | tee -a "$out/summary.txt"
done
cmake --build "$build" --target superslm_precache_e1 > /dev/null 2>&1
