#!/usr/bin/env bash
# Runs apply_e2_mutant.py's mutants (T-3060 F3, F6, F7, F14; plan §9 E2 receipt (3)) against the E2 restore cells.
#
# usage: run_e2_mutants.sh <source tree, a clean checkout> <configured build dir> <output dir> [mutant ...]
#
# The caller sets SUPERSLM_PAGED_KV_FIXTURE_DIR and SUPERSLM_PRECACHE_SYNTHETIC_DIR. For each mutant (default: all) it
# applies the mutant, rebuilds superslm_precache_e1, runs the cells below, restores src/ with git checkout, and writes
# <out>/<mutant>.log. <out>/summary.txt lists, per run, the cells red beyond the unmutated baseline's. A mutant is
# killed when at least one cell turns red that is green unmutated (or the runner crashes).
set -u
src=$(cd "$1" && pwd); build=$2; out=$3; shift 3
mkdir -p "$out"
cells=(2.1/E2 2.2/E2 2.3/E2 2.3b/E2 2.4/E2 2.5/E2 2.13/E2 2.14/E2 5.1/E2 5.2/E2 5.6/E2 5.7/E2 5.11/E2 6.4/E2 6.7/E2
       7.1/E2 7.2/E2 7.3/E2 7.11/E2 7.12/E2:independent-parser 7.12/E2:format 8.4/E2 9.4/E2 11.1/E2 11.2/E2 11.6/E2
       1.1/E2 1.3/E2)
args=(); for c in "${cells[@]}"; do args+=("=$c"); done
if [ -n "$(git -C "$src" status --porcelain -- src)" ]; then echo "src/ is not clean" >&2; exit 2; fi
mutants=("$@")
if [ ${#mutants[@]} -eq 0 ]; then mapfile -t mutants < <(python3 "$src/tests/precache/apply_e2_mutant.py" --list); fi

red_cells() { grep -oE '^\[E[0-9]\] [^ ]+ RED' "$1" | awk '{print $2}' | sort; }

run_one() {  # name
	cmake --build "$build" --target superslm_precache_e1 > "$out/$1.build.log" 2>&1 || { echo "BUILD-FAIL"; return; }
	(cd "$build" && timeout 3600 ./superslm_precache_e1 "${args[@]}" > "$out/$1.log" 2>&1); echo "exit=$?"
}

: > "$out/summary.txt"
echo "baseline $(run_one baseline) $(tail -1 "$out/baseline.log")" | tee -a "$out/summary.txt"
red_cells "$out/baseline.log" > "$out/baseline.red"
for m in "${mutants[@]}"; do
	python3 "$src/tests/precache/apply_e2_mutant.py" "$src" "$m" > /dev/null || { echo "$m APPLY-FAIL" | tee -a "$out/summary.txt"; continue; }
	result=$(run_one "$m")
	git -C "$src" checkout -q -- src
	extra=$(comm -13 "$out/baseline.red" <(red_cells "$out/$m.log") | tr '\n' ' ')
	verdict=survived; { [ -n "$extra" ] || [[ "$result" != "exit=0" && "$result" != "exit=1" ]]; } && verdict=killed
	echo "$m $verdict $result $(tail -1 "$out/$m.log" 2>/dev/null) newly-red: ${extra:-none}" | tee -a "$out/summary.txt"
done
cmake --build "$build" --target superslm_precache_e1 > /dev/null 2>&1
