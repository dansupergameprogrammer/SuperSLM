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
#   --only     unmodified | suites | tests | pkvg   run one leg
#   --with-pkvg  also run the pkvg leg (off by default: under lavapipe its lifetime cells, e.g. 1.9's 1,000
#              prefix cycles, run for hours)
# Each run uses its own WINEPREFIX (<out>/wineprefix-<pid>) unless WINEPREFIX is set.
#
# Legs, each printed PASS or FAIL, exit 0 only when every leg run passes:
#   unmodified  `git diff <base> HEAD` is empty over tests/ (except tests/multirow-gpu/ and the two ordinal cells
#               below), and build.bat's
#               superslm_tests compile command (sources, /D, /I) equals the tag's: the suite that runs is the
#               tag's. build.bat may gain shader steps (the widened variants). The one allowed edit (errata E-2,
#               rulings item 9, the precedent of 1948e1f2): t2791's and t2956's cell_status_ordinals.cpp gain the
#               appended statuses. Every ordinal pin and switch case the tag's copies hold must still be there.
#   suites      build.bat's T-2791 and T-2956 suites (the cells their .bat files build) compiled with mingw and
#               -Werror=switch, standing in for MSVC's /we4062 on their exhaustive status switches. Before it, a
#               control: the tag's cell_status_ordinals.cpp against this header must fail -Wswitch whenever the
#               header has statuses the tag lacks (the leg can see the defect it exists for).
#   tests       superslm_tests.exe, built from build.bat's own superslm_tests command (sources, /D and /I
#               parsed from the file, so the cloud build follows the box build), run under Wine. Green when its
#               FAIL lines are exactly p41_wine_baseline_<base>.txt (the base tag's own under Wine), or, where
#               p41_wine_branch_<base>.txt exists (ruling 23, MRX), exactly its FAIL lines with the base still
#               failing exactly its own pin. Ruling 29 (N3): the count delta is pinned per suite too (its SUITE lines;
#               test_main.cpp is compiled from an instrumented copy that prints each top-level call's checks).
#   pkvg        (opt-in) the paged-KV GPU suite's step-G5 runner (tests/paged-kv-gpu, pkvg_g5) over
#               tools/gen_paged_kv_fixture.py's fixtures (pkv_def, pkv_qk, ...), real-artifact cells skipped; exit 0.
#
#               And its total check count equals the base tag's own superslm_tests, built the same way from a
#               detached worktree (<out>/base-tree) and run in the same session, plus the branch pin's DELTA (0
#               without one): a silent drop is red.
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
# A prefix this run made (not a caller's) is removed at exit, with its per-model children: each is a full Wine tree.
if [ -z "${WINEPREFIX_P41:-}" ]; then
	trap 'for p in "$WINEPREFIX" "$WINEPREFIX"-*; do [ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }; done; true' EXIT
fi
mkdir -p "$OUT/shaders" "$OUT/obj"
FAILED=()
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
ORD_CELLS=(tests/t2791-gpu-prefill-read-red-suite/cell_status_ordinals.cpp tests/t2956-token-finish-red-suite/cell_status_ordinals.cpp)
# Errata E-3 (R-17): 1.17's G0.2 status-mirror checker admits the branch-provisional status after 31 and 32.
E3_FILE=tests/schema-fast-decode-gpu/check_gpu_status_mirrors.py
# Ruling 23's P4.1 amendment (MRX, as E-2/E-3 were): MRX's listed edits under tests/ outside multirow-gpu. 1.17's CPU
# schema-fast-decode suite's Canonical cells retired or re-pointed to Exact (the table in red-suite-mrx.md), the
# 1183-1185 relabel (21b), and the te425 census rows that move with the verb (e66f893b).
# Ruling 29 (the MRX review's M1, the same precedent): 1.17's SFD-GPU cells G12.6 and G7.1, re-pointed from Canonical's
# contract and pinned Canonical streams to Exact expectations computed without the jump verb (red-suite-mr5.md); as
# widened at MR5: G9.1, G11.1, G12.1-3/9 (sfdg_cell_g12_verb), G12.7, G12.10 and sfdg_helpers.h's loop and Off pins.
MRX_FILES=(tests/test_schema_fast_decode.cpp tests/schema_fast_decode/sfd_cells_x.inc
	tests/te425-gpu-alloc-status-red-suite/gpu_status_sites.txt
	tests/schema-fast-decode-gpu/sfdg_cell_g12_6.cpp tests/schema-fast-decode-gpu/sfdg_cell_g7_1.cpp
	tests/schema-fast-decode-gpu/sfdg_helpers.h tests/schema-fast-decode-gpu/sfdg_cell_g9_1.cpp
	tests/schema-fast-decode-gpu/sfdg_cell_g11_1.cpp tests/schema-fast-decode-gpu/sfdg_cell_g12_verb.cpp
	tests/schema-fast-decode-gpu/sfdg_cell_g12_7.cpp tests/schema-fast-decode-gpu/sfdg_cell_g12_10.cpp)
EXCL=(':(exclude)tests/multirow-gpu' ":(exclude)${ORD_CELLS[0]}" ":(exclude)${ORD_CELLS[1]}" ":(exclude)$E3_FILE")
for f in "${MRX_FILES[@]}"; do EXCL+=(":(exclude)$f"); done

if want unmodified; then
	echo "== P4.1 unmodified: the suite's sources and build.bat's superslm_tests command against $BASE"
	ok=1
	if ! git diff --quiet "$BASE" HEAD -- tests "${EXCL[@]}"; then
		echo "    tests/ outside tests/multirow-gpu differs from $BASE:"
		git diff --stat "$BASE" HEAD -- tests "${EXCL[@]}" | sed 's/^/        /'
		ok=0
	fi
	# The ordinal cells may only gain the appended statuses: every pin and case line of the tag's copy stays.
	for c in "${ORD_CELLS[@]}"; do
		missing="$(git show "$BASE:$c" | grep -E '^\s*(PIN\(|static_assert\(static_cast<uint32_t>|case )' | grep -vxF -f "$c" || true)"
		if [ -n "$missing" ]; then
			echo "    $c lost lines the tag's copy pins:"; sed 's/^/        /' <<<"$missing"; ok=0
		fi
	done
	# build.bat may gain shader steps (the widened variants); the superslm_tests compile line may not change.
	if ! diff <(git show "$BASE:build.bat" | python3 -I "$HERE/p41_parse_build_bat.py" -) \
		<(python3 -I "$HERE/p41_parse_build_bat.py" build.bat) >/dev/null; then
		echo "    build.bat's superslm_tests command (sources, /D, /I) differs from $BASE"
		ok=0
	fi
	if [ $ok = 1 ]; then echo "PASS unmodified"; else echo "FAIL unmodified"; FAILED+=(unmodified); fi
fi

if want suites; then
	echo "== P4.1 suites: the T-2791 and T-2956 cells compiled with -Werror=switch (MSVC's /we4062 here)"
	ok=1
	SFLAGS="-std=c++20 -Iinclude -Isrc -Isrc/gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -Werror=switch -fsyntax-only"  # no -w: it silences -Werror=switch too
	# Control: with statuses appended since the tag, the tag's own ordinal cells must fail here on -Wswitch.
	if [ -n "$(git diff "$BASE" HEAD -- include/superslm/gpu_1p0.h | grep -E '^\+\s*SSLM_[A-Z0-9_]+,?\s*(//.*)?$')" ]; then
		mkdir -p "$OUT/suites"
		for c in "${ORD_CELLS[@]}"; do
			t="$OUT/suites/tag_$(basename "$(dirname "$c")").cpp"; git show "$BASE:$c" >"$t"
			# shellcheck disable=SC2086
			if "$CXX" $SFLAGS "$t" >"$t.log" 2>&1 || ! grep -q "not handled in switch" "$t.log"; then
				echo "    control: $BASE's $c compiles against this header: the leg cannot see a missing case"; ok=0
			else
				echo "    control: $BASE's $c fails -Wswitch against this header, as it must"
			fi
		done
	fi
	# The cells each .bat builds, parsed from its own list. Not compiled here: cell_prompt_guard_removal, whose
	# ID3D12Device5 mingw-w64 11's d3d12.h lacks; it may hold no switch, which is checked.
	t2791="$(grep -A3 '^for %%f in (cell_status_ordinals' tests/t2791-gpu-prefill-read-red-suite/build_red_suite.bat | tr -d '^\r' | sed -n 's/.*(\(.*\)/\1/p;s/^ *\(cell[^)]*\).*/\1/p' | tr '\n' ' ')"
	t2956="$(grep -m1 '^for %%C in (' tests/t2956-token-finish-red-suite/build_cells.bat | sed 's/.*(\(.*\)).*/\1/')"
	n=0
	for cell in $t2791; do
		f=tests/t2791-gpu-prefill-read-red-suite/$cell.cpp
		if [ "$cell" = cell_prompt_guard_removal ]; then
			grep -q 'switch' "$f" && { echo "    $f now holds a switch and is not compiled here"; ok=0; }
			continue
		fi
		def=-DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION; [ "$cell" = cell_env_pins_shipping_leg ] && def=
		# shellcheck disable=SC2086
		"$CXX" $SFLAGS $def "$f" >"$OUT/suites.$cell.log" 2>&1 || { echo "    FAIL $f:"; grep -m5 'error' "$OUT/suites.$cell.log" | sed 's/^/        /'; ok=0; }
		n=$((n + 1))
	done
	for cell in $t2956; do
		f=tests/t2956-token-finish-red-suite/$cell.cpp
		# shellcheck disable=SC2086
		"$CXX" $SFLAGS -DT2956_CANDIDATE "$f" >"$OUT/suites.t2956.$cell.log" 2>&1 || { echo "    FAIL $f:"; grep -m5 'error' "$OUT/suites.t2956.$cell.log" | sed 's/^/        /'; ok=0; }
		n=$((n + 1))
	done
	[ "$n" -ge 25 ] || { echo "    only $n cells parsed from the two .bat files"; ok=0; }
	if [ $ok = 1 ]; then echo "PASS suites ($n cells, -Werror=switch)"; else echo "FAIL suites"; FAILED+=(suites); fi
fi

# shaders ROOT DIR: ROOT's src/gpu/shaders compiled into DIR/shaders, plus the vkd3d-proton DLLs into DIR.
shaders() {
	local root="$1" dir="$2"; mkdir -p "$dir/shaders"
	for f in "$root"/src/gpu/shaders/*.hlsl; do
		local o="$dir/shaders/$(basename "$f" .hlsl).cso"
		if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ -n "$(find "$root/src/gpu/shaders" -name '*.hlsli' -newer "$o")" ]; then
			"$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX >/dev/null
		fi
	done
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$dir/"
}

# build_tests ROOT DIR: superslm_tests.exe from ROOT's own build.bat command (sources, /D, /I), built in ROOT
# into DIR. Same flags, compiler and link line for the branch and for the base tag: one build config.
build_tests() {
	local root="$1" dir="$2" parsed srcs defs incs flags obj
	parsed="$(python3 -I "$HERE/p41_parse_build_bat.py" "$root/build.bat")"
	srcs="$(sed -n 1p <<<"$parsed")"; defs="$(sed -n 2p <<<"$parsed")"; incs="$(sed -n 3p <<<"$parsed")"
	flags="-std=c++20 -O2 $incs $defs -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h"
	shaders "$root" "$dir"
	obj="$dir/obj/tests"; mkdir -p "$obj"
	# Ruling 29 (MRX review N3): per-suite counts. test_main.cpp is compiled from an instrumented copy whose main()
	# prints each top-level call's checks (p41_count_suites.py); same calls, same order, same checks. -iquote keeps
	# its quoted includes resolving from <tree>/tests.
	if grep -qw "tests/test_main.cpp" <<<"$srcs"; then
		python3 -I "$HERE/p41_count_suites.py" instrument "$root/tests/test_main.cpp" "$dir/obj/p41_test_main.cpp" >/dev/null \
			|| { echo "FAIL tests: instrumenting $root/tests/test_main.cpp"; return 1; }
		srcs="$(sed "s#\btests/test_main.cpp\b#$dir/obj/p41_test_main.cpp#" <<<"$srcs")"
		flags="$flags -iquote $root/tests"
	fi
	# shellcheck disable=SC2086
	(cd "$root" && printf '%s\n' $srcs | xargs -P "$JOBS" -I{} bash -c \
		'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX $flags"' -w -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$obj" {}) \
		|| { echo "FAIL tests: compile in $root"; return 1; }
	"$CXX" -O2 "$obj"/*.o -o "$dir/superslm_tests.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi \
		-static -static-libgcc -static-libstdc++ >"$dir/superslm_tests.link.log" 2>&1 \
		|| { echo "FAIL tests: link (see $dir/superslm_tests.link.log)"; sed -n '1,20p' "$dir/superslm_tests.link.log"; return 1; }
}

# run_tests ROOT DIR: DIR/superslm_tests.exe under Wine from ROOT (the shader staleness check walks up to
# ROOT/src/gpu/shaders), with the CI fixtures; log in DIR/superslm_tests.log.
run_tests() {
	local root="$1" dir="$2"
	start_xvfb; wine_env
	set +e
	(cd "$root" && "$WINE" "$dir/superslm_tests.exe" >"$dir/superslm_tests.log" 2>&1)
	local rc=$?
	set -e
	tail -3 "$dir/superslm_tests.log" | sed 's/^/    /'
	return $rc
}
total_checks() { sed -n 's/^superslm tests: \([0-9]*\) checks.*/\1/p' "$1" | tail -1; }

if want tests; then
	echo "== P4.1 tests: superslm_tests.exe from build.bat's command, on the branch and on $BASE"
	# The base tag's own suite, built the same way from a detached worktree of $BASE and run in the same session:
	# its total check count is the pin. The suite's count depends on the host (the sha256 hardware-path cells run
	# only where the CPU has the SHA extensions), so a count pinned in a file would go red on a host change and
	# green on a real drop that happened to coincide with one; the tag measured here, now, cannot.
	BT="$OUT/base-tree"
	if [ ! -d "$BT/src" ]; then
		git worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
		git worktree add --detach "$BT" "$BASE" >/dev/null
	fi
	[ "$(git -C "$BT" rev-parse HEAD)" = "$(git rev-parse "$BASE^{commit}")" ] || { echo "FAIL tests: $BT is not $BASE"; exit 1; }
	if ! build_tests "$REPO" "$OUT"; then
		FAILED+=(tests-build)
	elif ! build_tests "$BT" "$BT/out/mr-p41"; then
		FAILED+=(tests-base-build)
	elif [ "$RUN" = 1 ]; then
		# The fixtures CI points the suite at (.github/workflows/tests.yml): generated, never committed.
		FIX="$OUT/fixtures"; mkdir -p "$FIX/dt" "$FIX/sfd"
		[ -f "$FIX/pkv_def.sslm" ] || python3 tools/gen_paged_kv_fixture.py "$FIX" >/dev/null
		[ -n "$(ls "$FIX/dt")" ] || python3 tools/gen_decode_threading_fixture.py "$FIX/dt" >/dev/null
		[ -n "$(ls "$FIX/sfd")" ] || python3 tools/gen_schema_fast_decode_fixture.py "$FIX/sfd" >/dev/null
		export SUPERSLM_PAGED_KV_FIXTURE_DIR="$FIX" SUPERSLM_DECODE_THREADING_FIXTURE_DIR="$FIX/dt" \
			SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR="$FIX/sfd"
		echo "   branch:"; rc=0; run_tests "$REPO" "$OUT" || rc=$?
		echo "   $BASE:"; brc=0; run_tests "$BT" "$BT/out/mr-p41" || brc=$?
		LOG="$OUT/superslm_tests.log"; BLOG="$BT/out/mr-p41/superslm_tests.log"
		N="$(total_checks "$LOG")"; BN="$(total_checks "$BLOG")"
		# Ruling 2 (rulings-mr1.md): no failure beyond the base tag's own under Wine. The baseline file pins the
		# base tag's three sfd_cells_x failure lines (1.X4, 5.X4, 11.X3) verbatim; any other FAIL line, a changed
		# or missing baseline line, or a run that did not reach its totals line is red.
		# The pin is per base (R-17: the base is 1.17's branch head until 1.17.0 is tagged); with none committed
		# for this base, the base run's own FAIL lines are written as the pin (commit it), as P4.2 does.
		BASELINE="$HERE/p41_wine_baseline_$BASE.txt"
		grep "^FAIL" "$LOG" >"$OUT/superslm_tests.failures" || true
		if [ ! -f "$BASELINE" ] && [ -n "$BN" ]; then
			echo "   no pin $BASELINE; writing it from $BASE's run (commit it)"
			grep "^FAIL" "$BLOG" >"$BASELINE" || true
		fi
		# Ruling 23 (MRX): the branch's expected FAIL set and its check count are re-recorded on the MRX tree, in
		# p41_wine_branch_<base>.txt: one "DELTA <n>" line (the branch's total minus the base's, measured in the same
		# session, so the host-dependent sha256 cells cancel) and the branch's FAIL lines verbatim. Every difference
		# from the base's pin maps to a row of red-suite-mrx.md's ruling-23 table. With the file present the base run
		# must still fail exactly its own pin; without it the branch is graded against the base's pin, N = BN.
		BRANCH_PIN="$HERE/p41_wine_branch_$BASE.txt"
		EXPECT="$BASELINE"; DELTA=0
		if [ -f "$BRANCH_PIN" ]; then
			EXPECT="$OUT/superslm_tests.expected"; grep "^FAIL" "$BRANCH_PIN" >"$EXPECT" || true
			DELTA="$(sed -n 's/^DELTA \(-\{0,1\}[0-9]\{1,\}\)$/\1/p' "$BRANCH_PIN")"
		fi
		ok=1
		if [ -z "$DELTA" ]; then
			echo "FAIL tests: $BRANCH_PIN has no DELTA line"; ok=0
		elif [ -z "$N" ]; then
			echo "FAIL tests: the branch run did not reach its totals line (exit $rc, log $LOG)"; ok=0
		elif [ -z "$BN" ]; then
			echo "FAIL tests: the $BASE run did not reach its totals line (exit $brc, log $BLOG)"; ok=0
		else
			if ! diff -u "$EXPECT" "$OUT/superslm_tests.failures" >"$OUT/superslm_tests.failures.diff"; then
				echo "FAIL tests: failures differ from the pinned set ($(basename "$EXPECT"); $OUT/superslm_tests.failures.diff):"
				sed -n '3,30p' "$OUT/superslm_tests.failures.diff" | sed 's/^/    /'; ok=0
			fi
			if [ -f "$BRANCH_PIN" ] && ! diff -u "$BASELINE" <(grep "^FAIL" "$BLOG" || true) >"$OUT/superslm_tests.base.diff"; then
				echo "FAIL tests: the $BASE run's failures differ from its own pin ($OUT/superslm_tests.base.diff)"; ok=0
			fi
			# A silent drop (a suite skipped, a cell gone) is red: the branch runs every check the tag runs here,
			# give or take the pinned delta.
			if [ "$((N - BN))" != "$DELTA" ]; then
				echo "FAIL tests: the branch ran $N checks, $BASE ran $BN under the same build and host (pinned delta $DELTA)"; ok=0
			fi
			# Ruling 29 (MRX review N3): the delta per suite, so a gain in one suite cannot hide a loss in another.
			# Every top-level call's count on the branch minus the base's must equal the pin's "SUITE <name> <n>" line
			# (0 when the name has none), "(unattributed)" included; a name on one side only counts as 0 on the other.
			python3 -I "$HERE/p41_count_suites.py" tally "$LOG" >"$OUT/suites.branch.txt" || ok=0
			python3 -I "$HERE/p41_count_suites.py" tally "$BLOG" >"$OUT/suites.base.txt" || ok=0
			if ! python3 -I - "$OUT/suites.branch.txt" "$OUT/suites.base.txt" "${BRANCH_PIN}" "$OUT/suites.delta.txt" <<'PY'
import os, sys
def load(p):
    d = {}
    for line in open(p):
        name, n = line.rsplit(" ", 1)
        d[name] = int(n)
    return d
br, ba = load(sys.argv[1]), load(sys.argv[2])
pin = {}
if os.path.exists(sys.argv[3]):
    for line in open(sys.argv[3]):
        f = line.split()
        if len(f) == 3 and f[0] == "SUITE":
            pin[f[1]] = int(f[2])
delta = {k: br.get(k, 0) - ba.get(k, 0) for k in set(br) | set(ba)}
with open(sys.argv[4], "w") as out:
    for k in sorted(delta):
        if delta[k]:
            out.write("SUITE %s %d\n" % (k, delta[k]))
bad = [(k, delta.get(k, 0), pin.get(k, 0)) for k in sorted(set(delta) | set(pin)) if delta.get(k, 0) != pin.get(k, 0)]
for k, got, want in bad[:30]:
    print("    tests: suite %s: branch - base = %d, pinned %d" % (k, got, want))
print("    tests: %d suites compared, %d with a nonzero delta (%s)" % (len(delta), sum(1 for v in delta.values() if v), ", ".join("%s %+d" % (k, v) for k, v in sorted(delta.items()) if v) or "none"))
sys.exit(1 if bad else 0)
PY
			then
				echo "FAIL tests: the per-suite deltas differ from the pin's SUITE lines ($OUT/suites.delta.txt has the measured ones)"; ok=0
			fi
		fi
		if [ $ok = 1 ]; then
			echo "PASS tests (exit $rc; $N checks = $BASE's $BN + ($DELTA) on this host; failures are exactly the $(wc -l <"$EXPECT") pinned lines of $(basename "$EXPECT"))"
		else
			FAILED+=(tests)
		fi
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
