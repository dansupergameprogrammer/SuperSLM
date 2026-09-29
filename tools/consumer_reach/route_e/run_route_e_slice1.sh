#!/usr/bin/env bash
# run_route_e_slice1.sh -- cell 10.0's route E leg at S1-C, on the real slice-1 candidate (not the stand-in).
# usage: run_route_e_slice1.sh <scratch-dir> <candidate-engine-clone> <superembedder-checkout> <artifact.sslm> [tag]
#
# The candidate clone carries the slice-1 commit tagged <tag> (default v1.9.0: the embedder's lock names a tag, and
# the pin tool resolves it in --source). This is the core of run_route_e_legs.sh -- the same reference derivation,
# the same leg construction (the consumer's own pin tool, configure with every guard active, build), the same code
# witness and the same grader (route_e_check.py) -- reduced to the four legs that decide the slice-1 question:
#   i   production + the seam            must PASS (E6: the embedder's encode reaches the tiled kernel)
#   ii  D-infinity + the seam            must FAIL on E6 alone (the XR must-reject: no tiled entry)
#   c   forced scalar + the seam         must FAIL on E3 alone (the engine linked is not the reference engine)
#   a   no seam                          must FAIL on E3, E4 and E6 (no counter can be read)
# The full kill-leg census of run_route_e_legs.sh (flag channels, fixtures, Q1 pairs) is the planner's rev-10
# evidence on the stand-in and is not re-run here.
set -u
mkdir -p "$1"
S=$(cd "$1" && pwd); CAND=$2; SEMB_SRC=$3; ART=$4; TAG=${5:-v1.9.0}; SEMB_COMMIT=1ee74bf
HERE=$(cd "$(dirname "$0")" && pwd)
SEAM="-DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT"
git -C "$SEMB_SRC" show "$SEMB_COMMIT:cmake/toolchain-pin.json" > "$S/toolchain-pin.json" || { echo "no toolchain pin at $SEMB_COMMIT"; exit 1; }
export CODEID_TOOLCHAIN_PIN="$S/toolchain-pin.json"

# ---- the reference, derived from the candidate commit before any leg runs (as run_route_e_legs.sh) ----
src="$S/refsrc"; rm -rf "$src"; mkdir -p "$src"; git -C "$CAND" archive "$TAG" | tar -x -C "$src"
commit=$(git -C "$CAND" rev-parse "$TAG^{commit}")
ver=$(sed -n 's/^project(superslm VERSION \([0-9.]*\).*/\1/p' "$src/CMakeLists.txt")
std=gnu++$(sed -n 's/^set(CMAKE_CXX_STANDARD \([0-9]*\)).*/\1/p' "$src/CMakeLists.txt")
R="python3 $HERE/buildcfg_record.py reference $src $std"
C() { local n=$1; shift; rm -rf "$S/refbuild-$n"; python3 "$HERE/codeid.py" build-reference "$src" "$S/refbuild-$n" "$@"; }
cat > "$S/reference.json" <<J
{"commit": "$commit", "version": "$ver", "std": "$std",
 "record": {"production": $($R SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT),
            "d_inf": $($R SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT SUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX)},
 "code": {"production": $(C production -DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT -I$src/tests),
          "d_inf": $(C d_inf -DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT -I$src/tests -DSUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX)}}
J
echo "reference: commit $commit, version $ver; code_sha256 $(python3 -c "
import json,sys; r=json.load(open(sys.argv[1])); print(', '.join(k+' '+v['code_sha256'][:16] for k,v in r['code'].items()))" "$S/reference.json")"
echo "host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); $(g++ --version | head -1)"

leg() {  # name CXXFLAGS (@WD@ = the pin workdir)
  local name=$1 flags=$2 L="$S/leg-$1"; rm -rf "$L"; mkdir -p "$L/semb"
  git -C "$SEMB_SRC" archive "$SEMB_COMMIT" | tar -x -C "$L/semb"
  printf 'tag=%s\n' "$TAG" > "$L/semb/superslm.lock"
  local cxx=${flags//@WD@/$L/work}
  echo "== leg $name: tag=$TAG CXXFLAGS='${cxx//$S/<S>}'"
  local t0; t0=$(date +%s)
  ( cd "$L" && env CXXFLAGS="$cxx" python3 semb/tools/pin-superslm/pin_superslm.py --lock semb/superslm.lock \
      --prefix "$L/eng" --source "$CAND" --workdir "$L/work" --keep-workdir ) > "$L/pin.log" 2>&1
  local prc=$?
  cmake -S "$L/semb" -B "$L/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$L/eng" > "$L/configure.log" 2>&1
  local crc=$?
  ninja -C "$L/build" superembedder semb > "$L/build.log" 2>&1
  local brc=$?
  g++ -O2 -std=c++20 -I"$L/semb/include" -I"$L/eng/include" -I"$src/tests" "$HERE/route_e_reach.cpp" \
      "$L/build/libsuperembedder.a" "$L/eng/lib/libsuperslm.a" -pthread -o "$L/route_e_reach" > "$L/tool-build.log" 2>&1
  local trc=$?
  [ $trc -eq 0 ] && "$L/route_e_reach" "$ART" > "$L/tool.out" 2>/dev/null
  python3 "$HERE/buildcfg_record.py" extract "$L/build/semb" "$L/route_e_reach" > "$L/buildcfg.json"
  python3 "$HERE/codeid.py" witness "$L/codeid.json" "$L/build" "$L/eng/lib/libsuperslm.a"
  echo "   rc: pin=$prc configure=$crc build=$brc tool=$trc | wall $(( $(date +%s) - t0 )) s"
  grep -h "pin verified" "$L/configure.log" | grep -o "engine pin verified.*" | cut -c1-80 | sed 's/^/   /'
  [ -f "$L/tool.out" ] && grep encode "$L/tool.out" | sed 's/^/   tool: /'
}
leg i  "$SEAM -I@WD@/src/tests"
leg ii "$SEAM -I@WD@/src/tests -DSUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX"
leg c  "$SEAM -I@WD@/src/tests -DSUPERSLM_FORCE_SCALAR_MATMUL"
leg a  ""

FAILS=0
g() {  # leg config expect
  local out rc got ok=no
  out=$(python3 "$HERE/route_e_check.py" "$S/reference.json" "$S/leg-$1" "$2"); rc=$?
  got=$(echo "$out" | awk '$3=="FAIL"||$4=="FAIL"{print $1}' | paste -sd, -)
  if [ "$3" = PASS ]; then [ $rc -eq 0 ] && ok=yes; else [ $rc -ne 0 ] && [ "FAIL:$got" = "$3" ] && ok=yes; fi
  echo "-- leg $1 graded as $2 (expected $3; got $([ $rc -eq 0 ] && echo PASS || echo "FAIL:$got")): $([ $ok = yes ] && echo as-expected || echo UNEXPECTED)"
  echo "$out" | sed "s#$S#<S>#g"
  [ $ok = yes ] || FAILS=$((FAILS+1))
}
echo; echo "== grading (route_e_check.py; E3 is code identity against the reference engine)"
g i production PASS
g ii d_inf FAIL:E6
g ii production FAIL:E3,E6
g c production FAIL:E3
g a production FAIL:E3,E4,E6
echo
[ $FAILS -eq 0 ] && echo "CELL 10.0 route E (slice 1): GREEN -- every leg graded on its exact failing set, as expected" \
                 || echo "CELL 10.0 route E (slice 1): RED -- $FAILS grading(s) not as expected"
exit $FAILS
