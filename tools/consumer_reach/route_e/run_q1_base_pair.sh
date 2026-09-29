#!/usr/bin/env bash
# run_q1_base_pair.sh -- Q1's B5 on the REAL pair shape (plan rev 8.1; rev 9).
# "before" = the shipped v1.8.0 base with only the build-configuration record constant applied in scratch;
# "after" = the candidate. Both production (no seam), both through the pin tool, each recorded by q1_record.py
# right after its build.
# Rev 9 (coverage-mutants-rev8 Qd, Qe; the conductor's rev-9 items 2 and 4):
#   * the shipped base is READ from the engine repository's v1.8.0 tag, not given as an argument;
#   * check D grades the diff scope (q1_before_scope.py): one file, no deletions, exactly the candidate's record block;
#   * check 1 is widened from matmul.cpp's .text to the whole reference engine: every other object identical;
#   * B5 is code identity: each side's linked engine against the reference engine built from ITS commit, and each
#     side's pin-verified commit against the expected one (before = the before tag, after = the candidate tag).
# usage: run_q1_base_pair.sh <scratch> <shipped engine repo, its v1.8.0 tag = the shipped base>
#                            <before clone, tagged v1.8.0> <candidate clone, tagged v1.8.0> <superembedder checkout>
# Ends "Q1 REAL-PAIR SHAPE: GREEN" when D, 1 and 2 hold, pair 3 is EQUAL and pair 4 is REFUSED; otherwise RED
# with the failing checks named.
set -u
mkdir -p "$1"; S=$(cd "$1" && pwd); SHIPPED=$2; BASE=$3; CAND=$4; SEMB_SRC=$5; SEMB_COMMIT=1ee74bf
HERE=$(cd "$(dirname "$0")" && pwd); FAILED=""
mkdir -p "$S/o0"; printf '#!/bin/sh\nexec "$@" -O0\n' > "$S/o0/wrap.sh"; chmod +x "$S/o0/wrap.sh"
SHIP=$(git -C "$SHIPPED" rev-parse v1.8.0^{commit}); BEF=$(git -C "$BASE" rev-parse v1.8.0^{commit}); AFT=$(git -C "$CAND" rev-parse v1.8.0^{commit})
echo "shipped: $SHIP (the engine repository's v1.8.0 tag)"
echo "before:  $BEF"
echo "after:   $AFT"
# D. diff scope
echo "D. before's diff from the shipped base"
python3 "$HERE/q1_before_scope.py" "$SHIPPED" "$BASE" v1.8.0 "$CAND" v1.8.0 && echo "   -- in scope" || { echo "   -- OUT OF SCOPE"; FAILED="$FAILED D"; }
# 1. the record changes no code but its own: reference engines of the shipped base and of before
refb() { local d="$S/src-$1"; rm -rf "$d" "$S/ref-$1"; mkdir -p "$d"; git -C "$2" archive "$3" | tar -x -C "$d"; python3 "$HERE/codeid.py" build-reference "$d" "$S/ref-$1" > "$S/ref-$1.json"; }
git -C "$BASE" cat-file -e "$SHIP" 2>/dev/null || git -C "$BASE" fetch -q "$SHIPPED" refs/tags/v1.8.0:refs/q1-shipped/v1.8.0
refb shipped "$BASE" "$SHIP"; refb before "$BASE" v1.8.0; refb after "$CAND" v1.8.0
python3 - "$S" "$HERE" <<'PY'
import json, sys, subprocess
S, H = sys.argv[1:3]
a = json.load(open(f"{S}/ref-shipped.json"))["members"]; b = json.load(open(f"{S}/ref-before.json"))["members"]
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
def text(p):
    subprocess.run(["ar", "x", "libsuperslm.a", "matmul.cpp.o"], cwd=f"{S}/ref-{p}", check=True)
    return subprocess.run(["python3", f"{H}/codeid.py", "text", f"{S}/ref-{p}/matmul.cpp.o"], capture_output=True, text=True, check=True).stdout.strip()
same_text = text("shipped") == text("before")
ok = diff in ([], ["matmul.cpp.o"]) and same_text
print(f"1. reference engines, shipped base and before: {len(a) - len(diff)} of {len(a)} objects identical; differing: {', '.join(diff) or 'none'}; "
      f"matmul.cpp.o .text {'identical' if same_text else 'DIFFERS'} -- {'as-expected' if ok else 'UNEXPECTED'}")
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || FAILED="$FAILED 1"
# 2. release records
rel() { python3 "$HERE/buildcfg_record.py" reference "$S/src-$1" gnu++20; }
RB=$(rel before); RC=$(rel after)
printf '{"record": {"release": %s}}\n' "$RC" > "$S/reference-release.json"
echo "2. release record, before: $RB"
echo "   release record, after:  $RC"
[ "$RB" = "$RC" ] && echo "   equal -- as-expected" || { echo "   differ -- UNEXPECTED"; FAILED="$FAILED 2"; }
# 3, 4. B5 pairs
q1() {  # name source-clone [pin-tool env]
  local D=$S/q1-$1; rm -rf "$D"; mkdir -p "$D/semb"; git -C "$SEMB_SRC" archive "$SEMB_COMMIT" | tar -x -C "$D/semb"; printf 'tag=v1.8.0\n' > "$D/semb/superslm.lock"
  ( cd "$D" && env CXXFLAGS= ${3:-} python3 -c "
import json,os; keys=['CXXFLAGS','CFLAGS','LDFLAGS','CPLUS_INCLUDE_PATH','CPATH','C_INCLUDE_PATH','GCC_EXEC_PREFIX','COMPILER_PATH','CMAKE_CXX_COMPILER_LAUNCHER','CXX','CMAKE_TOOLCHAIN_FILE']
json.dump({k: os.environ.get(k,'') for k in keys}, open('pin-env.json','w'))" && env CXXFLAGS= ${3:-} python3 semb/tools/pin-superslm/pin_superslm.py --lock semb/superslm.lock --prefix "$D/eng" \
      --source "$2" --workdir "$D/work" --keep-workdir ) > "$D/pin.log" 2>&1
  cmake -S "$D/semb" -B "$D/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$D/eng" > "$D/configure.log" 2>&1
  ninja -C "$D/build" semb > "$D/build.log" 2>&1
  python3 "$HERE/q1_record.py" "$D" > "$D/record.json"
  echo "   q1-$1: $(grep -o 'engine pin verified.*' "$D/configure.log" | cut -c1-70)"
}
pair() {  # n label before after expected
  local out rc; out=$(python3 "$HERE/q1_record.py" --compare "$S/q1-$3/record.json" "$S/q1-$4/record.json" --code-ref-before "$S/ref-before.json" \
      --code-ref-after "$S/ref-after.json" --commit-before "$BEF" --commit-after "$AFT" --record-reference "$S/reference-release.json"); rc=$?
  echo "$1. B5 pair: $2 (expected $5)"; echo "$out" | sed "s#$S#<S>#g" | cut -c1-220
  local got; case $rc in 0) got=EQUAL;; *) got=REFUSED;; esac
  [ "$got" = "$5" ] && echo "   got $got -- as-expected" || { echo "   got $got -- UNEXPECTED"; FAILED="$FAILED $1"; }
}
q1 before "$BASE"; q1 after "$CAND"; q1 before-o0 "$BASE" "CMAKE_CXX_COMPILER_LAUNCHER=$S/o0/wrap.sh"
pair 3 "before, after, both clean" before after EQUAL
pair 4 "before with an -O0 launcher, after clean" before-o0 after REFUSED
echo; [ -z "$FAILED" ] && echo "Q1 REAL-PAIR SHAPE: GREEN -- D, 1, 2, 3 and 4 as expected" || echo "Q1 REAL-PAIR SHAPE: RED -- failing:$FAILED"
