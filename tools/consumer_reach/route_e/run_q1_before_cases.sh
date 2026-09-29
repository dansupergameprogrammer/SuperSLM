#!/usr/bin/env bash
# run_q1_before_cases.sh -- run_q1_base_pair.sh on the real "before" and on the coverage auditor's two rev-8.1 beats
# (plan rev 9; coverage-mutants-rev8 Qd, Qe), each graded on its expected verdict.
#   real : before = the shipped base + the record constant                      -> GREEN
#   Qd   : before = the shipped base + the record + a code edit to sslm_abi.cpp  -> RED, failing D and 1
#   Qe   : before = the candidate itself                                         -> RED, failing D (and 1)
# usage: run_q1_before_cases.sh <scratch> <shipped engine repo> <base+record clone> <candidate clone> <superembedder checkout>
set -u
mkdir -p "$1"; S=$(cd "$1" && pwd); SHIPPED=$2; BASE=$3; CAND=$4; SEMB=$5; HERE=$(cd "$(dirname "$0")" && pwd); BAD=0
rm -rf "$S/qd-clone"; git clone -q "$BASE" "$S/qd-clone"; git -C "$S/qd-clone" -c advice.detachedHead=false checkout -q v1.8.0
git -C "$S/qd-clone" tag -d v1.8.0 >/dev/null
printf '\n// Qd fixture (plan rev 9, coverage-mutants-rev8 Qd): an edit outside matmul.cpp.\nextern "C" int superslm_qd_fixture(void) { return 7; }\n' >> "$S/qd-clone/src/sslm_abi.cpp"
git -C "$S/qd-clone" -c user.name=planner -c user.email=planner@localhost commit -qam "Qd fixture: base + record + an sslm_abi.cpp edit"; git -C "$S/qd-clone" tag v1.8.0
case_() {  # name before-clone expected-final-line-regex
  echo; echo "######## case $1 (expected: $3)"
  bash "$HERE/run_q1_base_pair.sh" "$S/$1" "$SHIPPED" "$2" "$CAND" "$SEMB" > "$S/$1.out" 2>&1
  sed "s#$S#<S>#g" "$S/$1.out"
  tail -1 "$S/$1.out" | grep -Eq "$3" && echo "######## case $1: as-expected" || { echo "######## case $1: UNEXPECTED"; BAD=$((BAD+1)); }
}
case_ real "$BASE" "GREEN"
case_ Qd "$S/qd-clone" "RED -- failing: D 1$"
case_ Qe "$CAND" "RED -- failing: D( 1)?$"
echo; [ $BAD -eq 0 ] && echo "Q1 BEFORE CASES: GREEN -- real GREEN; Qd and Qe RED on the diff scope" || echo "Q1 BEFORE CASES: RED -- $BAD unexpected"
