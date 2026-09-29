#!/usr/bin/env bash
# run_route_p_hook_witness.sh -- cell 10.0's H for route P, at run time (plan rev 6; rev 7 adds the simulators,
# the dead-counter fixture and k = 2, 8, 256 on every engine; rev 8 adds the engine-identity check and Hk_lt16;
# rev 9: identity also hashes the library the witness links against the pin step's record;
# rev 10: and compares its code with the reference engine built from the candidate commit, adversary round 5 F1).
# usage: run_route_p_hook_witness.sh <plugin Source/SuperSLMUnreal dir> <engine include dir> <scratch>
#                                    <artifact.sslm> <name>=<engine prefix>=<expected commit>=<reference engine json> ...
# Each plugin variant's SuperSLMFinishHook.cpp is compiled verbatim (control) or with a one-line mutant, then:
#   value print: what Make(k) produces (report only);
#   identity:    rev 8 (coverage-mutants-rev7 row 7a): the prefix the witness links must be the expected
#                commit, read from the provenance the pin tool wrote at install (engine_identity.py). Otherwise
#                every reading on that engine is "VOID (engine identity ...)" and H is not read;
#   run count:   Make(k)'s value installed on a real CPU workspace of each engine, run calls counted in a
#                64-token single-call prefill and one decode step (route_p_run_count.cpp) -> H, by the rule
#                stated there (accepted AND decode live AND prefill runs > 0).
set -u
SRC=$1; INC=$2; W=$3; ART=$4; shift 4; ENGINES="$*"; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"
echo "== engine identity (rev 8; rev 9; rev 10): provenance commit, pinned-library hash, and code against the reference engine"
for spec in $ENGINES; do r=${spec#*=}; e=${r%%=*}; r=${r#*=}; echo "   [${spec%%=*}] $(python3 "$HERE/engine_identity.py" "$e" "${r%%=*}" "$e/lib/libsuperslm.a" "${r#*=}")"; done
runcount() { # variant-dir engine-name engine-prefix=expected-commit [extra -D]
  local V=$1 n=$2 e=${3%%=*} r=${3#*=} x=${4:-} o="$1/rc-$2${4:+-dead}" id; local want=${r%%=*} ref=${r#*=}
  id=$(python3 "$HERE/engine_identity.py" "$e" "$want" "$e/lib/libsuperslm.a" "$ref")   # rev 9: the library linked below; rev 10: against the reference engine
  if [ "$id" != ok ]; then for k in 2 8 256; do echo "   [$n${x:+, dead counter}] H(k=$k) = VOID (engine identity: $id)"; done; return; fi
  g++ -std=c++20 -O1 $x -I"$V" -I"$HERE/shim" -I"$e/include" "$V/SuperSLMFinishHook.cpp" "$HERE/route_p_run_count.cpp" \
      "$e/lib/libsuperslm.a" -lpthread -o "$o" 2>>"$V/cc.err" || { echo "   [$n] run-count: COMPILE FAIL"; return; }
  for k in 2 8 256; do "$o" "$ART" $k | grep "H(k" | sed "s/^route P: /   [$n${x:+, dead counter}] /"; done
}
variant() { # name regex replacement
  local V="$W/$1"; rm -rf "$V"; mkdir -p "$V"; cp "$SRC/Private/SuperSLMFinishHook.cpp" "$SRC/Private/SuperSLMFinishHook.h" "$V/"
  [ -n "$2" ] && python3 -c "import sys,re;p=sys.argv[1];t=open(p).read();n=re.sub(sys.argv[2],sys.argv[3],t,count=1);assert n!=t,'mutant did not apply';open(p,'w').write(n)" "$V/SuperSLMFinishHook.cpp" "$2" "$3"
  g++ -std=c++20 -I"$V" -I"$HERE/shim" -I"$INC" "$V/SuperSLMFinishHook.cpp" "$HERE/route_p_hook_witness.cpp" -o "$V/w" 2>"$V/cc.err" || { echo "== $1: COMPILE FAIL"; head -5 "$V/cc.err"; return; }
  echo "== $1"; "$V/w" | tail -1 | sed "s/^route P: H = /   value print (report): /"
  for spec in $ENGINES; do runcount "$V" "${spec%%=*}" "${spec#*=}"; done
}
variant control "" ""
variant H1_hex_zero  'Hook\.reserved = 0;' 'Hook.reserved = 0x0;'
variant H2_named_zero 'Hook\.reserved = 0;' 'constexpr uint32_t kSslmHookReservedNone = 0; Hook.reserved = kSslmHookReservedNone;'
variant H3_reserved_deleted_Hook_value_init '(?s)sslm_parallel_for Hook;(.*?)\n\tHook\.reserved = 0;' 'sslm_parallel_for Hook{};\1'
variant Hplus_prefill_bit 'Hook\.reserved = 0;' 'Hook.reserved = 1u;'
variant Hbad_other_bit 'Hook\.reserved = 0;' 'Hook.reserved = 2u;'
variant Hplus_named_bit 'Hook\.reserved = 0;' 'constexpr uint32_t kPrefill = 1u << 0; Hook.reserved = /* 0 */ kPrefill;'
variant Hk16_bit_only_at_k_ge_16 'Hook\.reserved = 0;' 'Hook.reserved = FinishParallelTasks >= 16 ? 1u : 0u;'
variant Hk_lt16 'Hook\.reserved = 0;' 'Hook.reserved = FinishParallelTasks < 16 ? 1u : 0u;'   # rev 8: the auditor's variant (5e2)
echo "== dead-counter fixture (the witness installs Make's value unwrapped; it must read VOID, never 'not hooked')"
for v in control Hplus_prefill_bit; do
  for spec in $ENGINES; do runcount "$W/$v" "${spec%%=*}" "${spec#*=}" -DWITNESS_DEAD_COUNTER | sed "s/^/  $v /"; done
done
