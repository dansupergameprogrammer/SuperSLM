#!/usr/bin/env bash
# run_merge_acceptance.sh -- slice 2's merge acceptance on route P (plan rev 10, §5; adversary strike round 5 F2).
# Rev 9 read merge acceptance's H at the witness's fixed 64-token prefill, so an engine that threads only at
# batches of 32 or more read "hooked" for a need recorded at B = 16. Rev 10: H and M are read AT THE B THAT
# REOPENED slice 2, on the same route, and acceptance is one verdict:
#   1. the reopening record: its route is P (this script is route P's), B >= 8, and the route's M print it cites
#      is for route P, run at that B, and shows GEMM calls at M = B only (§5's reopening rule, re-checked here);
#   2. engine identity (engine_identity.py: provenance commit, pinned-library hash, code against the reference
#      engine built from the candidate commit -- rev 10 F1); otherwise VOID;
#   3. H at B: the plugin's Make with the prefill bit set (the Hplus_prefill_bit variant: the one-line change the
#      plugin's own follow-up makes, wired in scratch) installed on the candidate engine, then ONE prefill call of
#      B tokens (route_p_run_count.cpp's B argument; the call must consume all B tokens, so its GEMMs ran at
#      M = B), at k = 2, 8 and 256. Every reading must be "hooked".
#   => ACCEPT only if 1, 2 and 3 hold; otherwise REFUSED, naming which. 10.2 then runs (not executed here).
# usage: run_merge_acceptance.sh <plugin Source/SuperSLMUnreal dir> <scratch> <artifact.sslm> <reopening.json>
#                                <name>=<engine prefix>=<expected commit>=<reference engine json>
# exit: 0 ACCEPT, 2 REFUSED
set -u
SRC=$1; W=$2; ART=$3; REC=$4; spec=$5; HERE=$(cd "$(dirname "$0")" && pwd)
n=${spec%%=*}; r=${spec#*=}; e=${r%%=*}; r=${r#*=}; want=${r%%=*}; ref=${r#*=}
B=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['B'])" "$REC")
echo "== merge acceptance [$n] at the reopening B = $B ($(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print('need: route '+r['route']+', B '+str(r['B'])+'; M print: route '+r['m_print']['route']+', run at B '+str(r['m_print']['B'])+', M '+','.join(map(str,r['m_print']['M'])))" "$REC"))"
why=()
m=$(python3 - "$REC" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); p = r["m_print"]; bad = []
if r["route"] != "P": bad.append(f"the need is on route {r['route']}, not route P")
if r["B"] < 8: bad.append(f"B = {r['B']} < 8")
if p["route"] != r["route"]: bad.append(f"the M print is route {p['route']}'s, the need is route {r['route']}'s")
if p["B"] != r["B"]: bad.append(f"the M print was run at B = {p['B']}, the need is at B = {r['B']}")
if not p["M"] or any(x != r["B"] for x in p["M"]): bad.append(f"the M print shows M = {p['M']}, not M = {r['B']} only")
print("; ".join(bad) or "ok")
PY
)
echo "   1. reopening record: $m"; [ "$m" = ok ] || why+=("M: $m")
id=$(python3 "$HERE/engine_identity.py" "$e" "$want" "$e/lib/libsuperslm.a" "$ref")
echo "   2. engine identity: $id"
if [ "$id" != ok ]; then
  why+=("VOID: engine identity: $id")
else
  V="$W/ma-$n"; rm -rf "$V"; mkdir -p "$V"; cp "$SRC/Private/SuperSLMFinishHook.cpp" "$SRC/Private/SuperSLMFinishHook.h" "$V/"
  python3 -c "import sys,re;p=sys.argv[1];t=open(p).read();n=re.sub(r'Hook\.reserved = 0;','Hook.reserved = 1u;',t,count=1);assert n!=t;open(p,'w').write(n)" "$V/SuperSLMFinishHook.cpp"
  g++ -std=c++20 -O1 -I"$V" -I"$HERE/shim" -I"$e/include" "$V/SuperSLMFinishHook.cpp" "$HERE/route_p_run_count.cpp" \
      "$e/lib/libsuperslm.a" -lpthread -o "$V/rc" 2>"$V/cc.err" || { echo "   COMPILE FAIL"; head -3 "$V/cc.err"; exit 2; }
  for k in 2 8 256; do
    line=$("$V/rc" "$ART" $k "$B" | grep "H(k"); echo "   3. ${line#route P: }"
    case "$line" in *"= hooked "*) ;; *) why+=("H(k=$k, B=$B): ${line#*= }");; esac
  done
  for k in 2 8 256; do echo "      (report, not graded: the rev-9 fixed prefill) $("$V/rc" "$ART" $k | grep "H(k" | sed 's/^route P: //')"; done
fi
if [ ${#why[@]} -eq 0 ]; then echo "   => ACCEPT (then 10.2)"; exit 0; fi
echo "   => REFUSED -- $(IFS='|'; echo "${why[*]}" | sed 's/|/ | /g')"; exit 2
