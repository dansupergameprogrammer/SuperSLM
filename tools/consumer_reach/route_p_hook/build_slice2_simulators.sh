#!/usr/bin/env bash
# build_slice2_simulators.sh -- scratch fixture engines for the route P hook witness (plan rev 7; rev 8; the coverage
# auditor's slice-2 simulator, coverage-mutants-rev6 §3). NOT slice 2: two edits to sslm_abi.cpp, so that the H
# witness can be shown to read "hooked" when prefill really calls `run`.
#   optin : the install accepts reserved bit 0; sslm_prefill calls run once (two no-op tasks) when the bit is
#           set, max_tasks >= 2 and the call's batch (min(count, chunk_budget)) >= 8.
#   defon : the same, without the bit term (a default-on slice 2).
#   sim32 : (rev 10, adversary strike round 5 F2) opt-in, but the batch floor is 32, not 8: a stand-in for a
#           commissioned kMinMacsPerTask that threads only at larger batches.
# Rev 10 (F1): each simulator's reference engine is built from its commit (codeid.py build-reference), <out>/<v>/ref.json.
# Rev 8 (coverage-mutants-rev7 row 7a): each simulator is COMMITTED in its own scratch clone of the candidate and
# installed by the consumer's own pin tool, so its prefix carries share/superslm-provenance/provenance.json with
# its commit, as a real slice-2 candidate branch's engine would. The witness checks that commit before reading H.
# Rev 9: the pin step also records the installed library's sha256 (pin_record_lib.sh); the witness checks it too.
# usage: build_slice2_simulators.sh <candidate clone> <tag> <out-dir> <superembedder checkout> <superembedder commit>
#        -> <out>/{optin,defon,sim32}/eng (include, lib, share/superslm-provenance); <out>/<v>/commit; <out>/<v>/ref.json
# Rev 10 (F3): the reference build needs the toolchain pin (CODEID_TOOLCHAIN_PIN, exported by the runner).
set -eu
CAND=$1; TAG=$2; OUT=$3; SEMB_SRC=$4; SEMB_COMMIT=$5; mkdir -p "$OUT"
rm -rf "$OUT/semb"; mkdir -p "$OUT/semb"; git -C "$SEMB_SRC" archive "$SEMB_COMMIT" | tar -x -C "$OUT/semb"
for v in optin defon sim32; do
  D=$OUT/$v; rm -rf "$D"; mkdir -p "$D"; git clone -q "$CAND" "$D/clone"
  git -C "$D/clone" -c advice.detachedHead=false checkout -q "$TAG"; git -C "$D/clone" tag -d "$TAG" >/dev/null
  python3 - "$D/clone/src/sslm_abi.cpp" "$v" <<'PY'
import sys; p, v = sys.argv[1:3]; t = open(p).read()
old = "if (pf->reserved != 0 || pf->max_tasks < 0"
assert t.count(old) == 1; t = t.replace(old, "if ((pf->reserved & ~1u) != 0 || pf->max_tasks < 0")
anchor = "\tconst sslm_status st = PrefillWholeTokens(model, seq->state, seq->kv_block, seq->block_size,"
assert t.count(anchor) == 1
bit = "(ws->parallel_for.reserved & 1u) && " if v in ("optin", "sim32") else ""
floor = "32" if v == "sim32" else "8"   # rev 10: sim32, the adversary's round-5 simulator (a commissioned MAC floor)
sim = ("\t// SLICE-2 SIMULATOR (" + v + "; scratch fixture, plan rev 7): not slice 2.\n"
       "\tif (ws && ws->parallel_for.run && " + bit + "ws->parallel_for.max_tasks >= 2 &&\n"
       "\t    (count < chunk_budget ? count : chunk_budget) >= " + floor + ")\n"
       "\t\tws->parallel_for.run(ws->parallel_for.host_ctx, 2, [](void*, int32_t) {}, nullptr);\n")
t = t.replace(anchor, sim + anchor); open(p, "w").write(t)
PY
  git -C "$D/clone" -c user.name=planner -c user.email=planner@localhost commit -qam "SLICE-2 SIMULATOR ($v): scratch fixture, not slice 2"
  git -C "$D/clone" tag "$TAG"; git -C "$D/clone" rev-parse HEAD > "$D/commit"
  printf 'tag=%s\n' "$TAG" > "$D/superslm.lock"
  ( cd "$OUT" && PYTHONDONTWRITEBYTECODE=1 python3 semb/tools/pin-superslm/pin_superslm.py --lock "$D/superslm.lock" \
      --prefix "$D/eng" --source "$D/clone" --workdir "$D/work" ) > "$D/pin.log" 2>&1 || { echo "simulator $v: PIN FAIL"; tail -5 "$D/pin.log"; exit 1; }
  bash "$(dirname "$0")/pin_record_lib.sh" "$D/eng"   # rev 9: the pin step records the installed library's sha256
  rm -rf "$D/refsrc" "$D/ref"; mkdir -p "$D/refsrc"; git -C "$D/clone" archive "$TAG" | tar -x -C "$D/refsrc"   # rev 10: its reference engine (F1)
  python3 "$(dirname "$0")/../route_e/codeid.py" build-reference "$D/refsrc" "$D/ref" > "$D/ref.json"
  echo "simulator $v: installed at commit $(cut -c1-12 "$D/commit") (provenance commit $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['commit'][:12])" "$D/eng/share/superslm-provenance/provenance.json"))"
done
