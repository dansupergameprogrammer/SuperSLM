#!/usr/bin/env bash
# run_codeid_repro.sh -- is a same-commit clean build byte-identical on this host? (plan rev 9; the conductor's rev-9
# item 2, "first show that same-commit clean builds are byte-identical; if not, name the normalisation").
# usage: run_codeid_repro.sh <scratch> <candidate clone, tagged v1.8.0> [<runner scratch of run_route_e_legs.sh>]
# Rev 10 (F3): codeid.py build-reference needs CODEID_TOOLCHAIN_PIN (the consumer commit's cmake/toolchain-pin.json).
# 1. Two reference builds of the same commit (codeid.py build-reference: env -i, Release), from two different
#    source paths into two different build paths, for the release and the production configuration: raw archive
#    bytes, raw member bytes, and code_sha256.
# 2. With the runner's scratch: the libraries the consumer's pin tool built (leg i, production; q1-clean-a and
#    q1-clean-b, release) against the reference builds, raw and by code_sha256.
set -u
mkdir -p "$1"; S=$(cd "$1" && pwd); CAND=$2; RUN=${3:-}; HERE=$(cd "$(dirname "$0")" && pwd); BAD=0
SEAM=-DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT
for x in A B; do rm -rf "$S/src$x" "$S/b$x"-*; done
mkdir -p "$S/srcA" "$S/srcB/a/much/longer/source/path"; A="$S/srcA"; B="$S/srcB/a/much/longer/source/path"
git -C "$CAND" archive v1.8.0 | tar -x -C "$A"; git -C "$CAND" archive v1.8.0 | tar -x -C "$B"
echo "commit $(git -C "$CAND" rev-parse v1.8.0^{commit}); $(g++ --version | head -1); $(ar --version | head -1)"
cmpl() {  # label libA libB jsonA jsonB
  local n d=0; n=$(ar t "$2" | wc -l); rm -rf "$S/mx" "$S/my"; mkdir -p "$S/mx" "$S/my"; (cd "$S/mx" && ar x "$2"); (cd "$S/my" && ar x "$3")
  for f in "$S/mx"/*; do cmp -s "$f" "$S/my/$(basename "$f")" || d=$((d+1)); done
  local ra; cmp -s "$2" "$3" && ra=byte-identical || ra=DIFFERS
  local ca cb; ca=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['code_sha256'])" "$4"); cb=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['code_sha256'])" "$5")
  echo "$1: raw archive $ra; raw members $((n-d)) of $n identical; code_sha256 $([ "$ca" = "$cb" ] && echo "equal ${ca:0:16}" || echo "DIFFERS ${ca:0:16} ${cb:0:16}")"
  [ "$ra" = byte-identical ] && [ "$ca" = "$cb" ] || BAD=$((BAD+1))
}
for cfg in release production; do
  f=""; [ $cfg = production ] && f="$SEAM"
  python3 "$HERE/codeid.py" build-reference "$A" "$S/bA-$cfg" $f ${f:+-I$A/tests} > "$S/bA-$cfg.json"
  python3 "$HERE/codeid.py" build-reference "$B" "$S/bB-$cfg/deeper" $f ${f:+-I$B/tests} > "$S/bB-$cfg.json"
  cmpl "1. $cfg: clean build A against clean build B (different source and build paths)" "$S/bA-$cfg/libsuperslm.a" "$S/bB-$cfg/deeper/libsuperslm.a" "$S/bA-$cfg.json" "$S/bB-$cfg.json"
done
if [ -n "$RUN" ]; then
  for pr in "leg-i production" "q1-clean-a release" "q1-clean-b release"; do
    set -- $pr; python3 "$HERE/codeid.py" digest "$RUN/$1/eng/lib/libsuperslm.a" | python3 -c "import json,sys;json.dump(list(json.load(sys.stdin).values())[0],open(sys.argv[1],'w'))" "$S/$1.json"
    cmpl "2. the pin tool's build ($1, installed) against clean build A ($2)" "$RUN/$1/eng/lib/libsuperslm.a" "$S/bA-$2/libsuperslm.a" "$S/$1.json" "$S/bA-$2.json"
  done
fi
ar tv "$S/bA-release/libsuperslm.a" | head -1 | sed 's/^/   archive header (ar tv): /'
echo; [ $BAD -eq 0 ] && echo "REPRODUCIBLE: byte-identical on this host, no normalisation needed here" || echo "NOT BYTE-IDENTICAL: $BAD comparison(s) differ"
