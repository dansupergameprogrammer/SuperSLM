#!/usr/bin/env bash
# run_route_e_legs.sh -- plan §11.R's route E leg, its must-rejects, the coverage auditor's kill legs and fixtures,
# Q1's B5 pairs and fixtures, and (with the plugin arguments) route P's H witness (plan rev 8; rev 9).
# usage: run_route_e_legs.sh <scratch-dir> <candidate-engine-clone> <superembedder-checkout> <artifact.sslm>
#                           [<plugin Source/SuperSLMUnreal> <plugin's vendored engine include>]
# The candidate clone carries the stand-in (standin_counter.patch + standin_buildcfg.patch, record /2) tagged
# v1.8.0, and 1ca2803 tagged v1.7.1 (leg x2). The runner derives further candidates by cloning it into scratch
# (x3: the same code at another commit; vB: VERSION 1.8.1; the slice-2 simulators). Neither the SuperSLM nor the
# SuperEmbedder checkout is modified.
# E3 reads the build-configuration record the compiler embedded in the engine object, from the consumer's linked
# binaries (buildcfg_record.py). Rev 8: the record carries the optimisation and assertion state, encodes defined
# macros as D:<value> and undefined as U, and is kept by `used, retain`. Threat model (plan C11): accidental
# misconfiguration and wrong-engine linkage by a cooperative builder; deliberate tampering is out of scope.
# Rev 9 (the conductor's call): E3 and B5 compare CODE IDENTITY (codeid.py): the engine library each consumer binary
# actually linked must have the code_sha256 of the harness's own reference engine, built from the candidate commit
# in a clean env -i build with the declared Release flags. The record above is kept as a readable diagnosis.
# tE1/tE2/tMix*/tNoW and Q1's F rows are FIXTURES: files arranged to give one check a kill leg of its own.
# Rev 10 (adversary strike round 5): every reference engine is built with a certified toolchain from the embedder's
# toolchain pin, read from the consumer commit (F3); the witness hashes .gnu.lto_* sections (the LTO plane); route P's
# identity compares the linked library with the reference engine (F1); merge acceptance reads H at the reopening B (F2).
# Every leg and pair is graded on its exact outcome; the runner ends GREEN only if every one is as expected.
set -u
mkdir -p "$1"
S=$(cd "$1" && pwd); CAND=$2; SEMB_SRC=$3; ART=$4; SEMB_COMMIT=1ee74bf
HERE=$(cd "$(dirname "$0")" && pwd)
SEAM="-DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT"
# rev 10 (F3): the certified pairs, read from the consumer commit's own cmake/toolchain-pin.json (never duplicated)
git -C "$SEMB_SRC" show "$SEMB_COMMIT:cmake/toolchain-pin.json" > "$S/toolchain-pin.json" || { echo "no toolchain pin at $SEMB_COMMIT"; exit 1; }
export CODEID_TOOLCHAIN_PIN="$S/toolchain-pin.json"
# ---- fixtures (the channels; coverage-mutants-rev6 §4 and coverage-mutants-rev7 §2, verbatim where given) ----
printf '#define SUPERSLM_FORCE_SCALAR_MATMUL 1\n' > "$S/force_scalar.h"
printf '#!/bin/sh\nexec "$@" -DSUPERSLM_FORCE_SCALAR_MATMUL\n' > "$S/inj.sh"
printf '#!/bin/sh\ncase " $* " in *" -E "*) exec "$@";; esac; exec "$@" -DSUPERSLM_FORCE_SCALAR_MATMUL\n' > "$S/injE.sh"
mkdir -p "$S/l1" "$S/l2" "$S/shadow" "$S/shadow16" "$S/o0" "$S/lp" "$S/prag"
printf '#!/bin/sh\nexec "$@"\n' > "$S/l1/wrap.sh"; cp "$S/injE.sh" "$S/l2/wrap.sh"
printf '#!/bin/sh\nexec "$@" -O0\n' > "$S/o0/wrap.sh"
printf '#!/bin/sh\nexec "$@"\n' > "$S/lp/wrap.sh"
chmod +x "$S/inj.sh" "$S/injE.sh" "$S/l1/wrap.sh" "$S/l2/wrap.sh" "$S/o0/wrap.sh" "$S/lp/wrap.sh"
printf '#define SUPERSLM_FORCE_SCALAR_MATMUL 1\n#include_next <cstdint>\n' > "$S/shadow/cstdint"
printf '#define SUPERSLM_TEST_TILED_MIN_TOKENS 16\n#include_next <cstdint>\n' > "$S/shadow16/cstdint"
printf '#pragma GCC optimize("O0")\n#include_next <cstdint>\n' > "$S/prag/cstdint"
# ---- derived candidates ----
rm -rf "$S/cand-x3" "$S/cand-vB"
git clone -q "$CAND" "$S/cand-x3"; git -C "$S/cand-x3" -c advice.detachedHead=false checkout -q v1.8.0; git -C "$S/cand-x3" tag -d v1.8.0 >/dev/null
echo "x3: same code, another commit (plan rev 7 fixture)." >> "$S/cand-x3/README.md"
git -C "$S/cand-x3" -c user.name=planner -c user.email=planner@localhost commit -qam "x3 fixture: README line only"; git -C "$S/cand-x3" tag v1.8.0
git clone -q "$CAND" "$S/cand-vB"; git -C "$S/cand-vB" -c advice.detachedHead=false checkout -q v1.8.0
sed -i 's/^project(superslm VERSION 1\.8\.0/project(superslm VERSION 1.8.1/' "$S/cand-vB/CMakeLists.txt"
git -C "$S/cand-vB" -c user.name=planner -c user.email=planner@localhost commit -qam "vB fixture: VERSION 1.8.1"; git -C "$S/cand-vB" tag v1.8.1
# ---- REFERENCE, written before any leg runs. Provenance of each field is recorded in the file. ----
reference() { # clone tag out.json
  local src="$S/refsrc-$(basename "$3" .json)"; rm -rf "$src"; mkdir -p "$src"; git -C "$1" archive "$2" | tar -x -C "$src"
  local commit ver std; commit=$(git -C "$1" rev-parse "$2^{commit}")
  ver=$(sed -n 's/^project(superslm VERSION \([0-9.]*\).*/\1/p' "$src/CMakeLists.txt")
  std=gnu++$(sed -n 's/^set(CMAKE_CXX_STANDARD \([0-9]*\)).*/\1/p' "$src/CMakeLists.txt")
  local R="python3 $HERE/buildcfg_record.py reference $src $std"
  local rn; rn=$(basename "$3" .json)   # C: the reference engine for one config (rev 9); bash's dynamic scope gives it src and rn
  C() { local n=$1; shift; rm -rf "$S/refbuild-$rn-$n"; python3 "$HERE/codeid.py" build-reference "$src" "$S/refbuild-$rn-$n" "$@"; }
  cat > "$3" <<J
{"commit": "$commit", "version": "$ver", "std": "$std",
 "configs": {"production": ["SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT"],
             "d_inf": ["SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT", "SUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX"],
             "t100": ["SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT", "SUPERSLM_TEST_TILED_MIN_TOKENS=100"],
             "release": []},
 "release_flags": "$(python3 -c "import sys;sys.path.insert(0,'$HERE');import buildcfg_record as b;print(' '.join(b.RELEASE_FLAGS))")",
 "record": {"production": $($R SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT),
            "d_inf": $($R SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT SUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX),
            "t100": $($R SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT SUPERSLM_TEST_TILED_MIN_TOKENS=100),
            "release": $($R)},
 "code": {"production": $(C production -DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT -I$src/tests),
          "d_inf": $(C d_inf -DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT -I$src/tests -DSUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX),
          "t100": $(C t100 -DSUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT -I$src/tests -DSUPERSLM_TEST_TILED_MIN_TOKENS=100),
          "release": $(C release)},
 "provenance": {"commit": "READ from the candidate source: git rev-parse $2",
                "version": "READ from the candidate source: project(superslm VERSION) in CMakeLists.txt",
                "std": "READ from the candidate source: CMAKE_CXX_STANDARD (CMake default extensions on)",
                "configs": "DECLARED by the harness author: which define set is production, D-inf, t100 and release (Q1)",
                "release_flags": "DECLARED (rev 8): CMake's GNU default CMAKE_CXX_FLAGS_RELEASE; the pin tool configures Release",
                "record": "DERIVED: the build-configuration record the compiler embeds when compiling src/matmul.cpp from a git archive of the commit, env -i, the release flags and the declared defines only",
                "code": "DERIVED (rev 9): the reference engine, libsuperslm.a built from a git archive of the commit by env -i cmake -DCMAKE_BUILD_TYPE=Release with the declared defines only; codeid.py code_sha256"}}
J
}
reference "$CAND" v1.8.0 "$S/reference.json"
reference "$S/cand-vB" v1.8.1 "$S/reference-vB.json"
# the auditor's 3e fixture: the same derivation with a shadow on CPLUS_INCLUDE_PATH in the HARNESS environment
CPLUS_INCLUDE_PATH="$S/shadow" reference "$CAND" v1.8.0 "$S/reference-dirty.json"
echo "reference (before any leg): $(python3 -c "
import json,sys; r=json.load(open(sys.argv[1])); r['code']={k:v['code_sha256'] for k,v in r['code'].items()}; print(json.dumps(r))" "$S/reference.json" | sed "s#$S#<S>#g")"
FAILS=0
echo "reference derived in a dirty harness environment (CPLUS_INCLUDE_PATH=<S>/shadow), records against the clean one: $(python3 -c "
import json,sys; a=json.load(open(sys.argv[1]))['record']; b=json.load(open(sys.argv[2]))['record']
print('equal (env -i holds)' if a==b else 'DIFFERS: '+str({k:(a[k],b[k]) for k in a if a[k]!=b.get(k)}))" "$S/reference.json" "$S/reference-dirty.json"); reference engines (rev 9): $(python3 -c "
import json,sys; a=json.load(open(sys.argv[1]))['code']; b=json.load(open(sys.argv[2]))['code']
print('equal code_sha256 in every config' if all(a[k]['code_sha256']==b[k]['code_sha256'] for k in a) else 'DIFFER')" "$S/reference.json" "$S/reference-dirty.json")"
cmp -s <(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r['record'],{k:v['code_sha256'] for k,v in r['code'].items()})" "$S/reference.json") \
       <(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r['record'],{k:v['code_sha256'] for k,v in r['code'].items()})" "$S/reference-dirty.json") || FAILS=$((FAILS+1))
# rev 9: reproducibility -- the release reference engine built twice from the same commit, in two different
# source and build paths, must be identical (raw archive bytes and code_sha256)
R2="$S/repro-src/a/longer/path"; rm -rf "$S/repro-src"; mkdir -p "$R2"; git -C "$CAND" archive v1.8.0 | tar -x -C "$R2"
python3 "$HERE/codeid.py" build-reference "$R2" "$S/repro-build/b" > "$S/repro.json"
echo "reproducibility (rev 9): release reference engine rebuilt in another source and build path: raw archive $(cmp -s "$S/repro-build/b/libsuperslm.a" "$S/refbuild-reference-release/libsuperslm.a" && echo byte-identical || echo DIFFERS); code_sha256 $(python3 -c "
import json,sys; a=json.load(open(sys.argv[1]))['code']['release']['code_sha256']; b=json.load(open(sys.argv[2]))['code_sha256']; print('equal '+a[:16] if a==b else 'DIFFERS')" "$S/reference.json" "$S/repro.json")"
cmp -s "$S/repro-build/b/libsuperslm.a" "$S/refbuild-reference-release/libsuperslm.a" || FAILS=$((FAILS+1))
# rev 10 (F3): the reference toolchain, and the adversary's leg C (CXX=clang++ with an LLVM directory first on PATH)
echo "reference toolchain (rev 10): $(python3 -c "import json,sys;t=json.load(open(sys.argv[1]))['code']['release']['toolchain'];print(t['id'],t['version'],'frontend='+t['frontend'],t['path'])" "$S/reference.json"), certified by the consumer commit's cmake/toolchain-pin.json"
mkdir -p "$S/llvmfirst"; ln -sf /usr/bin/clang++-18 "$S/llvmfirst/clang++"
f3() { # label expected(refused|accepted) command...
  local lab=$1 want=$2; shift 2; local out rc got
  out=$("$@" 2>&1 >"$S/f3.json"); rc=$?
  got=$([ $rc -eq 3 ] && echo refused || { [ $rc -eq 0 ] && echo "accepted, code_sha256 $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['code_sha256'][:16])" "$S/f3.json")" || echo "error rc=$rc"; })
  echo "   F3 $lab: $got${out:+ -- ${out//$S/<S>}}"
  case "$got" in "$want"*) ;; *) echo "      UNEXPECTED (expected $want)"; FAILS=$((FAILS+1));; esac
}
echo "== F3: the reference build fails closed on an uncertified toolchain (codeid.py build-reference, exit 3)"
f3 "leg C: CXX=clang++, <S>/llvmfirst first on PATH" refused env CXX=clang++ PATH="$S/llvmfirst:$PATH" python3 "$HERE/codeid.py" build-reference "$R2" "$S/f3-c"
f3 "--cxx clang++-18 (Clang 18.1.3; the pin certifies Clang 18.1.8 with the MSVC frontend only)" refused python3 "$HERE/codeid.py" build-reference "$R2" "$S/f3-d" --cxx clang++-18
f3 "no toolchain pin" refused env -u CODEID_TOOLCHAIN_PIN python3 "$HERE/codeid.py" build-reference "$R2" "$S/f3-e"
f3 "CXX='g++ -march=native' (a compiler with arguments)" refused env CXX="g++ -march=native" python3 "$HERE/codeid.py" build-reference "$R2" "$S/f3-f"
f3 "--cxx g++-13 (GNU 13.3.0, certified)" "accepted, code_sha256 $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['code']['release']['code_sha256'][:16])" "$S/reference.json")" python3 "$HERE/codeid.py" build-reference "$R2" "$S/f3-g" --cxx g++-13
# rev 10 (the LTO plane): the adversary's A2 at object level -- matmul.cpp.o by the reference build's own compile
# command, then with fat and slim LTO appended; rev 9's digest (codeid_rev9.py) beside rev 10's
echo "== LTO plane (rev 10): matmul.cpp.o, the reference compile command as-is and with LTO appended (per-object digest)"
MMC=$(ninja -C "$S/refbuild-reference-release" -t commands CMakeFiles/superslm.dir/src/matmul.cpp.o | tail -1 | sed 's/ -MD -MT [^ ]* -MF [^ ]*//; s/ -o [^ ]* -c / -c /')
mkdir -p "$S/lto"
for v in ref:"" fat:"-flto=auto -ffat-lto-objects" slim:"-flto=auto" fatS:"-flto=auto -ffat-lto-objects -DSUPERSLM_FORCE_SCALAR_MATMUL"; do
  ( cd "$S/refbuild-reference-release" && eval "$MMC ${v#*:} -o $S/lto/${v%%:*}.o" )
done
python3 - "$S/lto" "$HERE" <<'PY'
import sys; sys.path.insert(0, sys.argv[2]); import codeid as n, codeid_rev9 as o
d = {v: open(f"{sys.argv[1]}/{v}.o", "rb").read() for v in ("ref", "fat", "slim", "fatS")}
r10 = {v: n.object_digest(b) for v, b in d.items()}; r9 = {v: o.object_digest(b) for v, b in d.items()}
for v in d:
    print(f"   {v:5} rev9 {r9[v][:16]} ({'= ref' if r9[v]==r9['ref'] else 'differs'})   rev10 {r10[v][:16]} ({'= ref' if r10[v]==r10['ref'] else 'differs'})")
ok = r9["fat"] == r9["ref"] and all(r10[v] != r10["ref"] for v in ("fat", "slim", "fatS"))
print("   LTO plane:", "as-expected (rev 9 could not see fat LTO; rev 10 does)" if ok else "UNEXPECTED")
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || FAILS=$((FAILS+1))
echo "host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); $(g++ --version | head -1); $(cmake --version | head -1); $(ld --version | head -1)"
leg() {  # name tag CXXFLAGS(@WD@ = the pin workdir, @L@ = the leg dir) [env for the pin tool only] [source clone] [post-build command] [env for the consumer configure and build] [rev 9: command run after the pin step, before the consumer configures]
  local name=$1 tag=$2 flags=$3 xenv=${4:-} src=${5:-$CAND} post=${6:-} cenv=${7:-} pre=${8:-} L="$S/leg-$1"; rm -rf "$L"; mkdir -p "$L/semb"
  git -C "$SEMB_SRC" archive "$SEMB_COMMIT" | tar -x -C "$L/semb"
  printf 'tag=%s\n' "$tag" > "$L/semb/superslm.lock"
  local cxx=${flags//@WD@/$L/work}; cxx=${cxx//@L@/$L}
  echo "== leg $name: tag=$tag CXXFLAGS='${cxx//$S/<S>}' ${xenv:+pin-tool env: ${xenv//$S/<S>}} ${cenv:+consumer env: $cenv} ${post:+post-build: ${post//$S/<S>}}"
  [ "$name" = dR2 ] && printf -- '-DSUPERSLM_FORCE_SCALAR_MATMUL\n' > "$L/force.rsp"
  local t0=$(date +%s.%N)
  ( cd "$L" && env CXXFLAGS="$cxx" $xenv python3 semb/tools/pin-superslm/pin_superslm.py --lock semb/superslm.lock \
      --prefix "$L/eng" --source "$src" --workdir "$L/work" --keep-workdir ) > "$L/pin.log" 2>&1
  local prc=$?
  [ $prc -eq 0 ] && bash "$HERE/../route_p_hook/pin_record_lib.sh" "$L/eng"   # rev 9: the pin step records the installed library's sha256
  [ -n "$pre" ] && ( eval "$pre" ) > "$L/pre.log" 2>&1
  env $cenv cmake -S "$L/semb" -B "$L/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$L/eng" > "$L/configure.log" 2>&1
  local crc=$?
  env $cenv ninja -C "$L/build" superembedder semb > "$L/build.log" 2>&1
  local brc=$? t3=$(date +%s.%N)
  g++ -O2 -std=c++20 -I"$L/semb/include" -I"$L/eng/include" -I"$S/refsrc-reference/tests" "$HERE/route_e_reach.cpp" \
      "$L/build/libsuperembedder.a" "$L/eng/lib/libsuperslm.a" -pthread -o "$L/route_e_reach" > "$L/tool-build.log" 2>&1
  local trc=$?
  [ $trc -eq 0 ] && "$L/route_e_reach" "$ART" > "$L/tool.out" 2>/dev/null
  [ -n "$post" ] && ( eval "$post" ) > "$L/post.log" 2>&1
  python3 "$HERE/buildcfg_record.py" extract "$L/build/semb" "$L/route_e_reach" > "$L/buildcfg.json"
  mkcid "$L/codeid.json" "$L/build" "$L/eng/lib/libsuperslm.a"   # rev 9: the code witness (E3)
  echo "   engine code linked: $(python3 -c "import json,sys;print('; '.join(k+' '+v.get('code_sha256','none')[:16] for k,v in json.load(open(sys.argv[1])).items()))" "$L/codeid.json")"
  printf '   rc: pin=%s configure=%s build=%s tool=%s | wall pin+configure+build %.1f s\n' $prc $crc $brc $trc "$(echo "$t3-$t0" | bc)"
  grep -h "pin verified" "$L/configure.log" | grep -o "engine pin verified.*" | cut -c1-80 | sed 's/^/   /'
  [ -f "$L/tool.out" ] && grep encode "$L/tool.out" | sed 's/^/   tool: /'
  echo "   embedded record (semb): $(python3 -c "import json,sys; f=json.load(open(sys.argv[1]))['files']; print(f[sys.argv[2]])" "$L/buildcfg.json" "$L/build/semb")"
  local O="$L/work/build/CMakeFiles/superslm.dir/src/matmul.cpp.o"
  echo "   report: vpmaddwd $(objdump -d "$O" 2>/dev/null | grep -c vpmaddwd); zmm $(objdump -d "$O" 2>/dev/null | grep -c zmm); threshold $(objdump -d --no-show-raw-insn "$O" 2>/dev/null | awk '/<_ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl>:/{f=1} f&&/cmp .*\$0x/{print $2, $3; exit}'); GemmInt8Accumulate insns $(objdump -d --no-show-raw-insn "$O" 2>/dev/null | awk '/^[0-9a-f]+ <_ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl>:/{f=1;next} f&&/^$/{exit} f{n++} END{print n+0}'); .text $(size -A "$O" 2>/dev/null | awk '$1==".text"{print $2}') B"
}
mkcid() {  # out.json consumer-build-dir tool-library : the code witness -- semb's engine resolved from the consumer's build tree, the tool's from the harness's own link command
  python3 "$HERE/codeid.py" witness "$@"
}
leg i    v1.8.0 "$SEAM -I@WD@/src/tests"
leg ii   v1.8.0 "$SEAM -I@WD@/src/tests -DSUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX"
leg d1   v1.8.0 "$SEAM -I@WD@/src/tests -D SUPERSLM_FORCE_SCALAR_MATMUL"
leg d2   v1.8.0 "$SEAM -I@WD@/src/tests -include $S/force_scalar.h"
leg d3   v1.8.0 "$SEAM -I@WD@/src/tests -D SUPERSLM_FORCE_SSE2_MATMUL"
leg d4   v1.8.0 "$SEAM -I@WD@/src/tests -D SUPERSLM_TEST_TILED_MIN_TOKENS=16"
leg d5   v1.8.0 "$SEAM -I@WD@/src/tests -D SUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX"
leg d6   v1.8.0 "$SEAM -I@WD@/src/tests -D SUPERSLM_FORCE_AVX2_MATMUL"
leg dL   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/inj.sh"
leg dI   v1.8.0 "$SEAM -I@WD@/src/tests -imacros $S/force_scalar.h"
leg a    v1.8.0 ""
leg c    v1.8.0 "$SEAM -I@WD@/src/tests -DSUPERSLM_FORCE_SCALAR_MATMUL"
leg x2   v1.7.1 "$SEAM -I@WD@/src/tests"
leg dE   v1.8.0 "$SEAM -I@WD@/src/tests" "CPLUS_INCLUDE_PATH=$S/shadow"
leg dE16 v1.8.0 "$SEAM -I@WD@/src/tests" "CPLUS_INCLUDE_PATH=$S/shadow16"
leg dLE  v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/injE.sh"
leg dR2  v1.8.0 "$SEAM -I@WD@/src/tests @@L@/force.rsp" "" "$CAND" ': > "$L/force.rsp"'
leg dS   v1.8.0 "$SEAM -I@WD@/src/tests -DSUPERSLM_FORCE_SCALAR_MATMUL" "" "$CAND" 'cmake -DCMAKE_CXX_FLAGS="$SEAM -I$L/work/src/tests" "$L/work/build"'
leg x3   v1.8.0 "$SEAM -I@WD@/src/tests" "" "$S/cand-x3"
leg vB   v1.8.1 "$SEAM -I@WD@/src/tests" "" "$S/cand-vB"
mkdir -p "$S/outside"; rm -rf "$S/outside/tests"; cp -r "$S/refsrc-reference/tests" "$S/outside/tests"
leg dH   v1.8.0 "$SEAM -I$S/outside/tests"
leg t100 v1.8.0 "$SEAM -I@WD@/src/tests -D SUPERSLM_TEST_TILED_MIN_TOKENS=100"
# rev 8: the coverage auditor's legs (coverage-mutants-rev7 §2.1, §2.2, §3) and the optimisation legs
leg d7   v1.8.0 "$SEAM -I@WD@/src/tests -D SUPERSLM_FORCE_AVX512_MATMUL"
leg d8   v1.8.0 "$SEAM -I@WD@/src/tests -DSUPERSLM_TILED_AVX512_MSVC=1"
leg dZ   v1.8.0 "$SEAM -I@WD@/src/tests -DSUPERSLM_FORCE_SCALAR_MATMUL="
leg dM   v1.8.0 "$SEAM -I@WD@/src/tests -DSUPERSLM_FORCE_SCALAR_MATMUL=-"
leg aT8  v1.8.0 "-DSUPERSLM_TEST_TILED_MIN_TOKENS=8"
leg dGC  v1.8.0 "$SEAM -I@WD@/src/tests" "" "$CAND" "" "LDFLAGS=-Wl,--gc-sections"
leg dO   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/o0/wrap.sh"
leg dOP  v1.8.0 "$SEAM -I@WD@/src/tests" "CPLUS_INCLUDE_PATH=$S/prag"
# rev 9: the coverage auditor's rev-8 legs (coverage-mutants-rev8 1d-1f, 2e) and the class the code witness closes
# (the conductor's rev-9 call): -O1/-O2, -march by launcher, by CXX and by a toolchain file, and a library swap
mkdir -p "$S/w" "$S/tc"
for f in Os:-Os UND:-UNDEBUG Og:-Og O1:-O1 O2:-O2 march:-march=native; do printf '#!/bin/sh\nexec "$@" %s\n' "${f#*:}" > "$S/w/${f%%:*}.sh"; done; chmod +x "$S/w/"*.sh
printf 'add_compile_options(-march=native)\n' > "$S/tc/march.cmake"
leg dOs   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/Os.sh"
leg dND   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/UND.sh"
leg dDBG  v1.8.0 "$SEAM -I@WD@/src/tests -D_DEBUG"
leg dOg   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/Og.sh"
leg dO1   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/O1.sh"
leg dO2   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/O2.sh"
leg dMa   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/march.sh"
CXX="g++ -march=native" leg dCXX v1.8.0 "$SEAM -I@WD@/src/tests"
leg dTC   v1.8.0 "$SEAM -I@WD@/src/tests" "CMAKE_TOOLCHAIN_FILE=$S/tc/march.cmake"
leg dSw   v1.8.0 "$SEAM -I@WD@/src/tests" "" "$CAND" "" "" 'cp "$S/leg-dO2/eng/lib/libsuperslm.a" "$L/eng/lib/libsuperslm.a"'
CXX="g++ -flto=auto -ffat-lto-objects" leg dLTO v1.8.0 "$SEAM -I@WD@/src/tests"   # rev 10: the adversary's round-5 leg
# fixtures: leg i's graded files with one input replaced (the only kill legs some checks can have)
fixture() { # name source-leg-for-files
  rm -rf "$S/leg-$1"; mkdir -p "$S/leg-$1/semb" "$S/leg-$1/eng/share"
  cp "$S/leg-$2/configure.log" "$S/leg-$2/tool.out" "$S/leg-$2/buildcfg.json" "$S/leg-$2/codeid.json" "$S/leg-$1/"; cp "$S/leg-$2/semb/superslm.lock" "$S/leg-$1/semb/"
  cp -r "$S/leg-$2/eng/share/superslm-provenance" "$S/leg-$1/eng/share/"
}
for T in tE1 tE2 tMix tMix2 tMix3 tNoW; do fixture $T i; done
X3C=$(git -C "$S/cand-x3" rev-parse v1.8.0^{commit}); RC=$(python3 -c "import json;print(json.load(open('$S/reference.json'))['commit'])")
sed -i "s/$RC/$X3C/" "$S/leg-tE1/configure.log"; sed -i "s/$RC/$X3C/" "$S/leg-tE2/semb/superslm.lock"
python3 "$HERE/buildcfg_record.py" extract "$S/leg-i/build/semb" "$S/leg-x2/route_e_reach" > "$S/leg-tMix/buildcfg.json"
python3 "$HERE/buildcfg_record.py" extract "$S/leg-i/build/semb" "$S/leg-d4/route_e_reach" > "$S/leg-tMix2/buildcfg.json"
python3 "$HERE/buildcfg_record.py" extract "$S/leg-d4/build/semb" "$S/leg-i/route_e_reach" > "$S/leg-tMix3/buildcfg.json"
mkcid "$S/leg-tMix/codeid.json" "$S/leg-i/build" "$S/leg-x2/eng/lib/libsuperslm.a"    # rev 9: the same mixes, for the code witness
mkcid "$S/leg-tMix2/codeid.json" "$S/leg-i/build" "$S/leg-d4/eng/lib/libsuperslm.a"
mkcid "$S/leg-tMix3/codeid.json" "$S/leg-d4/build" "$S/leg-i/eng/lib/libsuperslm.a"
rm -f "$S/leg-tNoW/buildcfg.json" "$S/leg-tNoW/codeid.json"
echo; echo "== fixtures: tE1/tE2 = leg i with the configure log's / the lock's commit replaced by x3's; tMix = leg i's semb + leg x2's tool (v1.7.1, no record); tMix2 = leg i's semb + leg d4's tool; tMix3 = leg d4's semb + leg i's tool; tNoW = leg i without buildcfg.json and codeid.json (rev 9: each mix also applied to the code witness)"
echo
echo "== grading (route_e_check.py against the reference; expected: PASS, or FAIL on exactly the named checks; rev 9: E3 is code identity)"
g() {  # leg config expect [reference] [label]
  local ref=${4:-$S/reference.json} out rc got
  out=$(python3 "$HERE/route_e_check.py" "$ref" "$S/leg-$1" "$2"); rc=$?
  got=$(echo "$out" | awk '$3=="FAIL"||$4=="FAIL"{print $1}' | paste -sd, -)
  local ok=no
  if [ "$3" = PASS ]; then [ $rc -eq 0 ] && ok=yes; else [ $rc -ne 0 ] && [ "FAIL:$got" = "$3" ] && ok=yes; fi
  echo "-- leg $1 graded as $2${4:+ against $(basename "$4")} (expected $3; got $([ $rc -eq 0 ] && echo PASS || echo "FAIL:$got")): $( [ $ok = yes ] && echo as-expected || echo UNEXPECTED)${5:+ -- $5}"
  echo "$out" | sed "s#$S#<S>#g"
  [ $ok = yes ] || FAILS=$((FAILS+1))
}
g i production PASS
g ii d_inf FAIL:E6
g d5 d_inf FAIL:E6
g ii production FAIL:E3,E6
g d5 production FAIL:E3,E6
for L in d1 d2 d3 d4 d6 dL dI c dE dE16 dLE dR2 dS; do g $L production FAIL:E3; done
g a production FAIL:E3,E4,E6
g x2 production FAIL:E1,E2,E3,E6
g x3 production FAIL:E1,E2
g vB production PASS "$S/reference-vB.json"
g vB production FAIL:E1,E2
g tE1 production FAIL:E1
g tE2 production FAIL:E2
g dH production FAIL:E4
g t100 t100 FAIL:E6
for L in d7 d8 dZ dM; do g $L production FAIL:E3; done
g aT8 production FAIL:E3,E4,E6
g dGC production PASS "" "consumer linked with -Wl,--gc-sections: the retained record survives"
g dO production FAIL:E3 "" "-O0 appended by a launcher: OPTIMIZE=U"
g dOP production FAIL:E3 "" "out of the model (a shadow header, plan §10); rev 9: the code witness sees the -O0 code, so it fails E3 as a side effect"
for L in dOs dND dDBG dOg dO1 dO2 dMa dCXX dTC; do g $L production FAIL:E3 "" "rev 9: the engine linked is not the reference engine"; done
g dSw production FAIL:E3 "" "rev 9: the -O2 library swapped under the pinned prefix; provenance (E1, E2) and the record still read the candidate"
g dLTO production FAIL:E3 "" "adversary round 5 (fat LTO through CXX); rev 10: matmul.cpp.o now differs too"
for L in tMix tMix2 tMix3 tNoW; do g $L production FAIL:E3; done
g i production PASS "$S/reference-dirty.json" "reference derived in a dirty harness environment"
g dE production FAIL:E3 "$S/reference-dirty.json" "reference derived in a dirty harness environment"
echo
[ $FAILS -eq 0 ] && echo "CELL 10.0 route E: GREEN -- every leg and fixture graded on its exact failing set, as expected" \
                 || echo "CELL 10.0 route E: RED -- $FAILS grading(s) not as expected"

# ---- Q1's B5: production pin-tool builds (no seam) and semb, recorded by q1_record.py IMMEDIATELY after each build ----
q1() {  # name CXXFLAGS [pin-tool env]
  local D=$S/q1-$1; rm -rf "$D"; mkdir -p "$D/semb"; git -C "$SEMB_SRC" archive "$SEMB_COMMIT" | tar -x -C "$D/semb"; printf 'tag=v1.8.0\n' > "$D/semb/superslm.lock"
  ( cd "$D" && env CXXFLAGS="$2" ${3:-} python3 -c "
import json,os,sys; keys=['CXXFLAGS','CFLAGS','LDFLAGS','CPLUS_INCLUDE_PATH','CPATH','C_INCLUDE_PATH','GCC_EXEC_PREFIX','COMPILER_PATH','CMAKE_CXX_COMPILER_LAUNCHER','CXX','CMAKE_TOOLCHAIN_FILE']
json.dump({k: os.environ.get(k,'') for k in keys}, open('pin-env.json','w'))" && env CXXFLAGS="$2" ${3:-} python3 semb/tools/pin-superslm/pin_superslm.py --lock semb/superslm.lock --prefix "$D/eng" \
      --source "$CAND" --workdir "$D/work" --keep-workdir ) > "$D/pin.log" 2>&1
  cmake -S "$D/semb" -B "$D/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$D/eng" > "$D/configure.log" 2>&1
  ninja -C "$D/build" semb > "$D/build.log" 2>&1
  python3 "$HERE/q1_record.py" "$D" > "$D/record.json"
  echo "   q1-$1: embedded record $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['embedded_buildcfg_record'])" "$D/record.json"); engine GemmInt8Accumulate insns $(objdump -d --no-show-raw-insn "$D/work/build/CMakeFiles/superslm.dir/src/matmul.cpp.o" 2>/dev/null | awk '/^[0-9a-f]+ <_ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl>:/{f=1;next} f&&/^$/{exit} f{n++} END{print n+0}') (report)"
}
pair() {  # label before after expected: EQUAL | REFUSED [note]  (rev 9: each side's engine against the candidate's reference engine, and its pin-verified commit)
  echo; echo "== Q1 B5 pair: $1 (expected $4)${5:+ -- $5}"
  local out rc; out=$(python3 "$HERE/q1_record.py" --compare "$S/q1-$2/record.json" "$S/q1-$3/record.json" --code-ref-before "$S/reference.json" \
      --code-ref-after "$S/reference.json" --commit-before "$RC" --commit-after "$RC" --record-reference "$S/reference.json"); rc=$?
  echo "$out" | sed "s#$S#<S>#g" | cut -c1-240
  local ok=no
  case $4 in EQUAL) [ $rc -eq 0 ] && ok=yes ;; REFUSED) [ $rc -eq 2 ] && ok=yes ;; esac
  [ $ok = yes ] && echo "   as-expected" || { echo "   UNEXPECTED (rc=$rc)"; FAILS=$((FAILS+1)); }
}
echo; echo "== Q1 builds"
q1 clean-a ""; q1 clean-b ""
q1 launcher "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/inj.sh"
q1 l1 "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/l1/wrap.sh"; q1 l2 "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/l2/wrap.sh"
q1 shadow "" "CPLUS_INCLUDE_PATH=$S/shadow"
q1 cx-a "-DSUPERSLM_FORCE_SCALAR_MATMUL"; q1 cx-b "-DSUPERSLM_FORCE_SCALAR_MATMUL"
# rev 8: the coverage auditor's pairs (coverage-mutants-rev7 §2.4) and the -O0 pairs
q1 sh-a "" "CPLUS_INCLUDE_PATH=$S/shadow"; q1 sh-b "" "CPLUS_INCLUDE_PATH=$S/shadow"
CXX="g++ -DSUPERSLM_FORCE_SCALAR_MATMUL" q1 cxx-a ""; CXX="g++ -DSUPERSLM_FORCE_SCALAR_MATMUL" q1 cxx-b ""
CXX=g++ q1 gxx-a ""
q1 lp-a "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/lp/wrap.sh"
printf '#!/bin/sh\nexec "$@" -march=native\n' > "$S/lp/wrap.sh"            # same path, content changed (A4): codegen changes, the record does not
q1 lp-b "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/lp/wrap.sh"
q1 o0-a "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/o0/wrap.sh"; q1 o0-b "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/o0/wrap.sh"
q1 prag-a "" "CPLUS_INCLUDE_PATH=$S/prag"; q1 prag-b "" "CPLUS_INCLUDE_PATH=$S/prag"
LDFLAGS=-Wl,--gc-sections q1 gc-a ""; LDFLAGS=-Wl,--gc-sections q1 gc-b ""
# rev 9: the coverage auditor's rev-8 pairs (coverage-mutants-rev8 4c, 4d, 4e) and -Og (2e) on the Q1 side
CXX="g++ -march=native" q1 cxxm ""
printf '# empty toolchain\n' > "$S/tc/tc.cmake"; q1 tcA "" "CMAKE_TOOLCHAIN_FILE=$S/tc/tc.cmake"
printf 'add_compile_options(-march=native)\n' > "$S/tc/tc.cmake"; q1 tcB "" "CMAKE_TOOLCHAIN_FILE=$S/tc/tc.cmake"   # same path, content changed (4e)
q1 og-a "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/Og.sh"; q1 og-b "" "CMAKE_CXX_COMPILER_LAUNCHER=$S/w/Og.sh"
CXX="g++ -flto=auto -ffat-lto-objects" q1 lto ""   # rev 10: the adversary's round-5 pair
# record-level fixtures (F1-F4, the auditor's; F5, rev 8: a record re-taken after its launcher was edited)
echo "$X3C" > "$S/x3.commit"
python3 - "$S" <<'PY'
import json, sys, os
S = sys.argv[1]; a = json.load(open(f"{S}/q1-clean-a/record.json")); b = json.load(open(f"{S}/q1-clean-b/record.json"))
def put(name, rec):
    os.makedirs(f"{S}/q1-{name}", exist_ok=True); json.dump(rec, open(f"{S}/q1-{name}/record.json", "w"), indent=1)
r = dict(a); r["pin_env"] = None; put("F1a", r)
r = dict(b); r["pin_env"] = None; put("F1b", r)
r = dict(b); r["pin_env"] = dict(b["pin_env"], CXXFLAGS="-O0"); put("F2", r)
r = dict(b); r["CMAKE_CXX_FLAGS"] = "-O0"; put("F3", r)
scalar = b["embedded_buildcfg_record"][0].replace("FORCE_SCALAR=U", "FORCE_SCALAR=D:1")
r = dict(b); r["embedded_buildcfg_record"] = sorted([b["embedded_buildcfg_record"][0], scalar]); put("F4", r)
# rev 9: fixtures for the code gate itself
r = dict(b); r.pop("_engine_code"); put("FC1", r)                                           # no code witness
lpb = json.load(open(f"{S}/q1-lp-b/record.json"))
r = dict(b); r["_engine_code"] = lpb["_engine_code"]; put("FC2", r)                          # another build's engine
x3 = open(f"{S}/x3.commit").read().strip()
r = dict(b); r["_pin_verified"] = x3; put("FC3", r)                                          # pin verified at another commit
PY
printf '#!/bin/sh\nexec "$@"\n' > "$S/lp/wrap.sh"                          # the launcher edited back, after lp-a and lp-b were linked
mkdir -p "$S/q1-F5"; python3 "$HERE/q1_record.py" "$S/q1-lp-a" > "$S/q1-F5/record.json"   # lp-a recorded late (4n)
pair "P1 positive control: two clean builds" clean-a clean-b EQUAL
pair "P2 rev 6's launcher pair" clean-a launcher REFUSED "rev 9: was VOID"
pair "P3 same-basename launchers (pass-through against -E-aware injector)" l1 l2 REFUSED "rev 9: was VOID"
pair "P4 include-path shadow during the pin tool only" clean-a shadow REFUSED "rev 9: was VOID"
pair "P5 both builds with the same non-empty CXXFLAGS" cx-a cx-b REFUSED
pair "A1 include-path shadow in BOTH builds" sh-a sh-b REFUSED
pair "A2 CXX carrying -DSUPERSLM_FORCE_SCALAR_MATMUL in both" cxx-a cxx-b REFUSED
pair "A3 CXX=g++ against A2's build" gxx-a cxx-a REFUSED
pair "A4 same launcher path, pass-through then -march=native" lp-a lp-b REFUSED "rev 9: was UNEQUAL; the after engine is not the reference engine"
pair "O1 -O0 launcher in BOTH builds (coverage-mutants-rev7 4l)" o0-a o0-b REFUSED
pair "O2 #pragma GCC optimize(\"O0\") shadow in BOTH builds (4m)" prag-a prag-b REFUSED "out of the model (a shadow header); rev 9: the code witness refuses it as a side effect (was EQUAL)"
pair "G1 consumer linked with -Wl,--gc-sections, both builds" gc-a gc-b EQUAL
pair "C1 CXX='g++ -march=native' against clean (coverage-mutants-rev8 4c)" clean-a cxxm REFUSED
pair "T1 an empty toolchain file against clean (4d)" clean-a tcA EQUAL "the same engine: an empty toolchain file changes no code"
pair "T2 the same toolchain path, empty then add_compile_options(-march=native) (4e, the auditor's beat)" tcA tcB REFUSED
pair "Og -Og launcher in BOTH builds (2e on the Q1 side)" og-a og-b REFUSED
pair "F1 neither record carries pin-env.json" F1a F1b EQUAL "rev 9: a configuration rule, now diagnosis only (was REFUSED); both engines are the reference"
pair "F2 after: CXXFLAGS=-O0 in the pin environment only" clean-a F2 EQUAL "rev 9: diagnosis only (was REFUSED)"
pair "F3 after: CMAKE_CXX_FLAGS=-O0 in the cache only" clean-a F3 EQUAL "rev 9: diagnosis only (was REFUSED)"
pair "F4 after: two records in one timing binary" clean-a F4 EQUAL "rev 9: diagnosis only (was REFUSED)"
pair "F5 lp-a recorded after its launcher was edited" clean-a F5 EQUAL "rev 9: diagnosis only (was REFUSED); lp-a's engine is the reference"
pair "L1 (adversary round 5) CXX='g++ -flto=auto -ffat-lto-objects' against clean" clean-a lto REFUSED "rev 10: matmul.cpp.o now differs too"
pair "FC1 after: no code witness" clean-a FC1 REFUSED
pair "FC2 after: the engine code of lp-b (-march=native)" clean-a FC2 REFUSED
pair "FC3 after: pin verified at x3's commit" clean-a FC3 REFUSED
echo; [ $FAILS -eq 0 ] && echo "ALL GRADINGS, PAIRS AND FIXTURES AS EXPECTED" || echo "$FAILS UNEXPECTED"
if [ -n "${5:-}" ] && [ -n "${6:-}" ]; then
  echo; echo "== route P: H at run time (engine identity, slice-2 simulators, plugin variants, dead-counter fixture)"
  bash "$HERE/../route_p_hook/build_slice2_simulators.sh" "$CAND" v1.8.0 "$S/sim" "$SEMB_SRC" "$SEMB_COMMIT" | sed "s#$S#<S>#g"
  OPT=$(cat "$S/sim/optin/commit"); DEF=$(cat "$S/sim/defon/commit")
  echo "   wrong-engine fixture: the default-on simulator's prefix handed to a merge-acceptance run whose candidate is the opt-in simulator's commit"
  rm -rf "$S/fx"; mkdir -p "$S/fx"; cp -r "$S/sim/optin/eng" "$S/fx/noprov"; rm -rf "$S/fx/noprov/share/superslm-provenance"
  cp -r "$S/sim/optin/eng" "$S/fx/libswap"; cp "$S/sim/defon/eng/lib/libsuperslm.a" "$S/fx/libswap/lib/libsuperslm.a"
  echo "   rev 9 fixtures: no-prov = the opt-in prefix without share/superslm-provenance (5a); lib-swap = the opt-in prefix with the default-on library installed over it (5d)"
  # rev 10 (F1): the adversary's pin-launcher leg -- the opt-in simulator's own clone pinned with a stray launcher that
  # forces scalar, pin_record_lib.sh run right after the pin tool (so checks 1 and 2 hold); check 3 must read VOID
  ( cd "$S/sim" && env CMAKE_CXX_COMPILER_LAUNCHER="$S/inj.sh" PYTHONDONTWRITEBYTECODE=1 python3 semb/tools/pin-superslm/pin_superslm.py \
      --lock "$S/sim/optin/superslm.lock" --prefix "$S/fx/pinlaunch" --source "$S/sim/optin/clone" --workdir "$S/fx/pinlaunch-work" ) > "$S/fx/pinlaunch.log" 2>&1 \
      && bash "$HERE/../route_p_hook/pin_record_lib.sh" "$S/fx/pinlaunch"
  echo "   rev 10 fixture: pin-launcher = the opt-in simulator pinned with CMAKE_CXX_COMPILER_LAUNCHER=<S>/inj.sh (pin rc=$?; provenance commit $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['commit'][:12])" "$S/fx/pinlaunch/share/superslm-provenance/provenance.json"))"
  python3 -c "import json,sys;json.dump(json.load(open(sys.argv[1]))['code']['production'],open(sys.argv[2],'w'))" "$S/reference.json" "$S/ref-standin.json"
  RO=$S/sim/optin/ref.json
  bash "$HERE/../route_p_hook/run_route_p_hook_witness.sh" "$5" "$6" "$S/rp" "$ART" \
      "standin=$S/leg-i/eng=$RC=$S/ref-standin.json" "sim-optin=$S/sim/optin/eng=$OPT=$RO" "sim-defon=$S/sim/defon/eng=$DEF=$S/sim/defon/ref.json" \
      "wrong-engine=$S/sim/defon/eng=$OPT=$RO" "no-prov=$S/fx/noprov=$OPT=$RO" "lib-swap=$S/fx/libswap=$OPT=$RO" \
      "pin-launcher=$S/fx/pinlaunch=$OPT=$RO" | sed "s#$S#<S>#g" | python3 "$HERE/../route_p_hook/grade_route_p.py" || FAILS=$((FAILS+1))
  # rev 10 (F2): merge acceptance at the reopening B (run_merge_acceptance.sh), the adversary's sim32 at B = 16 first
  echo; echo "== route P merge acceptance at the reopening B (rev 10, F2)"
  S32=$(cat "$S/sim/sim32/commit"); PSRC=$5
  mkrec() { printf '{"route": "%s", "B": %s, "m_print": {"route": "%s", "B": %s, "M": [%s]}}\n' "$2" "$3" "$4" "$5" "$6" > "$S/reopen-$1.json"; }
  mkrec P16 P 16 P 16 "16, 16"; mkrec P64 P 64 P 64 "64, 64"; mkrec P16m64 P 16 P 64 "64, 64"; mkrec E16 E 16 E 16 "16"; mkrec P4 P 4 P 4 "4"
  ma() { # label reopening engine-spec expected(ACCEPT|REFUSED)
    local out rc; out=$(bash "$HERE/../route_p_hook/run_merge_acceptance.sh" "$PSRC" "$S/ma" "$ART" "$S/reopen-$2.json" "$3" 2>&1); rc=$?
    echo "-- $1 (expected $4)"; echo "$out" | sed "s#$S#<S>#g"
    local ok=no; case $4 in ACCEPT) [ $rc -eq 0 ] && ok=yes;; REFUSED) [ $rc -eq 2 ] && ok=yes;; esac
    [ $ok = yes ] && echo "   as-expected" || { echo "   UNEXPECTED (rc=$rc)"; FAILS=$((FAILS+1)); }
  }
  ma "MA1 the adversary's leg: sim32 (threads only at batch >= 32), need at B = 16" P16 "sim32=$S/sim/sim32/eng=$S32=$S/sim/sim32/ref.json" REFUSED
  ma "MA2 positive control: the opt-in simulator (floor 8), need at B = 16" P16 "sim-optin=$S/sim/optin/eng=$OPT=$RO" ACCEPT
  ma "MA3 sim32 with a need at B = 64: the batch decides, not the engine" P64 "sim32=$S/sim/sim32/eng=$S32=$S/sim/sim32/ref.json" ACCEPT
  ma "MA4 need at B = 16 cited with an M print run at 64" P16m64 "sim-optin=$S/sim/optin/eng=$OPT=$RO" REFUSED
  ma "MA5 a route E need handed to route P's acceptance" E16 "sim-optin=$S/sim/optin/eng=$OPT=$RO" REFUSED
  ma "MA6 need at B = 4" P4 "sim-optin=$S/sim/optin/eng=$OPT=$RO" REFUSED
  ma "MA7 the pin-launcher engine (F1) at B = 16" P16 "pin-launcher=$S/fx/pinlaunch=$OPT=$RO" REFUSED
  echo; [ $FAILS -eq 0 ] && echo "RUN GREEN: route E, Q1 and route P all as expected" || echo "RUN RED: $FAILS unexpected"
fi
