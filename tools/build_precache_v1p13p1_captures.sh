#!/usr/bin/env bash
# Precached-prefixes plan, cell 9.9 [E5] (rev 1.18.1; E5 row: "Capture and pin 9.9's ... sequence saves on a
# v1.13.1 build (d697927)"): capture the 1.13.1 sequence saves the release tree must read.
#
# Builds the library at the frozen tag v1.13.1 in a throwaway worktree (refusing unless the tag peels to
# d697927, the commit plan 9.9 names), builds tests/precache/reference/capture_v1p13p1_seq_saves.cpp
# against it, generates pkv_def with tools/gen_paged_kv_fixture.py, runs the capture twice and requires the
# two runs bit-identical, then writes:
#   tests/precache/reference/v1.13.1_pkv_def_seqsaves.ref       one record per case
#   tests/precache/reference/pins/v1.13.1_pkv_def_<case>.zrl     each saved blob, zero-run encoded
# The tree under work is never built here: the oracle is the 1.13.1 build's own output.
#
# Usage: tools/build_precache_v1p13p1_captures.sh WORK_DIR
set -euo pipefail
work="${1:?usage: build_precache_v1p13p1_captures.sh WORK_DIR}"
repo="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
ref="$repo/tests/precache/reference"
tag=v1.13.1
want=d697927f6d5ce5fa73ecbb8660020a425518b923
got="$(git -C "$repo" rev-parse "$tag^{commit}")"
if [ "$got" != "$want" ]; then
	echo "$tag peels to $got, not $want (plan 9.9's predecessor commit)" >&2
	exit 1
fi
mkdir -p "$work/fixtures" "$work/pins-a" "$work/pins-b" "$ref/pins"
python3 "$repo/tools/gen_paged_kv_fixture.py" "$work/fixtures"
src="$work/src-$tag"
[ -d "$src" ] || git -C "$repo" worktree add --detach "$src" "$want"
cmake -S "$src" -B "$work/build-$tag" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$work/build-$tag" --target superslm -j"$(nproc)" >/dev/null
"${CXX:-g++}" -O2 -std=c++17 -I"$src/include" -I"$repo/tests/precache/reference" \
	"$repo/tests/precache/reference/capture_v1p13p1_seq_saves.cpp" \
	"$work/build-$tag/libsuperslm.a" -lpthread -o "$work/capture-$tag"
for run in a b; do
	"$work/capture-$tag" "$tag" "$want" "$work/fixtures/pkv_def.sslm" "$work/seqsaves-$run.ref" "$work/pins-$run"
done
cmp "$work/seqsaves-a.ref" "$work/seqsaves-b.ref"
diff -r "$work/pins-a" "$work/pins-b"
cp "$work/seqsaves-a.ref" "$ref/${tag}_pkv_def_seqsaves.ref"
cp "$work/pins-a/${tag}"_pkv_def_*.zrl "$ref/pins/"
echo "captures written to $ref"
