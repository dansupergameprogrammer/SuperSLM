#!/usr/bin/env bash
# Paged-KV plan (rev 16.1) §8 and step R0: regenerate the byte-equality oracle's reference.
#
# Builds the library at two frozen tags in throwaway worktrees -- v1.11.0, the reference, and
# v1.8.1, the last SSB4 writer -- builds tests/paged-kv/reference/pkv_reference.cpp against each,
# generates the pkv fixtures with tools/gen_paged_kv_fixture.py, and writes:
#   tests/paged-kv/reference/<tag>_<fixture>.ref    every fixture, both tags
#   tests/paged-kv/reference/pins/<tag>_pkv_def_*.zrl  the pinned blobs (pkv_def only: its
#                                                    fixture is hermetic; pkv_qk's is not)
# The tree under work is never built here: the flat path through the new code is never the
# reference. Run twice; the outputs must be bit-identical (R0's gate).
#
# Usage: tools/build_paged_kv_reference.sh WORK_DIR
set -euo pipefail
work="${1:?usage: build_paged_kv_reference.sh WORK_DIR}"
repo="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
ref="$repo/tests/paged-kv/reference"
mkdir -p "$work/fixtures" "$work/pins" "$ref/pins"
python3 "$repo/tools/gen_paged_kv_fixture.py" "$work/fixtures"
for tag in v1.8.1 v1.11.0; do
	src="$work/src-$tag"
	[ -d "$src" ] || git -C "$repo" worktree add --detach "$src" "$tag"
	cmake -S "$src" -B "$work/build-$tag" -DCMAKE_BUILD_TYPE=Release >/dev/null
	cmake --build "$work/build-$tag" --target superslm -j"$(nproc)" >/dev/null
	"${CXX:-g++}" -O2 -std=c++17 -I"$src/include" "$ref/pkv_reference.cpp" \
		"$work/build-$tag/libsuperslm.a" -lpthread -o "$work/pkv_reference-$tag"
	for fx in pkv_def pkv_qk pkv_odd pkv_32k; do
		"$work/pkv_reference-$tag" "$tag" "$work/fixtures/$fx.sslm" "$ref/${tag}_$fx.ref" "$work/pins"
	done
	cp "$work/pins/${tag}"_pkv_def_*.zrl "$ref/pins/"
done
echo "reference written to $ref"
