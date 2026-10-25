#!/usr/bin/env bash
# Precached-prefixes plan, cell 9.5 [E5 pin] (rev 1.18.14, section 15.34 (2); E5 run sheet piece R2): pin one 1.14.0
# SPX1 blob per geometry (pkv_def, pkv_odd, pkv_32k, pkv_qk) for F4's 9.5 on 1.16.0.
#
# Run on the release tree after R1 (the 1.14.0 bump). It refuses unless CMake and sslm_version.h both read 1.14.0 and
# the tree is clean, builds the library and tests/precache/reference/pin_9p5_spx1.cpp against it, generates the
# fixtures with tools/gen_paged_kv_fixture.py, runs the pin twice and requires the two runs bit-identical, reads each
# blob's producer independently (tests/precache/check_version_values.py --expect 1.14.0 --blob), then writes:
#   tests/precache/reference/pins-9.5/1.14.0_<stem>.spx1   each blob, raw
#   tests/precache/reference/pins-9.5/manifest.txt         per blob: fixture and blob SHA-256, D, cap, B, L, the commit
#
# Usage: tools/build_precache_9p5_pins.sh WORK_DIR [--allow-dirty]
#   --allow-dirty: a dry run on an uncommitted tree; the manifest's commit is then marked "-dirty" and nothing is copied
#   into the tree.
set -euo pipefail
work="${1:?usage: build_precache_9p5_pins.sh WORK_DIR [--allow-dirty]}"
dirty_ok="${2:-}"
repo="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
dest="$repo/tests/precache/reference/pins-9.5"
python3 "$repo/tests/precache/check_version_values.py" --expect 1.14.0
commit="$(git -C "$repo" rev-parse HEAD)"
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
	if [ "$dirty_ok" != "--allow-dirty" ]; then
		echo "the tree has uncommitted changes: the pin records the 1.14.0 commit it was built from" >&2
		exit 1
	fi
	commit="$commit-dirty"
fi
mkdir -p "$work/fixtures" "$work/pins-a" "$work/pins-b"
python3 "$repo/tools/gen_paged_kv_fixture.py" "$work/fixtures"
cmake -S "$repo" -B "$work/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$work/build" --target superslm -j"$(nproc)" >/dev/null
"${CXX:-g++}" -O2 -std=c++17 -I"$repo/include" -I"$repo/tests/precache/reference" \
	"$repo/tests/precache/reference/pin_9p5_spx1.cpp" \
	"$work/build/libsuperslm.a" -lpthread -o "$work/pin_9p5_spx1"
for run in a b; do
	"$work/pin_9p5_spx1" "$commit" "$work/fixtures" "$work/pins-$run" >/dev/null
done
diff -r "$work/pins-a" "$work/pins-b"
python3 "$repo/tests/precache/check_version_values.py" --expect 1.14.0 $(printf -- '--blob %s ' "$work"/pins-a/*.spx1)
cat "$work/pins-a/manifest.txt"
if [ "$dirty_ok" = "--allow-dirty" ]; then
	echo "dry run: pins left in $work/pins-a"
	exit 0
fi
mkdir -p "$dest"
cp "$work"/pins-a/*.spx1 "$work/pins-a/manifest.txt" "$dest/"
echo "pins written to $dest"
