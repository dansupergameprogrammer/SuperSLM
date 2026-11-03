#!/usr/bin/env bash
# mingw-w64 g++ wrapper for the multirow-gpu suite's cloud builds: passes every argument through and, on a
# compile (-c or -fsyntax-only), force-includes mingw_d3d12_compat.h after the caller's own -include list.
# Use as CXX=tests/multirow-gpu/mingw_cxx.sh with tools/cloud_gpu/wine_gpu.sh or any mingw build script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="${MR_REAL_CXX:-x86_64-w64-mingw32-g++-posix}"
for a in "$@"; do
	case "$a" in
		-c|-fsyntax-only) exec "$REAL" "$@" -include "$HERE/mingw_d3d12_compat.h" ;;
	esac
done
exec "$REAL" "$@"
