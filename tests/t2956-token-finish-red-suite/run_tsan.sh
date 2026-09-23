#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tests/t2956-token-finish-red-suite/out
exe=tests/t2956-token-finish-red-suite/out/cell_reference_run_tsan
g++ -std=c++20 -O1 -g -Wall -Wextra -pthread -fsanitize=thread -fno-omit-frame-pointer \
    -I. -Iinclude tests/t2956-token-finish-red-suite/cell_reference_run_tsan.cpp -o "$exe"
TSAN_OPTIONS=halt_on_error=1:exitcode=66 "$exe" r \
    >tests/t2956-token-finish-red-suite/out/reference_tsan.log 2>&1
set +e
TSAN_OPTIONS=halt_on_error=1:exitcode=66 "$exe" p \
    >tests/t2956-token-finish-red-suite/out/probe_tsan.log 2>&1
status=$?
set -e
if [ "$status" -ne 66 ] || ! grep -q 'WARNING: ThreadSanitizer: data race' \
    tests/t2956-token-finish-red-suite/out/probe_tsan.log; then
    cat tests/t2956-token-finish-red-suite/out/probe_tsan.log
    echo "FAIL probe Pool did not fire a ThreadSanitizer data race" >&2
    exit 1
fi
cat tests/t2956-token-finish-red-suite/out/reference_tsan.log
echo "PASS probe Pool rejected by ThreadSanitizer"
