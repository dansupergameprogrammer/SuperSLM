#!/usr/bin/env python3
"""CI/local structural check (T-1574 test suite split, §3, §7.2):
`SSLM_TEST(` occurrences across tests/test_main.cpp + tests/areas/*.cpp,
counted statically, must equal the running binary's own `--count-tests`
report. **What this is independent of: CMake and build.bat** -- a file
present on disk but omitted from either build surface's source list makes the
static count exceed the runtime one, and this check needs no build-system
cooperation to see that (Stage 1's drill, D-SLM594, induces exactly this).
**What this is NOT independent of: the SSLM_TEST macro itself** -- both
operands are counts of things the macro produced, so this check is
structurally blind to a test-shaped function that never received the macro at
all; tests/ci/check_test_definitions_registered.py (D-SLM592) is the
mechanism that closes that case.

Same invocation also performs §7.1's name+order+body-hash diff against
tests/support/test_order.txt and the symbol+body-hash diff against
tests/support/check_site_hashes.txt -- "one pass, two outputs, not two
scans": the static scan that locates each `SSLM_TEST(` call to count it is
the same scan whose balanced-brace extraction recomputes each test's body
hash.

Exit 0 iff: the static/runtime count matches, the name+order+body-hash diff
against test_order.txt is empty, and the symbol+body-hash diff against
check_site_hashes.txt is empty. Exit 1 otherwise, naming every discrepancy.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import _source_scan as scan  # noqa: E402

_SSLM_TEST_CALL_RE = re.compile(r"\bSSLM_TEST\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^,)]+?)\s*\)\s*")


def _candidate_binaries() -> list[str]:
    root = scan.REPO_ROOT
    return [
        os.path.join(root, "build", "Release", "superslm_tests.exe"),
        os.path.join(root, "build", "superslm_tests.exe"),
        os.path.join(root, "build", "superslm_tests"),
        os.path.join(root, "out", "superslm_tests.exe"),
        os.path.join(root, "out", "superslm_tests"),
    ]


def locate_or_build_binary() -> str:
    """Returns a path to a built superslm_tests binary, building it via
    `cmake --build build --target superslm_tests` (assumes `build/` is
    already configured, matching this project's CI convention) if none of
    the conventional output paths exist yet."""
    for path in _candidate_binaries():
        if os.path.isfile(path):
            return path
    build_dir = os.path.join(scan.REPO_ROOT, "build")
    if os.path.isdir(build_dir):
        subprocess.run(
            ["cmake", "--build", "build", "--target", "superslm_tests", "--config", "Release"],
            cwd=scan.REPO_ROOT, check=True,
        )
        for path in _candidate_binaries():
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(
        "no built superslm_tests binary found and `build/` is not configured -- run "
        "`cmake -B build` first, or build.bat, then re-run this check"
    )


def static_sslm_test_calls() -> dict[str, dict]:
    """name -> {order, hash, file, line} for every `SSLM_TEST(Name, Order) { ... }`
    call across the checked source files."""
    result: dict[str, dict] = {}
    for path in scan.checked_source_files():
        text = open(path, encoding="utf-8").read()
        masked = scan.mask_comments_and_literals(text)
        pairs = scan.match_braces(masked)
        lines = scan.line_of(text)
        for m in _SSLM_TEST_CALL_RE.finditer(masked):
            name = m.group(1)
            order = int(m.group(2))
            after = masked[m.end():]
            brace_offset = len(after) - len(after.lstrip())
            open_idx = m.end() + brace_offset
            if open_idx >= len(masked) or masked[open_idx] != "{":
                continue  # not a definition-site match (e.g. macro's own #define)
            close_idx = pairs.get(open_idx)
            if close_idx is None:
                continue
            body_text = text[open_idx:close_idx + 1]
            result[name] = {
                "order": order,
                "hash": scan.normalized_hash(body_text),
                "file": os.path.relpath(path, scan.REPO_ROOT),
                "line": lines[m.start()],
            }
    return result


def static_non_test_symbol_hashes() -> dict[str, str]:
    """qualified symbol name -> body hash, for every non-test function-like
    definition across the checked source files PLUS tests/support/*.{h,cpp}
    (mirrors the population tests/support/check_site_hashes.txt was generated
    from -- a relocated shared fixture like TwoLayerFixture lives in
    tests/support/ from Stage 0 onward, not in a checked test-content file)."""
    result: dict[str, str] = {}
    for path in scan.checked_source_files() + scan.support_source_files():
        text = open(path, encoding="utf-8").read()
        for fn in scan.find_function_definitions(text):
            body_text = text[fn["open"]:fn["close"] + 1]
            result[fn["qualified"]] = scan.normalized_hash(body_text)
    return result


def load_test_order() -> list[tuple[str, int, str]]:
    path = os.path.join(scan.REPO_ROOT, "tests", "support", "test_order.txt")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, order, h = line.split(",")
            rows.append((name, int(order), h))
    return rows


def load_check_site_hashes() -> list[tuple[str, str]]:
    path = os.path.join(scan.REPO_ROOT, "tests", "support", "check_site_hashes.txt")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, h = line.split(",")
            rows.append((name, h))
    return rows


def run_binary(binary: str, *args: str) -> str:
    proc = subprocess.run([binary, *args], cwd=scan.REPO_ROOT, capture_output=True, text=True, check=True)
    return proc.stdout


def main() -> int:
    failures: list[str] = []

    static_tests = static_sslm_test_calls()
    static_count = len(static_tests)

    binary = locate_or_build_binary()
    runtime_count = int(run_binary(binary, "--count-tests").strip())

    if static_count != runtime_count:
        failures.append(
            f"static SSLM_TEST( count ({static_count}) != runtime --count-tests ({runtime_count})"
        )

    runtime_rows: dict[str, int] = {}
    for line in run_binary(binary, "--list-tests").splitlines():
        line = line.strip()
        if not line:
            continue
        name, order = line.rsplit(",", 1)
        runtime_rows[name] = int(order)

    for name, order, expected_hash in load_test_order():
        if name not in runtime_rows:
            failures.append(f"test_order.txt: '{name}' is not currently registered at runtime")
            continue
        if runtime_rows[name] != order:
            failures.append(
                f"test_order.txt: '{name}' registered at order {runtime_rows[name]}, expected {order}"
            )
        if name not in static_tests:
            failures.append(f"test_order.txt: '{name}' has no static SSLM_TEST( call in source")
            continue
        actual_hash = static_tests[name]["hash"]
        if actual_hash != expected_hash:
            failures.append(
                f"test_order.txt: '{name}' body hash {actual_hash} != recorded {expected_hash} "
                f"({static_tests[name]['file']}:{static_tests[name]['line']})"
            )

    non_test_hashes = static_non_test_symbol_hashes()
    for name, expected_hash in load_check_site_hashes():
        if name not in non_test_hashes:
            failures.append(f"check_site_hashes.txt: symbol '{name}' not found in current source")
            continue
        actual_hash = non_test_hashes[name]
        if actual_hash != expected_hash:
            failures.append(
                f"check_site_hashes.txt: symbol '{name}' body hash {actual_hash} != recorded "
                f"{expected_hash}"
            )

    if failures:
        print("check_test_registration_complete.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"check_test_registration_complete.py: OK -- static {static_count} == runtime "
        f"{runtime_count}; {len(load_test_order())} test_order.txt rows and "
        f"{len(load_check_site_hashes())} check_site_hashes.txt rows all match"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
