"""Schema fast-decode plan rev 1.4.4 §11.2, cell G0.2 (slice G1-F; rev 1.4.2, strike N2): the status mirrors.

Every file in the tree that spells the status enum carries SSLM_GPU_FORCED_TABLE_ABSENT = 31 and
SSLM_GPU_CONTEXT_CAP_EXCEEDED = 32, appended after SSLM_RESTORE_PRODUCER_MISMATCH = 30, the seven files plan
§7.3 names included; and tests/ci/check_api_ordinals.py and tests/precache/check_prefix_status_mirrors.py pass.

A file spells the enum when it names 15 or more of gpu_1p0.h's SslmGpuStatus enumerators, or states
SSLM_RESTORE_PRODUCER_MISMATCH's ordinal (the name and the integer 30 on one line), or is one of the files §7.3
names. For each such file:
  - both new names appear, after SSLM_RESTORE_PRODUCER_MISMATCH's first appearance (an appended enumerator is
    written after it, in an enum, an X-macro, a name table or a switch);
  - when the file states ordinal 30 beside SSLM_RESTORE_PRODUCER_MISMATCH, it states 31 and 32 beside the new
    names the same way.
Records, release notes and the changelog are history, not mirrors, and are not read; nor are the engine's .cpp
sources under src/, which return the statuses rather than spell the enum.

Multi-row GPU errata E-3: the enum may end with exactly one more status, the branch-provisional
SSLM_GPU_ROWS_CHUNK_OPEN, until R-20; any other appended status is red. Each run first self-tests that rule (an appended
unruled status must go red; review M2).

Host-only: it runs in the cloud and in CI. Usage: python tests/schema-fast-decode-gpu/check_gpu_status_mirrors.py
"""

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LAST = "SSLM_RESTORE_PRODUCER_MISMATCH"
NEW = {"SSLM_GPU_FORCED_TABLE_ABSENT": 31, "SSLM_GPU_CONTEXT_CAP_EXCEEDED": 32}
# Plan §7.3 (rev 1.4.1 N3, rev 1.4.2 N2): the files the plan names; each must be found as a mirror.
NAMED = [
    "include/superslm/gpu_1p0.h",
    "include/superslm/sslm_abi.h",
    "tools/t2139_gate_c_type_identity_check.cpp",
    "tests/t2130-g5-red-suite/sslm_g5.h",
    "tests/ci/check_api_ordinals.py",
    "tests/precache/check_prefix_status_mirrors.py",
    "tests/t2138-abi-red-suite/sslm_abi.h",
    "tests/precache/precache_abi.h",
    "tests/precache-gpu/pcg_helpers.h",
    "tests/t2791-gpu-prefill-read-red-suite/cell_status_ordinals.cpp",
    "tests/t2956-token-finish-red-suite/cell_status_ordinals.cpp",
    "tests/t2112-gpu-1p0-red-suite/sslm_gpu_1p0.h",
]
SKIP_DIRS = {".git", "out", "build", "node_modules", "__pycache__"}
SKIP_FILES = {"CHANGELOG.md", "tests/ci/test_check_api_ordinals.py",
              "tests/schema-fast-decode-gpu/check_gpu_status_mirrors.py"}
SKIP_PREFIXES = ("docs/releases/", "Claude/", ".github/")
EXTS = {".h", ".hpp", ".cpp", ".inc", ".py", ".md", ".txt", ".hlsl", ".ps1", ".bat", ".cmake"}


def gpu_enumerators() -> list:
    text = (ROOT / "include/superslm/gpu_1p0.h").read_text(encoding="utf-8")
    m = re.search(r"enum\s+class\s+SslmGpuStatus[^{]*\{(.*?)\};", text, re.S)
    if not m:
        raise SystemExit("G0.2: no SslmGpuStatus enum in gpu_1p0.h")
    body = re.sub(r"/\*.*?\*/", " ", m.group(1), flags=re.S)
    body = re.sub(r"//[^\n]*", " ", body)
    return re.findall(r"\b(SSLM_[A-Z0-9_]+)\b", body)


# Multi-row GPU plan errata E-3 (rulings items 10 and 18, review M2): the one status allowed after 31 and 32 is the
# branch-provisional SSLM_GPU_ROWS_CHUNK_OPEN, until R-20 restores "the enum ends with the two".
E3_PROVISIONAL = ["SSLM_GPU_ROWS_CHUNK_OPEN"]


def tail_rule(names: list):
    """None when the enum's tail is exactly [LAST, *NEW] or E-3's [LAST, *NEW, *E3_PROVISIONAL]; else the failure."""
    want, allowed = [LAST, *NEW], [LAST, *NEW, *E3_PROVISIONAL]
    tail = names[-len(want):] if len(names) >= len(want) else names
    tail_e3 = names[-len(allowed):] if len(names) >= len(allowed) else names
    if tail == want or tail_e3 == allowed:
        return None
    return (f"include/superslm/gpu_1p0.h: the enum ends {names[-(len(allowed) + 1):]}, want it to end {want} "
            f"or, under errata E-3, {allowed}")


def self_test() -> list:
    """The tail rule sees the defects it exists for: an appended unruled status goes red (review M2)."""
    head = ["SSLM_OK", "SSLM_BUSY", LAST]
    cases = [
        ("1.17's own tail", head + list(NEW), True),
        ("E-3's provisional CHUNK_OPEN", head + list(NEW) + E3_PROVISIONAL, True),
        ("an unruled status after CHUNK_OPEN", head + list(NEW) + E3_PROVISIONAL + ["SSLM_GPU_UNRULED"], False),
        ("two unruled statuses after CHUNK_OPEN", head + list(NEW) + E3_PROVISIONAL + ["SSLM_GPU_A", "SSLM_GPU_B"], False),
        ("an unruled status instead of CHUNK_OPEN", head + list(NEW) + ["SSLM_GPU_UNRULED"], False),
        ("CHUNK_OPEN before 31 and 32", head + E3_PROVISIONAL + list(NEW), False),
        ("31 and 32 swapped", head + list(reversed(list(NEW))), False),
        ("a status between 30 and 31", head + ["SSLM_GPU_UNRULED"] + list(NEW), False),
    ]
    bad = []
    for name, names, green in cases:
        if (tail_rule(names) is None) != green:
            bad.append(f"self-test: the tail rule calls '{name}' {'red' if green else 'green'}, want {'green' if green else 'red'}")
    return bad


def states_ordinal(text: str, name: str, value: int) -> bool:
    return any(name in line and re.search(rf"(?<![\d.]){value}(?![\d.])", line) for line in text.splitlines())


def main() -> int:
    names = gpu_enumerators()
    failures = self_test()
    # The enum ends with the two, or (errata E-3) with the two and the provisional CHUNK_OPEN, and nothing else.
    tail_failure = tail_rule(names)
    if tail_failure:
        failures.append(tail_failure)
    mirrors = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if (path.is_dir() or path.suffix not in EXTS or any(p in SKIP_DIRS for p in path.parts)
                or rel in SKIP_FILES or rel.startswith(SKIP_PREFIXES)):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if LAST not in text and not any(n in text for n in NEW):
            continue
        # An engine source that returns the statuses is not a mirror (src/**/*.cpp); its headers are read.
        if rel.startswith("src/") and path.suffix == ".cpp":
            continue
        spelled = sum(1 for n in set(names) if re.search(rf"\b{n}\b", text))
        if spelled < 15 and not states_ordinal(text, LAST, 30) and rel not in NAMED:
            continue
        mirrors.append(rel)
        at = text.find(LAST)
        for n, v in NEW.items():
            pos = text.find(n, at if at >= 0 else 0)
            if pos < 0:
                failures.append(f"{rel}: spells the status enum but not {n} after {LAST}")
            elif states_ordinal(text, LAST, 30) and not states_ordinal(text, n, v):
                failures.append(f"{rel}: states {LAST} at 30 but not {n} at {v}")
    for rel in NAMED:
        if rel not in mirrors:
            failures.append(f"{rel}: named by plan §7.3 but not found as a status mirror")
    for script in ("tests/ci/check_api_ordinals.py", "tests/precache/check_prefix_status_mirrors.py"):
        r = subprocess.run([sys.executable, str(ROOT / script)], capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(f"{script} fails: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
    print(f"G0.2: {len(mirrors)} status mirror(s) read:")
    for rel in mirrors:
        print(f"  {rel}")
    for f in failures:
        print(f"FAIL {f}")
    print(f"check_gpu_status_mirrors: {'GREEN' if not failures else 'RED'} ({len(failures)} failure(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
