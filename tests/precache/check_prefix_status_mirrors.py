"""Gate C source census for the appended prefix producer status."""

from pathlib import Path
import re


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    status = "SSLM_RESTORE_PRODUCER_MISMATCH"
    paths = {
        "public X-macro": root / "include/superslm/sslm_abi.h",
        "G5 mirror": root / "tests/t2130-g5-red-suite/sslm_g5.h",
        "T2138 mirror": root / "tests/t2138-abi-red-suite/sslm_abi.h",
        "Gate C identity mirror": root / "tools/t2139_gate_c_type_identity_check.cpp",
        "API documentation": root / "docs/api.md",
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    assert re.search(rf"X\({status}\)", text["public X-macro"]), "public status absent"
    for name in ("G5 mirror", "Gate C identity mirror"):
        assert re.search(rf"\b{status}\s*=\s*30\b", text[name]), f"{name} lacks ordinal 30"
    positional = text["T2138 mirror"]
    assert re.search(rf"\bSSLM_KV_BUDGET_EXCEEDED\s*,.*?\b{status}\s*,"
                     rf".*?\bSSLM_STATUS_NEXT_FREE\b", positional, re.S), (
        "T2138 positional mirror lacks the appended status before its sentinel")
    assert re.search(rf"\b{status}\b[^\n]*\b30\b", text["API documentation"]), (
        "API documentation lacks the status and ordinal")
    # 1.18.0 (delta prefix saves, cell D14/E6): the parent status, appended after it at 31.
    parent = "SSLM_RESTORE_PARENT_MISMATCH"
    assert re.search(rf"X\({parent}\)", text["public X-macro"]), "parent status absent"
    assert re.search(r"#define SSLM_HAS_PREFIX_DELTA_ABI 1\b", text["public X-macro"]), "delta ABI macro absent"
    for name in ("G5 mirror", "Gate C identity mirror"):
        assert re.search(rf"\b{parent}\s*=\s*31\b", text[name]), f"{name} lacks ordinal 31"
    assert re.search(rf"\b{status}\s*,\s*\n\s*{parent}\s*,.*?\bSSLM_STATUS_NEXT_FREE\b", positional, re.S), (
        "T2138 positional mirror lacks the parent status right after the producer status")
    assert re.search(rf"\b{parent}\b[^\n]*\b31\b", text["API documentation"]), (
        "API documentation lacks the parent status and ordinal")
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    for gate in ("t2139_gate_c_must_accept_compiles",
                 "t2139_gate_c_real_suite_side_check_compiles",
                 "t2141_gate_c_t2138_suite_side_must_accept_compiles"):
        assert gate in cmake, f"Gate C compile entry {gate} is absent"
    print("prefix status mirrors: both statuses, the delta macro, three mirrors, docs, Gate C entries present")


if __name__ == "__main__":
    main()
