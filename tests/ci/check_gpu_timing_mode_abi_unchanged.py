"""check_gpu_timing_mode_abi_unchanged.py -- GPU per-token overhead plan rev 3.1, §7 cell 7.9.

"No ABI change": the timing-mode work (S1, and S2/S3 after it) adds only to the internal C++
surface in include/superslm/gpu_port.h. The public 1.0 GPU header and the C ABI verb list the
plugin compiles against must not move. This check pins both by SHA-256 of their bytes as they
stand at v1.11.0 -- the plan names v1.9.0, and the common brief re-anchors the slice on v1.11.0,
whose gpu_1p0.h differs from v1.9.0's by one comment paragraph (the host parallel-for hook's
`reserved` bits), so v1.11.0 is the baseline that can be byte-identical.

Line ends are normalized to LF before hashing: build.bat runs this on a Windows checkout, and a
CRLF checkout of the same bytes is the same header.

The Gate A header-parity check (tools/t2139_gate_a_header_parity_check.cpp) is the other half of
cell 7.9 and already runs in build.bat; this script does not restate it.

Usage: python tests/ci/check_gpu_timing_mode_abi_unchanged.py [--root DIR]
Exit 0 when both pins hold, 1 otherwise.
"""
import argparse
import hashlib
import pathlib
import sys

PINS = {
    # v1.11.0 (tag 8d85b7b): include/superslm/gpu_1p0.h
    "include/superslm/gpu_1p0.h": "f3401a496de189bef2e64164b17a35fe359340c824e456ae9a11f892fa39530a",
    # v1.11.0 (tag 8d85b7b): tests/t2807-api-slot/c_abi_verbs.txt
    "tests/t2807-api-slot/c_abi_verbs.txt": "095de52c4b960cf6b594de6dc756dd03660fc6f0420aefb29cd25c18b7352458",
}


def lf_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    failures = []
    for rel, want in PINS.items():
        path = root / rel
        if not path.is_file():
            failures.append(f"{rel}: missing")
            continue
        got = lf_sha256(path)
        if got != want:
            failures.append(f"{rel}: sha256 {got}, pinned {want} (v1.11.0)")
    if failures:
        print("check_gpu_timing_mode_abi_unchanged.py: FAILED -- the GPU per-token overhead plan "
              "(rev 3.1, I5 / cell 7.9) changes no public ABI surface:")
        for f in failures:
            print("  " + f)
        return 1
    print("check_gpu_timing_mode_abi_unchanged.py: OK -- gpu_1p0.h and c_abi_verbs.txt are "
          "byte-identical to v1.11.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
