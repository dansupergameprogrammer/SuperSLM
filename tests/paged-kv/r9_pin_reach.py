"""Paged-KV plan rev 16.21 cell 4.12 [T1], "absent from production": the test-only pin
KvPagesTestPinPagePositions is in superslm_test_injection and nowhere in the production superslm.

Three checks, in order, each a verdict line; exit 0 only when all three pass.
  1. Positive control: the symbol's name is in the test library's archive (a scan that cannot find it
     there proves nothing about its absence elsewhere).
  2. The production library's archive does not carry the name.
  3. The link probe (tests/paged-kv/pkv_r9_pin_link_probe.cpp, target superslm_pkv_r9_pin_link_probe),
     linked against superslm alone, fails, and its log names the pin as unresolved.

The scan reads the archive bytes for the symbol's name: a static library's member objects and its
symbol table carry every external name they define or reference, so a library that defines the pin
carries the string, and one that does not carry it neither defines nor references it.

Usage: python -I tests/paged-kv/r9_pin_reach.py BUILD_DIR [CONFIG]
(run from a shell where cmake builds the tree, e.g. after vcvars64; CONFIG defaults to Release)
"""

import os
import subprocess
import sys

SYMBOLS = (b"KvPagesTestPinPagePositions", b"KvPagesTestPoolGeometry")  # the pin, and the geometry seam (rev 16.23)


def find_lib(build, config, name):
    for cand in (os.path.join(build, config, name + ".lib"), os.path.join(build, "lib" + name + ".a"),
                 os.path.join(build, name + ".lib")):
        if os.path.isfile(cand):
            return cand
    return None


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    build = argv[1]
    config = argv[2] if len(argv) > 2 else "Release"
    ok = True
    test_lib = find_lib(build, config, "superslm_test_injection")
    prod_lib = find_lib(build, config, "superslm")
    if not test_lib or not prod_lib:
        print(f"INFRASTRUCTURE: libraries not found under {build} ({test_lib}, {prod_lib})")
        return 3
    with open(test_lib, "rb") as f:
        test_bytes = f.read()
    with open(prod_lib, "rb") as f:
        prod_bytes = f.read()
    for sym in SYMBOLS:
        in_test = sym in test_bytes
        print(f"{'PASS' if in_test else 'FAIL'} positive control: {sym.decode()} in {test_lib}: {in_test}")
        in_prod = sym in prod_bytes
        print(f"{'PASS' if not in_prod else 'FAIL'} production: {sym.decode()} in {prod_lib}: {in_prod}")
        ok = ok and in_test and not in_prod
    run = subprocess.run(["cmake", "--build", build, "--config", config, "--target", "superslm_pkv_r9_pin_link_probe"],
                         capture_output=True, text=True)
    log = run.stdout + run.stderr
    unresolved = run.returncode != 0 and all(s.decode() in log for s in SYMBOLS) and ("unresolved" in log or "undefined" in log)
    print(f"{'PASS' if unresolved else 'FAIL'} link probe against superslm: exit {run.returncode}, "
          f"unresolved pin named: {unresolved}")
    ok = ok and unresolved
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
