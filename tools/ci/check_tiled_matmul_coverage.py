#!/usr/bin/env python3
"""Tiled-matmul plan slice 1, cell 11.4: named-branch coverage of the tiled kernel.

A per-file branch percentage can stay above its floor while the one path that matters never runs. This
check reads the llvm-cov JSON export and asserts a nonzero count on each NAMED span of the tiled kernel:
the in-loop int64 flush in every micro-kernel instantiation (the MR tile and the one-row tail), the
token (MR) tail, the panel tail (the store guard discarding rows past the range), the packer's K tail
and its K pad -- the AVX2 half always, the AVX-512 half only when the AVX-512-forced binary ran.

How a span is named, so the map cannot drift silently:
  * each span is pinned by its exact source text (`anchor`), which must occur EXACTLY ONCE inside the
    named function's own line span in src/matmul.cpp -- a span that resolves to nothing, or to more
    than one line, is a failure, never a zero-count pass;
  * counts are read per function record, so each template instantiation (TiledMicroAvx2<4> and
    TiledMicroAvx2<1>) is checked on its own -- a flush deleted in the tail kernel only is caught;
  * `branch-false` spans read the false count of the branch whose condition starts at `cond`.

The AVX-512 half reports SKIPPED, never PASS, when the AVX-512-forced binary exited 132 (SIGILL, no
AVX-512 on the runner) or left an empty or missing profile. That fact is read from the binary's own exit
code and its own raw profile, never from the merged export (which could carry AVX-512 counts from
another binary).

Vitality (the plan's X11 rows): the check over a profile from the D-infinity build (tiled path never
taken) fails on every named span; a misspelt anchor fails on that span.

Usage:
  check_tiled_matmul_coverage.py EXPORT.json [--source src/matmul.cpp]
      [--avx512-exit-code N --avx512-profraw PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

FLUSH_AVX2 = "for (int i = 0; i < 8; ++i) acc64[m * 16 + h * 8 + i] += static_cast<int64_t>(lanes[i]);"
FLUSH_AVX512 = "for (int i = 0; i < 16; ++i) acc64[m * 32 + p * 16 + i] += static_cast<int64_t>(lanes[i]);"
STORE_GUARD = "if (n < j_end) out_acc[(t0 + r) * out_channels + n] = acc64[r * kNw + i];  // the store guard"

# (half, name, function as demangled (prefix match on the part before the argument list), kind, anchor, cond)
SPANS = [
    ("common", "packer K tail", "TiledPackPanel16", "region",
     "for (; k < kp; k += 2) {  // the K tail, a partial panel, and the K pad", None),
    ("common", "packer K pad", "TiledPackPanel16", "branch-false",
     "dst[(k / 2) * 32 + i * 2 + e] = (rows[i] != nullptr && kk < in_channels) ? rows[i][kk] : 0;",
     "kk < in_channels"),
    ("avx2", "AVX2 flush, MR tile", "TiledMicroAvx2<4>", "region", FLUSH_AVX2, None),
    ("avx2", "AVX2 flush, one-row tail", "TiledMicroAvx2<1>", "region", FLUSH_AVX2, None),
    ("avx2", "AVX2 token (MR) tail", "TiledGemmAvx2", "region",
     "for (size_t r = 0; r < mr; ++r) TiledMicroAvx2<1>(a16 + (t0 + r) * kp, kp, scratch, kp, acc64 + r * kNw);",
     None),
    ("avx2", "AVX2 panel tail", "TiledGemmAvx2", "branch-false", STORE_GUARD, "n < j_end"),
    ("avx512", "AVX-512 flush, MR tile", "TiledMicroAvx512<8>", "region", FLUSH_AVX512, None),
    ("avx512", "AVX-512 flush, one-row tail", "TiledMicroAvx512<1>", "region", FLUSH_AVX512, None),
    ("avx512", "AVX-512 token (MR) tail", "TiledGemmAvx512", "region",
     "for (size_t r = 0; r < mr; ++r) TiledMicroAvx512<1>(a16 + (t0 + r) * kp, kp, scratch, kp, acc64 + r * kNw);",
     None),
    ("avx512", "AVX-512 panel tail", "TiledGemmAvx512", "branch-false", STORE_GUARD, "n < j_end"),
]


def demangle(names: list[str]) -> list[str]:
    tool = shutil.which("llvm-cxxfilt-18") or shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    raw = [n.split(":", 1)[1] if ":" in n and not n.startswith("_Z") else n for n in names]
    if not tool:
        return raw
    out = subprocess.run([tool], input="\n".join(raw), capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def short_name(demangled: str) -> str:
    """'void superslm::(anonymous namespace)::TiledMicroAvx2<4>(short const*, ...)' -> 'TiledMicroAvx2<4>'."""
    s = demangled.replace("(anonymous namespace)::", "")
    depth, cut, last_sep, last_space = 0, len(s), -1, -1
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0 and ch == "(":
            cut = i
            break
        elif depth == 0 and ch == " ":
            last_space = i
        elif depth == 0 and s.startswith("::", i):
            last_sep = i + 1
    start = max(last_sep, last_space) + 1
    return s[start:cut]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--source", default="src/matmul.cpp")
    ap.add_argument("--avx512-exit-code", type=int, default=None)
    ap.add_argument("--avx512-profraw", default=None)
    args = ap.parse_args()

    lines = open(args.source, encoding="utf-8").read().splitlines()
    data = json.load(open(args.export, encoding="utf-8"))["data"][0]
    funcs = [f for f in data["functions"] if any(fn.endswith(os.path.basename(args.source)) for fn in f["filenames"])]
    names = demangle([f["name"] for f in funcs])

    avx512_ran = (args.avx512_exit_code == 0 and args.avx512_profraw is not None
                  and os.path.exists(args.avx512_profraw) and os.path.getsize(args.avx512_profraw) > 0)
    if args.avx512_exit_code is None:
        avx512_why = "no --avx512-exit-code given"
    elif args.avx512_exit_code == 132:
        avx512_why = "the AVX-512-forced binary exited 132 (SIGILL: no AVX-512 on this runner)"
    elif args.avx512_exit_code != 0:
        avx512_why = f"the AVX-512-forced binary exited {args.avx512_exit_code}"
    else:
        avx512_why = "the AVX-512-forced binary left no profile"

    failures = []
    report = {"common": [], "avx2": [], "avx512": []}
    for half, label, fname, kind, anchor, cond in SPANS:
        if half == "avx512" and not avx512_ran:
            report[half].append(f"SKIPPED  {label}")
            continue
        recs = [(f, n) for f, n in zip(funcs, names) if short_name(n) == fname]
        if not recs:
            failures.append(f"{label}: no function record named {fname} (renamed, or never compiled)")
            report[half].append(f"FAIL     {label}: no record for {fname}")
            continue
        best = 0
        resolved = False
        for f, _ in recs:
            code = [r for r in f["regions"] if r[5] == 0]
            lo = min(r[0] for r in code)
            hi = max(r[2] for r in code)
            hits = [i + 1 for i in range(lo - 1, min(hi, len(lines))) if lines[i].strip() == anchor]
            if len(hits) != 1:
                continue
            line = hits[0]
            col = lines[line - 1].index(anchor) + 1
            if kind == "region":
                cands = [r for r in code if (r[0], r[1]) <= (line, col) <= (r[2], r[3]) and r[7] == 0]
                if not cands:
                    continue
                inner = min(cands, key=lambda r: ((r[2] - r[0]) * 10000 + (r[3] - r[1])))
                resolved = True
                best = max(best, inner[4])
            else:
                ccol = lines[line - 1].index(cond) + 1
                br = [b for b in f.get("branches", []) if b[0] == line and b[1] == ccol and b[6] == 0]
                if not br:
                    continue
                resolved = True
                best = max(best, max(b[5] for b in br))
        if not resolved:
            failures.append(f"{label}: the anchor resolves to no single span in {fname}")
            report[half].append(f"FAIL     {label}: unresolved in {fname}")
        elif best == 0:
            failures.append(f"{label}: count 0 in {fname}")
            report[half].append(f"FAIL     {label}: count 0")
        else:
            report[half].append(f"ok       {label}: {best}")

    for half in ("common", "avx2", "avx512"):
        print(f"[{half}]" + ("" if half != "avx512" or avx512_ran else f" SKIPPED: {avx512_why}"))
        for r in report[half]:
            print("  " + r)
    print("halves run: common, AVX2" + (", AVX-512" if avx512_ran else " (AVX-512 SKIPPED)"))
    if failures:
        print("check_tiled_matmul_coverage: FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print("check_tiled_matmul_coverage: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
