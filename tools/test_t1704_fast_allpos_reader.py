#!/usr/bin/env python3
"""T-1704 -- checks `t1704_fast_allpos_reader.iter_chain_records_fast` against
the canonical `shadow_layer_recompute.load_site_dump_all_positions` on a real
captured all-position site-dump: every chain record's site/codes/d_prime/
position/token_id must match, in order. Run once against a real captured
file before trusting the fast reader for any Part 4 analysis.

Usage: python tools\\test_t1704_fast_allpos_reader.py <path-to-.sitedump>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import shadow_layer_recompute as SLR  # noqa: E402
import t1704_fast_allpos_reader as FAST  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-.sitedump>", file=sys.stderr)
        return 2
    path = sys.argv[1]

    canonical_records, _kv = SLR.load_site_dump_all_positions(path)
    fast_records = list(FAST.iter_chain_records_fast(path))

    if len(canonical_records) != len(fast_records):
        print(f"FAILED: record count mismatch canonical={len(canonical_records)} "
              f"fast={len(fast_records)}", file=sys.stderr)
        return 1

    mismatches = 0
    for i, (c, f) in enumerate(zip(canonical_records, fast_records)):
        ok = (c.site == f.site and c.d_prime == f.d_prime and c.position == f.position and
              c.token_id == f.token_id and np.array_equal(c.codes, f.codes))
        if not ok:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH at record {i}: canonical=({c.site!r},{c.d_prime},{c.position},"
                      f"{c.token_id}) fast=({f.site!r},{f.d_prime},{f.position},{f.token_id}) "
                      f"codes_equal={np.array_equal(c.codes, f.codes)}")

    if mismatches:
        print(f"FAILED: {mismatches}/{len(canonical_records)} chain records mismatched between the "
              f"canonical and fast readers", file=sys.stderr)
        return 1

    print(f"PASS: {len(canonical_records)} chain records identical (site, codes, d_prime, position, "
          f"token_id) between shadow_layer_recompute.load_site_dump_all_positions (canonical, decodes "
          f"x_int too) and t1704_fast_allpos_reader.iter_chain_records_fast (skips x_int) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
