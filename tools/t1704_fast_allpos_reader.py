#!/usr/bin/env python3
"""T-1704 -- a fast, x_int-skipping streaming reader over the all-position
site-dump format (`sslm_layer_trace.cpp --site-dump --site-dump-all-
positions`, `WriteSiteDumpAllPositions`).

This is NOT a replacement for `shadow_layer_recompute.load_site_dump_all_
positions` (the canonical reader, decodes every field into a real
`SiteRecordPositional`/`KvLandingRecordPositional`) -- it is a bulk-analysis
optimization used only by this task's own Part 4 scripts, which need only
`site`, `codes`, `d_prime` (the per-channel magnitude proxy), `position`,
and `token_id` from the chain-record section, never `x_int`
(potentially 8960 int64 values per record) or the K/V-landing section at
all. Skipping `x_int`'s bytes (rather than decoding them into a numpy
array) is the whole speed difference -- on a ~470MB all-position dump this
takes low single-digit seconds instead of tens of seconds.

Correctness is not assumed: `tools/test_t1704_fast_allpos_reader.py` checks
this reader's own output against `shadow_layer_recompute.load_site_dump_
all_positions` (decodes everything) on a real captured file and asserts
every field this reader emits matches, record-for-record.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_ALL_POSITIONS_MAGIC = b"SSLMAPD2"


class FastChainRecord:
    __slots__ = ("site", "codes", "d_prime", "position", "token_id")

    def __init__(self, site, codes, d_prime, position, token_id):
        self.site = site
        self.codes = codes
        self.d_prime = d_prime
        self.position = position
        self.token_id = token_id


def iter_chain_records_fast(path):
    """Yields `FastChainRecord` for every CHAIN record in an all-position
    site-dump, skipping x_int's bytes and never touching the K/V-landing
    section (this task's own analysis never reads it). Same magic check and
    the same "fail loud, not silent" contract as the canonical reader."""
    data = Path(path).read_bytes()
    if data[:8] != _ALL_POSITIONS_MAGIC:
        raise ValueError(f"iter_chain_records_fast: {path} does not start with the all-positions "
                          f"magic {_ALL_POSITIONS_MAGIC!r}")
    off = 8
    (n,) = struct.unpack_from("<Q", data, off)
    off += 8
    for _ in range(n):
        (name_len,) = struct.unpack_from("<Q", data, off)
        off += 8
        site = data[off:off + name_len].decode()
        off += name_len
        (n_x,) = struct.unpack_from("<Q", data, off)
        off += 8
        off += 8 * n_x  # SKIP x_int -- never decoded, this reader's whole speed advantage.
        (n_codes,) = struct.unpack_from("<Q", data, off)
        off += 8
        codes = np.frombuffer(data, dtype=np.int8, count=n_codes, offset=off).copy()
        off += n_codes
        d_prime, dn, s, r, m_out, e_out = struct.unpack_from("<qqiqqq", data, off)
        off += 8 + 8 + 4 + 8 + 8 + 8
        (position,) = struct.unpack_from("<Q", data, off)
        off += 8
        (token_id,) = struct.unpack_from("<i", data, off)
        off += 4
        yield FastChainRecord(site=site, codes=codes, d_prime=d_prime, position=position,
                               token_id=token_id)
    # Deliberately does not parse the K/V-landing section or verify
    # off == len(data) at the very end (the canonical reader does both) --
    # this reader stops once every chain record is yielded, since its own
    # analysis never needs the K/V section. Correctness of THIS reader
    # against the canonical one is what test_t1704_fast_allpos_reader.py
    # checks, not a self-contained total-consumption guard.
