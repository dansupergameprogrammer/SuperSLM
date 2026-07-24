"""Pinned Unicode data for the tokenizer's determinism: NFC normalization and the
\\p{L} / \\p{N} property classes, as data tables plus a from-tables implementation.

The runtime must not call a platform Unicode library (D-SLM13), so the converter
generates these tables from a pinned `unicodedata` version and the runtime consumes
them. This module both generates the tables and implements NFC + property lookup
*from the tables* (no `unicodedata` calls in the hot path) — the exact algorithm the
C++ mirrors. `self_test()` proves the table-driven implementation reproduces
`unicodedata` so the emitted tables are known-sufficient before any C++ is written.

NFC is: full canonical decomposition -> canonical ordering (stable, by combining
class) -> canonical composition. Hangul is handled by the algorithmic rules (UAX #15)
so its ~11k syllables need no table entries.
"""

import itertools
import struct
import unicodedata

from sslm_convert_validate import PINNED_UNICODE_VERSION

# Hangul composition constants (Unicode UAX #15).
SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT

UNI_MAGIC = b"UNI1"


def _ranges(pred):
    out, start = [], None
    for cp in range(0x110000):
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1)); start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


class UnicodeVersionMismatch(Exception):
    """Unicode.build()'s pinned-version guard failed. Mirrors
    sslm_convert_validate.check_unicode_version_coherence's own diagnostic --
    that check runs downstream, on the CFG1 fields the converter writes, after
    Unicode.build() has already run to completion; this guard moves the same
    coherence check to the point bytes would first be produced, before any of
    the range/ccc/decomp/compose loops over the 0x110000 codepoint space run
    (S-HARDEN-5, F8)."""


class Unicode:
    def __init__(self, version, letter_ranges, number_ranges, space_ranges, ccc, decomp, compose):
        self.version = version
        self.letter_ranges = letter_ranges     # list[(lo,hi)]
        self.number_ranges = number_ranges
        self.space_ranges = space_ranges       # \s (whitespace)
        self.ccc = ccc                          # dict cp->class (nonzero only)
        self.decomp = decomp                    # dict cp->tuple(full canonical NFD), non-Hangul
        self.compose = compose                  # dict (a,b)->c primary composites, non-Hangul

    # --- generation from the pinned unicodedata ---------------------------------
    @classmethod
    def build(cls, running_version=None):
        """`running_version` is injectable for testing a version mismatch without
        depending on the test runner's own Python build -- the same pattern
        sslm_convert_validate.check_unicode_version_coherence's own
        `running_version` parameter already supports. It affects only the guard
        below; every `unicodedata` call in this method still reads the actual
        running interpreter's real tables regardless of what is injected here.
        """
        raw = running_version if running_version is not None else unicodedata.unidata_version
        parts = [int(p) for p in raw.split(".")]
        while len(parts) < 3:
            parts.append(0)
        actual = tuple(parts[:3])
        if actual != PINNED_UNICODE_VERSION:
            pin = PINNED_UNICODE_VERSION
            raise UnicodeVersionMismatch(
                f"Unicode.build() requires unicodedata {pin[0]}.{pin[1]}.{pin[2]}, but the "
                f"running interpreter's unicodedata.unidata_version is "
                f"{actual[0]}.{actual[1]}.{actual[2]}"
            )

        letter = _ranges(lambda cp: unicodedata.category(chr(cp))[0] == "L")
        number = _ranges(lambda cp: unicodedata.category(chr(cp))[0] == "N")
        space = _ranges(lambda cp: chr(cp).isspace())
        ccc = {}
        for cp in range(0x110000):
            c = unicodedata.combining(chr(cp))
            if c:
                ccc[cp] = c
        decomp, compose = {}, {}
        for cp in range(0x110000):
            if SBASE <= cp < SBASE + SCOUNT:
                continue  # Hangul: algorithmic
            ch = chr(cp)
            nfd = unicodedata.normalize("NFD", ch)
            if nfd != ch:
                decomp[cp] = tuple(ord(x) for x in nfd)
            # primary composite: single-step canonical decomposition of exactly two,
            # round-tripping through NFC (which knows the composition exclusions).
            ds = unicodedata.decomposition(ch)
            if ds and not ds.startswith("<"):
                parts = ds.split()
                if len(parts) == 2:
                    a, b = int(parts[0], 16), int(parts[1], 16)
                    if unicodedata.normalize("NFC", chr(a) + chr(b)) == ch:
                        compose[(a, b)] = cp
        return cls(unicodedata.unidata_version, letter, number, space, ccc, decomp, compose)

    # --- table-driven lookups (no unicodedata) ----------------------------------
    @staticmethod
    def _in_ranges(cp, ranges):
        lo, hi = 0, len(ranges) - 1
        while lo <= hi:
            m = (lo + hi) // 2
            a, b = ranges[m]
            if cp < a:
                hi = m - 1
            elif cp > b:
                lo = m + 1
            else:
                return True
        return False

    def is_letter(self, cp):
        return self._in_ranges(cp, self.letter_ranges)

    def is_number(self, cp):
        return self._in_ranges(cp, self.number_ranges)

    def is_space(self, cp):
        return self._in_ranges(cp, self.space_ranges)

    def _ccc(self, cp):
        return self.ccc.get(cp, 0)

    def _decompose(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if SBASE <= cp < SBASE + SCOUNT:  # Hangul algorithmic decomposition
                s = cp - SBASE
                l = LBASE + s // NCOUNT
                v = VBASE + (s % NCOUNT) // TCOUNT
                t = s % TCOUNT
                out.append(l); out.append(v)
                if t:
                    out.append(TBASE + t)
            elif cp in self.decomp:
                out.extend(self.decomp[cp])
            else:
                out.append(cp)
        return out

    def _canonical_order(self, cps):
        # stable insertion sort of each combining run by combining class
        for i in range(1, len(cps)):
            cc = self._ccc(cps[i])
            if cc == 0:
                continue
            j = i
            while j > 0 and 0 < self._ccc(cps[j - 1]) > cc:
                cps[j - 1], cps[j] = cps[j], cps[j - 1]
                j -= 1

    def _compose(self, cps):
        if not cps:
            return cps
        result = [cps[0]]
        last_starter = 0 if self._ccc(cps[0]) == 0 else None
        prev_cc = self._ccc(cps[0])
        for i in range(1, len(cps)):
            cp = cps[i]
            cc = self._ccc(cp)
            if last_starter is not None and (prev_cc == 0 or prev_cc < cc):
                s = result[last_starter]
                comp = None
                # Hangul algorithmic composition
                if LBASE <= s < LBASE + LCOUNT and VBASE <= cp < VBASE + VCOUNT:
                    comp = SBASE + ((s - LBASE) * VCOUNT + (cp - VBASE)) * TCOUNT
                elif SBASE <= s < SBASE + SCOUNT and (s - SBASE) % TCOUNT == 0 and \
                        TBASE < cp < TBASE + TCOUNT:
                    comp = s + (cp - TBASE)
                else:
                    comp = self.compose.get((s, cp))
                if comp is not None:
                    result[last_starter] = comp
                    continue
            result.append(cp)
            if cc == 0:
                last_starter = len(result) - 1
            prev_cc = cc
        return result

    def nfc(self, text):
        cps = self._decompose(text)
        self._canonical_order(cps)
        return "".join(chr(c) for c in self._compose(cps))

    # --- serialization (the .sslm UNICODE_TABLES blob) --------------------------
    def serialize(self):
        b = bytearray()
        b += UNI_MAGIC
        b += struct.pack("<I", 1)  # blob version
        for ranges in (self.letter_ranges, self.number_ranges, self.space_ranges):
            b += struct.pack("<I", len(ranges))
            for lo, hi in ranges:
                b += struct.pack("<II", lo, hi)
        items = sorted(self.ccc.items())
        b += struct.pack("<I", len(items))
        for cp, c in items:
            b += struct.pack("<II", cp, c)
        # decompositions: cp, offset table into a flat seq
        ditems = sorted(self.decomp.items())
        seq, offs = [], [0]
        for _, s in ditems:
            seq.extend(s); offs.append(len(seq))
        b += struct.pack("<I", len(ditems))
        for cp, _ in ditems:
            b += struct.pack("<I", cp)
        for o in offs:
            b += struct.pack("<I", o)
        b += struct.pack("<I", len(seq))
        for cp in seq:
            b += struct.pack("<I", cp)
        citems = sorted(self.compose.items())
        b += struct.pack("<I", len(citems))
        for (a, bb), c in citems:
            b += struct.pack("<III", a, bb, c)
        return bytes(b)

    def self_test(self, samples=None):
        """Prove the table-driven NFC + property lookups reproduce unicodedata."""
        # property classes: exhaustive over all codepoints
        for cp in range(0x110000):
            cat0 = unicodedata.category(chr(cp))[0]
            assert self.is_letter(cp) == (cat0 == "L"), f"letter {cp:04X}"
            assert self.is_number(cp) == (cat0 == "N"), f"number {cp:04X}"
        # whitespace: exhaustive over all codepoints, the same shape as
        # is_letter/is_number above, extended to the third table that already
        # exists (space_ranges) but was never checked (S-HARDEN-5, F8).
        for cp in range(0x110000):
            assert self.is_space(cp) == chr(cp).isspace(), f"space {cp:04X}"
        # NFC: single codepoints (exhaustive) + adversarial combining sequences
        for cp in range(0x110000):
            ch = chr(cp)
            assert self.nfc(ch) == unicodedata.normalize("NFC", ch), f"nfc U+{cp:04X}"
        # `combining` deliberately carries more than one mark per combining
        # class: 0x300/0x301/0x30A/0x1DC0 all sit at ccc 230, and 0x323/0x316
        # both sit at ccc 220 -- a stable sort and an unstable sort produce
        # identical output on any input where no two elements share a sort
        # key, so a tied pair must be present in the base fixture set itself
        # for any cell to be able to distinguish the two (Mendeleev STRUCTURAL,
        # S-HARDEN-5 design S6).
        combining = [0x300, 0x301, 0x323, 0x328, 0x30A, 0x1DC0, 0x0F71, 0x0F72, 0x0316]
        bases = [ord("a"), ord("e"), ord("o"), 0x1100, 0xAC00, 0xC544, 0x304B, 0x1F1E6]
        # True multi-mark permutation and reordering: every ordering of every
        # subset of `combining`, not only the list's fixed original order --
        # the previous sweep (`combining[:k]` alone) never presented an
        # out-of-order input, so `_canonical_order`'s stable sort was never
        # actually exercised against one (Mendeleev Gap, S-HARDEN-5 design S6).
        # The upper bound is explicit and maximal: k runs through
        # len(combining) inclusive -- the all-marks-present case is its own
        # named line below, not left to be read off this loop's own bound.
        _MAX_MARKS = len(combining)
        for base in bases:
            for k in range(_MAX_MARKS + 1):
                for perm in itertools.permutations(combining, k):
                    s = chr(base) + "".join(chr(c) for c in perm)
                    assert self.nfc(s) == unicodedata.normalize("NFC", s), repr(s)
        # Named case: all `_MAX_MARKS` marks present at once, in strictly
        # descending combining-class order -- the single input shape most
        # likely to force a multi-position stable-sort move, rather than
        # leaving the maximal-reordering case to chance inside the sweep
        # above (Mendeleev Gap, S-HARDEN-5 design S6).
        _descending = sorted(combining, key=lambda c: -unicodedata.combining(chr(c)))
        for base in bases:
            s = chr(base) + "".join(chr(c) for c in _descending)
            assert self.nfc(s) == unicodedata.normalize("NFC", s), repr(s)
        # Named tied-combining-class case: 0x323 (COMBINING DOT BELOW) and
        # 0x0316 (COMBINING GRAVE ACCENT BELOW) both carry ccc 220. Presenting
        # the pair in each of its two relative input orders and checking
        # against unicodedata.normalize (the same oracle every other cell in
        # this method uses) proves `_canonical_order` preserves each order's
        # own relative placement of the tied pair -- the property that
        # distinguishes a stable sort from an ordinary one, which no
        # permutation of a fully-distinct set can ever exercise (Mendeleev
        # STRUCTURAL, S-HARDEN-5 design S6).
        assert unicodedata.combining(chr(0x323)) == unicodedata.combining(chr(0x0316)), (
            "the tied-class fixture's own premise: 0x323 and 0x0316 must share one combining class"
        )
        for base in bases:
            s_first = chr(base) + chr(0x323) + chr(0x0316)
            s_second = chr(base) + chr(0x0316) + chr(0x323)
            assert self.nfc(s_first) == unicodedata.normalize("NFC", s_first), repr(s_first)
            assert self.nfc(s_second) == unicodedata.normalize("NFC", s_second), repr(s_second)
        # extra: dense scan of BMP pairs (base + one combining mark)
        for base in range(0x20, 0x3000, 7):
            for cm in combining:
                s = chr(base) + chr(cm)
                assert self.nfc(s) == unicodedata.normalize("NFC", s), repr(s)
        return True


if __name__ == "__main__":
    u = Unicode.build()
    u.self_test()
    blob = u.serialize()
    print(f"Unicode {u.version}: {len(u.letter_ranges)} letter ranges, "
          f"{len(u.number_ranges)} number ranges, {len(u.space_ranges)} space ranges, "
          f"{len(u.ccc)} ccc, {len(u.decomp)} decomp, {len(u.compose)} compose; "
          f"blob {len(blob)} bytes")
    print("self_test PASSED (table-driven NFC + \\p{L}/\\p{N} == unicodedata, exhaustive)")
