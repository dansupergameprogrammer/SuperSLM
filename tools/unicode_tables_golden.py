"""Emit tools/unicode_tables_golden.txt -- the SHA-256 digest of the exact UNI1 blob
bytes convert_tokenizer.py embeds into every converted model (S-HARDEN-5, F8).

unicode_tables.py has no existing golden today: Unicode.build() is invoked LIVE, at
conversion time, and its serialized bytes are baked directly into each model's
UnicodeTables section, never written to a standalone tracked file. This script is
that missing instrument: it builds the tables from the pinned unicodedata version
(Unicode.build() itself asserts the version before any table-building loop runs --
tools/unicode_tables.py's UnicodeVersionMismatch guard), computes the SHA-256 of
Unicode.build().serialize(), and writes a tracked, human-reviewable file recording
the pinned version, the digest, and the per-table entry counts.

A digest is the right instrument here, not a full inline dump of the tables the way
gen_matmul_golden.py inlines C arrays: the tables are consumed by Python at
conversion time, not by test_main.cpp, so there is no C++ caller for a
#include-able header to serve, and a single strong hash over the full
deterministic blob has the same discriminating power as a full diff for a
"did this regenerate identically" gate.

Run from tools/:  python unicode_tables_golden.py
"""

import hashlib
import pathlib

from unicode_tables import Unicode

HERE = pathlib.Path(__file__).parent
OUT_PATH = HERE / "unicode_tables_golden.txt"


def format_report(u: Unicode) -> str:
    """Format `u`'s report from its CURRENT in-memory state -- reserializing and
    rehashing every time, never caching a prior digest. Split out from
    `build_report()` so the digest's own discriminating power (does the report
    actually change when the object's tables change) can be tested directly on
    a mutated object, independent of `Unicode.build()`/`self_test()` (S-HARDEN-5
    design S8 step 3a)."""
    blob = u.serialize()
    digest = hashlib.sha256(blob).hexdigest()
    lines = [
        f"unicode_version={u.version}",
        f"blob_sha256={digest}",
        f"blob_bytes={len(blob)}",
        f"letter_ranges={len(u.letter_ranges)}",
        f"number_ranges={len(u.number_ranges)}",
        f"space_ranges={len(u.space_ranges)}",
        f"ccc_entries={len(u.ccc)}",
        f"decomp_entries={len(u.decomp)}",
        f"compose_entries={len(u.compose)}",
    ]
    return "\n".join(lines) + "\n"


def build_report() -> str:
    u = Unicode.build()
    u.self_test()
    return format_report(u)


def main() -> None:
    report = build_report()
    OUT_PATH.write_text(report, encoding="ascii", newline="\n")
    print(f"wrote {OUT_PATH}")
    print(report)


if __name__ == "__main__":
    main()
