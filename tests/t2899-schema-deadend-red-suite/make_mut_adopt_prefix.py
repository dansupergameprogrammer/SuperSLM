"""T-2916 (Curie), closing TE-370 M1: generates the `sslm_abi.cpp` reference/mutant variants
`run_mutants_adopt_prefix.bat` builds, FROM THE LIVE TIP (`src/sslm_abi.cpp`), replacing the
out-of-repo `D:\\_t2909\\refs_adopt\\sslm_abi_fixed.cpp`/`sslm_abi_mutant_forced.cpp`/
`sslm_abi_mutant_walkreset.cpp` (static copies of an earlier fold, differing from the live tip
today only by prose/comment rewording plus the live tip's own later additions -- confirmed by
`diff --strip-trailing-cr`, 139 changed lines, none of them semantic).

Modes:
  FIXED         : no change -- the live tip verbatim (T-2896/T-2897/T-2898's own folds are
                  already landed in the committed tree).
  MUT_WALKRESET : `adopt_prefix`'s own `seq->dfa_walk_state = seq->bound_schema ? 0u :
                  kDfaWalkStateUnused;` reverted (commented out) -- reproduces the pre-T-2898
                  stale-walk defect TE-363's census found (2 of 16 resting states). Anchored on
                  the unique comment immediately above it ("...as at 1.5.0.") so the OTHER,
                  textually identical reset line in `sslm_seq_reset` (`:1920`) is never touched.
  MUT_FORCED    : `adopt_prefix`'s own `seq->forced_token_count = 0;` reverted (commented out) --
                  the TE-364 mutant this cell exists to kill. Anchored on the unique closing line
                  of the comment immediately above it (T-2917 folding TE-370 M2 rewrote the body
                  of that comment; the anchor tracks only its last line plus the code line, so a
                  future prose edit above it does not also require re-anchoring this generator).

The source file is CRLF throughout (`sslm_abi.cpp`, confirmed by execution); every anchor and
replacement below uses `\\r\\n` to match it exactly, opened with `newline=""` so nothing is
silently translated. Refuses (nonzero exit, no output written) unless each anchor occurs exactly
once.

Usage: python make_mut_adopt_prefix.py <FIXED|MUT_WALKRESET|MUT_FORCED> <path-to-sslm_abi.cpp> <out.cpp>
"""
import sys

_WALKRESET_ANCHOR = (
    "\t\t// as at 1.5.0.\r\n"
    "\t\tseq->dfa_walk_state = seq->bound_schema ? 0u : kDfaWalkStateUnused;\r\n"
)
_WALKRESET_REPLACEMENT = (
    "\t\t// as at 1.5.0.\r\n"
    "\t\t// T-2916 MUT_WALKRESET (T-2898 must-reject): the stale-walk reset reverted --\r\n"
    "\t\t// reproduces the pre-fix defect TE-363's census found.\r\n"
)

_FORCED_ANCHOR = (
    "\t// of the prefix's history.\r\n"
    "\tseq->forced_token_count = 0;\r\n"
)
_FORCED_REPLACEMENT = (
    "\t// of the prefix's history.\r\n"
    "\t// T-2916 MUT_FORCED (TE-364): seq->forced_token_count = 0; deleted.\r\n"
)

_MODES = {
    "FIXED": None,
    "MUT_WALKRESET": (_WALKRESET_ANCHOR, _WALKRESET_REPLACEMENT),
    "MUT_FORCED": (_FORCED_ANCHOR, _FORCED_REPLACEMENT),
}


def main(argv: list[str]) -> None:
    if len(argv) != 4 or argv[1] not in _MODES:
        sys.exit("usage: make_mut_adopt_prefix.py <FIXED|MUT_WALKRESET|MUT_FORCED> "
                  "<path-to-sslm_abi.cpp> <out.cpp>")
    mode, src, dst = argv[1], argv[2], argv[3]
    with open(src, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    if _MODES[mode] is None:
        out = text
    else:
        anchor, replacement = _MODES[mode]
        count = text.count(anchor)
        if count != 1:
            sys.exit(f"expected exactly 1 occurrence of the {mode} anchor, found {count} -- "
                      "source may have moved, refusing to guess")
        out = text.replace(anchor, replacement, 1)

    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"wrote {dst} (mode={mode})")


if __name__ == "__main__":
    main(sys.argv)
