"""T-2909 (Curie) -- generates the CPU-side reference/mutant `sslm_abi.cpp` variants
`run_mutants_cpu_deadend.bat` builds, straight from the LIVE TIP (`src/sslm_abi.cpp`), rather
than depending on external scratch reference files staged by an earlier ticket
(`Claude/Vitruvius/t2898-probe/`'s own `sslm_abi_cpu_fixed_v5.cpp`/mutant files) that no longer
exist on disk. Every fold those files were cumulative through (T-2866/T-2894/T-2896/T-2897/
T-2898) is already landed in the committed tree at this tip, so the live source IS the fixed
reference; the two mutants below are single-line reverts of it, at BOTH of the greedy branch's
own miss sites (the ready-branch's `sslm_decode_stepImpl`, `src/sslm_abi.cpp`'s greedy path --
the damped-greedy branch carries its own independent copy of the same two lines, reverted
identically at both sites so the mutant is total across both decode modes).

Each site is located structurally, not by line number: the three-line sequence
`out_tokens[i] = -2;` / `seq->state.layer_index = 0;` / `seq->ready_for_logits = true;`
(whatever its indentation), which is unique to the two miss sites in this file -- an ordinary
`ready_for_logits = true;` also appears elsewhere (prefill's own ready-arm), but never
immediately after this exact two-line preface. Refuses (nonzero exit, no output written)
unless EXACTLY TWO sites are found, so a source that has moved is caught rather than silently
mutating the wrong line or missing a site.

Modes:
  FIXED       : no change -- the live tip verbatim (the committed fixes already landed).
  NOREARM     : both sites' `seq->ready_for_logits = true;` line commented out. Must turn
                Cell 1(iii)'s own resumability assertion red on every route.
  NORESET     : both sites' `seq->state.layer_index = 0;` line commented out. Must turn Cell
                1(iv)'s own resettability assertion red on exactly routes D and D1 while (iii)
                stays green (the two assertions are independent, TE-361's own finding).

Usage: python make_mut_cpu_deadend.py <path-to-sslm_abi.cpp> <mode: FIXED|NOREARM|NORESET> <out.cpp>
"""
import re
import sys

# Matches either miss site, any indentation, capturing the indent and the three lines as a
# whole so exactly one of the middle two can be commented out per mode.
_SITE_RE = re.compile(
    r"(?P<indent>[ \t]+)out_tokens\[i\] = -2;\r?\n"
    r"(?P<indent2>[ \t]+)seq->state\.layer_index = 0;\r?\n"
    r"(?P<indent3>[ \t]+)seq->ready_for_logits = true;\r?\n"
)


def main(argv):
    if len(argv) != 4:
        sys.exit("usage: make_mut_cpu_deadend.py <path-to-sslm_abi.cpp> <FIXED|NOREARM|NORESET> <out.cpp>")
    src, mode, dst = argv[1], argv[2], argv[3]
    if mode not in ("FIXED", "NOREARM", "NORESET"):
        sys.exit("mode must be one of FIXED, NOREARM, NORESET")
    with open(src, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    hits = list(_SITE_RE.finditer(text))
    if len(hits) != 2:
        sys.exit("expected exactly 2 miss sites (greedy + damped-greedy), found %d -- source "
                  "may have moved, refusing to guess" % len(hits))

    if mode == "FIXED":
        out = text
    else:
        def repl(m):
            layer_line = "%sseq->state.layer_index = 0;\r\n" % m.group("indent2")
            rearm_line = "%sseq->ready_for_logits = true;\r\n" % m.group("indent3")
            if mode == "NOREARM":
                rearm_line = "%s// T-2909 MUT_NOREARM: seq->ready_for_logits = true; deleted.\r\n" % m.group("indent3")
            elif mode == "NORESET":
                layer_line = "%s// T-2909 MUT_NORESET: seq->state.layer_index = 0; deleted.\r\n" % m.group("indent2")
            return "%sout_tokens[i] = -2;\r\n%s%s" % (m.group("indent"), layer_line, rearm_line)

        out = _SITE_RE.sub(repl, text)

    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print("wrote %s (mode=%s, %d site(s) matched)" % (dst, mode, len(hits)))


if __name__ == "__main__":
    main(sys.argv)
