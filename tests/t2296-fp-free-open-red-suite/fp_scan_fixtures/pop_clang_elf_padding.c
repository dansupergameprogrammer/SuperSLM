// T-2342 (Curie) -- the clang/ELF/x86-64 leg's own padding-detector gap
// (design's own residual, Claude/Popper/t2340-fp-scan-instrument-
// commissioning-2026-08-27.md Sec1/Sec3: "refuse=True, unclassified=37/52,
// 0 verdicts" on both the must-accept AND the must-reject -- the leg REFUSES
// on everything, FP-carrying or not, and decides nothing). The cause: this
// design's own `_is_padding_run` (check_fp_free_scan.py:365-373) recognises
// only bytewise-repeated 0x90/0xCC as x86 padding, while clang (and GCC) pad
// function alignment with MULTI-BYTE NOP encodings (e.g. the 13-byte
// `66 66 66 2e 0f 1f 84 00 00 00 00 00 00` this session's own compile of this
// exact file produces) -- bytes that are neither 0x90 nor 0xCC, so the
// padding test fails and the whole object REFUSEs, confirmed by direct
// execution this session (refuse=True, unclassified=19 of 42 bytes, on the
// integer-only half alone).
//
// IntegerOnly (no FP) and GenuineDivide (one real fdiv-shaped double divide)
// are the must-accept/must-reject pair this leg needs to discriminate rather
// than REFUSE on both.

__attribute__((noinline)) int HelperA(int x) { return x + 1; }
__attribute__((noinline)) int HelperB(int x, int y) { return x * y - HelperA(x); }
__attribute__((aligned(16))) __attribute__((noinline)) int IntegerOnly(int x) { return HelperB(x, x); }

__attribute__((noinline)) double HelperC(double x, double y) { return x / y; }
__attribute__((aligned(16))) __attribute__((noinline)) double GenuineDivide(double x, double y) { return HelperC(x, y) + 1.0; }
