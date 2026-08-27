// T-2326 (Curie) -- Sec7 dimension 11's second commissioning population: a
// must-reject construction for §4.1's own text, "machine code that performs a
// genuine floating-point operation invisible to §4.2/§4.3's MXCSR-observing
// cells, whether because the operation class itself never raises an exception
// or because the corpus happens to be exact for that one run."
//
// PROVENANCE. The design's own text names this population as "TE-32's Part D2
// legs, TE-32's Part D exact-operand-set legs on the four always-invisible-to-
// MXCSR classes, and the compiled BodyUnderTest() artifact from Part E/E2" --
// TE-32's own historical artifact lives at Claude/Loki/te32-probe/, OUTSIDE
// this session's granted read-only scope. NOT a reproduction of that file;
// this is a fresh construction satisfying the SAME documented property
// (SSE-class comparison/min instructions that do not raise any IEEE-754
// exception under a normal, non-NaN, non-zero-denominator operand set, so a
// trap-observing liveness control -- §4.2/§4.3's own masks-cleared/sticky-flag
// cells -- would see nothing even though the instrument's own register-file
// check (A) must still REJECT: any instruction naming an xmm/ymm/zmm/st
// register is rejected unless on the closed VEC_MOVE_ALLOW list, and NONE of
// comisd/ucomisd/minsd/maxsd is on it).
//
// Executed check, this fixture's own defining claim: compiled and RUN under
// default (masked) MXCSR state, comparing and taking the min of two ordinary,
// non-exceptional doubles raises no trap and returns normally -- confirmed by
// the test that drives this fixture, independent of and prior to grading it
// through the (absent) instrument, per StandardsDocument.md §5.4 ("exactness
// is verified at source or by execution").

extern "C" __declspec(dllexport) int BodyCompare(double a, double b) {
    // comisd/ucomisd -- an SSE2 floating-point comparison. Never raises an
    // IEEE exception for a normal, non-NaN operand pair (only #I -- invalid --
    // on a NaN operand, which this fixture's own must-reject claim does not
    // need to exercise): a trap-observing liveness control sees nothing.
    return a < b ? 1 : 0;
}

extern "C" __declspec(dllexport) double BodyMin(double a, double b) {
    // minsd -- an SSE2 floating-point minimum. Also exception-free for a
    // normal operand pair.
    return a < b ? a : b;
}
