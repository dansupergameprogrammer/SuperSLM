// T-2342 (Curie) -- Poirot's own Critical finding, C1
// (Claude/Poirot/78535ed-t2339-fp-scan-instrument-review.md): `movsd` --
// MSVC's ordinary eight-byte copy of a double-sized value, PURE DATA
// MOVEMENT, no arithmetic -- is missing from `_X86_VEC_MOVE_ALLOW`
// (check_fp_free_scan.py:480-505), which carries `movss`/`movaps`/`movdqa`/
// `movd`/`movq` and the AVX form `vmovsd`, but not the plain SSE2 form. 62
// real symbols across 6 real translation units currently REJECT solely on
// this omission (Poirot's own table), including two members of Sec3.1's own
// must-accept commissioning population (FixedIntMap's own
// `_Emplace_back`/`_Uninitialized_fill_n`). Confirmed by direct execution
// this session: this exact construction, compiled fresh, decodes to
// `movsd xmm0, mmword ptr [rcx]` / `movsd mmword ptr [rcx+8], xmm0` -- no
// arithmetic anywhere -- and scan_object (the built instrument, unmodified)
// REJECTs it today.

extern "C" double LoadStoreDouble(double* p) {
    double x = *p;
    *(p + 1) = x;
    return x;
}
