// t2139_gate_c_real_suite_side_check.cpp -- Gate C's THIRD TU (S1, Claude/Poirot/
// 3bcbe43-t2139-fourth-confirmation-review.md, owed remedy 5): the structural half of S1's fix.
// tools/t2139_gate_c_type_identity_check.cpp's own "suite side" is a hand transcription -- its own
// top comment discloses that and gives the [dcl.link] reason -- so it can never itself detect a
// divergence a human forgets to re-transcribe (P5/P6, S1's own commissioned probes: appending an
// enumerator to the REAL sslm_g5.h, or moving an existing shared enumerator's ordinal in the REAL
// sslm_g5.h, both compiled clean through the existing two-gate build because nothing in it ever
// re-reads the real header's own enum body). THIS file closes that gap: it includes
// tests/t2130-g5-red-suite/sslm_g5.h DIRECTLY -- not a transcription -- alongside the library's
// own SSLM_STATUS_ENUM_LIST expansion, and asserts per-name equality and sentinel identity against
// the REAL header, not a copy of it.
//
// WHY THIS NEEDS ITS OWN TU (the collision tools/t2139_gate_c_type_identity_check.cpp's own header
// comment already names, and why namespacing the suite side there was not enough on its own to let
// this be added to THAT file instead): two `extern "C"` function declarations with the SAME name
// but DIFFERENT C++ parameter types are ill-formed ([dcl.link]) REGARDLESS of which C++ namespace
// encloses each one -- C linkage collides by symbol name alone, ignoring the enclosing namespace.
// tests/t2130-g5-red-suite/sslm_g5.h and include/superslm/sslm_abi.h both declare `sslm_model_map`
// (and 17 other shared verb names) with parameter types that are structurally distinct once BOTH
// real headers are included in one TU (each header's own opaque handle typedefs resolve to a
// DIFFERENT, separately-forward-declared struct tag, even where the two tag names are spelled
// identically) -- so simply namespacing the suite side's own #include of the real header does NOT
// avoid the collision, because namespacing only separates the TYPES, never the extern "C" function
// NAMES. Verified at source before writing this file: tests/t2130-g5-red-suite/sslm_g5.h itself
// declares 18 extern "C" functions (grep-confirmed, `sslm_model_map`/`sslm_prefix_begin`/
// `sslm_seq_create`/... at its own real names) -- the premise that the SUITE header carries no
// function declarations at all is FALSE; what makes this TU legal is not the suite header's
// absence of functions, but this TU's own suppression of the LIBRARY side's functions instead (see
// SUPERSLM_ABI_ENUM_ONLY below), which removes the only thing they would otherwise collide with.
//
// THE FIX: include/superslm/sslm_abi.h gained a SUPERSLM_ABI_ENUM_ONLY opt-out (that header's own
// comment, immediately above its `#include "superslm/sslm_abi_functions.inc"` line) -- defined
// here, before including the library header, it suppresses ONLY the function-declaration include;
// every type/enum/struct above it, INCLUDING sslm_status and the SSLM_STATUS_ENUM_LIST macro
// itself (the single generation source both the library's own enum body and Gate C's per-name
// checks re-expand -- F1's own structural remedy, unchanged here), still declares exactly as
// normal. With the library side's own functions suppressed, the REAL tests/t2130-g5-red-suite/
// sslm_g5.h can then be included in FULL -- types AND its own 18 real functions -- with nothing at
// global scope for its functions to collide with.
//
// MUST-ACCEPT construction (design Sec9 shape, matching Gate C's sibling file): never linked,
// compile-only.
#include <cstddef>
#include <type_traits>

#define SUPERSLM_ABI_ENUM_ONLY
#include "superslm/sslm_abi.h"
#undef SUPERSLM_ABI_ENUM_ONLY

namespace t2139_gate_c_real_suite_side {
// The REAL header, included whole -- not a transcription. Its own 18 function declarations land
// here, namespaced; nothing at global scope declares those same names in this TU (the library
// side's own functions are suppressed above), so there is nothing for them to collide with.
#include "t2130-g5-red-suite/sslm_g5.h"
}  // namespace t2139_gate_c_real_suite_side

// --- sslm_status: per-shared-enumerator-name value equality, REAL suite header vs REAL library
// header -- no transcription on either side. GENERATED, not hand-written (F1's own remedy,
// unchanged): re-expands include/superslm/sslm_abi.h's own SSLM_STATUS_ENUM_LIST X-macro, the
// SAME list tools/t2139_gate_c_type_identity_check.cpp's own per-name checks re-expand. ---
#define T2139_GATE_C_REAL_STATUS_CHECK(NAME)                                                     \
	static_assert(                                                                                 \
	    static_cast<int>(t2139_gate_c_real_suite_side::NAME) == static_cast<int>(::NAME), \
	    #NAME " diverges (REAL tests/t2130-g5-red-suite/sslm_g5.h vs REAL include/superslm/"       \
	          "sslm_abi.h)")
#define T2139_GATE_C_REAL_STATUS_CHECK_GEN_(NAME) T2139_GATE_C_REAL_STATUS_CHECK(NAME);
SSLM_STATUS_ENUM_LIST(T2139_GATE_C_REAL_STATUS_CHECK_GEN_)
#undef T2139_GATE_C_REAL_STATUS_CHECK_GEN_
#undef T2139_GATE_C_REAL_STATUS_CHECK

// --- registry-top sentinel identity, REAL suite header vs REAL library header (S1, F1's Probe C
// class): catches a shared ordinal claimed by two different names, which the per-name checks above
// structurally cannot see -- and, unlike tools/t2139_gate_c_type_identity_check.cpp's own sentinel
// assertion (which compares the library's real header against a HAND-TRANSCRIBED copy of the
// suite's sentinel), this one compares against the REAL suite header's own sentinel, so a one-sided
// append to EITHER real header -- P5/P6, the exact commissioned mutations that compiled clean
// through the existing two-gate build -- now fails HERE. ---
static_assert(
    static_cast<int>(t2139_gate_c_real_suite_side::SSLM_STATUS_NEXT_FREE) ==
        static_cast<int>(::SSLM_STATUS_NEXT_FREE),
    "registry-top divergence: the REAL tests/t2130-g5-red-suite/sslm_g5.h does not top out at the "
    "same SSLM_STATUS_NEXT_FREE sentinel the REAL include/superslm/sslm_abi.h declares -- one side "
    "has appended an enumerator, or moved an existing enumerator's ordinal, the other has not taken "
    "(S1, Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md P5/P6); reconcile both to design "
    "Sec6, the single-authority complete ordinal registry (T-2133 ruling, design commit "
    "4f4eb23896)");

int main() { return 0; }
