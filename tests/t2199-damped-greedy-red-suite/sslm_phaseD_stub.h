// T-2199 (Curie) -- Phase D stub, SUPERSEDED 2026-08-20 by the real build
// (brunel/t2199-phaseD@5ced7a6, merged forward onto this branch). Every symbol this header
// used to declare-but-not-define now has a real production definition in
// include/superslm/sslm_phaseD.h (same namespace, superslm_test_phaseD -- a naming quirk
// inherited from this suite's own original stub, kept because C++ linkage requires the
// mangled name to match; a records-only rename is open for later without any functional
// change). Rather than carry two declarations of the same surface (this file's stub copy and
// the real header), this file now simply re-exports the production header -- the same
// supersession shape include/superslm/sslm_damped_greedy.h already established for this
// suite's own Phase A/C stub.
//
// SYNC NOTE (conductor's dispute-resolution commission, 2026-08-20): the production build
// renamed the artifact flag constant from this stub's own original
// superslm_test_phaseD::kDampedGreedyConstantsScaleFlag (0x2, PRESUMPTIVE at authoring) to
// superslm::kDampedGreedyArtifactConstantsFlag (0x2, same value, superslm namespace) --
// necessary because a translation unit bringing both `namespace superslm` and
// `namespace superslm_test_phaseD` into scope (every cell file in this suite does) hit
// C2872 ambiguous symbol otherwise. Every test file in this suite is updated (this same
// commit) to reference superslm::kDampedGreedyArtifactConstantsFlag directly -- the
// presumptive local constant this stub used to declare is REMOVED here, not aliased, so a
// stale reference fails to compile rather than silently resolving to a name nobody maintains.
//
// The stub's own historical PRESUMPTIVE-bit framing turned out correct in substance (0x2 was
// genuinely unclaimed at authoring and is exactly the bit Phase D1 claimed) -- what changed is
// only the SYMBOL NAME, not the value, confirming the stub header's own "if D1 claims a
// different bit, update in the same edit" contingency never had to fire on the bit itself.
#ifndef SSLM_T2199_PHASED_STUB_H
#define SSLM_T2199_PHASED_STUB_H

#include "superslm/sslm_phaseD.h"

#endif  // SSLM_T2199_PHASED_STUB_H
