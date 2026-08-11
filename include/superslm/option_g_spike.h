// T-1891 -- Option G (T-1822 §29.4/§30, D-SLM2305) C++ measurement spike.
//
// DISPOSABLE. This header and its implementation (src/forward/forward_sites.cpp's
// Option-G-tagged additions) exist only on branch brunel/t1891-optionG-spike, in the
// worktree D:\SuperSLM\.worktrees\t1891-optionG-spike. They are an instrument, not a
// production interface, and this branch is never merged (T-1891's own commissioning
// brief). Nothing outside this spike includes this header.
//
// What it exposes: a runtime toggle between the engine's two K-landing paths (the
// shipped land-then-rotate order, and Option G's fused rotate-then-land order), and a
// per-(layer, kv_head) saturation-count instrument for the K landing's ClampRopeCode,
// wired for both paths so gate G5's old-vs-fused delta is directly readable. Q, V, the
// K/V store layout, and the artifact schema are unmodified by anything declared here.
#ifndef SUPERSLM_OPTION_G_SPIKE_H
#define SUPERSLM_OPTION_G_SPIKE_H

#include <cstdint>

namespace superslm {

// The rotated wide pair, matching `RopePair`'s own shape (intmath.h) but named
// distinctly since this is a spike-local type, not a change to the shared header.
struct RopePairWide {
	int64_t x = 0;
	int64_t y = 0;
};

// Option G's own wide RoPE-pair primitive (T-1822 §29.4's "int64-input,
// __int128-intermediate sibling of the RoPE pair primitive, Q2.30 tables
// unchanged"). Defined in forward_sites.cpp (this toolchain -- MSVC -- has no
// native __int128; the definition uses the same portable-128-bit substitution
// that file's own U128 facility already makes for LandingRescale's C27
// composite). Combination and rounding are IDENTICAL to `RopeApplyPair`
// (intmath.cpp): `(x*cos - y*sin, x*sin + y*cos)`, one C3 (ties-away-from-zero)
// rounding at ROPE_FRAC_BITS -- only the input width and the intermediate's own
// width differ. UNCLAMPED (matches `RopeApplyPair`'s own contract: "clamping ...
// is the caller's") -- the K-landing call site clamps through the EXISTING
// LandingRescale+ClampRopeCode pair, unchanged, per D-SLM2305's construction;
// this primitive is not a second clamp. `*out_in_domain` is false (refuse, not
// wrap) whenever either rotated component's ROUNDED value (the value this
// function returns -- T-1892 Minor 1: the check is computed on the value AFTER
// C3's rounding, not on some unrounded "true" value the function never
// materializes) does not fit int64_t -- T-1891 gate G2 exercises this directly.
RopePairWide RopeApplyPairWide(int64_t x, int64_t y, int32_t cos_q30, int32_t sin_q30,
                                bool* out_in_domain);

// Reads the environment variable SSLM_OPTION_G_FUSED_K_LANDING once (cached in a
// function-local static -- this spike's driver and test processes are single-threaded
// over the forward pass, so no synchronization is needed for the cache), non-empty and
// not "0" meaning ON (the fused path). Unset or "0" means OFF (the shipped path,
// byte-identical to main@c6cfa03 -- T-1891 gate G1). This is the ONE flag both the K/V
// landing block (forward_sites.cpp) and any spike tool read to select a path; there is
// no second toggle mechanism anywhere in this spike.
bool OptionGFusedKLandingEnabled();

// Per-(layer, kv_head) K-landing ClampRopeCode saturation counts (T-1891 gate G5),
// separate from `SequenceLayerState::kv_saturation_count` (the shipped, per-sequence,
// K-and-V-shared production counter, which this spike does not modify). Reset before a
// run (mandatory -- every accessor below aborts with a diagnostic if queried first,
// T-1892 Minor 3: an unreset instrument must refuse, not silently report a believable
// zero), then read after.
//
// T-1892 Critical 2: the shipped K path clamps TWICE -- once at the K/V landing
// (site 4, `ClampRopeCode(LandingRescale(...))`) and once after the post-landing
// rotation (site 7, `RopeApplySite`). The fused path clamps ONCE, at the landing.
// THREE counters, not two, because the like-for-like comparison and the
// boundary-being-deleted are different questions:
//   - `OptionGSaturationCountOld`      -- the SHIPPED path's own LANDING clamp (site
//                                         4). The like-for-like baseline: the SAME
//                                         clamp `Fused` counts, before the rotation
//                                         moves in front of it.
//   - `OptionGSaturationCountFused`    -- the FUSED path's single landing clamp
//                                         (site 4, post-rotation-then-land).
//   - `OptionGSaturationCountOldSite7` -- the SHIPPED path's post-rotation clamp
//                                         (site 7) -- the SECOND boundary Option G
//                                         deletes. Not the baseline for the
//                                         old-vs-fused delta; reported separately so
//                                         it is not silently conflated with it again.
// All three exist unconditionally (not gated behind the runtime flag above) so a
// single process that runs both paths in sequence (as gate G5's own tool does)
// accumulates each path's own counts independently, in the same run, without a
// rebuild.
void OptionGResetSaturationCounters(uint32_t num_layers, uint32_t num_kv_heads);
uint64_t OptionGSaturationCountOld(uint32_t layer, uint32_t kv_head);
uint64_t OptionGSaturationCountFused(uint32_t layer, uint32_t kv_head);
uint64_t OptionGSaturationCountOldSite7(uint32_t layer, uint32_t kv_head);

}  // namespace superslm

#endif  // SUPERSLM_OPTION_G_SPIKE_H
