// T-2199 (Curie) -- Phase D stub declarations: the ABI-surface (D1), wiring (D2/D3/D4), and
// artifact-flag (D1) symbols this suite calls but no production .cpp defines yet. Declared,
// not defined, per this suite's own established Phase A/C convention (sslm_damped_greedy.h's
// own header comment, this directory) -- signatures here are what a real Phase D build is
// expected to satisfy; the build seat's own production copy supersedes this file exactly the
// way include/superslm/sslm_damped_greedy.h superseded this directory's Phase A/C stub.
//
// Design of record: Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md Sec8 Phase D, Sec9
// dimensions 1-3/5/6/7/9/10 (the D-sited cells). Binding rulings: D-SLM3794 (Decision A
// mechanism -- a NEW FIELD gated behind a flags bit, no version bump, a pre-damped-greedy
// runtime rejects a flagged artifact with BadHeader per docs/sslm_format.md's own versioning
// semantics), D-SLM3795 (Decision B -- the decode digest is decoder-agnostic, covers the
// configured decoder's tokens), D-SLM3719/D-SLM3723/D-SLM3727/D-SLM3756 as before.
//
// Mechanism choice, stated so a reader does not mistake this suite's own scaffolding for a
// design ruling: Sec8 D1 leaves "an additive field or a versioned successor struct" open. This
// suite pins the OBSERVABLE contract (mode selection reaches the primitive's own exact output;
// greedy mode is bit-unchanged; validation is symmetric across entry points) against a
// successor-struct/successor-entry-point surface that either real mechanism can satisfy --
// an additive-field build simply makes sslm_decode_step_damped_greedy a thin wrapper that
// copies the extra fields into the sslm_decode_params it already extended and calls straight
// through to the (now mode-aware) sslm_decode_stepImpl.
//
// Flag-bit choice, stated for the same reason: kDampedGreedyConstantsScaleFlag below is
// PRESUMPTIVE (0x2, the next bit after Option-G's shipped 0x1, include/superslm/artifact.h) --
// chosen because it is genuinely unclaimed as of this suite's own authoring commit. If Phase
// D1 claims a DIFFERENT bit, the two cells in phaseD1_artifact_flag_red.cpp that reference this
// constant BY VALUE (TestD1a_UnflaggedArtifact_ByteLevelRegression's flags=0 half is
// unaffected either way; TestD1c_PreDampedGreedyLoader_RejectsPresumptiveFlagBit_BadHeader is
// the one that would need updating) must be updated in the SAME edit that lands D1, matching
// this codebase's own established convention for widening kKnownArtifactFlagsMask
// (include/superslm/artifact.h's own header comment on kOptionGFusedKLandingFlag).
#ifndef SSLM_T2199_PHASED_STUB_H
#define SSLM_T2199_PHASED_STUB_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/sslm_abi.h"   // the REAL, already-shipped production CPU ABI header (Phase
                                  // 1.1) -- Phase D builds on top of it, so this suite reads it
                                  // directly rather than maintaining a promoted copy the way
                                  // tests/t2138-abi-red-suite does for an ABI that was itself
                                  // unbuilt at authoring time.
#include "sslm_damped_greedy.h"  // Phase A/C's own promoted copy -- AntiLmState, FsdTopK,
                                  // TopKRenormalizeQ15, DampedGreedyScoreAndArgmax[Diag]. Real
                                  // as of feature/t2199-damped-greedy@c02b156.

namespace superslm_test_phaseD {

// --- D1: Decision A's artifact-carried scale constants (plan Sec8 B1, Sec3 Option 1) -------

// PRESUMPTIVE bit -- see this file's own header comment above.
inline constexpr uint32_t kDampedGreedyConstantsScaleFlag = 0x2u;

// Phase B1's own per-model (m, e) for the LM head's logit scale (plan Sec8 B1 -- "mirrors the
// existing per-head attention derivation"). This is the ONLY damped-greedy-specific data
// Decision A puts in the artifact; alpha/n/k are Phase D1's own ABI-surface build-time
// constants (sslm_decode_params_damped_greedy below), never artifact fields.
struct DampedGreedyScaleConstants {
	int64_t scale_mantissa_m;
	int32_t scale_exponent_e;
};

// True iff `art`'s header flags set kDampedGreedyConstantsScaleFlag AND a well-formed
// constants blob is present. Declared, not defined -- Phase D1's own artifact-reader
// obligation.
bool ArtifactHasDampedGreedyConstants(const superslm::SslmArtifact& art) noexcept;

// Reads the constants. Domain (plan Sec2.3's own M <= INT64_MAX / vocab_size constraint,
// Phase B2): a malformed or out-of-domain blob is a DEFINED rejection (returns false, *out
// left untouched), matching this codebase's own reject-over-degrade discipline for every other
// artifact-carried field (Sec9 dim2's hostile-input standard, applied to this new field).
bool ReadDampedGreedyScaleConstants(const superslm::SslmArtifact& art,
                                     DampedGreedyScaleConstants* out) noexcept;

// --- D1: the ABI-surface mode selector + build-time (alpha, n, k) (plan Sec8 D1) -----------

enum class DampedGreedyMode : int32_t { kGreedy = 0, kDampedGreedy = 1 };

struct sslm_decode_params_damped_greedy {
	int32_t layer_budget;
	DampedGreedyMode mode;
	int64_t alpha_q15;
	int32_t anti_lm_max_order;  // n
	int32_t top_k;              // k
	// The runtime (q_ln2, q_b, q_c) triple Phase B3 derives once per model load from the
	// artifact-carried (m, e) above -- taken directly here (not re-derived per call) matching
	// Phase B3's own "computed once and cached, unlike attention's per-position dynamic scale"
	// framing (plan Sec8 B3). A real D1 build may instead cache these on the model handle;
	// this suite passes them explicitly so the wiring cells below do not also have to
	// construct or exercise Phase B's own derivation path, which is out of this suite's scope.
	int64_t q_ln2;
	int64_t q_b;
	int64_t q_c;
};

// --- D2: the batched-ABI entry point (plan Sec8 D2, sslm_decode_stepImpl) -------------------
// Mirrors sslm_decode_step's own signature and caller-facing contract exactly
// (superslm/sslm_abi.h), plus the extended params above. Declared, not defined.
extern "C" sslm_status sslm_decode_step_damped_greedy(sslm_model model, sslm_seq* seqs, int32_t n,
                                                        const sslm_decode_params_damped_greedy* params,
                                                        sslm_workspace ws, int32_t* out_tokens);

// --- D3: the free-text CLI/RunGreedyDecodeLoop entry point (plan Sec8 D3, forward_sites.cpp) -
// Mirrors superslm::RunGreedyDecodeLoop's own signature exactly (superslm/forward_sites.h),
// plus the same mode/alpha/n/k/scale extension D2 carries -- the "one implementation, never
// two" primitive (DampedGreedyScoreAndArgmax) is what both call sites are specified to share
// (plan Sec7, Sec8 D3's own "one implementation" citation); this suite's cross-entry
// validation-symmetry cell (phaseD2_wiring_red.cpp) is what confirms that sharing was actually
// honored rather than reimplemented per site. Declared, not defined.
superslm::SslmForwardStatus RunGreedyOrDampedGreedyDecodeLoop(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    const int32_t* prompt_tokens, size_t prompt_len, const int8_t* embed_weights,
    superslm::CarriedScale embed_site_constant, const int32_t* final_norm_gain,
    superslm::CarriedScale final_norm_site_constant, const int8_t* head_weights, int32_t vocab_size,
    const int32_t* stop_ids, size_t stop_count, size_t max_new_tokens, uint8_t* workspace,
    size_t workspace_size, int32_t* out_tokens, int32_t* out_logit_rows, size_t out_tokens_capacity,
    size_t* out_tokens_produced, superslm::SslmDecodeStopReason* out_stop_reason,
    superslm::SslmKvPrecision kv_precision, bool option_g_fused_k_landing, DampedGreedyMode mode,
    int64_t alpha_q15, int32_t anti_lm_max_order, int32_t top_k, int64_t q_ln2, int64_t q_b,
    int64_t q_c);

// --- D4: the GPU parity bridge entry point (plan Sec8 D4, src/gpu/gpu_1p0.cpp:2057-2075) ----
// A minimal surface -- this suite has no GPU device to drive on this machine, so this
// declaration exists only so the cross-entry validation-symmetry cell can name all three wired
// sites in one place; the GPU half of that symmetry is routed with an honest SKIP, matching
// this suite's own established discipline (dim5's adversarial-search cells) and T-2112's own
// GPU-conditional convention, never fabricated.
struct DampedGreedyValidationParams {
	DampedGreedyMode mode;
	int64_t alpha_q15;
	int32_t anti_lm_max_order;
	int32_t top_k;
};

// --- D2/D3 shared: params validation, callable directly so the cross-entry symmetry cell can
// assert the SAME rejection status without needing three fully-wired call stacks live at once
// (the CLI/GPU stacks are not otherwise exercised by this suite -- see the SKIP note on the
// validation-symmetry cell itself for why this is not circular). Declared, not defined; a real
// D1/D2/D3/D4 build satisfies this by construction if (and only if) every entry point's own
// domain check is the SAME function, which is the property this cell exists to catch a
// divergence in.
bool ValidateDampedGreedyParams(const DampedGreedyValidationParams& p, int32_t vocab_size) noexcept;

}  // namespace superslm_test_phaseD

#endif  // SSLM_T2199_PHASED_STUB_H
