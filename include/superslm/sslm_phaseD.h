// sslm_phaseD.h -- T-2199 Phase D production surface: the artifact-carried scale constants
// (D1), the mode selector + validation (D2), the free-text decode-loop wrapper (D3), and the
// shared params-validation function D2/D3/D4 all route through.
//
// T-2199 Phase D review fix S5 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md), 2026-08-20: this
// surface now lives in `namespace superslm` -- the real production namespace, not the
// suite-scaffolding `superslm_test_phaseD` this file previously matched (the review: "production
// C-ABI surface ships under a test-named namespace and header ... on a repo that is a public
// portfolio artifact, that should not be what ships"). The suite's own
// `tests/t2199-damped-greedy-red-suite/sslm_phaseD_stub.h` simply `#include`s this header and
// every suite `.cpp` file brings both `using namespace superslm;` and
// `using namespace superslm_test_phaseD;` into scope (verified: no suite file qualifies any
// symbol as `superslm_test_phaseD::`), so relocating here needs no suite-side edit to keep
// linking -- an empty `namespace superslm_test_phaseD {}` is kept at the bottom of this file
// purely so that `using namespace superslm_test_phaseD;` remains valid syntax.
//
// `sslm_decode_step_damped_greedy`/`sslm_decode_params_damped_greedy` (below) are KEPT, not
// deleted, even though S5 also flags them as "precisely the versioned successor struct plus
// parallel entry point Sec8 D1 ruled against, now shipping alongside the additive-field shape
// the ruling chose. Only one was ruled." -- correct, and NOT fixed here: `phaseD2_wiring_red.cpp`
// / `phaseD2a_cost_ratio_red.cpp` / `phaseD3_teardown_red.cpp` (test files, outside this build's
// writable scope) construct their own primary cells by calling
// `sslm_decode_step_damped_greedy(...)` directly -- deleting it would be red-by-link across three
// suite files, not a build-time fix. The wrapper below is now explicitly a thin, suite-
// compatibility SHIM over the real, additive-field `sslm_decode_step`/`sslm_decode_params`
// (which IS the production-recommended path -- see `tools/sslm_generate.cpp`'s own S4 CLI wiring,
// which calls the real ABI directly, never this wrapper). Routed to the suite owner: migrate
// those three cells onto `sslm_decode_step`/`sslm_decode_params` directly, then this wrapper and
// `sslm_decode_params_damped_greedy` can be deleted in the same pass.
//
// Design of record: Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md Sec8 Phase D, Sec9
// dimensions 1-3/5/6/7/9/10. Binding rulings: D-SLM3794 (Decision A mechanism -- additive
// field, no version bump; a pre-damped-greedy runtime rejects a flagged artifact with
// BadHeader), D-SLM3795 (Decision B -- the decode digest covers the configured decoder's
// tokens).
#ifndef SUPERSLM_SSLM_PHASED_H
#define SUPERSLM_SSLM_PHASED_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/sslm_abi.h"

namespace superslm {

// --- D1: Decision A's artifact-carried scale constants (plan Sec8 B1, Sec3 Option 1) ---------
//
// The real flag value lives at superslm::kDampedGreedyArtifactConstantsFlag (include/superslm/
// artifact.h).

struct DampedGreedyScaleConstants {
	int64_t scale_mantissa_m;
	int32_t scale_exponent_e;
};

// The real target vocab this design's own domain check is against (plan Sec2.3: "the derived M
// overflows int64_t at the real 151,936-wide vocab" -- both target checkpoints, 0.5B and 1.5B,
// share this exact vocab_size). A fixed constant rather than read from the artifact's own
// Config section: the DGC1 section this file defines carries no vocab_size of its own (Config
// is opaque JSON at this layer, per sslm_fixtures.h's own "no cell asserts its meaning" fixture
// convention), and Phase B2's own obligation is verification against these two real
// checkpoints specifically, not an arbitrary artifact's declared width.
inline constexpr int64_t kRealVocabSizeForDomainCheck = 151936;

[[nodiscard]] bool ArtifactHasDampedGreedyConstants(const superslm::SslmArtifact& art) noexcept;
[[nodiscard]] bool ReadDampedGreedyScaleConstants(const superslm::SslmArtifact& art,
                                                   DampedGreedyScaleConstants* out) noexcept;

// The converter-side/hostile-fixture writer (plan Sec8 D1's own disposition: "whoever builds
// D1's section layout writes the byte-writer against it, in the same pass"). Appends a
// DampedGreedyConstants (DGC1) section carrying `constants` to `sections`, in the exact byte
// layout ReadDampedGreedyScaleConstants above parses -- usable both by a real converter
// (Python emits the equivalent bytes independently, see tools/convert_model.py) and by a test
// fixture that needs a REAL, byte-level hostile (m, e) pair (TestD1d's own owed fixture
// upgrade, sslm_phaseD_stub.h) rather than a struct set directly on the output. Does not set
// the artifact's own flags word -- the caller does that (matching MakeConfigSection's own
// "one concern per helper" shape, sslm_fixtures.h).
struct RawSection {
	uint32_t type;
	uint32_t dtype;
	std::vector<uint8_t> data;
	uint32_t alignment;
};
[[nodiscard]] RawSection MakeDampedGreedyConstantsSection(
    const DampedGreedyScaleConstants& constants);

// --- D1: the ABI-surface mode selector + build-time (alpha, n, k) (plan Sec8 D1) -------------

enum class DampedGreedyMode : int32_t { kGreedy = 0, kDampedGreedy = 1 };

struct sslm_decode_params_damped_greedy {
	int32_t layer_budget;
	DampedGreedyMode mode;
	int32_t alpha_q15;  // C2 fix: int32_t per plan Sec2.5, matches sslm_decode_params (sslm_abi.h)
	int32_t anti_lm_max_order;
	int32_t top_k;
	int64_t q_ln2;
	int64_t q_b;
	int64_t q_c;
};

// --- D2/D3 shared: params validation (plan Sec9 dim2) -----------------------------------------
struct DampedGreedyValidationParams {
	DampedGreedyMode mode;
	int32_t alpha_q15;  // C2 fix: int32_t, see sslm_decode_params_damped_greedy's own comment
	int32_t anti_lm_max_order;
	int32_t top_k;
};
[[nodiscard]] bool ValidateDampedGreedyParams(const DampedGreedyValidationParams& p,
                                               int32_t vocab_size) noexcept;

// --- D2: the batched-ABI entry point (plan Sec8 D2, sslm_decode_stepImpl) --------------------
extern "C" sslm_status sslm_decode_step_damped_greedy(sslm_model model, sslm_seq* seqs, int32_t n,
                                                        const sslm_decode_params_damped_greedy* params,
                                                        sslm_workspace ws, int32_t* out_tokens);

// --- D3: the free-text CLI/RunGreedyDecodeLoop entry point (plan Sec8 D3, forward_sites.cpp) -
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
    int32_t alpha_q15, int32_t anti_lm_max_order, int32_t top_k, int64_t q_ln2, int64_t q_b,
    int64_t q_c);

}  // namespace superslm

// S5: empty on purpose -- see this file's own top comment. Kept so
// `using namespace superslm_test_phaseD;` (every suite `.cpp` file in
// tests/t2199-damped-greedy-red-suite/) stays valid syntax without pulling in a second copy of
// anything.
namespace superslm_test_phaseD {}

#endif  // SUPERSLM_SSLM_PHASED_H
