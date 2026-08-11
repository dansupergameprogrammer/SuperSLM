// T-1899 -- Curie's red suite for T-1894 (T-1822 design Sec31.2, the
// production Option-G build shape: fused post-RoPE K landing). Fixtures and
// golden constants for the RunLayerLoop-level cells (Sec12's "Engine/
// reference bit-parity", "Mutation vitality on the deleted landing order",
// "Selection dispatch", "Fused-site domain check, dynamic", and
// "Microstep/whole-token parity, compiled engine" bullets).
//
// Test-design record: Claude/Curie/t1899-optionG-red-suite-2026-08-11.md
// (records worktree, D:\Wizard).
//
// GOLDEN-CONSTANT DERIVATION (StandardsDocument.md Sec5.4 -- exactness
// verified at source or by execution, never by construction). Every value
// below was obtained by EXECUTING the real, already-shipped primitives this
// design leaves untouched (RmsNormSite, GemmInt8AccumulateRow,
// ApplyWeightScaleFold, LandingRescale, ClampRopeCode, RopeApplyPair,
// DynamicScaleReciprocal -- all defined on main@c6cfa03, none of them
// touched by this suite or by T-1894's own build), via a disposable scratch
// program (never committed) that linked against those real objects and
// printed their outputs for this file's own hand-chosen inputs. The ONLY
// arithmetic not exercised through the real, shipped code is the wide
// rotation itself (RopeApplyPairWide does not exist on main), which is
// computed here by transcribing its own documented formula --
// "(x*cos - y*sin, x*sin + y*cos), one C3 (ties-away-from-zero) rounding at
// ROPE_FRAC_BITS" (forward_sites.h, this design) -- into exact int64
// arithmetic, chosen deliberately small enough (kacc magnitudes here are
// int8-GEMM-scale, cos/sin <= 2^30) that no 128-bit intermediate is needed
// to stay exact, unlike the domain-EXTREMITY cells (test_main.cpp's own
// T-1899 section), which deliberately DO reach magnitudes needing one and
// are derived independently in Python instead (arbitrary-precision, the
// same convention T-1839 already established).
#ifndef SSLM_T1899_OPTIONG_FIXTURES_H
#define SSLM_T1899_OPTIONG_FIXTURES_H

#include <cstdint>

#include "superslm/artifact.h"
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "sslm_fixtures.h"
#include "sslm_cfg1_hostile_fixtures.h"   // Cfg1Spec, BuildCfg1
#include "sslm_sil1_hostile_fixtures.h"   // MakeSigmoidLutSection
#include "sslm_model_hostile_fixtures.h"  // ManifestTensorSpec, BuildManifest

namespace superslm_test {

// Self-contained re-implementation of test_main.cpp's own file-local
// MakeRop1SectionMultiRow (test_main.cpp:11484) -- that helper is `static` in
// test_main.cpp's own translation unit and this header must not depend on
// inclusion ORDER relative to it, so the identical construction (a ROP1
// section with `context_cap` rows of `pairs` cos/sin entries each) is
// reproduced here under its own name.
inline FixtureSection MakeOptionGRop1SectionMultiRow(int32_t context_cap, int32_t pairs,
                                                       const int64_t* cos_flat,
                                                       const int64_t* sin_flat) {
	std::vector<ManifestTensorSpec> tensors = {
	    {"cos", {static_cast<uint32_t>(context_cap), static_cast<uint32_t>(pairs)}},
	    {"sin", {static_cast<uint32_t>(context_cap), static_cast<uint32_t>(pairs)}},
	};
	auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	const size_t n = static_cast<size_t>(context_cap) * static_cast<size_t>(pairs);
	for (size_t i = 0; i < n; ++i) {
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + i * 8,
		       static_cast<uint64_t>(cos_flat[i]));
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + i * 8,
		       static_cast<uint64_t>(sin_flat[i]));
	}
	return MakeSection(superslm::SslmSectionType::RopeTables, superslm::SslmDtype::Int64,
	                   manifest.bytes, /*alignment=*/64);
}

// The 45-degree-class Q2.30 rotation table entry this suite's cells share --
// round(2^30 / sqrt(2)) = 759250125. Chosen (not a real angle from a real
// artifact) because it is the SMALLEST-degree rotation that genuinely MIXES
// both components -- a 0-degree (identity) or 90-degree (pure swap-negate)
// table entry would make clamp(rotate(x)) == rotate(clamp(x)) by
// construction (confirmed by this fixture's own derivation run: a 90-degree
// probe produced IDENTICAL fused and legacy outputs on the discriminating
// fixture below, because ClampRopeCode's range is symmetric and a
// swap-negate permutation commutes with a symmetric per-component clamp) --
// silently hiding the exact divergence the mutation-vitality cell exists to
// detect. 45 degrees forces a real weighted sum before the C3 rounding,
// which is where the fused (rotate-wide-then-land-once) and legacy
// (land-then-rotate-narrow) orders genuinely diverge.
inline constexpr int32_t kOptionG45DegQ30 = 759250125;

// --- Single-layer, single-(kv-)head fixture for the K-landing cells -------
//
// hidden_size=2, head_dim=2 (ONE RoPE pair per head, matching TwoLayerFixture's
// own established minimal geometry), num_attention_heads=num_key_value_heads=1,
// context_cap=2 (two committed positions: 0 carries an IDENTITY rope-table row
// for the "null configuration" cells, 1 carries the 45-degree row for the
// "non-null configuration" cells). Self-contained (its own identity-fold and
// weight arrays) rather than sharing test_main.cpp's file-scope
// kIdentityFoldArr/kZeroFoldArr globals, so this header has no ordering
// dependency on where it is included relative to those definitions.
struct OptionGKvLandingFixture {
	static constexpr int32_t kContextCap = 2;

	superslm::SslmModelView view;
	superslm::LayerWeights layers[1];

	int64_t kv_landing_r_t_arr[1];
	int64_t kv_landing_e_t_arr[1] = {0};
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};
	int32_t norm_gain[2] = {16384, 16384};  // Q16 0.25 -- TwoLayerFixture's own vetted-safe choice
	int8_t identity2x2[4] = {1, 0, 0, 1};
	int32_t identity_fold_arr[2] = {1, 1};
	int32_t zero_fold_arr[2] = {0, 0};
	int64_t iexp_softmax_khead_m_arr[1] = {INT64_C(1073741824)};
	int64_t iexp_softmax_khead_e_arr[1] = {-86};

	OptionGKvLandingFixture(OptionGKvLandingFixture const&) = delete;
	OptionGKvLandingFixture& operator=(OptionGKvLandingFixture const&) = delete;
	OptionGKvLandingFixture(OptionGKvLandingFixture&&) = delete;
	OptionGKvLandingFixture& operator=(OptionGKvLandingFixture&&) = delete;

	OptionGKvLandingFixture() {
		using superslm::CarriedScale;

		OptionGKvLandingFixture& f = *this;
		Cfg1Spec spec{};
		spec.hidden_size = 2;
		spec.num_hidden_layers = 1;
		spec.num_attention_heads = 1;
		spec.num_key_value_heads = 1;
		spec.head_dim = 2;
		spec.intermediate_size = 2;
		spec.context_cap = OptionGKvLandingFixture::kContextCap;
		spec.kv_precision = 0;  // Int8
		spec.kv_block_size = 1;
		FixtureSection config = MakeSection(superslm::SslmSectionType::Config,
		                                    superslm::SslmDtype::Raw, BuildCfg1(spec));

		// Row 0: identity (cos=ROPE_ONE, sin=0) -- the "null configuration"
		// cells' own position. Row 1: the 45-degree entry above -- the
		// "non-null configuration" cells' own position.
		const int64_t cos_flat[2] = {INT64_C(1073741824), static_cast<int64_t>(kOptionG45DegQ30)};
		const int64_t sin_flat[2] = {INT64_C(0), static_cast<int64_t>(kOptionG45DegQ30)};
		FixtureSection rope =
		    MakeOptionGRop1SectionMultiRow(/*context_cap=*/OptionGKvLandingFixture::kContextCap,
		                                   /*pairs=*/1, cos_flat, sin_flat);
		auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope});
		std::string err;
		const auto status =
		    superslm::SslmModel::Load(built.bytes.data(), built.bytes.size(), f.view, &err);
		if (status != superslm::SslmModelStatus::Ok) {
			std::fprintf(stderr, "OptionGKvLandingFixture's own minimal artifact failed to load: "
			                     "%s (%s)\n",
			             superslm::SslmModelStatusName(status), err.c_str());
			std::abort();
		}

		// canonical CarriedScale, matching TwoLayerFixture's own convention --
		// m=2^30, e=-30.
		const CarriedScale canonical{INT64_C(1073741824), -30};
		f.kv_landing_r_t_arr[0] = superslm::DynamicScaleReciprocal(canonical.m);

		superslm::LayerWeights& lw = f.layers[0];
		lw.attn_norm_gain = f.norm_gain;
		lw.attn_norm_site_constant = canonical;
		lw.q_weight = f.identity2x2;
		lw.k_weight = f.identity2x2;
		lw.v_weight = f.identity2x2;
		lw.o_weight = f.identity2x2;
		lw.q_fold_identity = f.identity_fold_arr;
		lw.q_fold_mult = f.zero_fold_arr;
		lw.q_fold_shift = f.zero_fold_arr;
		lw.k_fold_identity = f.identity_fold_arr;
		lw.k_fold_mult = f.zero_fold_arr;
		lw.k_fold_shift = f.zero_fold_arr;
		lw.v_fold_identity = f.identity_fold_arr;
		lw.v_fold_mult = f.zero_fold_arr;
		lw.v_fold_shift = f.zero_fold_arr;
		lw.o_fold_identity = f.identity_fold_arr;
		lw.o_fold_mult = f.zero_fold_arr;
		lw.o_fold_shift = f.zero_fold_arr;
		lw.gate_fold_identity = f.identity_fold_arr;
		lw.gate_fold_mult = f.zero_fold_arr;
		lw.gate_fold_shift = f.zero_fold_arr;
		lw.up_fold_identity = f.identity_fold_arr;
		lw.up_fold_mult = f.zero_fold_arr;
		lw.up_fold_shift = f.zero_fold_arr;
		lw.down_fold_identity = f.identity_fold_arr;
		lw.down_fold_mult = f.zero_fold_arr;
		lw.down_fold_shift = f.zero_fold_arr;
		lw.q_site_constant = canonical;
		lw.o_site_constant = canonical;
		lw.kv_landing_r_t_k = f.kv_landing_r_t_arr;
		lw.kv_landing_e_t_k = f.kv_landing_e_t_arr;
		lw.kv_landing_r_t_v = f.kv_landing_r_t_arr;
		lw.kv_landing_e_t_v = f.kv_landing_e_t_arr;
		lw.ctx_fold_identity = f.ctx_fold_identity_arr;
		lw.ctx_fold_mult = f.ctx_fold_mult_arr;
		lw.ctx_fold_shift = f.ctx_fold_shift_arr;
		lw.ctx_fold_site_constant = canonical;
		lw.attn_residual_site_constant = canonical;
		lw.iexp_softmax_khead_m = f.iexp_softmax_khead_m_arr;
		lw.iexp_softmax_khead_e = f.iexp_softmax_khead_e_arr;
		lw.mlp_norm_gain = f.norm_gain;
		lw.mlp_norm_site_constant = canonical;
		lw.gate_weight = f.identity2x2;
		lw.up_weight = f.identity2x2;
		lw.down_weight = f.identity2x2;
		lw.gate_site_constant = canonical;
		lw.up_site_constant = canonical;
		lw.mlp_act_site_constant = CarriedScale{INT64_C(1073741824), -96};  // TwoLayerFixture's own
		                                                                     // in-domain choice
		lw.down_site_constant = canonical;
		lw.mlp_residual_site_constant = canonical;
	}
};

// T-1894 build round 4 (T-1901 Significant 2/D-SLM2419, D-SLM2427): a
// GENUINELY two-layer sibling of `OptionGKvLandingFixture`, for the
// compiled-engine microstep/whole-token parity cell -- design Sec12's own
// respecified text explicitly excludes a one-layer/call-twice construction
// ("a one-layer fixture, or any construction where both halves of the
// comparison route through one call with identical arguments, is not a
// valid instance of this cell"). Both layers carry IDENTICAL per-layer
// weights/scales to `OptionGKvLandingFixture`'s own single layer -- this
// fixture's own test does not need a pre-derived golden K value at layer 1
// (the residual stream carries layer 0's own output into layer 1's input,
// so layer 1's kacc is NOT the same value layer 0's own golden constants
// describe): the cell's own claim is that a GENUINELY resumed multi-call
// path (layer_index advancing 0->1->2 across two separate RunLayerLoop
// calls) computes the IDENTICAL K-store output a single whole-token call
// (layer_budget=2) does -- a self-consistency claim, not a value-match one.
struct OptionGTwoLayerKvLandingFixture {
	static constexpr int32_t kContextCap = 2;

	superslm::SslmModelView view;
	superslm::LayerWeights layers[2];

	int64_t kv_landing_r_t_arr[1];
	int64_t kv_landing_e_t_arr[1] = {0};
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};
	int32_t norm_gain[2] = {16384, 16384};
	int8_t identity2x2[4] = {1, 0, 0, 1};
	int32_t identity_fold_arr[2] = {1, 1};
	int32_t zero_fold_arr[2] = {0, 0};
	int64_t iexp_softmax_khead_m_arr[1] = {INT64_C(1073741824)};
	int64_t iexp_softmax_khead_e_arr[1] = {-86};

	OptionGTwoLayerKvLandingFixture(OptionGTwoLayerKvLandingFixture const&) = delete;
	OptionGTwoLayerKvLandingFixture& operator=(OptionGTwoLayerKvLandingFixture const&) = delete;
	OptionGTwoLayerKvLandingFixture(OptionGTwoLayerKvLandingFixture&&) = delete;
	OptionGTwoLayerKvLandingFixture& operator=(OptionGTwoLayerKvLandingFixture&&) = delete;

	OptionGTwoLayerKvLandingFixture() {
		using superslm::CarriedScale;

		OptionGTwoLayerKvLandingFixture& f = *this;
		Cfg1Spec spec{};
		spec.hidden_size = 2;
		spec.num_hidden_layers = 2;
		spec.num_attention_heads = 1;
		spec.num_key_value_heads = 1;
		spec.head_dim = 2;
		spec.intermediate_size = 2;
		spec.context_cap = OptionGTwoLayerKvLandingFixture::kContextCap;
		spec.kv_precision = 0;  // Int8
		spec.kv_block_size = 1;
		FixtureSection config = MakeSection(superslm::SslmSectionType::Config,
		                                    superslm::SslmDtype::Raw, BuildCfg1(spec));

		const int64_t cos_flat[2] = {INT64_C(1073741824), static_cast<int64_t>(kOptionG45DegQ30)};
		const int64_t sin_flat[2] = {INT64_C(0), static_cast<int64_t>(kOptionG45DegQ30)};
		FixtureSection rope = MakeOptionGRop1SectionMultiRow(
		    /*context_cap=*/OptionGTwoLayerKvLandingFixture::kContextCap, /*pairs=*/1, cos_flat, sin_flat);
		auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope});
		std::string err;
		const auto status =
		    superslm::SslmModel::Load(built.bytes.data(), built.bytes.size(), f.view, &err);
		if (status != superslm::SslmModelStatus::Ok) {
			std::fprintf(stderr, "OptionGTwoLayerKvLandingFixture's own minimal artifact failed to "
			                     "load: %s (%s)\n",
			             superslm::SslmModelStatusName(status), err.c_str());
			std::abort();
		}

		const CarriedScale canonical{INT64_C(1073741824), -30};
		f.kv_landing_r_t_arr[0] = superslm::DynamicScaleReciprocal(canonical.m);

		for (int l = 0; l < 2; ++l) {
			superslm::LayerWeights& lw = f.layers[l];
			lw.attn_norm_gain = f.norm_gain;
			lw.attn_norm_site_constant = canonical;
			lw.q_weight = f.identity2x2;
			lw.k_weight = f.identity2x2;
			lw.v_weight = f.identity2x2;
			lw.o_weight = f.identity2x2;
			lw.q_fold_identity = f.identity_fold_arr; lw.q_fold_mult = f.zero_fold_arr; lw.q_fold_shift = f.zero_fold_arr;
			lw.k_fold_identity = f.identity_fold_arr; lw.k_fold_mult = f.zero_fold_arr; lw.k_fold_shift = f.zero_fold_arr;
			lw.v_fold_identity = f.identity_fold_arr; lw.v_fold_mult = f.zero_fold_arr; lw.v_fold_shift = f.zero_fold_arr;
			lw.o_fold_identity = f.identity_fold_arr; lw.o_fold_mult = f.zero_fold_arr; lw.o_fold_shift = f.zero_fold_arr;
			lw.gate_fold_identity = f.identity_fold_arr; lw.gate_fold_mult = f.zero_fold_arr; lw.gate_fold_shift = f.zero_fold_arr;
			lw.up_fold_identity = f.identity_fold_arr; lw.up_fold_mult = f.zero_fold_arr; lw.up_fold_shift = f.zero_fold_arr;
			lw.down_fold_identity = f.identity_fold_arr; lw.down_fold_mult = f.zero_fold_arr; lw.down_fold_shift = f.zero_fold_arr;
			lw.q_site_constant = canonical;
			lw.o_site_constant = canonical;
			lw.kv_landing_r_t_k = f.kv_landing_r_t_arr;
			lw.kv_landing_e_t_k = f.kv_landing_e_t_arr;
			lw.kv_landing_r_t_v = f.kv_landing_r_t_arr;
			lw.kv_landing_e_t_v = f.kv_landing_e_t_arr;
			lw.ctx_fold_identity = f.ctx_fold_identity_arr;
			lw.ctx_fold_mult = f.ctx_fold_mult_arr;
			lw.ctx_fold_shift = f.ctx_fold_shift_arr;
			lw.ctx_fold_site_constant = canonical;
			lw.attn_residual_site_constant = canonical;
			lw.iexp_softmax_khead_m = f.iexp_softmax_khead_m_arr;
			lw.iexp_softmax_khead_e = f.iexp_softmax_khead_e_arr;
			lw.mlp_norm_gain = f.norm_gain;
			lw.mlp_norm_site_constant = canonical;
			lw.gate_weight = f.identity2x2;
			lw.up_weight = f.identity2x2;
			lw.down_weight = f.identity2x2;
			lw.gate_site_constant = canonical;
			lw.up_site_constant = canonical;
			lw.mlp_act_site_constant = CarriedScale{INT64_C(1073741824), -96};
			lw.down_site_constant = canonical;
			lw.mlp_residual_site_constant = canonical;
		}
	}
};

// --- Golden constants, execution-derived (see this file's own header
// comment) for OptionGKvLandingFixture's own hidden_codes=[100,-50] input at
// each position. Every intermediate is exactly reproducible: RmsNormSite on
// h=[100,-50], g=[16384,16384], incoming={0,0} (annihilated, RmsNormSite's
// own contract), site_constant={2^30,-30} gives normed=[127,-64],
// normed_scale={m=1358184448, e=0} -- then GemmInt8AccumulateRow against the
// identity 2x2 weight (identity fold) leaves kacc==normed exactly:
// kacc=[127,-64]. r_t = DynamicScaleReciprocal(2^30) = 4294967296. e_t = 0
// (this fixture's own kv_landing_e_t_arr[0]).
inline constexpr int8_t kOptionGFixtureInputH[2] = {100, -50};

// LEGACY order (land at kacc, clamp, THEN RopeApplySite rotates the clamped
// int8 row): LandingRescale(127,...)=161->clamp 127; LandingRescale(-64,...)
// =-81->clamp -81 (in [-127,127], unclamped). At position 1 (45-degree
// table), RopeApplyPair(127,-81,cos=759250125,sin=759250125) -> [81,127]
// (post its own ClampRopeCode; both components already in range).
inline constexpr int8_t kOptionGLegacyK_Pos1[2] = {81, 127};

// FUSED order at position 0 (IDENTITY rotation -- the "null configuration"
// cell): rotated == kacc exactly (no rounding at cos=2^30,sin=0), so fused
// output equals what LEGACY would ALSO compute at position 0's own identity
// table -- LandingRescale(127,...)=161->clamp 127; LandingRescale(-64,...)
// =-81->clamp -81. Fused and legacy MUST agree here; a build that diverges
// at the null configuration is wrong regardless of which order is "correct"
// downstream.
inline constexpr int8_t kOptionGFusedK_Pos0_NullConfiguration[2] = {127, -81};

// FUSED order at position 1 (45-degree rotation, applied to the WIDE
// pre-landing kacc=[127,-64] BEFORE landing): rot_x = round_c3(127*759250125
// - (-64)*759250125, 30) = round_c3(191*759250125, 30) = 135; rot_y =
// round_c3(127*759250125 + (-64)*759250125, 30) = round_c3(63*759250125, 30)
// = 45. Then land ONCE: LandingRescale(135,...)=171->clamp 127;
// LandingRescale(45,...)=57->clamp 57 (in range, unclamped) -- DIVERGES from
// kOptionGLegacyK_Pos1's own second component (57 != 127), execution-
// confirmed (this file's own header comment). This is the discriminating
// fixture the "Mutation vitality on the deleted landing order" cell and the
// non-null half of "Engine/reference bit-parity" both drive.
inline constexpr int8_t kOptionGFusedK_Pos1_NonNullConfiguration[2] = {127, 57};

// --- T-1894 build round 4 (T-1901 Critical 1/D-SLM2416, D-SLM2423-2424): ---
// --- the Selection-dispatch END-TO-END fixture, and the shared fixture ----
// --- the Python composed-path bit-parity cell mirrors (D-SLM2420, D-SLM2428) --
//
// A single-layer, single-(kv-)head, vocab_size=1 model driven through the
// REAL production entry point (SslmModel::Load -> SslmModelView ->
// RunGreedyDecodeLoop), never RunLayerLoop invoked directly with a
// hand-supplied selector -- design Sec12's own "Selection dispatch" cell
// text. `flags` is a constructor parameter, byte-patched into the header
// after BuildArtifact (matching this file's own established
// PutU32(...)/RecomputeIntegrityHash convention), so both a flags=0 and a
// flags=kOptionGFusedKLandingFlag instance of this SAME fixture shape can be
// built and compared.
//
// GOLDEN-CONSTANT DERIVATION (StandardsDocument.md Sec5.4). Every
// composition constant below was obtained by EXECUTING the real Python
// reference's own OFFLINE derivation functions
// (`pipeline._derive_scales`/`_derive_composition_constants`, unmodified by
// this design) against this fixture's own weight/scale inputs (uniform
// per-channel weight scale 1/127, identity 2x2 projections, a constant
// per-site calibration floor standing in for a real corpus pass -- see
// `tests/ci/test_t1899_dynamic_engine_python_reference.py`'s own
// `_minimal_two_head_model` docstring for why), EXCEPT the K/V landing
// reciprocal/exponent, overridden to a finer pair
// (`r_t=DynamicScaleReciprocal(2^30)`, `e_t=-30`, matching `e_a`) after the
// auto-derivation: the auto-derived pair landed both orders at
// magnitude-~2 int8 codes, too coarse to resolve an 8-unit difference in a
// ~135-magnitude rotated accumulator (both orders rounded to the identical
// small integer -- a real, executed false-negative this session found and
// corrected, not a hypothetical one).
//
// Then CROSS-CHECKED by executing `intmath.residual_reconcile` (the
// reference's own K-landing composite, bit-equal to the compiled
// `LandingRescale` by this codebase's own standing claim) and
// `rope.rope_apply_pair` (bit-equal to `RopeApplyPair`) at the derived
// `(m_a, r_t, e_a, e_t)`: landing kacc=[127,-64] unrotated ->
// [127,-82] (both orders agree here -- position 0's own identity rotation);
// the 45-degree FUSED rotation of the SAME kacc -> [135,45] (identical
// rotation this file's own `kOptionGFusedK_Pos1_NonNullConfiguration`
// derivation already used, since kacc and the 45-degree table entry are
// unchanged) -> landed [127,58]; the LEGACY order narrow-rotates the
// ALREADY-LANDED [127,-82] at the same 45-degree entry -> [127,32]
// (unclamped [148,32], clamped). Neither figure was re-derived through the
// C++ side independently this pass: this fixture's own
// `TestOptionGSelectionDispatch_EndToEndProductionPath` cell (test_main.cpp)
// is what proves the compiled engine reproduces [127,32]/[127,58] too,
// which IS the cross-language claim -- there is no separate, independent
// third derivation to cross-check against, matching every other golden
// constant in this file's own convention (the compiled primitive IS the
// oracle; the Python side is the second, independent implementation).
struct OptionGComposedPathFixture {
	static constexpr int32_t kContextCap = 2;
	static constexpr int32_t kVocabSize = 1;
	static constexpr size_t kHiddenSize = 2;

	superslm::SslmModelView view;
	superslm::LayerWeights layers[1];

	int8_t embed_weights[2] = {127, -64};        // vocab_size=1, hidden_size=2
	superslm::CarriedScale embed_site_constant{INT64_C(1090717716), INT64_C(-44)};
	int32_t final_norm_gain_arr[2] = {16384, 16384};
	superslm::CarriedScale final_norm_site_constant{INT64_C(1090717716), INT64_C(-60)};

	int64_t kv_landing_r_t_arr[1] = {INT64_C(4294967296)};
	int64_t kv_landing_e_t_arr[1] = {INT64_C(-30)};
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};
	int32_t norm_gain[2] = {16384, 16384};
	int8_t identity2x2[4] = {1, 0, 0, 1};
	int32_t identity_fold_arr[2] = {1, 1};
	int32_t zero_fold_arr[2] = {0, 0};
	int64_t iexp_softmax_khead_m_arr[1] = {INT64_C(1195669488)};
	int64_t iexp_softmax_khead_e_arr[1] = {INT64_C(-31)};

	OptionGComposedPathFixture(OptionGComposedPathFixture const&) = delete;
	OptionGComposedPathFixture& operator=(OptionGComposedPathFixture const&) = delete;
	OptionGComposedPathFixture(OptionGComposedPathFixture&&) = delete;
	OptionGComposedPathFixture& operator=(OptionGComposedPathFixture&&) = delete;

	// `flags`: 0 for the legacy artifact, `superslm::kOptionGFusedKLandingFlag`
	// for the Option-G one -- the ONE byte this fixture's two instances
	// differ by, matching design Sec12's own "byte-identical except the
	// header flags field" text.
	explicit OptionGComposedPathFixture(uint32_t flags) {
		using superslm::CarriedScale;
		OptionGComposedPathFixture& f = *this;

		Cfg1Spec spec{};
		spec.hidden_size = 2;
		spec.num_hidden_layers = 1;
		spec.num_attention_heads = 1;
		spec.num_key_value_heads = 1;
		spec.head_dim = 2;
		spec.intermediate_size = 2;
		spec.context_cap = OptionGComposedPathFixture::kContextCap;
		spec.kv_precision = 0;  // Int8
		spec.kv_block_size = 1;
		FixtureSection config = MakeSection(superslm::SslmSectionType::Config,
		                                    superslm::SslmDtype::Raw, BuildCfg1(spec));

		const int64_t cos_flat[2] = {INT64_C(1073741824), static_cast<int64_t>(kOptionG45DegQ30)};
		const int64_t sin_flat[2] = {INT64_C(0), static_cast<int64_t>(kOptionG45DegQ30)};
		FixtureSection rope = MakeOptionGRop1SectionMultiRow(
		    /*context_cap=*/OptionGComposedPathFixture::kContextCap, /*pairs=*/1, cos_flat, sin_flat);

		auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope});
		PutU32(built.bytes, 16, flags);
		RecomputeIntegrityHash(built.bytes);
		std::string err;
		const auto status =
		    superslm::SslmModel::Load(built.bytes.data(), built.bytes.size(), f.view, &err);
		if (status != superslm::SslmModelStatus::Ok) {
			std::fprintf(stderr,
			             "OptionGComposedPathFixture's own minimal artifact (flags=%u) failed to "
			             "load: %s (%s)\n",
			             flags, superslm::SslmModelStatusName(status), err.c_str());
			std::abort();
		}

		// The offline composition constants (StandardsDocument.md Sec5.4 --
		// executed derivation, this file's own header comment).
		const CarriedScale c_attn_norm{INT64_C(1090717716), INT64_C(-60)};
		const CarriedScale c_proj{INT64_C(1090717716), INT64_C(-44)};
		const CarriedScale c_ctx{INT64_C(1704246432), INT64_C(-53)};
		const CarriedScale c_mlp_act{INT64_C(1082196484), INT64_C(-52)};
		const CarriedScale c_residual{INT64_C(1082196484), INT64_C(-37)};

		superslm::LayerWeights& lw = f.layers[0];
		lw.attn_norm_gain = f.norm_gain;
		lw.attn_norm_site_constant = c_attn_norm;
		lw.q_weight = f.identity2x2;
		lw.k_weight = f.identity2x2;
		lw.v_weight = f.identity2x2;
		lw.o_weight = f.identity2x2;
		lw.q_fold_identity = f.identity_fold_arr; lw.q_fold_mult = f.zero_fold_arr; lw.q_fold_shift = f.zero_fold_arr;
		lw.k_fold_identity = f.identity_fold_arr; lw.k_fold_mult = f.zero_fold_arr; lw.k_fold_shift = f.zero_fold_arr;
		lw.v_fold_identity = f.identity_fold_arr; lw.v_fold_mult = f.zero_fold_arr; lw.v_fold_shift = f.zero_fold_arr;
		lw.o_fold_identity = f.identity_fold_arr; lw.o_fold_mult = f.zero_fold_arr; lw.o_fold_shift = f.zero_fold_arr;
		lw.gate_fold_identity = f.identity_fold_arr; lw.gate_fold_mult = f.zero_fold_arr; lw.gate_fold_shift = f.zero_fold_arr;
		lw.up_fold_identity = f.identity_fold_arr; lw.up_fold_mult = f.zero_fold_arr; lw.up_fold_shift = f.zero_fold_arr;
		lw.down_fold_identity = f.identity_fold_arr; lw.down_fold_mult = f.zero_fold_arr; lw.down_fold_shift = f.zero_fold_arr;
		lw.q_site_constant = c_proj;
		lw.o_site_constant = c_proj;
		lw.kv_landing_r_t_k = f.kv_landing_r_t_arr;
		lw.kv_landing_e_t_k = f.kv_landing_e_t_arr;
		lw.kv_landing_r_t_v = f.kv_landing_r_t_arr;
		lw.kv_landing_e_t_v = f.kv_landing_e_t_arr;
		lw.ctx_fold_identity = f.ctx_fold_identity_arr;
		lw.ctx_fold_mult = f.ctx_fold_mult_arr;
		lw.ctx_fold_shift = f.ctx_fold_shift_arr;
		lw.ctx_fold_site_constant = c_ctx;
		lw.attn_residual_site_constant = c_residual;
		lw.iexp_softmax_khead_m = f.iexp_softmax_khead_m_arr;
		lw.iexp_softmax_khead_e = f.iexp_softmax_khead_e_arr;
		lw.mlp_norm_gain = f.norm_gain;
		lw.mlp_norm_site_constant = c_attn_norm;
		lw.gate_weight = f.identity2x2;
		lw.up_weight = f.identity2x2;
		lw.down_weight = f.identity2x2;
		lw.gate_site_constant = c_proj;
		lw.up_site_constant = c_proj;
		lw.mlp_act_site_constant = c_mlp_act;
		lw.down_site_constant = c_proj;
		lw.mlp_residual_site_constant = c_residual;
	}
};

// Golden K-store values for `OptionGComposedPathFixture` at position 1 (the
// 45-degree table row), execution-derived (this struct's own header
// comment): `kOptionGComposedLegacyK` is the LEGACY order's own FINAL stored
// value -- landed unrotated, THEN narrow-rotated via `RopeApplySite` and
// written back (the real production write-back this fixture's own decode
// exercises), NOT the transient unrotated-landing value ([127,-82]) that
// exists only before the write-back overwrites it.
// `kOptionGComposedFusedK_Pos1` is the FUSED order's own value: the wide
// kacc rotated first, landed once. Identical to each other at position 0
// (identity rotation, not asserted by name here -- both orders compute
// [127,-82] there, confirmed by `TestOptionGSelectionDispatch_EndToEndProductionPath`'s
// own flags=0/flags=1 comparison at position 1 only, per the design's own
// "at null AND non-null configuration" text being carried by the
// mutation-vitality cell's own null-configuration assertion elsewhere in
// this file).
inline constexpr int8_t kOptionGComposedLegacyK[2] = {127, 32};
inline constexpr int8_t kOptionGComposedFusedK_Pos1[2] = {127, 58};

}  // namespace superslm_test

#endif  // SSLM_T1899_OPTIONG_FIXTURES_H
