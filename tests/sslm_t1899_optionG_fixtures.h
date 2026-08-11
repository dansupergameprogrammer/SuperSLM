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

}  // namespace superslm_test

#endif  // SSLM_T1899_OPTIONG_FIXTURES_H
