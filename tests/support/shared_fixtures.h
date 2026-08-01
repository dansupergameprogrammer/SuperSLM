// Fixtures shared across a candidate test area boundary (T-1574 test suite
// split, Stage 0 / §2, §3, §4 of the plan of record). Owned by neither
// consuming area; every area file that needs one of these #includes this
// header rather than redefining it.
//
// TwoLayerFixture -- moved verbatim from tests/test_main.cpp:12835 -- is used
// by candidate area #12 (forward_mlp_residual.cpp) directly and by candidate
// area #13 (forward_decode.cpp) as a struct member and via two static-const
// reads.
//
// ChainTraceSinkRecord + ChainTraceSinkHookFn -- moved verbatim from
// tests/test_main.cpp:8411/:8424 -- are used by candidate area #6
// (checked_chain_funnel.cpp) and again, composed with
// superslm::RunLayerLoop, by a join test that files under candidate area #12
// per the plan's composing-function rule (§4).
//
// MakeRop1SectionMultiRow -- moved verbatim from tests/test_main.cpp, and
// promoted from `static` to `inline` so every tests/test_main.cpp call site
// that still names it directly continues to resolve it through this header
// -- is TwoLayerFixture's own constructor's dependency (T-1632 Minor 4: this
// promotion was previously undocumented here).
//
// No production or fixture *behavior* changes here: every struct/function
// body below is relocated exactly as it read at tests/test_main.cpp@9fc75b0.
#pragma once

#include "superslm/artifact.h"
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/model.h"
#include "superslm/trace_hook.h"
#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_fixtures.h"
#include "sslm_s3_2_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

#include "test_harness.h"

#include <cstdint>
#include <string>
#include <vector>

// T-1632 Minor 2: this header previously declared `using namespace superslm;
// using namespace superslm_test;` at namespace scope here, to preserve the
// name resolution ChainTraceSinkRecord/ChainTraceSinkHookFn had at their
// original site in tests/test_main.cpp (which opens both namespaces itself).
// That leaked both directives into every current and future includer
// whether or not it opens them itself, with no way to opt out. Every
// declaration below already names its superslm/superslm_test types fully
// qualified (ChainTraceSinkHookFn's parameters, MakeRop1SectionMultiRow's
// return type, TwoLayerFixture's members), and the two functions that use
// unqualified names from those namespaces internally (MakeRop1SectionMultiRow,
// TwoLayerFixture's constructor) already scope their own
// `using namespace superslm_test;` to the function body -- so this header
// needs no namespace-scope using-directive of its own, and both current
// includers (tests/test_main.cpp, tests/areas/proof_manifest.cpp) already
// declare the same two directives themselves at file scope.

struct ChainTraceSinkRecord {
	std::string site;
	size_t token_index = 0;
	std::vector<int64_t> x_int;
	int64_t d_prime = 0;
	int64_t dn = 0;
	int32_t s = 0;
	int64_t r = 0;
	std::vector<int8_t> codes;
	int64_t m_out = 0;
	int64_t e_out = 0;
};

void ChainTraceSinkHookFn(const superslm::SslmChainTraceRecord* chain,
                           const superslm::SslmKvLandingTraceRecord* kv, void* user);

// Sec13.1 cell 9's sibling join for the RoPE table itself: the real, pinned
// cos/sin rows the site will eventually read are carried byte-for-byte
// through the real `SslmModel::Load`, at every row a small (context_cap=4)
// artifact declares -- proving the artifact-carries-the-real-table half of
// the site's own join independently of the site itself. Moved verbatim from
// tests/test_main.cpp (T-1574 Stage 0) -- TwoLayerFixture's constructor below
// calls it, and it stays `inline` (was `static`) so every remaining
// tests/test_main.cpp call site continues to resolve it through this header.
inline superslm_test::FixtureSection MakeRop1SectionMultiRow(int32_t context_cap, int32_t pairs,
                                                               const int64_t* cos_flat, const int64_t* sin_flat) {
	using namespace superslm_test;
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
	return MakeSection(superslm::SslmSectionType::RopeTables, superslm::SslmDtype::Int64, manifest.bytes,
	                    /*alignment=*/64);
}

struct TwoLayerFixture {
	// S3.7: the fixture's own CFG1 `context_cap` and ROP1 row count (T-1571,
	// the multi-position bump below). Not the `context_cap` any individual
	// call site passes to RunLayerLoop/RunGreedyDecodeLoop -- those remain
	// each call's own explicit argument.
	static constexpr int32_t kContextCap = 16;

	superslm::SslmModelView view;  // owns the backing store rope_tables points into
	superslm::LayerWeights layers[2];
	// Backing arrays LayerWeights points into -- kept alive for the fixture's
	// own lifetime.
	// T-1374/Significant 4: this fixture's own geometry has exactly one head
	// (num_attention_heads=1), so the per-head arrays below have one entry
	// each -- the same value every element used before this struct gained a
	// head axis, replicated across K and V since this fixture does not
	// distinguish them.
	int64_t kv_landing_r_t_arr[1];
	int64_t kv_landing_e_t_arr[1] = {0};
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};
	// T-1356: a Q16 gain of 1.0 (65536) is OUT of C29's chain-input domain at
	// this composition, by construction rather than by bad luck. RmsNormSite
	// forms `FloorDivI64(h[i] << 2*NORM_FRAC_BITS, root) * g[i]`, and for a row
	// whose elements are all at the RMS the first factor is ~2^16, so a unit
	// Q16 gain lands the wide row at exactly 2^32 -- twice C29's `d_prime >
	// 2^31` ceiling. Executed at this fixture's own {5,-5}: root == 327680,
	// wide == +/-4294967296 at gain 65536, +/-2147483648 at 32768 (the ceiling
	// itself), +/-1073741824 at 16384. The declaring pass could not have seen
	// this -- the stub returned before any funnel call, so nothing in the suite
	// ever drove the fixture through C29. Same class as D-SLM426, where a
	// latent fixture defect stayed silent until a real body first ran against
	// it. 16384 (Q16 0.25) is chosen over 32768 to sit clear of the boundary
	// rather than exactly on it.
	int32_t norm_gain[2] = {16384, 16384};
	int8_t identity2x2[4] = {1, 0, 0, 1};   // [[1,0],[0,1]] row-major [out,in]

	// Default-constructs in place, wiring every LayerWeights pointer field to
	// point at THIS object's own sibling array members (`norm_gain`,
	// `identity2x2`, etc). This used to be a `static Build()` factory that
	// built a separate local `f`, wired the pointers to `f`'s own members, and
	// `return f;`d it -- a named-local return that only stays valid if NRVO
	// elides the copy, which C++ never guarantees (D-SLM487). Doing the wiring
	// in the constructor body instead means it runs directly on `*this` at
	// its FINAL address (the caller's own local, `TwoLayerFixture fixture;`)
	// -- there is no intermediate object for the pointers to dangle from, so
	// the property does not depend on an optimization the compiler is free to
	// skip. Confirmed by direct address trace, not by construction alone: see
	// this pass's `TestTwoLayerFixtureSelfPointersSurviveConstructionByAddress`
	// below, which prints `&identity2x2` against `layers[*].q_weight` and
	// requires them equal -- the same measurement D-SLM487 used to prove the
	// old shape broken, now asserted permanently as a regression guard.
	TwoLayerFixture() {
		using namespace superslm_test;
		using superslm::CarriedScale;

		TwoLayerFixture& f = *this;
		Cfg1Spec spec{};
		spec.hidden_size = 2;
		spec.num_hidden_layers = 2;
		spec.num_attention_heads = 1;
		spec.num_key_value_heads = 1;
		spec.head_dim = 2;
		spec.intermediate_size = 2;
		// S3.7: bumped from 1 to kTwoLayerFixtureContextCap (16) so this
		// fixture's own ROP1 table carries enough rows for a multi-position
		// caller to drive `context_length` past 0 -- every existing call site
		// that passes RunLayerLoop its own explicit `context_cap=1` literal is
		// unaffected (CheckPositionOverCap/RopeApplySite bound `position`
		// against the CALLER's own `context_cap` argument, never CFG1's, so a
		// bigger table changes nothing for a caller that only ever asks for
		// position 0). The K/V workspace size is likewise driven by each
		// call's own `context_cap` argument, not this fixture's CFG1 value.
		spec.context_cap = TwoLayerFixture::kContextCap;
		spec.kv_precision = 0;  // Int8
		spec.kv_block_size = 1;
		FixtureSection config =
		    MakeSection(superslm::SslmSectionType::Config, superslm::SslmDtype::Raw, BuildCfg1(spec));
		// Identity rotation (cos(0)==2^30, sin==0) at every one of the
		// fixture's `kContextCap` rows -- every existing cell in this suite
		// that reads this fixture's rope table still sees identity at
		// position 0, unchanged from before this bump.
		std::vector<int64_t> cos_flat(TwoLayerFixture::kContextCap, INT64_C(1073741824));
		std::vector<int64_t> sin_flat(TwoLayerFixture::kContextCap, INT64_C(0));
		FixtureSection rope = MakeRop1SectionMultiRow(/*context_cap=*/TwoLayerFixture::kContextCap,
		                                               /*pairs=*/1, cos_flat.data(), sin_flat.data());
		auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope});
		std::string err;
		const auto status = superslm::SslmModel::Load(built.bytes.data(), built.bytes.size(), f.view, &err);
		CHECK_MSG(status == superslm::SslmModelStatus::Ok,
		          "TwoLayerFixture's own minimal artifact failed to load: got %s (%s)",
		          superslm::SslmModelStatusName(status), err.c_str());

		const CarriedScale canonical{/*m=*/INT64_C(1073741824), /*e=*/-30};
		const int64_t r_t = superslm::DynamicScaleReciprocal(canonical.m);
		f.kv_landing_r_t_arr[0] = r_t;
		for (int l = 0; l < 2; ++l) {
			superslm::LayerWeights& lw = f.layers[l];
			lw.attn_norm_gain = f.norm_gain;
			lw.attn_norm_site_constant = canonical;
			lw.q_weight = f.identity2x2;
			lw.k_weight = f.identity2x2;
			lw.v_weight = f.identity2x2;
			lw.o_weight = f.identity2x2;
			lw.proj_identity = 1;
			lw.proj_mult = 0;
			lw.proj_shift = 0;
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
			// A domain-checked, non-degenerate C30 triple (e=-52 against this
			// fixture's own canonical m=2^30-adjacent mantissa) -- derived by
			// direct execution of the vendored reference's iexp_scale_constants,
			// never hand re-derived (Curie's own discipline, dimension 10):
			// python -c "import sys; sys.path.insert(0,'tests/reference/
			// superslm_spike'); sys.path.insert(0,'tests/reference'); import
			// intmath as im; QFMT=30; LN2_Q=int(im._LN2*(1<<QFMT));
			// B_Q=int(im._POLY_B*(1<<QFMT)); CA_Q=int((im._POLY_C/im._POLY_A)*(1<<QFMT));
			// print(im.iexp_scale_constants(1500000000,-52,LN2_Q,QFMT,B_Q,QFMT,CA_Q,QFMT))"
			// -> (2081104, 4062246, 8649804928567); M=q_b^2+q_c == 25151647493083,
			// under kSoftmaxRowMaxSafeExponent (2^47 == 140737488355328).
			lw.q_ln2 = INT64_C(2081104);
			lw.q_b_iexp = INT64_C(4062246);
			lw.q_c_iexp = INT64_C(8649804928567);
			lw.mlp_norm_gain = f.norm_gain;
			lw.mlp_norm_site_constant = canonical;
			lw.gate_weight = f.identity2x2;
			lw.up_weight = f.identity2x2;
			lw.down_weight = f.identity2x2;
			// T-1355: gate_proj/up_proj each funnel in their own right and so
			// each carries its own site constant, on the same `canonical`
			// pattern every other site constant in this fixture uses.
			lw.gate_site_constant = canonical;
			lw.up_site_constant = canonical;
			// T-1356: `canonical` here would compound into the SAME
			// chain-input rejection D-SLM434 already found one site later
			// (the whole finding this fixture pass closes). Derived by
			// execution, not tuned: MlpActSite's own funnel call folds THREE
			// factors left-associated (gate_scale, up_scale, this site
			// constant -- forward_sites.cpp:504), and the trace hook
			// (SslmSetTraceHook + ChainTraceSinkHookFn, this suite's own
			// S3.1a instrumentation) reads the real per-site (dn, s, m_out,
			// e_out) with no re-implementation needed.
			//   1. With this field at `canonical` (e=-30), a trace-hooked
			//      run of RunLayerLoop(budget=1) on this fixture's layer 0
			//      gives mlp_act e_out=72 (executed) -- gate/up both e=6, so
			//      the three-factor fold plus this site's own D'=528515072
			//      (~127*2^15*127, MlpActSite's own documented Q15 product
			//      bound) drives the OUTPUT nearly 66 exponent-bits above
			//      its INPUTS. Downstream this composes through down_proj
			//      (e_out=79) into mlp_residual, whose wide row then reads
			//      exactly the values Poirot's review recorded --
			//      branch=(2064898088,79), stream=(1140850688,6) -- and
			//      rejects C29 (`d_prime > 2^31`), reproducing D-SLM434's
			//      "each correction moves rejection one site later" exactly.
			//   2. The +66 drift lives entirely in this ONE site constant's
			//      own `e`, because CombineCarriedScale's exponent output is
			//      `a.e + b.e + 31` (checked_chain_funnel.cpp) at each fold
			//      step, and this constant's own `m` stays at the same
			//      canonical mantissa (2^30) every other site constant in
			//      this tree uses -- so shifting only its `e` shifts
			//      mlp_act's own output `e` by the identical amount, with no
			//      other combine step touched. Re-running the same
			//      trace-hooked probe at e=-96 (30+66 below canonical)
			//      gives mlp_act e_out=6 exactly (executed) -- landing on
			//      attn_residual's own carried-scale exponent for this same
			//      fixture, which keeps down_proj (e_out=13) and
			//      mlp_residual (e_out=20, wide-row d_prime=29550, far
			//      inside C29's 2^31 ceiling) in domain at every enumerated
			//      budget and every resume/poison cell (§8 below).
			// Reproduction: build this commit, install the trace hook on a
			// RunLayerLoop(budget=1) call over this fixture's layer 0, and
			// diff mlp_act's own e_out between `canonical` and this value --
			// the delta is exactly -66, matching the shift applied here.
			// C34's own domain check (CheckSiluCompositionScaleDomain) is
			// unaffected: it examines gate_scale (the INPUT), never this
			// site constant, so no upstream cell's C34 coverage changes.
			lw.mlp_act_site_constant = CarriedScale{INT64_C(1073741824), INT64_C(-96)};
			lw.down_site_constant = canonical;
			lw.mlp_residual_site_constant = canonical;
		}
	}

	// D-SLM492/T-1407: every `LayerWeights` pointer field wired above points at
	// THIS object's own sibling array members. A copy or a move would produce a
	// second object whose `LayerWeights` still point at the ORIGINAL object's
	// arrays (a shallow-copied pointer, not a deep one) -- harmless only for as
	// long as the original outlives the copy, and silently dangling the moment
	// it does not. The constructor-body wiring (T-1403) makes direct
	// construction safe by removing the return-by-value NRVO dependency; it
	// does nothing to stop a FUTURE caller from relocating an already-built
	// fixture (a helper that builds one and returns it by value, a
	// `std::vector<TwoLayerFixture>`, an assignment). Deleting copy and move
	// converts that reintroduction from a silent dangling pointer into
	// `error C2280` at compile time -- proven: a hand-built two-return-path
	// factory returning `TwoLayerFixture` by value (the exact shape D-SLM487
	// found broken) fails to compile with copy/move deleted, where it
	// previously compiled and passed every existing check while corrupting the
	// four bytes `TestTwoLayerFixtureSelfPointersSurviveConstructionByAddress`
	// (below) reads back. No existing call site constructs this fixture by
	// relocation (independently swept, §4 of the review this repairs), so
	// nothing that exists today needs the move.
	TwoLayerFixture(const TwoLayerFixture&) = delete;
	TwoLayerFixture& operator=(const TwoLayerFixture&) = delete;
	TwoLayerFixture(TwoLayerFixture&&) = delete;
	TwoLayerFixture& operator=(TwoLayerFixture&&) = delete;
};
