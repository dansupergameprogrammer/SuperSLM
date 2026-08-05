// T-1420 adversary probe -- DISPOSABLE. Not product code, not a suite test.
//
// Attacks the T-1409 promise's universal clause: "a change to any load-bearing
// layer weight changes what the model decodes."
//
// Two axes the shipped experiments/decode_discrimination.cpp mutation matrix
// does not sweep:
//
//   AXIS 1 -- the per-layer, artifact-carried SCALE operands. WSC1's
//   `layer{L}.weight_scale` fold triple (proj_identity/proj_mult/proj_shift,
//   C24) and WSC1's `layer{L}.ctx_fold` triple (ctx_fold_identity/mult/shift,
//   C27/D-SLM57), plus KVC1's per-head landing scale (kv_landing_r_t/e_t,
//   C25/S8.1). Each is a per-layer weight the forward reads; the shipped
//   matrix sweeps only the seven int8 projection tensors, the norm gains,
//   embed and head.
//
//   AXIS 2 -- the DECODE half of "what the model decodes". The shipped matrix
//   calls a mutation a MOVE when status, produced count, token vector OR
//   int32 logit-row vector differ. This probe reports the token vector and
//   the logit rows separately, over the same 56 weight elements.
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/model.h"
#include "superslm/trace_hook.h"

#include "sslm_fixtures.h"
#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

namespace {

using superslm::CarriedScale;
using superslm::LayerWeights;
using superslm::SslmForwardStatus;

superslm_test::FixtureSection MakeRopeSection(int32_t context_cap, int32_t pairs,
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

// Same geometry as SolverDecodeFixture, with one correction the strike needs:
// the per-layer scale operands are PER-LAYER arrays here. The shipped fixture
// aliases one ctx_fold triple and one kv_landing pair across BOTH layers, so a
// layer-0 mutation of them is indistinguishable from a layer-1 one.
struct StrikeFixture {
	superslm::SslmModelView view;
	LayerWeights layers[2];

	int64_t kv_landing_r_t_arr[2][1];
	int64_t kv_landing_e_t_arr[2][1] = {{0}, {0}};
	int64_t kv_landing_m_target_arr[2][1] = {{INT64_C(1073741824)}, {INT64_C(1073741824)}};
	int64_t kv_landing_e_target_arr[2][1] = {{0}, {0}};
	int32_t ctx_fold_identity_arr[2][1] = {{1}, {1}};
	int32_t ctx_fold_mult_arr[2][1] = {{0}, {0}};
	int32_t ctx_fold_shift_arr[2][1] = {{0}, {0}};
	int32_t proj_identity[2] = {1, 1};
	int32_t proj_mult[2] = {0, 0};
	int32_t proj_shift[2] = {0, 0};

	int8_t w[2][7][4];
	int32_t attn_gain[2][2];
	int32_t mlp_gain[2][2];
	int32_t final_gain[2];

	static constexpr int32_t kVocabSize = 3;
	int8_t embed_weights[6];
	int8_t head_weights[6];
	CarriedScale embed_site_constant{INT64_C(1073741824), INT64_C(0)};
	CarriedScale final_norm_site_constant{INT64_C(1073741824), INT64_C(-30)};

	StrikeFixture(const int8_t weights[2][7][4], const int8_t embed[6], const int8_t head[6],
	              int32_t gain_q16) {
		using namespace superslm_test;
		std::memcpy(w, weights, sizeof(w));
		std::memcpy(embed_weights, embed, 6);
		std::memcpy(head_weights, head, 6);
		for (int l = 0; l < 2; ++l) {
			attn_gain[l][0] = attn_gain[l][1] = gain_q16;
			mlp_gain[l][0] = mlp_gain[l][1] = gain_q16;
		}
		final_gain[0] = final_gain[1] = gain_q16;

		Cfg1Spec spec{};
		spec.hidden_size = 2;
		spec.num_hidden_layers = 2;
		spec.num_attention_heads = 1;
		spec.num_key_value_heads = 1;
		spec.head_dim = 2;
		spec.intermediate_size = 2;
		spec.context_cap = 1;
		spec.kv_precision = 0;
		spec.kv_block_size = 1;
		FixtureSection config =
		    MakeSection(superslm::SslmSectionType::Config, superslm::SslmDtype::Raw, BuildCfg1(spec));
		const int64_t cos_flat[1] = {INT64_C(1073741824)};
		const int64_t sin_flat[1] = {0};
		FixtureSection rope = MakeRopeSection(1, 1, cos_flat, sin_flat);
		auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope});
		std::string err;
		const auto status =
		    superslm::SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		if (status != superslm::SslmModelStatus::Ok) {
			std::fprintf(stderr, "FATAL: fixture artifact failed to load: %s (%s)\n",
			             superslm::SslmModelStatusName(status), err.c_str());
			std::abort();
		}

		const CarriedScale canonical{INT64_C(1073741824), INT64_C(-30)};
		const int64_t o_e_boost[2] = {0, 20};
		const int64_t down_e_boost[2] = {0, 12};
		for (int l = 0; l < 2; ++l) {
			kv_landing_r_t_arr[l][0] = superslm::DynamicScaleReciprocal(canonical.m);
			LayerWeights& lw = layers[l];
			lw.attn_norm_gain = attn_gain[l];
			lw.attn_norm_site_constant = canonical;
			lw.q_weight = w[l][0];
			lw.k_weight = w[l][1];
			lw.v_weight = w[l][2];
			lw.o_weight = w[l][3];
			lw.proj_identity = proj_identity[l];
			lw.proj_mult = proj_mult[l];
			lw.proj_shift = proj_shift[l];
			lw.q_site_constant = canonical;
			lw.o_site_constant = CarriedScale{canonical.m, canonical.e + o_e_boost[l]};
			lw.kv_landing_r_t_k = kv_landing_r_t_arr[l];
			lw.kv_landing_e_t_k = kv_landing_e_t_arr[l];
			lw.kv_landing_r_t_v = kv_landing_r_t_arr[l];
			lw.kv_landing_e_t_v = kv_landing_e_t_arr[l];
			lw.kv_landing_m_target_k = kv_landing_m_target_arr[l];
			lw.kv_landing_e_target_k = kv_landing_e_target_arr[l];
			lw.kv_landing_m_target_v = kv_landing_m_target_arr[l];
			lw.kv_landing_e_target_v = kv_landing_e_target_arr[l];
			lw.ctx_fold_identity = ctx_fold_identity_arr[l];
			lw.ctx_fold_mult = ctx_fold_mult_arr[l];
			lw.ctx_fold_shift = ctx_fold_shift_arr[l];
			lw.ctx_fold_site_constant = canonical;
			lw.attn_residual_site_constant = canonical;
			lw.q_ln2 = INT64_C(2081104);
			lw.q_b_iexp = INT64_C(4062246);
			lw.q_c_iexp = INT64_C(8649804928567);
			lw.mlp_norm_gain = mlp_gain[l];
			lw.mlp_norm_site_constant = canonical;
			lw.gate_weight = w[l][4];
			lw.up_weight = w[l][5];
			lw.down_weight = w[l][6];
			lw.gate_site_constant = canonical;
			lw.up_site_constant = canonical;
			lw.mlp_act_site_constant = CarriedScale{INT64_C(1073741824), INT64_C(-96)};
			lw.down_site_constant = CarriedScale{canonical.m, canonical.e + down_e_boost[l]};
			lw.mlp_residual_site_constant = canonical;
		}
	}
	// Re-publish the per-layer scale operands into LayerWeights after a knob
	// is turned (LayerWeights holds proj_* by value, the arrays by pointer).
	void Republish() {
		for (int l = 0; l < 2; ++l) {
			layers[l].proj_identity = proj_identity[l];
			layers[l].proj_mult = proj_mult[l];
			layers[l].proj_shift = proj_shift[l];
		}
	}
	StrikeFixture(const StrikeFixture&) = delete;
	StrikeFixture& operator=(const StrikeFixture&) = delete;
};

struct DecodeResult {
	SslmForwardStatus status = SslmForwardStatus::Ok;
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;
	size_t produced = 0;

	bool SameTokensAs(const DecodeResult& o) const {
		return status == o.status && produced == o.produced && tokens == o.tokens;
	}
	bool SameLogitsAs(const DecodeResult& o) const {
		return status == o.status && produced == o.produced && logit_rows == o.logit_rows;
	}
};

DecodeResult RunDecode(StrikeFixture& f, const std::vector<int32_t>& prompt, size_t max_new) {
	DecodeResult r;
	int8_t hidden_codes[2] = {0, 0};
	superslm::SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes;
	uint8_t workspace[64];
	std::memset(workspace, 0xEE, sizeof(workspace));
	superslm::SslmDecodeStopReason stop{};
	r.tokens.assign(max_new, INT32_C(-99));
	r.logit_rows.assign(max_new * StrikeFixture::kVocabSize, INT32_C(-99));
	r.status = superslm::RunGreedyDecodeLoop(
	    seq, f.layers, 2, 2, 2, 2, 1, f.view.rope_tables, prompt.data(), prompt.size(),
	    f.embed_weights, f.embed_site_constant, f.final_gain, f.final_norm_site_constant,
	    f.head_weights, StrikeFixture::kVocabSize, nullptr, 0, max_new, workspace,
	    sizeof(workspace), r.tokens.data(), r.logit_rows.data(), r.tokens.size(), &r.produced,
	    &stop);
	return r;
}

void CandidateWeights(int8_t w[2][7][4]) {
	static const int8_t q0[4] = {1, 0, 0, 1}, k0[4] = {1, 0, 0, 1};
	static const int8_t v0[4] = {1, 1, 0, 1};
	static const int8_t o0[4] = {1, 0, 1, 1};
	static const int8_t g0[4] = {1, 1, 1, 0};
	static const int8_t u0[4] = {2, 1, 1, 1};
	static const int8_t d0[4] = {1, 1, 0, 1};
	static const int8_t q1[4] = {1, 0, 0, 1}, k1[4] = {1, 0, 0, 1};
	static const int8_t v1[4] = {0, 1, 1, 0};
	static const int8_t o1[4] = {1, 1, 1, 0};
	static const int8_t g1[4] = {1, 0, 1, 1};
	static const int8_t u1[4] = {1, 1, 0, 2};
	static const int8_t d1[4] = {2, 1, 1, 1};
	const int8_t* l0[7] = {q0, k0, v0, o0, g0, u0, d0};
	const int8_t* l1[7] = {q1, k1, v1, o1, g1, u1, d1};
	for (int t = 0; t < 7; ++t) {
		std::memcpy(w[0][t], l0[t], 4);
		std::memcpy(w[1][t], l1[t], 4);
	}
}

const int8_t kEmbed[6] = {5, -3, 7, 2, -4, 6};
const int8_t kHead[6] = {9, -2, -3, 8, 5, 5};
const char* kTensorName[7] = {"q", "k", "v", "o", "gate", "up", "down"};

DecodeResult g_base[3];

// Runs the three prompts against `f` and classifies against the baseline.
// Returns a two-character verdict: token movement, logit movement.
struct Verdict {
	bool tokens_moved = false;
	bool logits_moved = false;
	bool status_changed = false;
};
Verdict Classify(StrikeFixture& f) {
	Verdict v;
	for (int32_t p = 0; p < 3; ++p) {
		DecodeResult r = RunDecode(f, {p}, 3);
		if (r.status != g_base[p].status) v.status_changed = true;
		if (!r.SameTokensAs(g_base[p])) v.tokens_moved = true;
		if (!r.SameLogitsAs(g_base[p])) v.logits_moved = true;
	}
	return v;
}
const char* Label(const Verdict& v) {
	if (v.status_changed) return "STATUS-CHANGE";
	if (v.tokens_moved) return "TOKENS-MOVE";
	if (v.logits_moved) return "LOGITS-ONLY";
	return "SILENT";
}

}  // namespace

// Dumps the K/V-landing records' `codes` spans, so the claim "the score GEMM
// reading v_store instead of k_store is a NON-equivalent mutant" is verified by
// execution rather than by construction (StandardsDocument 5.4).
namespace {
void DumpKvCodesHook(const superslm::SslmChainTraceRecord*,
                     const superslm::SslmKvLandingTraceRecord* kv, void*) {
	if (kv == nullptr) return;
	std::printf("  tok=%zu %-24s head=%u codes=[", kv->token_index,
	            std::string(kv->site).c_str(), kv->head);
	for (size_t i = 0; i < kv->codes.size(); ++i)
		std::printf("%s%d", i ? "," : "", static_cast<int>(kv->codes[i]));
	std::printf("]\n");
}
}  // namespace

int main() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	StrikeFixture f(wc, kEmbed, kHead, 16384);
	{
		std::printf("=== K vs V store codes (prompt {0}, 3 new tokens) ===\n");
		superslm::SslmTraceHookState hs{};
		superslm::SslmSetTraceHook(hs, &DumpKvCodesHook, nullptr);
		int8_t hidden_codes[2] = {0, 0};
		superslm::SequenceLayerState seq{};
		seq.hidden_codes = hidden_codes;
		uint8_t workspace[64];
		std::memset(workspace, 0xEE, sizeof(workspace));
		superslm::SslmDecodeStopReason stop{};
		std::vector<int32_t> toks(3, -99), rows(9, -99);
		size_t produced = 0;
		const int32_t prompt[1] = {0};
		superslm::RunGreedyDecodeLoop(seq, f.layers, 2, 2, 2, 2, 1, f.view.rope_tables, prompt, 1,
		                              f.embed_weights, f.embed_site_constant, f.final_gain,
		                              f.final_norm_site_constant, f.head_weights,
		                              StrikeFixture::kVocabSize, nullptr, 0, 3, workspace,
		                              sizeof(workspace), toks.data(), rows.data(), toks.size(),
		                              &produced, &stop, /*site_prefix=*/{}, &hs);
		std::printf("\n");
	}
	for (int32_t p = 0; p < 3; ++p) {
		g_base[p] = RunDecode(f, {p}, 3);
		if (g_base[p].status != SslmForwardStatus::Ok) {
			std::printf("BASELINE NOT OK prompt=%d\n", p);
			return 1;
		}
	}
	std::printf("baseline tokens: p0=[%d,%d,%d] p1=[%d,%d,%d] p2=[%d,%d,%d]\n",
	            g_base[0].tokens[0], g_base[0].tokens[1], g_base[0].tokens[2], g_base[1].tokens[0],
	            g_base[1].tokens[1], g_base[1].tokens[2], g_base[2].tokens[0], g_base[2].tokens[1],
	            g_base[2].tokens[2]);

	// ---- AXIS 1: the per-layer artifact-carried scale operands -------------
	std::printf("\n=== AXIS 1: per-layer scale operands (WSC1 weight_scale, WSC1 ctx_fold, KVC1 landing) ===\n");
	for (int l = 0; l < 2; ++l) {
		// C24 near-identity control: identity=0 with (mult, shift) = (2^30, 30),
		// the exact "near-identity" form SuperSLM_Plan.md:2194-2197 names.
		{
			const int32_t oi = f.proj_identity[l], om = f.proj_mult[l], os = f.proj_shift[l];
			f.proj_identity[l] = 0;
			f.proj_mult[l] = INT32_C(1073741824);
			f.proj_shift[l] = 30;
			f.Republish();
			Verdict v = Classify(f);
			std::printf("layer%d.weight_scale identity->near-identity(2^30,30): %s\n", l, Label(v));
			f.proj_identity[l] = oi;
			f.proj_mult[l] = om;
			f.proj_shift[l] = os;
			f.Republish();
		}
		// A genuine, large weight_scale change: x2 and x0.5.
		for (int32_t sh : {29, 31}) {
			const int32_t oi = f.proj_identity[l], om = f.proj_mult[l], os = f.proj_shift[l];
			f.proj_identity[l] = 0;
			f.proj_mult[l] = INT32_C(1073741824);
			f.proj_shift[l] = sh;
			f.Republish();
			Verdict v = Classify(f);
			std::printf("layer%d.weight_scale identity->(2^30,shift=%d) [x%s]: %s\n", l, sh,
			            sh == 29 ? "2" : "0.5", Label(v));
			f.proj_identity[l] = oi;
			f.proj_mult[l] = om;
			f.proj_shift[l] = os;
			f.Republish();
		}
		// C27 ctx_fold near-identity control, same form.
		{
			f.ctx_fold_identity_arr[l][0] = 0;
			f.ctx_fold_mult_arr[l][0] = INT32_C(1073741824);
			f.ctx_fold_shift_arr[l][0] = 30;
			Verdict v = Classify(f);
			std::printf("layer%d.ctx_fold  identity->near-identity(2^30,30): %s\n", l, Label(v));
			f.ctx_fold_identity_arr[l][0] = 1;
			f.ctx_fold_mult_arr[l][0] = 0;
			f.ctx_fold_shift_arr[l][0] = 0;
		}
		for (int32_t sh : {29, 31}) {
			f.ctx_fold_identity_arr[l][0] = 0;
			f.ctx_fold_mult_arr[l][0] = INT32_C(1073741824);
			f.ctx_fold_shift_arr[l][0] = sh;
			Verdict v = Classify(f);
			std::printf("layer%d.ctx_fold  identity->(2^30,shift=%d) [x%s]: %s\n", l, sh,
			            sh == 29 ? "2" : "0.5", Label(v));
			f.ctx_fold_identity_arr[l][0] = 1;
			f.ctx_fold_mult_arr[l][0] = 0;
			f.ctx_fold_shift_arr[l][0] = 0;
		}
		// KVC1 landing: the per-head target exponent, +/-1.
		for (int64_t d : {INT64_C(1), INT64_C(-1)}) {
			f.kv_landing_e_t_arr[l][0] = d;
			Verdict v = Classify(f);
			std::printf("layer%d.kv_landing_e_t %+" PRId64 ": %s\n", l, d, Label(v));
			f.kv_landing_e_t_arr[l][0] = 0;
		}
		// KVC1 landing reciprocal, +/-1 ulp.
		for (int64_t d : {INT64_C(1), INT64_C(-1)}) {
			const int64_t orig = f.kv_landing_r_t_arr[l][0];
			f.kv_landing_r_t_arr[l][0] = orig + d;
			Verdict v = Classify(f);
			std::printf("layer%d.kv_landing_r_t %+" PRId64 " ulp: %s\n", l, d, Label(v));
			f.kv_landing_r_t_arr[l][0] = orig;
		}
	}

	// ---- AXIS 2: tokens vs logits over the 56 shipped weight elements ------
	std::printf("\n=== AXIS 2: the 56 weight elements the shipped matrix calls MOVES/SILENT ===\n");
	int tok_moves = 0, logit_only = 0, silent = 0, status_change = 0;
	for (int l = 0; l < 2; ++l) {
		for (int t = 0; t < 7; ++t) {
			for (int e = 0; e < 4; ++e) {
				Verdict best;
				for (int8_t delta : {int8_t{1}, int8_t{-1}, int8_t{2}, int8_t{-2}}) {
					const int8_t orig = f.w[l][t][e];
					f.w[l][t][e] = static_cast<int8_t>(orig + delta);
					Verdict v = Classify(f);
					f.w[l][t][e] = orig;
					if (v.tokens_moved) { best = v; break; }
					if (v.logits_moved || v.status_changed) best = v;
				}
				if (best.status_changed) ++status_change;
				else if (best.tokens_moved) ++tok_moves;
				else if (best.logits_moved) ++logit_only;
				else ++silent;
				std::printf("layer%d.%-4s[%d]: %s\n", l, kTensorName[t], e, Label(best));
			}
		}
	}
	std::printf("\nAXIS2 SUMMARY over 56 elements: TOKENS-MOVE=%d LOGITS-ONLY=%d SILENT=%d "
	            "STATUS-CHANGE=%d\n",
	            tok_moves, logit_only, silent, status_change);
	return 0;
}
