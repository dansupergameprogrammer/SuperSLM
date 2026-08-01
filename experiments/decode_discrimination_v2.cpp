// T-1641 solver experiment harness — decode-level weight discrimination at the
// S3.7 multi-position surface (all seven layer weight tensors, q/k included).
//
// DISPOSABLE. Laplace's experiment rig against main@9fc75b0, never product
// code and never a suite test. Successor to T-1409's decode_discrimination.cpp
// (branch claude/solver-oracle-discrimination@9fa1391), rebuilt for the S3.7
// surface: RunLayerLoop's width is now seq.context_length + 1, the K/V store
// is position-minor per (layer, head), and RunGreedyDecodeLoop takes
// kv_precision.
//
// BUILD BASE. This file is BANKED on claude/solver-oracle-discrimination beside
// T-1409's rig, but it does NOT compile against that branch's own tree, which
// predates S3.7. Build it the way it was authored: extract main@9fc75b0 with
// `git archive`, drop this file and build_v2.bat into experiments/, run
// build_v2.bat. See Claude/Laplace/superslm-t1641-decode-discriminating-
// calibration-2026-08-01.md section 8 in D:\Wizard.
//
// Modes (argv[1]):
//   instrument — baseline calibration instrument: replicated decode loop with
//                the chain trace hook installed, per-site (d_prime, e_out) per
//                token, landed K/V rows read back through KeyRow/ValueRow,
//                replicated attention scores/probs per (token, layer), and a
//                bit-identity check of the replicated loop against the real
//                RunGreedyDecodeLoop.
//   mutate     — the matrix: single-element mutation over every weight tensor
//                element (2 layers x 7 tensors x 4 elements), plus norm gains,
//                embed, head; decode output diffed bit-exactly against
//                baseline across all prompts; MOVES counted only when every
//                status in the mutated run is Ok.
//   zero       — controls: whole-tensor zeroing per (layer, tensor) plus the
//                full zero-map, each diffed against baseline.
//   domain     — every counted run of the matrix re-run with the trace hook,
//                reporting the global max d_prime across every funnel site and
//                the kv saturation counter, against C29's 2^31 ceiling.
//   refusals   — the refusal boundary: every (layer, tensor, element) at every
//                delta in {+-1, +-2, +-3} over every prompt, printing each run
//                whose forward status is not Ok, with a total.
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/model.h"
#include "superslm/trace_hook.h"

#include "sslm_fixtures.h"
#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_model_hostile_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

namespace {

using superslm::CarriedScale;
using superslm::LayerWeights;
using superslm::SequenceLayerState;
using superslm::SslmForwardStatus;

// ---------------------------------------------------------------------------
// The candidate calibration. Every number here is the derivation's output;
// the record (Claude/Laplace/superslm-t1641-...) carries the arithmetic.
// ---------------------------------------------------------------------------
struct Calib {
	// C30 i-exp triple per layer, derived by executing the vendored
	// reference's iexp_scale_constants(1500000000, e_iexp, ...) — e_iexp is
	// the knob; the triples for e in {-41..-44} are pinned below.
	int e_iexp[2] = {-43, -43};
	// Residual-balance boosts on the o_proj / down_proj site-constant
	// exponents per layer (T-1409's mechanism, re-derived at this geometry
	// from the instrument's own e-gap readings).
	int64_t o_e_boost[2] = {0, 0};
	int64_t down_e_boost[2] = {0, 0};
	// K/V landing exponent per layer (r_t fixed at reciprocal(2^30) = 2^32);
	// e_t shifts the landed code magnitude by 2^-e_t relative to e_norm.
	int64_t e_t_k[2] = {0, 0};
	int64_t e_t_v[2] = {0, 0};
};

struct IExpTriple { int64_t q_ln2, q_b, q_c; };
IExpTriple TripleFor(int e_iexp) {
	// Executed from tests/reference/superslm_spike/intmath.py
	// iexp_scale_constants(1500000000, e, LN2_Q, 30, B_Q, 30, CA_Q, 30):
	switch (e_iexp) {
		case -41: return {INT64_C(1016), INT64_C(1983), INT64_C(2062274)};
		case -42: return {INT64_C(2032), INT64_C(3967), INT64_C(8249096)};
		case -43: return {INT64_C(4064), INT64_C(7934), INT64_C(32996387)};
		case -44: return {INT64_C(8129), INT64_C(15868), INT64_C(131985548)};
		case -45: return {INT64_C(16258), INT64_C(31736), INT64_C(527942195)};
		case -52: return {INT64_C(2081104), INT64_C(4062246), INT64_C(8649804928567)};  // TwoLayerFixture's own
		default:
			std::fprintf(stderr, "FATAL: no pinned triple for e_iexp=%d\n", e_iexp);
			std::abort();
	}
}

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
	return MakeSection(superslm::SslmSectionType::RopeTables, superslm::SslmDtype::Int64,
	                   manifest.bytes, /*alignment=*/64);
}

// Geometry: hidden 2, two layers, one head, head_dim 2, intermediate 2,
// vocab 3, context_cap 4 (a 4-whole-token run reaches widths 1,2,3,4).
struct FixtureV2 {
	static constexpr int32_t kVocabSize = 3;
	static constexpr int64_t kContextCap = 4;
	static constexpr size_t kWorkspaceBytes = 2 * 4 * 1 * 2 * 2;  // layers*cap*heads*head_dim*2

	superslm::SslmModelView view;
	LayerWeights layers[2];

	int64_t kv_r_t_k[2][1], kv_e_t_k[2][1], kv_r_t_v[2][1], kv_e_t_v[2][1];
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};

	// Per-layer, per-tensor weight storage. [layer][tensor][elem], tensor
	// order: q, k, v, o, gate, up, down.
	int8_t w[2][7][4];
	int32_t attn_gain[2][2];
	int32_t mlp_gain[2][2];
	int32_t final_gain[2];
	int8_t embed_weights[6];
	int8_t head_weights[6];
	CarriedScale embed_site_constant{INT64_C(1073741824), INT64_C(0)};
	CarriedScale final_norm_site_constant{INT64_C(1073741824), INT64_C(-30)};

	FixtureV2(const int8_t weights[2][7][4], const int8_t embed[6], const int8_t head[6],
	          int32_t gain_q16, const Calib& cal) {
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
		spec.context_cap = static_cast<int32_t>(kContextCap);
		spec.kv_precision = 0;
		spec.kv_block_size = 1;
		FixtureSection config =
		    MakeSection(superslm::SslmSectionType::Config, superslm::SslmDtype::Raw, BuildCfg1(spec));
		// Identity rotation at every position (cos = 2^30, sin = 0): RoPE is
		// deliberately inert so the attention arithmetic below is exactly
		// scores = q·K with the landed codes.
		std::vector<int64_t> cos_flat(kContextCap, INT64_C(1073741824));
		std::vector<int64_t> sin_flat(kContextCap, INT64_C(0));
		FixtureSection rope = MakeRopeSection(static_cast<int32_t>(kContextCap), 1,
		                                       cos_flat.data(), sin_flat.data());
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
		const int64_t r_t = superslm::DynamicScaleReciprocal(canonical.m);
		for (int l = 0; l < 2; ++l) {
			kv_r_t_k[l][0] = r_t;
			kv_r_t_v[l][0] = r_t;
			kv_e_t_k[l][0] = cal.e_t_k[l];
			kv_e_t_v[l][0] = cal.e_t_v[l];
			LayerWeights& lw = layers[l];
			lw.attn_norm_gain = attn_gain[l];
			lw.attn_norm_site_constant = canonical;
			lw.q_weight = w[l][0];
			lw.k_weight = w[l][1];
			lw.v_weight = w[l][2];
			lw.o_weight = w[l][3];
			lw.proj_identity = 1;
			lw.proj_mult = 0;
			lw.proj_shift = 0;
			lw.q_site_constant = canonical;
			lw.o_site_constant = CarriedScale{canonical.m, canonical.e + cal.o_e_boost[l]};
			lw.kv_landing_r_t_k = kv_r_t_k[l];
			lw.kv_landing_e_t_k = kv_e_t_k[l];
			lw.kv_landing_r_t_v = kv_r_t_v[l];
			lw.kv_landing_e_t_v = kv_e_t_v[l];
			lw.ctx_fold_identity = ctx_fold_identity_arr;
			lw.ctx_fold_mult = ctx_fold_mult_arr;
			lw.ctx_fold_shift = ctx_fold_shift_arr;
			lw.ctx_fold_site_constant = canonical;
			lw.attn_residual_site_constant = canonical;
			const IExpTriple t = TripleFor(cal.e_iexp[l]);
			lw.q_ln2 = t.q_ln2;
			lw.q_b_iexp = t.q_b;
			lw.q_c_iexp = t.q_c;
			lw.mlp_norm_gain = mlp_gain[l];
			lw.mlp_norm_site_constant = canonical;
			lw.gate_weight = w[l][4];
			lw.up_weight = w[l][5];
			lw.down_weight = w[l][6];
			lw.gate_site_constant = canonical;
			lw.up_site_constant = canonical;
			lw.mlp_act_site_constant = CarriedScale{INT64_C(1073741824), INT64_C(-96)};
			lw.down_site_constant = CarriedScale{canonical.m, canonical.e + cal.down_e_boost[l]};
			lw.mlp_residual_site_constant = canonical;
		}
	}
	FixtureV2(const FixtureV2&) = delete;
	FixtureV2& operator=(const FixtureV2&) = delete;
	FixtureV2(FixtureV2&&) = delete;
	FixtureV2& operator=(FixtureV2&&) = delete;
};

struct DecodeResult {
	SslmForwardStatus status = SslmForwardStatus::Ok;
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;
	size_t produced = 0;
	superslm::SslmDecodeStopReason stop_reason{};
	uint64_t kv_saturation = 0;

	bool SameOutputAs(const DecodeResult& o) const {
		return status == o.status && produced == o.produced && tokens == o.tokens &&
		       logit_rows == o.logit_rows;
	}
};

DecodeResult RunDecode(FixtureV2& f, const std::vector<int32_t>& prompt, size_t max_new_tokens) {
	DecodeResult r;
	int8_t hidden_codes[2] = {0, 0};
	SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes;
	uint8_t workspace[FixtureV2::kWorkspaceBytes];
	std::memset(workspace, 0xEE, sizeof(workspace));
	r.tokens.assign(max_new_tokens, INT32_C(-99));
	r.logit_rows.assign(max_new_tokens * FixtureV2::kVocabSize, INT32_C(-99));
	r.status = superslm::RunGreedyDecodeLoop(
	    seq, f.layers, /*num_hidden_layers=*/2, /*hidden_size=*/2, /*head_dim=*/2,
	    /*intermediate_size=*/2, FixtureV2::kContextCap, f.view.rope_tables, prompt.data(),
	    prompt.size(), f.embed_weights, f.embed_site_constant, f.final_gain,
	    f.final_norm_site_constant, f.head_weights, FixtureV2::kVocabSize,
	    /*stop_ids=*/nullptr, /*stop_count=*/0, max_new_tokens, workspace, sizeof(workspace),
	    r.tokens.data(), r.logit_rows.data(), r.tokens.size(), &r.produced, &r.stop_reason);
	r.kv_saturation = seq.kv_saturation_count;
	return r;
}

void PrintResult(const char* tag, const DecodeResult& r) {
	std::printf("%s: status=%s produced=%zu sat=%" PRIu64 " tokens=[",
	            tag, superslm::SslmForwardStatusName(r.status), r.produced, r.kv_saturation);
	for (size_t i = 0; i < r.produced; ++i) std::printf("%s%d", i ? "," : "", r.tokens[i]);
	std::printf("] logits=[");
	for (size_t i = 0; i < r.produced * FixtureV2::kVocabSize; ++i)
		std::printf("%s%d", i ? "," : "", r.logit_rows[i]);
	std::printf("]\n");
}

// --- trace capture ------------------------------------------------------------
struct TraceCapture {
	struct Row {
		std::string site;
		size_t token_index;
		int64_t d_prime;
		int64_t e_out;
		std::vector<int8_t> codes;
	};
	std::vector<Row> rows;
	static void Hook(const superslm::SslmChainTraceRecord* r,
	                 const superslm::SslmKvLandingTraceRecord* kv, void* user) {
		auto* self = static_cast<TraceCapture*>(user);
		(void)kv;
		if (r != nullptr) {
			Row row{std::string(r->site), r->token_index, r->d_prime, r->e_out, {}};
			row.codes.assign(r->codes.begin(), r->codes.end());
			self->rows.push_back(std::move(row));
		}
	}
};

// Replicated decode loop: RunWholeToken (EmbedEntry + RunLayerLoop) + final
// norm + logits + argmax, exactly RunGreedyDecodeLoop's own composition, with
// the trace hook installed and the K/V store readable between tokens. Used by
// `instrument` and `domain` only; the matrix runs through the real driver.
struct ReplicatedDecode {
	FixtureV2& f;
	superslm::SslmTraceHookState hook_state{};
	TraceCapture capture;
	uint8_t workspace[FixtureV2::kWorkspaceBytes];
	int8_t hidden_codes[2] = {0, 0};
	SequenceLayerState seq{};
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;
	SslmForwardStatus status = SslmForwardStatus::Ok;

	explicit ReplicatedDecode(FixtureV2& fixture) : f(fixture) {
		seq.hidden_codes = hidden_codes;
		std::memset(workspace, 0xEE, sizeof(workspace));
		superslm::SslmSetTraceHook(hook_state, &TraceCapture::Hook, &capture);
	}

	SslmForwardStatus RunWholeToken(int32_t token, size_t token_index) {
		int8_t embed_codes[2];
		CarriedScale embed_scale{};
		auto st = superslm::EmbedEntry(token, FixtureV2::kVocabSize, f.embed_weights, 2,
		                               f.embed_site_constant, embed_codes, &embed_scale,
		                               "embed", token_index, &hook_state);
		if (st != SslmForwardStatus::Ok) return st;
		seq.hidden_codes[0] = embed_codes[0];
		seq.hidden_codes[1] = embed_codes[1];
		seq.hidden_scale = embed_scale;
		seq.layer_index = 0;
		return superslm::RunLayerLoop(seq, f.layers, 2, /*layer_budget=*/2, 2, 2, 2,
		                              FixtureV2::kContextCap, f.view.rope_tables, workspace,
		                              sizeof(workspace), /*site_prefix=*/{}, token_index,
		                              &hook_state);
	}

	// Runs the whole decode; mirrors RunGreedyDecodeLoop's prefill/generation
	// split and its whole-token indexing.
	void Run(const std::vector<int32_t>& prompt, size_t max_new_tokens) {
		size_t token_index = 0;
		for (size_t i = 0; i + 1 < prompt.size(); ++i) {
			status = RunWholeToken(prompt[i], token_index++);
			if (status != SslmForwardStatus::Ok) return;
		}
		int32_t current = prompt.back();
		while (tokens.size() < max_new_tokens) {
			status = RunWholeToken(current, token_index);
			if (status != SslmForwardStatus::Ok) return;
			int8_t final_codes[2];
			CarriedScale final_scale{};
			status = superslm::RmsNormSite(seq.hidden_codes, f.final_gain, 2, seq.hidden_scale,
			                               f.final_norm_site_constant, final_codes, &final_scale,
			                               "final_norm", token_index, &hook_state);
			if (status != SslmForwardStatus::Ok) return;
			int64_t wide[FixtureV2::kVocabSize];
			int32_t row[FixtureV2::kVocabSize];
			status = superslm::LogitsSite(final_codes, 2, f.head_weights, FixtureV2::kVocabSize,
			                              wide, row);
			if (status != SslmForwardStatus::Ok) return;
			const int32_t tok = superslm::ArgmaxLowestIndexTieBreak(row, FixtureV2::kVocabSize);
			tokens.push_back(tok);
			logit_rows.insert(logit_rows.end(), row, row + FixtureV2::kVocabSize);
			current = tok;
			++token_index;
		}
	}
};

// --- weight sets --------------------------------------------------------------

// The candidate: per-tensor distinct, channel-mixing int8 matrices, q/k now
// non-identity (the S3.7 surface makes them observable). Row-major [out,in].
void CandidateWeights(int8_t w[2][7][4]) {
	static const int8_t q0[4] = {2, 1, 1, 1},  k0[4] = {1, 1, 1, 2};
	static const int8_t v0[4] = {1, 1, 0, 1},  o0[4] = {1, 0, 1, 1};
	static const int8_t g0[4] = {1, 1, 1, 0},  u0[4] = {2, 1, 1, 1};
	static const int8_t d0[4] = {1, 1, 0, 1};
	static const int8_t q1[4] = {1, 2, 1, 0},  k1[4] = {2, 0, 1, 1};
	static const int8_t v1[4] = {0, 1, 1, 0},  o1[4] = {1, 1, 1, 0};
	static const int8_t g1[4] = {1, 0, 1, 1},  u1[4] = {1, 1, 0, 2};
	static const int8_t d1[4] = {2, 1, 1, 1};
	const int8_t* l0[7] = {q0, k0, v0, o0, g0, u0, d0};
	const int8_t* l1[7] = {q1, k1, v1, o1, g1, u1, d1};
	for (int t = 0; t < 7; ++t) {
		std::memcpy(w[0][t], l0[t], 4);
		std::memcpy(w[1][t], l1[t], 4);
	}
}

void ZeroWeights(int8_t w[2][7][4]) { std::memset(w, 0, 2 * 7 * 4); }

const int8_t kEmbed[6] = {5, -3, 7, 2, -4, 6};
const int8_t kHead[6] = {9, -2, -3, 8, 5, 5};
const char* kTensorName[7] = {"q", "k", "v", "o", "gate", "up", "down"};

// One prompt set: three multi-token prompts (distinct token orders so the K/V
// store carries genuinely different rows per position) + one single-token.
const std::vector<std::vector<int32_t>> kPrompts = {{0, 1}, {1, 2}, {2, 0}, {1}};
constexpr size_t kMaxNew = 3;

Calib ActiveCalib() {
	Calib c;
	const char* env = std::getenv("V2CAL");
	if (env != nullptr) {
		// "o0,o1,d0,d1,ek0,ek1,ev0,ev1,ei0,ei1" — the sweep interface.
		long v[10] = {0};
		std::sscanf(env, "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld", &v[0], &v[1], &v[2], &v[3],
		            &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]);
		c.o_e_boost[0] = v[0]; c.o_e_boost[1] = v[1];
		c.down_e_boost[0] = v[2]; c.down_e_boost[1] = v[3];
		c.e_t_k[0] = v[4]; c.e_t_k[1] = v[5];
		c.e_t_v[0] = v[6]; c.e_t_v[1] = v[7];
		c.e_iexp[0] = static_cast<int>(v[8]); c.e_iexp[1] = static_cast<int>(v[9]);
	}
	return c;
}

int RunInstrument() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	const Calib cal = ActiveCalib();
	std::printf("calib: o_boost={%lld,%lld} down_boost={%lld,%lld} e_t_k={%lld,%lld} "
	            "e_t_v={%lld,%lld} e_iexp={%d,%d}\n",
	            (long long)cal.o_e_boost[0], (long long)cal.o_e_boost[1],
	            (long long)cal.down_e_boost[0], (long long)cal.down_e_boost[1],
	            (long long)cal.e_t_k[0], (long long)cal.e_t_k[1],
	            (long long)cal.e_t_v[0], (long long)cal.e_t_v[1], cal.e_iexp[0], cal.e_iexp[1]);

	for (const auto& prompt : kPrompts) {
		FixtureV2 f(wc, kEmbed, kHead, 16384, cal);
		ReplicatedDecode rep(f);
		rep.Run(prompt, kMaxNew);
		std::printf("\n=== prompt {");
		for (size_t i = 0; i < prompt.size(); ++i) std::printf("%s%d", i ? "," : "", prompt[i]);
		std::printf("} status=%s tokens=[", superslm::SslmForwardStatusName(rep.status));
		for (size_t i = 0; i < rep.tokens.size(); ++i)
			std::printf("%s%d", i ? "," : "", rep.tokens[i]);
		std::printf("] sat=%" PRIu64 "\n", rep.seq.kv_saturation_count);

		// Cross-check against the real driver.
		FixtureV2 f2(wc, kEmbed, kHead, 16384, cal);
		DecodeResult real = RunDecode(f2, prompt, kMaxNew);
		bool same = (real.status == rep.status) && (real.produced == rep.tokens.size());
		if (same) {
			for (size_t i = 0; i < real.produced; ++i)
				if (real.tokens[i] != rep.tokens[i]) same = false;
			for (size_t i = 0; i < real.produced * FixtureV2::kVocabSize; ++i)
				if (real.logit_rows[i] != rep.logit_rows[i]) same = false;
		}
		std::printf("replicated-vs-real identical=%s\n", same ? "YES" : "NO");

		// Per-site trace table.
		for (const auto& r : rep.capture.rows) {
			std::printf("  tok=%zu site=%-24s d_prime=%-11" PRId64 " e_out=%-4" PRId64 " codes=[",
			            r.token_index, r.site.c_str(), r.d_prime, r.e_out);
			for (size_t i = 0; i < r.codes.size(); ++i)
				std::printf("%s%d", i ? "," : "", (int)r.codes[i]);
			std::printf("]\n");
		}

		// Landed K/V rows (post-RoPE; identity table so as-landed), both layers.
		const int64_t width = rep.seq.context_length;
		for (uint32_t l = 0; l < 2; ++l) {
			for (int64_t p = 0; p < width; ++p) {
				const int8_t* k = superslm::KeyRow(rep.workspace, l, FixtureV2::kContextCap, 1, 2, 0, p);
				const int8_t* v = superslm::ValueRow(rep.workspace, l, FixtureV2::kContextCap, 1, 2, 0, p);
				std::printf("  layer%u pos%lld K=[%d,%d] V=[%d,%d]\n", l, (long long)p,
				            (int)k[0], (int)k[1], (int)v[0], (int)v[1]);
			}
		}

		// Replicated attention probabilities per (token, layer): scores from
		// the traced q_proj codes (identity RoPE => q_rot == q codes) against
		// the landed K rows AT THE TIME of that token. The K store is
		// append-only across this run (each token writes only its own
		// position), so the final store carries every position's row
		// unchanged; token t at width w reads rows 0..w-1.
		// q codes per (token, layer) come from the trace rows named
		// "layerL.q_proj.requant".
		for (const auto& r : rep.capture.rows) {
			if (r.site.find("q_proj.requant") == std::string::npos) continue;
			const uint32_t l = (r.site.find("layer0") != std::string::npos) ? 0u : 1u;
			// width for whole token index t is t+1 (fresh sequence).
			const int64_t wdt = static_cast<int64_t>(r.token_index) + 1;
			if (wdt > width) continue;
			std::vector<int64_t> scores(wdt), probs(wdt);
			const int8_t* k_base = superslm::KeyRow(rep.workspace, l, FixtureV2::kContextCap, 1, 2, 0, 0);
			superslm::GemmInt8AccumulateRow(r.codes.data(), k_base, 2, static_cast<size_t>(wdt),
			                                scores.data());
			const IExpTriple t = TripleFor(cal.e_iexp[l]);
			const bool ok = superslm::SoftmaxRowQ15(scores.data(), static_cast<size_t>(wdt),
			                                        t.q_ln2, t.q_b, t.q_c, probs.data());
			std::printf("  tok=%zu layer%u scores=[", r.token_index, l);
			for (int64_t i = 0; i < wdt; ++i) std::printf("%s%lld", i ? "," : "", (long long)scores[i]);
			std::printf("] probs=[");
			for (int64_t i = 0; i < wdt; ++i) std::printf("%s%lld", i ? "," : "", (long long)probs[i]);
			std::printf("] wf=%d\n", ok ? 1 : 0);
		}
	}
	return 0;
}

struct MutationOutcome {
	bool moves = false, refuses = false;
	int used_delta = 0;
	std::string detail;
};

MutationOutcome ProbeElement(const int8_t base_w[2][7][4], int l, int t, int e,
                             const DecodeResult baseline[], const Calib& cal) {
	MutationOutcome out;
	for (int delta : {1, -1, 2, -2, 3, -3}) {
		int8_t wm[2][7][4];
		std::memcpy(wm, base_w, 2 * 7 * 4);
		wm[l][t][e] = static_cast<int8_t>(wm[l][t][e] + delta);
		FixtureV2 fm(wm, kEmbed, kHead, 16384, cal);
		bool diff = false, refusal = false;
		std::string detail;
		for (size_t p = 0; p < kPrompts.size(); ++p) {
			DecodeResult r = RunDecode(fm, kPrompts[p], kMaxNew);
			if (r.status != SslmForwardStatus::Ok) refusal = true;
			if (!r.SameOutputAs(baseline[p])) {
				if (!diff && !refusal) {
					// first difference, for the record
					char buf[128];
					for (size_t i = 0; i < r.produced * FixtureV2::kVocabSize; ++i) {
						if (r.logit_rows[i] != baseline[p].logit_rows[i]) {
							std::snprintf(buf, sizeof(buf), "prompt#%zu logit[%zu] %d->%d", p, i,
							              baseline[p].logit_rows[i], r.logit_rows[i]);
							detail = buf;
							break;
						}
					}
					if (detail.empty()) {
						for (size_t i = 0; i < r.produced; ++i) {
							if (r.tokens[i] != baseline[p].tokens[i]) {
								std::snprintf(buf, sizeof(buf), "prompt#%zu token[%zu] %d->%d", p, i,
								              baseline[p].tokens[i], r.tokens[i]);
								detail = buf;
								break;
							}
						}
					}
					if (detail.empty()) detail = "output differs (count/status shape)";
				}
				diff = true;
			}
		}
		if (diff && !refusal) {
			out.moves = true;
			out.used_delta = delta;
			out.detail = detail;
			return out;
		}
		if (refusal) out.refuses = true;
	}
	return out;
}

int RunMutate() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	const Calib cal = ActiveCalib();
	FixtureV2 base(wc, kEmbed, kHead, 16384, cal);
	DecodeResult baseline[8];
	for (size_t p = 0; p < kPrompts.size(); ++p) {
		baseline[p] = RunDecode(base, kPrompts[p], kMaxNew);
		char tag[32];
		std::snprintf(tag, sizeof(tag), "baseline#%zu", p);
		PrintResult(tag, baseline[p]);
		if (baseline[p].status != SslmForwardStatus::Ok) {
			std::printf("BASELINE NOT OK\n");
			return 1;
		}
	}

	int total = 0, moved = 0, silent = 0, refused = 0;
	int per_tensor_moved[7] = {0};
	for (int l = 0; l < 2; ++l) {
		for (int t = 0; t < 7; ++t) {
			for (int e = 0; e < 4; ++e) {
				++total;
				MutationOutcome o = ProbeElement(wc, l, t, e, baseline, cal);
				const char* verdict = o.moves ? "MOVES" : (o.refuses ? "REFUSES" : "SILENT");
				if (o.moves) { ++moved; ++per_tensor_moved[t]; }
				else if (o.refuses) ++refused;
				else ++silent;
				std::printf("layer%d.%s[%d]: %s", l, kTensorName[t], e, verdict);
				if (o.moves) std::printf(" delta=%+d  first-diff: %s", o.used_delta, o.detail.c_str());
				std::printf("\n");
			}
		}
	}

	// Norm gains, embed, head — same protocol.
	auto probe_i32 = [&](const char* name, int32_t* slot) {
		const int32_t orig = *slot;
		bool any = false;
		for (int32_t delta : {4096, -4096, 8192, -8192}) {
			*slot = orig + delta;
			bool diff = false, refusal = false;
			for (size_t p = 0; p < kPrompts.size(); ++p) {
				DecodeResult r = RunDecode(base, kPrompts[p], kMaxNew);
				if (r.status != SslmForwardStatus::Ok) refusal = true;
				if (!r.SameOutputAs(baseline[p])) diff = true;
			}
			if (diff && !refusal) { any = true; break; }
		}
		*slot = orig;
		std::printf("%s: %s\n", name, any ? "MOVES" : "SILENT-OR-REFUSES");
	};
	probe_i32("layer0.attn_norm_gain[0]", &base.attn_gain[0][0]);
	probe_i32("layer0.mlp_norm_gain[0]", &base.mlp_gain[0][0]);
	probe_i32("layer1.attn_norm_gain[1]", &base.attn_gain[1][1]);
	probe_i32("layer1.mlp_norm_gain[1]", &base.mlp_gain[1][1]);
	probe_i32("final_norm_gain[0]", &base.final_gain[0]);

	auto probe_i8 = [&](const char* name, int8_t* slot) {
		const int8_t orig = *slot;
		bool any = false;
		for (int delta : {1, -1, 2, -2}) {
			*slot = static_cast<int8_t>(orig + delta);
			bool diff = false, refusal = false;
			for (size_t p = 0; p < kPrompts.size(); ++p) {
				DecodeResult r = RunDecode(base, kPrompts[p], kMaxNew);
				if (r.status != SslmForwardStatus::Ok) refusal = true;
				if (!r.SameOutputAs(baseline[p])) diff = true;
			}
			if (diff && !refusal) { any = true; break; }
		}
		*slot = orig;
		std::printf("%s: %s\n", name, any ? "MOVES" : "SILENT-OR-REFUSES");
	};
	for (int i = 0; i < 6; ++i) {
		char nm[32];
		std::snprintf(nm, sizeof(nm), "embed[%d]", i);
		probe_i8(nm, &base.embed_weights[i]);
	}
	for (int i = 0; i < 6; ++i) {
		char nm[32];
		std::snprintf(nm, sizeof(nm), "head[%d]", i);
		probe_i8(nm, &base.head_weights[i]);
	}

	std::printf("\nSUMMARY weight elements: total=%d MOVES=%d REFUSES=%d SILENT=%d\n", total,
	            moved, refused, silent);
	std::printf("per-tensor (of 8 elements each):");
	for (int t = 0; t < 7; ++t) std::printf(" %s=%d", kTensorName[t], per_tensor_moved[t]);
	std::printf("\n");
	return silent == 0 && refused == 0 ? 0 : 3;
}

int RunZero() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	const Calib cal = ActiveCalib();
	FixtureV2 base(wc, kEmbed, kHead, 16384, cal);
	DecodeResult baseline[8];
	for (size_t p = 0; p < kPrompts.size(); ++p) baseline[p] = RunDecode(base, kPrompts[p], kMaxNew);

	// Whole-tensor zero controls.
	for (int l = 0; l < 2; ++l) {
		for (int t = 0; t < 7; ++t) {
			int8_t wm[2][7][4];
			std::memcpy(wm, wc, sizeof(wm));
			std::memset(wm[l][t], 0, 4);
			FixtureV2 fm(wm, kEmbed, kHead, 16384, cal);
			bool diff = false;
			bool all_ok = true;
			for (size_t p = 0; p < kPrompts.size(); ++p) {
				DecodeResult r = RunDecode(fm, kPrompts[p], kMaxNew);
				if (r.status != SslmForwardStatus::Ok) all_ok = false;
				if (!r.SameOutputAs(baseline[p])) diff = true;
			}
			std::printf("zero layer%d.%s: %s%s\n", l, kTensorName[t],
			            diff ? "MOVES" : "SILENT", all_ok ? "" : " (some refusal)");
		}
	}
	// Full zero map (D-SLM493's control).
	int8_t wz[2][7][4];
	ZeroWeights(wz);
	FixtureV2 fz(wz, kEmbed, kHead, 16384, cal);
	bool diff = false;
	for (size_t p = 0; p < kPrompts.size(); ++p) {
		DecodeResult r = RunDecode(fz, kPrompts[p], kMaxNew);
		if (!r.SameOutputAs(baseline[p])) diff = true;
	}
	std::printf("zero ALL (the D-SLM493 control): %s\n", diff ? "MOVES" : "SILENT");
	return 0;
}

// Every counted matrix run re-executed through the replicated loop with the
// hook installed, tracking the global max d_prime per site class and the
// saturation counter — the C29 margin evidence.
int RunDomain() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	const Calib cal = ActiveCalib();

	int64_t global_max_dprime = 0;
	std::string max_site;
	uint64_t max_sat = 0;
	int runs = 0, not_ok = 0;

	auto measure = [&](const int8_t w[2][7][4]) {
		for (const auto& prompt : kPrompts) {
			FixtureV2 f(w, kEmbed, kHead, 16384, cal);
			ReplicatedDecode rep(f);
			rep.Run(prompt, kMaxNew);
			++runs;
			if (rep.status != SslmForwardStatus::Ok) ++not_ok;
			for (const auto& r : rep.capture.rows) {
				if (r.d_prime > global_max_dprime) {
					global_max_dprime = r.d_prime;
					max_site = r.site;
				}
			}
			if (rep.seq.kv_saturation_count > max_sat) max_sat = rep.seq.kv_saturation_count;
		}
	};

	measure(wc);  // baseline
	for (int l = 0; l < 2; ++l)
		for (int t = 0; t < 7; ++t)
			for (int e = 0; e < 4; ++e)
				for (int delta : {1, -1, 2, -2, 3, -3}) {
					int8_t wm[2][7][4];
					std::memcpy(wm, wc, sizeof(wm));
					wm[l][t][e] = static_cast<int8_t>(wm[l][t][e] + delta);
					measure(wm);
				}

	const int64_t ceiling = INT64_C(1) << 31;
	std::printf("runs=%d not_ok=%d\n", runs, not_ok);
	std::printf("global max d_prime = %" PRId64 " at %s (C29 ceiling 2^31 = %" PRId64
	            ", margin %.3fx)\n",
	            global_max_dprime, max_site.c_str(), ceiling,
	            (double)ceiling / (double)global_max_dprime);
	std::printf("max kv_saturation_count across runs = %" PRIu64 "\n", max_sat);
	return 0;
}

// Names every refusing (layer, tensor, elem, delta, prompt) with its status —
// the honest characterization of the delta boundary.
int RunRefusals() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	const Calib cal = ActiveCalib();
	int refusing_runs = 0;
	for (int l = 0; l < 2; ++l)
		for (int t = 0; t < 7; ++t)
			for (int e = 0; e < 4; ++e)
				for (int delta : {1, -1, 2, -2, 3, -3}) {
					int8_t wm[2][7][4];
					std::memcpy(wm, wc, sizeof(wm));
					wm[l][t][e] = static_cast<int8_t>(wm[l][t][e] + delta);
					FixtureV2 fm(wm, kEmbed, kHead, 16384, cal);
					for (size_t p = 0; p < kPrompts.size(); ++p) {
						DecodeResult r = RunDecode(fm, kPrompts[p], kMaxNew);
						if (r.status != SslmForwardStatus::Ok) {
							++refusing_runs;
							std::printf("layer%d.%s[%d] delta=%+d prompt#%zu: %s\n", l,
							            kTensorName[t], e, delta, p,
							            superslm::SslmForwardStatusName(r.status));
						}
					}
				}
	std::printf("refusing runs total=%d\n", refusing_runs);
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	const std::string mode = argc > 1 ? argv[1] : "";
	if (mode == "instrument") return RunInstrument();
	if (mode == "mutate") return RunMutate();
	if (mode == "zero") return RunZero();
	if (mode == "domain") return RunDomain();
	if (mode == "refusals") return RunRefusals();
	std::fprintf(stderr, "usage: %s instrument|mutate|zero|domain|refusals   [V2CAL=o0,o1,d0,d1,ek0,ek1,ev0,ev1,ei0,ei1]\n",
	             argv[0]);
	return 2;
}
