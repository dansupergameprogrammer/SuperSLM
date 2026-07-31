// T-1409 solver experiment harness — decode-level weight discrimination.
//
// DISPOSABLE. This is Laplace's experiment rig, not product code and not a
// suite test. It exists to measure whether RunGreedyDecodeLoop's output
// discriminates the transformer layer weights under (A) the shipped
// TwoLayerFixture geometry and (B) the candidate mixing-weight geometry, and
// to produce the mutation matrix the result packet reports.
//
// Modes (argv[1]):
//   null      — Experiment A: reproduce D-SLM493 at decode level through the
//               real decode driver (identity weights vs zero map).
//   candidate — Experiment B: candidate fixture, baseline run + zero-map
//               control, with per-site trace capture (d_prime margins).
//   mutate    — Experiment C: single-element mutation matrix over every
//               weight array (+1 first, then -1 where +1 is silent), decode
//               output diffed bit-exactly against the baseline.
//   qk        — Experiment D: q/k width-1 invariance probe.
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

// Same shape as tests/test_main.cpp's MakeRop1SectionMultiRow (that one is
// static inside the test TU, so it is reproduced here byte-for-byte in
// spirit: two [context_cap, pairs] int64 tensors named cos/sin).
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

// The solver's decode fixture. Geometry matches TwoLayerFixture /
// DecodeLoopFixture exactly (hidden=2, two layers, one head, head_dim=2,
// intermediate=2, context_cap=1, vocab=3); every weight tensor is its OWN
// array (no aliasing), so a single-element mutation names one tensor.
struct SolverDecodeFixture {
	superslm::SslmModelView view;
	LayerWeights layers[2];

	int64_t kv_landing_r_t_arr[1];
	int64_t kv_landing_e_t_arr[1] = {0};
	// Trace-only landing targets (T-1412): the (m_target, e_target) the
	// landing record's m_out/e_out carry. This fixture's landed codes sit at
	// the canonical scale the reciprocal was derived from, exponent 0.
	int64_t kv_landing_m_target_arr[1] = {INT64_C(1073741824)};
	int64_t kv_landing_e_target_arr[1] = {0};
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};

	// Per-layer, per-tensor weight storage. Indexed [layer][tensor], tensor
	// order: q, k, v, o, gate, up, down.
	int8_t w[2][7][4];
	// Per-layer norm gains (attn, mlp) and the final norm gain — distinct
	// arrays so each is independently mutable.
	int32_t attn_gain[2][2];
	int32_t mlp_gain[2][2];
	int32_t final_gain[2];

	static constexpr int32_t kVocabSize = 3;
	int8_t embed_weights[6];
	int8_t head_weights[6];
	CarriedScale embed_site_constant{INT64_C(1073741824), INT64_C(0)};
	CarriedScale final_norm_site_constant{INT64_C(1073741824), INT64_C(-30)};

	// The layer-1 residual calibration (derived from the trace instrument, not
	// tuned blind): the stream's carried-scale exponent grows through each
	// residual funnel while the branch restarts at the norm's gain-derived
	// scale, so layer L>0's branch needs its own site-constant exponent boost
	// to land within reconciliation range of the stream. Measured at the
	// candidate geometry: layer1 o_proj e_out=-2..-1 vs stream e=17..21
	// (attn_residual), layer1 down_proj e_out=16 vs stream e=24..28
	// (mlp_residual). Boosts below re-center e_b - e_h near 0 at both sites.
	struct Calibration {
		int64_t o_e_boost[2] = {0, 20};
		int64_t down_e_boost[2] = {0, 12};
	};

	SolverDecodeFixture(const int8_t weights[2][7][4], const int8_t embed[6], const int8_t head[6],
	                    int32_t gain_q16, Calibration cal = Calibration{}) {
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
		const int64_t cos_flat[1] = {INT64_C(1073741824)};  // cos(0) == 2^30
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
		kv_landing_r_t_arr[0] = superslm::DynamicScaleReciprocal(canonical.m);
		for (int l = 0; l < 2; ++l) {
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
			lw.kv_landing_r_t_k = kv_landing_r_t_arr;
			lw.kv_landing_e_t_k = kv_landing_e_t_arr;
			lw.kv_landing_r_t_v = kv_landing_r_t_arr;
			lw.kv_landing_e_t_v = kv_landing_e_t_arr;
			lw.kv_landing_m_target_k = kv_landing_m_target_arr;
			lw.kv_landing_e_target_k = kv_landing_e_target_arr;
			lw.kv_landing_m_target_v = kv_landing_m_target_arr;
			lw.kv_landing_e_target_v = kv_landing_e_target_arr;
			lw.ctx_fold_identity = ctx_fold_identity_arr;
			lw.ctx_fold_mult = ctx_fold_mult_arr;
			lw.ctx_fold_shift = ctx_fold_shift_arr;
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
			lw.down_site_constant = CarriedScale{canonical.m, canonical.e + cal.down_e_boost[l]};
			lw.mlp_residual_site_constant = canonical;
		}
	}
	SolverDecodeFixture(const SolverDecodeFixture&) = delete;
	SolverDecodeFixture& operator=(const SolverDecodeFixture&) = delete;
	SolverDecodeFixture(SolverDecodeFixture&&) = delete;
	SolverDecodeFixture& operator=(SolverDecodeFixture&&) = delete;
};

struct DecodeResult {
	SslmForwardStatus status = SslmForwardStatus::Ok;
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;
	size_t produced = 0;
	superslm::SslmDecodeStopReason stop_reason{};

	bool SameOutputAs(const DecodeResult& o) const {
		return status == o.status && produced == o.produced && tokens == o.tokens &&
		       logit_rows == o.logit_rows;
	}
};

// Per-site trace capture: name, token_index, d_prime, e_out.
struct TraceCapture {
	struct Row {
		std::string site;
		size_t token_index;
		int64_t d_prime;
		int64_t e_out;
	};
	std::vector<Row> rows;
	struct KvRow {
		std::string site;
		size_t token_index;
		uint32_t head;
		int64_t m_in, e_in, m_out, e_out;
	};
	std::vector<KvRow> kv_rows;
	static void Hook(const superslm::SslmChainTraceRecord* r,
	                 const superslm::SslmKvLandingTraceRecord* kv, void* user) {
		auto* self = static_cast<TraceCapture*>(user);
		if (r != nullptr) {
			self->rows.push_back(Row{std::string(r->site), r->token_index, r->d_prime, r->e_out});
		}
		if (kv != nullptr) {
			self->kv_rows.push_back(KvRow{std::string(kv->site), kv->token_index, kv->head,
			                              kv->m_in, kv->e_in, kv->m_out, kv->e_out});
		}
	}
};

DecodeResult RunDecode(SolverDecodeFixture& f, const std::vector<int32_t>& prompt,
                       size_t max_new_tokens
#ifdef SOLVER_TRACE_WIRED
                       ,
                       superslm::SslmTraceHookState* hook_state = nullptr
#endif
) {
	DecodeResult r;
	int8_t hidden_codes[2] = {0, 0};
	superslm::SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes;
	uint8_t workspace[64];
	std::memset(workspace, 0xEE, sizeof(workspace));
	r.tokens.assign(max_new_tokens, INT32_C(-99));
	r.logit_rows.assign(max_new_tokens * SolverDecodeFixture::kVocabSize, INT32_C(-99));
	r.status = superslm::RunGreedyDecodeLoop(
	    seq, f.layers, /*num_hidden_layers=*/2, /*hidden_size=*/2, /*head_dim=*/2,
	    /*intermediate_size=*/2, /*context_cap=*/1, f.view.rope_tables, prompt.data(),
	    prompt.size(), f.embed_weights, f.embed_site_constant, f.final_gain,
	    f.final_norm_site_constant, f.head_weights, SolverDecodeFixture::kVocabSize,
	    /*stop_ids=*/nullptr, /*stop_count=*/0, max_new_tokens, workspace, sizeof(workspace),
	    r.tokens.data(), r.logit_rows.data(), r.tokens.size(), &r.produced, &r.stop_reason
#ifdef SOLVER_TRACE_WIRED
	    ,
	    /*site_prefix=*/{}, hook_state
#endif
	);
	return r;
}

void PrintResult(const char* tag, const DecodeResult& r) {
	std::printf("%s: status=%s produced=%zu tokens=[", tag,
	            superslm::SslmForwardStatusName(r.status), r.produced);
	for (size_t i = 0; i < r.produced; ++i) std::printf("%s%d", i ? "," : "", r.tokens[i]);
	std::printf("] logits=[");
	for (size_t i = 0; i < r.produced * 3; ++i)
		std::printf("%s%d", i ? "," : "", r.logit_rows[i]);
	std::printf("]\n");
}

// --- weight sets --------------------------------------------------------------

// Shipped TwoLayerFixture geometry: every tensor the 2x2 identity.
void IdentityWeights(int8_t w[2][7][4]) {
	static const int8_t id[4] = {1, 0, 0, 1};
	for (int l = 0; l < 2; ++l)
		for (int t = 0; t < 7; ++t) std::memcpy(w[l][t], id, 4);
}

void ZeroWeights(int8_t w[2][7][4]) { std::memset(w, 0, 2 * 7 * 4); }

// Candidate: per-tensor distinct, channel-mixing matrices, small ints.
// Row-major [out,in]. q/k stay identity (structurally dead at width=1; kept
// identity so their mutation probes measure the pure q/k axis).
void CandidateWeights(int8_t w[2][7][4]) {
	static const int8_t q0[4] = {1, 0, 0, 1}, k0[4] = {1, 0, 0, 1};
	static const int8_t v0[4] = {1, 1, 0, 1};   // shear
	static const int8_t o0[4] = {1, 0, 1, 1};   // opposite shear
	static const int8_t g0[4] = {1, 1, 1, 0};
	static const int8_t u0[4] = {2, 1, 1, 1};
	static const int8_t d0[4] = {1, 1, 0, 1};
	static const int8_t q1[4] = {1, 0, 0, 1}, k1[4] = {1, 0, 0, 1};
	static const int8_t v1[4] = {0, 1, 1, 0};   // swap
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
// DecodeLoopFixture's own embed/head (basis rows + sum) for the null repro.
const int8_t kBasisEmbed[6] = {1, 0, 0, 1, 1, 1};
const int8_t kBasisHead[6] = {1, 0, 0, 1, 1, 1};

const char* kTensorName[7] = {"q", "k", "v", "o", "gate", "up", "down"};

int RunNull() {
	int8_t wid[2][7][4], wz[2][7][4];
	IdentityWeights(wid);
	ZeroWeights(wz);
	// Exact DecodeLoopFixture conditions: basis embed/head, gain 16384.
	SolverDecodeFixture fid(wid, kBasisEmbed, kBasisHead, 16384);
	SolverDecodeFixture fz(wz, kBasisEmbed, kBasisHead, 16384);
	for (int32_t p = 0; p < 3; ++p) {
		DecodeResult a = RunDecode(fid, {p}, 3);
		DecodeResult b = RunDecode(fz, {p}, 3);
		char tag[64];
		std::snprintf(tag, sizeof(tag), "identity prompt=%d", p);
		PrintResult(tag, a);
		std::snprintf(tag, sizeof(tag), "zeromap  prompt=%d", p);
		PrintResult(tag, b);
		std::printf("prompt=%d decode-identical=%s\n", p, a.SameOutputAs(b) ? "YES" : "NO");
	}
	return 0;
}

int RunCandidate() {
	int8_t wc[2][7][4], wz[2][7][4];
	CandidateWeights(wc);
	ZeroWeights(wz);
	SolverDecodeFixture fc(wc, kEmbed, kHead, 16384);
	SolverDecodeFixture fz(wz, kEmbed, kHead, 16384);
	for (int32_t p = 0; p < 3; ++p) {
		DecodeResult a = RunDecode(fc, {p}, 3);
		DecodeResult b = RunDecode(fz, {p}, 3);
		char tag[64];
		std::snprintf(tag, sizeof(tag), "candidate prompt=%d", p);
		PrintResult(tag, a);
		std::snprintf(tag, sizeof(tag), "zeromap   prompt=%d", p);
		PrintResult(tag, b);
		std::printf("prompt=%d decode-identical=%s\n", p, a.SameOutputAs(b) ? "YES" : "NO");
	}
	return 0;
}

// Experiment C: the mutation matrix. For every weight tensor element, apply
// delta and diff the decode output against the baseline (all three prompts).
int RunMutate() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	SolverDecodeFixture base(wc, kEmbed, kHead, 16384);
	DecodeResult baseline[3];
	for (int32_t p = 0; p < 3; ++p) {
		baseline[p] = RunDecode(base, {p}, 3);
		if (baseline[p].status != SslmForwardStatus::Ok) {
			std::printf("BASELINE NOT OK at prompt %d: %s\n", p,
			            superslm::SslmForwardStatusName(baseline[p].status));
			return 1;
		}
	}

	int total = 0, moved = 0, silent = 0, refused = 0;
	for (int l = 0; l < 2; ++l) {
		for (int t = 0; t < 7; ++t) {
			for (int e = 0; e < 4; ++e) {
				++total;
				bool any_diff = false, any_refusal = false;
				int used_delta = 0;
				for (int8_t delta : {int8_t{1}, int8_t{-1}, int8_t{2}, int8_t{-2}}) {
					int8_t wm[2][7][4];
					std::memcpy(wm, wc, sizeof(wm));
					wm[l][t][e] = static_cast<int8_t>(wm[l][t][e] + delta);
					SolverDecodeFixture fm(wm, kEmbed, kHead, 16384);
					bool diff = false, refusal = false;
					for (int32_t p = 0; p < 3; ++p) {
						DecodeResult r = RunDecode(fm, {p}, 3);
						if (r.status != SslmForwardStatus::Ok) refusal = true;
						if (!r.SameOutputAs(baseline[p])) diff = true;
					}
					if (diff && !refusal) {
						any_diff = true;
						used_delta = delta;
						break;
					}
					if (diff && refusal) any_refusal = true;
				}
				const char* verdict = any_diff ? "MOVES" : (any_refusal ? "REFUSES" : "SILENT");
				if (any_diff) ++moved;
				else if (any_refusal) ++refused;
				else ++silent;
				std::printf("layer%d.%s[%d]: %s%s%s\n", l, kTensorName[t], e, verdict,
				            any_diff ? " delta=" : "",
				            any_diff ? std::to_string(used_delta).c_str() : "");
			}
		}
	}

	// Norm gains, embed, head — same protocol, one representative element set.
	auto probe_gain = [&](const char* name, int32_t* slot) {
		const int32_t orig = *slot;
		bool any_diff = false;
		for (int32_t delta : {4096, -4096, 8192, -8192}) {
			*slot = orig + delta;
			bool diff = false, refusal = false;
			for (int32_t p = 0; p < 3; ++p) {
				DecodeResult r = RunDecode(base, {p}, 3);
				if (r.status != SslmForwardStatus::Ok) refusal = true;
				if (!r.SameOutputAs(baseline[p])) diff = true;
			}
			if (diff && !refusal) { any_diff = true; break; }
		}
		*slot = orig;
		std::printf("%s: %s\n", name, any_diff ? "MOVES" : "SILENT-OR-REFUSES");
	};
	probe_gain("layer0.attn_norm_gain[0]", &base.attn_gain[0][0]);
	probe_gain("layer0.mlp_norm_gain[0]", &base.mlp_gain[0][0]);
	probe_gain("layer1.attn_norm_gain[1]", &base.attn_gain[1][1]);
	probe_gain("layer1.mlp_norm_gain[1]", &base.mlp_gain[1][1]);
	probe_gain("final_norm_gain[0]", &base.final_gain[0]);

	auto probe_i8 = [&](const char* name, int8_t* slot) {
		const int8_t orig = *slot;
		bool any_diff = false;
		for (int8_t delta : {int8_t{1}, int8_t{-1}, int8_t{2}, int8_t{-2}}) {
			*slot = static_cast<int8_t>(orig + delta);
			bool diff = false, refusal = false;
			for (int32_t p = 0; p < 3; ++p) {
				DecodeResult r = RunDecode(base, {p}, 3);
				if (r.status != SslmForwardStatus::Ok) refusal = true;
				if (!r.SameOutputAs(baseline[p])) diff = true;
			}
			if (diff && !refusal) { any_diff = true; break; }
		}
		*slot = orig;
		std::printf("%s: %s\n", name, any_diff ? "MOVES" : "SILENT-OR-REFUSES");
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

	std::printf("\nSUMMARY weight-tensor elements: total=%d MOVES=%d REFUSES=%d SILENT=%d\n",
	            total, moved, refused, silent);
	return 0;
}

// Experiment D: q/k invariance at width=1, plus the direct kernel probe.
int RunQk() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	SolverDecodeFixture base(wc, kEmbed, kHead, 16384);
	DecodeResult baseline[3];
	for (int32_t p = 0; p < 3; ++p) baseline[p] = RunDecode(base, {p}, 3);

	for (int l = 0; l < 2; ++l) {
		for (int t = 0; t < 2; ++t) {  // q, k
			for (int e = 0; e < 4; ++e) {
				for (int8_t delta : {int8_t{1}, int8_t{-1}, int8_t{3}, int8_t{-3}}) {
					int8_t wm[2][7][4];
					std::memcpy(wm, wc, sizeof(wm));
					wm[l][t][e] = static_cast<int8_t>(wm[l][t][e] + delta);
					SolverDecodeFixture fm(wm, kEmbed, kHead, 16384);
					for (int32_t p = 0; p < 3; ++p) {
						DecodeResult r = RunDecode(fm, {p}, 3);
						if (!r.SameOutputAs(baseline[p])) {
							std::printf("layer%d.%s[%d] delta=%d prompt=%d: OUTPUT MOVED "
							            "(status=%s)\n",
							            l, kTensorName[t], e, delta, p,
							            superslm::SslmForwardStatusName(r.status));
						}
					}
				}
			}
		}
	}
	std::printf("qk sweep done (silence above == invariant, as hypothesized)\n");

	// Direct kernel probe: SoftmaxRowQ15 width 1 at several scores.
	const int64_t q_ln2 = INT64_C(2081104), q_b = INT64_C(4062246),
	              q_c = INT64_C(8649804928567);
	int64_t probs_ref = -1;
	for (int64_t score : {INT64_C(0), INT64_C(1), INT64_C(-5), INT64_C(1000), INT64_C(-123456)}) {
		int64_t prob = 0;
		const bool ok = superslm::SoftmaxRowQ15(&score, 1, q_ln2, q_b, q_c, &prob);
		std::printf("SoftmaxRowQ15(width=1, score=%" PRId64 "): ok=%d prob=%" PRId64 "\n", score,
		            ok ? 1 : 0, prob);
		if (probs_ref < 0) probs_ref = prob;
		else if (prob != probs_ref) std::printf("  ^ DIFFERS from first probe!\n");
	}
	return 0;
}

#ifdef SOLVER_TRACE_WIRED
// Experiment E: the trace wiring. (1) hooked run captures per-site records
// with real site names and whole-token indices; (2) hooked and unhooked
// runs produce bit-identical decode output; (3) per-site d_prime/e_out are
// printed — the calibration instrument for the layer-1 blindness.
int RunTrace() {
	int8_t wc[2][7][4];
	CandidateWeights(wc);
	SolverDecodeFixture f(wc, kEmbed, kHead, 16384);
	SolverDecodeFixture f2(wc, kEmbed, kHead, 16384);

	superslm::SslmTraceHookState hook_state{};
	TraceCapture capture;
	superslm::SslmSetTraceHook(hook_state, &TraceCapture::Hook, &capture);

	const std::vector<int32_t> prompt = {2, 0};
	DecodeResult hooked = RunDecode(f, prompt, 2, &hook_state);
	DecodeResult unhooked = RunDecode(f2, prompt, 2);
	PrintResult("hooked  ", hooked);
	PrintResult("unhooked", unhooked);
	std::printf("hooked-vs-unhooked identical=%s\n",
	            hooked.SameOutputAs(unhooked) ? "YES" : "NO");
	std::printf("records=%zu\n", capture.rows.size());
	for (const auto& r : capture.rows) {
		std::printf("  tok=%zu site=%-28s d_prime=%-12" PRId64 " e_out=%" PRId64 "\n",
		            r.token_index, r.site.c_str(), r.d_prime, r.e_out);
	}
	std::printf("kv_landing_records=%zu\n", capture.kv_rows.size());
	for (const auto& r : capture.kv_rows) {
		std::printf("  tok=%zu site=%-28s head=%u m_in=%" PRId64 " e_in=%" PRId64
		            " m_out=%" PRId64 " e_out=%" PRId64 "\n",
		            r.token_index, r.site.c_str(), r.head, r.m_in, r.e_in, r.m_out, r.e_out);
	}
	return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
	const std::string mode = argc > 1 ? argv[1] : "";
	if (mode == "null") return RunNull();
	if (mode == "candidate") return RunCandidate();
	if (mode == "mutate") return RunMutate();
	if (mode == "qk") return RunQk();
#ifdef SOLVER_TRACE_WIRED
	if (mode == "trace") return RunTrace();
#endif
	std::fprintf(stderr, "usage: %s null|candidate|mutate|qk|trace\n", argv[0]);
	return 2;
}
