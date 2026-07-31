// T-1409 Popper probe — falsification pass over the solver's result packet.
//
// DISPOSABLE. Not product code, not a suite test. Independent of
// experiments/decode_discrimination.cpp (the solver's rig): this file rebuilds
// the fixture from the packet's stated §5.1 values and then runs attacks the
// solver's rig does not run.
//
// Modes (argv[1]):
//   control   — N1: the null-delta control the solver's mutation matrix omits.
//               A freshly constructed fixture with UNCHANGED weights must
//               produce decode output bit-identical to the baseline fixture's.
//               If it does not, every "MOVES" verdict is a construction
//               artifact rather than a weight effect.
//   heldout   — N2: the overfit null. Re-run the whole mutation matrix on
//               weight geometries Popper chose, not the solver's, holding the
//               solver's layer-1 calibration boosts (+20 / +12) fixed. If the
//               boosts are a property of the solver's chosen weight VALUES
//               rather than of the geometry class, layer 1 goes silent again.
//   gain      — N3: the calibration's dependence on norm_gain, which the packet
//               holds fixed at 16384.
//   deltas    — N4: refusal-space. Sweep single-element deltas well past the
//               packet's {+-1,+-2} and record where statuses stop being Ok.
//   defects   — N5: the independently-found population. Semantic defect SHAPES
//               (transposed tensor, swapped tensor roles, cross-wired layers,
//               dropped projection) rather than single-element +-1.
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

// Rebuilt from the packet's §5.1 description. Same geometry cell.
struct ProbeFixture {
	superslm::SslmModelView view;
	LayerWeights layers[2];

	int64_t kv_landing_r_t_arr[1];
	int64_t kv_landing_e_t_arr[1] = {0};
	int64_t kv_landing_m_target_arr[1] = {INT64_C(1073741824)};
	int64_t kv_landing_e_target_arr[1] = {0};
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};

	int8_t w[2][7][4];
	int32_t attn_gain[2][2];
	int32_t mlp_gain[2][2];
	int32_t final_gain[2];

	static constexpr int32_t kVocabSize = 3;
	int8_t embed_weights[6];
	int8_t head_weights[6];
	CarriedScale embed_site_constant{INT64_C(1073741824), INT64_C(0)};
	CarriedScale final_norm_site_constant{INT64_C(1073741824), INT64_C(-30)};

	struct Calibration {
		int64_t o_e_boost[2] = {0, 20};
		int64_t down_e_boost[2] = {0, 12};
	};

	ProbeFixture(const int8_t weights[2][7][4], const int8_t embed[6], const int8_t head[6],
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
	ProbeFixture(const ProbeFixture&) = delete;
	ProbeFixture& operator=(const ProbeFixture&) = delete;
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

DecodeResult RunDecode(ProbeFixture& f, const std::vector<int32_t>& prompt, size_t max_new_tokens) {
	DecodeResult r;
	int8_t hidden_codes[2] = {0, 0};
	superslm::SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes;
	uint8_t workspace[64];
	std::memset(workspace, 0xEE, sizeof(workspace));
	r.tokens.assign(max_new_tokens, INT32_C(-99));
	r.logit_rows.assign(max_new_tokens * ProbeFixture::kVocabSize, INT32_C(-99));
	r.status = superslm::RunGreedyDecodeLoop(
	    seq, f.layers, /*num_hidden_layers=*/2, /*hidden_size=*/2, /*head_dim=*/2,
	    /*intermediate_size=*/2, /*context_cap=*/1, f.view.rope_tables, prompt.data(), prompt.size(),
	    f.embed_weights, f.embed_site_constant, f.final_gain, f.final_norm_site_constant,
	    f.head_weights, ProbeFixture::kVocabSize, /*stop_ids=*/nullptr, /*stop_count=*/0,
	    max_new_tokens, workspace, sizeof(workspace), r.tokens.data(), r.logit_rows.data(),
	    r.tokens.size(), &r.produced, &r.stop_reason
#ifdef SOLVER_TRACE_WIRED
	    ,
	    /*site_prefix=*/{}, nullptr
#endif
	);
	return r;
}

void PrintResult(const char* tag, const DecodeResult& r) {
	std::printf("%s: status=%s produced=%zu tokens=[", tag,
	            superslm::SslmForwardStatusName(r.status), r.produced);
	for (size_t i = 0; i < r.produced; ++i) std::printf("%s%d", i ? "," : "", r.tokens[i]);
	std::printf("] logits=[");
	for (size_t i = 0; i < r.produced * 3; ++i) std::printf("%s%d", i ? "," : "", r.logit_rows[i]);
	std::printf("]\n");
}

// --- weight sets --------------------------------------------------------------

// The solver's §5.1 candidate, transcribed from the packet.
void SolverWeights(int8_t w[2][7][4]) {
	static const int8_t l0[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 0, 1}, {1, 0, 1, 1},
	                                {1, 1, 1, 0}, {2, 1, 1, 1}, {1, 1, 0, 1}};
	static const int8_t l1[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 1}, {0, 1, 1, 0}, {1, 1, 1, 0},
	                                {1, 0, 1, 1}, {1, 1, 0, 2}, {2, 1, 1, 1}};
	for (int t = 0; t < 7; ++t) {
		std::memcpy(w[0][t], l0[t], 4);
		std::memcpy(w[1][t], l1[t], 4);
	}
}
const int8_t kSolverEmbed[6] = {5, -3, 7, 2, -4, 6};
const int8_t kSolverHead[6] = {9, -2, -3, 8, 5, 5};

// Popper's held-out geometry A: same class (per-tensor distinct, channel-mixing,
// small ints), values chosen independently of the solver's.
void HeldOutA(int8_t w[2][7][4]) {
	static const int8_t l0[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 2, 1, 1}, {2, 1, 0, 1},
	                                {0, 1, 1, 1}, {1, 0, 1, 2}, {1, 1, 2, 1}};
	static const int8_t l1[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 2, 0}, {0, 1, 1, 1},
	                                {2, 1, 0, 1}, {1, 2, 1, 1}, {1, 0, 1, 2}};
	for (int t = 0; t < 7; ++t) {
		std::memcpy(w[0][t], l0[t], 4);
		std::memcpy(w[1][t], l1[t], 4);
	}
}
const int8_t kHeldOutAEmbed[6] = {2, 7, -5, 3, 6, -1};
const int8_t kHeldOutAHead[6] = {4, 6, -7, 1, 2, -8};

// Held-out geometry B: larger magnitudes, still int8, still channel-mixing.
void HeldOutB(int8_t w[2][7][4]) {
	static const int8_t l0[7][4] = {{1, 0, 0, 1},  {1, 0, 0, 1}, {7, -3, 2, 5}, {-4, 6, 1, 3},
	                                {5, 2, -6, 1}, {3, 7, 2, -5}, {6, -1, 4, 2}};
	static const int8_t l1[7][4] = {{1, 0, 0, 1},  {1, 0, 0, 1}, {-2, 8, 3, 1}, {5, 1, -7, 4},
	                                {2, -6, 5, 3}, {8, 2, 1, -4}, {3, 5, -2, 7}};
	for (int t = 0; t < 7; ++t) {
		std::memcpy(w[0][t], l0[t], 4);
		std::memcpy(w[1][t], l1[t], 4);
	}
}
const int8_t kHeldOutBEmbed[6] = {-11, 13, 9, -6, 15, 4};
const int8_t kHeldOutBHead[6] = {12, -7, 3, 14, -9, 6};

// Held-out geometry C: the minimal off-diagonal perturbation of the identity —
// the smallest departure from the geometry the null was demonstrated on.
void HeldOutC(int8_t w[2][7][4]) {
	static const int8_t id[4] = {1, 0, 0, 1};
	for (int l = 0; l < 2; ++l)
		for (int t = 0; t < 7; ++t) std::memcpy(w[l][t], id, 4);
	w[0][2][1] = 1;  // layer0 v gains one off-diagonal element
	w[1][2][1] = 1;  // layer1 v likewise
}

const char* kTensorName[7] = {"q", "k", "v", "o", "gate", "up", "down"};

struct MatrixResult {
	int moved = 0, silent = 0, refused = 0;
	int l0_moved = 0, l1_moved = 0;  // over v/o/gate/up/down only (20 each)
};

// The solver's Experiment C protocol, verbatim in behavior, parameterised so it
// can run on a geometry the solver did not choose.
MatrixResult RunMatrix(const int8_t wc[2][7][4], const int8_t* embed, const int8_t* head,
                       int32_t gain, ProbeFixture::Calibration cal, bool verbose) {
	MatrixResult res;
	ProbeFixture base(wc, embed, head, gain, cal);
	DecodeResult baseline[3];
	for (int32_t p = 0; p < 3; ++p) {
		baseline[p] = RunDecode(base, {p}, 3);
		if (baseline[p].status != SslmForwardStatus::Ok) {
			std::printf("  BASELINE NOT OK at prompt %d: %s\n", p,
			            superslm::SslmForwardStatusName(baseline[p].status));
			res.refused = -1;
			return res;
		}
	}
	for (int l = 0; l < 2; ++l) {
		for (int t = 0; t < 7; ++t) {
			for (int e = 0; e < 4; ++e) {
				bool any_diff = false, any_refusal = false;
				int used_delta = 0;
				for (int8_t delta : {int8_t{1}, int8_t{-1}, int8_t{2}, int8_t{-2}}) {
					int8_t wm[2][7][4];
					std::memcpy(wm, wc, sizeof(wm));
					wm[l][t][e] = static_cast<int8_t>(wm[l][t][e] + delta);
					ProbeFixture fm(wm, embed, head, gain, cal);
					bool diff = false, refusal = false;
					for (int32_t p = 0; p < 3; ++p) {
						DecodeResult r = RunDecode(fm, {p}, 3);
						if (r.status != SslmForwardStatus::Ok) refusal = true;
						if (!r.SameOutputAs(baseline[p])) diff = true;
					}
					if (diff && !refusal) { any_diff = true; used_delta = delta; break; }
					if (diff && refusal) any_refusal = true;
				}
				const char* verdict = any_diff ? "MOVES" : (any_refusal ? "REFUSES" : "SILENT");
				if (any_diff) {
					++res.moved;
					if (t >= 2) { (l == 0 ? res.l0_moved : res.l1_moved) += 1; }
				} else if (any_refusal) ++res.refused;
				else ++res.silent;
				if (verbose)
					std::printf("  layer%d.%s[%d]: %s%s%d\n", l, kTensorName[t], e, verdict,
					            any_diff ? " delta=" : "", any_diff ? used_delta : 0);
			}
		}
	}
	return res;
}

// --- N1: the null-delta control ------------------------------------------------

int RunControl() {
	int8_t wc[2][7][4];
	SolverWeights(wc);
	ProbeFixture base(wc, kSolverEmbed, kSolverHead, 16384);
	DecodeResult baseline[3];
	for (int32_t p = 0; p < 3; ++p) baseline[p] = RunDecode(base, {p}, 3);

	// (a) Same fixture object, re-run: run-to-run stability.
	int rerun_diffs = 0;
	for (int rep = 0; rep < 8; ++rep)
		for (int32_t p = 0; p < 3; ++p)
			if (!RunDecode(base, {p}, 3).SameOutputAs(baseline[p])) ++rerun_diffs;
	std::printf("N1a same-object re-runs (24): differing=%d\n", rerun_diffs);

	// (b) Freshly constructed fixtures with UNCHANGED weights — the delta=0
	// control the mutation matrix never runs. This is the arm that decides
	// whether "MOVES" is a weight effect or a construction artifact.
	int fresh_diffs = 0;
	for (int rep = 0; rep < 8; ++rep) {
		ProbeFixture f0(wc, kSolverEmbed, kSolverHead, 16384);
		for (int32_t p = 0; p < 3; ++p)
			if (!RunDecode(f0, {p}, 3).SameOutputAs(baseline[p])) ++fresh_diffs;
	}
	std::printf("N1b fresh-fixture delta=0 runs (24): differing=%d\n", fresh_diffs);

	// (c) Delta=0 applied through the mutation loop's own code path (memcpy of
	// the weight block, then a fresh fixture), for all 56 elements.
	int zero_delta_moves = 0;
	for (int l = 0; l < 2; ++l)
		for (int t = 0; t < 7; ++t)
			for (int e = 0; e < 4; ++e) {
				int8_t wm[2][7][4];
				std::memcpy(wm, wc, sizeof(wm));
				wm[l][t][e] = static_cast<int8_t>(wm[l][t][e] + 0);
				ProbeFixture fm(wm, kSolverEmbed, kSolverHead, 16384);
				for (int32_t p = 0; p < 3; ++p)
					if (!RunDecode(fm, {p}, 3).SameOutputAs(baseline[p])) ++zero_delta_moves;
			}
	std::printf("N1c delta=0 through the mutation path (56 elements x 3 prompts = 168): "
	            "differing=%d\n",
	            zero_delta_moves);

	// (d) Heap-state perturbation: allocate and free between constructions, so a
	// fresh fixture lands at a different address. Address-dependent output would
	// show here.
	int addr_diffs = 0;
	for (int rep = 0; rep < 6; ++rep) {
		std::vector<uint8_t> churn(static_cast<size_t>(1 + rep) * 4096, 0xA5);
		ProbeFixture* fp = new ProbeFixture(wc, kSolverEmbed, kSolverHead, 16384);
		for (int32_t p = 0; p < 3; ++p)
			if (!RunDecode(*fp, {p}, 3).SameOutputAs(baseline[p])) ++addr_diffs;
		delete fp;
		churn.clear();
	}
	std::printf("N1d heap-perturbed fresh fixtures (18): differing=%d\n", addr_diffs);
	return 0;
}

// --- N2: the overfit null ------------------------------------------------------

void ReportMatrix(const char* label, const MatrixResult& r) {
	std::printf("%-46s MOVES=%d SILENT=%d REFUSES=%d  (layer0 v/o/g/u/d=%d/20, "
	            "layer1 v/o/g/u/d=%d/20)\n",
	            label, r.moved, r.silent, r.refused, r.l0_moved, r.l1_moved);
}

int RunHeldOut() {
	int8_t wc[2][7][4];
	ProbeFixture::Calibration shipped{};  // {0,20} / {0,12}

	SolverWeights(wc);
	ReportMatrix("solver geometry (tuned-on), shipped cal:",
	             RunMatrix(wc, kSolverEmbed, kSolverHead, 16384, shipped, false));

	HeldOutA(wc);
	ReportMatrix("held-out A (small mixing), shipped cal:",
	             RunMatrix(wc, kHeldOutAEmbed, kHeldOutAHead, 16384, shipped, false));

	HeldOutB(wc);
	ReportMatrix("held-out B (large mixing), shipped cal:",
	             RunMatrix(wc, kHeldOutBEmbed, kHeldOutBHead, 16384, shipped, false));

	HeldOutC(wc);
	ReportMatrix("held-out C (near-identity), shipped cal:",
	             RunMatrix(wc, kSolverEmbed, kSolverHead, 16384, shipped, false));

	// The same geometries with the calibration REMOVED, to establish that the
	// boosts are what carry layer 1 (rather than the geometry alone).
	ProbeFixture::Calibration none{};
	none.o_e_boost[1] = 0;
	none.down_e_boost[1] = 0;
	SolverWeights(wc);
	ReportMatrix("solver geometry, NO layer-1 calibration:",
	             RunMatrix(wc, kSolverEmbed, kSolverHead, 16384, none, false));
	HeldOutA(wc);
	ReportMatrix("held-out A, NO layer-1 calibration:",
	             RunMatrix(wc, kHeldOutAEmbed, kHeldOutAHead, 16384, none, false));

	// And the boosts perturbed off their derived values, to measure how wide the
	// working window is.
	for (int64_t o_boost : {int64_t{14}, int64_t{17}, int64_t{20}, int64_t{23}, int64_t{26}}) {
		for (int64_t d_boost : {int64_t{6}, int64_t{12}, int64_t{18}}) {
			ProbeFixture::Calibration cal{};
			cal.o_e_boost[1] = o_boost;
			cal.down_e_boost[1] = d_boost;
			SolverWeights(wc);
			char label[96];
			std::snprintf(label, sizeof(label), "solver geometry, o_boost=%" PRId64
			              " down_boost=%" PRId64 ":", o_boost, d_boost);
			ReportMatrix(label, RunMatrix(wc, kSolverEmbed, kSolverHead, 16384, cal, false));
		}
	}
	return 0;
}

// --- N3: the gain axis ---------------------------------------------------------

int RunGain() {
	int8_t wc[2][7][4];
	SolverWeights(wc);
	ProbeFixture::Calibration shipped{};
	for (int32_t gain : {1024, 4096, 8192, 16384, 32768, 65536}) {
		char label[96];
		std::snprintf(label, sizeof(label), "solver geometry, norm_gain=%d:", gain);
		ReportMatrix(label, RunMatrix(wc, kSolverEmbed, kSolverHead, gain, shipped, false));
	}
	return 0;
}

// --- N4: refusal space ---------------------------------------------------------

int RunDeltas() {
	int8_t wc[2][7][4];
	SolverWeights(wc);
	ProbeFixture base(wc, kSolverEmbed, kSolverHead, 16384);
	DecodeResult baseline[3];
	for (int32_t p = 0; p < 3; ++p) baseline[p] = RunDecode(base, {p}, 3);

	std::printf("Per-element sweep over deltas {+-1,+-2,+-4,+-8,+-16,+-32,+-64,+-127}; "
	            "reporting counts over the 56 elements.\n");
	const int deltas[] = {1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 127, -127};
	for (int d : deltas) {
		int moves = 0, silent = 0, refuse = 0, sat = 0;
		for (int l = 0; l < 2; ++l)
			for (int t = 0; t < 7; ++t)
				for (int e = 0; e < 4; ++e) {
					const int nv = static_cast<int>(wc[l][t][e]) + d;
					if (nv > 127 || nv < -128) { ++sat; continue; }
					int8_t wm[2][7][4];
					std::memcpy(wm, wc, sizeof(wm));
					wm[l][t][e] = static_cast<int8_t>(nv);
					ProbeFixture fm(wm, kSolverEmbed, kSolverHead, 16384);
					bool diff = false, refusal = false;
					for (int32_t p = 0; p < 3; ++p) {
						DecodeResult r = RunDecode(fm, {p}, 3);
						if (r.status != SslmForwardStatus::Ok) refusal = true;
						if (!r.SameOutputAs(baseline[p])) diff = true;
					}
					if (refusal) ++refuse;
					else if (diff) ++moves;
					else ++silent;
				}
		std::printf("  delta=%+5d : MOVES(Ok)=%2d SILENT=%2d REFUSES=%2d out-of-int8=%2d\n", d,
		            moves, silent, refuse, sat);
	}
	return 0;
}

// --- N5: the independently-found defect population -----------------------------

struct Defect {
	const char* name;
	void (*apply)(int8_t w[2][7][4]);
};

void D_TransposeL0V(int8_t w[2][7][4]) { std::swap(w[0][2][1], w[0][2][2]); }
void D_TransposeL0O(int8_t w[2][7][4]) { std::swap(w[0][3][1], w[0][3][2]); }
void D_TransposeL0Down(int8_t w[2][7][4]) { std::swap(w[0][6][1], w[0][6][2]); }
void D_TransposeL1Gate(int8_t w[2][7][4]) { std::swap(w[1][4][1], w[1][4][2]); }
void D_TransposeL1Up(int8_t w[2][7][4]) { std::swap(w[1][5][1], w[1][5][2]); }
void D_SwapGateUpL0(int8_t w[2][7][4]) { for (int i = 0; i < 4; ++i) std::swap(w[0][4][i], w[0][5][i]); }
void D_SwapGateUpL1(int8_t w[2][7][4]) { for (int i = 0; i < 4; ++i) std::swap(w[1][4][i], w[1][5][i]); }
void D_SwapVOL0(int8_t w[2][7][4]) { for (int i = 0; i < 4; ++i) std::swap(w[0][2][i], w[0][3][i]); }
void D_SwapVOL1(int8_t w[2][7][4]) { for (int i = 0; i < 4; ++i) std::swap(w[1][2][i], w[1][3][i]); }
void D_SwapQKL0(int8_t w[2][7][4]) { for (int i = 0; i < 4; ++i) std::swap(w[0][0][i], w[0][1][i]); }
void D_SwapQKBoth(int8_t w[2][7][4]) {
	for (int l = 0; l < 2; ++l)
		for (int i = 0; i < 4; ++i) std::swap(w[l][0][i], w[l][1][i]);
}
void D_SwapLayers(int8_t w[2][7][4]) {
	for (int t = 0; t < 7; ++t)
		for (int i = 0; i < 4; ++i) std::swap(w[0][t][i], w[1][t][i]);
}
void D_L0WeightsAtL1(int8_t w[2][7][4]) { std::memcpy(w[1], w[0], sizeof(w[0])); }
void D_ZeroL0V(int8_t w[2][7][4]) { std::memset(w[0][2], 0, 4); }
void D_ZeroL0O(int8_t w[2][7][4]) { std::memset(w[0][3], 0, 4); }
void D_ZeroL1Down(int8_t w[2][7][4]) { std::memset(w[1][6], 0, 4); }
void D_ZeroL1Gate(int8_t w[2][7][4]) { std::memset(w[1][4], 0, 4); }
void D_IdentityL0V(int8_t w[2][7][4]) { const int8_t id[4] = {1, 0, 0, 1}; std::memcpy(w[0][2], id, 4); }
void D_IdentityL1Down(int8_t w[2][7][4]) { const int8_t id[4] = {1, 0, 0, 1}; std::memcpy(w[1][6], id, 4); }
void D_ZeroQKBoth(int8_t w[2][7][4]) {
	for (int l = 0; l < 2; ++l) { std::memset(w[l][0], 0, 4); std::memset(w[l][1], 0, 4); }
}

int RunDefects() {
	const Defect defects[] = {
	    {"transpose layer0.v", D_TransposeL0V},
	    {"transpose layer0.o", D_TransposeL0O},
	    {"transpose layer0.down", D_TransposeL0Down},
	    {"transpose layer1.gate", D_TransposeL1Gate},
	    {"transpose layer1.up", D_TransposeL1Up},
	    {"swap gate<->up, layer0", D_SwapGateUpL0},
	    {"swap gate<->up, layer1", D_SwapGateUpL1},
	    {"swap v<->o, layer0", D_SwapVOL0},
	    {"swap v<->o, layer1", D_SwapVOL1},
	    {"swap q<->k, layer0", D_SwapQKL0},
	    {"swap q<->k, both layers", D_SwapQKBoth},
	    {"zero q and k, both layers", D_ZeroQKBoth},
	    {"swap layer0 <-> layer1 entirely", D_SwapLayers},
	    {"layer0 weights reused at layer1", D_L0WeightsAtL1},
	    {"drop layer0.v (zeroed)", D_ZeroL0V},
	    {"drop layer0.o (zeroed)", D_ZeroL0O},
	    {"drop layer1.down (zeroed)", D_ZeroL1Down},
	    {"drop layer1.gate (zeroed)", D_ZeroL1Gate},
	    {"layer0.v collapsed to identity", D_IdentityL0V},
	    {"layer1.down collapsed to identity", D_IdentityL1Down},
	};

	int8_t wc[2][7][4];
	SolverWeights(wc);
	ProbeFixture base(wc, kSolverEmbed, kSolverHead, 16384);
	DecodeResult baseline[3];
	for (int32_t p = 0; p < 3; ++p) baseline[p] = RunDecode(base, {p}, 3);

	int caught = 0, missed = 0, refused = 0;
	for (const Defect& d : defects) {
		int8_t wm[2][7][4];
		std::memcpy(wm, wc, sizeof(wm));
		d.apply(wm);
		if (std::memcmp(wm, wc, sizeof(wm)) == 0) {
			std::printf("  %-38s NO-OP ON THIS GEOMETRY (defect not expressible)\n", d.name);
			continue;
		}
		ProbeFixture fm(wm, kSolverEmbed, kSolverHead, 16384);
		bool diff = false, refusal = false;
		for (int32_t p = 0; p < 3; ++p) {
			DecodeResult r = RunDecode(fm, {p}, 3);
			if (r.status != SslmForwardStatus::Ok) refusal = true;
			if (!r.SameOutputAs(baseline[p])) diff = true;
		}
		const char* v = refusal ? "REFUSES" : (diff ? "CAUGHT" : "MISSED (output identical)");
		if (refusal) ++refused; else if (diff) ++caught; else ++missed;
		std::printf("  %-38s %s\n", d.name, v);
	}
	std::printf("\nDefect-shape population: CAUGHT=%d MISSED=%d REFUSES=%d\n", caught, missed,
	            refused);
	return 0;
}

// --- N6: the A-vs-B confound ---------------------------------------------------
//
// The packet's Experiment A runs identity layer weights with BASIS embed/head;
// Experiment B runs mixing layer weights with NON-BASIS embed/head. Two
// variables move between them, so "geometry closes the gap" is not isolated by
// that pair. This mode runs the missing two cells of the 2x2.

void IdentityWeights(int8_t w[2][7][4]) {
	static const int8_t id[4] = {1, 0, 0, 1};
	for (int l = 0; l < 2; ++l)
		for (int t = 0; t < 7; ++t) std::memcpy(w[l][t], id, 4);
}
void ZeroWeights(int8_t w[2][7][4]) { std::memset(w, 0, 2 * 7 * 4); }
const int8_t kBasisEmbed[6] = {1, 0, 0, 1, 1, 1};
const int8_t kBasisHead[6] = {1, 0, 0, 1, 1, 1};

// Does identity-vs-zeromap stay bit-identical (the D-SLM493 null) for this
// embed/head choice? "identical" == the null still holds == layer weights
// invisible.
void NullCell(const char* label, void (*wf)(int8_t[2][7][4]), const int8_t* embed,
              const int8_t* head, ProbeFixture::Calibration cal) {
	int8_t wa[2][7][4], wz[2][7][4];
	wf(wa);
	ZeroWeights(wz);
	ProbeFixture fa(wa, embed, head, 16384, cal);
	ProbeFixture fz(wz, embed, head, 16384, cal);
	bool all_same = true;
	bool ok = true;
	for (int32_t p = 0; p < 3; ++p) {
		DecodeResult a = RunDecode(fa, {p}, 3);
		DecodeResult b = RunDecode(fz, {p}, 3);
		if (a.status != SslmForwardStatus::Ok || b.status != SslmForwardStatus::Ok) ok = false;
		if (!a.SameOutputAs(b)) all_same = false;
	}
	std::printf("%-58s statuses=%s  weights-invisible(null holds)=%s\n", label, ok ? "Ok" : "NOT-Ok",
	            all_same ? "YES" : "NO");
}

int RunConfound() {
	ProbeFixture::Calibration shipped{};
	ProbeFixture::Calibration none{};
	none.o_e_boost[1] = 0;
	none.down_e_boost[1] = 0;

	std::printf("The 2x2 the packet's A/B pair does not span (weights x embed/head):\n");
	NullCell("A  : identity weights, BASIS embed/head (packet Exp A)", IdentityWeights, kBasisEmbed,
	         kBasisHead, none);
	NullCell("A' : identity weights, SOLVER embed/head (missing cell)", IdentityWeights,
	         kSolverEmbed, kSolverHead, none);
	NullCell("A'': identity weights, SOLVER embed/head, WITH calibration", IdentityWeights,
	         kSolverEmbed, kSolverHead, shipped);
	NullCell("B' : mixing weights, BASIS embed/head (missing cell)", SolverWeights, kBasisEmbed,
	         kBasisHead, shipped);
	NullCell("B  : mixing weights, SOLVER embed/head (packet Exp B)", SolverWeights, kSolverEmbed,
	         kSolverHead, shipped);
	NullCell("C  : near-identity (one off-diagonal in v), BASIS embed/head", HeldOutC, kBasisEmbed,
	         kBasisHead, shipped);

	std::printf("\nFull mutation matrix at the two cells that isolate the variable:\n");
	int8_t w[2][7][4];
	SolverWeights(w);
	ReportMatrix("  mixing weights + BASIS embed/head:",
	             RunMatrix(w, kBasisEmbed, kBasisHead, 16384, shipped, false));
	HeldOutC(w);
	ReportMatrix("  near-identity weights + BASIS embed/head:",
	             RunMatrix(w, kBasisEmbed, kBasisHead, 16384, shipped, false));

	// Is the mixing-weight geometry NECESSARY? Run the matrix on the SHIPPED
	// identity weights with only embed/head changed.
	std::printf("\nIs the mixing geometry necessary? Identity layer weights throughout:\n");
	IdentityWeights(w);
	ReportMatrix("  identity weights + SOLVER embed/head, no cal:",
	             RunMatrix(w, kSolverEmbed, kSolverHead, 16384, none, false));
	IdentityWeights(w);
	ReportMatrix("  identity weights + SOLVER embed/head, shipped cal:",
	             RunMatrix(w, kSolverEmbed, kSolverHead, 16384, shipped, false));
	IdentityWeights(w);
	ReportMatrix("  identity weights + BASIS embed/head, shipped cal:",
	             RunMatrix(w, kBasisEmbed, kBasisHead, 16384, shipped, false));
	IdentityWeights(w);
	ReportMatrix("  identity weights + held-out-A embed/head, shipped cal:",
	             RunMatrix(w, kHeldOutAEmbed, kHeldOutAHead, 16384, shipped, false));
	return 0;
}

// --- N7: the counter-construction ----------------------------------------------
//
// The packet's §2 states: "No calibration of site constants or gains alone can
// fix this while the weights remain diagonal." This mode executes that claim on
// the SHIPPED DecodeLoopFixture geometry (identity weights everywhere, basis
// embed/head, gain 16384) with ONLY the two layer-1 site-constant exponent
// boosts added — no geometry change at all.

int RunShipped() {
	ProbeFixture::Calibration shipped{};
	ProbeFixture::Calibration none{};
	none.o_e_boost[1] = 0;
	none.down_e_boost[1] = 0;
	int8_t wid[2][7][4], wz[2][7][4];
	IdentityWeights(wid);
	ZeroWeights(wz);

	std::printf("--- shipped geometry (identity weights, basis embed/head, gain 16384) ---\n");
	std::printf("(a) WITHOUT the calibration — the packet's Experiment A:\n");
	{
		ProbeFixture fa(wid, kBasisEmbed, kBasisHead, 16384, none);
		ProbeFixture fz(wz, kBasisEmbed, kBasisHead, 16384, none);
		for (int32_t p = 0; p < 3; ++p) {
			DecodeResult a = RunDecode(fa, {p}, 3), b = RunDecode(fz, {p}, 3);
			char t1[48], t2[48];
			std::snprintf(t1, sizeof(t1), "  identity p=%d", p);
			std::snprintf(t2, sizeof(t2), "  zero-map p=%d", p);
			PrintResult(t1, a);
			PrintResult(t2, b);
			std::printf("  -> identical=%s\n", a.SameOutputAs(b) ? "YES" : "NO");
		}
	}

	std::printf("\n(b) WITH the calibration only (o_e +20, down_e +12 on layer 1); "
	            "weights untouched:\n");
	{
		ProbeFixture fa(wid, kBasisEmbed, kBasisHead, 16384, shipped);
		ProbeFixture fz(wz, kBasisEmbed, kBasisHead, 16384, shipped);
		for (int32_t p = 0; p < 3; ++p) {
			DecodeResult a = RunDecode(fa, {p}, 3), b = RunDecode(fz, {p}, 3);
			char t1[48], t2[48];
			std::snprintf(t1, sizeof(t1), "  identity p=%d", p);
			std::snprintf(t2, sizeof(t2), "  zero-map p=%d", p);
			PrintResult(t1, a);
			PrintResult(t2, b);
			std::printf("  -> identical=%s\n", a.SameOutputAs(b) ? "YES" : "NO");
		}
	}

	std::printf("\n(c) delta=0 control at this cell (56 elements x 3 prompts = 168):\n");
	{
		ProbeFixture base(wid, kBasisEmbed, kBasisHead, 16384, shipped);
		DecodeResult baseline[3];
		for (int32_t p = 0; p < 3; ++p) baseline[p] = RunDecode(base, {p}, 3);
		int diffs = 0;
		for (int l = 0; l < 2; ++l)
			for (int t = 0; t < 7; ++t)
				for (int e = 0; e < 4; ++e) {
					int8_t wm[2][7][4];
					std::memcpy(wm, wid, sizeof(wm));
					ProbeFixture fm(wm, kBasisEmbed, kBasisHead, 16384, shipped);
					for (int32_t p = 0; p < 3; ++p)
						if (!RunDecode(fm, {p}, 3).SameOutputAs(baseline[p])) ++diffs;
				}
		std::printf("  differing=%d\n", diffs);
	}

	std::printf("\n(d) the full mutation matrix at this cell:\n");
	IdentityWeights(wid);
	MatrixResult r = RunMatrix(wid, kBasisEmbed, kBasisHead, 16384, shipped, true);
	ReportMatrix("  TOTAL:", r);
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	const std::string mode = argc > 1 ? argv[1] : "";
	if (mode == "control") return RunControl();
	if (mode == "confound") return RunConfound();
	if (mode == "shipped") return RunShipped();
	if (mode == "heldout") return RunHeldOut();
	if (mode == "gain") return RunGain();
	if (mode == "deltas") return RunDeltas();
	if (mode == "defects") return RunDefects();
	std::fprintf(stderr, "usage: %s control|heldout|gain|deltas|defects\n", argv[0]);
	return 2;
}
