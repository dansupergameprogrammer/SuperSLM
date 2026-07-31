// T-1421 probe: what does the pinned golden decode row actually discriminate?
//
// The committed suite pins DecodeLoopFixture's decode output to
// kDecodeLoopExpectedLogits = {127,0,127}/{0,127,127}/{127,127,254}. This
// probe reproduces that geometry through the real RunGreedyDecodeLoop and
// then asks, per fixture-level ablation, whether the pinned row moves. It
// writes nothing to product code.
//
// Modes:
//   pin      -- reproduce the pinned rows and run the ablation table.
//   scale    -- pure-magnitude family: every projection scaled by k.
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

struct PinFixture {
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

	PinFixture(const int8_t weights[2][7][4], const int8_t embed[6], const int8_t head[6],
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
			std::fprintf(stderr, "FATAL: fixture artifact failed to load\n");
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
			lw.o_site_constant = canonical;
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
			lw.down_site_constant = canonical;
			lw.mlp_residual_site_constant = canonical;
		}
	}
	PinFixture(const PinFixture&) = delete;
	PinFixture& operator=(const PinFixture&) = delete;
};

struct DecodeResult {
	SslmForwardStatus status = SslmForwardStatus::Ok;
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;
	size_t produced = 0;
	superslm::SslmDecodeStopReason stop_reason{};
};

DecodeResult RunDecode(PinFixture& f, const std::vector<int32_t>& prompt, size_t max_new) {
	DecodeResult r;
	int8_t hidden_codes[2] = {0, 0};
	superslm::SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes;
	uint8_t workspace[64];
	std::memset(workspace, 0xEE, sizeof(workspace));
	r.tokens.assign(max_new, INT32_C(-99));
	r.logit_rows.assign(max_new * PinFixture::kVocabSize, INT32_C(-99));
	r.status = superslm::RunGreedyDecodeLoop(
	    seq, f.layers, 2, 2, 2, 2, 1, f.view.rope_tables, prompt.data(), prompt.size(),
	    f.embed_weights, f.embed_site_constant, f.final_gain, f.final_norm_site_constant,
	    f.head_weights, PinFixture::kVocabSize, nullptr, 0, max_new, workspace, sizeof(workspace),
	    r.tokens.data(), r.logit_rows.data(), r.tokens.size(), &r.produced, &r.stop_reason
#ifdef SOLVER_TRACE_WIRED
	    ,
	    /*site_prefix=*/{}, nullptr
#endif
	);
	return r;
}

// The committed pin, verbatim from tests/test_main.cpp.
const int32_t kPinnedLogits[3][3] = {{127, 0, 127}, {0, 127, 127}, {127, 127, 254}};

const int8_t kBasisEmbed[6] = {1, 0, 0, 1, 1, 1};
const int8_t kBasisHead[6] = {1, 0, 0, 1, 1, 1};

void Identity(int8_t w[2][7][4]) {
	static const int8_t id[4] = {1, 0, 0, 1};
	for (int l = 0; l < 2; ++l)
		for (int t = 0; t < 7; ++t) std::memcpy(w[l][t], id, 4);
}

// Runs the pin's three vocabulary entries and reports whether the FIRST decode
// step's logit row equals the pinned row for each.
bool PinHolds(PinFixture& f, bool verbose, const char* tag) {
	bool all = true;
	for (int32_t p = 0; p < 3; ++p) {
		DecodeResult r = RunDecode(f, {p}, 1);
		bool ok = r.status == SslmForwardStatus::Ok && r.produced == 1;
		if (ok) {
			for (int v = 0; v < 3; ++v)
				if (r.logit_rows[v] != kPinnedLogits[p][v]) ok = false;
		}
		if (!ok) all = false;
		if (verbose) {
			std::printf("  %s entry=%d status=%s row=[", tag, p,
			            superslm::SslmForwardStatusName(r.status));
			for (int v = 0; v < 3; ++v) std::printf("%s%d", v ? "," : "", r.logit_rows[v]);
			std::printf("] pinned=[%d,%d,%d] %s\n", kPinnedLogits[p][0], kPinnedLogits[p][1],
			            kPinnedLogits[p][2], ok ? "MATCH" : "MOVED");
		}
	}
	return all;
}

void Row(const char* name, int8_t w[2][7][4], const int8_t* embed, const int8_t* head,
         int32_t gain) {
	PinFixture f(w, embed, head, gain);
	const bool held = PinHolds(f, false, name);
	std::printf("%-52s pin %s\n", name, held ? "HOLDS  (blind)" : "MOVES  (discriminates)");
}

int RunPin() {
	int8_t w[2][7][4];
	Identity(w);
	std::printf("=== the pinned row, reproduced through RunGreedyDecodeLoop ===\n");
	{
		PinFixture f(w, kBasisEmbed, kBasisHead, 16384);
		PinHolds(f, true, "identity");
	}

	std::printf("\n=== fixture-level ablations: does the pinned row move? ===\n");
	const char* tname[7] = {"q", "k", "v", "o", "gate", "up", "down"};

	Identity(w);
	Row("baseline identity everywhere", w, kBasisEmbed, kBasisHead, 16384);

	// Whole stack ablated.
	Identity(w);
	std::memset(w, 0, sizeof(w));
	Row("ALL layer weights zeroed (whole stack ablated)", w, kBasisEmbed, kBasisHead, 16384);

	// Per-tensor zeroing, both layers.
	for (int t = 0; t < 7; ++t) {
		Identity(w);
		for (int l = 0; l < 2; ++l) std::memset(w[l][t], 0, 4);
		char nm[80];
		std::snprintf(nm, sizeof(nm), "zeroed %s in both layers", tname[t]);
		Row(nm, w, kBasisEmbed, kBasisHead, 16384);
	}

	// Whole layer ablated.
	for (int l = 0; l < 2; ++l) {
		Identity(w);
		std::memset(w[l], 0, sizeof(w[l]));
		char nm[80];
		std::snprintf(nm, sizeof(nm), "layer %d zeroed entirely", l);
		Row(nm, w, kBasisEmbed, kBasisHead, 16384);
	}

	// Pure magnitude: every projection scaled by k (direction preserved).
	for (int k : {2, 3, 5}) {
		Identity(w);
		for (int l = 0; l < 2; ++l)
			for (int t = 0; t < 7; ++t)
				for (int e = 0; e < 4; ++e) w[l][t][e] = static_cast<int8_t>(w[l][t][e] * k);
		char nm[80];
		std::snprintf(nm, sizeof(nm), "all projections = %dx identity (pure magnitude)", k);
		Row(nm, w, kBasisEmbed, kBasisHead, 16384);
	}

	// Direction change: channel swap in one projection.
	for (int t = 2; t < 7; ++t) {
		Identity(w);
		const int8_t swap[4] = {0, 1, 1, 0};
		std::memcpy(w[0][t], swap, 4);
		char nm[80];
		std::snprintf(nm, sizeof(nm), "layer0 %s = channel swap (direction change)", tname[t]);
		Row(nm, w, kBasisEmbed, kBasisHead, 16384);
	}

	// Shear: one off-diagonal element.
	for (int t = 2; t < 7; ++t) {
		Identity(w);
		w[0][t][1] = 1;
		char nm[80];
		std::snprintf(nm, sizeof(nm), "layer0 %s += one off-diagonal 1 (shear)", tname[t]);
		Row(nm, w, kBasisEmbed, kBasisHead, 16384);
	}

	// Norm gains.
	for (int32_t g : {1024, 4096, 8192, 32768}) {
		Identity(w);
		char nm[80];
		std::snprintf(nm, sizeof(nm), "all norm gains = %d", g);
		Row(nm, w, kBasisEmbed, kBasisHead, g);
	}

	// Embed / head perturbation.
	{
		Identity(w);
		int8_t e2[6];
		std::memcpy(e2, kBasisEmbed, 6);
		e2[0] = 2;
		Row("embed[0] 1 -> 2", w, e2, kBasisHead, 16384);
		int8_t h2[6];
		std::memcpy(h2, kBasisHead, 6);
		h2[0] = 2;
		Row("head[0] 1 -> 2", w, kBasisEmbed, h2, 16384);
	}
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	const std::string mode = argc > 1 ? argv[1] : "pin";
	if (mode == "pin") return RunPin();
	std::fprintf(stderr, "usage: %s pin\n", argv[0]);
	return 2;
}
