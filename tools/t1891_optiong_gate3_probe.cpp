// T-1891 gate G3 -- reference parity, the ENGINE-SIDE half.
//
// DISPOSABLE. Branch brunel/t1891-optionG-spike only, never merged.
//
// Builds a small, hermetic, 2-layer GQA fixture through the REAL production load path
// (BuildArtifact -> SslmModel::Load, the same construction test_main.cpp's own
// GqaGroupingFixture uses and this codebase's 24042-check suite already exercises) --
// same geometry and same weight/scale/bias values as GqaGroupingFixture, so every value
// here is already proven to clear every domain check RunLayerLoop enforces. The ONE
// axis this fixture changes from that proven baseline is the ROPE TABLE: real,
// non-degenerate Q2.30 cos/sin values (GqaGroupingFixture's own tables are the trivial
// identity rotation, cos=1/sin=0 at every position, which would bit-match under Option G
// for a reason that has nothing to do with the rotation arithmetic actually agreeing).
//
// Runs `RunGreedyDecodeLoop` (the same production entry point sslm_generate.cpp calls)
// over a small set of REAL-ENGLISH-TEXT documents, byte-tokenized into this fixture's
// small vocabulary (`byte % vocab_size`, the IDENTICAL scheme
// `superslm_spike.pipeline._fixture_tokenize_prompt` already uses on the Python
// reference side -- one convention, not two independently-authored ones), with Option G
// on (`SSLM_OPTION_G_FUSED_K_LANDING=1`), and dumps the resulting K/V store for
// `tools/t1891_optiong_gate3_check.py` to compare against the mirrored Python reference
// (`tests/reference/superslm_spike/dynamic_engine.py`,
// `option_g_fused_k_landing=True`) run on the SAME documents.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/option_g_spike.h"

#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_fixtures.h"
#include "sslm_model_hostile_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

using namespace superslm;
using namespace superslm_test;

namespace {

// --- Fixture geometry -- IDENTICAL to test_main.cpp's own GqaGroupingFixture --------
constexpr uint32_t kHiddenSize = 8;
constexpr uint32_t kNumLayers = 2;
constexpr uint32_t kNumHeads = 4;
constexpr uint32_t kNumKvHeads = 2;
constexpr uint32_t kHeadDim = 2;
constexpr uint32_t kIntermediateSize = 8;
constexpr int32_t kContextCap = 16;
constexpr uint32_t kVocabSize = 32;  // small, tied embedding == head

// T-1891's own non-degenerate rope table: cos(position)/sin(position) at Q2.30 (inv_freq
// = theta^0 = 1.0 for pair 0 of a head_dim=2 head), computed once in Python and pasted
// here verbatim -- both this array and the Python reference's OWN literal copy
// (tools/t1891_optiong_gate3_reference.py) are transcriptions of the SAME computation,
// not two independent derivations, which is deliberate: G3 is testing whether the
// ENGINE's fused rotation agrees with the REFERENCE's fused rotation over the SAME
// table, not whether two table-generation methods agree with each other (that is G2's
// job, over the primitive alone, already covered by an independent Python reference).
const int64_t kCosQ30[kContextCap] = {
    1073741824, 580145183, -446834263, -1062996349, -701844494, 304579952,
    1030974995, 809496382, -156229472, -978318669, -900946194, 4752057,
    906081289, 974363562, 146820470, -815708685,
};
const int64_t kSinQ30[kContextCap] = {
    0, 903522590, 976350678, 151526455, -812610492, -1029637100,
    -300020107, 705433989, 1062315328, 442508854, -584138220, -1073731308,
    -576140784, 451150921, 1063656549, 698241252,
};

// 256, not GqaGroupingFixture's own 16384 -- T-1891's own reference build
// (tools/t1891_optiong_gate3_reference.py) found 16384 drove the calibration-derived
// softmax_khead scale fine enough to trip C30's own coarse-scale rejection
// (i_exp requires a positive ln2 quantum; got q_ln2=0). 256 is this fixture's OWN
// value, not GqaGroupingFixture's -- transcribed here from that script's own printed
// gain_array, and every CarriedScale/kv_landing constant below is transcribed from
// the SAME script run at this SAME gain value.
const int32_t kNormGain[kHiddenSize] = {256, 256, 256, 256, 256, 256, 256, 256};
const int8_t kIdentity8x8[64] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
};
const int8_t kKSelector4x8[32] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
};
const int8_t kVSelector4x8[32] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
};
const int32_t kIdentityFold[kHiddenSize] = {1, 1, 1, 1, 1, 1, 1, 1};
const int32_t kZeroFold[kHiddenSize] = {0, 0, 0, 0, 0, 0, 0, 0};
const int32_t kCtxFoldIdentity[kNumHeads] = {1, 1, 1, 1};
const int32_t kCtxFoldZero[kNumHeads] = {0, 0, 0, 0};

// T-1891's own embed/head table (GqaGroupingFixture has none -- it drives RunLayerLoop
// directly; this probe drives RunGreedyDecodeLoop, sslm_generate.cpp's own entry point,
// so it needs one). Deterministic, no domain content: embed[t][i] = ((t*7 + i*13) % 255)
// - 127 -- an arbitrary but REPRODUCIBLE int8 pattern, restated identically (same
// formula, not a copied table) in the Python reference so both sides derive the same
// values from the same rule rather than one side depending on a literal transcription
// of the other's output.
int8_t g_embed[kVocabSize * kHiddenSize];
void BuildEmbedTable() {
	for (uint32_t t = 0; t < kVocabSize; ++t) {
		for (uint32_t i = 0; i < kHiddenSize; ++i) {
			g_embed[t * kHiddenSize + i] = static_cast<int8_t>(((t * 7 + i * 13) % 255) - 127);
		}
	}
}

// Copied verbatim from tests/test_main.cpp's own file-local
// `MakeRop1SectionMultiRow` (that function is `static`, not header-shared, so this
// probe tool -- which cannot #include test_main.cpp -- carries its own copy rather
// than duplicating its logic differently). Builds a spec-faithful ROP1 tensor
// manifest ("cos"/"sin", each [context_cap, pairs] int64) from flat cos/sin arrays.
FixtureSection MakeRop1SectionMultiRow(int32_t context_cap, int32_t pairs,
                                        const int64_t* cos_flat, const int64_t* sin_flat) {
	std::vector<ManifestTensorSpec> tensors = {
	    {"cos", {static_cast<uint32_t>(context_cap), static_cast<uint32_t>(pairs)}},
	    {"sin", {static_cast<uint32_t>(context_cap), static_cast<uint32_t>(pairs)}},
	};
	auto manifest = BuildManifest(kRopeMagic, /*element_size=*/8, tensors);
	const size_t n = static_cast<size_t>(context_cap) * static_cast<size_t>(pairs);
	for (size_t i = 0; i < n; ++i) {
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + i * 8,
		       static_cast<uint64_t>(cos_flat[i]));
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + i * 8,
		       static_cast<uint64_t>(sin_flat[i]));
	}
	return MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
}

// Byte-tokenize into [0, kVocabSize) -- IDENTICAL scheme to
// superslm_spike.pipeline._fixture_tokenize_prompt ("one token per byte... byte %
// cfg.vocab_size").
std::vector<int32_t> Tokenize(const std::string& text) {
	std::vector<int32_t> out;
	out.reserve(text.size());
	for (unsigned char c : text) out.push_back(static_cast<int32_t>(c % kVocabSize));
	return out;
}

}  // namespace

int main(int argc, char** argv) {
	const char* out_path = argc > 1 ? argv[1] : "out/t1891_gate3_kv.bin";

	BuildEmbedTable();
	// Transcribed from dump_cpp_constants() (comment on Layer1891Scales, below).
	const CarriedScale kEmbedSiteConstant{1090717716, -44};
	const CarriedScale kFinalNormSiteConstant{1082196484, -45};

	// --- Build the artifact (real load path) ---------------------------------------
	Cfg1Spec spec{};
	spec.hidden_size = kHiddenSize;
	spec.num_hidden_layers = kNumLayers;
	spec.num_attention_heads = kNumHeads;
	spec.num_key_value_heads = kNumKvHeads;
	spec.head_dim = kHeadDim;
	spec.intermediate_size = kIntermediateSize;
	spec.vocab_size = kVocabSize;
	spec.context_cap = static_cast<uint32_t>(kContextCap);
	spec.kv_precision = 0;  // Int8
	spec.kv_block_size = 1;
	spec.tie_word_embeddings = 1;
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope =
	    MakeRop1SectionMultiRow(kContextCap, /*pairs=*/1, kCosQ30, kSinQ30);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope});

	SslmModelView view;
	std::string err;
	const auto load_status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED: fixture artifact load: %s (%s)\n",
		             SslmModelStatusName(load_status), err.c_str());
		return 1;
	}

	// --- Wire LayerWeights: every scale below is a LITERAL TRANSCRIPTION of
	// `tools/t1891_optiong_gate3_reference.py`'s own `dump_cpp_constants()` output --
	// this fixture's weight matrices (identity/selector) and calibration documents are
	// IDENTICAL to that script's, so both sides' `_derive_scales`/
	// `_derive_composition_constants`-equivalent numbers had to come from the SAME
	// computation to be comparable at all (StandardsDocument §5.4: "the reference is
	// independent of what it grades" -- here the reverse risk applies, two INDEPENDENT
	// derivations of a scale that happen to share a name, which a hand-picked
	// "canonical" constant on this side would have been). Regenerate by re-running that
	// script if the fixture's weights, documents, or geometry ever change.
	struct Layer1891Scales {
		CarriedScale attn_norm, mlp_norm, q_proj, o_proj, gate_proj, up_proj, down_proj;
		CarriedScale attn_ctx, mlp_act, attn_residual, mlp_residual;
		int64_t k_r_t[kNumKvHeads], k_e_t[kNumKvHeads], v_r_t[kNumKvHeads], v_e_t[kNumKvHeads];
		int64_t iexp_m[kNumKvHeads], iexp_e[kNumKvHeads];
	};
	Layer1891Scales L[kNumLayers] = {
	    // layer 0 -- transcribed from t1891_optiong_gate3_reference.py at gain=256
	    {
	        CarriedScale{1082196484, -45}, CarriedScale{1082196484, -45},
	        CarriedScale{1090717716, -44}, CarriedScale{1090717716, -44},
	        CarriedScale{1090717716, -44}, CarriedScale{1090717716, -44},
	        CarriedScale{1090717716, -44},
	        CarriedScale{1990802028, -67}, CarriedScale{1082196484, -52},
	        CarriedScale{1082196484, -37}, CarriedScale{1082196484, -37},
	        {3682832993LL, 3682832993LL}, {-37, -37}, {2353120427LL, 2353120427LL}, {-38, -38},
	        {1784838611LL, 1784838611LL}, {-45, -45},
	    },
	    // layer 1 -- transcribed from t1891_optiong_gate3_reference.py at gain=256
	    {
	        CarriedScale{1082196484, -45}, CarriedScale{1082196484, -45},
	        CarriedScale{1090717716, -44}, CarriedScale{1090717716, -44},
	        CarriedScale{1090717716, -44}, CarriedScale{1090717716, -44},
	        CarriedScale{1090717716, -44},
	        CarriedScale{1990802290, -67}, CarriedScale{1082196484, -52},
	        CarriedScale{1082196484, -37}, CarriedScale{1082196484, -37},
	        {3682849240LL, 3682849240LL}, {-37, -37}, {2353120118LL, 2353120118LL}, {-38, -38},
	        {1784830738LL, 1784830738LL}, {-45, -45},
	    },
	};

	LayerWeights layers[kNumLayers];
	for (uint32_t l = 0; l < kNumLayers; ++l) {
		LayerWeights& lw = layers[l];
		const Layer1891Scales& s = L[l];
		lw.attn_norm_gain = kNormGain;
		lw.attn_norm_site_constant = s.attn_norm;
		lw.q_weight = kIdentity8x8;
		lw.k_weight = kKSelector4x8;
		lw.v_weight = kVSelector4x8;
		lw.o_weight = kIdentity8x8;
		// Uniform per-channel weight_scales on the Python side (dump_cpp_constants's
		// own precondition) makes `_reference_fold` return None at every channel --
		// the exact match to `identity=1` here (comment on Layer1891Scales, above).
		lw.q_fold_identity = kIdentityFold; lw.q_fold_mult = kZeroFold; lw.q_fold_shift = kZeroFold;
		lw.k_fold_identity = kIdentityFold; lw.k_fold_mult = kZeroFold; lw.k_fold_shift = kZeroFold;
		lw.v_fold_identity = kIdentityFold; lw.v_fold_mult = kZeroFold; lw.v_fold_shift = kZeroFold;
		lw.o_fold_identity = kIdentityFold; lw.o_fold_mult = kZeroFold; lw.o_fold_shift = kZeroFold;
		lw.gate_fold_identity = kIdentityFold; lw.gate_fold_mult = kZeroFold; lw.gate_fold_shift = kZeroFold;
		lw.up_fold_identity = kIdentityFold; lw.up_fold_mult = kZeroFold; lw.up_fold_shift = kZeroFold;
		lw.down_fold_identity = kIdentityFold; lw.down_fold_mult = kZeroFold; lw.down_fold_shift = kZeroFold;
		lw.q_site_constant = s.q_proj;
		lw.o_site_constant = s.o_proj;
		lw.kv_landing_r_t_k = s.k_r_t;
		lw.kv_landing_e_t_k = s.k_e_t;
		lw.kv_landing_r_t_v = s.v_r_t;
		lw.kv_landing_e_t_v = s.v_e_t;
		lw.ctx_fold_identity = kCtxFoldIdentity;
		lw.ctx_fold_mult = kCtxFoldZero;
		lw.ctx_fold_shift = kCtxFoldZero;
		lw.ctx_fold_site_constant = s.attn_ctx;
		lw.attn_residual_site_constant = s.attn_residual;
		lw.iexp_softmax_khead_m = s.iexp_m;
		lw.iexp_softmax_khead_e = s.iexp_e;
		lw.mlp_norm_gain = kNormGain;
		lw.mlp_norm_site_constant = s.mlp_norm;
		lw.gate_weight = kIdentity8x8;
		lw.up_weight = kIdentity8x8;
		lw.down_weight = kIdentity8x8;
		lw.gate_site_constant = s.gate_proj;
		lw.up_site_constant = s.up_proj;
		lw.mlp_act_site_constant = s.mlp_act;
		lw.down_site_constant = s.down_proj;
		lw.mlp_residual_site_constant = s.mlp_residual;
	}

	// --- Documents: real English text, byte-tokenized into the small vocab ---------
	// `c % 32` (Tokenize(), above) -- lengths (13, 14, 14) plus the one extra
	// generated position each document's decode produces (comment above the
	// RunGreedyDecodeLoop call) stay within this fixture's context_cap (16).
	const std::vector<std::string> documents = {
	    "the quick fox",
	    "a gray cat sat",
	    "dog barks",
	    "cold rain falls",
	    "we ate soup",
	    "red bird sings",
	};

	// T-1891 gate G5: reset ONCE, before any document, so the counters accumulate
	// over the WHOLE document set (this run's own flag state -- old-path counts only
	// accumulate on a flag-off run, fused-path counts only on a flag-on run, per
	// `OptionGFusedKLandingEnabled`'s own process-lifetime cache).
	OptionGResetSaturationCounters(kNumLayers, kNumKvHeads);

	std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
	if (!out) {
		std::fprintf(stderr, "FAILED: could not open %s for writing\n", out_path);
		return 1;
	}
	const uint64_t num_docs = documents.size();
	out.write(reinterpret_cast<const char*>(&num_docs), sizeof(num_docs));
	const uint64_t nl = kNumLayers, nkv = kNumKvHeads, hd = kHeadDim;
	out.write(reinterpret_cast<const char*>(&nl), sizeof(nl));
	out.write(reinterpret_cast<const char*>(&nkv), sizeof(nkv));
	out.write(reinterpret_cast<const char*>(&hd), sizeof(hd));

	for (const std::string& doc : documents) {
		const std::vector<int32_t> tokens = Tokenize(doc);
		const size_t kv_bytes = static_cast<size_t>(kNumLayers) * static_cast<size_t>(kContextCap) *
		                        kNumKvHeads * kHeadDim * 2;
		std::vector<uint8_t> workspace(kv_bytes);
		std::vector<int8_t> hidden_codes(kHiddenSize);
		SequenceLayerState seq;
		seq.hidden_codes = hidden_codes.data();

		std::vector<int32_t> out_tokens(tokens.size());
		std::vector<int32_t> out_logit_rows(tokens.size() * kVocabSize);
		size_t produced = 0;
		SslmDecodeStopReason stop_reason;
		// No stop ids (nullptr/0, matching sslm_generate.cpp's own
		// `stop_ids.empty() ? nullptr : ...` convention): an out-of-vocabulary
		// sentinel stop id is itself rejected by the SAME TokenIdOutOfRange check
		// EmbedEntry enforces on prompt/generated tokens (found by execution --
		// the first version of this probe used a -1 sentinel and failed exactly
		// this way). `max_new_tokens=1` still lands every prompt position during
		// prefill and produces exactly one extra generated position past the
		// prompt (ignored by the comparator, ordinary decode behaviour, not a
		// defect this probe works around).
		const auto status = RunGreedyDecodeLoop(
		    seq, layers, kNumLayers, kHiddenSize, kHeadDim, kNumKvHeads, kIntermediateSize,
		    kContextCap, view.rope_tables, tokens.data(), tokens.size(), g_embed,
		    kEmbedSiteConstant, kNormGain, kFinalNormSiteConstant, g_embed,
		    static_cast<int32_t>(kVocabSize), nullptr, 0,
		    /*max_new_tokens=*/1, workspace.data(), workspace.size(), out_tokens.data(),
		    out_logit_rows.data(), out_tokens.size(), &produced, &stop_reason,
		    view.config.kv_precision);
		if (status != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED: decode on document \"%s\": %s\n", doc.c_str(),
			             SslmForwardStatusName(status));
			return 1;
		}

		const uint64_t doc_len = tokens.size();
		out.write(reinterpret_cast<const char*>(&doc_len), sizeof(doc_len));
		out.write(reinterpret_cast<const char*>(workspace.data()),
		          static_cast<std::streamsize>(workspace.size()));
		std::fprintf(stderr, "document \"%s\": %zu tokens, decode status Ok\n", doc.c_str(),
		             tokens.size());
	}

	std::fprintf(stderr, "wrote %llu documents' K/V store to %s (option_g_fused=%d)\n",
	             static_cast<unsigned long long>(num_docs), out_path,
	             OptionGFusedKLandingEnabled() ? 1 : 0);

	// T-1891 gate G5: per-(layer, kv_head) saturation report, this run's own flag
	// state, over the whole document set above. Machine-parseable (one "GATE5" line
	// per (layer, kv_head)) so a small driver script can run this binary twice (flag
	// off, flag on) and combine both runs' lines into the old-vs-fused delta table.
	for (uint32_t l = 0; l < kNumLayers; ++l) {
		for (uint32_t h = 0; h < kNumKvHeads; ++h) {
			std::fprintf(stdout, "GATE5 flag=%d layer=%u kv_head=%u old=%llu fused=%llu\n",
			             OptionGFusedKLandingEnabled() ? 1 : 0, l, h,
			             static_cast<unsigned long long>(OptionGSaturationCountOld(l, h)),
			             static_cast<unsigned long long>(OptionGSaturationCountFused(l, h)));
		}
	}
	return 0;
}
