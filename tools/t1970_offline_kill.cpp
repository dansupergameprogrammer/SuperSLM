// t1970_offline_kill.cpp -- T-1970 (Brunel fix round, disposable, never
// merges). The coordinator's own C1-v2 critique of T-1968's position-0
// restriction: RoPE's rotation angle at position 0 is IDENTITY (cos=1,
// sin=0) for every pair simultaneously, so legacy (quantize -> rotate ->
// clamp) and dynamic-fused (rotate -> quantize) are THE SAME FUNCTION at
// that one captured position -- T-1968's own fix bought arm-independence at
// the price of a kill rule that reads UNRESOLVED forever by construction
// (its own reported legacy==dynamic-fused equality, delta exactly 0.0, is
// proof of this, not an unplanned confirmation).
//
// Remedy (the "same-row, three-treatment offline design"):
//   1. Capture the WIDE pre-landing Q row (post-projection, pre-any-
//      rotation, the int64 accumulator) per (layer, position) from ONE
//      named engine run -- the LEGACY arm's own prefill over a real,
//      multi-token prompt (positions >= 1 rotate substantially at the
//      low-frequency pairs; this is the measurement). One run, so the
//      captured rows are arm-independent by definition -- the GEMM/fold/
//      bias steps that produce them never inspect which Q construction is
//      selected.
//   2. Apply all three landing treatments OFFLINE to each captured row,
//      through `OptionGApplyOfflineThreeTreatments`
//      (forward_sites.h/.cpp) -- composed EXCLUSIVELY from already-shipped,
//      certified primitives (RequantChainChecked, RopeApplySite,
//      RopeApplyPairWide, LandingRescale, ClampRopeCode), never a
//      reimplementation of their arithmetic.
//   3. ANCHOR the offline legacy treatment's own pre-rotation quantize step
//      against a SECOND named engine run's own REAL q_codes (also
//      captured from the SAME legacy run, since ProjectAndFunnel's own
//      output IS that run's real legacy answer), and the offline dynamic
//      treatment against a THIRD run's own REAL q_rot (the dynamic-fused
//      arm, same prompt) -- both refusing, bit-exact, quoted counts.
//   4. Grade all three against the wide post-RoPE exact target (computable
//      from the SAME captured row -- also arm-independent), over positions
//      >= 1 (RoPE's rotation is non-identity there, verified from the REAL
//      resolved table, not assumed from the position label alone).
//      Position 0 stays in the dump, reported as its own separate cell,
//      excluded from the deciding pool.
//
// This tool performs steps 1-3 (capture + offline treatments + anchoring)
// and writes ONE JSON dump covering every (layer, position) cell; step 4
// (the graded kill-rule computation) is tools/t1970_offline_kill_compare.py,
// which also carries the NEW vitality check the coordinator's own remedy
// requires: the deciding pool must contain only positions whose rotation is
// non-identity, refused otherwise.
//
// Usage: t1970_offline_kill <model.sslm> <tokenizer.sslm> <out.json> <derived_constants.txt>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/tokenizer.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

namespace {

struct DerivedQLanding {
	int64_t m_out, e_out, e_t, r_t;
};

bool LoadDerivedConstants(const std::string& path, std::map<uint32_t, DerivedQLanding>* out,
                           std::string* err) {
	std::ifstream f(path);
	if (!f) {
		*err = "could not open \"" + path + "\"";
		return false;
	}
	std::string line;
	while (std::getline(f, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		uint32_t l;
		DerivedQLanding d{};
		if (!(iss >> l >> d.m_out >> d.e_out >> d.e_t >> d.r_t)) {
			*err = "malformed line: \"" + line + "\"";
			return false;
		}
		(*out)[l] = d;
	}
	return true;
}

// Runs one full prefill of `prompt_tokens` (plus one trailing decode step,
// harmless -- only positions < prompt_tokens.size() are read back by the
// caller) with `SSLM_OPTION_G_FUSED_Q_LANDING` set to `arm_env_value`
// beforehand. Fresh `SequenceLayerState`/workspace each call, so position
// resets to 0 -- the SAME prompt tokenizes to the SAME positions on both
// the legacy and dynamic-fused runs.
bool RunOneArm(const char* arm_env_value, const std::vector<LayerWeights>& layers,
                uint32_t num_hidden_layers, size_t hidden_size, const SslmModelView& model_view,
                const std::vector<int32_t>& prompt_tokens, const int8_t* embed_weights,
                CarriedScale embed_site_constant, const std::vector<int32_t>& final_norm_gain,
                CarriedScale final_norm_site_constant, const int8_t* head_weights,
                std::vector<int32_t>* out_decode_tokens, std::string* err) {
	_putenv_s("SSLM_OPTION_G_FUSED_Q_LANDING", arm_env_value ? arm_env_value : "");

	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * model_view.config.head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	std::vector<int32_t> out_tokens(1);
	std::vector<int32_t> out_logit_rows(1 * static_cast<size_t>(model_view.config.vocab_size));
	size_t out_tokens_produced = 0;
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;
	const std::vector<int32_t> stop_ids = {151645, 151643};

	const SslmForwardStatus decode_status = RunGreedyDecodeLoop(
	    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim, num_kv_heads,
	    model_view.config.intermediate_size, context_cap, model_view.rope_tables, prompt_tokens.data(),
	    prompt_tokens.size(), embed_weights, embed_site_constant, final_norm_gain.data(),
	    final_norm_site_constant, head_weights, static_cast<int32_t>(model_view.config.vocab_size),
	    stop_ids.data(), stop_ids.size(), 1, workspace.data(), workspace.size(), out_tokens.data(),
	    out_logit_rows.data(), out_tokens.size(), &out_tokens_produced, &stop_reason,
	    model_view.config.kv_precision, /*option_g_fused_k_landing=*/false);
	if (decode_status != SslmForwardStatus::Ok) {
		*err = std::string("decode status=") + SslmForwardStatusName(decode_status);
		return false;
	}
	out_decode_tokens->assign(out_tokens.begin(), out_tokens.begin() + out_tokens_produced);
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr,
		             "usage: %s <model.sslm> <tokenizer.sslm> <out.json> <derived_constants.txt>\n",
		             argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string out_path = argv[3];
	const std::string derived_path = argv[4];

	std::map<uint32_t, DerivedQLanding> derived;
	std::string derived_err;
	if (!LoadDerivedConstants(derived_path, &derived, &derived_err)) {
		std::fprintf(stderr, "FAILED stage=derived_constants_load diagnostic=\"%s\"\n",
		             derived_err.c_str());
		return 1;
	}

	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_file_read path=\"%s\"\n", tokenizer_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=tokenizer_artifact_open status=%s\n",
		             SslmStatusName(tok_open_err.code));
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_view_open diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED stage=model_file_read path=\"%s\"\n", model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=model_load status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED stage=layer_weights_marshal layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
		const auto it = derived.find(l);
		if (it == derived.end()) {
			std::fprintf(stderr, "FAILED stage=derived_constants_lookup: no entry for layer=%u\n", l);
			return 1;
		}
		// Injected for EVERY layer, unconditionally -- unlike
		// t1966_arm_capture.cpp (which only needs these for the static-fused
		// arm), this tool's offline static-treatment computation needs them
		// on every (layer, position) cell regardless of which arm the two
		// engine runs below select.
		layers[l].q_landing_m_out = it->second.m_out;
		layers[l].q_landing_e_out = it->second.e_out;
		layers[l].q_landing_e_t = it->second.e_t;
		layers[l].q_landing_r_t = it->second.r_t;
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	const SslmTensorView* final_gain_w = model_view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing embed or final_norm.gain\n");
		return 1;
	}
	std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	CarriedScale final_norm_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "final_norm", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing site constants\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const int8_t* head_weights =
	    model_view.config.tie_word_embeddings
	        ? embed_weights
	        : reinterpret_cast<const int8_t*>(model_view.weights.Tensor("lm_head")->data);

	// A real, natural-language, multi-token prompt -- long enough that the
	// low-frequency RoPE pairs rotate substantially well before the prompt
	// ends (theta_0 == 1 rad/position, so position 1 alone already leaves
	// identity at the fastest-rotating pair; later positions bring
	// progressively slower pairs off identity too). Positions >= 1 are the
	// coordinator's own named measurement; position 0 stays in the dump,
	// reported separately, per the remedy's own instruction.
	const std::string prompt =
	    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the "
	    "distant mountains, painting the sky in brilliant shades of orange and purple as "
	    "evening approaches and the day comes to its natural end.";
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED stage=tokenize: zero tokens\n");
		return 1;
	}
	const uint32_t max_positions = static_cast<uint32_t>(prompt_tokens.size());
	std::printf("prompt tokenized to %u tokens (positions 0..%u captured)\n", max_positions,
	            max_positions - 1);

	_putenv_s("SSLM_OPTION_G_OFFLINE_KILL_CAPTURE", "1");
	OptionGOfflineKillReset(num_hidden_layers, max_positions);

	std::vector<int32_t> legacy_decode_tokens, dynamic_decode_tokens;
	std::string run_err;
	if (!RunOneArm(nullptr, layers, num_hidden_layers, hidden_size, model_view, prompt_tokens,
	               embed_weights, embed_site_constant, final_norm_gain, final_norm_site_constant,
	               head_weights, &legacy_decode_tokens, &run_err)) {
		std::fprintf(stderr, "FAILED stage=legacy_run diagnostic=\"%s\"\n", run_err.c_str());
		return 1;
	}
	const std::vector<OptionGOfflineKillSample> legacy_samples = OptionGOfflineKillDumpLegacy();
	std::printf("legacy run complete: %zu decode token(s), %zu (layer,position) slots\n",
	            legacy_decode_tokens.size(), legacy_samples.size());

	if (!RunOneArm("dynamic", layers, num_hidden_layers, hidden_size, model_view, prompt_tokens,
	               embed_weights, embed_site_constant, final_norm_gain, final_norm_site_constant,
	               head_weights, &dynamic_decode_tokens, &run_err)) {
		std::fprintf(stderr, "FAILED stage=dynamic_run diagnostic=\"%s\"\n", run_err.c_str());
		return 1;
	}
	const std::vector<OptionGOfflineKillSample> dynamic_samples = OptionGOfflineKillDumpDynamic();
	std::printf("dynamic-fused run complete: %zu decode token(s), %zu (layer,position) slots\n",
	            dynamic_decode_tokens.size(), dynamic_samples.size());

	// --- Apply the three offline treatments to every captured cell, ANCHOR
	// each against its own named engine run, per the remedy's own step 3.
	size_t total_cells = 0, legacy_anchor_checked = 0, legacy_anchor_matched = 0;
	size_t dynamic_anchor_checked = 0, dynamic_anchor_matched = 0;
	size_t legacy_anchor_total_mismatches = 0, dynamic_anchor_total_mismatches = 0;
	size_t skipped_uncaptured = 0;
	// T-1970/T-1968-C1-final-form (coordinator's own re-scoped remedy): the
	// DECIDING POOL is layer 0, positions 1..40 -- the intersection of
	// arm-independent (layer 0: embedding-fed input, no upstream Q
	// influence) and genuinely rotated (position >= 1: non-identity RoPE).
	// Tracked separately from the global 1148-cell grid, whose own
	// out-of-pool mismatches are now an EXPECTED, explained fact (§24.3),
	// not a refusing condition -- the pool's own 100% is.
	size_t pool_cells = 0, pool_legacy_matched = 0, pool_dynamic_matched = 0, pool_non_identity = 0;

	std::ofstream out(out_path);
	if (!out) {
		std::fprintf(stderr, "FAILED: could not open \"%s\" for writing\n", out_path.c_str());
		return 1;
	}
	out << "{\n  \"hidden_size\": " << hidden_size << ",\n  \"num_hidden_layers\": " << num_hidden_layers
	    << ",\n  \"max_positions\": " << max_positions << ",\n  \"legacy_decode_tokens\": [";
	for (size_t i = 0; i < legacy_decode_tokens.size(); ++i) {
		if (i) out << ",";
		out << legacy_decode_tokens[i];
	}
	out << "],\n  \"dynamic_decode_tokens\": [";
	for (size_t i = 0; i < dynamic_decode_tokens.size(); ++i) {
		if (i) out << ",";
		out << dynamic_decode_tokens[i];
	}
	out << "],\n  \"cells\": [\n";
	bool first_cell = true;
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		for (uint32_t p = 0; p < max_positions; ++p) {
			const size_t idx = static_cast<size_t>(l) * max_positions + p;
			const OptionGOfflineKillSample& leg = legacy_samples[idx];
			const OptionGOfflineKillSample& dyn = dynamic_samples[idx];
			if (!leg.captured || !dyn.captured) {
				++skipped_uncaptured;
				continue;
			}
			const CarriedScale normed_scale{leg.normed_scale_m, leg.normed_scale_e};
			const CarriedScale site_constant{leg.site_constant_m, leg.site_constant_e};
			OptionGOfflineTreatmentResult result = OptionGApplyOfflineThreeTreatments(
			    leg.qacc_pre_rotation.data(), hidden_size, head_dim, num_heads,
			    static_cast<int64_t>(p), context_cap, model_view.rope_tables, normed_scale,
			    site_constant, layers[l].q_landing_m_out, layers[l].q_landing_e_out,
			    layers[l].q_landing_e_t, layers[l].q_landing_r_t, leg.engine_codes.data(),
			    dyn.engine_codes.data(), "t1970_offline", 0, nullptr);
			if (result.status != SslmForwardStatus::Ok) {
				std::fprintf(stderr,
				             "FAILED stage=offline_treatment layer=%u position=%u status=%s\n", l, p,
				             SslmForwardStatusName(result.status));
				return 1;
			}
			++total_cells;
			if (result.legacy_anchor_checked) {
				++legacy_anchor_checked;
				if (result.legacy_anchor_match) {
					++legacy_anchor_matched;
				} else {
					legacy_anchor_total_mismatches += result.legacy_anchor_mismatches;
				}
			}
			if (result.dynamic_anchor_checked) {
				++dynamic_anchor_checked;
				if (result.dynamic_anchor_match) {
					++dynamic_anchor_matched;
				} else {
					dynamic_anchor_total_mismatches += result.dynamic_anchor_mismatches;
				}
			}
			// Deciding-pool admissibility tally: layer 0, positions 1..40 --
			// see this function's own comment above `pool_cells`.
			if (l == 0 && p >= 1 && p <= 40) {
				++pool_cells;
				if (result.legacy_anchor_match) ++pool_legacy_matched;
				if (result.dynamic_anchor_match) ++pool_dynamic_matched;
				if (!result.is_identity_rotation) ++pool_non_identity;
			}

			if (!first_cell) out << ",\n";
			first_cell = false;
			out << "    {\"layer\": " << l << ", \"position\": " << p
			    << ", \"is_identity_rotation\": " << (result.is_identity_rotation ? "true" : "false")
			    << ", \"legacy_anchor_match\": " << (result.legacy_anchor_match ? "true" : "false")
			    << ", \"dynamic_anchor_match\": " << (result.dynamic_anchor_match ? "true" : "false")
			    << ", \"normed_scale_m\": " << leg.normed_scale_m
			    << ", \"normed_scale_e\": " << leg.normed_scale_e
			    << ", \"site_constant_m\": " << leg.site_constant_m
			    << ", \"site_constant_e\": " << leg.site_constant_e << ", \"target_wide_rotated\": [";
			for (size_t i = 0; i < result.target_wide_rotated.size(); ++i) {
				if (i) out << ",";
				out << result.target_wide_rotated[i];
			}
			out << "], \"legacy_codes\": [";
			for (size_t i = 0; i < result.legacy_codes.size(); ++i) {
				if (i) out << ",";
				out << static_cast<int>(result.legacy_codes[i]);
			}
			out << "], \"legacy_scale_m\": " << result.legacy_scale.m
			    << ", \"legacy_scale_e\": " << result.legacy_scale.e << ", \"static_codes\": [";
			for (size_t i = 0; i < result.static_codes.size(); ++i) {
				if (i) out << ",";
				out << static_cast<int>(result.static_codes[i]);
			}
			out << "], \"static_scale_m\": " << result.static_scale.m
			    << ", \"static_scale_e\": " << result.static_scale.e << ", \"dynamic_codes\": [";
			for (size_t i = 0; i < result.dynamic_codes.size(); ++i) {
				if (i) out << ",";
				out << static_cast<int>(result.dynamic_codes[i]);
			}
			out << "], \"dynamic_scale_m\": " << result.dynamic_scale.m
			    << ", \"dynamic_scale_e\": " << result.dynamic_scale.e << ", \"k_row_head0\": [";
			for (size_t i = 0; i < leg.k_row_head0.size(); ++i) {
				if (i) out << ",";
				out << static_cast<int>(leg.k_row_head0[i]);
			}
			out << "]}";
		}
	}
	out << "\n  ]\n}\n";
	out.close();

	std::printf("\n=== Anchor counts, GLOBAL grid (executed, bit-exact; the coordinator's own T-1970 "
	            "enumeration, re-confirmed) ===\n");
	std::printf("cells with both engine runs captured: %zu (skipped, uncaptured: %zu)\n", total_cells,
	            skipped_uncaptured);
	std::printf("legacy anchor: %zu/%zu cells bit-exact vs engine q_codes (total element mismatches "
	            "across failing cells: %zu)\n",
	            legacy_anchor_matched, legacy_anchor_checked, legacy_anchor_total_mismatches);
	std::printf("dynamic anchor: %zu/%zu cells bit-exact vs engine q_rot (total element mismatches "
	            "across failing cells: %zu) -- EXPECTED to be well short of 100%%: only {layer 0, any "
	            "position} union {any layer, position 0} is arm-independent (D-SLM2809/2810); this is "
	            "no longer this tool's own refusing condition, the DECIDING POOL below is.\n",
	            dynamic_anchor_matched, dynamic_anchor_checked, dynamic_anchor_total_mismatches);

	std::printf("\n=== Deciding-pool admissibility, layer 0 positions 1..40 (T-1968-C1-final-form) "
	            "===\n");
	std::printf("pool cells: %zu (expected 40)\n", pool_cells);
	std::printf("pool legacy anchor bit-exact: %zu/%zu\n", pool_legacy_matched, pool_cells);
	std::printf("pool dynamic anchor bit-exact: %zu/%zu\n", pool_dynamic_matched, pool_cells);
	std::printf("pool non-identity rotation: %zu/%zu\n", pool_non_identity, pool_cells);

	if (pool_cells != 40) {
		std::fprintf(stderr,
		             "FAILED: deciding pool has %zu cells, expected exactly 40 (layer 0, positions "
		             "1..40) -- refusing\n",
		             pool_cells);
		return 1;
	}
	if (pool_legacy_matched != pool_cells) {
		std::fprintf(stderr,
		             "FAILED: deciding pool's own legacy anchor is not 100%% bit-exact (%zu/%zu) -- "
		             "the offline reimplementation cannot be trusted on this pool; refusing (this is "
		             "the C1 class again)\n",
		             pool_legacy_matched, pool_cells);
		return 1;
	}
	if (pool_dynamic_matched != pool_cells) {
		std::fprintf(stderr,
		             "FAILED: deciding pool's own dynamic anchor is not 100%% bit-exact (%zu/%zu) -- "
		             "the offline reimplementation cannot be trusted on this pool; refusing (this is "
		             "the C1 class again)\n",
		             pool_dynamic_matched, pool_cells);
		return 1;
	}
	if (pool_non_identity != pool_cells) {
		std::fprintf(stderr,
		             "FAILED: deciding pool contains %zu/%zu cells with non-identity rotation, "
		             "expected all 40 -- refusing (a position-0-shaped cell has leaked into the pool)\n",
		             pool_non_identity, pool_cells);
		return 1;
	}
	std::printf("\nDECIDING POOL ADMISSIBLE: 40/40 cells pass both anchor gates AND non-identity "
	            "rotation -- wrote %zu total cells (full grid) to \"%s\"\n",
	            total_cells, out_path.c_str());
	return 0;
}
