// sslm_generate.cpp -- the real-model decode driver (T-1664, WSC1 marshaling
// completed T-1666-driver). Loads a `.sslm` model artifact and a `.sslm`
// tokenizer artifact, encodes a prompt, marshals the artifact's raw
// manifests into a `LayerWeights[]` for all 28 layers, and runs
// `RunGreedyDecodeLoop` to produce output token ids.
//
// THIS BUILD'S STATUS, STATED PLAINLY. T-1657/T-1654/T-1655/T-1656 closed the
// four blockers a prior pass (T-1652/T-1653, branch claude/smoke-driver)
// found: the BIA1 load-time gate is gone (SslmModel::Load accepts the real
// artifact with NO relaxation, on `main`'s own gate), RunLayerLoop is
// GQA-capable, LayerWeights carries a bias field with a live caller, and the
// per-query i-exp constants are derived from artifact composition constants
// rather than fixed per-layer.
//
// THE FIELD T-1664 FOUND NOT REPRESENTABLE IS NOW REPRESENTABLE. T-1664
// discovered and verified against the real artifact's own bytes that the
// real Qwen2.5-1.5B-Instruct artifact's WSC1 weight-scale-fold data is
// genuinely PER-OUTPUT-CHANNEL for all seven projections -- q_proj/o_proj/
// down_proj carry 1536 distinct (identity,mult,shift) triples, k_proj/v_proj
// carry 256, gate_proj/up_proj carry 8960 -- while `LayerWeights` (at that
// time) carried only a single scalar triple shared across a whole
// projection. T-1666 (design doc
// Claude/Vitruvius/superslm-t1666-wsc1-per-channel-fold-design-2026-08-02.md)
// closed that gap in production: `LayerWeights` now carries seven per-
// projection `{proj}_fold_identity`/`{proj}_fold_mult`/`{proj}_fold_shift`
// arrays (forward_sites.h), one triple per output channel, and
// `ProjectAndFunnel`/the k/v-landing fold loop index them by output channel
// (forward_sites.cpp). This driver's job -- marshaling the artifact's real
// WSC1 rows into those arrays, for every layer and all seven projections --
// is what this file now does. Every array is sized to the exact output-
// channel count `ProjectAndFunnel`/the k/v-landing loop read it at: hidden_size
// for q/o/down, `num_key_value_heads * head_dim` for k/v, intermediate_size
// for gate/up (§8.2 "Weight-scale fold blob", docs/sslm_format.md: each WSC1
// tensor is a row-major `[num_channels, 3]` int32 array, row i's triple at
// elements `[3i, 3i+3)` -- identity, mult, shift in that column order).
//
// Usage: sslm_generate <model.sslm> <tokenizer.sslm> "<prompt>" [--max-new N]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/model.h"
#include "superslm/sslm_phaseD.h"  // T-2199 Phase D review fix S4: --decode-mode damped-greedy
#include "superslm/tokenizer.h"
#include "sslm_marshal.h"
#include "sslm_adapter_loader.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;
using superslm_adapter::AdapterHandle;
using superslm_adapter::AdapterLoadStatus;
using superslm_adapter::ApplyAdapterToLayers;
using superslm_adapter::BaseModelGeometry;
using superslm_adapter::LoadAdapterArtifact;

namespace {

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" [--max-new N] [--stop "
	             "a,b,c] [--dump-logits <path>] [--adapter <adapter.sslm>] [--decode-mode "
	             "greedy|damped-greedy [--alpha-q15 N] [--anti-lm-order N] [--top-k N]]\n",
	             argv0);
}

// T-1681: writes `out_logit_rows` (RunGreedyDecodeLoop's own raw int32 logit
// output, one row of `vocab_size` elements per produced token, populated
// immediately before that token's argmax -- see forward_sites.h's
// RunGreedyDecodeLoop comment) to a flat binary file for offline analysis.
// This is a read-only dump of an array the driver already computes and
// already owns (`out_logit_rows` was already sized and passed into
// RunGreedyDecodeLoop before this flag existed) -- no forward-path
// computation changes. Format: `rows_produced` (uint64 LE) then
// `vocab_size` (uint64 LE) then `rows_produced * vocab_size` little-endian
// int32 values, row-major (row i = the logit row that produced out_tokens[i]).
bool DumpLogitRows(const char* path, const int32_t* rows, size_t rows_produced,
                    size_t vocab_size) {
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) return false;
	const uint64_t rp = static_cast<uint64_t>(rows_produced);
	const uint64_t vs = static_cast<uint64_t>(vocab_size);
	f.write(reinterpret_cast<const char*>(&rp), sizeof(rp));
	f.write(reinterpret_cast<const char*>(&vs), sizeof(vs));
	f.write(reinterpret_cast<const char*>(rows), sizeof(int32_t) * rows_produced * vocab_size);
	return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompt = argv[3];
	size_t max_new_tokens = 32;
	// The .sslm format carries no end-of-sequence id and the tokenizer view exposes none
	// (tokenizer.h: "No BOS/EOS/chat markers are added -- that is the caller's job"), so the
	// stop set is supplied here rather than defaulted. Qwen2.5-instruct uses 151645
	// (<|im_end|>) and 151643 (<|endoftext|>); a different model family uses different ids.
	std::vector<int32_t> stop_ids;
	std::string dump_logits_path;
	std::string adapter_path;  // T-2102: --adapter <path> -- runtime LoRA, empty means base-only
	// T-2199 Phase D review fix S4/S3 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md): the CLI's
	// own damped-greedy selector -- plan Sec8 D3's stated deliverable ("so tools/sslm_generate.cpp
	// ... can select damped greedy mode too"), previously unmet (the engine wiring existed with
	// no caller). Defaults to greedy (bit-identical to this driver's pre-Phase-D behavior, this
	// flag block a pure addition, nothing existing re-parses differently).
	superslm::DampedGreedyMode decode_mode = superslm::DampedGreedyMode::kGreedy;
	int32_t alpha_q15 = SSLM_DAMPED_GREEDY_DEFAULT_ALPHA_Q15;
	int32_t anti_lm_max_order = SSLM_DAMPED_GREEDY_DEFAULT_ANTI_LM_ORDER;
	int32_t top_k = SSLM_DAMPED_GREEDY_DEFAULT_TOP_K;
	bool have_top_k = false;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
			dump_logits_path = argv[++i];
		} else if (std::strcmp(argv[i], "--adapter") == 0 && i + 1 < argc) {
			adapter_path = argv[++i];
		} else if (std::strcmp(argv[i], "--decode-mode") == 0 && i + 1 < argc) {
			const std::string val = argv[++i];
			if (val == "greedy") {
				decode_mode = superslm::DampedGreedyMode::kGreedy;
			} else if (val == "damped-greedy") {
				decode_mode = superslm::DampedGreedyMode::kDampedGreedy;
			} else {
				std::fprintf(stderr, "invalid --decode-mode value: \"%s\" (expected \"greedy\" or "
				                      "\"damped-greedy\")\n", val.c_str());
				PrintUsage(argv[0]);
				return 2;
			}
		} else if (std::strcmp(argv[i], "--alpha-q15") == 0 && i + 1 < argc) {
			try {
				alpha_q15 = static_cast<int32_t>(std::stol(argv[++i]));
			} catch (const std::exception&) {
				std::fprintf(stderr, "invalid --alpha-q15 value: \"%s\"\n", argv[i]);
				PrintUsage(argv[0]);
				return 2;
			}
		} else if (std::strcmp(argv[i], "--anti-lm-order") == 0 && i + 1 < argc) {
			try {
				anti_lm_max_order = static_cast<int32_t>(std::stol(argv[++i]));
			} catch (const std::exception&) {
				std::fprintf(stderr, "invalid --anti-lm-order value: \"%s\"\n", argv[i]);
				PrintUsage(argv[0]);
				return 2;
			}
		} else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
			try {
				top_k = static_cast<int32_t>(std::stol(argv[++i]));
				have_top_k = true;
			} catch (const std::exception&) {
				std::fprintf(stderr, "invalid --top-k value: \"%s\"\n", argv[i]);
				PrintUsage(argv[0]);
				return 2;
			}
		} else if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
			const std::string val = argv[++i];
			try {
				max_new_tokens = static_cast<size_t>(std::stoul(val));
			} catch (const std::exception&) {
				std::fprintf(stderr, "invalid --max-new value: \"%s\" (expected an unsigned integer)\n",
				             val.c_str());
				PrintUsage(argv[0]);
				return 2;
			}
		} else if (std::strcmp(argv[i], "--stop") == 0 && i + 1 < argc) {
			const std::string spec = argv[++i];
			size_t pos = 0;
			try {
				while (pos < spec.size()) {
					const size_t comma = spec.find(',', pos);
					const std::string one = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
					if (!one.empty()) stop_ids.push_back(static_cast<int32_t>(std::stol(one)));
					if (comma == std::string::npos) break;
					pos = comma + 1;
				}
			} catch (const std::exception&) {
				std::fprintf(stderr, "invalid --stop value: \"%s\" (expected comma-separated integers)\n",
				             spec.c_str());
				PrintUsage(argv[0]);
				return 2;
			}
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}
	const auto t_start = std::chrono::steady_clock::now();

	// --- Stage 1: load the tokenizer artifact and encode the prompt. -------
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_file_read: could not read \"%s\"\n",
		             tokenizer_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact,
	                                  &tok_open_err) != SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_artifact_open: status=%s diagnostic=\"%s\"\n",
		             SslmStatusName(tok_open_err.code), tok_open_err.message.c_str());
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_view_open: diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);

	std::printf("prompt: %s\n", prompt.c_str());
	std::printf("prompt_tokens (%zu):", prompt_tokens.size());
	for (int32_t t : prompt_tokens) std::printf(" %d", t);
	std::printf("\n");
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_encode: prompt encoded to zero tokens\n");
		return 1;
	}

	// --- Stage 2: load the model artifact through the real production ------
	// entry point (SslmModel::Load). T-1657 removed the BIA1 load-time gate
	// this driver's predecessor needed a branch-only relaxation for; this
	// call needs no relaxation of any kind.
	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED at stage=model_file_read: could not read \"%s\"\n",
		             model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u intermediate=%u "
	            "vocab=%u context_cap=%u tie=%d\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim, model_view.config.intermediate_size,
	            model_view.config.vocab_size, model_view.config.context_cap,
	            model_view.config.tie_word_embeddings ? 1 : 0);

	// T-2199 Phase D review fix S3 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md): "an artifact
	// missing the Decision-A scale field under damped greedy mode must be a defined rejection,
	// not a silent fallback to garbage constants" (plan Sec9 dim2) -- unreachable before this fix
	// because ArtifactHasDampedGreedyConstants/ReadDampedGreedyScaleConstants had no production
	// caller anywhere. SslmModelView now exposes the already-validated backing section, so this
	// driver does not parse and hash the same model bytes a second time merely to reach DGC1.
	int64_t q_ln2 = 0, q_b = 0, q_c = 0;
	if (decode_mode == superslm::DampedGreedyMode::kDampedGreedy) {
		// Match sslm_decode_params_init: the ruled default is six candidates when the model has
		// at least six tokens, otherwise the whole vocabulary. An explicit out-of-range override
		// remains an error rather than being silently rewritten.
		if (!have_top_k) {
			top_k = std::min<int32_t>(SSLM_DAMPED_GREEDY_DEFAULT_TOP_K,
			                          static_cast<int32_t>(model_view.config.vocab_size));
		}
		if (!model_view.DampedGreedyConstantsFlagSet()) {
			std::fprintf(stderr, "FAILED at stage=damped_greedy_constants: --decode-mode "
			                      "damped-greedy requires the artifact's own DampedGreedyConstants "
			                      "(DGC1) section and its flag bit -- neither is present on this "
			                      "artifact. No forward pass was attempted.\n");
			return 1;
		}
		superslm::DampedGreedyScaleConstants scale{};
		if (!superslm::ReadDampedGreedyScaleConstants(model_view, &scale)) {
			std::fprintf(stderr, "FAILED at stage=damped_greedy_constants: the artifact's "
			                      "DampedGreedyConstants section is present but out of domain\n");
			return 1;
		}
		// Phase B3's own derivation (plan Sec2.3/Sec8 B3): the artifact carries the raw (m, e)
		// source pair; the runtime (q_ln2, q_b, q_c) triple this decode step actually needs is
		// derived ONCE per model load through the certified IExpScaleConstants primitive --
		// exactly the recipe the suite's own shared fixture (sslm_phaseD_fixture.h) documents,
		// reused here rather than re-derived ad hoc.
		if (superslm::IExpScaleConstants(scale.scale_mantissa_m, scale.scale_exponent_e,
		                                  superslm::kIExpLn2Q, 30, superslm::kIExpBQ, 30,
		                                  superslm::kIExpCaQ, 30, &q_ln2, &q_b,
		                                  &q_c) != superslm::IExpScaleDomain::kOk) {
			std::fprintf(stderr, "FAILED at stage=damped_greedy_constants: the artifact's (m, e) "
			                      "scale pair is out of the certified i-exp domain\n");
			return 1;
		}
		const superslm::DampedGreedyValidationParams vp{decode_mode, alpha_q15, anti_lm_max_order,
		                                                 top_k};
		if (!superslm::ValidateDampedGreedyParams(
		        vp, static_cast<int32_t>(model_view.config.vocab_size))) {
			std::fprintf(stderr, "FAILED at stage=damped_greedy_constants: --alpha-q15/"
			                      "--anti-lm-order/--top-k failed domain validation "
			                      "(ValidateDampedGreedyParams)\n");
			return 1;
		}
	}

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;

	// --- Stage 3: the marshaling adapter. Full artifact-wide scan first ----
	// (this file's header comment; §3.1-style extent confirmation), then a
	// real attempt at every layer, in order -- stopping at the first field
	// this tree's LayerWeights cannot represent (missing/wrong-shaped
	// tensors remain possible; the per-channel WSC1 fold gap T-1664 found is
	// closed as of T-1666).
	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n",
			             l, marshal_err.c_str());
			std::fprintf(stderr,
			             "The tokenizer and model both load and are real. The LayerWeights[] "
			             "marshaling adapter cannot represent this layer's data in the current struct "
			             "shape -- see the diagnostic above for the exact field. No forward pass was "
			             "attempted; RunGreedyDecodeLoop was never called.\n");
			return 1;
		}
	}

	// --- Stage 3b: the runtime adapter (T-2102), if --adapter was given. ---
	// LayerWeights::adapter defaults to nullptr on every layer (LayerWeights' own field default),
	// which is already the correct "no adapter bound" state -- this stage only runs, and only
	// mutates that default, when the flag is present.
	AdapterHandle adapter_handle;
	if (!adapter_path.empty()) {
		// T-2104 (Poirot 8e07d0c review, Significant 2): the base's own RawIntegrityHash, read
		// directly off `model_view` -- no second `SslmArtifact::OpenFromMemory` over model_bytes.
		// The ORIGINAL comment here justified that second open by claiming SslmModel::Load's view
		// dangles once Load returns; that claim was false at source (SslmModelView::backing_ owns
		// the bytes, model.h) and the real reason a second open was needed was simply that
		// SslmModelView exposed no hash accessor of its own. It now does
		// (SslmModelView::RawIntegrityHash(), added this round) -- the SAME "lifted out rather than
		// reaching into backing_" precedent option_g_fused_k_landing already set.
		BaseModelGeometry geo;
		geo.num_hidden_layers = num_hidden_layers;
		geo.hidden_size = hidden_size;
		geo.intermediate_size = model_view.config.intermediate_size;
		geo.kv_hidden_size = static_cast<uint64_t>(num_kv_heads) * model_view.config.head_dim;
		geo.base_artifact_hash = model_view.RawIntegrityHash();

		const auto t_adapter_start = std::chrono::steady_clock::now();
		std::string adapter_err;
		const auto adapter_status =
		    LoadAdapterArtifact(adapter_path, geo, adapter_handle, &adapter_err);
		const auto t_adapter_end = std::chrono::steady_clock::now();
		if (adapter_status != AdapterLoadStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=adapter_load: status=%s diagnostic=\"%s\"\n",
			             superslm_adapter::AdapterLoadStatusName(adapter_status), adapter_err.c_str());
			return 1;
		}
		ApplyAdapterToLayers(&adapter_handle, layers.data(), num_hidden_layers);
		std::printf("adapter loaded: path=%s rank=%u target_modules_mask=0x%x source=\"%s\" "
		            "load_seconds=%.4f\n",
		            adapter_path.c_str(), adapter_handle.meta.rank, adapter_handle.meta.target_modules_mask,
		            adapter_handle.meta.source_adapter_name.c_str(),
		            std::chrono::duration<double>(t_adapter_end - t_adapter_start).count());
	}

	// --- embed/final_norm/head marshaling, workspace sizing, and the -------
	// RunGreedyDecodeLoop call.
	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	const SslmTensorView* final_gain_w = model_view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed or final_norm.gain tensor\n");
		return 1;
	}
	std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	CarriedScale embed_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	CarriedScale final_norm_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "final_norm", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed/final_norm site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	// tie_word_embeddings is the CALLER's resolution (forward_sites.h:828,
	// "weights[tie ? \"embed\" : \"lm_head\"]"): when the artifact ties the
	// head to the embedding matrix, the head reuses embed_weights; when it
	// does not, a dedicated "lm_head" WGT1 tensor must be present. This
	// artifact happens to tie (tie=1, printed above), but that is a fact
	// about THIS artifact, not a fact this driver may assume -- CFG1's
	// tie_word_embeddings is a legal 0 (accepted by SslmModel::Load,
	// src/model.cpp:483; already produced by
	// tools/sslm_pinned_calibration_fixture.py and
	// tools/test_sslm_convert_loader_join.py). An unconditional embed
	// fallback would compute logits against the wrong matrix on a tie=0
	// artifact and exit 0 -- the only silent-wrong-answer path in a file
	// whose every other stage fails loudly.
	const int8_t* head_weights = nullptr;
	if (model_view.config.tie_word_embeddings) {
		head_weights = embed_weights;
	} else {
		const SslmTensorView* lm_head_w = model_view.weights.Tensor("lm_head");
		if (!lm_head_w) {
			std::fprintf(stderr,
			             "FAILED at stage=head_marshal: tie_word_embeddings=0 but no \"lm_head\" WGT1 "
			             "tensor is present -- cannot resolve the head weight matrix\n");
			return 1;
		}
		head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * model_view.config.head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	std::vector<int32_t> out_tokens(max_new_tokens);
	std::vector<int32_t> out_logit_rows(max_new_tokens * static_cast<size_t>(model_view.config.vocab_size));
	size_t out_tokens_produced = 0;
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;

	// T-1894 (design Sec31.2.1, round 4/D-SLM2423, link 4 of 5): passes
	// `model_view.option_g_fused_k_landing` -- the header `flags` bit
	// SslmModel::Load already read into this view (link 1/2) -- as
	// RunGreedyDecodeLoop's own new trailing parameter, mirroring exactly how
	// this call already passes `model_view.config.kv_precision` as its
	// previous last argument.
	// T-2199 Phase D review fix S4: damped-greedy is a SEPARATE call to the parallel loop
	// (RunGreedyOrDampedGreedyDecodeLoop), never a branch inside this existing call -- the
	// greedy path below is byte-for-byte the pre-Phase-D call, zero regression risk, matching
	// this loop function's own "new, parallel function rather than an edit to RunGreedyDecodeLoop
	// itself" design (damped_greedy_phaseD_loop.cpp's header comment).
	SslmForwardStatus decode_status;
	if (decode_mode == superslm::DampedGreedyMode::kDampedGreedy) {
		decode_status = superslm::RunGreedyOrDampedGreedyDecodeLoop(
		    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim,
		    num_kv_heads, model_view.config.intermediate_size, context_cap, model_view.rope_tables,
		    prompt_tokens.data(), prompt_tokens.size(), embed_weights, embed_site_constant,
		    final_norm_gain.data(), final_norm_site_constant, head_weights,
		    static_cast<int32_t>(model_view.config.vocab_size),
		    stop_ids.empty() ? nullptr : stop_ids.data(), stop_ids.size(), max_new_tokens,
		    workspace.data(), workspace.size(), out_tokens.data(), out_logit_rows.data(),
		    out_tokens.size(), &out_tokens_produced, &stop_reason, model_view.config.kv_precision,
		    model_view.option_g_fused_k_landing, decode_mode, alpha_q15, anti_lm_max_order, top_k,
		    q_ln2, q_b, q_c);
	} else {
		decode_status = RunGreedyDecodeLoop(
		    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim, num_kv_heads,
		    model_view.config.intermediate_size, context_cap, model_view.rope_tables, prompt_tokens.data(),
		    prompt_tokens.size(), embed_weights, embed_site_constant, final_norm_gain.data(),
		    final_norm_site_constant, head_weights, static_cast<int32_t>(model_view.config.vocab_size),
		    stop_ids.empty() ? nullptr : stop_ids.data(), stop_ids.size(), max_new_tokens, workspace.data(), workspace.size(),
		    out_tokens.data(), out_logit_rows.data(), out_tokens.size(), &out_tokens_produced, &stop_reason,
		    model_view.config.kv_precision, model_view.option_g_fused_k_landing);
	}

	if (decode_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=decode: status=%s\n",
		             SslmForwardStatusName(decode_status));
		return 1;
	}

	std::printf("output_tokens (%zu):", out_tokens_produced);
	for (size_t i = 0; i < out_tokens_produced; ++i) std::printf(" %d", out_tokens[i]);
	std::printf("\n");
	const std::vector<int32_t> output_token_vector(out_tokens.begin(),
	                                               out_tokens.begin() + out_tokens_produced);
	const std::string output_text = tokenizer.Decode(output_token_vector);
	std::printf("output_text_bytes (%zu):\n", output_text.size());
	if (!output_text.empty()) std::fwrite(output_text.data(), 1, output_text.size(), stdout);
	std::printf("\noutput_text_end\n");
	std::printf("stop_reason: %d\n", static_cast<int>(stop_reason));

	if (!dump_logits_path.empty()) {
		if (!DumpLogitRows(dump_logits_path.c_str(), out_logit_rows.data(), out_tokens_produced,
		                    static_cast<size_t>(model_view.config.vocab_size))) {
			std::fprintf(stderr, "FAILED at stage=dump_logits: could not write \"%s\"\n",
			             dump_logits_path.c_str());
			return 1;
		}
		std::printf("logit_rows_dumped: %zu rows x %u vocab -> %s\n", out_tokens_produced,
		            model_view.config.vocab_size, dump_logits_path.c_str());
	}

	const auto t_end = std::chrono::steady_clock::now();
	std::printf("wall_time_seconds: %.3f\n", std::chrono::duration<double>(t_end - t_start).count());
	return 0;
}
