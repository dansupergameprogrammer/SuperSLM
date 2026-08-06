// sslm_clamp_census.cpp -- T-1781 cells C8/C9: does RmsNormSite's int8
// floor-at-1 clamp, or RopeApplySite's ClampRopeCode, ever actually engage
// on real data -- and how often.
//
// WHY THIS TOOL EXISTS. D-SLM770 (Claude/Decisions/DecisionLog.md:16883-
// 16891) rules out the missing-epsilon hypothesis for a signature this
// campaign was chasing and states in its own text: "whether the int8
// floor-at-1 clamp ever actually engages on real data is unmeasured -- no
// site-dump instrumentation captures the intermediate root." T-1689 (board
// row) closed the adjacent K/V-landing saturation census (zero saturation,
// nine prompts, 28 layers) but explicitly left `ClampRopeCode` itself
// uninstrumented. Both are coverage questions -- does a real, exercised
// clamp path ever fire -- prior to any correctness question about what it
// computes when it does.
//
// This tool runs the PRODUCTION decode path (RunGreedyDecodeLoop, unchanged)
// over a real prompt and prints the two clamps' own engagement counters
// (forward_sites.h's GetRmsNormFloorClampEngagements/Calls,
// GetRopeClampEngagements/Calls -- T-1781, this ticket) after the run. It
// performs no correctness comparison of its own: engagement count is a
// pass/fail-free observation, not a pass/fail check, and this tool exits 0
// regardless of what the counters read (a nonzero engagement count is not a
// failure -- it is exactly the finding D-SLM770/T-1689 asked this ticket to
// produce).
//
// Usage: sslm_clamp_census <model.sslm> <tokenizer.sslm> "<prompt>" [--max-new N]
//
// This tool runs ONE prompt per process invocation and prints that
// invocation's own engagement/call counts ("_this_run" fields, computed as
// a before/after delta so --reset never matters for a single-prompt run --
// it is provided only in case this tool is later extended to run more than
// one prompt per process). A driving script sums "_this_run" across
// per-prompt invocations to get the population total.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
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
void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" [--max-new N] [--reset]\n",
	             argv0);
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
	size_t max_new = 32;
	bool reset = false;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
			max_new = static_cast<size_t>(std::stoul(argv[++i]));
		} else if (std::strcmp(argv[i], "--reset") == 0) {
			reset = true;
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}
	if (reset) ResetClampCensusCounters();

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
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_encode: prompt encoded to zero tokens\n");
		return 1;
	}

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

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n",
			             l, marshal_err.c_str());
			return 1;
		}
	}

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
	const size_t vocab_size_z = static_cast<size_t>(model_view.config.vocab_size);

	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	std::vector<int32_t> out_tokens(max_new);
	std::vector<int32_t> out_logit_rows(max_new * vocab_size_z);
	size_t tokens_produced = 0;
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;
	const int32_t stop_ids[2] = {151645, 151643};

	const uint64_t rms_calls_before = GetRmsNormFloorClampCalls();
	const uint64_t rms_eng_before = GetRmsNormFloorClampEngagements();
	const uint64_t rope_calls_before = GetRopeClampCalls();
	const uint64_t rope_eng_before = GetRopeClampEngagements();

	const SslmForwardStatus status = RunGreedyDecodeLoop(
	    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim, num_kv_heads,
	    model_view.config.intermediate_size, context_cap, model_view.rope_tables, prompt_tokens.data(),
	    prompt_tokens.size(), embed_weights, embed_site_constant, final_norm_gain.data(),
	    final_norm_site_constant, head_weights, static_cast<int32_t>(model_view.config.vocab_size),
	    stop_ids, 2, max_new, workspace.data(), workspace.size(), out_tokens.data(),
	    out_logit_rows.data(), out_tokens.size(), &tokens_produced, &stop_reason,
	    model_view.config.kv_precision);
	if (status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=production_decode: status=%s\n",
		             SslmForwardStatusName(status));
		return 1;
	}

	const uint64_t rms_calls_this_run = GetRmsNormFloorClampCalls() - rms_calls_before;
	const uint64_t rms_eng_this_run = GetRmsNormFloorClampEngagements() - rms_eng_before;
	const uint64_t rope_calls_this_run = GetRopeClampCalls() - rope_calls_before;
	const uint64_t rope_eng_this_run = GetRopeClampEngagements() - rope_eng_before;

	// THIS_RUN figures are this process's own delta (each process starts
	// with fresh, zero-initialized globals, so THIS_RUN == the process
	// total regardless of --reset) -- a driving script sums THIS_RUN across
	// per-prompt invocations for the population total; --reset only matters
	// if this tool is ever extended to run more than one prompt in a single
	// process.
	std::printf(
	    "census: tokens_produced=%zu prompt_tokens=%zu "
	    "rmsnorm_floor_clamp_engagements_this_run=%llu rmsnorm_floor_clamp_calls_this_run=%llu "
	    "rope_clamp_engagements_this_run=%llu rope_clamp_calls_this_run=%llu\n",
	    tokens_produced, prompt_tokens.size(),
	    static_cast<unsigned long long>(rms_eng_this_run),
	    static_cast<unsigned long long>(rms_calls_this_run),
	    static_cast<unsigned long long>(rope_eng_this_run),
	    static_cast<unsigned long long>(rope_calls_this_run));
	return 0;
}
