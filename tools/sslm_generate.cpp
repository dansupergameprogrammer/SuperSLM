// sslm_generate.cpp -- the real-model smoke driver (T-1646 continuation build,
// 2026-08-01). Loads a `.sslm` model artifact and a `.sslm` tokenizer artifact,
// encodes a prompt, and (once a model artifact actually passes SslmModel::Load)
// is where the LayerWeights[] marshaling adapter and RunGreedyDecodeLoop call
// belong.
//
// THIS BUILD'S STATUS, STATED PLAINLY: the LayerWeights marshaling adapter is
// NOT implemented below. Two independent, load-bearing blockers were found
// while standing this driver up against the real Qwen2.5-1.5B-Instruct w8a8
// calibrated artifact, and both are upstream of this file -- neither is a
// thing this driver, or an adapter written inside it, can work around without
// inventing production behavior that does not exist:
//
//   1. SslmModel::Load rejects the artifact outright, deterministically, on
//      every conversion attempted: BiasCodeOutOfDomain, because ~97.6% of the
//      BIA1 dynamic-bias codes the calibration pipeline produced (every one
//      of the 84 q/k/v-projection sites, all 28 layers) carry magnitudes up
//      to ~2^46.6, against the format's documented bound of +/-INT32_MAX
//      (~2^31). See the build log for the full scan. No artifact that fails
//      this gate can ever reach a LayerWeights marshaling step, because
//      SslmModel::Load is the one entry point every consumer (this driver
//      included) must go through, and it returns before any tensor is
//      exposed.
//   2. Independent of (1): LayerWeights (forward_sites.h) bakes q_ln2/
//      q_b_iexp/q_c_iexp in as fixed PER-LAYER scalars, but the reference
//      engine (Tools/superslm_spike/dynamic_engine.py:410-428) derives them
//      PER QUERY, per decode step, from the runtime q-projection's own
//      carried scale composed with an artifact-carried per-(layer, kv-head)
//      static factor. forward_sites.h's own header comment states plainly
//      that the C++ site which would form these three constants from a
//      per-query carried scale "is not yet built anywhere in this tree."
//      Baking a static per-layer value into an adapter would not be the
//      reference's actual per-query quantity, and presenting it as if it
//      were is exactly the "quietly invent a design to fill the gap" failure
//      the builder persona's own discipline exists to prevent.
//
// So this driver does the part that IS well-defined today -- load the
// tokenizer, encode the prompt, load the model artifact through the real
// production entry point -- and fails loudly and specifically (status name,
// diagnostic, which stage) rather than press on past a rejection. Once the
// upstream calibration defect in (1) is fixed and (2)'s per-query derivation
// site is built, the marshaling adapter and the RunGreedyDecodeLoop call are
// the next thing to add here.
//
// Usage: sslm_generate <model.sslm> <tokenizer.sslm> "<prompt>" [--max-new N]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/model.h"
#include "superslm/tokenizer.h"

using namespace superslm;

namespace {

bool ReadFile(const char* path, std::vector<uint8_t>& out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	std::streamoff size = f.tellg();
	if (size < 0) return false;
	f.seekg(0, std::ios::beg);
	out.resize(static_cast<size_t>(size));
	if (size > 0) f.read(reinterpret_cast<char*>(out.data()), size);
	return static_cast<bool>(f) || f.eof();
}

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" [--max-new N]\n",
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
	size_t max_new_tokens = 32;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
			max_new_tokens = static_cast<size_t>(std::stoul(argv[++i]));
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
		std::fprintf(stderr,
		             "FAILED at stage=tokenizer_artifact_open: status=%s diagnostic=\"%s\"\n",
		             SslmStatusName(tok_open_err.code), tok_open_err.message.c_str());
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_view_open: diagnostic=\"%s\"\n",
		             tok_err.c_str());
		return 1;
	}
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);

	std::printf("prompt: %s\n", prompt.c_str());
	std::printf("prompt_tokens (%zu):", prompt_tokens.size());
	for (int32_t t : prompt_tokens) std::printf(" %d", t);
	std::printf("\n");

	// --- Stage 2: load the model artifact through the real production ------
	// entry point (SslmModel::Load) -- the same call sslm_verify makes, and
	// the same call any future adapter here would sit downstream of. This is
	// the gate the real Qwen2.5-1.5B conversion does not clear (see the file
	// header above and the build log).
	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED at stage=model_file_read: could not read \"%s\"\n",
		             model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status = SslmModel::Load(
	    model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr,
		             "FAILED at stage=model_load: status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		std::fprintf(stderr,
		             "SslmModel::Load rejected this artifact before any weight was marshaled; "
		             "no forward pass was attempted.\n");
		return 1;
	}

	// --- Stage 3: NOT REACHED by the real Qwen2.5-1.5B artifact today. ------
	// This is where the LayerWeights[] marshaling adapter and
	// RunGreedyDecodeLoop call belong once stage 2 actually returns Ok and
	// the per-query q_ln2/q_b_iexp/q_c_iexp derivation site (blocker 2 above)
	// exists in production. Left unimplemented rather than filled with an
	// invented approximation.
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u "
	            "intermediate=%u vocab=%u context_cap=%u\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim, model_view.config.intermediate_size,
	            model_view.config.vocab_size, model_view.config.context_cap);
	std::fprintf(stderr,
	             "FAILED at stage=not_implemented: the model artifact loaded, but the "
	             "LayerWeights marshaling adapter and RunGreedyDecodeLoop call are not yet "
	             "written (see this file's header comment) -- max_new_tokens=%zu was not run\n",
	             max_new_tokens);

	const auto t_end = std::chrono::steady_clock::now();
	const double wall_seconds =
	    std::chrono::duration<double>(t_end - t_start).count();
	std::printf("wall_time_seconds: %.3f\n", wall_seconds);
	return 1;
}
