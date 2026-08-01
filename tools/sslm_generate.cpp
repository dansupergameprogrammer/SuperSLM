// sslm_generate.cpp -- the real-model smoke driver (T-1652 continuation build,
// 2026-08-01). Loads a `.sslm` model artifact and a `.sslm` tokenizer artifact,
// encodes a prompt, and (once a model artifact actually passes SslmModel::Load)
// is where the LayerWeights[] marshaling adapter and RunGreedyDecodeLoop call
// belong.
//
// THIS BUILD'S STATUS, STATED PLAINLY. The BiasCodeOutOfDomain blocker that
// stopped the prior pass (T-1646) is resolved ON THIS BRANCH ONLY: `src/
// model.cpp`'s `kBia1MagnitudeBound` carries a SMOKE-BRANCH-ONLY relaxation,
// marked at its definition, and SslmModel::Load now accepts the real
// Qwen2.5-1.5B-Instruct w8a8 calibrated artifact (verified: `sslm_inspect`
// reports "MODELVIEW OK" against `qwen2.5-1.5b-instruct.sslm`). See that
// site's own comment for why the relaxation changes no computed value in
// this tree today, and why it must never reach `main`.
//
// The LayerWeights marshaling adapter is STILL NOT implemented below.
// Standing it up against the real, now-loadable artifact surfaced a THIRD
// blocker, structural rather than data-domain, and more fundamental than the
// two the prior pass found -- all three are upstream of this file, and none
// is a thing this driver, or an adapter written inside it, can work around
// without a production change to RunLayerLoop itself (out of this build's
// scope: "no production changes beyond the marked gate relaxation; if the
// adapter needs another production change, STOP and report"):
//
//   1. [RESOLVED on this branch] BIA1 magnitude gate -- see src/model.cpp.
//
//   2. [Confirmed, not resolved] `LayerWeights` (forward_sites.h) bakes
//      q_ln2/q_b_iexp/q_c_iexp in as fixed PER-LAYER scalars. Reading the
//      real reference (D:\Wizard\Tools\superslm_spike\dynamic_engine.py:
//      410-428) end to end shows the true derivation is
//      `iexp_scale_constants(carried_scale_product(q_scale[t], softmax_static), ...)`
//      -- `softmax_static` IS artifact data (`CompositionConstants["layer{L}.
//      softmax_khead{h}"]`, confirmed present for all 28 layers of this
//      artifact, and identical across both kv-heads in every layer checked),
//      but `q_scale[t]` is the q-projection's own dynamic, PER-DECODE-STEP
//      carried scale -- the output of `RequantChainChecked`'s C19/C20/C21
//      max-abs-reduce chain on that token's own activations
//      (checked_chain_funnel.h's own documented contract: the funnel's
//      `*out_scale` is a product of the artifact's fixed site constant AND
//      the row's own data-dependent reduction, never the site constant
//      alone). RunLayerLoop already computes this value every layer (the
//      discarded `CarriedScale q_scale` local, forward_sites.cpp) but never
//      composes it with `softmax_static` or calls `iexp_scale_constants` --
//      confirming forward_sites.h's own header comment that this derivation
//      site does not exist in this tree. A per-layer constant computed
//      offline from `softmax_static` alone (the only defensible offline
//      substitution) would silently omit the query's own dynamic factor on
//      every token; wiring the real composition is a RunLayerLoop change,
//      not an adapter field.
//
//   3. [NEW, and the one actually blocking this build] `RunLayerLoop` is
//      MHA-only, and the real artifact is GQA. Read at source, not inferred:
//      `num_heads` (forward_sites.cpp line ~890) is computed as
//      `hidden_size / head_dim` -- the QUERY head count (12 for this model)
//      -- and is the ONLY head count RunLayerLoop knows; there is no
//      `num_key_value_heads` parameter anywhere in its signature. The k/v
//      projection call (`GemmInt8AccumulateRow(normed, lw.k_weight,
//      hidden_size, hidden_size, kacc)`) assumes `k_weight`/`v_weight` are
//      `[hidden_size, hidden_size]`, and the attention loop reads
//      `KeyRow`/`ValueRow` at the SAME head index `h` used for the query
//      head, one-to-one. The real artifact's `layer{L}.k_proj`/`v_proj`
//      weights are `[256, 1536]` (`num_key_value_heads * head_dim = 2 * 128`,
//      confirmed via `sslm_inspect`/a manifest dump against the real file,
//      against `num_attention_heads=12, num_key_value_heads=2` in CFG1) --
//      256 output channels, not 1536. Feeding that matrix through a call
//      shaped for a 1536-wide output reads past the tensor's real extent.
//      The KV-landing constant arrays compound this: `LayerWeights.
//      kv_landing_r_t_k`/`_e_t_k`/`_r_t_v`/`_e_t_v` are documented as
//      "num_heads entries each" (12), but the artifact only carries
//      KvLandingScales/Reciprocals entries for 2 kv-heads
//      (`layer0.k_head0`/`k_head1`, confirmed -- no `k_head2..k_head11`
//      exist to fill the other ten). There is no artifact data to populate
//      an adapter field at that width, because the real model has no such
//      quantity. Making RunLayerLoop GQA-correct means: adding a
//      `num_key_value_heads` (or equivalent) parameter, resizing the k/v
//      projection call to `num_key_value_heads * head_dim` outputs, and
//      grouping query heads onto kv-heads at the attention read
//      (`kv_head = head / (num_attention_heads / num_key_value_heads)`,
//      matching `dynamic_engine.py`'s own `group = num_attention_heads //
//      num_key_value_heads` / `kv_head = head // group`). That is a real
//      change to RunLayerLoop's control flow and LayerWeights' shape, not an
//      adapter-side field derivation -- exactly the "another production
//      change" this build stops on rather than making.
//
// So this driver does the part that IS well-defined today -- load the
// tokenizer, encode the prompt, and load the model artifact through the real
// production entry point, which now succeeds on this branch -- and fails
// loudly and specifically past that point (status name, diagnostic, which
// stage) rather than press on past blocker 3 with an adapter that cannot be
// correct for this model's architecture. Once RunLayerLoop is made
// GQA-capable and (2)'s per-query i-exp composition is wired in, the
// marshaling adapter and the RunGreedyDecodeLoop call are the next thing to
// add here.
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
	// the same call any future adapter here would sit downstream of. On this
	// branch (the smoke-branch-only BIA1 relaxation, src/model.cpp), the real
	// Qwen2.5-1.5B-Instruct conversion now clears this gate.
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

	// --- Stage 3: NOT REACHED. The model artifact loads (stage 2 is Ok on --
	// this branch), but RunLayerLoop's own attention/K-V design is MHA-only
	// (blocker 3, this file's header) and this artifact is GQA
	// (num_attention_heads=12, num_key_value_heads=2) -- writing the
	// LayerWeights[] adapter and calling RunGreedyDecodeLoop here would mean
	// feeding a [256,1536] k/v weight matrix and 2 kv-head landing constants
	// into a loop built for a [1536,1536] matrix and 12 one-to-one head
	// slots, which is not an adapter decision, it is silently wrong output.
	// Left unimplemented rather than filled with an invented approximation.
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u "
	            "intermediate=%u vocab=%u context_cap=%u\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim, model_view.config.intermediate_size,
	            model_view.config.vocab_size, model_view.config.context_cap);
	std::fprintf(stderr,
	             "FAILED at stage=not_implemented: the model artifact loaded (BIA1 gate "
	             "relaxation, this branch only), but RunLayerLoop is MHA-only and this "
	             "artifact is GQA (heads=%u/%u) -- see this file's header comment, blocker 3 "
	             "-- max_new_tokens=%zu was not run\n",
	             model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	             max_new_tokens);

	const auto t_end = std::chrono::steady_clock::now();
	const double wall_seconds =
	    std::chrono::duration<double>(t_end - t_start).count();
	std::printf("wall_time_seconds: %.3f\n", wall_seconds);
	return 1;
}
