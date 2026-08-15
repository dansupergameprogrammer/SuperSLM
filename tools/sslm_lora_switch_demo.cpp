// sslm_lora_switch_demo.cpp -- T-2102 (RESUME-2026-08-14-gpu-throughput.md §4, D-SLM3304): the
// demonstration vehicle Dan's own dispatch asked for -- "the demonstration that runtime LoRAs do
// real work and how well they do vs a baked LoRA, under standard allocations."
//
// One process, three model configurations, run back to back so the "switch" is a real in-process
// event rather than three separate CLI invocations:
//   BASE     -- <base.sslm> with no adapter bound.
//   RUNTIME  -- the SAME <base.sslm>, with <adapter.sslm> loaded and bound via
//               superslm_adapter::LoadAdapterArtifact / ApplyAdapterToLayers (T-2102's own loader).
//   BAKED    -- <baked.sslm>, the merged-and-converted checkpoint (the SAME adapted weights, folded
//               into the base offline rather than composed at runtime).
//
// For each of a small, fixed prompt set (--prompts, comma-separated, or this file's own default
// three), all three configurations run the SAME prompt through the SAME greedy decode loop and
// print: the decoded text, the token ids, and per-token decode-only wall time (the full
// RunGreedyDecodeLoop call divided by tokens actually produced -- prefill is included, matching
// what an operator actually experiences per call, and is reported alongside the token count so a
// reader can see it is not being hidden).
//
// RUNTIME's own logits are additionally compared against BAKED's, token position by token
// position, against the SAME prompt -- max|delta| and the fraction of positions where the argmax
// token disagrees, which is what "agreement, and whether any divergence is within int8 rounding
// expectations" (T-2102's own brief) asks for. This is a comparison of two INDEPENDENTLY COMPUTED
// quantities (runtime's own composed int8 accumulator path vs baked's own merged-then-quantized
// path) -- neither is derived from the other, so the comparison is not circular.
//
// Timing: model load (BASE, BAKED) and adapter load+apply (RUNTIME's own extra step) are each
// repeated >=3 times (--timing-reps, default 3) and reported as min/median/max -- this machine may
// be shared with unrelated CPU/GPU load tonight (RESUME-2026-08-14 §6), so the spread is the
// honest unit, not a single point figure.
//
// Usage: sslm_lora_switch_demo <base.sslm> <tokenizer.sslm> <adapter.sslm> <baked.sslm>
//          [--max-new N] [--timing-reps N] [--prompts "a|b|c"] [--json <out.json>]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/tokenizer.h"
#include "sslm_marshal.h"
#include "sslm_adapter_loader.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;
using superslm_adapter::AdapterHandle;
using superslm_adapter::AdapterLoadStatus;
using superslm_adapter::ApplyAdapterToLayers;
using superslm_adapter::BaseModelGeometry;
using superslm_adapter::LoadAdapterArtifact;

namespace {

// The stop ids sslm_generate.cpp's own header comment documents: Qwen2.5-instruct's <|im_end|>/
// <|endoftext|>. Fixed here rather than a flag -- this demo targets exactly this model family
// (the shopkeeper adapters are trained against qwen2.5-1.5b-instruct).
constexpr int32_t kStopIds[2] = {151645, 151643};

struct LoadedModel {
	std::vector<uint8_t> bytes;
	SslmModelView view;
	std::vector<LayerBacking> backings;
	std::vector<LayerWeights> layers;
	uint32_t num_hidden_layers = 0;
	uint32_t num_heads = 0;
	uint32_t num_kv_heads = 0;
	size_t hidden_size = 0;
	const int8_t* embed_weights = nullptr;
	const int8_t* head_weights = nullptr;
	std::vector<int32_t> final_norm_gain;
	CarriedScale embed_site_constant;
	CarriedScale final_norm_site_constant;
	int64_t context_cap = 0;
};

// Loads and fully marshals one model artifact -- the same Stage 1-3 sequence sslm_generate.cpp's
// own main() performs, factored so this driver can repeat it (timing) and run it three times
// (BASE/RUNTIME's base/BAKED) without duplicating the marshaling logic three times over.
bool LoadAndMarshal(const std::string& path, LoadedModel& out, std::string* err) {
	if (!ReadFile(path.c_str(), out.bytes)) {
		*err = "could not read " + path;
		return false;
	}
	std::string load_err;
	if (SslmModel::Load(out.bytes.data(), out.bytes.size(), out.view, &load_err) != SslmModelStatus::Ok) {
		*err = "model load rejected: " + load_err;
		return false;
	}
	out.num_heads = out.view.config.num_attention_heads;
	out.num_kv_heads = out.view.config.num_key_value_heads;
	out.num_hidden_layers = out.view.config.num_hidden_layers;
	out.hidden_size = out.view.config.hidden_size;
	out.context_cap = static_cast<int64_t>(out.view.config.context_cap);

	out.backings.assign(out.num_hidden_layers, LayerBacking{});
	out.layers.assign(out.num_hidden_layers, LayerWeights{});
	for (uint32_t l = 0; l < out.num_hidden_layers; ++l) {
		std::string merr;
		if (!MarshalLayer(out.view, l, out.num_heads, out.num_kv_heads, out.backings[l], out.layers[l], &merr)) {
			*err = "layer " + std::to_string(l) + " marshal failed: " + merr;
			return false;
		}
	}

	const SslmTensorView* embed_w = out.view.weights.Tensor("embed");
	const SslmTensorView* final_gain_w = out.view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		*err = "missing embed or final_norm.gain tensor";
		return false;
	}
	out.final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	out.embed_site_constant = ReadCarriedScale(out.view.composition_constants, "embed", &ok);
	out.final_norm_site_constant = ReadCarriedScale(out.view.composition_constants, "final_norm", &ok);
	if (!ok) {
		*err = "missing embed/final_norm site constant";
		return false;
	}
	out.embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	if (out.view.config.tie_word_embeddings) {
		out.head_weights = out.embed_weights;
	} else {
		const SslmTensorView* lm_head_w = out.view.weights.Tensor("lm_head");
		if (!lm_head_w) {
			*err = "tie_word_embeddings=0 but no lm_head tensor present";
			return false;
		}
		out.head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}
	return true;
}

struct GenResult {
	std::vector<int32_t> prompt_tokens;
	std::vector<int32_t> out_tokens;
	std::vector<int32_t> out_logit_rows;  // rows_produced * vocab_size, row-major
	size_t tokens_produced = 0;
	uint32_t vocab_size = 0;
	SslmForwardStatus status = SslmForwardStatus::Ok;
	double decode_seconds = 0.0;  // the WHOLE RunGreedyDecodeLoop call (prefill + every step)
};

// Runs one prompt to completion under `m`'s own layers[] (which already carries whatever
// adapter -- or none -- the caller bound via ApplyAdapterToLayers before this call). Times only
// the decode call itself, not tokenization/marshaling (those are one-time model-load costs,
// already reported separately by the timing harness below).
bool GenerateOne(const LoadedModel& m, const TokenizerView& tok, const std::string& prompt,
                  size_t max_new_tokens, GenResult& out, std::string* err) {
	out.prompt_tokens = tok.Encode(prompt);
	if (out.prompt_tokens.empty()) {
		*err = "prompt encoded to zero tokens";
		return false;
	}
	out.vocab_size = m.view.config.vocab_size;
	const size_t kv_bytes = static_cast<size_t>(m.num_hidden_layers) * static_cast<size_t>(m.context_cap) *
	                         m.num_kv_heads * m.view.config.head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(m.hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	out.out_tokens.assign(max_new_tokens, 0);
	out.out_logit_rows.assign(max_new_tokens * static_cast<size_t>(out.vocab_size), 0);
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;

	const auto t0 = std::chrono::steady_clock::now();
	out.status = RunGreedyDecodeLoop(
	    seq, m.layers.data(), m.num_hidden_layers, m.hidden_size, m.view.config.head_dim, m.num_kv_heads,
	    m.view.config.intermediate_size, m.context_cap, m.view.rope_tables, out.prompt_tokens.data(),
	    out.prompt_tokens.size(), m.embed_weights, m.embed_site_constant, m.final_norm_gain.data(),
	    m.final_norm_site_constant, m.head_weights, static_cast<int32_t>(out.vocab_size), kStopIds, 2,
	    max_new_tokens, workspace.data(), workspace.size(), out.out_tokens.data(), out.out_logit_rows.data(),
	    out.out_tokens.size(), &out.tokens_produced, &stop_reason, m.view.config.kv_precision,
	    m.view.option_g_fused_k_landing);
	const auto t1 = std::chrono::steady_clock::now();
	out.decode_seconds = std::chrono::duration<double>(t1 - t0).count();
	if (out.status != SslmForwardStatus::Ok) {
		*err = std::string("decode failed: ") + SslmForwardStatusName(out.status);
		return false;
	}
	return true;
}

double Median(std::vector<double> v) {
	std::sort(v.begin(), v.end());
	const size_t n = v.size();
	if (n == 0) return 0.0;
	return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct TimingStats {
	double min_s = 0, median_s = 0, max_s = 0;
	std::vector<double> samples;
};

TimingStats Summarize(const std::vector<double>& samples) {
	TimingStats t;
	t.samples = samples;
	t.min_s = *std::min_element(samples.begin(), samples.end());
	t.max_s = *std::max_element(samples.begin(), samples.end());
	t.median_s = Median(samples);
	return t;
}

void PrintTiming(const char* label, const TimingStats& t) {
	std::printf("  %-28s min=%.4fs median=%.4fs max=%.4fs  (samples:", label, t.min_s, t.median_s, t.max_s);
	for (double s : t.samples) std::printf(" %.4f", s);
	std::printf(")\n");
}

std::string JsonEscape(const std::string& s) {
	std::string out;
	for (char c : s) {
		if (c == '"' || c == '\\') out += '\\';
		if (c == '\n') { out += "\\n"; continue; }
		out += c;
	}
	return out;
}

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <base.sslm> <tokenizer.sslm> <adapter.sslm> <baked.sslm> "
	             "[--max-new N] [--timing-reps N] [--prompts \"a|b|c\"] [--json <out.json>]\n",
	             argv0);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string base_path = argv[1];
	const std::string tok_path = argv[2];
	const std::string adapter_path = argv[3];
	const std::string baked_path = argv[4];
	size_t max_new_tokens = 24;
	int timing_reps = 3;
	std::string json_out_path;
	std::vector<std::string> prompts = {
	    "Customer: Do you have any healing potions?\nShopkeeper:",
	    "Customer: What's the price of that sword?\nShopkeeper:",
	    "Customer: I need something to protect me from goblins.\nShopkeeper:",
	};
	for (int i = 5; i < argc; ++i) {
		if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
			max_new_tokens = static_cast<size_t>(std::stoul(argv[++i]));
		} else if (std::strcmp(argv[i], "--timing-reps") == 0 && i + 1 < argc) {
			timing_reps = std::stoi(argv[++i]);
		} else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
			json_out_path = argv[++i];
		} else if (std::strcmp(argv[i], "--prompts") == 0 && i + 1 < argc) {
			prompts.clear();
			std::string spec = argv[++i];
			size_t pos = 0;
			while (pos < spec.size()) {
				size_t bar = spec.find('|', pos);
				prompts.push_back(spec.substr(pos, bar == std::string::npos ? std::string::npos : bar - pos));
				if (bar == std::string::npos) break;
				pos = bar + 1;
			}
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}
	if (timing_reps < 1) timing_reps = 1;

	std::printf("================================================================================\n");
	std::printf("T-2102 runtime-LoRA switch demonstration\n");
	std::printf("  base:    %s\n", base_path.c_str());
	std::printf("  adapter: %s\n", adapter_path.c_str());
	std::printf("  baked:   %s\n", baked_path.c_str());
	std::printf("  max_new_tokens=%zu  timing_reps=%d  prompts=%zu\n", max_new_tokens, timing_reps,
	            prompts.size());
	std::printf("================================================================================\n\n");

	// --- Tokenizer (shared across base/runtime/baked -- same architecture/vocab). -------------
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tok_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED reading tokenizer %s\n", tok_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED opening tokenizer artifact: %s\n", SslmStatusName(tok_err.code));
		return 1;
	}
	TokenizerView tokenizer;
	std::string tv_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tv_err)) {
		std::fprintf(stderr, "FAILED opening tokenizer view: %s\n", tv_err.c_str());
		return 1;
	}

	// --- Timing: model load (BASE, BAKED) and adapter load+apply, each repeated. --------------
	std::vector<double> base_load_times, baked_load_times, adapter_load_times;
	LoadedModel base_model, baked_model;
	AdapterHandle adapter_handle;

	std::printf("--- timing: cold load, %d repetitions each -------------------------------------\n",
	            timing_reps);
	for (int rep = 0; rep < timing_reps; ++rep) {
		LoadedModel m;
		std::string err;
		const auto t0 = std::chrono::steady_clock::now();
		const bool ok = LoadAndMarshal(base_path, m, &err);
		const auto t1 = std::chrono::steady_clock::now();
		if (!ok) {
			std::fprintf(stderr, "FAILED loading base model (rep %d): %s\n", rep, err.c_str());
			return 1;
		}
		base_load_times.push_back(std::chrono::duration<double>(t1 - t0).count());
		if (rep == timing_reps - 1) base_model = std::move(m);  // keep the LAST load for generation
	}
	for (int rep = 0; rep < timing_reps; ++rep) {
		LoadedModel m;
		std::string err;
		const auto t0 = std::chrono::steady_clock::now();
		const bool ok = LoadAndMarshal(baked_path, m, &err);
		const auto t1 = std::chrono::steady_clock::now();
		if (!ok) {
			std::fprintf(stderr, "FAILED loading baked model (rep %d): %s\n", rep, err.c_str());
			return 1;
		}
		baked_load_times.push_back(std::chrono::duration<double>(t1 - t0).count());
		if (rep == timing_reps - 1) baked_model = std::move(m);
	}
	for (int rep = 0; rep < timing_reps; ++rep) {
		SslmArtifact base_artifact;
		SslmError aerr;
		if (SslmArtifact::OpenFromMemory(base_model.bytes.data(), base_model.bytes.size(), base_artifact,
		                                  &aerr) != SslmStatus::Ok) {
			std::fprintf(stderr, "FAILED re-opening base artifact for base-hash: %s\n",
			             SslmStatusName(aerr.code));
			return 1;
		}
		BaseModelGeometry geo;
		geo.num_hidden_layers = base_model.num_hidden_layers;
		geo.hidden_size = base_model.hidden_size;
		geo.intermediate_size = base_model.view.config.intermediate_size;
		geo.kv_hidden_size = static_cast<uint64_t>(base_model.num_kv_heads) * base_model.view.config.head_dim;
		geo.base_artifact_hash = base_artifact.RawIntegrityHash();

		AdapterHandle h;
		std::string err;
		const auto t0 = std::chrono::steady_clock::now();
		const auto status = LoadAdapterArtifact(adapter_path, geo, h, &err);
		const auto t1 = std::chrono::steady_clock::now();
		if (status != AdapterLoadStatus::Ok) {
			std::fprintf(stderr, "FAILED loading adapter (rep %d): %s (%s)\n", rep,
			             superslm_adapter::AdapterLoadStatusName(status), err.c_str());
			return 1;
		}
		adapter_load_times.push_back(std::chrono::duration<double>(t1 - t0).count());
		if (rep == timing_reps - 1) adapter_handle = std::move(h);
	}

	const TimingStats base_load_stats = Summarize(base_load_times);
	const TimingStats baked_load_stats = Summarize(baked_load_times);
	const TimingStats adapter_load_stats = Summarize(adapter_load_times);
	PrintTiming("base model cold load", base_load_stats);
	PrintTiming("baked model cold load", baked_load_stats);
	PrintTiming("adapter load+apply (switch cost)", adapter_load_stats);
	std::printf("\n");

	// --- Generation: base-only, runtime(base+adapter), baked-only, per prompt. ----------------
	// Runtime generation reuses `base_model`'s own layers[] directly (bind the adapter
	// immediately before the runtime call, unbind immediately after) rather than a second,
	// separately-loaded model object -- this IS the switch: the same in-memory base, the adapter
	// pointer swapped in and out, never a second model load. `LoadedModel` is intentionally not
	// copied for this (its pointer fields point into its own `bytes` buffer; a struct copy would
	// leave them dangling into the wrong buffer).
	std::vector<double> base_tok_per_sec, runtime_tok_per_sec, baked_tok_per_sec;
	std::vector<double> logit_max_abs_diffs;
	std::vector<int> argmax_agree_counts, argmax_total_counts;

	std::ostringstream json;
	json << "{\n \"prompts\": [\n";

	for (size_t pi = 0; pi < prompts.size(); ++pi) {
		const std::string& prompt = prompts[pi];
		std::printf("--- prompt %zu: %s\n", pi, prompt.c_str());

		GenResult base_res, runtime_res, baked_res;
		std::string err;

		if (!GenerateOne(base_model, tokenizer, prompt, max_new_tokens, base_res, &err)) {
			std::fprintf(stderr, "FAILED base generation: %s\n", err.c_str());
			return 1;
		}
		// Runtime: base_model's layers[], with adapter bound (base_model.layers[l].adapter is set
		// by ApplyAdapterToLayers below immediately before this call; cleared immediately after,
		// so `base_res` above was genuinely base-only, not accidentally adapter-bound).
		ApplyAdapterToLayers(&adapter_handle, base_model.layers.data(), base_model.num_hidden_layers);
		if (!GenerateOne(base_model, tokenizer, prompt, max_new_tokens, runtime_res, &err)) {
			std::fprintf(stderr, "FAILED runtime generation: %s\n", err.c_str());
			return 1;
		}
		ApplyAdapterToLayers(nullptr, base_model.layers.data(), base_model.num_hidden_layers);  // unbind

		if (!GenerateOne(baked_model, tokenizer, prompt, max_new_tokens, baked_res, &err)) {
			std::fprintf(stderr, "FAILED baked generation: %s\n", err.c_str());
			return 1;
		}

		const std::string base_text = tokenizer.Decode(
		    std::vector<int32_t>(base_res.out_tokens.begin(), base_res.out_tokens.begin() +
		                                                          static_cast<long>(base_res.tokens_produced)));
		const std::string runtime_text = tokenizer.Decode(std::vector<int32_t>(
		    runtime_res.out_tokens.begin(), runtime_res.out_tokens.begin() + static_cast<long>(runtime_res.tokens_produced)));
		const std::string baked_text = tokenizer.Decode(std::vector<int32_t>(
		    baked_res.out_tokens.begin(), baked_res.out_tokens.begin() + static_cast<long>(baked_res.tokens_produced)));

		std::printf("  BASE    (%zu tok, %.4fs, %.2f tok/s): %s\n", base_res.tokens_produced,
		            base_res.decode_seconds, base_res.tokens_produced / std::max(1e-9, base_res.decode_seconds),
		            base_text.c_str());
		std::printf("  RUNTIME (%zu tok, %.4fs, %.2f tok/s): %s\n", runtime_res.tokens_produced,
		            runtime_res.decode_seconds,
		            runtime_res.tokens_produced / std::max(1e-9, runtime_res.decode_seconds), runtime_text.c_str());
		std::printf("  BAKED   (%zu tok, %.4fs, %.2f tok/s): %s\n", baked_res.tokens_produced,
		            baked_res.decode_seconds,
		            baked_res.tokens_produced / std::max(1e-9, baked_res.decode_seconds), baked_text.c_str());

		const bool switch_did_real_work = (runtime_res.tokens_produced != base_res.tokens_produced) ||
		                                   (runtime_res.out_tokens != base_res.out_tokens);
		std::printf("  switch changed output vs base: %s\n", switch_did_real_work ? "YES" : "NO");

		base_tok_per_sec.push_back(base_res.tokens_produced / std::max(1e-9, base_res.decode_seconds));
		runtime_tok_per_sec.push_back(runtime_res.tokens_produced / std::max(1e-9, runtime_res.decode_seconds));
		baked_tok_per_sec.push_back(baked_res.tokens_produced / std::max(1e-9, baked_res.decode_seconds));

		// --- runtime vs baked: token-level agreement + logit-level divergence. -----------------
		const size_t common_n = std::min(runtime_res.tokens_produced, baked_res.tokens_produced);
		int agree = 0;
		double max_abs_logit_diff = 0.0;
		for (size_t t = 0; t < common_n; ++t) {
			if (runtime_res.out_tokens[t] == baked_res.out_tokens[t]) ++agree;
			const int32_t* rt_row = runtime_res.out_logit_rows.data() + t * runtime_res.vocab_size;
			const int32_t* bk_row = baked_res.out_logit_rows.data() + t * baked_res.vocab_size;
			for (uint32_t v = 0; v < runtime_res.vocab_size; ++v) {
				const double d = std::fabs(static_cast<double>(rt_row[v]) - static_cast<double>(bk_row[v]));
				if (d > max_abs_logit_diff) max_abs_logit_diff = d;
			}
		}
		argmax_agree_counts.push_back(agree);
		argmax_total_counts.push_back(static_cast<int>(common_n));
		logit_max_abs_diffs.push_back(max_abs_logit_diff);
		std::printf("  runtime-vs-baked: token agreement %d/%zu, max|logit delta| over all positions/vocab = %.1f\n\n",
		            agree, common_n, max_abs_logit_diff);

		json << "  {\"prompt\": \"" << JsonEscape(prompt) << "\", "
		     << "\"base_text\": \"" << JsonEscape(base_text) << "\", "
		     << "\"runtime_text\": \"" << JsonEscape(runtime_text) << "\", "
		     << "\"baked_text\": \"" << JsonEscape(baked_text) << "\", "
		     << "\"base_tokens\": " << base_res.tokens_produced << ", "
		     << "\"runtime_tokens\": " << runtime_res.tokens_produced << ", "
		     << "\"baked_tokens\": " << baked_res.tokens_produced << ", "
		     << "\"base_decode_seconds\": " << base_res.decode_seconds << ", "
		     << "\"runtime_decode_seconds\": " << runtime_res.decode_seconds << ", "
		     << "\"baked_decode_seconds\": " << baked_res.decode_seconds << ", "
		     << "\"switch_changed_output\": " << (switch_did_real_work ? "true" : "false") << ", "
		     << "\"runtime_vs_baked_argmax_agree\": " << agree << ", "
		     << "\"runtime_vs_baked_common_tokens\": " << common_n << ", "
		     << "\"runtime_vs_baked_max_abs_logit_diff\": " << max_abs_logit_diff << "}"
		     << (pi + 1 < prompts.size() ? ",\n" : "\n");
	}
	json << " ],\n";

	auto avg = [](const std::vector<double>& v) {
		double s = 0;
		for (double x : v) s += x;
		return v.empty() ? 0.0 : s / v.size();
	};

	std::printf("================================================================================\n");
	std::printf("SUMMARY\n");
	std::printf("  mean tok/s   base=%.2f  runtime=%.2f  baked=%.2f\n", avg(base_tok_per_sec),
	            avg(runtime_tok_per_sec), avg(baked_tok_per_sec));
	if (avg(base_tok_per_sec) > 0.0) {
		std::printf("  runtime-adapter overhead vs base:  %.2f%%\n",
		            100.0 * (avg(base_tok_per_sec) - avg(runtime_tok_per_sec)) / avg(base_tok_per_sec));
	}
	if (avg(baked_tok_per_sec) > 0.0) {
		std::printf("  runtime vs baked throughput ratio: %.4f (1.0 = identical)\n",
		            avg(runtime_tok_per_sec) / avg(baked_tok_per_sec));
	}
	int total_agree = 0, total_common = 0;
	for (size_t i = 0; i < argmax_agree_counts.size(); ++i) {
		total_agree += argmax_agree_counts[i];
		total_common += argmax_total_counts[i];
	}
	std::printf("  runtime-vs-baked argmax agreement: %d/%d positions across all prompts\n", total_agree,
	            total_common);
	double overall_max_logit_diff = 0.0;
	for (double d : logit_max_abs_diffs) overall_max_logit_diff = std::max(overall_max_logit_diff, d);
	std::printf("  runtime-vs-baked max|logit delta| across all prompts/positions/vocab: %.1f\n",
	            overall_max_logit_diff);
	PrintTiming("base model cold load", base_load_stats);
	PrintTiming("baked model cold load", baked_load_stats);
	PrintTiming("adapter load+apply (switch cost)", adapter_load_stats);
	std::printf("================================================================================\n");

	json << " \"summary\": {\n"
	     << "  \"mean_tok_per_sec_base\": " << avg(base_tok_per_sec) << ",\n"
	     << "  \"mean_tok_per_sec_runtime\": " << avg(runtime_tok_per_sec) << ",\n"
	     << "  \"mean_tok_per_sec_baked\": " << avg(baked_tok_per_sec) << ",\n"
	     << "  \"argmax_agree\": " << total_agree << ",\n"
	     << "  \"argmax_total\": " << total_common << ",\n"
	     << "  \"max_abs_logit_diff\": " << overall_max_logit_diff << ",\n"
	     << "  \"base_load_seconds\": [";
	for (size_t i = 0; i < base_load_stats.samples.size(); ++i)
		json << base_load_stats.samples[i] << (i + 1 < base_load_stats.samples.size() ? "," : "");
	json << "],\n  \"baked_load_seconds\": [";
	for (size_t i = 0; i < baked_load_stats.samples.size(); ++i)
		json << baked_load_stats.samples[i] << (i + 1 < baked_load_stats.samples.size() ? "," : "");
	json << "],\n  \"adapter_load_seconds\": [";
	for (size_t i = 0; i < adapter_load_stats.samples.size(); ++i)
		json << adapter_load_stats.samples[i] << (i + 1 < adapter_load_stats.samples.size() ? "," : "");
	json << "]\n }\n}\n";

	if (!json_out_path.empty()) {
		std::ofstream jf(json_out_path, std::ios::trunc);
		jf << json.str();
		std::printf("wrote %s\n", json_out_path.c_str());
	}

	return 0;
}
