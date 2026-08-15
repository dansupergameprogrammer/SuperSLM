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
// RUNTIME's own logits are compared against BAKED's TWO ways (T-2104, Poirot 8e07d0c review,
// Significant 3 -- corrected in-tool rather than left to an ad hoc offline script):
//   (a) SAME-CONTEXT, position 0 of the shared prompt -- the one position both sides are provably
//       the same quantity (same input, same context, two independently-computed compositions:
//       runtime's own composed int8 accumulator path vs baked's own merged-then-quantized path,
//       neither derived from the other). Argmax match, top-5 candidate-set match, and max|delta|
//       over the union of the two top-5s. THIS is what "agreement, and whether any divergence is
//       within int8 rounding expectations" (T-2102's own brief) actually asks, and answers.
//   (b) TRAJECTORY divergence, informational only -- printed and serialized under its own clearly
//       labeled key, never as "agreement." Past the first position where RUNTIME's and BAKED's own
//       argmax choices differ, greedy decode means each side is conditioned on a DIFFERENT token
//       history, so comparing later positions compares two different quantities, not composition
//       error (StandardsDocument.md §5.4).
//
// Timing: model load (BASE, BAKED), adapter load+apply (RUNTIME's own extra step), AND the
// generation cells themselves (three configurations x three prompts) are ALL repeated >=3 times
// (--timing-reps, default 3) and reported as min/median/max -- this machine may be shared with
// unrelated CPU/GPU load tonight (RESUME-2026-08-14 §6), so the spread is the honest unit, never a
// single point figure at more precision than the sample supports.
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

// T-2104 (Poirot 8e07d0c review, Significant 3): the CORRECTED runtime-vs-baked comparison, built
// into the tool itself rather than left to an ad hoc offline script. Position 0 of `out_logit_rows`
// is the logit row produced from the SHARED prompt context, before either side has emitted a token
// of its own -- the one position where RUNTIME and BAKED are provably the same quantity (same
// input, same context, two independently-computed compositions). Any position past 0 in a greedy
// decode is conditioned on whichever token THAT side already chose, which can differ between the
// two sides the instant they first disagree -- comparing those positions compares two different
// conditioning contexts, not the same quantity (StandardsDocument.md Sec5.4).
struct SingleTokenComparison {
	int32_t argmax_runtime = -1;
	int32_t argmax_baked = -1;
	bool argmax_match = false;
	bool top5_set_match = false;
	double max_abs_delta_over_top5_union = 0.0;
	int32_t runtime_top_logit = 0;
	int32_t baked_top_logit = 0;
};

// Indices of the `k` largest elements of `row[0..n)`, descending. `n`/`k` are small enough (k=5)
// that a linear partial-selection is simpler and plenty fast against a single row.
std::vector<uint32_t> TopKIndices(const int32_t* row, uint32_t n, uint32_t k) {
	std::vector<uint32_t> idx(n);
	for (uint32_t i = 0; i < n; ++i) idx[i] = i;
	std::partial_sort(idx.begin(), idx.begin() + std::min<uint32_t>(k, n), idx.end(),
	                   [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
	idx.resize(std::min<uint32_t>(k, n));
	return idx;
}

SingleTokenComparison CompareSingleToken(const int32_t* runtime_row0, const int32_t* baked_row0,
                                          uint32_t vocab_size) {
	SingleTokenComparison c;
	const auto rt_top5 = TopKIndices(runtime_row0, vocab_size, 5);
	const auto bk_top5 = TopKIndices(baked_row0, vocab_size, 5);
	c.argmax_runtime = static_cast<int32_t>(rt_top5[0]);
	c.argmax_baked = static_cast<int32_t>(bk_top5[0]);
	c.argmax_match = (c.argmax_runtime == c.argmax_baked);
	c.runtime_top_logit = runtime_row0[rt_top5[0]];
	c.baked_top_logit = baked_row0[bk_top5[0]];
	std::vector<uint32_t> rt_sorted(rt_top5), bk_sorted(bk_top5);
	std::sort(rt_sorted.begin(), rt_sorted.end());
	std::sort(bk_sorted.begin(), bk_sorted.end());
	c.top5_set_match = (rt_sorted == bk_sorted);
	std::vector<uint32_t> u = rt_top5;
	u.insert(u.end(), bk_top5.begin(), bk_top5.end());
	std::sort(u.begin(), u.end());
	u.erase(std::unique(u.begin(), u.end()), u.end());
	for (uint32_t i : u) {
		const double d = std::fabs(static_cast<double>(runtime_row0[i]) - static_cast<double>(baked_row0[i]));
		if (d > c.max_abs_delta_over_top5_union) c.max_abs_delta_over_top5_union = d;
	}
	return c;
}

// T-2104 (Poirot 8e07d0c review, Minor 4): every control byte is escaped, not just `"`/`\`/`\n` --
// prompts are user-supplied (`--prompts`), and an unescaped `\r`/`\t`/other control byte produces a
// `--json` file that is not valid JSON.
std::string JsonEscape(const std::string& s) {
	std::string out;
	for (unsigned char c : s) {
		switch (c) {
			case '"': out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n"; break;
			case '\r': out += "\\r"; break;
			case '\t': out += "\\t"; break;
			case '\b': out += "\\b"; break;
			case '\f': out += "\\f"; break;
			default:
				if (c < 0x20) {
					char buf[8];
					std::snprintf(buf, sizeof(buf), "\\u%04x", c);
					out += buf;
				} else {
					out += static_cast<char>(c);
				}
		}
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

	// T-2104 (Poirot 8e07d0c review, Minor 6): "cold load" is wrong for reps 2+, which run against
	// the OS's own warm file cache once rep 1 has read the file -- labeled plainly below rather than
	// claimed cold.
	std::printf("--- timing: model load, %d repetitions each (rep 1 may be cold; reps 2+ are "
	            "warm-cache) ---------\n",
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

	// T-2104 (Poirot 8e07d0c review, Minor 3): the runtime-vs-baked logit comparison below indexes
	// BAKED's row by `baked_res.vocab_size` and iterates `v < runtime_res.vocab_size` -- sound only
	// if the two agree. Checked once, here, rather than trusted per-prompt.
	if (base_model.view.config.vocab_size != baked_model.view.config.vocab_size) {
		std::fprintf(stderr,
		             "FAILED at stage=geometry_check: base vocab_size=%u != baked vocab_size=%u -- "
		             "the runtime-vs-baked logit comparison assumes a shared vocabulary and geometry\n",
		             base_model.view.config.vocab_size, baked_model.view.config.vocab_size);
		return 1;
	}

	for (int rep = 0; rep < timing_reps; ++rep) {
		// T-2104 (Poirot 8e07d0c review, Significant 2): the base's own hash, read directly off
		// `base_model.view` -- no second `SslmArtifact::OpenFromMemory`. This was already outside
		// the timed t0/t1 region below even before this fix; corrected here for the same reason
		// sslm_generate.cpp's own Stage 3b was (SslmModelView::RawIntegrityHash(), added this round).
		BaseModelGeometry geo;
		geo.num_hidden_layers = base_model.num_hidden_layers;
		geo.hidden_size = base_model.hidden_size;
		geo.intermediate_size = base_model.view.config.intermediate_size;
		geo.kv_hidden_size = static_cast<uint64_t>(base_model.num_kv_heads) * base_model.view.config.head_dim;
		geo.base_artifact_hash = base_model.view.RawIntegrityHash();

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
	PrintTiming("base model load (see banner)", base_load_stats);
	PrintTiming("baked model load (see banner)", baked_load_stats);
	PrintTiming("adapter load+apply (switch cost)", adapter_load_stats);
	std::printf("\n");

	// --- Generation: base-only, runtime(base+adapter), baked-only, per prompt, REPEATED. -------
	// T-2104 (Poirot 8e07d0c review, Significant 4): the generation cell now repeats under the
	// SAME `--timing-reps` protocol the loads already used, rather than being measured once and
	// reported to two decimals. Runtime generation reuses `base_model`'s own layers[] directly
	// (bind the adapter immediately before the runtime call, unbind immediately after) rather than
	// a second, separately-loaded model object -- this IS the switch: the same in-memory base, the
	// adapter pointer swapped in and out, never a second model load.
	std::vector<double> all_base_tps, all_runtime_tps, all_baked_tps;  // every (prompt, rep) sample
	int runtime_slower_than_base_prompts = 0;  // paired-direction count (median vs median, per prompt)

	std::ostringstream json;
	json << "{\n \"prompts\": [\n";

	for (size_t pi = 0; pi < prompts.size(); ++pi) {
		const std::string& prompt = prompts[pi];
		std::printf("--- prompt %zu: %s\n", pi, prompt.c_str());

		std::vector<double> base_tps, runtime_tps, baked_tps;
		GenResult base_res, runtime_res, baked_res;  // kept: the LAST rep (text/logits reported from it)
		std::string err;
		for (int rep = 0; rep < timing_reps; ++rep) {
			if (!GenerateOne(base_model, tokenizer, prompt, max_new_tokens, base_res, &err)) {
				std::fprintf(stderr, "FAILED base generation (rep %d): %s\n", rep, err.c_str());
				return 1;
			}
			// base_res above is genuinely base-only: the adapter is bound immediately before the
			// runtime call and cleared immediately after, every rep.
			ApplyAdapterToLayers(&adapter_handle, base_model.layers.data(), base_model.num_hidden_layers);
			if (!GenerateOne(base_model, tokenizer, prompt, max_new_tokens, runtime_res, &err)) {
				std::fprintf(stderr, "FAILED runtime generation (rep %d): %s\n", rep, err.c_str());
				return 1;
			}
			ApplyAdapterToLayers(nullptr, base_model.layers.data(), base_model.num_hidden_layers);
			if (!GenerateOne(baked_model, tokenizer, prompt, max_new_tokens, baked_res, &err)) {
				std::fprintf(stderr, "FAILED baked generation (rep %d): %s\n", rep, err.c_str());
				return 1;
			}
			base_tps.push_back(base_res.tokens_produced / std::max(1e-9, base_res.decode_seconds));
			runtime_tps.push_back(runtime_res.tokens_produced / std::max(1e-9, runtime_res.decode_seconds));
			baked_tps.push_back(baked_res.tokens_produced / std::max(1e-9, baked_res.decode_seconds));
		}
		const TimingStats base_tps_stats = Summarize(base_tps);
		const TimingStats runtime_tps_stats = Summarize(runtime_tps);
		const TimingStats baked_tps_stats = Summarize(baked_tps);
		all_base_tps.insert(all_base_tps.end(), base_tps.begin(), base_tps.end());
		all_runtime_tps.insert(all_runtime_tps.end(), runtime_tps.begin(), runtime_tps.end());
		all_baked_tps.insert(all_baked_tps.end(), baked_tps.begin(), baked_tps.end());
		if (runtime_tps_stats.median_s < base_tps_stats.median_s) ++runtime_slower_than_base_prompts;

		const std::string base_text = tokenizer.Decode(
		    std::vector<int32_t>(base_res.out_tokens.begin(), base_res.out_tokens.begin() +
		                                                          static_cast<long>(base_res.tokens_produced)));
		const std::string runtime_text = tokenizer.Decode(std::vector<int32_t>(
		    runtime_res.out_tokens.begin(), runtime_res.out_tokens.begin() + static_cast<long>(runtime_res.tokens_produced)));
		const std::string baked_text = tokenizer.Decode(std::vector<int32_t>(
		    baked_res.out_tokens.begin(), baked_res.out_tokens.begin() + static_cast<long>(baked_res.tokens_produced)));

		std::printf("  BASE    (last rep, %zu tok): %s\n", base_res.tokens_produced, base_text.c_str());
		std::printf("  RUNTIME (last rep, %zu tok): %s\n", runtime_res.tokens_produced, runtime_text.c_str());
		std::printf("  BAKED   (last rep, %zu tok): %s\n", baked_res.tokens_produced, baked_text.c_str());
		PrintTiming("  tok/s BASE", base_tps_stats);
		PrintTiming("  tok/s RUNTIME", runtime_tps_stats);
		PrintTiming("  tok/s BAKED", baked_tps_stats);

		const bool switch_did_real_work = (runtime_res.tokens_produced != base_res.tokens_produced) ||
		                                   (runtime_res.out_tokens != base_res.out_tokens);
		std::printf("  switch changed output vs base (last rep): %s\n", switch_did_real_work ? "YES" : "NO");

		// --- CORRECTED (T-2104 S3): position-0 same-context comparison -- the valid apples-to-apples
		// test, computed by the tool itself so it is a committed, reproducible artifact rather than
		// an ad hoc offline script's untracked output. -------------------------------------------
		const SingleTokenComparison stc = CompareSingleToken(
		    runtime_res.out_logit_rows.data(), baked_res.out_logit_rows.data(), runtime_res.vocab_size);
		std::printf("  runtime-vs-baked SAME-CONTEXT position 0 (the valid comparison): argmax rt=%d "
		            "bk=%d %s, top-5 candidate set %s, max|delta| over top-5 union=%.1f "
		            "(top logits rt=%d bk=%d)\n",
		            stc.argmax_runtime, stc.argmax_baked, stc.argmax_match ? "MATCH" : "DIFFER",
		            stc.top5_set_match ? "MATCH" : "differ", stc.max_abs_delta_over_top5_union,
		            stc.runtime_top_logit, stc.baked_top_logit);

		// --- Trajectory divergence (last rep) -- NOT a composition-error measurement past the
		// first argmax fork (StandardsDocument.md Sec5.4: RUNTIME position t and BAKED position t
		// stop sharing a context the instant either side's own argmax has diverged from the
		// other's, so comparing later positions compares two different quantities). Reported for
		// context only, explicitly labeled, never folded into an "agreement" headline. ----------
		const size_t common_n = std::min(runtime_res.tokens_produced, baked_res.tokens_produced);
		size_t first_fork = common_n;
		int trajectory_agree = 0;
		for (size_t t = 0; t < common_n; ++t) {
			if (runtime_res.out_tokens[t] == baked_res.out_tokens[t]) {
				++trajectory_agree;
			} else if (first_fork == common_n) {
				first_fork = t;
			}
		}
		std::printf("  runtime-vs-baked TRAJECTORY divergence (last rep, informational -- NOT composition "
		            "error past the first fork): token-id match %d/%zu, first fork at position %zu\n\n",
		            trajectory_agree, common_n, first_fork);

		json << "  {\"prompt\": \"" << JsonEscape(prompt) << "\", "
		     << "\"base_text\": \"" << JsonEscape(base_text) << "\", "
		     << "\"runtime_text\": \"" << JsonEscape(runtime_text) << "\", "
		     << "\"baked_text\": \"" << JsonEscape(baked_text) << "\", "
		     << "\"base_tok_per_sec_median\": " << base_tps_stats.median_s << ", "
		     << "\"runtime_tok_per_sec_median\": " << runtime_tps_stats.median_s << ", "
		     << "\"baked_tok_per_sec_median\": " << baked_tps_stats.median_s << ", "
		     << "\"switch_changed_output\": " << (switch_did_real_work ? "true" : "false") << ", "
		     << "\"single_token_same_context\": {"
		     << "\"argmax_runtime\": " << stc.argmax_runtime << ", "
		     << "\"argmax_baked\": " << stc.argmax_baked << ", "
		     << "\"argmax_match\": " << (stc.argmax_match ? "true" : "false") << ", "
		     << "\"top5_set_match\": " << (stc.top5_set_match ? "true" : "false") << ", "
		     << "\"max_abs_delta_over_top5_union\": " << stc.max_abs_delta_over_top5_union << ", "
		     << "\"runtime_top_logit\": " << stc.runtime_top_logit << ", "
		     << "\"baked_top_logit\": " << stc.baked_top_logit << "}, "
		     << "\"trajectory_divergence_informational_only\": {"
		     << "\"token_id_match\": " << trajectory_agree << ", "
		     << "\"common_tokens\": " << common_n << ", "
		     << "\"first_fork_position\": " << first_fork << "}}"
		     << (pi + 1 < prompts.size() ? ",\n" : "\n");
	}
	json << " ],\n";

	const TimingStats base_gen_stats = Summarize(all_base_tps);
	const TimingStats runtime_gen_stats = Summarize(all_runtime_tps);
	const TimingStats baked_gen_stats = Summarize(all_baked_tps);

	std::printf("================================================================================\n");
	std::printf("SUMMARY\n");
	PrintTiming("tok/s BASE (all prompts/reps)", base_gen_stats);
	PrintTiming("tok/s RUNTIME (all prompts/reps)", runtime_gen_stats);
	PrintTiming("tok/s BAKED (all prompts/reps)", baked_gen_stats);
	// T-2104 (Poirot 8e07d0c review, Significant 4): the direction (paired per prompt, median vs
	// median) is what n=`prompts.size()` paired samples on a shared machine actually supports --
	// stated as a count, not smoothed into a percentage with no resolving power beside it.
	std::printf("  RUNTIME slower than BASE (median tok/s) on %d/%zu prompts\n",
	            runtime_slower_than_base_prompts, prompts.size());
	if (base_gen_stats.median_s > 0.0) {
		std::printf("  runtime-adapter overhead vs base, MEDIANS: %.1f%% (spread: base %.2f-%.2f "
		            "tok/s, runtime %.2f-%.2f tok/s -- treat the percentage as approximate at this "
		            "resolution, not exact to the decimal)\n",
		            100.0 * (base_gen_stats.median_s - runtime_gen_stats.median_s) / base_gen_stats.median_s,
		            base_gen_stats.min_s, base_gen_stats.max_s, runtime_gen_stats.min_s, runtime_gen_stats.max_s);
	}
	if (baked_gen_stats.median_s > 0.0) {
		std::printf("  runtime vs baked throughput ratio, MEDIANS: %.3f (1.0 = identical; spread: "
		            "baked %.2f-%.2f tok/s)\n",
		            runtime_gen_stats.median_s / baked_gen_stats.median_s, baked_gen_stats.min_s,
		            baked_gen_stats.max_s);
	}
	PrintTiming("base model load (see banner)", base_load_stats);
	PrintTiming("baked model load (see banner)", baked_load_stats);
	PrintTiming("adapter load+apply (switch cost)", adapter_load_stats);
	std::printf("================================================================================\n");

	auto write_samples = [&](const char* key, const TimingStats& t) {
		json << "  \"" << key << "\": [";
		for (size_t i = 0; i < t.samples.size(); ++i)
			json << t.samples[i] << (i + 1 < t.samples.size() ? "," : "");
		json << "]";
	};
	json << " \"summary\": {\n"
	     << "  \"runtime_slower_than_base_prompts\": " << runtime_slower_than_base_prompts << ",\n"
	     << "  \"prompts_total\": " << prompts.size() << ",\n  ";
	write_samples("base_tok_per_sec_all_prompts_reps", base_gen_stats);
	json << ",\n  ";
	write_samples("runtime_tok_per_sec_all_prompts_reps", runtime_gen_stats);
	json << ",\n  ";
	write_samples("baked_tok_per_sec_all_prompts_reps", baked_gen_stats);
	json << ",\n  ";
	write_samples("base_load_seconds", base_load_stats);
	json << ",\n  ";
	write_samples("baked_load_seconds", baked_load_stats);
	json << ",\n  ";
	write_samples("adapter_load_seconds", adapter_load_stats);
	json << "\n }\n}\n";

	if (!json_out_path.empty()) {
		std::ofstream jf(json_out_path, std::ios::trunc);
		jf << json.str();
		std::printf("wrote %s\n", json_out_path.c_str());
	}

	return 0;
}
