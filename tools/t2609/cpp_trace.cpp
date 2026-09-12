// PROBE-ONLY: dump selected full-stack integer attention sites for T-2609.
// Usage: sslm_t2609_trace <model.sslm> <comma-separated-token-ids> <dump.jsonl>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;

namespace {

template <typename T>
void WriteArray(std::ostream& out, std::span<const T> values) {
	out << '[';
	for (size_t i = 0; i < values.size(); ++i) {
		if (i != 0) out << ',';
		out << static_cast<int64_t>(values[i]);
	}
	out << ']';
}

bool SelectedLayer(std::string_view site) {
	return site.starts_with("layer1.") || site.starts_with("layer14.") ||
	       site.starts_with("layer27.");
}

bool SelectedChain(std::string_view site) {
	return SelectedLayer(site) && site.ends_with(".q_norm");
}

bool SelectedSite(std::string_view site) {
	if (!SelectedLayer(site)) return false;
	return site.ends_with(".rope_q") || site.ends_with(".rope_k") ||
	       site.ends_with(".attention_scores") || site.ends_with(".softmax") ||
	       site.ends_with(".weighted_sum");
}

struct TraceWriter {
	std::ofstream out;
	std::unordered_map<std::string, uint32_t> occurrence;

	explicit TraceWriter(const char* path) : out(path, std::ios::trunc) {}

	uint32_t Head(std::string_view site, size_t token) {
		const std::string key = std::string(site) + '#' + std::to_string(token);
		return occurrence[key]++;
	}

	void Chain(const SslmChainTraceRecord& r) {
		if (!SelectedChain(r.site)) return;
		out << "{\"kind\":\"chain\",\"site\":\"" << r.site << "\",\"token\":"
		    << r.token_index << ",\"head\":" << Head(r.site, r.token_index) << ",\"codes\":";
		WriteArray(out, r.codes);
		out << ",\"m_out\":" << r.m_out << ",\"e_out\":" << r.e_out << "}\n";
	}

	void Site(const SslmSiteTraceRecord& r) {
		if (!SelectedSite(r.site)) return;
		out << "{\"kind\":\"site\",\"site\":\"" << r.site << "\",\"token\":"
		    << r.token_index << ",\"head\":" << r.head << ",\"x_int\":";
		WriteArray(out, r.x_int);
		out << ",\"codes\":";
		WriteArray(out, r.codes);
		out << "}\n";
	}
};

void TraceCallback(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord*, void* user) {
	if (chain != nullptr) static_cast<TraceWriter*>(user)->Chain(*chain);
}

void SiteCallback(const SslmSiteTraceRecord* site, void* user) {
	static_cast<TraceWriter*>(user)->Site(*site);
}

bool ParseIds(const char* text, std::vector<int32_t>& ids) {
	const char* p = text;
	while (*p != '\0') {
		char* end = nullptr;
		const long value = std::strtol(p, &end, 10);
		if (end == p || value < INT32_MIN || value > INT32_MAX) return false;
		ids.push_back(static_cast<int32_t>(value));
		if (*end == '\0') return true;
		if (*end != ',') return false;
		p = end + 1;
	}
	return !ids.empty();
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 4) {
		std::fprintf(stderr, "usage: %s <model.sslm> <comma-separated-token-ids> <dump.jsonl>\n", argv[0]);
		return 2;
	}
	std::vector<int32_t> ids;
	if (!ParseIds(argv[2], ids)) return 2;

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(argv[1], model_bytes)) return 1;
	SslmModelView model;
	std::string error;
	if (SslmModel::Load(model_bytes.data(), model_bytes.size(), model, &error) != SslmModelStatus::Ok) {
		std::fprintf(stderr, "model load failed: %s\n", error.c_str());
		return 1;
	}
	PreflightScanWscFolds(model);
	const uint32_t layers_n = model.config.num_hidden_layers;
	std::vector<LayerBacking> backings(layers_n);
	std::vector<LayerWeights> layers(layers_n);
	for (uint32_t l = 0; l < layers_n; ++l) {
		if (!MarshalLayer(model, l, model.config.num_attention_heads,
		                  model.config.num_key_value_heads, backings[l], layers[l], &error)) return 1;
	}
	const SslmTensorView* embed = model.weights.Tensor("embed");
	bool ok = true;
	const CarriedScale embed_constant = ReadCarriedScale(model.composition_constants, "embed", &ok);
	if (embed == nullptr || !ok) return 1;

	TraceWriter writer(argv[3]);
	if (!writer.out) return 1;
	writer.out << "{\"kind\":\"header\",\"token_ids\":";
	WriteArray(writer.out, std::span<const int32_t>(ids));
	writer.out << "}\n";
	SslmTraceHookState hook;
	SslmSetTraceHook(hook, TraceCallback, &writer);
	SslmSetSiteTraceHook(hook, SiteCallback, &writer);

	const size_t hidden = model.config.hidden_size;
	const int64_t cap = model.config.context_cap;
	const size_t kv_bytes = static_cast<size_t>(layers_n) * static_cast<size_t>(cap) *
	                        model.config.num_key_value_heads * model.config.head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();
	const OptionGKLandingMode mode = model.option_g_fused_k_landing
	                                    ? OptionGKLandingMode::kFused
	                                    : OptionGKLandingMode::kLegacy;
	const size_t q_width = static_cast<size_t>(model.config.num_attention_heads) *
	                       model.config.head_dim;
	for (size_t t = 0; t < ids.size(); ++t) {
		SslmForwardStatus st = EmbedEntry(
		    ids[t], static_cast<int32_t>(model.config.vocab_size),
		    reinterpret_cast<const int8_t*>(embed->data), hidden, embed_constant,
		    hidden_codes.data(), &seq.hidden_scale, "embed", t, &hook);
		if (st != SslmForwardStatus::Ok) return 1;
		seq.layer_index = 0;
		st = RunLayerLoop(seq, layers.data(), layers_n, layers_n, hidden, model.config.head_dim,
		                  model.config.num_key_value_heads, model.config.intermediate_size, cap,
		                  model.rope_tables, workspace.data(), workspace.size(), mode, {}, t, &hook,
		                  q_width);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "forward failed token %zu: %s\n", t, SslmForwardStatusName(st));
			return 1;
		}
	}
	return writer.out ? 0 : 1;
}
