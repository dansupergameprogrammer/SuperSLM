// PROBE-ONLY: execute Qwen3 layers from supplied carried-scale hidden rows for T-2616.
// Usage: sslm_t2616_probe <local|sites|propagated> <model.sslm> <input.txt> <output.jsonl>

#include <cstdint>
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

struct InputRow {
	CarriedScale scale;
	std::vector<int8_t> codes;
};

struct LocalCase {
	std::string id;
	uint32_t layer = 0;
	std::vector<InputRow> rows;
};

struct PropagatedCase {
	std::string id;
	std::vector<int32_t> ids;
};

bool ReadLocalCases(const char* path, size_t hidden, std::vector<LocalCase>& cases) {
	std::ifstream in(path);
	std::string magic;
	uint32_t version = 0;
	size_t declared_hidden = 0, count = 0;
	if (!(in >> magic >> version >> declared_hidden >> count) || magic != "T2616_LOCAL" ||
	    version != 1 || declared_hidden != hidden) return false;
	cases.resize(count);
	for (LocalCase& c : cases) {
		size_t steps = 0;
		if (!(in >> c.id >> c.layer >> steps)) return false;
		c.rows.resize(steps);
		for (InputRow& row : c.rows) {
			if (!(in >> row.scale.m >> row.scale.e)) return false;
			row.codes.resize(hidden);
			for (size_t i = 0; i < hidden; ++i) {
				int value = 0;
				if (!(in >> value) || value < -127 || value > 127) return false;
				row.codes[i] = static_cast<int8_t>(value);
			}
		}
	}
	return true;
}

bool ReadPropagatedCases(const char* path, std::vector<PropagatedCase>& cases) {
	std::ifstream in(path);
	std::string magic;
	uint32_t version = 0;
	size_t count = 0;
	if (!(in >> magic >> version >> count) || magic != "T2616_PROP" || version != 1) return false;
	cases.resize(count);
	for (PropagatedCase& c : cases) {
		size_t steps = 0;
		if (!(in >> c.id >> steps)) return false;
		c.ids.resize(steps);
		for (int32_t& id : c.ids) {
			int64_t value = 0;
			if (!(in >> value) || value < INT32_MIN || value > INT32_MAX) return false;
			id = static_cast<int32_t>(value);
		}
	}
	return true;
}

struct TraceWriter {
	std::ofstream out;
	std::string case_id;
	bool all_sites = false;
	std::string selected_prefix;
	std::unordered_map<std::string, uint32_t> occurrence;

	explicit TraceWriter(const char* path) : out(path, std::ios::trunc) {}

	void StartCase(const std::string& id) {
		case_id = id;
		occurrence.clear();
	}

	uint32_t Head(std::string_view site, size_t token) {
		const std::string key = std::string(site) + '#' + std::to_string(token);
		return occurrence[key]++;
	}

	void Chain(const SslmChainTraceRecord& r) {
		if (!selected_prefix.empty() && !r.site.starts_with(selected_prefix)) return;
		if (!all_sites && !r.site.ends_with(".mlp_residual")) return;
		out << "{\"kind\":\"chain\",\"case\":\"" << case_id << "\",\"site\":\"" << r.site
		    << "\",\"token\":" << r.token_index << ",\"head\":" << Head(r.site, r.token_index)
		    << ",\"x_int\":";
		WriteArray(out, r.x_int);
		out << ",\"d_prime\":" << r.d_prime << ",\"dn\":" << r.dn << ",\"s\":" << r.s
		    << ",\"r\":" << r.r << ",\"codes\":";
		WriteArray(out, r.codes);
		out << ",\"m_out\":" << r.m_out << ",\"e_out\":" << r.e_out << "}\n";
	}

	void Site(const SslmSiteTraceRecord& r) {
		if (!all_sites) return;
		if (!selected_prefix.empty() && !r.site.starts_with(selected_prefix)) return;
		out << "{\"kind\":\"site\",\"case\":\"" << case_id << "\",\"site\":\"" << r.site
		    << "\",\"token\":" << r.token_index << ",\"head\":" << r.head << ",\"x_int\":";
		WriteArray(out, r.x_int);
		out << ",\"codes\":";
		WriteArray(out, r.codes);
		out << "}\n";
	}

	void Output(const LocalCase& c, size_t token, const SequenceLayerState& seq) {
		out << "{\"kind\":\"output\",\"case\":\"" << c.id << "\",\"layer\":" << c.layer
		    << ",\"token\":" << token << ",\"codes\":";
		WriteArray(out, std::span<const int8_t>(seq.hidden_codes, c.rows[token].codes.size()));
		out << ",\"m_out\":" << seq.hidden_scale.m << ",\"e_out\":" << seq.hidden_scale.e << "}\n";
	}
};

void TraceCallback(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord*, void* user) {
	if (chain != nullptr) static_cast<TraceWriter*>(user)->Chain(*chain);
}

void SiteCallback(const SslmSiteTraceRecord* site, void* user) {
	static_cast<TraceWriter*>(user)->Site(*site);
}

bool LoadModel(const char* path, std::vector<uint8_t>& bytes, SslmModelView& model,
	             std::vector<LayerBacking>& backings, std::vector<LayerWeights>& layers,
	             std::string& error) {
	if (!ReadFile(path, bytes)) return false;
	if (SslmModel::Load(bytes.data(), bytes.size(), model, &error) != SslmModelStatus::Ok) return false;
	PreflightScanWscFolds(model);
	backings.resize(model.config.num_hidden_layers);
	layers.resize(model.config.num_hidden_layers);
	for (uint32_t l = 0; l < model.config.num_hidden_layers; ++l) {
		if (!MarshalLayer(model, l, model.config.num_attention_heads,
		                  model.config.num_key_value_heads, backings[l], layers[l], &error)) return false;
	}
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5 || argc > 6) {
		std::fprintf(stderr, "usage: %s <local|sites|propagated|propagated-sites> <model.sslm> <input.txt> <output.jsonl> [layer]\n", argv[0]);
		return 2;
	}
	const std::string mode_name = argv[1];
	const bool sites = mode_name == "sites";
	const bool local = mode_name == "local" || sites;
	const bool propagated_sites = mode_name == "propagated-sites";
	const bool propagated = mode_name == "propagated" || propagated_sites;
	if (!local && !propagated) return 2;
	if (propagated_sites && argc != 6) return 2;

	std::vector<uint8_t> model_bytes;
	SslmModelView model;
	std::vector<LayerBacking> backings;
	std::vector<LayerWeights> layers;
	std::string error;
	if (!LoadModel(argv[2], model_bytes, model, backings, layers, error)) {
		std::fprintf(stderr, "model load/marshal failed: %s\n", error.c_str());
		return 1;
	}

	TraceWriter writer(argv[4]);
	if (!writer.out) return 1;
	writer.all_sites = sites || propagated_sites;
	if (propagated_sites) writer.selected_prefix = "layer" + std::string(argv[5]) + ".";
	SslmTraceHookState hook;
	SslmSetTraceHook(hook, TraceCallback, &writer);
	SslmSetSiteTraceHook(hook, SiteCallback, &writer);

	const uint32_t layers_n = model.config.num_hidden_layers;
	const size_t hidden = model.config.hidden_size;
	const int64_t cap = model.config.context_cap;
	const size_t kv_bytes = static_cast<size_t>(layers_n) * static_cast<size_t>(cap) *
	                        model.config.num_key_value_heads * model.config.head_dim * 2;
	const size_t q_width = static_cast<size_t>(model.config.num_attention_heads) * model.config.head_dim;
	const OptionGKLandingMode landing_mode = model.option_g_fused_k_landing
	                                                ? OptionGKLandingMode::kFused
	                                                : OptionGKLandingMode::kLegacy;

	if (local) {
		std::vector<LocalCase> cases;
		if (!ReadLocalCases(argv[3], hidden, cases)) {
			std::fprintf(stderr, "invalid local input\n");
			return 1;
		}
		for (const LocalCase& c : cases) {
			if (c.layer >= layers_n || c.rows.empty()) return 1;
			writer.StartCase(c.id);
			std::vector<uint8_t> workspace(kv_bytes);
			std::vector<int8_t> hidden_codes(hidden);
			SequenceLayerState seq;
			seq.hidden_codes = hidden_codes.data();
			const uint64_t kv0 = seq.kv_landing_saturation_count;
			const uint64_t kn0 = seq.k_normed_landing_saturation_count;
			const uint64_t rq0 = seq.rope_q_saturation_count;
			const uint64_t rk0 = seq.rope_k_saturation_count;
			for (size_t t = 0; t < c.rows.size(); ++t) {
				hidden_codes = c.rows[t].codes;
				seq.hidden_scale = c.rows[t].scale;
				seq.layer_index = c.layer;
				seq.context_length = static_cast<int64_t>(t);
				const SslmForwardStatus st = RunLayerLoop(
				    seq, layers.data(), layers_n, 1, hidden, model.config.head_dim,
				    model.config.num_key_value_heads, model.config.intermediate_size, cap,
				    model.rope_tables, workspace.data(), workspace.size(), landing_mode, {}, t,
				    sites ? &hook : nullptr, q_width);
				if (st != SslmForwardStatus::Ok) {
					std::fprintf(stderr, "case %s layer %u token %zu failed: %s\n", c.id.c_str(),
					             c.layer, t, SslmForwardStatusName(st));
					return 1;
				}
				writer.Output(c, t, seq);
			}
			writer.out << "{\"kind\":\"clamps\",\"case\":\"" << c.id << "\",\"layer\":" << c.layer
			           << ",\"kv_landing\":" << (seq.kv_landing_saturation_count - kv0)
			           << ",\"k_norm_landing\":" << (seq.k_normed_landing_saturation_count - kn0)
			           << ",\"rope_q\":" << (seq.rope_q_saturation_count - rq0)
			           << ",\"rope_k\":" << (seq.rope_k_saturation_count - rk0) << "}\n";
		}
	} else {
		std::vector<PropagatedCase> cases;
		if (!ReadPropagatedCases(argv[3], cases)) return 1;
		const SslmTensorView* embed = model.weights.Tensor("embed");
		bool ok = true;
		const CarriedScale embed_constant = ReadCarriedScale(model.composition_constants, "embed", &ok);
		if (embed == nullptr || !ok) return 1;
		for (const PropagatedCase& c : cases) {
			writer.StartCase(c.id);
			std::vector<uint8_t> workspace(kv_bytes);
			std::vector<int8_t> hidden_codes(hidden);
			SequenceLayerState seq;
			seq.hidden_codes = hidden_codes.data();
			for (size_t t = 0; t < c.ids.size(); ++t) {
				SslmForwardStatus st = EmbedEntry(
				    c.ids[t], static_cast<int32_t>(model.config.vocab_size),
				    reinterpret_cast<const int8_t*>(embed->data), hidden, embed_constant,
				    hidden_codes.data(), &seq.hidden_scale, "embed", t, nullptr);
				if (st != SslmForwardStatus::Ok) return 1;
				seq.layer_index = 0;
				st = RunLayerLoop(seq, layers.data(), layers_n, layers_n, hidden,
				                  model.config.head_dim, model.config.num_key_value_heads,
				                  model.config.intermediate_size, cap, model.rope_tables,
				                  workspace.data(), workspace.size(), landing_mode, {}, t, &hook, q_width);
				if (st != SslmForwardStatus::Ok) return 1;
			}
		}
	}
	return writer.out ? 0 : 1;
}
