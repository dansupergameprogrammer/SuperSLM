// T-2801 / T-2814 (Curie) -- O-CPU, the CPU forward oracle the guard-status cells label tokens with
// and check their per-cell preconditions against: EmbedEntry plus RunLayerLoop per token, over layers
// this header marshals from the cell's own SslmModel::Load parse (GpuModelFixture::view). It takes no
// input from the GPU path. Moved here unchanged from cell_prompt_guard_status.cpp (T-2801) so that
// cell_schema_guard_status.cpp (T-2814, Q5-1) uses the identical construction.
//
// `seq_cap` sizes the K/V workspace and is passed to RunLayerLoop as the context cap. It defaults to the
// artifact's own context_cap, which is what T-2801 used (64 on the U1 pair model and G-an, so those
// cells are unchanged). Q5-1's 1.5B artifact has context_cap 32,768, and a workspace of that size is
// ~470 MB per call; the cell passes 64, the cap of every sequence it creates, since no call it checks
// runs past a handful of positions.
#ifndef SSLM_T2791_CPU_ORACLE_H
#define SSLM_T2791_CPU_ORACLE_H

#include "fixture_common.h"

struct CpuResult {
	superslm::SslmForwardStatus st = superslm::SslmForwardStatus::Ok;
	int refused_index = -1;
	uint32_t refused_layer = 0;
	Frame frame;  // status SSLM_OK with the final_norm frame when every token passed
};

class CpuOracle {
public:
	explicit CpuOracle(GpuModelFixture& fx, int64_t seq_cap = 0) : fx_(fx) {
		superslm::SslmModelView& v = fx.view;
		cap_ = seq_cap > 0 ? seq_cap : static_cast<int64_t>(v.config.context_cap);
		superslm_marshal::PreflightScanWscFolds(v);
		backing_.resize(fx.layers);
		layers_.resize(fx.layers);
		std::string err;
		ok_ = true;
		for (uint32_t l = 0; l < fx.layers; ++l) {
			if (!superslm_marshal::MarshalLayer(v, l, v.config.num_attention_heads, v.config.num_key_value_heads,
			                                    backing_[l], layers_[l], &err)) {
				ok_ = false;
			}
		}
		embed_scale_ = superslm_marshal::ReadCarriedScale(v.composition_constants, "embed", &ok_);
		const superslm::SslmTensorView* e = v.weights.Tensor("embed");
		if (!e) ok_ = false;
		embed_ = e ? reinterpret_cast<const int8_t*>(e->data) : nullptr;
	}
	bool ok() const { return ok_; }

	// `layer_budget` layers per token; 0 is full depth. A label taken at layer 0 only passes 1.
	CpuResult Run(const std::vector<int32_t>& toks, uint32_t layer_budget = 0) {
		superslm::SslmModelView& v = fx_.view;
		const size_t need = static_cast<size_t>(v.config.num_hidden_layers) * static_cast<size_t>(cap_) *
		                    v.config.num_key_value_heads * v.config.head_dim * 2;
		if (ws_.size() != need) ws_.assign(need, 0);
		std::fill(ws_.begin(), ws_.end(), static_cast<uint8_t>(0));
		std::vector<int8_t> codes(fx_.hidden, 0);
		superslm::SequenceLayerState seq{};
		seq.hidden_codes = codes.data();
		const auto k_mode = v.option_g_fused_k_landing ? superslm::OptionGKLandingMode::kFused
		                                               : superslm::OptionGKLandingMode::kLegacy;
		CpuResult r;
		r.frame.status = SSLM_SEQUENCE_REJECTED;
		for (size_t i = 0; i < toks.size(); ++i) {
			superslm::CarriedScale s{};
			superslm::EmbedEntry(toks[i], fx_.vocab, embed_, fx_.hidden, embed_scale_, codes.data(), &s);
			seq.hidden_scale = s;
			seq.layer_index = 0;
			const auto st = superslm::RunLayerLoop(
			    seq, layers_.data(), fx_.layers, layer_budget ? layer_budget : fx_.layers, fx_.hidden,
			    v.config.head_dim, v.config.num_key_value_heads, v.config.intermediate_size, cap_, v.rope_tables,
			    ws_.data(), ws_.size(), k_mode, {}, 0, &v.trace_hook,
			    static_cast<size_t>(v.config.num_attention_heads) * v.config.head_dim);
			if (st != superslm::SslmForwardStatus::Ok) {
				r.st = st;
				r.refused_index = static_cast<int>(i);
				r.refused_layer = seq.layer_index;
				return r;
			}
		}
		r.frame.codes.assign(fx_.hidden, 0);
		superslm::CarriedScale out{};
		const auto fst = superslm::RmsNormSite(codes.data(), fx_.final_gain.data(), fx_.hidden, superslm::CarriedScale{},
		                                       fx_.final_const, r.frame.codes.data(), &out, "final_norm");
		r.frame.status = fst == superslm::SslmForwardStatus::Ok ? SSLM_OK : SSLM_SEQUENCE_REJECTED;
		r.frame.m = out.m;
		r.frame.e = out.e;
		r.frame.required = fx_.hidden;
		return r;
	}

private:
	GpuModelFixture& fx_;
	int64_t cap_ = 0;
	std::vector<uint8_t> ws_;
	std::vector<superslm_marshal::LayerBacking> backing_;
	std::vector<superslm::LayerWeights> layers_;
	superslm::CarriedScale embed_scale_{};
	const int8_t* embed_ = nullptr;
	bool ok_ = false;
};

inline std::string Ids(const std::vector<int32_t>& t) {
	std::string s = "[";
	for (size_t i = 0; i < t.size(); ++i) s += (i ? "," : "") + std::to_string(t[i]);
	return s + "]";
}

inline bool FileSha256(const std::string& path, std::string* out) {
	std::vector<uint8_t> b;
	if (!ReadFileBytes(path, &b)) return false;
	*out = Sha256Hex(b.data(), b.size());
	return true;
}

#endif  // SSLM_T2791_CPU_ORACLE_H
