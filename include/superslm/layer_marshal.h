// layer_marshal.h -- the shared "how an artifact becomes a LayerWeights[]" code: parses a
// loaded model artifact's tensor sections into the per-layer weight views the forward pass
// consumes.
//
// This is production code: `sslm_gpu_model_map` (src/gpu/gpu_1p0.cpp) and the CPU-side
// generation tools both marshal a `SslmModelView` into a `LayerWeights[]` through this one
// shared implementation, so there is exactly one copy of the marshaling logic rather than two
// that can silently drift apart.
//
// `tools/sslm_marshal.h` is a compatibility shim that includes this header, so existing
// `#include "sslm_marshal.h"` call sites keep working unchanged.
#ifndef SUPERSLM_INCLUDE_LAYER_MARSHAL_H
#define SUPERSLM_INCLUDE_LAYER_MARSHAL_H

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"

namespace superslm_marshal {

inline bool ReadFile(const char* path, std::vector<uint8_t>& out) {
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

// Little-endian byte-assembly read of one int32 element -- WGT1/WSC1/BIA1
// tensor data IS alignment-guaranteed by the loader (TensorMisaligned,
// model.h), so a reinterpret_cast would also be sound here, but this driver
// reads every artifact-carried array through explicit byte assembly to match
// this codebase's own established convention for untrusted-provenance data
// (forward_sites.cpp's own RdI64 comment: "the same discipline... for every
// other untrusted-alignment array in this tree").
inline int32_t RdI32(const uint8_t* p) noexcept {
	return static_cast<int32_t>(static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	                             (static_cast<uint32_t>(p[2]) << 16) |
	                             (static_cast<uint32_t>(p[3]) << 24));
}

inline int64_t RdI64(const uint8_t* p) noexcept {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

// Sign-extends an int8-coded WGT1 gain tensor into an owned int32_t array --
// RmsNormSite's own contract (`const int32_t* g`, forward_sites.h) needs a
// widened container; the wire format stores gain codes as int8
// (tools/sslm_model_writer.py / convert_model.py's "Weights (WGT1, int8)").
inline std::vector<int32_t> WidenGainToInt32(const superslm::SslmTensorView& t) {
	std::vector<int32_t> out(t.elem_count);
	for (uint64_t i = 0; i < t.elem_count; ++i) {
		out[i] = static_cast<int32_t>(static_cast<int8_t>(t.data[i]));
	}
	return out;
}

inline superslm::CarriedScale ReadCarriedScale(const superslm::SslmKeyedConstants& kc,
                                                const std::string& name, bool* ok) {
	const superslm::SslmConstantEntry* e = kc.Entry(name);
	if (e == nullptr || e->value_words < 2) {
		*ok = false;
		return superslm::CarriedScale{};
	}
	return superslm::CarriedScale{superslm::SslmKeyedConstants::Value(*e, 0),
	                               superslm::SslmKeyedConstants::Value(*e, 1)};
}

// Backing storage for one layer's marshaled arrays -- LayerWeights itself
// holds only pointers, so every owned array a layer needs lives here, one
// instance per layer, kept alive for the whole decode call.
struct LayerBacking {
	std::vector<int32_t> attn_norm_gain, mlp_norm_gain;
	std::vector<int64_t> kv_r_t_k, kv_e_t_k, kv_r_t_v, kv_e_t_v;
	std::vector<int64_t> iexp_m, iexp_e;
	std::vector<int32_t> ctx_identity, ctx_mult, ctx_shift;
	// T-1666: one (identity, mult, shift) array per output channel, per
	// projection -- the per-channel WSC1 fold backing this driver marshals
	// from the artifact's own WSC1 rows (see this file's header comment).
	std::vector<int32_t> q_fold_identity, q_fold_mult, q_fold_shift;
	std::vector<int32_t> k_fold_identity, k_fold_mult, k_fold_shift;
	std::vector<int32_t> v_fold_identity, v_fold_mult, v_fold_shift;
	std::vector<int32_t> o_fold_identity, o_fold_mult, o_fold_shift;
	std::vector<int32_t> gate_fold_identity, gate_fold_mult, gate_fold_shift;
	std::vector<int32_t> up_fold_identity, up_fold_mult, up_fold_shift;
	std::vector<int32_t> down_fold_identity, down_fold_mult, down_fold_shift;
};

// Marshals one projection's per-output-channel WSC1 fold tensor into three
// owned arrays. `t` is the artifact's WSC1 tensor for this projection
// (row-major [channels, 3] int32, per docs/sslm_format.md "Weight-scale fold
// blob"); `channels` is the exact output-channel count ProjectAndFunnel/the
// k-v-landing loop will index these arrays at (forward_sites.cpp). Returns
// false and a diagnostic in `*err` if the tensor is missing or its row count
// does not match `channels` -- this is the structural check T-1664's driver
// used to fail loudly on; now it is the input-shape check the per-channel
// marshal itself performs before trusting the artifact's row count.
inline bool MarshalProjectionFold(const superslm::SslmTensorView* t, uint64_t channels,
                                   const std::string& label, std::vector<int32_t>& id,
                                   std::vector<int32_t>& mult, std::vector<int32_t>& shift,
                                   std::string* err) {
	if (!t) {
		*err = label + ": missing WSC1 weight-scale tensor";
		return false;
	}
	if (t->elem_count != channels * 3) {
		*err = label + ": WSC1 carries " + std::to_string(t->elem_count / 3) +
		       " per-output-channel (identity,mult,shift) rows, expected " + std::to_string(channels) +
		       " (one per output channel this projection's ProjectAndFunnel/k-v-landing call uses)";
		return false;
	}
	id.resize(channels);
	mult.resize(channels);
	shift.resize(channels);
	for (uint64_t i = 0; i < channels; ++i) {
		id[i] = RdI32(t->data + (i * 3 + 0) * 4);
		mult[i] = RdI32(t->data + (i * 3 + 1) * 4);
		shift[i] = RdI32(t->data + (i * 3 + 2) * 4);
	}
	return true;
}

// Attempts to marshal layer `l` into `out`/`backing`. Returns true and a
// populated LayerWeights on success. Returns false and a diagnostic in `err`
// the FIRST time a field cannot be represented -- specifically, the WSC1
// per-channel fold gap this file's header documents. Every field marshaled
// before that point in this function is real, artifact-sourced data, not a
// placeholder -- the function does the whole job up to the point the
// production struct's own shape stops it.
inline bool MarshalLayer(const superslm::SslmModelView& view, uint32_t l, uint32_t num_heads,
                          uint32_t num_key_value_heads, LayerBacking& backing,
                          superslm::LayerWeights& out, std::string* err) {
	const std::string prefix = "layer" + std::to_string(l);
	auto Wgt = [&](const char* suffix) -> const superslm::SslmTensorView* {
		return view.weights.Tensor(prefix + "." + suffix);
	};
	auto Wsc = [&](const char* suffix) -> const superslm::SslmTensorView* {
		return view.weight_scales.Tensor(prefix + "." + suffix);
	};
	auto Bia = [&](const char* suffix) -> const superslm::SslmTensorView* {
		return view.biases.Tensor(prefix + "." + suffix);
	};

	// --- weights (WGT1, int8) --------------------------------------------
	const superslm::SslmTensorView *q_w = Wgt("q_proj"), *k_w = Wgt("k_proj"), *v_w = Wgt("v_proj"),
	                                *o_w = Wgt("o_proj"), *gate_w = Wgt("gate_proj"),
	                                *up_w = Wgt("up_proj"), *down_w = Wgt("down_proj"),
	                                *attn_gain = Wgt("attn_norm.gain"), *mlp_gain = Wgt("mlp_norm.gain");
	if (!q_w || !k_w || !v_w || !o_w || !gate_w || !up_w || !down_w || !attn_gain || !mlp_gain) {
		*err = prefix + ": missing a required WGT1 tensor";
		return false;
	}
	out.q_weight = reinterpret_cast<const int8_t*>(q_w->data);
	out.k_weight = reinterpret_cast<const int8_t*>(k_w->data);
	out.v_weight = reinterpret_cast<const int8_t*>(v_w->data);
	out.o_weight = reinterpret_cast<const int8_t*>(o_w->data);
	out.gate_weight = reinterpret_cast<const int8_t*>(gate_w->data);
	out.up_weight = reinterpret_cast<const int8_t*>(up_w->data);
	out.down_weight = reinterpret_cast<const int8_t*>(down_w->data);
	backing.attn_norm_gain = WidenGainToInt32(*attn_gain);
	backing.mlp_norm_gain = WidenGainToInt32(*mlp_gain);
	out.attn_norm_gain = backing.attn_norm_gain.data();
	out.mlp_norm_gain = backing.mlp_norm_gain.data();

	// --- WSC1 per-output-channel fold (T-1666): one (identity, mult, shift)
	// array per output channel, per projection. Channel counts match exactly
	// what ProjectAndFunnel/the k-v-landing fold loop index these arrays at
	// (forward_sites.cpp): hidden_size for q/o/down, num_key_value_heads *
	// head_dim for k/v, intermediate_size for gate/up.
	const uint64_t hidden_size = view.config.hidden_size;
	const uint64_t intermediate_size = view.config.intermediate_size;
	const uint64_t kv_hidden_size = static_cast<uint64_t>(num_key_value_heads) * view.config.head_dim;

	if (!MarshalProjectionFold(Wsc("q_proj"), hidden_size, prefix + ".q_proj", backing.q_fold_identity,
	                            backing.q_fold_mult, backing.q_fold_shift, err) ||
	    !MarshalProjectionFold(Wsc("k_proj"), kv_hidden_size, prefix + ".k_proj",
	                            backing.k_fold_identity, backing.k_fold_mult, backing.k_fold_shift,
	                            err) ||
	    !MarshalProjectionFold(Wsc("v_proj"), kv_hidden_size, prefix + ".v_proj",
	                            backing.v_fold_identity, backing.v_fold_mult, backing.v_fold_shift,
	                            err) ||
	    !MarshalProjectionFold(Wsc("o_proj"), hidden_size, prefix + ".o_proj", backing.o_fold_identity,
	                            backing.o_fold_mult, backing.o_fold_shift, err) ||
	    !MarshalProjectionFold(Wsc("gate_proj"), intermediate_size, prefix + ".gate_proj",
	                            backing.gate_fold_identity, backing.gate_fold_mult,
	                            backing.gate_fold_shift, err) ||
	    !MarshalProjectionFold(Wsc("up_proj"), intermediate_size, prefix + ".up_proj",
	                            backing.up_fold_identity, backing.up_fold_mult, backing.up_fold_shift,
	                            err) ||
	    !MarshalProjectionFold(Wsc("down_proj"), hidden_size, prefix + ".down_proj",
	                            backing.down_fold_identity, backing.down_fold_mult,
	                            backing.down_fold_shift, err)) {
		return false;
	}
	out.q_fold_identity = backing.q_fold_identity.data();
	out.q_fold_mult = backing.q_fold_mult.data();
	out.q_fold_shift = backing.q_fold_shift.data();
	out.k_fold_identity = backing.k_fold_identity.data();
	out.k_fold_mult = backing.k_fold_mult.data();
	out.k_fold_shift = backing.k_fold_shift.data();
	out.v_fold_identity = backing.v_fold_identity.data();
	out.v_fold_mult = backing.v_fold_mult.data();
	out.v_fold_shift = backing.v_fold_shift.data();
	out.o_fold_identity = backing.o_fold_identity.data();
	out.o_fold_mult = backing.o_fold_mult.data();
	out.o_fold_shift = backing.o_fold_shift.data();
	out.gate_fold_identity = backing.gate_fold_identity.data();
	out.gate_fold_mult = backing.gate_fold_mult.data();
	out.gate_fold_shift = backing.gate_fold_shift.data();
	out.up_fold_identity = backing.up_fold_identity.data();
	out.up_fold_mult = backing.up_fold_mult.data();
	out.up_fold_shift = backing.up_fold_shift.data();
	out.down_fold_identity = backing.down_fold_identity.data();
	out.down_fold_mult = backing.down_fold_mult.data();
	out.down_fold_shift = backing.down_fold_shift.data();

	// --- ctx_fold (WSC1, per-head -- LayerWeights already carries this as an
	// array, T-518/D-SLM57) ------------------------------------------------
	const superslm::SslmTensorView* ctx_wsc = Wsc("ctx_fold");
	if (!ctx_wsc || ctx_wsc->elem_count != static_cast<uint64_t>(num_heads) * 3) {
		*err = prefix + ".ctx_fold: missing or wrong-sized WSC1 tensor";
		return false;
	}
	backing.ctx_identity.resize(num_heads);
	backing.ctx_mult.resize(num_heads);
	backing.ctx_shift.resize(num_heads);
	for (uint32_t h = 0; h < num_heads; ++h) {
		backing.ctx_identity[h] = RdI32(ctx_wsc->data + (h * 3 + 0) * 4);
		backing.ctx_mult[h] = RdI32(ctx_wsc->data + (h * 3 + 1) * 4);
		backing.ctx_shift[h] = RdI32(ctx_wsc->data + (h * 3 + 2) * 4);
	}
	out.ctx_fold_identity = backing.ctx_identity.data();
	out.ctx_fold_mult = backing.ctx_mult.data();
	out.ctx_fold_shift = backing.ctx_shift.data();

	// --- biases (BIA1, int64; T-1656/D-SLM642) -----------------------------
	const superslm::SslmTensorView* qb = Bia("q_proj");
	const superslm::SslmTensorView* kb = Bia("k_proj");
	const superslm::SslmTensorView* vb = Bia("v_proj");
	out.q_bias = qb ? reinterpret_cast<const int64_t*>(qb->data) : nullptr;
	out.k_bias = kb ? reinterpret_cast<const int64_t*>(kb->data) : nullptr;
	out.v_bias = vb ? reinterpret_cast<const int64_t*>(vb->data) : nullptr;

	// --- KV landing reciprocals (KVC1, r_t/e_t -- LandingRescale's own two
	// runtime inputs are BOTH KvLandingReciprocals' word 2 (R_t) and word 1
	// (e_t); KvLandingScales' own word 1 (e_target) is a documented
	// pending-consumer field LandingRescale does not read, per model.h's
	// KvLandingScaleOutOfDomain/KvLandingReciprocalOutOfDomain comments) ----
	backing.kv_r_t_k.resize(num_key_value_heads);
	backing.kv_e_t_k.resize(num_key_value_heads);
	backing.kv_r_t_v.resize(num_key_value_heads);
	backing.kv_e_t_v.resize(num_key_value_heads);
	for (uint32_t h = 0; h < num_key_value_heads; ++h) {
		const std::string kname = prefix + ".k_head" + std::to_string(h);
		const std::string vname = prefix + ".v_head" + std::to_string(h);
		const superslm::SslmConstantEntry* ke = view.kv_landing_reciprocals.Entry(kname);
		const superslm::SslmConstantEntry* ve = view.kv_landing_reciprocals.Entry(vname);
		if (!ke || !ve || ke->value_words < 3 || ve->value_words < 3) {
			*err = prefix + ": missing kv_landing_reciprocals entry for head " + std::to_string(h);
			return false;
		}
		backing.kv_e_t_k[h] = superslm::SslmKeyedConstants::Value(*ke, 1);
		backing.kv_r_t_k[h] = superslm::SslmKeyedConstants::Value(*ke, 2);
		backing.kv_e_t_v[h] = superslm::SslmKeyedConstants::Value(*ve, 1);
		backing.kv_r_t_v[h] = superslm::SslmKeyedConstants::Value(*ve, 2);
	}
	out.kv_landing_r_t_k = backing.kv_r_t_k.data();
	out.kv_landing_e_t_k = backing.kv_e_t_k.data();
	out.kv_landing_r_t_v = backing.kv_r_t_v.data();
	out.kv_landing_e_t_v = backing.kv_e_t_v.data();

	// --- per-query i-exp composition inputs (KVC1 composition_constants,
	// T-1655/D-SLM620) -------------------------------------------------------
	backing.iexp_m.resize(num_key_value_heads);
	backing.iexp_e.resize(num_key_value_heads);
	for (uint32_t h = 0; h < num_key_value_heads; ++h) {
		const std::string kname = prefix + ".softmax_khead" + std::to_string(h);
		const superslm::SslmConstantEntry* e = view.composition_constants.Entry(kname);
		if (!e || e->value_words < 2) {
			*err = prefix + ": missing composition_constants entry \"" + kname + "\"";
			return false;
		}
		backing.iexp_m[h] = superslm::SslmKeyedConstants::Value(*e, 0);
		backing.iexp_e[h] = superslm::SslmKeyedConstants::Value(*e, 1);
	}
	out.iexp_softmax_khead_m = backing.iexp_m.data();
	out.iexp_softmax_khead_e = backing.iexp_e.data();

	// --- per-layer site constants (KVC1 composition_constants) -------------
	bool ok = true;
	out.attn_norm_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".attn_norm", &ok);
	out.q_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".q_proj", &ok);
	out.o_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".o_proj", &ok);
	out.ctx_fold_site_constant =
	    ReadCarriedScale(view.composition_constants, prefix + ".attn_ctx", &ok);
	out.attn_residual_site_constant =
	    ReadCarriedScale(view.composition_constants, prefix + ".attn_residual", &ok);
	out.mlp_norm_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".mlp_norm", &ok);
	out.gate_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".gate_proj", &ok);
	out.up_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".up_proj", &ok);
	out.mlp_act_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".mlp_act", &ok);
	out.down_site_constant = ReadCarriedScale(view.composition_constants, prefix + ".down_proj", &ok);
	out.mlp_residual_site_constant =
	    ReadCarriedScale(view.composition_constants, prefix + ".mlp_residual", &ok);
	if (!ok) {
		*err = prefix + ": missing a required composition_constants site entry";
		return false;
	}

	return true;
}

// Scans every layer's WSC1 fold tensor for every one of the seven
// projections BEFORE any per-layer marshaling is attempted -- the same
// "scanned independently, not assumed from one element" discipline T-1652's
// build log applied to the BIA1 defect it found. Prints the full extent
// (layers affected, worst-case row counts) so the report below is grounded
// in the whole artifact, not a sample.
inline void PreflightScanWscFolds(const superslm::SslmModelView& view) {
	const char* names[] = {"q_proj", "k_proj", "v_proj", "o_proj",
	                        "gate_proj", "up_proj", "down_proj"};
	uint32_t layers_affected = 0;
	uint64_t max_rows_seen = 0;
	for (uint32_t l = 0; l < view.config.num_hidden_layers; ++l) {
		bool this_layer_affected = false;
		for (const char* name : names) {
			const std::string key = "layer" + std::to_string(l) + "." + name;
			const superslm::SslmTensorView* t = view.weight_scales.Tensor(key);
			if (!t) continue;
			const uint64_t rows = t->elem_count / 3;
			if (rows > max_rows_seen) max_rows_seen = rows;
			if (rows != 1) this_layer_affected = true;
		}
		if (this_layer_affected) ++layers_affected;
	}
	std::fprintf(stderr,
	             "preflight: %u/%u layers carry a non-degenerate (per-output-channel) WSC1 fold "
	             "tensor on at least one of q/k/v/o/gate/up/down_proj; worst case %llu rows in a "
	             "single tensor (LayerWeights now carries one fold triple per output channel per "
	             "projection, T-1666; MarshalLayer marshals every row)\n",
	             layers_affected, view.config.num_hidden_layers,
	             static_cast<unsigned long long>(max_rows_seen));
}

}  // namespace superslm_marshal

#endif  // SUPERSLM_INCLUDE_LAYER_MARSHAL_H
