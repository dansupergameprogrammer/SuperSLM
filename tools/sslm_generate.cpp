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
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
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
	std::fprintf(stderr, "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" [--max-new N]\n",
	             argv0);
}

// Little-endian byte-assembly read of one int32 element -- WGT1/WSC1/BIA1
// tensor data IS alignment-guaranteed by the loader (TensorMisaligned,
// model.h), so a reinterpret_cast would also be sound here, but this driver
// reads every artifact-carried array through explicit byte assembly to match
// this codebase's own established convention for untrusted-provenance data
// (forward_sites.cpp's own RdI64 comment: "the same discipline... for every
// other untrusted-alignment array in this tree").
int32_t RdI32(const uint8_t* p) noexcept {
	return static_cast<int32_t>(static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	                             (static_cast<uint32_t>(p[2]) << 16) |
	                             (static_cast<uint32_t>(p[3]) << 24));
}

int64_t RdI64(const uint8_t* p) noexcept {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

// Sign-extends an int8-coded WGT1 gain tensor into an owned int32_t array --
// RmsNormSite's own contract (`const int32_t* g`, forward_sites.h) needs a
// widened container; the wire format stores gain codes as int8
// (tools/sslm_model_writer.py / convert_model.py's "Weights (WGT1, int8)").
std::vector<int32_t> WidenGainToInt32(const SslmTensorView& t) {
	std::vector<int32_t> out(t.elem_count);
	for (uint64_t i = 0; i < t.elem_count; ++i) {
		out[i] = static_cast<int32_t>(static_cast<int8_t>(t.data[i]));
	}
	return out;
}

CarriedScale ReadCarriedScale(const SslmKeyedConstants& kc, const std::string& name, bool* ok) {
	const SslmConstantEntry* e = kc.Entry(name);
	if (e == nullptr || e->value_words < 2) {
		*ok = false;
		return CarriedScale{};
	}
	return CarriedScale{SslmKeyedConstants::Value(*e, 0), SslmKeyedConstants::Value(*e, 1)};
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
bool MarshalProjectionFold(const SslmTensorView* t, uint64_t channels, const std::string& label,
                            std::vector<int32_t>& id, std::vector<int32_t>& mult,
                            std::vector<int32_t>& shift, std::string* err) {
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
bool MarshalLayer(const SslmModelView& view, uint32_t l, uint32_t num_heads,
                   uint32_t num_key_value_heads, LayerBacking& backing, LayerWeights& out,
                   std::string* err) {
	const std::string prefix = "layer" + std::to_string(l);
	auto Wgt = [&](const char* suffix) -> const SslmTensorView* {
		return view.weights.Tensor(prefix + "." + suffix);
	};
	auto Wsc = [&](const char* suffix) -> const SslmTensorView* {
		return view.weight_scales.Tensor(prefix + "." + suffix);
	};
	auto Bia = [&](const char* suffix) -> const SslmTensorView* {
		return view.biases.Tensor(prefix + "." + suffix);
	};

	// --- weights (WGT1, int8) --------------------------------------------
	const SslmTensorView *q_w = Wgt("q_proj"), *k_w = Wgt("k_proj"), *v_w = Wgt("v_proj"),
	                      *o_w = Wgt("o_proj"), *gate_w = Wgt("gate_proj"), *up_w = Wgt("up_proj"),
	                      *down_w = Wgt("down_proj"), *attn_gain = Wgt("attn_norm.gain"),
	                      *mlp_gain = Wgt("mlp_norm.gain");
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
	const uint64_t kv_hidden_size =
	    static_cast<uint64_t>(num_key_value_heads) * view.config.head_dim;

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
	const SslmTensorView* ctx_wsc = Wsc("ctx_fold");
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
	const SslmTensorView* qb = Bia("q_proj");
	const SslmTensorView* kb = Bia("k_proj");
	const SslmTensorView* vb = Bia("v_proj");
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
		const SslmConstantEntry* ke = view.kv_landing_reciprocals.Entry(kname);
		const SslmConstantEntry* ve = view.kv_landing_reciprocals.Entry(vname);
		if (!ke || !ve || ke->value_words < 3 || ve->value_words < 3) {
			*err = prefix + ": missing kv_landing_reciprocals entry for head " + std::to_string(h);
			return false;
		}
		backing.kv_e_t_k[h] = SslmKeyedConstants::Value(*ke, 1);
		backing.kv_r_t_k[h] = SslmKeyedConstants::Value(*ke, 2);
		backing.kv_e_t_v[h] = SslmKeyedConstants::Value(*ve, 1);
		backing.kv_r_t_v[h] = SslmKeyedConstants::Value(*ve, 2);
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
		const SslmConstantEntry* e = view.composition_constants.Entry(kname);
		if (!e || e->value_words < 2) {
			*err = prefix + ": missing composition_constants entry \"" + kname + "\"";
			return false;
		}
		backing.iexp_m[h] = SslmKeyedConstants::Value(*e, 0);
		backing.iexp_e[h] = SslmKeyedConstants::Value(*e, 1);
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
void PreflightScanWscFolds(const SslmModelView& view) {
	const char* names[] = {"q_proj", "k_proj", "v_proj", "o_proj",
	                        "gate_proj", "up_proj", "down_proj"};
	uint32_t layers_affected = 0;
	uint64_t max_rows_seen = 0;
	for (uint32_t l = 0; l < view.config.num_hidden_layers; ++l) {
		bool this_layer_affected = false;
		for (const char* name : names) {
			const std::string key = "layer" + std::to_string(l) + "." + name;
			const SslmTensorView* t = view.weight_scales.Tensor(key);
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
	// tie_word_embeddings: the head reuses the embedding matrix; no separate
	// lm_head WGT1 tensor exists in this artifact (verified: only "embed" is
	// present at the [vocab_size, hidden_size] shape).
	const int8_t* head_weights = embed_weights;

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

	const SslmForwardStatus decode_status = RunGreedyDecodeLoop(
	    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim, num_kv_heads,
	    model_view.config.intermediate_size, context_cap, model_view.rope_tables, prompt_tokens.data(),
	    prompt_tokens.size(), embed_weights, embed_site_constant, final_norm_gain.data(),
	    final_norm_site_constant, head_weights, static_cast<int32_t>(model_view.config.vocab_size),
	    /*stop_ids=*/nullptr, /*stop_count=*/0, max_new_tokens, workspace.data(), workspace.size(),
	    out_tokens.data(), out_logit_rows.data(), out_tokens.size(), &out_tokens_produced, &stop_reason,
	    model_view.config.kv_precision);

	if (decode_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=decode: status=%s\n",
		             SslmForwardStatusName(decode_status));
		return 1;
	}

	std::printf("output_tokens (%zu):", out_tokens_produced);
	for (size_t i = 0; i < out_tokens_produced; ++i) std::printf(" %d", out_tokens[i]);
	std::printf("\n");
	std::printf("stop_reason: %d\n", static_cast<int>(stop_reason));

	const auto t_end = std::chrono::steady_clock::now();
	std::printf("wall_time_seconds: %.3f\n", std::chrono::duration<double>(t_end - t_start).count());
	return 0;
}
