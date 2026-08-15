// T-2039: C5's first composed bit-identity token. Loads the real
// qwen2.5-1.5b-instruct.sslm artifact through the same production entry
// point every consumer uses (SslmModel::Load), marshals it into a
// LayerWeights[] via the SAME adapter tools/sslm_generate.cpp already
// proves correct (tools/sslm_marshal.h), embeds one token, and runs it
// through all 28 layers on BOTH the CPU oracle (production RunLayerLoop)
// and the GPU port (RunLayerLoopGpu) from the IDENTICAL initial
// SequenceLayerState and workspace. Compares the full SequenceLayerState-
// complete surface (residual codes+scale, layer_index, K/V cache rows,
// kv_saturation_count, context_length) plus the final_norm+logits path.
//
// Throwaway harness, matching tools/t1657_load_harness.cpp's own precedent
// ("Not part of the build.bat/CMake build graph -- compiled and run
// directly for this session's own verification").
//
// Usage: t2039_c5_harness <model.sslm> [token_id]
#include <cstdio>
#include <cstring>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [token_id]\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const int32_t token_id = argc >= 3 ? std::atoi(argv[2]) : 0;

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED: could not read \"%s\"\n", model_path.c_str());
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
	std::printf(
	    "model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u intermediate=%u vocab=%u "
	    "context_cap=%u tie=%d\n",
	    model_view.config.hidden_size, model_view.config.num_hidden_layers,
	    model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	    model_view.config.head_dim, model_view.config.intermediate_size, model_view.config.vocab_size,
	    model_view.config.context_cap, model_view.config.tie_word_embeddings ? 1 : 0);

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);

	PreflightScanWscFolds(model_view);
	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	if (!embed_w) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);

	if (token_id < 0 || token_id >= model_view.config.vocab_size) {
		std::fprintf(stderr, "FAILED: token_id=%d out of range [0,%u)\n", token_id,
		             model_view.config.vocab_size);
		return 1;
	}

	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est = EmbedEntry(token_id, model_view.config.vocab_size, embed_weights,
	                                          hidden_size, embed_site_constant, embed_codes.data(),
	                                          &embed_scale);
	if (est != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=embed: status=%s\n", SslmForwardStatusName(est));
		return 1;
	}
	std::printf("token_id=%d embedded: scale.m=%lld scale.e=%lld\n", token_id,
	            (long long)embed_scale.m, (long long)embed_scale.e);

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;
	std::printf("workspace (KV cache) bytes = %zu (%.2f MiB)\n", kv_bytes, kv_bytes / (1024.0 * 1024.0));

	// --- CPU oracle: production RunLayerLoop, layer_budget = all layers. ---
	std::vector<int8_t> cpu_codes(hidden_size);
	std::memcpy(cpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState cpu_seq;
	cpu_seq.hidden_codes = cpu_codes.data();
	cpu_seq.hidden_scale = embed_scale;
	cpu_seq.layer_index = 0;
	std::vector<uint8_t> cpu_ws(kv_bytes, 0);
	const SslmForwardStatus cpu_status =
	    RunLayerLoop(cpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers,
	                 hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
	                 model_view.rope_tables, cpu_ws.data(), cpu_ws.size());
	std::printf("CPU oracle: status=%s layer_index=%u kv_saturation_count=%llu context_length=%lld\n",
	            SslmForwardStatusName(cpu_status), cpu_seq.layer_index,
	            (unsigned long long)cpu_seq.kv_saturation_count, (long long)cpu_seq.context_length);

	// --- GPU port: RunLayerLoopGpu, IDENTICAL inputs. ---
	std::vector<int8_t> gpu_codes(hidden_size);
	std::memcpy(gpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState gpu_seq;
	gpu_seq.hidden_codes = gpu_codes.data();
	gpu_seq.hidden_scale = embed_scale;
	gpu_seq.layer_index = 0;
	std::vector<uint8_t> gpu_ws(kv_bytes, 0);
	const SslmForwardStatus gpu_status = superslm_gpu::RunLayerLoopGpu(
	    gpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, gpu_ws.data(),
	    gpu_ws.size());
	std::printf("GPU port:   status=%s layer_index=%u kv_saturation_count=%llu context_length=%lld\n",
	            SslmForwardStatusName(gpu_status), gpu_seq.layer_index,
	            (unsigned long long)gpu_seq.kv_saturation_count, (long long)gpu_seq.context_length);

	// --- Compare, verbatim, the FIRST divergence if any. ---
	bool all_match = true;
	if (cpu_status != gpu_status) {
		std::printf("DIVERGENCE: status: CPU=%s GPU=%s\n", SslmForwardStatusName(cpu_status),
		            SslmForwardStatusName(gpu_status));
		all_match = false;
	}
	if (cpu_seq.layer_index != gpu_seq.layer_index) {
		std::printf("DIVERGENCE: layer_index: CPU=%u GPU=%u\n", cpu_seq.layer_index, gpu_seq.layer_index);
		all_match = false;
	}
	if (cpu_seq.hidden_scale.m != gpu_seq.hidden_scale.m || cpu_seq.hidden_scale.e != gpu_seq.hidden_scale.e) {
		std::printf("DIVERGENCE: hidden_scale: CPU=(%lld,%lld) GPU=(%lld,%lld)\n",
		            (long long)cpu_seq.hidden_scale.m, (long long)cpu_seq.hidden_scale.e,
		            (long long)gpu_seq.hidden_scale.m, (long long)gpu_seq.hidden_scale.e);
		all_match = false;
	}
	if (cpu_seq.kv_saturation_count != gpu_seq.kv_saturation_count) {
		std::printf("DIVERGENCE: kv_saturation_count: CPU=%llu GPU=%llu\n",
		            (unsigned long long)cpu_seq.kv_saturation_count,
		            (unsigned long long)gpu_seq.kv_saturation_count);
		all_match = false;
	}
	if (cpu_seq.context_length != gpu_seq.context_length) {
		std::printf("DIVERGENCE: context_length: CPU=%lld GPU=%lld\n", (long long)cpu_seq.context_length,
		            (long long)gpu_seq.context_length);
		all_match = false;
	}
	int first_code_mismatch = -1;
	for (size_t i = 0; i < hidden_size; ++i) {
		if (cpu_codes[i] != gpu_codes[i]) {
			first_code_mismatch = static_cast<int>(i);
			break;
		}
	}
	if (first_code_mismatch >= 0) {
		std::printf("DIVERGENCE: hidden_codes[%d]: CPU=%d GPU=%d (first mismatch of %zu elements)\n",
		            first_code_mismatch, cpu_codes[first_code_mismatch], gpu_codes[first_code_mismatch],
		            hidden_size);
		all_match = false;
	}

	// K/V cache: every layer, every kv_head, position 0 (the only position
	// this single-token run lands), compared byte-for-byte.
	bool kv_match = true;
	for (uint32_t l = 0; l < num_hidden_layers && kv_match; ++l) {
		for (uint32_t h = 0; h < num_kv_heads && kv_match; ++h) {
			const int8_t* cpu_k = KeyRow(cpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			const int8_t* gpu_k =
			    superslm_gpu::KeyRowGpu(gpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			const int8_t* cpu_v = ValueRow(cpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			const int8_t* gpu_v =
			    superslm_gpu::ValueRowGpu(gpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			for (size_t d = 0; d < head_dim; ++d) {
				if (cpu_k[d] != gpu_k[d]) {
					std::printf("DIVERGENCE: K[layer=%u][kv_head=%u][dim=%zu]: CPU=%d GPU=%d\n", l, h, d,
					            cpu_k[d], gpu_k[d]);
					kv_match = false;
					all_match = false;
					break;
				}
				if (cpu_v[d] != gpu_v[d]) {
					std::printf("DIVERGENCE: V[layer=%u][kv_head=%u][dim=%zu]: CPU=%d GPU=%d\n", l, h, d,
					            cpu_v[d], gpu_v[d]);
					kv_match = false;
					all_match = false;
					break;
				}
			}
		}
	}

	if (all_match) {
		std::printf("RESULT: BIT-IDENTICAL across the full SequenceLayerState-complete surface "
		            "(hidden_codes[%zu], hidden_scale, layer_index, kv_saturation_count, "
		            "context_length) and every K/V cache row across all %u layers.\n",
		            hidden_size, num_hidden_layers);
	} else {
		std::printf("RESULT: DIVERGENCE FOUND (see above) -- not bit-identical.\n");
	}

	// --- Downstream: final_norm + logits, from each side's own final state.
	// T-2045 (S4, M4, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md):
	// this section is NOT an independent GPU measurement. RmsNormSite and
	// LogitsSite are CPU functions, run here on the HOST over cpu_codes and
	// gpu_codes -- both sides are the identical deterministic function of
	// hidden_codes/hidden_scale, already compared bit-for-bit above. A real
	// GPU-side final_norm+head dispatch is outside this design's ratified
	// B-step list (Sec11 covers the 16 composed sites + commit, never the
	// head projection), so building one is out of this remedy round's own
	// scope -- reported here as a DERIVED CONSEQUENCE of the already-real
	// SequenceLayerState/K-V comparison, never as a third independent line
	// (S4's own alternative: "reported as a derived consequence, or made a
	// real measurement by computing the head on GPU" -- this takes the
	// first). Every return status is now checked (M4); a missing tensor or
	// constant now prints an explicit skip note rather than silently
	// omitting the section.
	if (cpu_status == SslmForwardStatus::Ok && gpu_status == SslmForwardStatus::Ok) {
		const SslmTensorView* final_gain_w = model_view.weights.Tensor("final_norm.gain");
		const int8_t* head_weights = nullptr;
		if (model_view.config.tie_word_embeddings) {
			head_weights = embed_weights;
		} else {
			const SslmTensorView* lm_head_w = model_view.weights.Tensor("lm_head");
			head_weights = lm_head_w ? reinterpret_cast<const int8_t*>(lm_head_w->data) : nullptr;
		}
		if (!final_gain_w || !head_weights) {
			std::printf("LOGITS: skipped -- final_norm.gain or the head tensor is absent from this "
			            "artifact.\n");
		} else {
			std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
			bool ok2 = true;
			CarriedScale final_norm_site_constant =
			    ReadCarriedScale(model_view.composition_constants, "final_norm", &ok2);
			if (!ok2) {
				std::printf("LOGITS: skipped -- no \"final_norm\" composition constant in this artifact.\n");
			} else {
				const size_t vocab_size_z = static_cast<size_t>(model_view.config.vocab_size);
				std::vector<int8_t> cpu_final(hidden_size), gpu_final(hidden_size);
				CarriedScale cpu_final_scale{}, gpu_final_scale{};
				const SslmForwardStatus cpu_norm_st =
				    RmsNormSite(cpu_codes.data(), final_norm_gain.data(), hidden_size, cpu_seq.hidden_scale,
				                final_norm_site_constant, cpu_final.data(), &cpu_final_scale, "final_norm");
				const SslmForwardStatus gpu_norm_st =
				    RmsNormSite(gpu_codes.data(), final_norm_gain.data(), hidden_size, gpu_seq.hidden_scale,
				                final_norm_site_constant, gpu_final.data(), &gpu_final_scale, "final_norm");
				if (cpu_norm_st != SslmForwardStatus::Ok || gpu_norm_st != SslmForwardStatus::Ok) {
					std::printf("LOGITS: skipped -- final_norm returned CPU=%s GPU=%s (not both Ok).\n",
					            SslmForwardStatusName(cpu_norm_st), SslmForwardStatusName(gpu_norm_st));
				} else {
					std::vector<int64_t> cpu_wide(vocab_size_z), gpu_wide(vocab_size_z);
					std::vector<int32_t> cpu_logits(vocab_size_z), gpu_logits(vocab_size_z);
					const SslmForwardStatus cpu_logit_st = LogitsSite(
					    cpu_final.data(), hidden_size, head_weights, model_view.config.vocab_size,
					    cpu_wide.data(), cpu_logits.data());
					const SslmForwardStatus gpu_logit_st = LogitsSite(
					    gpu_final.data(), hidden_size, head_weights, model_view.config.vocab_size,
					    gpu_wide.data(), gpu_logits.data());
					if (cpu_logit_st != SslmForwardStatus::Ok || gpu_logit_st != SslmForwardStatus::Ok) {
						std::printf("LOGITS: skipped -- LogitsSite returned CPU=%s GPU=%s (not both Ok).\n",
						            SslmForwardStatusName(cpu_logit_st), SslmForwardStatusName(gpu_logit_st));
					} else {
						int first_logit_mismatch = -1;
						for (size_t i = 0; i < vocab_size_z; ++i) {
							if (cpu_logits[i] != gpu_logits[i]) {
								first_logit_mismatch = static_cast<int>(i);
								break;
							}
						}
						if (first_logit_mismatch >= 0) {
							std::printf(
							    "DIVERGENCE: logits[%d] (post final_norm+head, off this run's own "
							    "hidden_codes): CPU=%d GPU=%d\n",
							    first_logit_mismatch, cpu_logits[first_logit_mismatch],
							    gpu_logits[first_logit_mismatch]);
						} else {
							const int32_t cpu_argmax =
							    ArgmaxLowestIndexTieBreak(cpu_logits.data(), vocab_size_z);
							const int32_t gpu_argmax =
							    ArgmaxLowestIndexTieBreak(gpu_logits.data(), vocab_size_z);
							std::printf(
							    "LOGITS (DERIVED, not an independent GPU measurement -- both sides run "
							    "CPU final_norm+head over the already bit-compared hidden_codes above): "
							    "identical across all %zu vocab entries. argmax token: CPU=%d GPU=%d\n",
							    vocab_size_z, cpu_argmax, gpu_argmax);
						}
					}
				}
			}
		}
	}

	return 0;
}
