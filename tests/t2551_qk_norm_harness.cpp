// T-2551 Track B acceptance harness. Committed (not disposable, unlike tools/t2039_c5_harness.cpp
// and T-2425's own spike harnesses this file's structure is drawn from) -- Track B's own
// "Acceptance for Track B alone" needs a real load-marshal-forward drive against a real converted
// artifact, and this is the record of exactly what ran. Not part of the CMake build graph (matches
// tools/t2432_geometry_harness.cpp's own precedent) -- compiled and run directly, per this
// ticket's own build log.
//
// Modeled on tools/t2432_geometry_harness.cpp (Track A's own committed acceptance harness): loads
// a real .sslm artifact, marshals every layer (a marshal failure's own diagnostic IS the
// rejection-cell check when this harness is pointed at a deliberately malformed artifact -- no
// separate tool needed), embeds one token, and runs it through every layer on BOTH the CPU oracle
// (production RunLayerLoop, the single-token path) and the GPU port (RunLayerLoopGpuSubmit/
// Finish) from an IDENTICAL initial SequenceLayerState and workspace -- the determinism crown,
// Track B's own separate, genuinely bit-exact claim (design §6 Track B's own "Acceptance for
// Track B alone").
//
// A trace hook captures the LAST "q_norm"/"k_norm" chain record fired (ApplyQkNormSite's own
// LayerSite naming, forward_sites.cpp) -- proof the call site actually ran, not merely that the
// forward pass returned Ok (a model with q_norm/k_norm tensors absent would also return Ok, and a
// broken gate that skipped the call silently would too).
//
// A second drive exercises RunLayerLoopChunkBatched directly (bypassing the ABI/sslm_prefill
// entirely, matching this ticket's own "No ABI verb" scope line -- Track D's sslm_seq_get_hidden_
// state is not called anywhere in this file) -- the chunk-batched path is what sslm_prefill
// actually calls in production (forward_sites.h's own header comment on RunLayerLoopChunkBatched),
// and it carries ApplyQkNormSite's own second call site, never exercised by the single-token drive
// above.
//
// The final hidden state from both drives is printed as a single parseable line
// ("HIDDEN_STATE_FOR_ORACLE: scale_m=... scale_e=... codes=c0,c1,...") for a separate Python step
// to compare against the (now QK-norm-bearing, D-SLM5676) float reference -- QUARANTINED per this
// ticket's own brief, recorded and never headlined as pass/fail pending the slice 3 tolerance.
//
// Usage: t2551_qk_norm_harness <model.sslm> [token_id]
#include <cstdio>
#include <cstring>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/trace_hook.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;

namespace {
struct QkNormCapture {
	bool q_captured = false, k_captured = false;
	std::vector<int8_t> q_codes, k_codes;
	// Materiality evidence (D-SLM5312's own established shape, T-2425): the PRE-norm q_proj
	// output, captured from the SAME run, at the SAME (last) layer, so it can be diffed
	// directly against the POST-norm q_codes above -- proof the call site changes the real
	// forward pass, not merely that it executes without crashing.
	bool q_proj_captured = false;
	std::vector<int8_t> q_proj_codes;
};

bool EndsWith(std::string_view s, const char* suffix, size_t suffix_len) {
	return s.size() >= suffix_len && s.compare(s.size() - suffix_len, suffix_len, suffix) == 0;
}

void QkNormHook(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord* kv, void* user) {
	(void)kv;
	if (chain == nullptr) return;
	QkNormCapture* cap = static_cast<QkNormCapture*>(user);
	// LayerSite's own convention: "layer{L}.q_norm" / "layer{L}.k_norm" / "layer{L}.q_proj.
	// requant" -- match the SUFFIX. Keeps the LAST fire per token (the highest head/layer index
	// run), matching t2432_geometry_harness.cpp's own established convention for this class of
	// hook. "q_proj.requant" is checked before "q_norm" is possible to match against it (both
	// end differently -- "requant" vs "q_norm" -- so there is no ambiguity; order here is
	// cosmetic).
	if (EndsWith(chain->site, "q_proj.requant", 14)) {
		cap->q_proj_captured = true;
		cap->q_proj_codes.assign(chain->codes.begin(), chain->codes.end());
	} else if (EndsWith(chain->site, "q_norm", 6)) {
		cap->q_captured = true;
		cap->q_codes.assign(chain->codes.begin(), chain->codes.end());
	} else if (EndsWith(chain->site, "k_norm", 6)) {
		cap->k_captured = true;
		cap->k_codes.assign(chain->codes.begin(), chain->codes.end());
	}
}

void PrintHiddenStateForOracle(const char* label, const int8_t* codes, size_t n, CarriedScale scale) {
	std::printf("HIDDEN_STATE_FOR_ORACLE[%s]: scale_m=%lld scale_e=%lld codes=", label,
	            (long long)scale.m, (long long)scale.e);
	for (size_t i = 0; i < n; ++i) std::printf("%s%d", i ? "," : "", (int)codes[i]);
	std::printf("\n");
}
}  // namespace

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
	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t q_width = static_cast<size_t>(num_heads) * head_dim;
	std::printf(
	    "model loaded: hidden_size=%zu layers=%u heads=%u/%u head_dim=%zu q_width=%zu "
	    "(square=%d) intermediate=%zu vocab=%u context_cap=%lld option_g_fused_k_landing=%d\n",
	    hidden_size, num_hidden_layers, num_heads, num_kv_heads, head_dim, q_width,
	    q_width == hidden_size ? 1 : 0, intermediate_size, model_view.config.vocab_size,
	    (long long)context_cap, model_view.option_g_fused_k_landing ? 1 : 0);

	PreflightScanWscFolds(model_view);
	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	uint32_t layers_with_qk_norm = 0;
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			// This IS the rejection-cell check: point this harness at an artifact carrying
			// asymmetric q_norm/k_norm presence, or option_g_fused_k_landing=true combined with
			// presence, or weight_scales without the matching composition_constants entry, and
			// this diagnostic is the recorded evidence.
			std::fprintf(stderr, "MARSHAL REJECTED: layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
		if (layers[l].q_norm_gain != nullptr) ++layers_with_qk_norm;
	}
	std::printf("MARSHAL: OK -- %u/%u layers carry q_norm/k_norm\n", layers_with_qk_norm,
	            num_hidden_layers);

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

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	// --- CPU oracle: production RunLayerLoop (single-token path, RunLayerLoopImpl), layer_budget
	//     = all layers, q_width explicit, a trace hook installed to capture q_norm/k_norm's own
	//     output row VALUES. ---
	std::vector<int8_t> cpu_codes(hidden_size);
	std::memcpy(cpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState cpu_seq;
	cpu_seq.hidden_codes = cpu_codes.data();
	cpu_seq.hidden_scale = embed_scale;
	cpu_seq.layer_index = 0;
	std::vector<uint8_t> cpu_ws(kv_bytes, 0);
	QkNormCapture cap;
	SslmTraceHookState hook_state;
	SslmSetTraceHook(hook_state, &QkNormHook, &cap);
	const SslmForwardStatus cpu_status = RunLayerLoop(
	    cpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, cpu_ws.data(),
	    cpu_ws.size(), /*site_prefix=*/{}, /*token_index=*/0, &hook_state, q_width);
	SslmSetTraceHook(hook_state, nullptr, nullptr);
	std::printf("CPU oracle (RunLayerLoopImpl): status=%s layer_index=%u\n",
	            SslmForwardStatusName(cpu_status), cpu_seq.layer_index);

	bool qk_norm_ran = true;
	if (layers_with_qk_norm > 0) {
		if (!cap.q_captured) {
			std::printf("QK_NORM CALL-SITE CHECK: FAILED -- q_norm trace hook never fired despite "
			            "%u layer(s) carrying q_norm\n", layers_with_qk_norm);
			qk_norm_ran = false;
		}
		if (!cap.k_captured) {
			std::printf("QK_NORM CALL-SITE CHECK: FAILED -- k_norm trace hook never fired despite "
			            "%u layer(s) carrying k_norm\n", layers_with_qk_norm);
			qk_norm_ran = false;
		}
		if (qk_norm_ran) {
			std::printf("QK_NORM CALL-SITE CHECK: PASS -- q_norm (width=%zu) and k_norm (width=%zu) "
			            "both fired (CPU, RunLayerLoopImpl, last layer run, token 0)\n",
			            cap.q_codes.size(), cap.k_codes.size());
		}
		// Materiality (D-SLM5312's own established shape, T-2425): the norm's own effect on the
		// real forward pass, not merely that the call site executes -- q_proj.requant's own
		// PRE-norm codes vs q_norm's own POST-norm codes, same run, same layer.
		if (cap.q_proj_captured && cap.q_captured) {
			bool q_materially_differs = cap.q_proj_codes != cap.q_codes;
			std::printf("QK_NORM MATERIALITY CHECK (Q): pre-norm=[");
			for (size_t i = 0; i < cap.q_proj_codes.size() && i < cap.q_codes.size(); ++i) {
				std::printf("%s%d", i ? "," : "", (int)cap.q_proj_codes[i]);
			}
			std::printf("] post-norm=[");
			for (size_t i = 0; i < cap.q_codes.size(); ++i) {
				std::printf("%s%d", i ? "," : "", (int)cap.q_codes[i]);
			}
			std::printf("] %s\n", q_materially_differs ? "DIFFERS (materiality confirmed)"
			                                            : "IDENTICAL (no measurable effect at "
			                                              "this fixture's own int8 code "
			                                              "granularity -- not itself a defect; "
			                                              "see build log)");
		}
	} else {
		std::printf("QK_NORM CALL-SITE CHECK: SKIPPED -- this artifact carries no q_norm/k_norm "
		            "tensors (red-state / non-QK-norm fixture)\n");
	}

	// --- GPU port: RunLayerLoopGpuSubmit/Finish, IDENTICAL inputs. ---
	std::vector<int8_t> gpu_codes(hidden_size);
	std::memcpy(gpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState gpu_seq;
	gpu_seq.hidden_codes = gpu_codes.data();
	gpu_seq.hidden_scale = embed_scale;
	gpu_seq.layer_index = 0;
	std::vector<uint8_t> gpu_ws(kv_bytes, 0);
	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const SslmForwardStatus gpu_submit_status = superslm_gpu::RunLayerLoopGpuSubmit(
	    gpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, gpu_ws.data(),
	    gpu_ws.size(), /*external_kv_resident=*/nullptr, /*io_external_kv_needs_resume_barrier=*/nullptr,
	    &inflight, /*external_weights_resident=*/nullptr, /*external_rope_cos_resident=*/nullptr,
	    /*external_rope_sin_resident=*/nullptr, /*external_rope_has=*/false,
	    /*external_rope_cos_elems=*/0, /*external_rope_sin_elems=*/0, /*adapter_bridge=*/nullptr,
	    q_width, /*out_q_codes=*/nullptr, /*out_q_codes_capacity=*/0);
	SslmForwardStatus gpu_status = gpu_submit_status;
	if (inflight) {
		int32_t ready = 0;
		gpu_status = superslm_gpu::RunLayerLoopGpuFinish(inflight, gpu_seq, gpu_ws.data(), /*block=*/1,
		                                                  &ready, /*out_q_codes=*/nullptr);
	}
	std::printf("GPU port (RunLayerLoopGpuSubmit/Finish): status=%s layer_index=%u\n",
	            SslmForwardStatusName(gpu_status), gpu_seq.layer_index);

	bool all_match = true;
	if (cpu_status != gpu_status) {
		std::printf("DIVERGENCE: status: CPU=%s GPU=%s\n", SslmForwardStatusName(cpu_status),
		            SslmForwardStatusName(gpu_status));
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
	if (cpu_seq.hidden_scale.m != gpu_seq.hidden_scale.m ||
	    cpu_seq.hidden_scale.e != gpu_seq.hidden_scale.e) {
		std::printf("DIVERGENCE: hidden_scale: CPU=(%lld,%lld) GPU=(%lld,%lld)\n",
		            (long long)cpu_seq.hidden_scale.m, (long long)cpu_seq.hidden_scale.e,
		            (long long)gpu_seq.hidden_scale.m, (long long)gpu_seq.hidden_scale.e);
		all_match = false;
	}
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
				if (cpu_k[d] != gpu_k[d] || cpu_v[d] != gpu_v[d]) {
					std::printf("DIVERGENCE: K/V[layer=%u][kv_head=%u][dim=%zu]: cpu_k=%d gpu_k=%d "
					            "cpu_v=%d gpu_v=%d\n",
					            l, h, d, cpu_k[d], gpu_k[d], cpu_v[d], gpu_v[d]);
					kv_match = false;
					all_match = false;
					break;
				}
			}
		}
	}
	if (all_match) {
		std::printf("DETERMINISM CROWN: PASS -- CPU/GPU bit-identical end-to-end across "
		            "hidden_codes[%zu], hidden_scale, and every K/V row (%u layers, single-token "
		            "path).\n", hidden_size, num_hidden_layers);
	} else {
		std::printf("DETERMINISM CROWN: FAIL (see DIVERGENCE lines above)\n");
	}

	if (cpu_status == SslmForwardStatus::Ok) {
		PrintHiddenStateForOracle("cpu_single_token", cpu_codes.data(), hidden_size,
		                          cpu_seq.hidden_scale);
	}

	// --- Chunk-batched CPU drive (RunLayerLoopChunkBatched -- the path sslm_prefill actually
	//     calls in production, forward_sites.h's own header comment) -- direct call, no ABI, no
	//     Track D verb, matching this ticket's own scope line. Three tokens, so the norm's own
	//     per-position/per-head loop runs more than once. ---
	{
		const size_t chunk_tokens = 3;
		std::vector<int32_t> chunk_ids;
		for (size_t i = 0; i < chunk_tokens; ++i) {
			chunk_ids.push_back(static_cast<int32_t>((token_id + static_cast<int32_t>(i)) %
			                                          model_view.config.vocab_size));
		}
		std::vector<int8_t> chunk_codes(chunk_tokens * hidden_size);
		std::vector<CarriedScale> chunk_scales(chunk_tokens);
		for (size_t t = 0; t < chunk_tokens; ++t) {
			CarriedScale sc{};
			const SslmForwardStatus e =
			    EmbedEntry(chunk_ids[t], model_view.config.vocab_size, embed_weights, hidden_size,
			               embed_site_constant, chunk_codes.data() + t * hidden_size, &sc);
			if (e != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=chunk_embed: token=%zu status=%s\n", t,
				             SslmForwardStatusName(e));
				return 1;
			}
			chunk_scales[t] = sc;
		}
		std::vector<uint8_t> chunk_ws(kv_bytes, 0);
		uint64_t kv_sat = 0;
		QkNormCapture chunk_cap;
		SslmTraceHookState chunk_hook;
		SslmSetTraceHook(chunk_hook, &QkNormHook, &chunk_cap);
		const SslmForwardStatus chunk_status = RunLayerLoopChunkBatched(
		    chunk_codes.data(), chunk_scales.data(), chunk_tokens, layers.data(), num_hidden_layers,
		    hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
		    /*context_length_start=*/0, model_view.rope_tables, chunk_ws.data(), chunk_ws.size(),
		    /*option_g_fused_k_landing=*/model_view.option_g_fused_k_landing, &kv_sat,
		    /*site_prefix=*/{}, &chunk_hook, q_width);
		SslmSetTraceHook(chunk_hook, nullptr, nullptr);
		std::printf("CPU chunk-batched (RunLayerLoopChunkBatched, %zu tokens): status=%s\n",
		            chunk_tokens, SslmForwardStatusName(chunk_status));
		if (layers_with_qk_norm > 0) {
			std::printf("QK_NORM CALL-SITE CHECK (chunk-batched): q_norm_fired=%d k_norm_fired=%d\n",
			            chunk_cap.q_captured ? 1 : 0, chunk_cap.k_captured ? 1 : 0);
			if (!chunk_cap.q_captured || !chunk_cap.k_captured) qk_norm_ran = false;
		}
		if (chunk_status == SslmForwardStatus::Ok) {
			PrintHiddenStateForOracle("cpu_chunk_last_token",
			                          chunk_codes.data() + (chunk_tokens - 1) * hidden_size,
			                          hidden_size, chunk_scales[chunk_tokens - 1]);
		}
	}

	if (all_match && qk_norm_ran && cpu_status == SslmForwardStatus::Ok &&
	    gpu_status == SslmForwardStatus::Ok) {
		std::printf("RESULT: PASS\n");
		return 0;
	}
	std::printf("RESULT: FAIL (see above)\n");
	return 1;
}
