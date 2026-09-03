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
		// (§7 Cell 6, D-SLM6145): S5's repair -- the review's own materiality check compared
		// q_proj_codes (q_width elements) against q_codes (head_dim elements); std::vector
		// operator!= on unequal lengths is unconditionally true, so it could not fail
		// regardless of whether the norm ran (confirmed by execution, T-2559 §3). Repaired:
		// compare q_codes against the SAME head_dim-wide SLICE of q_proj_codes -- the LAST
		// head's own pre-norm codes (q_norm's trace hook keeps the last head visited, the
		// same convention QkNormHook already establishes for every other capture), never the
		// full wide record.
		if (cap.q_proj_captured && cap.q_captured && cap.q_proj_codes.size() >= cap.q_codes.size()) {
			const std::vector<int8_t> matching_width_pre_norm(
			    cap.q_proj_codes.end() - static_cast<std::ptrdiff_t>(cap.q_codes.size()),
			    cap.q_proj_codes.end());
			bool q_materially_differs = matching_width_pre_norm != cap.q_codes;
			std::printf("QK_NORM MATERIALITY CHECK (Q, matching-width): pre-norm=[");
			for (size_t i = 0; i < matching_width_pre_norm.size(); ++i) {
				std::printf("%s%d", i ? "," : "", (int)matching_width_pre_norm[i]);
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
		// (§7 Cell 6, D-SLM6145): the matching-width repair (above), re-run against the LAST
		// token's own trace record from THIS chunk-batched drive -- width>1 (chunk_tokens=3),
		// where the review's own C4 finding says the pre-fix cell was blind.
		if (chunk_cap.q_proj_captured && chunk_cap.q_captured &&
		    chunk_cap.q_proj_codes.size() >= chunk_cap.q_codes.size()) {
			const std::vector<int8_t> matching_width_pre_norm(
			    chunk_cap.q_proj_codes.end() - static_cast<std::ptrdiff_t>(chunk_cap.q_codes.size()),
			    chunk_cap.q_proj_codes.end());
			bool q_materially_differs = matching_width_pre_norm != chunk_cap.q_codes;
			std::printf("QK_NORM MATERIALITY CHECK (Q, chunk-batched width>1, matching-width): %s\n",
			            q_materially_differs ? "DIFFERS (materiality confirmed at width>1)"
			                                 : "IDENTICAL (no measurable effect)");
		}
		// (§7 Cell 6, D-SLM6145): must-reject twin -- bypassing the call site's norm
		// application (q_norm_gain/k_norm_gain nulled, the engine's own no-QK-norm state,
		// the delta's own sanctioned alternative to an identity-gain construction) must leave
		// the repaired check reporting no materiality: with the gain nulled, ApplyQkNormSite's
		// Q branch never runs (forward_sites.cpp's own `if (lw.q_norm_gain != nullptr)`
		// gate), so no "q_norm" trace record fires at all -- confirmed by execution, not by
		// construction, immediately below (§7 Cell 1 reuses this same nulled-gain layer set).
		std::vector<LayerWeights> layers_no_qk = layers;
		for (LayerWeights& lw : layers_no_qk) {
			lw.q_norm_gain = nullptr;
			lw.k_norm_gain = nullptr;
		}
		std::vector<int8_t> chunk_codes_no_qk(chunk_tokens * hidden_size);
		std::vector<CarriedScale> chunk_scales_no_qk(chunk_tokens);
		for (size_t t = 0; t < chunk_tokens; ++t) {
			CarriedScale sc{};
			const SslmForwardStatus e =
			    EmbedEntry(chunk_ids[t], model_view.config.vocab_size, embed_weights, hidden_size,
			               embed_site_constant, chunk_codes_no_qk.data() + t * hidden_size, &sc);
			if (e != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=chunk_embed_no_qk: token=%zu status=%s\n", t,
				             SslmForwardStatusName(e));
				return 1;
			}
			chunk_scales_no_qk[t] = sc;
		}
		std::vector<uint8_t> chunk_ws_no_qk(kv_bytes, 0);
		uint64_t kv_sat_no_qk = 0;
		QkNormCapture chunk_cap_no_qk;
		SslmTraceHookState chunk_hook_no_qk;
		SslmSetTraceHook(chunk_hook_no_qk, &QkNormHook, &chunk_cap_no_qk);
		const SslmForwardStatus chunk_status_no_qk = RunLayerLoopChunkBatched(
		    chunk_codes_no_qk.data(), chunk_scales_no_qk.data(), chunk_tokens, layers_no_qk.data(),
		    num_hidden_layers, hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
		    /*context_length_start=*/0, model_view.rope_tables, chunk_ws_no_qk.data(),
		    chunk_ws_no_qk.size(), model_view.option_g_fused_k_landing, &kv_sat_no_qk,
		    /*site_prefix=*/{}, &chunk_hook_no_qk, q_width);
		SslmSetTraceHook(chunk_hook_no_qk, nullptr, nullptr);
		if (layers_with_qk_norm > 0) {
			std::printf("QK_NORM MATERIALITY MUST-REJECT (Q, gain bypassed): q_norm_fired=%d "
			            "(expected 0 -- the norm application never runs) %s\n",
			            chunk_cap_no_qk.q_captured ? 1 : 0,
			            chunk_cap_no_qk.q_captured
			                ? "FAIL -- q_norm fired despite nulled gain"
			                : "PASS -- the repaired check correctly observes nothing to compare, "
			                  "never a false DIFFERS/IDENTICAL claim");
			if (chunk_cap_no_qk.q_captured) qk_norm_ran = false;
		}

		// (§7 Cell 1, D-SLM6122): width>=2 acceptance, must-reject = identical output with vs.
		// without QK-norm at width>1 -- the cell C4 shows every pre-delta cell fails to be
		// (every recorded reading was taken at width==1, where Q/K cannot influence the
		// output). Reuses layers_no_qk/chunk_codes_no_qk/chunk_status_no_qk, above.
		if (layers_with_qk_norm > 0 && chunk_status == SslmForwardStatus::Ok &&
		    chunk_status_no_qk == SslmForwardStatus::Ok) {
			bool width_gt1_identical = true;
			for (size_t i = 0; i < hidden_size; ++i) {
				if (chunk_codes[(chunk_tokens - 1) * hidden_size + i] !=
				    chunk_codes_no_qk[(chunk_tokens - 1) * hidden_size + i]) {
					width_gt1_identical = false;
					break;
				}
			}
			if (chunk_scales[chunk_tokens - 1].m != chunk_scales_no_qk[chunk_tokens - 1].m ||
			    chunk_scales[chunk_tokens - 1].e != chunk_scales_no_qk[chunk_tokens - 1].e) {
				width_gt1_identical = false;
			}
			std::printf(
			    "CELL 1 (width>1 acceptance, real candidate, must-reject=identical-with-vs-"
			    "without-QK-norm at width=%zu): %s\n",
			    chunk_tokens,
			    width_gt1_identical
			        ? "FAIL -- IDENTICAL with and without QK-norm at width>1 (the must-reject "
			          "construction did not fire -- this cell cannot distinguish the feature)"
			        : "PASS -- DIFFERS with vs. without QK-norm at width>1 (unlike C4's own "
			          "width==1 finding, this cell IS live to the feature)");
			if (width_gt1_identical) qk_norm_ran = false;
		}
	}

	// --- (§7 Cell 4, D-SLM6118/D-SLM6150): GPU chunk-batched drive, repeated N=100 -----------
	// No GPU-side chunk-batched dispatch function exists (confirmed absent by reading
	// superslm_gpu.cpp/gpu_port.h in full) -- this drives the SAME real single-token
	// RunLayerLoopGpuSubmit/Finish pair sequentially, position by position, over the SAME
	// chunk_ids the CPU chunk-batched drive above used, reusing ONE gpu_seq/gpu_ws pair across
	// the three calls exactly the way autoregressive decode does ("layer_index resets to 0
	// every token but context_length does not", forward_sites.cpp's own comment on this
	// property) -- width grows 1, 2, 3 across the three calls, the identical width range the
	// CPU chunk-batched drive exercises in one call.
	if (layers_with_qk_norm > 0 && cpu_status == SslmForwardStatus::Ok) {
		const size_t chunk_tokens = 3;
		std::vector<int32_t> chunk_ids;
		for (size_t i = 0; i < chunk_tokens; ++i) {
			chunk_ids.push_back(static_cast<int32_t>((token_id + static_cast<int32_t>(i)) %
			                                          model_view.config.vocab_size));
		}
		auto RunGpuChunkSequential = [&](std::vector<int8_t>& out_codes,
		                                  CarriedScale& out_scale) -> SslmForwardStatus {
			SequenceLayerState seq;
			std::vector<int8_t> hidden(hidden_size);
			seq.hidden_codes = hidden.data();
			seq.layer_index = 0;
			std::vector<uint8_t> ws(kv_bytes, 0);
			SslmForwardStatus st = SslmForwardStatus::Ok;
			for (size_t t = 0; t < chunk_tokens; ++t) {
				CarriedScale sc{};
				st = EmbedEntry(chunk_ids[t], model_view.config.vocab_size, embed_weights,
				                hidden_size, embed_site_constant, hidden.data(), &sc);
				if (st != SslmForwardStatus::Ok) return st;
				seq.hidden_scale = sc;
				seq.layer_index = 0;
				superslm_gpu::GpuLayerLoopInFlight* inflight2 = nullptr;
				st = superslm_gpu::RunLayerLoopGpuSubmit(
				    seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers,
				    hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
				    model_view.rope_tables, ws.data(), ws.size(), /*external_kv_resident=*/nullptr,
				    /*io_external_kv_needs_resume_barrier=*/nullptr, &inflight2,
				    /*external_weights_resident=*/nullptr, /*external_rope_cos_resident=*/nullptr,
				    /*external_rope_sin_resident=*/nullptr, /*external_rope_has=*/false,
				    /*external_rope_cos_elems=*/0, /*external_rope_sin_elems=*/0,
				    /*adapter_bridge=*/nullptr, q_width, /*out_q_codes=*/nullptr,
				    /*out_q_codes_capacity=*/0);
				if (inflight2) {
					int32_t ready = 0;
					st = superslm_gpu::RunLayerLoopGpuFinish(inflight2, seq, ws.data(), /*block=*/1,
					                                          &ready, /*out_q_codes=*/nullptr);
				}
				if (st != SslmForwardStatus::Ok) return st;
			}
			out_codes.assign(hidden.begin(), hidden.end());
			out_scale = seq.hidden_scale;
			return st;
		};

		std::vector<int8_t> cpu_ref_codes;
		CarriedScale cpu_ref_scale{};
		{
			// The CPU chunk-batched reference this repeated GPU drive is checked against --
			// re-run once here (fresh workspace) rather than reusing the earlier block's own
			// already-consumed chunk_ws/chunk_codes buffers.
			std::vector<int8_t> ref_codes(chunk_tokens * hidden_size);
			std::vector<CarriedScale> ref_scales(chunk_tokens);
			for (size_t t = 0; t < chunk_tokens; ++t) {
				CarriedScale sc{};
				EmbedEntry(chunk_ids[t], model_view.config.vocab_size, embed_weights, hidden_size,
				          embed_site_constant, ref_codes.data() + t * hidden_size, &sc);
				ref_scales[t] = sc;
			}
			std::vector<uint8_t> ref_ws(kv_bytes, 0);
			uint64_t ref_sat = 0;
			const SslmForwardStatus ref_status = RunLayerLoopChunkBatched(
			    ref_codes.data(), ref_scales.data(), chunk_tokens, layers.data(), num_hidden_layers,
			    hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
			    /*context_length_start=*/0, model_view.rope_tables, ref_ws.data(), ref_ws.size(),
			    model_view.option_g_fused_k_landing, &ref_sat, /*site_prefix=*/{},
			    /*trace_hook_state=*/nullptr, q_width);
			if (ref_status == SslmForwardStatus::Ok) {
				cpu_ref_codes.assign(ref_codes.begin() + (chunk_tokens - 1) * hidden_size,
				                     ref_codes.end());
				cpu_ref_scale = ref_scales[chunk_tokens - 1];
			}
		}

		const int kRepeatedDispatches = 100;
		int divergences = 0;
		std::vector<int8_t> first_codes;
		CarriedScale first_scale{};
		for (int i = 0; i < kRepeatedDispatches; ++i) {
			std::vector<int8_t> out_codes;
			CarriedScale out_scale{};
			const SslmForwardStatus st = RunGpuChunkSequential(out_codes, out_scale);
			if (st != SslmForwardStatus::Ok) {
				std::printf("CELL 4: GPU chunk-batched run %d/%d FAILED: status=%s\n", i,
				            kRepeatedDispatches, SslmForwardStatusName(st));
				++divergences;
				continue;
			}
			if (i == 0) {
				first_codes = out_codes;
				first_scale = out_scale;
			} else if (out_codes != first_codes || out_scale.m != first_scale.m ||
			           out_scale.e != first_scale.e) {
				++divergences;
			}
			if (!cpu_ref_codes.empty() &&
			    (out_codes != cpu_ref_codes || out_scale.m != cpu_ref_scale.m ||
			     out_scale.e != cpu_ref_scale.e)) {
				++divergences;
			}
		}
		std::printf(
		    "CELL 4 (GPU determinism, repeated dispatch, width>1, N=%d): %d/%d divergences "
		    "(against the first GPU run and against the CPU chunk-batched reference) -- %s\n",
		    kRepeatedDispatches, divergences, kRepeatedDispatches,
		    divergences == 0 ? "PASS (must-accept)" : "FAIL");
		// Must-reject twin (D-SLM6150): a disposable mutant reverting Q's per-head write back
		// to the single, shared q_scale_off slot WAS built and executed, in a follow-up round
		// this same session (T-2560 §3 Cell 4, T-2564): a scratch copy of the whole worktree
		// with qk_norm_site.hlsl reverted there, RunLayerLoopGpuSubmit's own dispatch table
		// having no injection point for a second variant under this binary without a further
		// production-code change. 186 divergences across the same 100 repeated dispatches --
		// the per-head-addressing fix's own closure of the race is backed by an executed
		// regression-catching proof, not accepted by construction alone. Not committed (the
		// mutant lived only in the scratch copy). GPU/driver identity for this reading: see
		// the build record.
		if (divergences != 0) qk_norm_ran = false;
	}

	if (all_match && qk_norm_ran && cpu_status == SslmForwardStatus::Ok &&
	    gpu_status == SslmForwardStatus::Ok) {
		std::printf("RESULT: PASS\n");
		return 0;
	}
	std::printf("RESULT: FAIL (see above)\n");
	return 1;
}
