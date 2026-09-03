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
// BUILD RECIPE -- the dxc step is part of it, not an optional prerequisite (T-2575, D-SLM6268).
// `superslm_gpu::harness::ShaderPath` resolves `.cso` files relative to the EXECUTABLE's own
// directory, so this harness dispatches whatever shader binaries happen to sit in
// `<exe_dir>\shaders\`. Compiling only the .cpp leaves the previous generation's shaders in
// place and the harness runs them silently. Measured: `out\shaders\rope_guard_site.cso` was a
// pre-D-SLM6263 compile carrying no saturation counter at all, so every GPU
// `kv_saturation_count` reading this harness produced across T-2572 and T-2574 came from a dead
// counter -- which is what the "GPU loses almost every RoPE saturation event" finding and its
// quarantine actually were. `ShaderPath` now refuses a `.cso` older than its own source rather
// than dispatching it, so the recipe below is enforced rather than remembered:
//
//   1. for %f in (src\gpu\shaders\*.hlsl) do dxc -T cs_6_2 -E main -Fo out\shaders\%~nf.cso %f
//        -O3 -HV 2018 -WX            (build.bat's own loop -- run it, or run build.bat)
//   2. cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools
//        src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp
//        src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp
//        src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp
//        src\decode_digest.cpp src\gpu\superslm_gpu.cpp tests\t2551_qk_norm_harness.cpp
//        /Fo:out\harness\ /Fe:out\t2551_qk_norm_harness.exe /link d3d12.lib dxgi.lib dxguid.lib
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

	// (T-2577, D-SLM6278): this process loads exactly ONE model, once -- every GPU call below
	// that drives this SAME loaded candidate passes this SAME identity, so the residency caches
	// (g_resident_weights/g_resident_kv/g_resident_rope, superslm_gpu.cpp) treat every fresh
	// sequence of it as a legitimate hit rather than paying T-2576's own `!fresh_sequence`-gated
	// repack+reupload cost every time. CELL 5 (below) deliberately uses a DIFFERENT generation
	// for its own mutated-tables arm, simulating a different model now occupying the identical
	// host address.
	constexpr uint64_t kModelGeneration = 1;

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
	    q_width, /*out_q_codes=*/nullptr, /*out_q_codes_capacity=*/0, kModelGeneration);
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
	// (D-SLM6263, external review Significant 1, closed T-2572 D-SLM6264):
	// kv_saturation_count now also carries RopeApplySite's own clamp count (both engines)
	// alongside the pre-existing K/V and QK-norm landing counts -- compared here so the
	// determinism crown covers the new counter on the real, 28-layer candidate, not only
	// the K/V store bytes above.
	if (cpu_seq.kv_saturation_count != gpu_seq.kv_saturation_count) {
		std::printf("DIVERGENCE: kv_saturation_count: CPU=%llu GPU=%llu\n",
		            (unsigned long long)cpu_seq.kv_saturation_count,
		            (unsigned long long)gpu_seq.kv_saturation_count);
		all_match = false;
	}
	std::printf("kv_saturation_count (single-token path): CPU=%llu GPU=%llu\n",
	            (unsigned long long)cpu_seq.kv_saturation_count,
	            (unsigned long long)gpu_seq.kv_saturation_count);

	// (T-2577, D-SLM6280): the per-site breakdown, single-token path -- GPU-equals-CPU per
	// site, not only in the aggregate above.
	{
		struct SiteReading { const char* name; uint64_t cpu; uint64_t gpu; };
		const SiteReading sites[] = {
		    {"kv_landing", cpu_seq.kv_landing_saturation_count, gpu_seq.kv_landing_saturation_count},
		    {"k_normed_landing", cpu_seq.k_normed_landing_saturation_count,
		     gpu_seq.k_normed_landing_saturation_count},
		    {"rope_q", cpu_seq.rope_q_saturation_count, gpu_seq.rope_q_saturation_count},
		    {"rope_k", cpu_seq.rope_k_saturation_count, gpu_seq.rope_k_saturation_count},
		};
		bool per_site_match = true;
		for (const auto& s : sites) {
			std::printf("kv_saturation_count per-site (single-token path) %s: CPU=%llu GPU=%llu%s\n",
			            s.name, (unsigned long long)s.cpu, (unsigned long long)s.gpu,
			            s.cpu == s.gpu ? "" : " DIVERGENCE");
			if (s.cpu != s.gpu) per_site_match = false;
		}
		if (!per_site_match) all_match = false;
	}
	if (all_match) {
		std::printf("DETERMINISM CROWN: PASS -- CPU/GPU bit-identical end-to-end across "
		            "hidden_codes[%zu], hidden_scale, every K/V row, and kv_saturation_count "
		            "(%u layers, single-token path).\n", hidden_size, num_hidden_layers);
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
		auto RunGpuChunkSequential = [&](std::vector<int8_t>& out_codes, CarriedScale& out_scale,
		                                  uint64_t& out_sat, uint64_t* out_rope_k,
		                                  int* out_peak_k_code) -> SslmForwardStatus {
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
				    /*out_q_codes_capacity=*/0, kModelGeneration);
				if (inflight2) {
					int32_t ready = 0;
					st = superslm_gpu::RunLayerLoopGpuFinish(inflight2, seq, ws.data(), /*block=*/1,
					                                          &ready, /*out_q_codes=*/nullptr);
				}
				if (st != SslmForwardStatus::Ok) return st;
			}
			out_codes.assign(hidden.begin(), hidden.end());
			out_scale = seq.hidden_scale;
			out_sat = seq.kv_saturation_count;
			if (out_rope_k != nullptr) *out_rope_k = seq.rope_k_saturation_count;
			// (T-2577, D-SLM6280): O2's own zero-margin observation, answered with the margin
			// the integer path actually has -- the peak |K store byte| this real drive reached,
			// against the [-127, 127] pinned code range every landing site clamps to.
			if (out_peak_k_code != nullptr) {
				int peak = 0;
				for (uint32_t l = 0; l < num_hidden_layers; ++l) {
					for (uint32_t h = 0; h < num_kv_heads; ++h) {
						for (int64_t pos = 0; pos < static_cast<int64_t>(chunk_tokens); ++pos) {
							const int8_t* k_row =
							    KeyRow(ws.data(), l, context_cap, num_kv_heads, head_dim, h, pos);
							for (size_t d = 0; d < head_dim; ++d) {
								const int mag = k_row[d] < 0 ? -static_cast<int>(k_row[d]) : k_row[d];
								if (mag > peak) peak = mag;
							}
						}
					}
				}
				*out_peak_k_code = peak;
			}
			return st;
		};

		std::vector<int8_t> cpu_ref_codes;
		CarriedScale cpu_ref_scale{};
		// (D-SLM6263): the CPU chunk-batched reference's own kv_saturation_count -- Cell 4's
		// repeated GPU drive (below) is checked against this too, alongside the codes/scale
		// it already checks, so RopeApplySite's own new counter is covered by the SAME N=100
		// repeated-dispatch determinism proof the codes/scale already have.
		uint64_t cpu_ref_sat = 0;
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
				cpu_ref_sat = ref_sat;
			}
		}

		const int kRepeatedDispatches = 100;
		int divergences = 0;
		int sat_divergences = 0;
		int rope_k_nonzero_runs = 0;
		uint64_t rope_k_max_observed = 0;
		int peak_k_code_observed = 0;
		std::vector<int8_t> first_codes;
		CarriedScale first_scale{};
		uint64_t first_sat = 0;
		for (int i = 0; i < kRepeatedDispatches; ++i) {
			std::vector<int8_t> out_codes;
			CarriedScale out_scale{};
			uint64_t out_sat = 0;
			uint64_t out_rope_k = 0;
			int out_peak_k_code = 0;
			const SslmForwardStatus st =
			    RunGpuChunkSequential(out_codes, out_scale, out_sat, &out_rope_k, &out_peak_k_code);
			if (st != SslmForwardStatus::Ok) {
				std::printf("CELL 4: GPU chunk-batched run %d/%d FAILED: status=%s\n", i,
				            kRepeatedDispatches, SslmForwardStatusName(st));
				++divergences;
				continue;
			}
			if (i == 0) {
				first_codes = out_codes;
				first_scale = out_scale;
				first_sat = out_sat;
			} else if (out_codes != first_codes || out_scale.m != first_scale.m ||
			           out_scale.e != first_scale.e) {
				++divergences;
			}
			if (out_sat != first_sat) ++sat_divergences;
			if (!cpu_ref_codes.empty() &&
			    (out_codes != cpu_ref_codes || out_scale.m != cpu_ref_scale.m ||
			     out_scale.e != cpu_ref_scale.e)) {
				++divergences;
			}
			if (out_sat != cpu_ref_sat) ++sat_divergences;
			if (out_rope_k != 0) ++rope_k_nonzero_runs;
			if (out_rope_k > rope_k_max_observed) rope_k_max_observed = out_rope_k;
			if (out_peak_k_code > peak_k_code_observed) peak_k_code_observed = out_peak_k_code;
		}
		// (T-2577, D-SLM6280, external review Significant 1's own required closure item 2 and
		// O2): the enclosure proof this ticket owes -- RoPE's own K clamp (rope_k) stays zero
		// across every repeated dispatch at width > 1 on the recalibrated candidate, and the peak
		// integer K code actually reached is reported against the pinned 127 clamp boundary, so
		// O2's own "zero margin in the float domain" reading is answered with the margin the
		// INTEGER path actually has (never negative here means the union calibration's own
		// enclosure holds through RoPE's rotation at this candidate's own real geometry).
		std::printf(
		    "CELL 4 rope_k enclosure (GPU, repeated dispatch, width>1, N=%d): %d/%d runs with "
		    "rope_k != 0 (max observed %llu), peak |K store code| observed = %d / 127 (integer-"
		    "path margin = %d) -- %s\n",
		    kRepeatedDispatches, rope_k_nonzero_runs, kRepeatedDispatches,
		    (unsigned long long)rope_k_max_observed, peak_k_code_observed, 127 - peak_k_code_observed,
		    rope_k_nonzero_runs == 0 ? "PASS -- rope_k == 0 at width > 1, the enclosure holds"
		                             : "FAIL -- RoPE's own K rotation clamped at least once");
		if (rope_k_nonzero_runs != 0) qk_norm_ran = false;
		// (D-SLM6263): kv_saturation_count's own N=100 determinism, reported alongside Cell
		// 4's own codes/scale divergence count -- the identical must-accept shape, over the
		// SAME 100 repeated dispatches, now also covering RopeApplySite's new counter.
		std::printf(
		    "CELL 4 kv_saturation_count (GPU determinism, repeated dispatch, width>1, N=%d): "
		    "%d/%d divergences (against the first GPU run and against the CPU chunk-batched "
		    "reference, cpu_ref_sat=%llu) -- %s\n",
		    kRepeatedDispatches, sat_divergences, kRepeatedDispatches * 2,
		    (unsigned long long)cpu_ref_sat, sat_divergences == 0 ? "PASS (must-accept)" : "FAIL");
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

	// --- (T-2576): CELL 5 -- is the RoPE-table residency cache's stale hit reachable at
	//     PRODUCTION geometry? Settled by construction, not by inference.
	//
	// `g_resident_rope` (superslm_gpu.cpp) is a process-global keyed on the source tensor's own
	// host ADDRESS and byte count. Nothing in that key is geometry-dependent, so "the degenerate
	// fixture broke and the real candidate did not" says only that the degenerate fixture's heap
	// addresses collided and the real candidate's did not. This cell removes the allocator from
	// the question: one full sequence populates the cache, then the cos/sin tables are rewritten
	// IN PLACE -- same address, same byte count, different content -- and a second sequence runs
	// on the mutated model. A caller reusing its own buffer for a second model is the production
	// shape of exactly that.
	//
	// Two properties this cell needs, both learned by executing it wrong first:
	//   - The rewrite must be a 45-degree rotation, NOT the identity. The real table's own row 0
	//     IS the identity, so an identity rewrite changes nothing and the cell passes vacuously
	//     (measured: the mutated CPU arm reproduced the pristine scale 1083582878/-28 exactly).
	//   - The forward must run at WIDTH > 1. At width 1 the softmax is over a single position and
	//     returns 1.0 whatever the score is, so Q/K rotations cannot reach the output at all --
	//     the same blindness C4 found in every pre-delta reading. Measured: at width 1 the 45-degree
	//     rewrite ALSO reproduced the pristine output, and the cell was still vacuous.
	// The must-reject below is what makes both of those visible instead of green.
	{
		const SslmTensorView* cos_t = model_view.rope_tables.Tensor("cos");
		const SslmTensorView* sin_t = model_view.rope_tables.Tensor("sin");
		if (cos_t == nullptr || sin_t == nullptr) {
			std::printf("CELL 5 (RoPE-table residency, production geometry): SKIPPED -- no cos/sin\n");
		} else {
			const size_t chunk_tokens = 3;
			std::vector<int32_t> ids;
			for (size_t i = 0; i < chunk_tokens; ++i) {
				ids.push_back(static_cast<int32_t>((token_id + static_cast<int32_t>(i)) %
				                                   model_view.config.vocab_size));
			}
			// One CPU chunk-batched drive over whichever tables are live when it is called.
			auto run_cpu_chunk = [&](std::vector<int8_t>& out_codes, CarriedScale& out_scale) {
				std::vector<int8_t> codes(chunk_tokens * hidden_size);
				std::vector<CarriedScale> scales(chunk_tokens);
				for (size_t t = 0; t < chunk_tokens; ++t) {
					CarriedScale sc{};
					EmbedEntry(ids[t], model_view.config.vocab_size, embed_weights, hidden_size,
					           embed_site_constant, codes.data() + t * hidden_size, &sc);
					scales[t] = sc;
				}
				std::vector<uint8_t> ws(kv_bytes, 0);
				uint64_t sat = 0;
				const SslmForwardStatus st = RunLayerLoopChunkBatched(
				    codes.data(), scales.data(), chunk_tokens, layers.data(), num_hidden_layers,
				    hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
				    /*context_length_start=*/0, model_view.rope_tables, ws.data(), ws.size(),
				    model_view.option_g_fused_k_landing, &sat, /*site_prefix=*/{},
				    /*trace_hook_state=*/nullptr, q_width);
				out_codes.assign(codes.begin() + (chunk_tokens - 1) * hidden_size, codes.end());
				out_scale = scales[chunk_tokens - 1];
				return st;
			};
			// The same three tokens driven sequentially on the GPU, one fresh sequence.
			// (T-2577, D-SLM6278): `generation` and `out_first_call_rope_hit` let this lambda
			// serve BOTH S1 cells this section now covers -- the recycled-address must-reject
			// (T-2576's own construction, generation bumped between `warm` and `mut_gpu` below)
			// and the "a second fresh sequence of the same model hits" must-accept, read directly
			// off `LastRopeUploadWasSkipped()` on this call's own FIRST (fresh-sequence) submit,
			// before `context_length` advances off zero.
			auto run_gpu_chunk = [&](std::vector<int8_t>& out_codes, CarriedScale& out_scale,
			                          uint64_t generation, bool* out_first_call_rope_hit) {
				SequenceLayerState seq;
				std::vector<int8_t> hidden(hidden_size);
				seq.hidden_codes = hidden.data();
				seq.layer_index = 0;
				std::vector<uint8_t> ws(kv_bytes, 0);
				SslmForwardStatus st = SslmForwardStatus::Ok;
				for (size_t t = 0; t < chunk_tokens; ++t) {
					CarriedScale sc{};
					st = EmbedEntry(ids[t], model_view.config.vocab_size, embed_weights, hidden_size,
					                embed_site_constant, hidden.data(), &sc);
					if (st != SslmForwardStatus::Ok) return st;
					seq.hidden_scale = sc;
					seq.layer_index = 0;
					superslm_gpu::GpuLayerLoopInFlight* infl = nullptr;
					st = superslm_gpu::RunLayerLoopGpuSubmit(
					    seq, layers.data(), num_hidden_layers, num_hidden_layers, hidden_size, head_dim,
					    num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, ws.data(),
					    ws.size(), nullptr, nullptr, &infl, nullptr, nullptr, nullptr, false, 0, 0,
					    nullptr, q_width, nullptr, 0, generation);
					if (t == 0 && out_first_call_rope_hit != nullptr) {
						*out_first_call_rope_hit = superslm_gpu::LastRopeUploadWasSkipped();
					}
					if (infl) {
						int32_t ready = 0;
						st = superslm_gpu::RunLayerLoopGpuFinish(infl, seq, ws.data(), 1, &ready, nullptr);
					}
					if (st != SslmForwardStatus::Ok) return st;
				}
				out_codes.assign(hidden.begin(), hidden.end());
				out_scale = seq.hidden_scale;
				return st;
			};

			const size_t cos_bytes = static_cast<size_t>(cos_t->elem_count) * 8u;
			const size_t sin_bytes = static_cast<size_t>(sin_t->elem_count) * 8u;
			uint8_t* cos_mut = const_cast<uint8_t*>(cos_t->data);
			uint8_t* sin_mut = const_cast<uint8_t*>(sin_t->data);
			std::vector<uint8_t> cos_orig(cos_mut, cos_mut + cos_bytes);
			std::vector<uint8_t> sin_orig(sin_mut, sin_mut + sin_bytes);

			std::vector<int8_t> pristine_cpu;
			CarriedScale pristine_cpu_scale{};
			const SslmForwardStatus pristine_st = run_cpu_chunk(pristine_cpu, pristine_cpu_scale);
			// Populate/confirm the cache with the pristine tables, generation kModelGeneration --
			// this process has already primed that generation (the DETERMINISM CROWN drive and
			// CELL 4's own 100 repeats, above), so `warm`'s own first-call cache read is the S1
			// must-accept in the same motion: a fresh sequence of the SAME model, same
			// generation, is a legitimate hit.
			std::vector<int8_t> warm_codes;
			CarriedScale warm_scale{};
			bool warm_first_call_hit = false;
			const SslmForwardStatus warm_st =
			    run_gpu_chunk(warm_codes, warm_scale, kModelGeneration, &warm_first_call_hit);

			{
				const int64_t cos45_q30 = INT64_C(759250125);
				for (size_t b = 0; b < cos_bytes; b += 8) std::memcpy(cos_mut + b, &cos45_q30, 8);
				for (size_t b = 0; b < sin_bytes; b += 8) std::memcpy(sin_mut + b, &cos45_q30, 8);
			}

			// (T-2577, D-SLM6278): a DIFFERENT generation for the mutated arm -- simulating a
			// different model now occupying the identical host address, T-2576's own recycled-
			// address construction. The must-reject: this must still MISS (and read the mutated
			// tables), even though the address and byte count are unchanged from `warm` above.
			constexpr uint64_t kMutatedGeneration = kModelGeneration + 1;
			std::vector<int8_t> mut_cpu, mut_gpu;
			CarriedScale mut_cpu_scale{}, mut_gpu_scale{};
			bool mut_gpu_first_call_hit = true;  // default true so a missing write cannot pass silently
			const SslmForwardStatus mut_cpu_st = run_cpu_chunk(mut_cpu, mut_cpu_scale);
			const SslmForwardStatus mut_gpu_st =
			    run_gpu_chunk(mut_gpu, mut_gpu_scale, kMutatedGeneration, &mut_gpu_first_call_hit);

			// (T-2577, D-SLM6278, property 3): the cache resumes hitting once the generation
			// is held steady again -- proves the miss above was the generation mismatch, not a
			// permanent cache trip. Same (still-mutated) tables, same kMutatedGeneration.
			std::vector<int8_t> mut_gpu2;
			CarriedScale mut_gpu2_scale{};
			bool mut_gpu2_first_call_hit = false;
			const SslmForwardStatus mut_gpu2_st =
			    run_gpu_chunk(mut_gpu2, mut_gpu2_scale, kMutatedGeneration, &mut_gpu2_first_call_hit);

			std::memcpy(cos_mut, cos_orig.data(), cos_bytes);
			std::memcpy(sin_mut, sin_orig.data(), sin_bytes);

			if (pristine_st != SslmForwardStatus::Ok || warm_st != SslmForwardStatus::Ok ||
			    mut_cpu_st != SslmForwardStatus::Ok || mut_gpu_st != SslmForwardStatus::Ok ||
			    mut_gpu2_st != SslmForwardStatus::Ok) {
				std::printf("CELL 5 (RoPE-table residency, production geometry): INCONCLUSIVE -- "
				            "a status was not Ok (pristine=%s warm=%s mut_cpu=%s mut_gpu=%s "
				            "mut_gpu2=%s); this cell says nothing about the cache unless every arm "
				            "returns Ok\n",
				            SslmForwardStatusName(pristine_st), SslmForwardStatusName(warm_st),
				            SslmForwardStatusName(mut_cpu_st), SslmForwardStatusName(mut_gpu_st),
				            SslmForwardStatusName(mut_gpu2_st));
				qk_norm_ran = false;
			} else {
				const bool mutation_is_live =
				    mut_cpu != pristine_cpu || mut_cpu_scale.m != pristine_cpu_scale.m ||
				    mut_cpu_scale.e != pristine_cpu_scale.e;
				const bool gpu_followed = mut_gpu == mut_cpu && mut_gpu_scale.m == mut_cpu_scale.m &&
				                          mut_gpu_scale.e == mut_cpu_scale.e;
				if (!mutation_is_live) {
					std::printf("CELL 5 (RoPE-table residency, production geometry): INCONCLUSIVE -- "
					            "the rewritten tables produce the CPU's own pristine output, so this "
					            "cell cannot tell a stale GPU hit from a live one\n");
					qk_norm_ran = false;
				} else {
					std::printf(
					    "CELL 5 (RoPE-table residency, production geometry: head_dim=%zu group=%u "
					    "context_cap=%lld, cos/sin %zu elems / %zu B each, width=%zu; tables rewritten "
					    "in place to a 45-degree rotation between two fresh sequences; must-reject live: "
					    "the rewrite moves the CPU arm off its pristine output): %s\n",
					    head_dim, num_heads / (num_kv_heads ? num_kv_heads : 1u), (long long)context_cap,
					    static_cast<size_t>(cos_t->elem_count), cos_bytes, chunk_tokens,
					    gpu_followed
					        ? "PASS -- the GPU read the mutated tables, so no stale hit"
					        : "FAIL -- the GPU disagrees with the CPU on the SAME mutated tables, so it "
					          "rotated with the cached pre-mutation tables (stale hit REACHABLE at "
					          "production geometry)");
					if (!gpu_followed) qk_norm_ran = false;

					// (T-2577, D-SLM6278): the three model_generation properties, on this real
					// candidate at production geometry.
					std::printf(
					    "CELL 5a model_generation must-accept (a fresh sequence of the SAME model, "
					    "same generation=%llu, is a cache hit): %s\n",
					    (unsigned long long)kModelGeneration,
					    warm_first_call_hit ? "PASS -- LastRopeUploadWasSkipped()==true"
					                        : "FAIL -- the rope table was repacked/re-uploaded for "
					                          "an unchanged, still-live model");
					if (!warm_first_call_hit) qk_norm_ran = false;
					std::printf(
					    "CELL 5b model_generation must-reject (T-2576's own recycled-address "
					    "construction, generation bumped %llu -> %llu for the mutated arm): %s\n",
					    (unsigned long long)kModelGeneration, (unsigned long long)kMutatedGeneration,
					    !mut_gpu_first_call_hit
					        ? "PASS -- LastRopeUploadWasSkipped()==false, a real miss (matches "
					          "gpu_followed reading the mutated tables above)"
					        : "FAIL -- the cache served a wrong-model hit through model_generation, "
					          "the exact hazard T-2576 closed via !fresh_sequence");
					if (mut_gpu_first_call_hit) qk_norm_ran = false;
					std::printf(
					    "CELL 5c model_generation resumes hitting (same generation=%llu as the "
					    "mutated arm, confirming the miss above was the generation mismatch, not a "
					    "permanent cache trip): %s\n",
					    (unsigned long long)kMutatedGeneration,
					    mut_gpu2_first_call_hit
					        ? "PASS -- LastRopeUploadWasSkipped()==true"
					        : "FAIL -- the cache never recovered to hitting on a steady generation");
					if (!mut_gpu2_first_call_hit) qk_norm_ran = false;
					const bool mut_gpu2_matches_mut_gpu =
					    mut_gpu2 == mut_gpu && mut_gpu2_scale.m == mut_gpu_scale.m &&
					    mut_gpu2_scale.e == mut_gpu_scale.e;
					if (!mut_gpu2_matches_mut_gpu) {
						std::printf("CELL 5c: mut_gpu2's landed output differs from mut_gpu's despite "
						            "an unmutated, cached hit on the same generation\n");
						qk_norm_ran = false;
					}
				}
			}
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
