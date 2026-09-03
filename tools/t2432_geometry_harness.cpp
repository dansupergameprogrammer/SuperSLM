// T-2432 Track A acceptance harness. Disposable, matching tools/t2039_c5_harness.cpp's own
// precedent ("Not part of the build.bat/CMake build graph -- compiled and run directly for
// this session's own verification"). Loads a real .sslm artifact, marshals it, embeds one
// token, and runs it through every layer on BOTH the CPU oracle (production RunLayerLoop) and
// the GPU port (RunLayerLoopGpuSubmit/Finish) from an IDENTICAL initial SequenceLayerState and
// workspace -- q_width threaded explicitly through both calls (design's own new parameter,
// forward_sites.h/gpu_port.h).
//
// T-2441 (Poirot 327ee29-t2438-ask5-tracka-review.md, Significant 1, D-SLM5434/D-SLM5435):
// this harness originally ran ONE token at position 0 and compared only the END-TO-END output
// (final hidden_codes, K/V cache) -- attention's softmax over a width-1 row is probability 1
// regardless of scores, so the context vector is exactly V independent of Q, making the
// comparison structurally incapable of detecting any defect in q_proj's own computed values
// (GS-14 through GS-17). Demonstrated by construction: with GS-14's own fix reverted
// (`plan.out_channels = hidden_size;` in place of the `q_width`-aware line), the harness
// reported PASS, byte-identical to the fixed build.
//
// TWO remedies were tried, in order, and this file's own history is left in place rather than
// silently deleted (StandardsDocument.md Sec6.6/Sec7): the record of what did not work is
// itself worth keeping.
//
//   1. A first attempt ran TWO tokens (later two DISTINCT ids, once same-id was found to
//      leave V identical at both positions -- RoPE rotates Q/K but never V, so a repeated id
//      makes the weighted-average context insensitive to Q for the identical reason the
//      width-1 case is). Executed: even with two distinct tokens and a genuinely non-degenerate
//      softmax, the GS-14 mutant STILL PASSED against a non-GQA (MHA) non-square fixture --
//      because the mutation leaves the missing channels' Q at exactly ZERO (uncomputed
//      GPU-resident memory, D3D12 zero-initialized), and a zero Q dot-products to a CONSTANT
//      zero score against every key regardless of position, so attention degenerates to a
//      UNIFORM average over however many tokens exist -- "make the softmax non-degenerate" does
//      not by itself make an all-zero-Q defect detectable, because the defect's OWN failure mode
//      is a second kind of degeneracy end-to-end propagation cannot see through. (Separately,
//      the same construction on a GQA fixture (num_key_value_heads < num_attention_heads) DID
//      diverge, by a small amount -- inconclusive whether that reflects a real, independent GQA-
//      specific residual or fixture-specific quantization luck; not run down further, named here
//      as an open question for whoever next touches GQA + non-square + multi-token GPU decoding,
//      not fixed or claimed by this ticket.)
//   2. What actually closes it, landed here: a DIRECT readback of the GPU's own intermediate
//      q_codes LayerScratch region (RunLayerLoopGpuSubmit/Finish's new optional `out_q_codes`
//      parameter, T-2441), compared byte-for-byte against the CPU's own q_codes (captured via
//      the trace hook below) -- exactly the design's own first-stated preference (§6 Track A:
//      "a GPU parity check specifically asserts every one of q_proj's q_width output channels
//      is populated and correct"). This does not depend on attention, o_proj, or the MLP at
//      all, so it cannot be defeated by a defect whose downstream effect happens to cancel out.
//
// Checks:
//   1. CPU q_proj.requant trace hook (VALUES, not just width) vs GPU's direct q_codes readback
//      -- byte-for-byte, the primary defect-detecting check (closes GS-14 through GS-17).
//   2. The pre-existing end-to-end CPU-vs-GPU bit-identity comparison (hidden_codes, K/V cache)
//      at width-1 -- unaffected by the above, kept as a broader regression check.
//
// Usage: t2432_geometry_harness <model.sslm> [token_id]
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
using superslm_marshal::WidenGainToInt32;

namespace {
struct QProjRowCapture {
	bool captured = false;
	std::vector<int8_t> codes;
};

// CORRECTED 2026-08-31 (T-2445, Claude/Poirot/ddbc57a-t2443-ask5-tracka-confirmation.md,
// Significant 1): this hook used to keep only its FIRST fire (`if (!cap->captured)`), which
// holds layer 0's row, while the GPU side below reads `q_codes` from `LayerScratch` AFTER the
// whole dispatch chain -- `RunLayerLoopGpuSubmit`'s own per-layer loop overwrites that region
// every layer, so it holds the LAST layer's row. At `num_hidden_layers == 1` the two coincide;
// above it they do not, and the check read `Q_CODES CHECK: FAILED` on an unmodified tree the
// moment a real (multi-layer) artifact was run. Fixed by keeping the LAST fire instead of the
// first -- both sides now agree on which layer they compare, independent of layer count, and
// the harness's own two calls below already run every layer (`layer_budget=num_hidden_layers`)
// on both paths, so no other change is needed to exercise this.
void QProjRowHook(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord* kv, void* user) {
	(void)kv;
	if (chain == nullptr) return;
	QProjRowCapture* cap = static_cast<QProjRowCapture*>(user);
	// LayerSite's own convention: "layer{L}.q_proj.requant" -- match the SUFFIX. Fires once per
	// layer; the LAST fire (the highest layer index run) is what is kept, matching the GPU
	// side's own LayerScratch readback below.
	if (chain->site.size() >= 14 && chain->site.substr(chain->site.size() - 14) == "q_proj.requant") {
		cap->captured = true;
		cap->codes.assign(chain->codes.begin(), chain->codes.end());
	}
}
}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [token_id]\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const int32_t token_id = argc >= 3 ? std::atoi(argv[2]) : 0;
	// (T-2577 round 2, D-SLM6278): this process loads exactly ONE model, once -- a constant
	// caller-owned identity, so the pre-1.0 residency caches (g_resident_weights/g_resident_kv/
	// g_resident_rope, superslm_gpu.cpp) treat every fresh sequence of it as a legitimate hit
	// rather than paying the repack+reupload cost every unwired caller (model_generation=0)
	// still pays.
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
	    "(square=%d) intermediate=%zu vocab=%u context_cap=%lld\n",
	    hidden_size, num_hidden_layers, num_heads, num_kv_heads, head_dim, q_width,
	    q_width == hidden_size ? 1 : 0, intermediate_size, model_view.config.vocab_size,
	    (long long)context_cap);

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

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	// --- CPU oracle: production RunLayerLoop, layer_budget = all layers, q_width explicit,
	//     a trace hook installed to capture q_proj.requant's own output row VALUES. ---
	std::vector<int8_t> cpu_codes(hidden_size);
	std::memcpy(cpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState cpu_seq;
	cpu_seq.hidden_codes = cpu_codes.data();
	cpu_seq.hidden_scale = embed_scale;
	cpu_seq.layer_index = 0;
	std::vector<uint8_t> cpu_ws(kv_bytes, 0);
	QProjRowCapture cap;
	SslmTraceHookState hook_state;
	SslmSetTraceHook(hook_state, &QProjRowHook, &cap);
	const SslmForwardStatus cpu_status = RunLayerLoop(
	    cpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, cpu_ws.data(),
	    cpu_ws.size(), /*site_prefix=*/{}, /*token_index=*/0, &hook_state, q_width);
	SslmSetTraceHook(hook_state, nullptr, nullptr);
	std::printf("CPU oracle: status=%s layer_index=%u\n", SslmForwardStatusName(cpu_status),
	            cpu_seq.layer_index);

	bool geometry_row_ok = true;
	if (!cap.captured) {
		std::printf("GEOMETRY CHECK: FAILED -- q_proj.requant trace hook never fired\n");
		geometry_row_ok = false;
	} else if (cap.codes.size() != q_width) {
		std::printf("GEOMETRY CHECK: FAILED -- q_proj.requant output row width=%zu, want q_width=%zu\n",
		            cap.codes.size(), q_width);
		geometry_row_ok = false;
	} else {
		std::printf("GEOMETRY CHECK: PASS -- q_proj.requant output row width=%zu == q_width (CPU, "
		            "last layer run, token 0)\n",
		            cap.codes.size());
	}

	// --- GPU port: RunLayerLoopGpuSubmit/Finish, IDENTICAL inputs, q_width explicit, PLUS a
	//     direct readback of the GPU's own q_codes LayerScratch region (T-2441's new optional
	//     out_q_codes/out_q_codes_capacity parameters). ---
	std::vector<int8_t> gpu_codes(hidden_size);
	std::memcpy(gpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState gpu_seq;
	gpu_seq.hidden_codes = gpu_codes.data();
	gpu_seq.hidden_scale = embed_scale;
	gpu_seq.layer_index = 0;
	std::vector<uint8_t> gpu_ws(kv_bytes, 0);
	std::vector<uint8_t> gpu_q_codes(q_width, 0xEE);  // poison value: FAILED-to-fire is visible
	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const SslmForwardStatus gpu_submit_status = superslm_gpu::RunLayerLoopGpuSubmit(
	    gpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, gpu_ws.data(),
	    gpu_ws.size(), /*external_kv_resident=*/nullptr, /*io_external_kv_needs_resume_barrier=*/nullptr,
	    &inflight, /*external_weights_resident=*/nullptr, /*external_rope_cos_resident=*/nullptr,
	    /*external_rope_sin_resident=*/nullptr, /*external_rope_has=*/false,
	    /*external_rope_cos_elems=*/0, /*external_rope_sin_elems=*/0, /*adapter_bridge=*/nullptr,
	    q_width, gpu_q_codes.data(), gpu_q_codes.size(), kModelGeneration);
	SslmForwardStatus gpu_status = gpu_submit_status;
	if (inflight) {
		int32_t ready = 0;
		gpu_status = superslm_gpu::RunLayerLoopGpuFinish(inflight, gpu_seq, gpu_ws.data(), /*block=*/1,
		                                                  &ready, gpu_q_codes.data());
	}
	std::printf("GPU port:   status=%s layer_index=%u\n", SslmForwardStatusName(gpu_status),
	            gpu_seq.layer_index);

	// --- Primary check (T-2441, Significant 1): CPU's own q_codes (captured by the trace hook)
	//     vs GPU's own q_codes (read back directly from LayerScratch), byte-for-byte. This is
	//     the design's own preferred assertion -- direct, not inferred from end-to-end
	//     propagation -- and it is what actually closes GS-14 through GS-17. ---
	bool q_codes_match = true;
	if (cpu_status == SslmForwardStatus::Ok && gpu_status == SslmForwardStatus::Ok) {
		if (cap.codes.size() != gpu_q_codes.size()) {
			std::printf("Q_CODES CHECK: FAILED -- CPU row width=%zu != GPU readback width=%zu\n",
			            cap.codes.size(), gpu_q_codes.size());
			q_codes_match = false;
		} else {
			int first_mismatch = -1;
			for (size_t i = 0; i < cap.codes.size(); ++i) {
				if (cap.codes[i] != static_cast<int8_t>(gpu_q_codes[i])) {
					first_mismatch = static_cast<int>(i);
					break;
				}
			}
			if (first_mismatch >= 0) {
				std::printf("Q_CODES CHECK: FAILED -- q_codes[%d]: CPU=%d GPU=%d (first mismatch of "
				            "%zu elements)\n",
				            first_mismatch, cap.codes[first_mismatch],
				            static_cast<int8_t>(gpu_q_codes[first_mismatch]), cap.codes.size());
				q_codes_match = false;
			} else {
				std::printf("Q_CODES CHECK: PASS -- CPU/GPU q_codes bit-identical across all %zu "
				            "q_width channels (direct LayerScratch readback, last layer run, token 0)\n",
				            cap.codes.size());
			}
		}
	} else {
		std::printf("Q_CODES CHECK: SKIPPED -- CPU or GPU status was not Ok\n");
		q_codes_match = false;
	}

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
	if (cpu_seq.hidden_scale.m != gpu_seq.hidden_scale.m || cpu_seq.hidden_scale.e != gpu_seq.hidden_scale.e) {
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
					std::printf("DIVERGENCE: K/V[layer=%u][kv_head=%u][dim=%zu]\n", l, h, d);
					kv_match = false;
					all_match = false;
					break;
				}
			}
		}
	}

	if (all_match && geometry_row_ok && q_codes_match) {
		std::printf("RESULT: PASS -- CPU q_proj row genuinely q_width-wide, GPU's own direct q_codes "
		            "readback bit-identical to it, and CPU/GPU bit-identical end-to-end across "
		            "hidden_codes[%zu], hidden_scale, and every K/V row (%u layers).\n",
		            hidden_size, num_hidden_layers);
		return 0;
	}
	std::printf("RESULT: FAIL (see above)\n");
	return 1;
}
