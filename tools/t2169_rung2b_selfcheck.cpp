// T-2169 (Rung 2b, design Claude/Vitruvius/t2169-gpu-batched-prefill-design-2026-08-18.md
// Sec5/Sec8's own exit condition (b)/(b2)): the chunk-submission primitive's own self-check --
// (a) `chunk_tokens=1`: driving one token through SubmitChunkToFullDepthForG5Bridge must produce
// the identical dispatch sequence RunLayerLoopGpu already produces for that same token
// (mechanism-only per the design's own fold-round-2 disposition -- every stale-host-input class
// member is trivially current at one token); (b) a 4-token self-check: driving four tokens
// through the SAME primitive, in one call, must produce the identical post-chunk SeqState AND
// K/V cache the unbatched per-token path (four separate RunLayerLoopGpu calls) produces for the
// same four tokens -- the discriminating instance (D-SLM3608/D-SLM3615), since a size-1 check
// alone cannot see the T-2175 stale-root-constant/stale-embedding defect class.
//
// This primitive has internal linkage in its own translation unit's header sense (no exported
// declaration in gpu_port.h -- Rung 3/4 have not yet wired a public bridge caller) and is
// therefore not reachable through any 1.0 API entry point yet; this harness forward-declares it
// with the exact matching signature (extern, same namespace) to call it directly, matching this
// project's own established throwaway-harness precedent (tools/t1657_load_harness.cpp,
// tools/t2039_c5_harness.cpp: "compiled and run directly for this session's own verification").
//
// Real Qwen2.5-1.5B artifact, real per-token embeddings via the SAME EmbedEntry arithmetic
// production uses (StandardsDocument.md Sec5.4: no synthetic-only fixture stands in for the
// product claim). Both arms start from an identical freshly-zeroed workspace and context_length
// 0; the reference arm's own four RunLayerLoopGpu calls are what the shipped
// DriveGpuSeqToFullDepthForG5Bridge loop performs at layer_budget=num_hidden_layers (one
// round-trip per token, not per whole-layer-quantum -- the fixture's own choice of the widest
// dispatch_budget, matching this cell's "same dispatch sequence" claim exactly, not the narrower
// per-layer-quantum stepping the real bridge functions use before Rung 3/4 lands).
//
// Usage: t2169_rung2b_selfcheck <model.sslm>
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

namespace superslm_gpu {
struct GpuLayerLoopInFlight;  // opaque, matches gpu_port.h's own forward declaration

// Exact signature match to the real definition in src/gpu/superslm_gpu.cpp (T-2169 Rung 2b) --
// forward-declared here so this harness can call it directly, ahead of Rung 3/4's own public
// bridge wiring.
superslm::SslmForwardStatus SubmitChunkToFullDepthForG5Bridge(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    uint8_t* workspace, size_t workspace_size, const uint8_t* chunk_embedding_bytes,
    uint32_t chunk_len, ID3D12Resource* external_kv_resident,
    bool* io_external_kv_needs_resume_barrier, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident,
    bool external_rope_has, uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge, GpuLayerLoopInFlight** out_inflight);
}  // namespace superslm_gpu

namespace {

// The identical SeqState prefix layout PrepareGpuLayerLoopChunkOpenState packs
// (src/gpu/superslm_gpu.cpp): hidden_codes[H] as i32 elements, Align8U32(H*4) bytes, then
// hidden_scale.m/.e as two i64 -- the exact `[0, SeqScaleOff(H)+16)` range design Sec5 2b names.
uint32_t Align8U32(uint32_t v) { return (v + 7u) & ~7u; }
uint32_t SeqScaleOffLocal(uint32_t hidden_size) { return Align8U32(hidden_size * 4u); }

void PackEmbeddingBlock(const int8_t* codes, size_t hidden_size, const CarriedScale& scale,
                         std::vector<uint8_t>* out) {
	const uint32_t H = static_cast<uint32_t>(hidden_size);
	const uint32_t block_bytes = SeqScaleOffLocal(H) + 16u;
	out->assign(block_bytes, 0);
	for (uint32_t i = 0; i < H; ++i) {
		int32_t v = static_cast<int32_t>(codes[i]);
		std::memcpy(out->data() + i * 4u, &v, 4);
	}
	std::memcpy(out->data() + SeqScaleOffLocal(H) + 0, &scale.m, 8);
	std::memcpy(out->data() + SeqScaleOffLocal(H) + 8, &scale.e, 8);
}

bool RunSelfCheck(const char* label, const SslmModelView& model_view,
                   const std::vector<LayerWeights>& layers, const SslmTensorView* embed_w,
                   const CarriedScale& embed_site_constant, const std::vector<int32_t>& token_ids) {
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t num_kv_heads = model_view.config.num_key_value_heads;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	// --- Reference arm: chunk.size() separate RunLayerLoopGpu calls, each one token, full
	// depth per call -- the granularity design Sec6 fold round 2 (D-SLM3610) names as
	// trivially current at every stage, never itself a claim under test. ---
	std::vector<int8_t> ref_codes(hidden_size, 0);
	SequenceLayerState ref_seq;
	ref_seq.hidden_codes = ref_codes.data();
	std::vector<uint8_t> ref_ws(kv_bytes, 0);
	SslmForwardStatus ref_status = SslmForwardStatus::Ok;
	for (size_t i = 0; i < token_ids.size(); ++i) {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token_ids[i], model_view.config.vocab_size, embed_weights, hidden_size,
		               embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) {
			std::printf("[%s] FAILED at stage=ref_embed token=%d status=%s\n", label, token_ids[i],
			            SslmForwardStatusName(est));
			return false;
		}
		std::memcpy(ref_seq.hidden_codes, embed_codes.data(), hidden_size);
		ref_seq.hidden_scale = embed_scale;
		ref_seq.layer_index = 0;
		ref_status = superslm_gpu::RunLayerLoopGpu(
		    ref_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers,
		    hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
		    model_view.rope_tables, ref_ws.data(), ref_ws.size());
		if (ref_status != SslmForwardStatus::Ok) {
			std::printf("[%s] FAILED at stage=ref_run token_index=%zu status=%s\n", label, i,
			            SslmForwardStatusName(ref_status));
			return false;
		}
	}

	// --- Candidate arm: ONE call into SubmitChunkToFullDepthForG5Bridge, every token's own
	// embedding pre-packed by the SAME EmbedEntry arithmetic, ahead of recording. ---
	SequenceLayerState cand_seq;
	std::vector<int8_t> cand_codes(hidden_size, 0);
	cand_seq.hidden_codes = cand_codes.data();
	std::vector<uint8_t> cand_ws(kv_bytes, 0);
	std::vector<uint8_t> chunk_bytes;
	const uint32_t block_bytes = SeqScaleOffLocal(static_cast<uint32_t>(hidden_size)) + 16u;
	chunk_bytes.resize(block_bytes * token_ids.size());
	for (size_t i = 0; i < token_ids.size(); ++i) {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token_ids[i], model_view.config.vocab_size, embed_weights, hidden_size,
		               embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) {
			std::printf("[%s] FAILED at stage=cand_embed token=%d status=%s\n", label, token_ids[i],
			            SslmForwardStatusName(est));
			return false;
		}
		std::vector<uint8_t> block;
		PackEmbeddingBlock(embed_codes.data(), hidden_size, embed_scale, &block);
		std::memcpy(chunk_bytes.data() + i * block_bytes, block.data(), block_bytes);
	}
	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const SslmForwardStatus submit_status = superslm_gpu::SubmitChunkToFullDepthForG5Bridge(
	    cand_seq, layers.data(), num_hidden_layers, hidden_size, head_dim, num_kv_heads,
	    intermediate_size, context_cap, model_view.rope_tables, cand_ws.data(), cand_ws.size(),
	    chunk_bytes.data(), static_cast<uint32_t>(token_ids.size()), nullptr, nullptr, nullptr,
	    nullptr, nullptr, false, 0, 0, nullptr, &inflight);
	if (submit_status != SslmForwardStatus::Ok || !inflight) {
		std::printf("[%s] FAILED at stage=cand_submit status=%s\n", label,
		            SslmForwardStatusName(submit_status));
		return false;
	}
	int32_t ready = 0;
	const SslmForwardStatus finish_status =
	    superslm_gpu::RunLayerLoopGpuFinish(inflight, cand_seq, cand_ws.data(), /*block=*/1, &ready);
	if (finish_status != SslmForwardStatus::Ok) {
		std::printf("[%s] FAILED at stage=cand_finish status=%s\n", label,
		            SslmForwardStatusName(finish_status));
		return false;
	}

	// --- Compare, full SequenceLayerState-complete surface + every K/V row. ---
	bool all_match = true;
	if (ref_seq.layer_index != cand_seq.layer_index) {
		std::printf("[%s] DIVERGENCE: layer_index: ref=%u cand=%u\n", label, ref_seq.layer_index,
		            cand_seq.layer_index);
		all_match = false;
	}
	if (ref_seq.hidden_scale.m != cand_seq.hidden_scale.m ||
	    ref_seq.hidden_scale.e != cand_seq.hidden_scale.e) {
		std::printf("[%s] DIVERGENCE: hidden_scale: ref=(%lld,%lld) cand=(%lld,%lld)\n", label,
		            (long long)ref_seq.hidden_scale.m, (long long)ref_seq.hidden_scale.e,
		            (long long)cand_seq.hidden_scale.m, (long long)cand_seq.hidden_scale.e);
		all_match = false;
	}
	if (ref_seq.kv_saturation_count != cand_seq.kv_saturation_count) {
		std::printf("[%s] DIVERGENCE: kv_saturation_count: ref=%llu cand=%llu\n", label,
		            (unsigned long long)ref_seq.kv_saturation_count,
		            (unsigned long long)cand_seq.kv_saturation_count);
		all_match = false;
	}
	if (ref_seq.context_length != cand_seq.context_length) {
		std::printf("[%s] DIVERGENCE: context_length: ref=%lld cand=%lld\n", label,
		            (long long)ref_seq.context_length, (long long)cand_seq.context_length);
		all_match = false;
	}
	for (size_t i = 0; i < hidden_size; ++i) {
		if (ref_codes[i] != cand_codes[i]) {
			std::printf("[%s] DIVERGENCE: hidden_codes[%zu]: ref=%d cand=%d\n", label, i, ref_codes[i],
			            cand_codes[i]);
			all_match = false;
			break;
		}
	}
	bool kv_match = true;
	for (uint32_t l = 0; l < num_hidden_layers && kv_match; ++l) {
		for (uint32_t h = 0; h < num_kv_heads && kv_match; ++h) {
			for (int64_t p = 0; p < static_cast<int64_t>(token_ids.size()) && kv_match; ++p) {
				const int8_t* ref_k =
				    superslm_gpu::KeyRowGpu(ref_ws.data(), l, context_cap, num_kv_heads, head_dim, h, p);
				const int8_t* cand_k = superslm_gpu::KeyRowGpu(cand_ws.data(), l, context_cap,
				                                                num_kv_heads, head_dim, h, p);
				const int8_t* ref_v = superslm_gpu::ValueRowGpu(ref_ws.data(), l, context_cap,
				                                                 num_kv_heads, head_dim, h, p);
				const int8_t* cand_v = superslm_gpu::ValueRowGpu(cand_ws.data(), l, context_cap,
				                                                  num_kv_heads, head_dim, h, p);
				for (size_t d = 0; d < head_dim; ++d) {
					if (ref_k[d] != cand_k[d]) {
						std::printf(
						    "[%s] DIVERGENCE: K[layer=%u][kv_head=%u][pos=%lld][dim=%zu]: ref=%d cand=%d\n",
						    label, l, h, (long long)p, d, ref_k[d], cand_k[d]);
						kv_match = false;
						all_match = false;
						break;
					}
					if (ref_v[d] != cand_v[d]) {
						std::printf(
						    "[%s] DIVERGENCE: V[layer=%u][kv_head=%u][pos=%lld][dim=%zu]: ref=%d cand=%d\n",
						    label, l, h, (long long)p, d, ref_v[d], cand_v[d]);
						kv_match = false;
						all_match = false;
						break;
					}
				}
			}
		}
	}
	if (all_match) {
		std::printf(
		    "[%s] RESULT: BIT-IDENTICAL -- %zu token(s), full SequenceLayerState-complete surface "
		    "(hidden_codes[%zu], hidden_scale, layer_index, kv_saturation_count, context_length) and "
		    "every K/V row across all %u layers at every committed position.\n",
		    label, token_ids.size(), hidden_size, num_hidden_layers);
	} else {
		std::printf("[%s] RESULT: DIVERGENCE FOUND (see above) -- not bit-identical.\n", label);
	}
	return all_match;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm>\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
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
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u vocab=%u context_cap=%u\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim, model_view.config.vocab_size, model_view.config.context_cap);

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;

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

	bool overall_ok = true;
	// Rung 2's own exit condition (b): chunk_tokens=1, mechanism-only (D-SLM3610).
	overall_ok &= RunSelfCheck("chunk_tokens=1", model_view, layers, embed_w, embed_site_constant, {90});
	// Rung 2's own exit condition (b2): the 4-token discriminating self-check (D-SLM3608/D-SLM3615).
	overall_ok &=
	    RunSelfCheck("4-token", model_view, layers, embed_w, embed_site_constant, {90, 1, 258, 1960});

	// D-SLM3649 (this rung's own fold, cdb-confirmed nvwgf2umx driver recursion above
	// kT2169TdrSafeMaxChunkTokens*2=8 tokens in one open command list): the sub-chunk split
	// wrapper's own pin. `chunk_tokens=8` straddles the split boundary exactly (2 internal
	// sub-chunks of 4) -- the cell that would have crashed the process outright before this
	// rung's own fix landed.
	std::vector<int32_t> straddle_tokens;
	for (int i = 0; i < 8; ++i) straddle_tokens.push_back(90 + (i % 50));
	overall_ok &=
	    RunSelfCheck("chunk_tokens=8 (split boundary)", model_view, layers, embed_w, embed_site_constant,
	                 straddle_tokens);

	// A much larger chunk (256 tokens, matching design Sec5's own pure-TDR-arithmetic figure) --
	// proves the split-wrapper fix SCALES, not merely clears the one crashing size (64 internal
	// sub-chunk submissions of 4 tokens each, each its own synchronous Finish per the wrapper's
	// own D-SLM3649 ruling).
	std::vector<int32_t> large_tokens;
	for (int i = 0; i < 256; ++i) large_tokens.push_back(90 + (i % 50));
	overall_ok &= RunSelfCheck("chunk_tokens=256 (TDR-arithmetic scale)", model_view, layers, embed_w,
	                            embed_site_constant, large_tokens);

	std::printf("\n=== T-2169 Rung 2b self-check: %s ===\n", overall_ok ? "PASS" : "FAIL");
	return overall_ok ? 0 : 1;
}
