// GPU per-token overhead plan rev 3.1, §7 cell 7.4: the four record-log fixtures, and the driver
// that captures each one's command stream through the record-log seam (`SetGpuRecordLogSink`,
// include/superslm/gpu_port.h, compiled only under SUPERSLM_ENABLE_GPU_RECORD_LOG).
//
// Cell 7.4 compares per-site mode's stream on THIS build with a capture the box makes from v1.11.0
// plus the same seam (plan §4.2 B0.6; the common brief re-anchors "v1.9.0" to v1.11.0, whose
// recorder is byte-for-byte d86aa3c's). Both sides must drive the identical fixtures, so the
// fixtures and the driver live here, header-only, and use only entry points whose signatures are
// unchanged since v1.11.0: SslmModel::Load, RunLayerLoopGpuSubmit / RunLayerLoopGpuFinish,
// SubmitChunkToFullDepthForG5Bridge, harness::Device::Upload, and the seam itself. Nothing here
// names a timing mode: v1.11.0 has only per-site timing, and a caller on a later build selects
// per-site mode itself before it captures (the cell does; WriteRecordLogFixtures says so).
//
// The capture program for B0.6 is a few lines over this header: include it, call
// WriteRecordLogFixtures(<dir>), and file <dir> as tests/fixtures/gpu_record_log/v1p11/ (the four
// .log files plus SHA256SUMS, sha256sum's own "<hex>  <name>" format). Files are written in
// binary mode with '\n' line ends, so the hash does not depend on the platform that wrote them
// (.gitattributes keeps git from converting them either).
//
// The four fixtures (plan §7 7.4):
//   legacy_2layer  -- two legacy layers (24 dispatches each), one token, single-token submit.
//   qknorm_2layer  -- the same two layers with K norm wired on both, so qk_norm_site records (25).
//   chunk_2token   -- the legacy model, two tokens in one chunk-bridge submission (one list).
//   adapter_2layer -- the legacy model with an adapter bridge whose every slot is present at rank
//                     1 with distinct offsets, so the tail root constants 11-27 are non-zero. Its
//                     A/B/fold buffers are zero-filled upload buffers: the cell compares RECORDED
//                     commands, which do not depend on what the adapter data holds, so the call's
//                     own finish status is reported and not asserted.
// Geometry is the suite's degenerate synthetic one (hidden 2, one head, one KV head, head_dim 2,
// intermediate 2) at context_cap 4: every site records, every group count is 1, and the stream
// still carries every dispatch, root constant, barrier, query and resolve in order.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/intmath.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "../sslm_t1899_optionG_fixtures.h"  // MakeOptionGRop1SectionMultiRow, BuildCfg1, BuildArtifact
#include "../../src/gpu/d3d12_harness.h"

namespace superslm_gpu_record_log {

inline constexpr size_t kFixtureCount = 4;
inline constexpr const char* kFixtureNames[kFixtureCount] = {"legacy_2layer", "qknorm_2layer",
                                                              "chunk_2token", "adapter_2layer"};

// Two synthetic layers, wired the way test_main.cpp's NLayerFixture / QkNormWiringFixture wire
// theirs (identity weights, identity WSC1 folds, canonical site constants, the off-canonical
// mlp_act constant), self-contained so a capture program built at v1.11.0 needs nothing from
// test_main.cpp.
struct RecordLogModel {
	static constexpr uint32_t kLayers = 2;
	static constexpr int32_t kContextCap = 4;
	static constexpr size_t kHidden = 2, kHeadDim = 2, kKvHeads = 1, kInter = 2;
	static constexpr size_t kWorkspaceBytes =
	    size_t{kLayers} * static_cast<size_t>(kContextCap) * kKvHeads * kHeadDim * 2;
	static constexpr size_t kFoldChannels = 8;  // every projection here is <= 2 channels wide

	superslm::SslmModelView view;
	superslm::LayerWeights layers[kLayers]{};
	bool loaded = false;
	std::string load_error;

	int32_t fold_identity[kFoldChannels] = {1, 1, 1, 1, 1, 1, 1, 1};
	int32_t fold_zero[kFoldChannels] = {};
	int64_t kv_landing_r_t_arr[1] = {0};
	int64_t kv_landing_e_t_arr[1] = {0};
	int32_t ctx_fold_identity_arr[1] = {1};
	int32_t ctx_fold_mult_arr[1] = {0};
	int32_t ctx_fold_shift_arr[1] = {0};
	int32_t norm_gain[kLayers][2] = {{16384, 16384}, {16384, 16384}};
	int8_t identity2x2[4] = {1, 0, 0, 1};
	int64_t iexp_softmax_khead_m_arr[1] = {INT64_C(1073741824)};
	int64_t iexp_softmax_khead_e_arr[1] = {-86};
	int32_t k_norm_gain[2] = {16384, 16384};
	int64_t k_norm_landing_r_t_arr[2] = {0, 0};
	int64_t k_norm_landing_e_t_arr[2] = {INT64_C(-20), INT64_C(-20)};
	int64_t k_channel_ratio_arr[2] = {INT64_C(2147483648), INT64_C(2147483648)};

	explicit RecordLogModel(bool with_k_norm) {
		using namespace superslm_test;
		using superslm::CarriedScale;
		Cfg1Spec spec{};
		spec.hidden_size = kHidden;
		spec.num_hidden_layers = kLayers;
		spec.num_attention_heads = 1;
		spec.num_key_value_heads = kKvHeads;
		spec.head_dim = kHeadDim;
		spec.intermediate_size = kInter;
		spec.context_cap = kContextCap;
		spec.kv_precision = 0;
		spec.kv_block_size = 1;
		FixtureSection config =
		    MakeSection(superslm::SslmSectionType::Config, superslm::SslmDtype::Raw, BuildCfg1(spec));
		std::vector<int64_t> cos_flat(kContextCap, INT64_C(1073741824));
		std::vector<int64_t> sin_flat(kContextCap, INT64_C(0));
		FixtureSection rope =
		    MakeOptionGRop1SectionMultiRow(kContextCap, /*pairs=*/1, cos_flat.data(), sin_flat.data());
		built_ = BuildArtifact({config, MakeSigmoidLutSection(), rope});
		loaded = superslm::SslmModel::Load(built_.bytes.data(), built_.bytes.size(), view, &load_error) ==
		         superslm::SslmModelStatus::Ok;

		const CarriedScale canonical{INT64_C(1073741824), INT64_C(-30)};
		kv_landing_r_t_arr[0] = superslm::DynamicScaleReciprocal(canonical.m);
		const CarriedScale landing_target{INT64_C(1073741824), INT64_C(-20)};
		k_norm_landing_r_t_arr[0] = superslm::DynamicScaleReciprocal(landing_target.m);
		k_norm_landing_r_t_arr[1] = k_norm_landing_r_t_arr[0];
		for (uint32_t l = 0; l < kLayers; ++l) {
			superslm::LayerWeights& lw = layers[l];
			lw.attn_norm_gain = norm_gain[l];
			lw.attn_norm_site_constant = canonical;
			lw.q_weight = identity2x2;
			lw.k_weight = identity2x2;
			lw.v_weight = identity2x2;
			lw.o_weight = identity2x2;
			lw.q_fold_identity = fold_identity; lw.q_fold_mult = fold_zero; lw.q_fold_shift = fold_zero;
			lw.k_fold_identity = fold_identity; lw.k_fold_mult = fold_zero; lw.k_fold_shift = fold_zero;
			lw.v_fold_identity = fold_identity; lw.v_fold_mult = fold_zero; lw.v_fold_shift = fold_zero;
			lw.o_fold_identity = fold_identity; lw.o_fold_mult = fold_zero; lw.o_fold_shift = fold_zero;
			lw.gate_fold_identity = fold_identity; lw.gate_fold_mult = fold_zero; lw.gate_fold_shift = fold_zero;
			lw.up_fold_identity = fold_identity; lw.up_fold_mult = fold_zero; lw.up_fold_shift = fold_zero;
			lw.down_fold_identity = fold_identity; lw.down_fold_mult = fold_zero; lw.down_fold_shift = fold_zero;
			lw.q_site_constant = canonical;
			lw.o_site_constant = canonical;
			lw.kv_landing_r_t_k = kv_landing_r_t_arr;
			lw.kv_landing_e_t_k = kv_landing_e_t_arr;
			lw.kv_landing_r_t_v = kv_landing_r_t_arr;
			lw.kv_landing_e_t_v = kv_landing_e_t_arr;
			lw.ctx_fold_identity = ctx_fold_identity_arr;
			lw.ctx_fold_mult = ctx_fold_mult_arr;
			lw.ctx_fold_shift = ctx_fold_shift_arr;
			lw.ctx_fold_site_constant = canonical;
			lw.attn_residual_site_constant = canonical;
			lw.iexp_softmax_khead_m = iexp_softmax_khead_m_arr;
			lw.iexp_softmax_khead_e = iexp_softmax_khead_e_arr;
			lw.mlp_norm_gain = norm_gain[l];
			lw.mlp_norm_site_constant = canonical;
			lw.gate_weight = identity2x2;
			lw.up_weight = identity2x2;
			lw.down_weight = identity2x2;
			lw.gate_site_constant = canonical;
			lw.up_site_constant = canonical;
			lw.mlp_act_site_constant = CarriedScale{INT64_C(1073741824), INT64_C(-96)};
			lw.down_site_constant = canonical;
			lw.mlp_residual_site_constant = canonical;
			if (with_k_norm) {
				lw.k_norm_gain = k_norm_gain;
				lw.k_norm_site_constant = canonical;
				lw.k_channel_r_t = k_norm_landing_r_t_arr;
				lw.k_channel_e_t = k_norm_landing_e_t_arr;
				lw.k_channel_ratio = k_channel_ratio_arr;
			}
		}
	}

	// Every LayerWeights pointer above points into this object's own arrays (the suite's
	// D-SLM492 rule for fixtures of this shape): no copy, no move.
	RecordLogModel(const RecordLogModel&) = delete;
	RecordLogModel& operator=(const RecordLogModel&) = delete;
	RecordLogModel(RecordLogModel&&) = delete;
	RecordLogModel& operator=(RecordLogModel&&) = delete;

private:
	superslm_test::BuiltArtifact built_;
};

// The adapter bridge for adapter_2layer: every (layer, projection) slot present at rank 1, with
// offsets that differ per slot so a constant written to the wrong position shows in the stream.
struct RecordLogAdapter {
	static constexpr uint64_t kBufferBytes = 64 * 1024;
	Microsoft::WRL::ComPtr<ID3D12Resource> ab;
	Microsoft::WRL::ComPtr<ID3D12Resource> fold;
	std::vector<superslm_gpu::AdapterProjSlot> slots;
	superslm_gpu::GpuAdapterBridge bridge{};

	bool Build(std::string* err) {
		superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
		if (!dev.available.load()) {
			if (err) *err = "D3D12 device unavailable";
			return false;
		}
		const std::vector<uint8_t> zeros(static_cast<size_t>(kBufferBytes), 0);
		try {
			ab = dev.Upload(zeros.data(), kBufferBytes);
			fold = dev.Upload(zeros.data(), kBufferBytes);
		} catch (const std::exception& e) {
			if (err) *err = std::string("adapter buffer upload threw: ") + e.what();
			return false;
		}
		slots.assign(static_cast<size_t>(RecordLogModel::kLayers) * 7u, superslm_gpu::AdapterProjSlot{});
		for (size_t i = 0; i < slots.size(); ++i) {
			superslm_gpu::AdapterProjSlot& s = slots[i];
			s.present = true;
			s.rank = 1;
			s.out_channels = 2;
			s.a_offset = 256u * i + 16u;
			s.b_offset = 256u * i + 80u;
			s.a_bytes = 2;
			s.b_bytes = 2;
			s.fold_offset = 256u * i + 144u;
		}
		bridge.lora_ab_resident = ab.Get();
		bridge.fold_resident = fold.Get();
		bridge.rank = 1;
		bridge.slots = slots.data();
		bridge.num_hidden_layers = RecordLogModel::kLayers;
		return true;
	}
};

// One fixture's call: submits it (single-token or chunk bridge), finishes it, and reports both
// statuses. The record stream is complete once Submit returns, so a caller that only wants the
// stream needs submit_status == Ok.
struct FixtureRunResult {
	bool ran = false;  // the fixture was built and its submit was reached
	superslm::SslmForwardStatus submit_status = superslm::SslmForwardStatus::Ok;
	superslm::SslmForwardStatus finish_status = superslm::SslmForwardStatus::Ok;
	std::string error;
};

inline std::vector<uint8_t> PackEmbeddingBlock(const int8_t* codes, size_t hidden_size,
                                               superslm::CarriedScale scale) {
	// The SeqState embedding-block packing PrepareGpuLayerLoopChunkOpenState reads (hidden codes
	// as i32, Align8(H*4) bytes, then the scale's m and e as i64) -- tools/t2169_rung2b_selfcheck's
	// PackEmbeddingBlock.
	const uint32_t H = static_cast<uint32_t>(hidden_size);
	const uint32_t aligned = (H * 4u + 7u) & ~7u;
	std::vector<uint8_t> block(aligned + 16u, 0);
	for (uint32_t i = 0; i < H; ++i) {
		const int32_t v = static_cast<int32_t>(codes[i]);
		std::memcpy(block.data() + i * 4u, &v, 4);
	}
	std::memcpy(block.data() + aligned + 0, &scale.m, 8);
	std::memcpy(block.data() + aligned + 8, &scale.e, 8);
	return block;
}

inline FixtureRunResult RunRecordLogFixture(const std::string& name) {
	FixtureRunResult r;
	const bool qk = name == "qknorm_2layer";
	const bool chunk = name == "chunk_2token";
	const bool adapter = name == "adapter_2layer";
	if (!qk && !chunk && !adapter && name != "legacy_2layer") {
		r.error = "unknown record-log fixture '" + name + "'";
		return r;
	}
	RecordLogModel model(qk);
	if (!model.loaded) {
		r.error = "fixture artifact failed to load: " + model.load_error;
		return r;
	}
	RecordLogAdapter adapter_backing;
	if (adapter && !adapter_backing.Build(&r.error)) return r;

	int8_t codes[RecordLogModel::kHidden] = {5, -5};
	const superslm::CarriedScale scale{INT64_C(1073741824), 0};
	superslm::SequenceLayerState seq;
	seq.hidden_codes = codes;
	seq.hidden_scale = scale;
	seq.layer_index = 0;
	std::vector<uint8_t> ws(RecordLogModel::kWorkspaceBytes, 0);
	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	r.ran = true;
	if (chunk) {
		std::vector<uint8_t> blocks;
		for (int t = 0; t < 2; ++t) {
			const std::vector<uint8_t> b = PackEmbeddingBlock(codes, RecordLogModel::kHidden, scale);
			blocks.insert(blocks.end(), b.begin(), b.end());
		}
		r.submit_status = superslm_gpu::SubmitChunkToFullDepthForG5Bridge(
		    seq, model.layers, RecordLogModel::kLayers, RecordLogModel::kHidden, RecordLogModel::kHeadDim,
		    RecordLogModel::kKvHeads, RecordLogModel::kInter, RecordLogModel::kContextCap,
		    model.view.rope_tables, ws.data(), ws.size(), blocks.data(), /*chunk_len=*/2,
		    /*external_kv_resident=*/nullptr, /*io_external_kv_needs_resume_barrier=*/nullptr,
		    /*external_weights_resident=*/nullptr, /*external_rope_cos_resident=*/nullptr,
		    /*external_rope_sin_resident=*/nullptr, /*external_rope_has=*/false,
		    /*external_rope_cos_elems=*/0, /*external_rope_sin_elems=*/0, /*adapter_bridge=*/nullptr,
		    &inflight, /*q_width=*/0, /*model_generation=*/0);
	} else {
		r.submit_status = superslm_gpu::RunLayerLoopGpuSubmit(
		    seq, model.layers, RecordLogModel::kLayers, /*layer_budget=*/RecordLogModel::kLayers,
		    RecordLogModel::kHidden, RecordLogModel::kHeadDim, RecordLogModel::kKvHeads,
		    RecordLogModel::kInter, RecordLogModel::kContextCap, model.view.rope_tables, ws.data(),
		    ws.size(), /*external_kv_resident=*/nullptr,
		    /*io_external_kv_needs_resume_barrier=*/nullptr, &inflight,
		    /*external_weights_resident=*/nullptr, /*external_rope_cos_resident=*/nullptr,
		    /*external_rope_sin_resident=*/nullptr, /*external_rope_has=*/false,
		    /*external_rope_cos_elems=*/0, /*external_rope_sin_elems=*/0,
		    adapter ? &adapter_backing.bridge : nullptr);
	}
	if (inflight != nullptr) {
		int32_t ready = 0;
		r.finish_status = superslm_gpu::RunLayerLoopGpuFinish(inflight, seq, ws.data(), /*block=*/1, &ready);
	}
	return r;
}

#if defined(SUPERSLM_ENABLE_GPU_RECORD_LOG)
// Captures one fixture's record log. False (with *err set) when the fixture could not be built or
// its submit did not return Ok -- a stream from a refused submit is empty and means nothing.
inline bool CaptureRecordLog(const std::string& name, std::vector<std::string>* out_log,
                             std::string* err) {
	out_log->clear();
	superslm_gpu::SetGpuRecordLogSink(out_log);
	FixtureRunResult r;
	try {
		r = RunRecordLogFixture(name);
	} catch (const std::exception& e) {
		superslm_gpu::SetGpuRecordLogSink(nullptr);
		if (err) *err = name + ": threw " + e.what();
		return false;
	}
	superslm_gpu::SetGpuRecordLogSink(nullptr);
	if (!r.ran) {
		if (err) *err = name + ": " + r.error;
		return false;
	}
	if (r.submit_status != superslm::SslmForwardStatus::Ok) {
		if (err) {
			*err = name + ": submit returned " +
			       std::string(superslm::SslmForwardStatusName(r.submit_status));
		}
		return false;
	}
	return true;
}

inline std::string JoinRecordLog(const std::vector<std::string>& log) {
	std::string text;
	for (const std::string& line : log) {
		text += line;
		text += '\n';
	}
	return text;
}

inline std::string Sha256Hex(const std::string& bytes) {
	uint8_t digest[32];
	superslm::Sha256Hash(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), digest);
	static const char* kHex = "0123456789abcdef";
	std::string hex;
	for (uint8_t b : digest) {
		hex += kHex[b >> 4];
		hex += kHex[b & 15];
	}
	return hex;
}

// Writes <dir>/<fixture>.log for all four fixtures and <dir>/SHA256SUMS. On a build that has
// timing modes, the caller selects per-site mode first: the capture is the per-site stream.
inline bool WriteRecordLogFixtures(const std::string& dir, std::string* err) {
	std::string sums;
	for (const char* name : kFixtureNames) {
		std::vector<std::string> log;
		if (!CaptureRecordLog(name, &log, err)) return false;
		const std::string text = JoinRecordLog(log);
		const std::string path = dir + "/" + name + ".log";
		std::FILE* f = std::fopen(path.c_str(), "wb");
		if (!f) {
			if (err) *err = "cannot open " + path + " for writing";
			return false;
		}
		const bool wrote = std::fwrite(text.data(), 1, text.size(), f) == text.size();
		std::fclose(f);
		if (!wrote) {
			if (err) *err = "short write on " + path;
			return false;
		}
		sums += Sha256Hex(text) + "  " + name + ".log\n";
	}
	const std::string sums_path = dir + "/SHA256SUMS";
	std::FILE* f = std::fopen(sums_path.c_str(), "wb");
	if (!f) {
		if (err) *err = "cannot open " + sums_path + " for writing";
		return false;
	}
	const bool wrote = std::fwrite(sums.data(), 1, sums.size(), f) == sums.size();
	std::fclose(f);
	if (!wrote && err) *err = "short write on " + sums_path;
	return wrote;
}
#endif  // SUPERSLM_ENABLE_GPU_RECORD_LOG

}  // namespace superslm_gpu_record_log
