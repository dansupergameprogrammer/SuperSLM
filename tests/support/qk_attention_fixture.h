// Attention and per-row sites plan (rev 3.1), cell 11.1(c): the widened QK-norm attention fixture.
//
// Built the way test_main.cpp's QkNormWiringFixture is (a minimal loaded artifact for the RoPE table,
// then LayerWeights wired to this object's own arrays, driving the real RunLayerLoop and
// RunLayerLoopChunkBatched), and widened so the Q31 score path runs at a real geometry: hidden 256,
// 4 query heads over 2 KV heads (G = 2), head_dim 64, intermediate 256, one layer, context_cap 32,
// q_norm and k_norm gains set, k_channel_ratio = 2^31 on every channel (the loader's maximum, G6), and
// int8 weights from a fixed-seed generator so the scores spread.
//
// Header-only and integer-only: the suite (tests/test_attn_rowsites.cpp) and the golden-pin generator
// (tools/gen_attn_rowsite_golden.cpp, built against the v1.9.0 tag's library) both include it, so it
// needs no build entry and takes nothing from the code it grades. It calls only entry points whose
// signatures are unchanged since v1.9.0 (SslmModel::Load, RunLayerLoop, RunLayerLoopChunkBatched,
// DynamicScaleReciprocal). The RoPE table is built from Pythagorean triples, never libm, so every
// platform builds the same bytes.
//
// Every run feeds the same 24 positions (fixed-seed hidden codes and scales) and emits, as int64 values:
// per position, the layer's 256 output codes and its output scale (m, e); then every byte of the K/V
// workspace. The decode loop (one RunLayerLoop per position), the chunk loop (one
// RunLayerLoopChunkBatched over all 24) and the decode loop with a capture sink installed must all emit
// the same stream, and its hash is the golden pin's fixture hash (plan §3.3 evidence 3, cell 6.3).
#ifndef SUPERSLM_TESTS_SUPPORT_QK_ATTENTION_FIXTURE_H
#define SUPERSLM_TESTS_SUPPORT_QK_ATTENTION_FIXTURE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/model.h"
#include "../sslm_cfg1_hostile_fixtures.h"   // Cfg1Spec, BuildCfg1
#include "../sslm_fixtures.h"                // BuildArtifact, MakeSection
#include "../sslm_model_hostile_fixtures.h"  // ManifestTensorSpec, BuildManifest, PutU64
#include "../sslm_sil1_hostile_fixtures.h"   // MakeSigmoidLutSection

namespace superslm_qk_fixture {

inline constexpr size_t kHidden = 256;
inline constexpr size_t kHeads = 4;
inline constexpr size_t kKvHeads = 2;
inline constexpr size_t kHeadDim = 64;
inline constexpr size_t kInter = 256;
inline constexpr int64_t kCap = 32;
inline constexpr size_t kPositions = 24;
inline constexpr size_t kKvWidth = kKvHeads * kHeadDim;
inline constexpr size_t kWorkspaceBytes = size_t{1} * static_cast<size_t>(kCap) * kKvHeads * kHeadDim * 2;
inline constexpr int64_t kRatioQ31 = INT64_C(2147483648);  // 2^31, every channel

// splitmix64, integer-only (the generator rowsite_cases.h and attention_cases.h use).
struct FixtureRng {
	uint64_t s;
	explicit FixtureRng(uint64_t seed) : s(seed) {}
	uint64_t Next() {
		s += 0x9e3779b97f4a7c15ULL;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		return z ^ (z >> 31);
	}
	int64_t InRange(int64_t lo, int64_t hi) {
		return lo + static_cast<int64_t>(Next() % (static_cast<uint64_t>(hi - lo) + 1ULL));
	}
};

// A Q2.30 rotation per (position, pair) from Pythagorean triples: cos = a/c, sin = +-b/c, exact integer
// division, so |cos|, |sin| <= 2^30 (the loader's RoPE entry bound) and no platform's libm is involved.
// Position 0 is the identity, as a real table's is.
inline superslm_test::FixtureSection MakeRopeSection() {
	static constexpr int64_t kTriples[][3] = {{3, 4, 5}, {5, 12, 13}, {8, 15, 17}, {7, 24, 25}, {20, 21, 29},
	                                          {12, 35, 37}, {9, 40, 41}, {28, 45, 53}, {11, 60, 61}, {33, 56, 65}};
	const size_t pairs = kHeadDim / 2;
	const size_t n = static_cast<size_t>(kCap) * pairs;
	std::vector<int64_t> cos_flat(n), sin_flat(n);
	for (size_t p = 0; p < static_cast<size_t>(kCap); ++p) {
		for (size_t i = 0; i < pairs; ++i) {
			const size_t at = p * pairs + i;
			if (p == 0) {
				cos_flat[at] = INT64_C(1) << 30;
				sin_flat[at] = 0;
				continue;
			}
			const int64_t* t = kTriples[(p * 7 + i * 3) % 10];
			const bool swap = ((p + i) & 1) != 0;
			const int64_t a = swap ? t[1] : t[0], b = swap ? t[0] : t[1];
			cos_flat[at] = ((p + 2 * i) % 3 == 0 ? -1 : 1) * ((a << 30) / t[2]);
			sin_flat[at] = ((p + i) % 4 < 2 ? -1 : 1) * ((b << 30) / t[2]);
		}
	}
	std::vector<superslm_test::ManifestTensorSpec> tensors = {
	    {"cos", {static_cast<uint32_t>(kCap), static_cast<uint32_t>(pairs)}},
	    {"sin", {static_cast<uint32_t>(kCap), static_cast<uint32_t>(pairs)}},
	};
	auto manifest = superslm_test::BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	for (size_t i = 0; i < n; ++i) {
		superslm_test::PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + i * 8,
		                      static_cast<uint64_t>(cos_flat[i]));
		superslm_test::PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + i * 8,
		                      static_cast<uint64_t>(sin_flat[i]));
	}
	return superslm_test::MakeSection(superslm::SslmSectionType::RopeTables, superslm::SslmDtype::Int64,
	                                  manifest.bytes, /*alignment=*/64);
}

struct QkAttentionFixture {
	superslm::SslmModelView view;
	superslm::LayerWeights layer{};
	bool loaded = false;
	std::string load_error;

	std::vector<int8_t> q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
	std::vector<int32_t> fold_identity, fold_zero;
	std::vector<int32_t> hidden_gain, q_norm_gain, k_norm_gain;
	int64_t kv_landing_r_t[kKvHeads], kv_landing_e_t[kKvHeads];
	int64_t k_channel_r_t[kKvWidth], k_channel_e_t[kKvWidth], k_channel_ratio[kKvWidth];
	int64_t softmax_khead_m[kKvHeads], softmax_khead_e[kKvHeads];
	int32_t ctx_fold_identity[kHeads], ctx_fold_mult[kHeads], ctx_fold_shift[kHeads];

	// The 24 positions' layer inputs.
	std::vector<int8_t> hidden_in;             // kPositions * kHidden
	std::vector<superslm::CarriedScale> scale_in;  // kPositions

	QkAttentionFixture() {
		using superslm::CarriedScale;
		superslm_test::Cfg1Spec spec{};
		spec.hidden_size = static_cast<uint32_t>(kHidden);
		spec.num_hidden_layers = 1;
		spec.num_attention_heads = static_cast<uint32_t>(kHeads);
		spec.num_key_value_heads = static_cast<uint32_t>(kKvHeads);
		spec.head_dim = static_cast<uint32_t>(kHeadDim);
		spec.intermediate_size = static_cast<uint32_t>(kInter);
		spec.context_cap = static_cast<uint32_t>(kCap);
		spec.kv_precision = 0;
		spec.kv_block_size = 1;
		auto built = superslm_test::BuildArtifact(
		    {superslm_test::MakeSection(superslm::SslmSectionType::Config, superslm::SslmDtype::Raw,
		                                superslm_test::BuildCfg1(spec)),
		     superslm_test::MakeSigmoidLutSection(), MakeRopeSection()});
		artifact_bytes = std::move(built.bytes);
		loaded = superslm::SslmModel::Load(artifact_bytes.data(), artifact_bytes.size(), view, &load_error) ==
		         superslm::SslmModelStatus::Ok;

		FixtureRng rng(0x51A7'7E57'0005'0011ULL);
		auto weights = [&](size_t n) {
			std::vector<int8_t> w(n);
			for (auto& x : w) x = static_cast<int8_t>(rng.InRange(-127, 127));
			return w;
		};
		q_w = weights(kHeads * kHeadDim * kHidden);
		k_w = weights(kKvWidth * kHidden);
		v_w = weights(kKvWidth * kHidden);
		o_w = weights(kHidden * kHeads * kHeadDim);
		gate_w = weights(kInter * kHidden);
		up_w = weights(kInter * kHidden);
		down_w = weights(kHidden * kInter);
		fold_identity.assign(kInter > kHidden ? kInter : kHidden, 1);
		fold_zero.assign(fold_identity.size(), 0);
		hidden_gain.assign(kHidden, 8192);
		q_norm_gain.resize(kHeadDim);
		k_norm_gain.resize(kHeadDim);
		for (size_t d = 0; d < kHeadDim; ++d) {
			q_norm_gain[d] = static_cast<int32_t>(rng.InRange(4096, 8192));
			k_norm_gain[d] = static_cast<int32_t>(rng.InRange(4096, 8192));
		}

		const CarriedScale canonical{INT64_C(1073741824), INT64_C(-30)};
		const int64_t canonical_r_t = superslm::DynamicScaleReciprocal(canonical.m);
		for (size_t h = 0; h < kKvHeads; ++h) {
			kv_landing_r_t[h] = canonical_r_t;
			kv_landing_e_t[h] = kKvLandingE;
			softmax_khead_m[h] = INT64_C(1073741824);
			softmax_khead_e[h] = kSoftmaxKheadE;
		}
		for (size_t c = 0; c < kKvWidth; ++c) {
			k_channel_r_t[c] = canonical_r_t;
			k_channel_e_t[c] = kKChannelE;
			k_channel_ratio[c] = kRatioQ31;
		}
		for (size_t h = 0; h < kHeads; ++h) {
			ctx_fold_identity[h] = 1;
			ctx_fold_mult[h] = 0;
			ctx_fold_shift[h] = 0;
		}

		superslm::LayerWeights& lw = layer;
		const int32_t* id = fold_identity.data();
		const int32_t* zero = fold_zero.data();
		lw.attn_norm_gain = hidden_gain.data();
		lw.attn_norm_site_constant = canonical;
		lw.q_weight = q_w.data();
		lw.k_weight = k_w.data();
		lw.v_weight = v_w.data();
		lw.o_weight = o_w.data();
		lw.q_fold_identity = id; lw.q_fold_mult = zero; lw.q_fold_shift = zero;
		lw.k_fold_identity = id; lw.k_fold_mult = zero; lw.k_fold_shift = zero;
		lw.v_fold_identity = id; lw.v_fold_mult = zero; lw.v_fold_shift = zero;
		lw.o_fold_identity = id; lw.o_fold_mult = zero; lw.o_fold_shift = zero;
		lw.gate_fold_identity = id; lw.gate_fold_mult = zero; lw.gate_fold_shift = zero;
		lw.up_fold_identity = id; lw.up_fold_mult = zero; lw.up_fold_shift = zero;
		lw.down_fold_identity = id; lw.down_fold_mult = zero; lw.down_fold_shift = zero;
		lw.q_site_constant = canonical;
		lw.o_site_constant = canonical;
		lw.kv_landing_r_t_k = kv_landing_r_t;
		lw.kv_landing_e_t_k = kv_landing_e_t;
		lw.kv_landing_r_t_v = kv_landing_r_t;
		lw.kv_landing_e_t_v = kv_landing_e_t;
		lw.ctx_fold_identity = ctx_fold_identity;
		lw.ctx_fold_mult = ctx_fold_mult;
		lw.ctx_fold_shift = ctx_fold_shift;
		lw.ctx_fold_site_constant = canonical;
		lw.attn_residual_site_constant = canonical;
		lw.iexp_softmax_khead_m = softmax_khead_m;
		lw.iexp_softmax_khead_e = softmax_khead_e;
		lw.mlp_norm_gain = hidden_gain.data();
		lw.mlp_norm_site_constant = canonical;
		lw.gate_weight = gate_w.data();
		lw.up_weight = up_w.data();
		lw.down_weight = down_w.data();
		lw.gate_site_constant = CarriedScale{INT64_C(1073741824), kGateSiteE};
		lw.up_site_constant = canonical;
		lw.mlp_act_site_constant = CarriedScale{INT64_C(1073741824), INT64_C(-96)};
		lw.down_site_constant = canonical;
		lw.mlp_residual_site_constant = canonical;
		lw.q_norm_gain = q_norm_gain.data();
		lw.q_norm_site_constant = canonical;
		lw.k_norm_gain = k_norm_gain.data();
		lw.k_norm_site_constant = canonical;
		lw.k_wide_source_scale = CarriedScale{INT64_C(1073741824), kKWideSourceE};
		lw.k_channel_r_t = k_channel_r_t;
		lw.k_channel_e_t = k_channel_e_t;
		lw.k_channel_ratio = k_channel_ratio;

		hidden_in.resize(kPositions * kHidden);
		for (auto& x : hidden_in) x = static_cast<int8_t>(rng.InRange(-127, 127));
		scale_in.resize(kPositions);
		for (size_t p = 0; p < kPositions; ++p)
			scale_in[p] = CarriedScale{INT64_C(1073741824) + rng.InRange(0, (INT64_C(1) << 30) - 1), kHiddenE};
	}

	QkAttentionFixture(const QkAttentionFixture&) = delete;
	QkAttentionFixture& operator=(const QkAttentionFixture&) = delete;

	// Tuned on the base (the v1.9.0 code) so that every step returns Ok, K lands mostly unclamped, and the
	// softmax rows are spread and inside §5.4's guard (docs/attention-rowsites/s5/fixture-premise.txt).
	static constexpr int64_t kHiddenE = -30;
	static constexpr int64_t kKvLandingE = 12;
	static constexpr int64_t kKWideSourceE = -60;
	static constexpr int64_t kKChannelE = -36;
	static constexpr int64_t kSoftmaxKheadE = -72;
	static constexpr int64_t kGateSiteE = -52;

 private:
	std::vector<uint8_t> artifact_bytes;
};

template <class Emit>
void EmitPosition(Emit& emit, const int8_t* codes, const superslm::CarriedScale& scale) {
	for (size_t i = 0; i < kHidden; ++i) emit(static_cast<int64_t>(codes[i]));
	emit(scale.m);
	emit(scale.e);
}

template <class Emit>
void EmitWorkspace(Emit& emit, const std::vector<uint8_t>& workspace) {
	for (uint8_t b : workspace) emit(static_cast<int64_t>(b));
}

struct NoPositionHook {
	void operator()(size_t) const {}
};

// Run (i), or run (iii) when `sink` is non-null: the decode loop, one RunLayerLoop call per position.
// `after(p)` runs after position p's call returns Ok (the suite reads its path counters there). Returns
// the first non-Ok status (the stream then stops there).
template <class Emit, class After = NoPositionHook>
superslm::SslmForwardStatus RunFixtureDecode(QkAttentionFixture& f, Emit& emit,
                                             superslm::AttentionCaptureSink* sink = nullptr,
                                             After after = After{}) {
	using superslm::SslmForwardStatus;
	f.layer.attention_capture_sink = sink;
	std::vector<uint8_t> workspace(kWorkspaceBytes, 0);
	std::vector<int8_t> hidden(kHidden);
	superslm::SequenceLayerState seq;
	seq.hidden_codes = hidden.data();
	SslmForwardStatus st = SslmForwardStatus::Ok;
	for (size_t p = 0; p < kPositions && st == SslmForwardStatus::Ok; ++p) {
		for (size_t i = 0; i < kHidden; ++i) hidden[i] = f.hidden_in[p * kHidden + i];
		seq.hidden_scale = f.scale_in[p];
		seq.layer_index = 0;
		st = superslm::RunLayerLoop(seq, &f.layer, 1, 1, kHidden, kHeadDim, kKvHeads, kInter, kCap,
		                            f.view.rope_tables, workspace.data(), workspace.size(), {}, p, nullptr,
		                            kHeads * kHeadDim);
		if (st != SslmForwardStatus::Ok) break;
		EmitPosition(emit, hidden.data(), seq.hidden_scale);
		after(p);
	}
	f.layer.attention_capture_sink = nullptr;
	if (st == SslmForwardStatus::Ok) EmitWorkspace(emit, workspace);
	return st;
}

// Run (ii): the chunk loop, one RunLayerLoopChunkBatched call over all 24 positions from position 0.
// `sink` is installed on the layer (t2701's chunk-mode configuration) and must never be called.
template <class Emit>
superslm::SslmForwardStatus RunFixtureChunk(QkAttentionFixture& f, Emit& emit,
                                            superslm::AttentionCaptureSink* sink = nullptr) {
	using superslm::SslmForwardStatus;
	f.layer.attention_capture_sink = sink;
	std::vector<uint8_t> workspace(kWorkspaceBytes, 0);
	std::vector<int8_t> chunk(f.hidden_in);
	std::vector<superslm::CarriedScale> scales(f.scale_in);
	uint64_t saturation = 0;
	const SslmForwardStatus st = superslm::RunLayerLoopChunkBatched(
	    chunk.data(), scales.data(), kPositions, &f.layer, 1, kHidden, kHeadDim, kKvHeads, kInter, kCap, 0,
	    f.view.rope_tables, workspace.data(), workspace.size(), /*option_g_fused_k_landing=*/false, &saturation,
	    {}, nullptr, kHeads * kHeadDim);
	f.layer.attention_capture_sink = nullptr;
	if (st != SslmForwardStatus::Ok) return st;
	for (size_t p = 0; p < kPositions; ++p) EmitPosition(emit, chunk.data() + p * kHidden, scales[p]);
	EmitWorkspace(emit, workspace);
	return st;
}

}  // namespace superslm_qk_fixture

#endif  // SUPERSLM_TESTS_SUPPORT_QK_ATTENTION_FIXTURE_H
