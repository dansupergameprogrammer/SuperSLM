// T-2791 (Curie) -- plan Sec3.4 row 6 (numerical edges and determinism), row 8 (composition) and
// row 7's "Non-mutating" contract claim (plan Sec2.6 M18).
//
//   D1 (row 6, FEATURE ORACLE) "The read equals RmsNormSite("final_norm") applied by the CPU path to
//      the CPU forward's last-position residual, byte for byte": for each row, the GPU read after
//      reset -> prefill equals the CPU frame. The CPU frame is computed here exactly as the T-2776
//      driver's CPU leg does (the oracle behind T-2780's 51616b69... commission): EmbedEntry then
//      RunLayerLoop per token, over layers marshalled from this file's own parse, then RmsNormSite
//      with the artifact's final_norm gain and constant. It takes no input from the GPU path.
//   C1 (row 8) "Prefill in 1, 3 and whole-document continuation calls, then read, gives identical
//      bytes": the same 12-token row fed three ways; every read equals the CPU frame.
//   N1 (row 7 / M18) "A decode step issued after a read produces the same token and logits as one
//      issued without it": two decode-wrapper steps after the prompt, with and without reads
//      interleaved; the two tokens, the logits of the second step (LogitsSite over the finished
//      residual, computed here) and every bench-visible field of the sequence must match.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden.
//
// Run: cell_determinism_composition.exe [--qwen3=PATH] [--synthetic=PATH] [--g5fixture=PATH]
#include "fixture_common.h"

namespace {

// The CPU oracle, built once per artifact.
struct CpuForward {
	std::vector<superslm_marshal::LayerBacking> backing;
	std::vector<superslm::LayerWeights> layers;
	const int8_t* embed = nullptr;
	superslm::CarriedScale embed_const{};
	std::vector<uint8_t> workspace;
	bool ok = false;

	bool Build(GpuModelFixture& fx) {
		superslm::SslmModelView& v = fx.view;
		superslm_marshal::PreflightScanWscFolds(v);
		backing.resize(fx.layers);
		layers.resize(fx.layers);
		std::string err;
		for (uint32_t l = 0; l < fx.layers; ++l) {
			if (!superslm_marshal::MarshalLayer(v, l, v.config.num_attention_heads, v.config.num_key_value_heads,
			                                    backing[l], layers[l], &err)) {
				std::printf("cpu oracle: marshal layer %u: %s\n", l, err.c_str());
				return false;
			}
		}
		const superslm::SslmTensorView* e = v.weights.Tensor("embed");
		bool cok = true;
		embed_const = superslm_marshal::ReadCarriedScale(v.composition_constants, "embed", &cok);
		if (!e || !cok) return false;
		embed = reinterpret_cast<const int8_t*>(e->data);
		workspace.assign(static_cast<size_t>(v.config.num_hidden_layers) * static_cast<size_t>(v.config.context_cap) *
		                     v.config.num_key_value_heads * v.config.head_dim * 2,
		                 0);
		ok = true;
		return true;
	}

	// The CPU frame of `tokens` from a fresh sequence.
	Frame Run(GpuModelFixture& fx, const std::vector<int32_t>& tokens) {
		superslm::SslmModelView& v = fx.view;
		std::fill(workspace.begin(), workspace.end(), 0);
		std::vector<int8_t> codes(fx.hidden, 0);
		superslm::SequenceLayerState seq{};
		seq.hidden_codes = codes.data();
		const superslm::OptionGKLandingMode k_mode = v.option_g_fused_k_landing ? superslm::OptionGKLandingMode::kFused
		                                                                         : superslm::OptionGKLandingMode::kLegacy;
		Frame f;
		f.status = SSLM_SEQUENCE_REJECTED;
		for (const int32_t t : tokens) {
			superslm::CarriedScale s{};
			if (superslm::EmbedEntry(t, fx.vocab, embed, fx.hidden, embed_const, codes.data(), &s) !=
			    superslm::SslmForwardStatus::Ok) {
				return f;
			}
			seq.hidden_scale = s;
			seq.layer_index = 0;
			const superslm::SslmForwardStatus st = superslm::RunLayerLoop(
			    seq, layers.data(), fx.layers, fx.layers, fx.hidden, v.config.head_dim, v.config.num_key_value_heads,
			    v.config.intermediate_size, v.config.context_cap, v.rope_tables, workspace.data(), workspace.size(),
			    k_mode, {}, 0, &v.trace_hook, static_cast<size_t>(v.config.num_attention_heads) * v.config.head_dim);
			if (st != superslm::SslmForwardStatus::Ok) {
				std::printf("cpu oracle: RunLayerLoop: %s\n", superslm::SslmForwardStatusName(st));
				return f;
			}
		}
		f.codes.assign(fx.hidden, 0);
		superslm::CarriedScale out{};
		if (superslm::RmsNormSite(codes.data(), fx.final_gain.data(), fx.hidden, seq.hidden_scale, fx.final_const,
		                          f.codes.data(), &out, "final_norm") != superslm::SslmForwardStatus::Ok) {
			return f;
		}
		f.status = SSLM_OK;
		f.m = out.m;
		f.e = out.e;
		return f;
	}
};

// The logits digest of the sequence's finished live residual: final_norm then LogitsSite, here.
std::string LogitsDigest(GpuModelFixture& fx, SslmGpuSequenceHandle* s) {
	const Frame fn = OracleFromLive(fx, s);
	const superslm::SslmTensorView* head = fx.view.config.tie_word_embeddings ? fx.view.weights.Tensor("embed")
	                                                                          : fx.view.weights.Tensor("lm_head");
	if (!head || !fn.IsOk()) return "none";
	std::vector<int64_t> wide(static_cast<size_t>(fx.vocab));
	std::vector<int32_t> logits(static_cast<size_t>(fx.vocab));
	if (superslm::LogitsSite(fn.codes.data(), fx.hidden, reinterpret_cast<const int8_t*>(head->data),
	                         static_cast<size_t>(fx.vocab), wide.data(), logits.data()) != superslm::SslmForwardStatus::Ok) {
		return "logits-rejected";
	}
	return Sha256Hex(logits.data(), logits.size() * sizeof(int32_t));
}

void RunOn(const std::string& path, const char* flag, SslmGpuContext* ctx) {
	if (path.empty()) {
		SKIP_MSG("determinism/composition: %s not supplied", flag);
		return;
	}
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) {
		CHECK_MSG(false, "could not open %s", path.c_str());
		return;
	}
	const char* tag = path.c_str();
	CpuForward cpu;
	CHECK_MSG(cpu.Build(fx), "[%s] SETUP: CPU oracle build", tag);
	SslmGpuSequenceHandle* s = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, fx.model, 64, &s) == SSLM_OK && s);
	if (!s || !cpu.ok) { fx.Close(); return; }

	// D1
	const std::vector<std::vector<int32_t>> rows = {fx.TokensP(), fx.Run(16, fx.TokensQ()[0]), {fx.SomeToken()}};
	for (size_t r = 0; r < rows.size(); ++r) {
		const Frame want = cpu.Run(fx, rows[r]);
		CHECK_MSG(want.IsOk(), "[%s] D1 SETUP: CPU frame of row %zu", tag, r);
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		CHECK(Prefill(fx, s, rows[r]) == SSLM_OK);
		const Frame got = ReadVerb(fx, s);
		CHECK_MSG(got.status == SSLM_OK && got.SameFrame(want, fx.hidden),
		          "[%s] D1 row %zu (%zu tokens): GPU read %s differs from the CPU forward's frame in %zu of %zu codes "
		          "(m %lld vs %lld, e %lld vs %lld)",
		          tag, r, rows[r].size(), StatusName(got.status), DiffCodes(got, want, fx.hidden), fx.hidden,
		          static_cast<long long>(got.m), static_cast<long long>(want.m), static_cast<long long>(got.e),
		          static_cast<long long>(want.e));
	}

	// C1
	{
		const std::vector<int32_t> row = fx.Run(12, fx.TokensP()[0]);
		const Frame want = cpu.Run(fx, row);
		for (size_t step : {size_t{1}, size_t{3}, row.size()}) {
			CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
			bool fed = true;
			for (size_t i = 0; i < row.size() && fed; i += step) {
				const std::vector<int32_t> part(row.begin() + i, row.begin() + std::min(row.size(), i + step));
				fed = Prefill(fx, s, part) == SSLM_OK;
			}
			CHECK_MSG(fed, "[%s] C1 SETUP: feeding in %zu-token calls", tag, step);
			const Frame got = ReadVerb(fx, s);
			CHECK_MSG(got.status == SSLM_OK && got.SameFrame(want, fx.hidden),
			          "[%s] C1 %zu-token continuation calls: read %s differs from the CPU frame in %zu codes", tag, step,
			          StatusName(got.status), DiffCodes(got, want, fx.hidden));
		}
	}

	// N1
	{
		auto base = [&]() { return sslm_gpu_seq_reset(ctx, s) == SSLM_OK && Prefill(fx, s, fx.TokensP()) == SSLM_OK; };
		int32_t a1 = -1, a2 = -1, b1 = -1, b2 = -1;
		CHECK(base());
		CHECK(SslmGpuSeqDecodeStepForG5Bridge(ctx, s, fx.OtherToken(), fx.one_layer_budget, &a1) == SSLM_OK);
		CHECK(SslmGpuSeqDecodeStepForG5Bridge(ctx, s, a1, fx.one_layer_budget, &a2) == SSLM_OK);
		const std::string da = LogitsDigest(fx, s);
		const SeqState sa = CaptureState(s);

		CHECK(base());
		const Frame FP = OracleFromLive(fx, s);
		const Frame r0 = ReadVerb(fx, s);
		CHECK(SslmGpuSeqDecodeStepForG5Bridge(ctx, s, fx.OtherToken(), fx.one_layer_budget, &b1) == SSLM_OK);
		const Frame r1 = ReadVerb(fx, s);
		CHECK(SslmGpuSeqDecodeStepForG5Bridge(ctx, s, b1, fx.one_layer_budget, &b2) == SSLM_OK);
		const Frame r2 = ReadVerb(fx, s);
		const std::string db = LogitsDigest(fx, s);
		const SeqState sb = CaptureState(s);
		CHECK_MSG(r0.SameFrame(FP, fx.hidden) && r1.SameFrame(FP, fx.hidden) && r2.SameFrame(FP, fx.hidden),
		          "[%s] N1: the interleaved reads must each return F_P (%s, %s, %s)", tag, StatusName(r0.status),
		          StatusName(r1.status), StatusName(r2.status));
		CHECK_MSG(a1 >= 0 && a1 == b1 && a2 == b2, "[%s] N1 tokens without reads %d,%d; with reads %d,%d", tag, a1, a2, b1, b2);
		CHECK_MSG(da == db && da != "none", "[%s] N1 second-step logits digest differs: %s vs %s", tag, da.c_str(), db.c_str());
		CHECK_MSG(sa == sb, "[%s] N1 the sequence's host-visible state differs with reads interleaved", tag);
		std::printf("    [%s] N1 tokens %d,%d logits %s\n", tag, a1, a2, da.substr(0, 16).c_str());
	}

	sslm_gpu_seq_release(ctx, s);
	fx.Close();
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	RunOn(g_qwen3_path, "--qwen3", ctx);
	RunOn(g_synthetic_path, "--synthetic", ctx);
	RunOn(g_g5_path, "--g5fixture", ctx);
	sslm_gpu_context_destroy(ctx);
	return FinishSuite("cell_determinism_composition");
}
