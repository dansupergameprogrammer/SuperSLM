// Multi-row GPU plan (rev 1.3.2), MR1 gate, the widened leg of the Wine one-layer smoke (rulings-mr1.md, ruling 1:
// ONE widened layer, layer 0 of pkv_def and pkv_qk, H rows at each variant 2/4/8, per-row output byte-equal to the
// one-row reference). Built only with -DSUPERSLM_MULTIROW_WIDE; calls the builder's test-only entry
// sslm_gpu_test_only_mr1_wide_one_layer_smoke (src/gpu/multirow_wide.h). Independent of the builder's runner
// (tools/multirow_wide_smoke.*): its own cases, its own token stream origin, its own expectations.
//
// Per case (fixture x (H, M) x p0), on a fresh context with two dedicated sequences fed the same p0 prefix tokens:
//   E1 the entry returns SSLM_OK and echoes M, H and p0;
//   E2 every real row's widened SeqState equals the one-row state after layer 0 (rows_equal == M);
//   E3 every padded row leaves as it entered (padded_rows_inert == H - M);
//   E4 the K/V comparison covered exactly M rows x KV heads x {K, V} x head_dim bytes, computed here from the
//      artifact's config, not from the report; none differ, and no byte outside those rows changed;
//   T  the entry's one-row half is the reference: `ref`'s SLM5 after the entry is byte-equal to this suite's own
//      twin capture (mr1_twin_capture.cpp, public verbs only) of the same stream at the same p0 -- the last row's
//      last-layer blob. Its mirror holds every row's layer-0 K/V rows, which are what E4 compared against.
// Cases: H/M in {2/1, 2/2, 4/3, 4/4, 8/5, 8/8} x p0 in {0, 3, 7}, plus MX1-host cases (code review N1: attention
// groups sized for row 0) at H/M {4/4, 8/5, 8/8} and p0 = 256 / NQH - 2, where row 0 fits one 256-thread group and
// a later real row does not.
// Every case runs twice: once on a dedicated `wide` (zero K/V background) and once with a CANARY background
// (code review S1: a padded row that writes K/V writes code 0, which a zero background cannot show). The canary
// `wide` is a budgeted sequence over a one-sequence pool whose budget (p0 + H rounded up to 16 positions) is first filled
// with a different token stream, then reset -- a pooled reset returns the pages to the sequence's own reserve and
// fills nothing (gpu_1p0.h, sslm_gpu_seq_reset) -- and then fed the case's prefix. Every position from p0 to the
// budget therefore holds the canary stream's non-zero K/V rows in the device pool and in the pool's mirror alike,
// so E4's stray count sees any write a padded row (or a real row, off its own row) makes there.
// And one must-reject per fixture (the entry's comparison can fail): `wide` is fed a different prefix token, so its
// layer-0 attention reads different K/V; the report must show a difference (rows_equal < M or K/V bytes differ).
//
// Not observable through the entry, so not graded here: the widened rows' layer-0 SeqState bytes themselves
// against this suite's per-row layer-0 blobs (the entry compares them to its own one-row snapshot and returns
// only counts). The link to the twin is T plus E2. Exposing the per-row bytes would close it (MR1 record).
//
// Usage: mr1_wide_smoke.exe --model=PATH [--shaders=DIR]     exit 0 when every case and the must-reject pass.
#define MR1_TWIN_NO_MAIN
#include "mr1_twin_capture.cpp"

#include "multirow_wide.h"

#if !defined(SUPERSLM_MULTIROW_WIDE)
#error mr1_wide_smoke.cpp needs -DSUPERSLM_MULTIROW_WIDE (the test-only widened entry)
#endif

namespace {

struct Env {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuSequenceHandle* ref = nullptr;
	SslmGpuSequenceHandle* wide = nullptr;
	SslmGpuKvPoolHandle* pool = nullptr;
	~Env() {
		if (ref) sslm_gpu_seq_release(ctx, ref);
		if (wide) sslm_gpu_seq_release(ctx, wide);
		if (pool) sslm_gpu_kv_pool_destroy(ctx, pool);
		if (model) sslm_gpu_model_unmap(ctx, model);
		if (ctx) sslm_gpu_context_destroy(ctx);
	}
};

bool FeedFull(SslmGpuContext* ctx, SslmGpuSequenceHandle* s, int32_t tok, uint32_t full_budget) {
	S st = sslm_gpu_seq_embed_token(ctx, s, tok);
	if (st != S::SSLM_OK) return Fail("prefix embed", st);
	if ((st = sslm_decode_step_gpu(ctx, s, nullptr, full_budget)) != S::SSLM_OK) return Fail("prefix decode", st);
	return Drain(ctx, s);
}

// Canary background: positions [0, budget) of the canary `wide` hold this stream's rows before the prefix is fed,
// budget = p0 + H rounded up to 16, so every padded row's position is on canary.
constexpr int kCanaryStreamOrigin = 977;
// Context capacity of `ref` and the dedicated `wide`: the twin capture's (T compares their SLM5 bytes).
constexpr int32_t kSeqCap = kTwinSeqCap;

// Builds the case's pair; `wide_prefix_bias` != 0 changes the wide sequence's last prefix token (must-reject);
// `canary` makes `wide` the canary-background pooled sequence described above.
bool Setup(const Args& a, const superslm::SslmModelView& view, int p0, uint32_t h, int wide_prefix_bias, bool canary,
           Env* e) {
	const int32_t canary_budget = static_cast<int32_t>((p0 + static_cast<int>(h) + 15) / 16 * 16);
	GpuContextConfig cfg{};
	cfg.shader_dir = a.shaders.empty() ? nullptr : a.shaders.c_str();
	S st = sslm_gpu_context_create(cfg, &e->ctx);
	if (st != S::SSLM_OK) return Fail("context", st);
	if ((st = sslm_gpu_model_map(e->ctx, &view, GpuResidencyConfig{}, &e->model)) != S::SSLM_OK) return Fail("map", st);
	if ((st = sslm_gpu_seq_create(e->ctx, e->model, kSeqCap, &e->ref)) != S::SSLM_OK) return Fail("ref create", st);
	const auto& c = view.config;
	if (!canary) {
		if ((st = sslm_gpu_seq_create(e->ctx, e->model, kSeqCap, &e->wide)) != S::SSLM_OK) return Fail("wide create", st);
	} else {
		if ((st = sslm_gpu_kv_pool_create(e->ctx, e->model, 32, &e->pool)) != S::SSLM_OK) return Fail("pool create", st);
		if ((st = sslm_gpu_seq_create_budgeted(e->ctx, e->model, e->pool, canary_budget, &e->wide)) != S::SSLM_OK)
			return Fail("canary wide create", st);
	}
	bool qk = false;
	for (uint32_t l = 0; l < c.num_hidden_layers; ++l)
		if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) qk = true;
	const uint32_t full = superslm_gpu::DispatchesPerLayer(qk) * c.num_hidden_layers;
	const int32_t vocab = static_cast<int32_t>(c.vocab_size);
	if (canary) {
		for (int i = 0; i < canary_budget; ++i)
			if (!FeedFull(e->ctx, e->wide, TokenAt(kCanaryStreamOrigin + i, vocab), full)) return false;
		if ((st = sslm_gpu_seq_reset(e->ctx, e->wide)) != S::SSLM_OK) return Fail("canary reset", st);
	}
	for (int i = 0; i < p0; ++i) {
		if (!FeedFull(e->ctx, e->ref, TokenAt(i, vocab), full)) return false;
		const int32_t wt = (wide_prefix_bias && i == p0 - 1) ? (TokenAt(i, vocab) + wide_prefix_bias) % vocab : TokenAt(i, vocab);
		if (!FeedFull(e->ctx, e->wide, wt, full)) return false;
	}
	return true;
}

int g_checks = 0, g_fail = 0;
void Check(bool ok, const char* tag, const std::string& what) {
	++g_checks;
	if (!ok) {
		++g_fail;
		std::printf("FAIL %s: %s\n", tag, what.c_str());
	}
}

}  // namespace

int main(int argc, char** argv) {
	Args a;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		if (s.rfind("--model=", 0) == 0) a.model = s.substr(8);
		else if (s.rfind("--shaders=", 0) == 0) a.shaders = s.substr(10);
		else { std::printf("usage: mr1_wide_smoke --model=PATH [--shaders=DIR]\n"); return 2; }
	}
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	std::string err;
	if (a.model.empty() || !ReadFileBytes(a.model, &bytes) ||
	    superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
		std::printf("FAIL cannot load model '%s' %s\n", a.model.c_str(), err.c_str());
		return 1;
	}
	const auto& c = view.config;
	const int32_t vocab = static_cast<int32_t>(c.vocab_size);
	const uint64_t kv_row_bytes_per_row = uint64_t{c.num_key_value_heads} * 2u * c.head_dim *
	                                      (c.kv_precision == superslm::SslmKvPrecision::Int16 ? 2u : 1u);
	struct Case { uint32_t h, m; int p0; };
	std::vector<Case> cases;
	for (const auto& hm : {std::pair{2u, 1u}, {2u, 2u}, {4u, 3u}, {4u, 4u}, {8u, 5u}, {8u, 8u}})
		for (int p0 : {0, 3, 7}) cases.push_back({hm.first, hm.second, p0});
	// MX1's host half (code review N1): attention groups sized for row 0's width instead of the widest row's.
	// A group is 256 threads over NQH x width items, so the cases need NQH * (p0 + 1) <= 256 < NQH * (p0 + M):
	// row 0 fits one group and a later real row does not. p0 = 256 / NQH - 2 gives that for every M >= 3.
	const int nqh = static_cast<int>(c.num_attention_heads);
	const int p0_mx1 = 256 / nqh - 2;
	if (p0_mx1 + 8 + 1 > kSeqCap) {
		std::printf("FAIL MX1-host cases need p0 %d + 8 rows within the capacity %d\n", p0_mx1, kSeqCap);
		return 1;
	}
	for (const auto& hm : {std::pair{4u, 4u}, {8u, 5u}, {8u, 8u}}) cases.push_back({hm.first, hm.second, p0_mx1});
	for (const bool canary : {false, true}) {
	for (const Case& cs : cases) {
		const struct { uint32_t h, m; } hm{cs.h, cs.m};
		const int p0 = cs.p0;
		{
			char tag[64];
			std::snprintf(tag, sizeof tag, "H%u M%u p0=%d %s", hm.h, hm.m, p0, canary ? "canary" : "zero-bg");
			Env e;
			if (!Setup(a, view, p0, hm.h, 0, canary, &e)) { Check(false, tag, "setup"); continue; }
			std::vector<int32_t> toks;
			for (uint32_t r = 0; r < hm.m; ++r) toks.push_back(TokenAt(p0 + static_cast<int>(r), vocab));
			SslmGpuMr1WideSmokeReport rep{};
			const S st = sslm_gpu_test_only_mr1_wide_one_layer_smoke(e.ctx, e.ref, e.wide, nullptr, toks.data(), hm.m,
			                                                         hm.h, &rep);
			Check(st == S::SSLM_OK, tag, "E1 status " + std::to_string(static_cast<int>(st)) + " " + rep.detail);
			if (st != S::SSLM_OK) continue;
			Check(rep.real_rows == hm.m && rep.variant_h == hm.h && rep.p0 == p0, tag, "E1 report echo");
			Check(rep.rows_equal == hm.m, tag, "E2 rows_equal " + std::to_string(rep.rows_equal) + " " + rep.detail);
			Check(rep.padded_rows_inert == hm.h - hm.m, tag, "E3 padded_rows_inert " + std::to_string(rep.padded_rows_inert));
			Check(rep.kv_row_bytes_compared == kv_row_bytes_per_row * hm.m, tag,
			      "E4 compared " + std::to_string(rep.kv_row_bytes_compared) + ", expected " +
			          std::to_string(kv_row_bytes_per_row * hm.m));
			Check(rep.kv_row_bytes_differing == 0 && rep.kv_stray_bytes == 0, tag,
			      "E4 K/V differing " + std::to_string(rep.kv_row_bytes_differing) + " stray " + std::to_string(rep.kv_stray_bytes));
			// T: the entry's one-row half against this suite's twin capture of the same stream.
			std::vector<uint8_t> got;
			if (!Save(e.ctx, e.ref, &got)) { Check(false, tag, "T save ref"); continue; }
			Args ta = a;
			ta.rows = static_cast<int>(hm.m);
			ta.start = p0;
			std::vector<Blob> twin;
			if (!Capture(ta, view, &twin) || twin.empty()) { Check(false, tag, "T twin capture"); continue; }
			Check(twin.back().bytes == got, tag, "T ref's SLM5 after the entry differs from the twin capture's last blob");
			std::printf("ok   %s\n", tag);
		}
	}
	}
	// Must-reject: different K/V history under the widened rows must show as a difference.
	{
		Env e;
		const char* tag = "must-reject (wide prefix differs)";
		if (!Setup(a, view, 3, 4, 1, false, &e)) Check(false, tag, "setup");
		else {
			std::vector<int32_t> toks;
			for (int r = 0; r < 4; ++r) toks.push_back(TokenAt(3 + r, vocab));
			SslmGpuMr1WideSmokeReport rep{};
			const S st = sslm_gpu_test_only_mr1_wide_one_layer_smoke(e.ctx, e.ref, e.wide, nullptr, toks.data(), 4, 4, &rep);
			Check(st == S::SSLM_OK, tag, "status " + std::to_string(static_cast<int>(st)));
			Check(rep.rows_equal < 4 || rep.kv_row_bytes_differing > 0, tag,
			      "the entry reported agreement although wide's K/V history differs (its comparison cannot fail)");
			if (st == S::SSLM_OK) std::printf("ok   %s: rows_equal %u, K/V differing %llu\n", tag, rep.rows_equal,
			                                  static_cast<unsigned long long>(rep.kv_row_bytes_differing));
		}
	}
	std::printf("%s widened one-layer smoke: %d checks, %d failures\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
	return g_fail ? 1 : 0;
}
