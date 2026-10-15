// Paged-KV plan (rev 16.2), step G1, group g1b: the cell parts this group owns that are bracketed [G3].
//   2.6/G3   a pool of the largest admissible page count plus one -> SSLM_GPU_KV_POOL_TOO_LARGE
//   2.11/G3  the budgeted create given a pool of another context or model, and pool create given a
//            model of another context -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH
//   5.7/G3   GPU equivalents of 5.1-5.3, and of 5.4 for create, prefill and decode
// 11.4's G3 part is g1b_g3_pool_ceiling.cpp (it needs the fault-injection build for its seam legs).
//
// RED NOW: link-red on sslm_gpu_kv_pool_create, sslm_gpu_kv_pool_destroy and
// sslm_gpu_seq_create_budgeted (declared at 63e7b0e, defined at G3).
// GREEN AT G3: every check below passes on both GPUs with the debug layer clean.
#include "g1b_helpers.h"

namespace g1b {
namespace {

constexpr uint32_t kForeignPool = 40;  // every 2.11 pool: probed whole, so 40 pages free

// ---- 2.6 [G3] ------------------------------------------------------------------------------------
// "A pool of the largest admissible page count plus one page -> SSLM_GPU_KV_POOL_TOO_LARGE." The
// page count is the test's own §3.1/§4 item 3 arithmetic: floor(2^32 / page_bytes) + 1 (699,051 at
// the tiny fixture's 6,144-byte page), never an engine query. The out-handle is null (gpu_1p0.h's
// page-pool block). The ceiling itself admitted is 11.4's verdict, not this cell's.
// Green: TOO_LARGE with a null out-handle at ceiling + 1 and at UINT32_MAX; a 2-page pool admitted.
void Cell26G3() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	const uint32_t ceil = g.CeilingPages();
	G1B_CHECK_MSG(uint64_t{ceil} * g.page_bytes <= (uint64_t{1} << 32) && uint64_t{ceil + 1} * g.page_bytes > (uint64_t{1} << 32),
	              "2.6: the test's own ceiling arithmetic (%u pages of %llu bytes)", ceil,
	              static_cast<unsigned long long>(g.page_bytes));
	for (uint32_t pages : {ceil + 1, 0xFFFFFFFFu}) {
		SslmGpuKvPoolHandle* p = reinterpret_cast<SslmGpuKvPoolHandle*>(uintptr_t{0x1});
		const S st = sslm_gpu_kv_pool_create(e.ctx1, e.m1, pages, &p);
		G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_POOL_TOO_LARGE, "2.6/G3: pool of %u pages", pages);
		G1B_CHECK_MSG(p == nullptr, "2.6/G3: pool of %u pages: out-handle not nulled", pages);
		if (st == S::SSLM_OK && p) sslm_gpu_kv_pool_destroy(e.ctx1, p);
	}
	// Control: a small pool is admitted, so the refusals above are not a verb that refuses everything.
	Scene sc;
	NewPool(sc, e.ctx1, e.m1, 2);
}

// ---- 2.11 [G3] -----------------------------------------------------------------------------------
// GPU handles of another context or model, swept as a class. Each refusal asserts
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH, a null out-handle and, after sslm_gpu_ready, the GPU fill probe
// unchanged on every pool the case involves (here the foreign pool, which the call names; it is
// fresh, so "unchanged" is all kForeignPool pages). The mutant that draws before it checks leaves the
// foreign pool one reservation (R(64) = 5 pages) short, and the probe's admission leg fails.
// The pool-create case states pool create's own operands only, the status and the null out-handle:
// there is no pool to probe and nothing to draw (the carrier's CLASS-G row).
// Each cross-handle case names the argument that carries the foreign handle.
void Cell211G3() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	struct Case {
		const char* name;
		SslmGpuContext* pool_ctx;
		SslmGpuModelHandle* pool_model;
	} cases[] = {{"create_budgeted, `pool` of another context", e.ctx2, e.m2},
	             {"create_budgeted, `pool` of another model", e.ctx1, e.m1b}};
	for (const Case& c : cases) {
		auto build = [&](Scene& sc, int) -> std::vector<ProbeTarget> {
			SslmGpuKvPoolHandle* foreign = NewPool(sc, c.pool_ctx, c.pool_model, kForeignPool);
			if (!foreign) return {};
			SslmGpuSequenceHandle* out = reinterpret_cast<SslmGpuSequenceHandle*>(uintptr_t{0x1});
			const S st = sslm_gpu_seq_create_budgeted(e.ctx1, e.m1, foreign, 64, &out);
			G1B_CHECK_STATUS(st, S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.11/G3 %s", c.name);
			G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "2.11/G3 %s: out-handle not nulled", c.name);
			if (st == S::SSLM_OK && out) sc.seqs.push_back({e.ctx1, out});
			return {{c.pool_ctx, c.pool_model, foreign}};
		};
		char label[160];
		std::snprintf(label, sizeof(label), "2.11/G3 %s (foreign pool)", c.name);
		GpuFillProbe(g, build, 0, kForeignPool, label);
	}
	// sslm_gpu_kv_pool_create given a `model` of another context.
	SslmGpuKvPoolHandle* p = reinterpret_cast<SslmGpuKvPoolHandle*>(uintptr_t{0x1});
	const S st = sslm_gpu_kv_pool_create(e.ctx1, e.m2, 8, &p);
	G1B_CHECK_STATUS(st, S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.11/G3 pool_create, `model` of another context");
	G1B_CHECK_MSG(p == nullptr || st == S::SSLM_OK, "2.11/G3 pool_create, `model` of another context: out-handle not nulled");
	if (st == S::SSLM_OK && p) sslm_gpu_kv_pool_destroy(e.ctx1, p);
}

// ---- 5.7 [G3] ------------------------------------------------------------------------------------
// GPU equivalents of 5.1-5.3, and of 5.4 for create, prefill and decode: budget, exhaustion and the
// token count of 5.2. Constructions follow tests/paged-kv/c5_dim5_failure.cpp's 5.1-5.4 (the CPU
// cells), on the GPU verbs, with the operands that exist at G3: statuses, tokens against an unsaved
// twin, and the GPU fill probe. (The 'SLM6' re-save that 5.1's CPU cell also reads is G5's; the
// adopter cases of 5.2 and 5.3 need GPU adopt, G4, and are not GPU cells of 5.7.)

// 5.1: a budget-100 holder given 150 tokens is refused with SSLM_GPU_KV_BUDGET_EXCEEDED, in one call
// and across chunked calls, and its state is the 100-token state: its one ready token equals the
// 100-token twin's and it is refused after it (1 + (limit - c) = 1 token, §3.4). A further one-token
// prefill is refused too. Reset, it prefills 100 again (resumable). A bad token id before the limit
// still reports SSLM_TOKEN_ID_OUT_OF_RANGE; past the limit the budget wins.
void Part51(Env& e) {
	const Geometry& g = e.geo;
	Scene sc;
	SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, 64);
	if (!pool) return;
	const std::vector<int32_t> t150 = Stream(1, 150, g.vocab);
	const std::vector<int32_t> t100(t150.begin(), t150.begin() + 100);
	std::vector<int32_t> twin_tok;
	{
		SslmGpuSequenceHandle* t = NewBudgetSeq(sc, e.ctx1, e.m1, pool, 100);
		if (!t) return;
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, t, t100), S::SSLM_OK, "5.1 twin prefill of 100");
		S r = S::SSLM_OK;
		G1B_CHECK_MSG(TokensUntilRefusal(g, e.ctx1, t, &r, 8192, &twin_tok) == 1, "5.1 twin: token count at the limit");
		G1B_CHECK_STATUS(r, S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.1 twin: refusal at the limit");
		sc.Release(t);
	}
	for (int chunked = 0; chunked <= 1; ++chunked) {
		SslmGpuSequenceHandle* s = NewBudgetSeq(sc, e.ctx1, e.m1, pool, 100);
		if (!s) continue;
		S st = S::SSLM_OK;
		if (chunked) {
			// 64, then 36 of the next 64, then nothing of the last 22.
			for (size_t off = 0; off < t150.size() && st == S::SSLM_OK; off += 64) {
				const std::vector<int32_t> chunk(t150.begin() + off, t150.begin() + std::min<size_t>(off + 64, t150.size()));
				st = PrefillPrompt(g, e.ctx1, s, chunk);
			}
		} else {
			st = PrefillPrompt(g, e.ctx1, s, t150);
		}
		// kills: the limit headroom missing from the admission pre-scan (150 consumed), or an
		// all-or-nothing refusal (0 consumed: the ready token below differs from the twin's).
		G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.1 (%s): prefill of 150 under budget 100", chunked ? "chunked" : "one call");
		const std::vector<int32_t> one(t150.begin() + 100, t150.begin() + 101);
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, s, one), S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.1 (%s): a further token at the limit",
		                 chunked ? "chunked" : "one call");
		std::vector<int32_t> tok;
		S r = S::SSLM_OK;
		G1B_CHECK_MSG(TokensUntilRefusal(g, e.ctx1, s, &r, 8192, &tok) == 1, "5.1 (%s): token count at the limit", chunked ? "chunked" : "one call");
		G1B_CHECK_STATUS(r, S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.1 (%s): refusal after the ready token", chunked ? "chunked" : "one call");
		G1B_CHECK_MSG(tok == twin_tok, "5.1 (%s): the stopped holder's ready token differs from the 100-token twin's",
		              chunked ? "chunked" : "one call");
		G1B_CHECK_STATUS(sslm_gpu_seq_reset(e.ctx1, s), S::SSLM_OK, "5.1: reset");  // resumable: the budget runs again
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, s, t100), S::SSLM_OK, "5.1 (%s): prefill of 100 after reset", chunked ? "chunked" : "one call");
		sc.Release(s);
	}
	struct {
		int at;
		S want;
	} ids[] = {{50, S::SSLM_TOKEN_ID_OUT_OF_RANGE}, {0, S::SSLM_TOKEN_ID_OUT_OF_RANGE}, {120, S::SSLM_GPU_KV_BUDGET_EXCEEDED}};
	for (auto& c : ids) {
		std::vector<int32_t> t = t150;
		t[static_cast<size_t>(c.at)] = g.vocab;  // one past the vocabulary
		SslmGpuSequenceHandle* s = NewBudgetSeq(sc, e.ctx1, e.m1, pool, 100);
		if (!s) continue;
		// kills: the budget check given precedence over the token-id check (or the reverse past the limit).
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, s, t), c.want, "5.1 bad id at %d", c.at);
		sc.Release(s);
	}
}

// 5.2: decode at the limit. A holder resting ready at context_length c emits exactly 1 + (limit - c)
// tokens (§3.4) and is then refused SSLM_GPU_KV_BUDGET_EXCEEDED: c = 25 under budget 40 emits 16; a
// holder resting at c = limit emits 1. In a batch the refusal has today's cap check's batch
// behaviour: X, a budget holder at its limit, and X', a dedicated holder at its cap (40), each with
// its next token embedded, are batched with Y in both orders; X gets the budget status, and the
// pattern (embed status of X, the batch call's success, Y's status and Y's next two tokens) equals
// X''s at the cap.
void Part52(Env& e) {
	const Geometry& g = e.geo;
	Scene sc;
	SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, 64);
	if (!pool) return;
	struct {
		const char* name;
		int32_t budget, prefill;
	} counts[] = {{"c 25, budget 40", 40, 25}, {"c = limit", 25, 25}};
	for (auto& c : counts) {
		SslmGpuSequenceHandle* s = NewBudgetSeq(sc, e.ctx1, e.m1, pool, c.budget);
		if (!s || PrefillPrompt(g, e.ctx1, s, Stream(1, c.prefill, g.vocab)) != S::SSLM_OK) {
			G1B_CHECK_MSG(false, "5.2 %s: setup", c.name);
			continue;
		}
		S refusal = S::SSLM_OK;
		// kills: the budget check on the ready path (one token fewer) or after the embed (a row past
		// the limit and one token more).
		const int64_t n = TokensUntilRefusal(g, e.ctx1, s, &refusal);
		G1B_CHECK_MSG(n == 1 + (c.budget - c.prefill), "5.2 %s: %lld tokens, want %d", c.name, static_cast<long long>(n),
		              1 + (c.budget - c.prefill));
		G1B_CHECK_STATUS(refusal, S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.2 %s: refusal", c.name);
		sc.Release(s);
	}
	struct Pattern {
		S embed_x = S::SSLM_DEVICE_LOST, batch = S::SSLM_DEVICE_LOST, out_x = S::SSLM_DEVICE_LOST, out_y = S::SSLM_DEVICE_LOST;
		std::vector<int32_t> y_tokens;
	};
	// X rests at its limit (or cap) with its last token produced and not yet embedded.
	auto at_limit = [&](SslmGpuSequenceHandle* x, int32_t* pending) {
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, x, Stream(1, 25, g.vocab)), S::SSLM_OK, "5.2 batch: X prefill");
		const std::vector<int32_t> t = DecodeN(g, e.ctx1, x, 16);
		G1B_CHECK_MSG(t.size() == 16, "5.2 batch: X decoded %zu of 16 tokens to its limit", t.size());
		*pending = t.empty() ? 1 : t.back();
	};
	auto run = [&](SslmGpuSequenceHandle* x, int32_t x_pending, bool x_first) {
		Pattern r;
		SslmGpuSequenceHandle* y = NewBudgetSeq(sc, e.ctx1, e.m1, pool, 64);
		if (!y) return r;
		PrefillPrompt(g, e.ctx1, y, Stream(9, 10, g.vocab));
		int32_t ty = -1;
		DecodeOne(g, e.ctx1, y, 1, &ty);  // the ready token; Y now waits for it to be embedded
		r.embed_x = sslm_gpu_seq_embed_token(e.ctx1, x, x_pending);
		sslm_gpu_seq_embed_token(e.ctx1, y, ty);
		SslmGpuSequenceHandle* batch[2] = {x_first ? x : y, x_first ? y : x};
		const SslmGpuAdapterHandle* adapters[2] = {nullptr, nullptr};
		S outs[2] = {S::SSLM_DEVICE_LOST, S::SSLM_DEVICE_LOST};
		r.batch = sslm_decode_step_batch_gpu(e.ctx1, batch, adapters, 2, g.decode_budget, outs);
		r.out_x = outs[x_first ? 0 : 1];
		r.out_y = outs[x_first ? 1 : 0];
		Scene::DrainOne(e.ctx1, x);
		Scene::DrainOne(e.ctx1, y);
		int32_t tok = -1;
		if (SslmGpuSeqFinishTokenForG5Bridge(e.ctx1, y, &tok) == S::SSLM_OK) r.y_tokens.push_back(tok);
		const std::vector<int32_t> more = DecodeN(g, e.ctx1, y, 1);
		r.y_tokens.insert(r.y_tokens.end(), more.begin(), more.end());
		sc.Release(y);
		return r;
	};
	for (bool x_first : {false, true}) {
		Scene xs;
		SslmGpuSequenceHandle* xb = NewBudgetSeq(xs, e.ctx1, e.m1, pool, 40);
		SslmGpuSequenceHandle* xl = NewDedicatedSeq(xs, e.ctx1, e.m1, 40);
		if (!xb || !xl) continue;
		int32_t pb = 1, pl = 1;
		at_limit(xb, &pb);
		at_limit(xl, &pl);
		const Pattern b = run(xb, pb, x_first), l = run(xl, pl, x_first);
		const char* order = x_first ? "X, Y" : "Y, X";
		G1B_CHECK_STATUS(b.out_x, S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.2 batch [%s]: X at its limit", order);
		G1B_CHECK_MSG(l.out_x != S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.2 batch [%s]: the dedicated holder at its cap reported the budget status", order);
		// kills: a budget refusal with its own batch semantics (aborting the batch, or skipping Y).
		G1B_CHECK_MSG(b.embed_x == l.embed_x && (b.batch == S::SSLM_OK) == (l.batch == S::SSLM_OK) && b.out_y == l.out_y &&
		                  b.y_tokens == l.y_tokens && b.y_tokens.size() == 2,
		              "5.2 batch [%s]: the budget refusal's batch behaviour differs from the cap's (embed %s/%s, batch %s/%s, Y %s/%s, Y tokens %zu/%zu)",
		              order, StatusName(b.embed_x), StatusName(l.embed_x), StatusName(b.batch), StatusName(l.batch),
		              StatusName(b.out_y), StatusName(l.out_y), b.y_tokens.size(), l.y_tokens.size());
	}
}

// 5.3: limit == cap reports today's cap status, never the budget status. A budget-cap holder from
// origin 0, filled to one short of the cap: decode emits 2 tokens (1 + (cap - (cap - 1))) and is then
// refused with the status a dedicated holder of the same cap returns at the same point; a 2-token
// prefill there is refused with the dedicated twin's prefill status. Neither is
// SSLM_GPU_KV_BUDGET_EXCEEDED. (The GPU cap statuses are today's: the plan names no new one, so the
// dedicated twin is the oracle rather than a literal.)
void Part53(Env& e) {
	const Geometry& g = e.geo;
	const int64_t cap = g.cap;
	Scene sc;
	SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, g.R(cap) + 4);
	if (!pool) return;
	const std::vector<int32_t> fill = Stream(5, static_cast<int32_t>(cap - 1), g.vocab);
	const std::vector<int32_t> two = Stream(6, 2, g.vocab);
	for (int via_decode = 0; via_decode <= 1; ++via_decode) {
		SslmGpuSequenceHandle* s = NewBudgetSeq(sc, e.ctx1, e.m1, pool, static_cast<int32_t>(cap));
		SslmGpuSequenceHandle* d = NewDedicatedSeq(sc, e.ctx1, e.m1, cap);
		if (!s || !d) continue;
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, s, fill), S::SSLM_OK, "5.3: fill to cap - 1");
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, d, fill), S::SSLM_OK, "5.3: dedicated twin fill to cap - 1");
		if (via_decode) {
			S rs = S::SSLM_OK, rd = S::SSLM_OK;
			const int64_t ns = TokensUntilRefusal(g, e.ctx1, s, &rs), nd = TokensUntilRefusal(g, e.ctx1, d, &rd);
			G1B_CHECK_MSG(ns == 2 && nd == 2, "5.3 decode: %lld tokens (twin %lld), want 2", static_cast<long long>(ns),
			              static_cast<long long>(nd));
			// kills: the budget check ahead of the cap check at limit == cap.
			G1B_CHECK_MSG(rs != S::SSLM_GPU_KV_BUDGET_EXCEEDED && rs == rd, "5.3 decode: refusal %s, dedicated twin %s",
			              StatusName(rs), StatusName(rd));
		} else {
			const S ss = PrefillPrompt(g, e.ctx1, s, two), sd = PrefillPrompt(g, e.ctx1, d, two);
			G1B_CHECK_MSG(ss != S::SSLM_GPU_KV_BUDGET_EXCEEDED && ss == sd && ss != S::SSLM_OK,
			              "5.3 prefill past the cap: %s, dedicated twin %s", StatusName(ss), StatusName(sd));
		}
		sc.Release(s);
		sc.Release(d);
	}
}

// 5.4 for create, prefill and decode. A pool of 10 pages holds A and B (budget 64, R = 5 each), so it
// is exhausted: a third budgeted create is refused SSLM_GPU_KV_POOL_EXHAUSTED with a null out-handle.
// Prefill and decode map pages from the holder's own reserve and never touch the pool (§3.4), so on
// the exhausted pool A still prefills 30 and decodes its whole budget, 1 + (64 - 30) = 35 tokens,
// before the budget refusal. Nothing leaked: with B released, the GPU fill probe reads exactly 5 free
// (B's reservation), after the refused create and A's whole run.
void Part54(Env& e) {
	const Geometry& g = e.geo;
	const uint32_t P = 2 * g.R(64);
	auto build = [&](Scene& sc, int leg) -> std::vector<ProbeTarget> {
		SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, P);
		SslmGpuSequenceHandle* a = pool ? NewBudgetSeq(sc, e.ctx1, e.m1, pool, 64) : nullptr;
		SslmGpuSequenceHandle* b = pool ? NewBudgetSeq(sc, e.ctx1, e.m1, pool, 64) : nullptr;
		if (!a || !b) return {};
		SslmGpuSequenceHandle* c = reinterpret_cast<SslmGpuSequenceHandle*>(uintptr_t{0x1});
		const S st = sslm_gpu_seq_create_budgeted(e.ctx1, e.m1, pool, 16, &c);  // R(16) = 2 > 0 free
		G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_POOL_EXHAUSTED, "5.4 (GPU) create against the exhausted pool");
		G1B_CHECK_MSG(c == nullptr || st == S::SSLM_OK, "5.4 (GPU) create: out-handle not nulled");
		if (st == S::SSLM_OK && c) sc.seqs.push_back({e.ctx1, c});
		G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, a, Stream(3, 30, g.vocab)), S::SSLM_OK, "5.4 (GPU) prefill on the exhausted pool");
		S refusal = S::SSLM_OK;
		const int64_t n = TokensUntilRefusal(g, e.ctx1, a, &refusal);
		G1B_CHECK_MSG(n == 35, "5.4 (GPU) decode on the exhausted pool: %lld tokens, want 35 (leg %d)", static_cast<long long>(n), leg);
		G1B_CHECK_STATUS(refusal, S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.4 (GPU) decode's refusal at the limit");
		sc.Release(b);
		return {{e.ctx1, e.m1, pool}};
	};
	GpuFillProbe(g, build, 0, g.R(64), "5.4 (GPU): nothing leaked by the refused create, the prefill or the decode");
}

void Cell57G3() {
	Env& e = GetEnv();
	if (!e.ok) return;
	Part51(e);
	Part52(e);
	Part53(e);
	Part54(e);
}

}  // namespace

G1B_CELL("2.6/G3", "G3", Cell26G3);
G1B_CELL("2.11/G3", "G3", Cell211G3);
G1B_CELL("5.7/G3", "G3", Cell57G3);

}  // namespace g1b
