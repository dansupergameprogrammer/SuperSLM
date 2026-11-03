// Schema fast-decode plan rev 1.4.4 §11.2, cell G12.6 (slice G1-J; rev 1.4.1, audit G-1): partial admission.
// A dedicated sequence at the context cap (256 on F-SFD) and a pooled one at its K/V budget, for every row of
// §5.5's table, h = 0 included. h is set by the prompt's length: a prompt of limit - h tokens (no pending token),
// or limit - h tokens and then one decode (a pending token). The expected values are §7.3's rule, written here
// before the build (XG15):
//   k = the rows asked for (the pending row, if any, plus the chain, <= chunk_budget 64);
//   h = 0        nothing admitted; the limit status; the sequence unchanged;
//   h < k        h rows admitted (the pending row first); the limit status; ready set; layer at full depth; the
//                walk over the admitted forced rows; out_forced_count = the forced rows among them;
//   h >= k       k rows; OK; as above.
// The limit status is SSLM_GPU_CONTEXT_CAP_EXCEEDED (dedicated) or SSLM_GPU_KV_BUDGET_EXCEEDED (pooled, its limit
// below the cap); never SSLM_DEVICE_LOST, and the ready flag is never left clear after admitting rows (XG15).
// After an admission the next decode finishes only: its token equals the CPU's (dedicated: the CPU verb at the same
// headroom; pooled: the dedicated GPU arm at the same h) and the context length does not rise. At h = 0 with a
// pending token, the next decode with that token returns what GPU off returns at that state.
// The dedicated arm also asserts the CPU verb's status class, forced ids and next token. Box only.
//
// MR5 RE-POINT (multi-row GPU plan, ruling 29: the MRX review's M1). 1.18 removes Canonical: NULL options mean Exact
// on the CPU verb (the dedicated arm's comparator) and, from MR5, on the GPU verb. The rule above is Canonical's
// headroom contract, so every expectation below is Exact's, computed without calling the jump verb (as the CPU suite's
// re-pointed cells do: OwnNext / ExactKept / DecodeOnly in tests/test_schema_fast_decode.cpp):
//   no pending token, h <= 1     a no-op (no token in hand, or an empty clipped draft: no row after the pending one):
//                                SSLM_OK, nothing forced, the sequence unchanged; the next decode and the saved state
//                                equal GPU off's at the same state (at h = 0 with a pending token that decode returns
//                                the limit status, as GPU off's does);
//   a pending token, h >= 2      kept = the chain after it, clipped to chunk_budget - 1 and to h - 1 rows, cut at its
//                                first token that differs from Off's own next one (decode-only, same history); SSLM_OK;
//                                out_forced = those kept ids; ready set; layer at full depth; context = limit - h + 1 +
//                                kept; the next decode finishes only and gives Off's next token after them.
// The dedicated arm's CPU verb must give the same (status, ids), and its next token Off's (CPU decode-only). The pooled
// arm equals a dedicated run whose budget clips the draft to the same h - 1 rows. Red until MR5 (the GPU refuses
// Exact); the CPU comparator's checks pass today.
// Run: sfdg_cell_g12_6.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

#include <algorithm>

using namespace sfdg;

namespace {

SslmGpuKvPoolHandle* g_pool = nullptr;
constexpr int32_t kBudgetLimit = 200;  // the pooled arm's limit: origin 0 + budget, below the cap

struct Outcome {
	GS st = GS::SSLM_OK;
	std::vector<int32_t> forced;
	int32_t k = 0;               // the rows this run's verb call asks for: the pending row, if any, plus the chain
	std::vector<int32_t> chain;  // the chain's first k - pending ids, read by the query before the call
	int32_t ready = -1;
	uint32_t layer = 0, walk = 0;
	int64_t ctx = -1;
	int32_t next = -9;
	GS next_st = GS::SSLM_OK;
	int64_t ctx_after_next = -1;
	std::vector<uint8_t> blob;
};

// One GPU run: limit - h prompt tokens (minus one when `pending`, whose decode adds no row), then the verb.
Outcome RunGpu(GRig& rig, const std::vector<int32_t>& base, int32_t limit, int32_t h, bool pending, bool pooled,
               bool call_verb = true, int32_t budget = 64) {
	Outcome o;
	GSeq s;
	if (pooled) {
		const GS st = sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), g_pool, kBudgetLimit, &s.h);
		CHECK_MSG(st == GS::SSLM_OK && s.h, "[G12.6] budgeted create: %s", Name(st));
		if (!s.h) return o;
		rig.seqs.push_back(s.h);
		SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, rig.Index(kPotion));
	} else {
		s = rig.New(kPotion);
		if (!s.h) return o;
	}
	const std::vector<int32_t> prompt = PromptOfLength(base, static_cast<size_t>(limit - h));
	CHECK_MSG(rig.Prompt(s, prompt) == GS::SSLM_OK, "[G12.6] prompt of %zu", prompt.size());
	int32_t t = -1;
	if (pending) {
		// The prompt's ready row is finished first: the start chain is skipped on purpose, so the choice is made at
		// the start state and the pending token's chain is the one measured. (The contract is not under test here.)
		GS st = GS::SSLM_OK;
		t = Token(rig.env, s, &st);
		CHECK_MSG(st == GS::SSLM_OK && t >= 0, "[G12.6] the choice: %s %d", Name(st), t);
	}
	// The chain the verb will be asked for, from this run's own walk state. With a pending token it follows this run's
	// choice, which the prompt's length can change, so a run at one h cannot supply another's expectations.
	const GQuery q = QueryGpu(rig.ctx(), s.h);
	const int32_t pend = pending ? 1 : 0;
	const int32_t take = std::min<int32_t>(static_cast<int32_t>(q.ids.size()), budget - pend);
	o.chain.assign(q.ids.begin(), q.ids.begin() + take);
	o.k = take + pend;
	const SeqObs before = Observe(rig, s.h);
	if (call_verb) {
		const JumpOut j = Jump(rig, s, t, budget);
		o.st = j.st;
		o.forced = j.forced;
		// Exact's no-ops: no pending token, or an empty clipped draft (no chain, or no row after the pending one).
		const bool noop = !pending || take == 0 || h <= 1;
		if (noop) CHECK_MSG(Observe(rig, s.h) == before, "[G12.6] a no-op (h = %d, %s, chain %d) changed the sequence", h,
		                    pending ? "a pending token" : "no pending token", take);
	}
	o.ready = ReadyFlag(rig.ctx(), s.h);
	o.layer = LayerIdx(s.h);
	o.walk = SslmGpuSeqWalkStateForG5Bridge(s.h);
	o.ctx = CtxLen(s.h);
	o.blob = rig.Blob(s.h);
	if (pending && (h == 0 || !call_verb)) s.last = t;
	o.next = Token(rig.env, s, &o.next_st);
	o.ctx_after_next = CtxLen(s.h);
	return o;
}

struct CpuOutcome {
	sslm_status st = SSLM_OK;
	std::vector<int32_t> forced;
	int32_t next = -9;
};
CpuOutcome RunCpu(CRig& c, const std::vector<int32_t>& base, int32_t limit, int32_t h, bool pending) {
	CpuOutcome o;
	sslm_seq s = c.New(kPotion);
	if (!s) return o;
	const std::vector<int32_t> prompt = PromptOfLength(base, static_cast<size_t>(limit - h));
	CHECK_MSG(c.Prefill(s, prompt, SSLM_SPAN_PROMPT) == SSLM_OK, "[G12.6] CPU prompt");
	if (pending) {
		sslm_status st = SSLM_OK;
		const int32_t t = cpu::Token(c.M(), s, &st);
		CHECK_MSG(st == SSLM_OK && t >= 0, "[G12.6] CPU choice: %d %d", static_cast<int>(st), t);
	}
	std::vector<int32_t> out(64, -7);
	int32_t n = -9;
	o.st = sslm_seq_jump_forward(c.M(), s, 64, nullptr, nullptr, nullptr, out.data(), 64, &n);
	out.resize(n >= 0 ? static_cast<size_t>(n) : 0);
	o.forced = out;
	sslm_status st = SSLM_OK;
	o.next = cpu::Token(c.M(), s, &st);
	if (st != SSLM_OK) o.next = -1000 - static_cast<int32_t>(st);
	sslm_seq_release(s);
	c.seqs.erase(std::remove(c.seqs.begin(), c.seqs.end(), s), c.seqs.end());
	return o;
}

// GPU off: the same history as RunGpu's, then up to `n` decode-only tokens after the pending one (or after the prompt):
// a token, -2 at a dead end, or -1000 - status at the first failure (and nothing after it).
std::vector<int32_t> OffNextGpu(GRig& rig, const std::vector<int32_t>& base, int32_t limit, int32_t h, bool pending,
                                bool pooled, int32_t n) {
	std::vector<int32_t> out;
	GSeq s;
	if (pooled) {
		if (sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), g_pool, kBudgetLimit, &s.h) != GS::SSLM_OK || !s.h) return out;
		rig.seqs.push_back(s.h);
		SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, rig.Index(kPotion));
	} else {
		s = rig.New(kPotion);
		if (!s.h) return out;
	}
	if (rig.Prompt(s, PromptOfLength(base, static_cast<size_t>(limit - h))) != GS::SSLM_OK) return out;
	GS st = GS::SSLM_OK;
	if (pending && Token(rig.env, s, &st) < 0) return out;
	for (int32_t i = 0; i < n; ++i) {
		const int32_t t = Token(rig.env, s, &st);
		if (st != GS::SSLM_OK) { out.push_back(-1000 - static_cast<int32_t>(st)); break; }
		out.push_back(t);
		if (t < 0) break;
	}
	return out;
}
// CPU decode-only, the same history as RunCpu's: up to `n` tokens after the pending one (or after the prompt).
std::vector<int32_t> OffNextCpu(CRig& c, const std::vector<int32_t>& base, int32_t limit, int32_t h, bool pending, int32_t n) {
	std::vector<int32_t> out;
	sslm_seq s = c.New(kPotion);
	if (!s) return out;
	if (c.Prefill(s, PromptOfLength(base, static_cast<size_t>(limit - h)), SSLM_SPAN_PROMPT) == SSLM_OK) {
		sslm_status st = SSLM_OK;
		if (!pending || cpu::Token(c.M(), s, &st) >= 0) {
			for (int32_t i = 0; i < n; ++i) {
				const int32_t t = cpu::Token(c.M(), s, &st);
				if (st != SSLM_OK) { out.push_back(-1000 - static_cast<int32_t>(st)); break; }
				out.push_back(t);
				if (t < 0) break;
			}
		}
	}
	sslm_seq_release(s);
	c.seqs.erase(std::remove(c.seqs.begin(), c.seqs.end(), s), c.seqs.end());
	return out;
}
// Exact's kept count, computed without the verb: the chain's leading ids that equal Off's own next tokens, at most
// `limit_rows`.
int32_t ExactKept(const std::vector<int32_t>& chain, const std::vector<int32_t>& off, int32_t limit_rows) {
	int32_t k = 0;
	while (k < limit_rows && static_cast<size_t>(k) < chain.size() && static_cast<size_t>(k) < off.size() &&
	       chain[static_cast<size_t>(k)] == off[static_cast<size_t>(k)])
		++k;
	return k;
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		const std::vector<int32_t> base = c.Tokenize(kPromptPotionText);
		const GS ps = sslm_gpu_kv_pool_create(rig.ctx(), rig.model(), 128, &g_pool);
		CHECK_MSG(ps == GS::SSLM_OK && g_pool, "[G12.6] pool: %s", Name(ps));
		const int32_t cap = static_cast<int32_t>(rig.env.fx.model_cap);
		for (const bool pooled : {false, true}) {
			if (pooled && !g_pool) continue;
			const int32_t limit = pooled ? kBudgetLimit : cap;
			const char* arm = pooled ? "pooled" : "dedicated";
			for (const bool pending : {false, true}) {
				// k, from a run with room to spare: the rows asked for (the pending row and the chain). It picks the h
				// values below; each run's expectations use its own chain, since a pending run's choice can differ with
				// the prompt's length (7 false fails on the RX 7900 XT, 2026-10-26, before this).
				const Outcome full = RunGpu(rig, base, limit, 64 + 2, pending, pooled);
				const int32_t k = full.k;
				CHECK_MSG(k >= 3, "[G12.6 %s] precondition: k = %d (a chain of 2 or more after the pending row)", arm, k);
				for (const int32_t h : {0, 1, 2, k - 1, k, k + 1}) {
					if (h < 0 || h > 64) continue;
					const Outcome g = RunGpu(rig, base, limit, h, pending, pooled);
					const Outcome off = RunGpu(rig, base, limit, h, pending, pooled, /*call_verb=*/false);
					const std::vector<int32_t> offn = OffNextGpu(rig, base, limit, h, pending, pooled, 64);
					// The pass runs when the clipped draft is not empty: a pending token, a chain, and a row after it.
					const bool runs = pending && !g.chain.empty() && h >= 2;
					// Exact, computed without the verb: the chain (already clipped to chunk_budget - 1) clipped to h - 1
					// rows after the pending row, cut at Off's first different token.
					const int32_t kept = runs ? ExactKept(g.chain, offn, h - 1) : 0;
					const std::vector<int32_t> want_forced(g.chain.begin(), g.chain.begin() + kept);
					CHECK_MSG(g.st == GS::SSLM_OK && g.forced == want_forced,
					          "[G12.6 %s%s h=%d k=%d] %s forced %s; want SSLM_OK forced %s (Exact: the chain cut at Off's own "
					          "tokens, clipped to %d)", arm, pending ? " pending" : "", h, g.k, Name(g.st), Str(g.forced).c_str(),
					          Str(want_forced).c_str(), runs ? h - 1 : 0);
					if (runs) {
						CHECK_MSG(g.ready == 1 && g.layer == rig.env.fx.layers && g.ctx == limit - h + 1 + kept,
						          "[G12.6 %s h=%d] ready %d layer %u context %lld (want 1, %u, %lld)", arm, h, g.ready, g.layer,
						          static_cast<long long>(g.ctx), rig.env.fx.layers, static_cast<long long>(limit - h + 1 + kept));
						const int32_t want_next = static_cast<size_t>(kept) < offn.size() ? offn[static_cast<size_t>(kept)] : -9;
						CHECK_MSG(g.next_st == GS::SSLM_OK && g.ctx_after_next == g.ctx && g.next == want_next,
						          "[G12.6 %s h=%d] the next decode: %s %d, context %lld -> %lld; want Off's %d, finish only", arm, h,
						          Name(g.next_st), g.next, static_cast<long long>(g.ctx), static_cast<long long>(g.ctx_after_next),
						          want_next);
					} else {
						CHECK_MSG(g.next == off.next && g.next_st == off.next_st,
						          "[G12.6 %s h=%d] the next decode %s %d; GPU off %s %d", arm, h, Name(g.next_st), g.next,
						          Name(off.next_st), off.next);
						// The saved state after a no-op is GPU off's at the same state, byte for byte (engine code review N4).
						CHECK_MSG(!g.blob.empty() && g.blob == off.blob, "[G12.6 %s h=%d] the blob (%zu bytes) differs from GPU "
						          "off's (%zu bytes)", arm, h, g.blob.size(), off.blob.size());
					}
					if (!pooled) {
						// The CPU verb (Exact since 1.18) against Exact computed from CPU decode-only, and against the GPU.
						const CpuOutcome cpu = RunCpu(c, base, limit, h, pending);
						const std::vector<int32_t> coff = OffNextCpu(c, base, limit, h, pending, 64);
						const int32_t ckept = runs ? ExactKept(g.chain, coff, h - 1) : 0;
						const std::vector<int32_t> cwant(g.chain.begin(), g.chain.begin() + ckept);
						CHECK_MSG(cpu.st == SSLM_OK && cpu.forced == cwant,
						          "[G12.6 dedicated%s h=%d] the CPU verb %d forced %s; want SSLM_OK forced %s (CPU decode-only)",
						          pending ? " pending" : "", h, static_cast<int>(cpu.st), Str(cpu.forced).c_str(), Str(cwant).c_str());
						if (runs) {
							const int32_t cnext = static_cast<size_t>(ckept) < coff.size() ? coff[static_cast<size_t>(ckept)] : -9;
							CHECK_MSG(cpu.next == cnext, "[G12.6 dedicated h=%d] the CPU's next token %d, its decode-only %d", h,
							          cpu.next, cnext);
							CHECK_MSG(cpu.next == g.next && cpu.forced == g.forced, "[G12.6 dedicated h=%d] CPU forced %s next %d; "
							          "GPU forced %s next %d", h, Str(cpu.forced).c_str(), cpu.next, Str(g.forced).c_str(), g.next);
						}
					} else if (h > 0) {
						// The same prompt on a dedicated sequence with room, the verb's budget clipping the draft to h - 1.
						const Outcome d = RunGpu(rig, base, limit, h, pending, false, true, std::min(h, 64));
						CHECK_MSG(d.forced == g.forced && d.next == g.next, "[G12.6 pooled h=%d] forced %s next %d; the "
						          "dedicated sequence clipped to %d rows: %s next %d", h, Str(g.forced).c_str(), g.next, h,
						          Str(d.forced).c_str(), d.next);
					}
					rig.ReleaseSeqs();
				}
			}
		}
		rig.ReleaseSeqs();
		if (g_pool) sslm_gpu_kv_pool_destroy(rig.ctx(), g_pool);
	}
	return Finish("sfdg_cell_g12_6");
}
