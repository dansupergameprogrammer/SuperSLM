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
		if (h == 0) CHECK_MSG(Observe(rig, s.h) == before, "[G12.6] h = 0 changed the sequence");
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
			const GS limit_status = pooled ? GS::SSLM_GPU_KV_BUDGET_EXCEEDED : kContextCapExceeded;
			const char* arm = pooled ? "pooled" : "dedicated";
			for (const bool pending : {false, true}) {
				// k, from a run with room to spare: the rows the verb admits when nothing limits it. It picks the h values
				// below; each run's expectations use its own k, since a pending run's choice can differ with the prompt's
				// length (7 false fails on the RX 7900 XT, 2026-10-26, before this).
				const Outcome full = RunGpu(rig, base, limit, 64 + 2, pending, pooled);
				const int32_t k = static_cast<int32_t>(full.forced.size()) + (pending ? 1 : 0);
				CHECK_MSG(full.st == GS::SSLM_OK && k >= 3 && full.k == k && full.chain == full.forced,
				          "[G12.6 %s] precondition: k = %d, queried %d (%s)", arm, k, full.k, Name(full.st));
				const Outcome off = RunGpu(rig, base, limit, 0, pending, pooled, /*call_verb=*/false);
				for (const int32_t h : {0, 1, 2, k - 1, k, k + 1}) {
					if (h < 0 || h > 64) continue;
					const Outcome g = RunGpu(rig, base, limit, h, pending, pooled);
					const int32_t admitted = std::min(h, g.k);
					const int32_t forced = admitted - (pending && admitted > 0 ? 1 : 0);
					const GS want_st = h < g.k ? limit_status : GS::SSLM_OK;
					std::vector<int32_t> want_forced(g.chain.begin(), g.chain.begin() + forced);
					CHECK_MSG(g.st == want_st && g.forced == want_forced,
					          "[G12.6 %s%s h=%d k=%d] %s forced %s; want %s forced %s", arm, pending ? " pending" : "", h, g.k,
					          Name(g.st), Str(g.forced).c_str(), Name(want_st), Str(want_forced).c_str());
					if (h > 0) {
						CHECK_MSG(g.ready == 1 && g.layer == rig.env.fx.layers && g.ctx == limit - h + admitted,
						          "[G12.6 %s h=%d] ready %d layer %u context %lld (want 1, %u, %lld)", arm, h, g.ready, g.layer,
						          static_cast<long long>(g.ctx), rig.env.fx.layers,
						          static_cast<long long>(limit - h + admitted));
						CHECK_MSG(g.next_st == GS::SSLM_OK && g.ctx_after_next == g.ctx,
						          "[G12.6 %s h=%d] the next decode: %s, context %lld -> %lld (finish only)", arm, h,
						          Name(g.next_st), static_cast<long long>(g.ctx), static_cast<long long>(g.ctx_after_next));
					} else {
						CHECK_MSG(g.next == off.next && g.next_st == off.next_st,
						          "[G12.6 %s h=0] the next decode %s %d; GPU off %s %d", arm, Name(g.next_st), g.next,
						          Name(off.next_st), off.next);
						// The saved state at h = 0 is GPU off's at the same state, byte for byte (engine code review N4).
						CHECK_MSG(!g.blob.empty() && g.blob == off.blob, "[G12.6 %s h=0] the blob (%zu bytes) differs from GPU "
						          "off's (%zu bytes)", arm, g.blob.size(), off.blob.size());
					}
					if (!pooled) {
						const CpuOutcome cpu = RunCpu(c, base, limit, h, pending);
						const bool cpu_limit = cpu.st == SSLM_CONTEXT_CAP_EXCEEDED;
						CHECK_MSG((h < g.k) == cpu_limit && (cpu.st == SSLM_OK || cpu_limit) && cpu.forced == g.forced,
						          "[G12.6 dedicated h=%d] the CPU verb %d forced %s; the GPU %s forced %s", h,
						          static_cast<int>(cpu.st), Str(cpu.forced).c_str(), Name(g.st), Str(g.forced).c_str());
						if (h > 0)
							CHECK_MSG(cpu.next == g.next, "[G12.6 dedicated h=%d] the next token: CPU %d, GPU %d", h, cpu.next,
							          g.next);
					} else if (h > 0) {
						// The same prompt on a dedicated sequence with room, the verb clipped to h rows by its budget.
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
