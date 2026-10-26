// Multi-row GPU plan (rev 1.3.2), slice MR3e (plan §16 erratum E-7): the embedder prompt path. Since E-7,
// SslmGpuSeqPrefillPromptForG5Bridge reads the prompt through the multi-row path at up to H = 8 rows per pass,
// bounded only by dispatch_budget and the row cap. This is the verb SuperEmbedder's GPU encode calls: a fresh
// dedicated handle, reset, one bridge call with dispatch_budget 0xFFFFFFFF (its kDispatchBudget), then
// sslm_gpu_seq_read_prefill_final_hidden. Wine + vkd3d-proton + lavapipe, on pkv_def (and pkv_qk) and
// pkv_def_refuse (MR2's fixtures).
//
// Cells:
//   sweep    E-7 (a). For n = 1..33, 64, 127, 500 (n mod H != 0 for H = 2, 4, 8 included), at dispatch budgets 1,
//            3 x one token at full depth, and 0xFFFFFFFF, on a dedicated (cap 512) and a pooled (budget 512) handle:
//            one bridge call of n tokens equals the one-row path, which is the same tokens fed one bridge call per
//            token. Status, the final hidden (codes and m, e from sslm_gpu_seq_read_prefill_final_hidden), the whole
//            save blob (K/V and state), the context length and the ready flag. Also the embedder's own shape: a
//            dedicated handle whose cap is exactly n, budget 0xFFFFFFFF, plus one decode-bridge step after it
//            (the produced token and the blob). This is a guard: the one-row bridge passes it today.
//   refusal  E-7 (b). Every existing refusal of the bridge, with the status pinned and, where the contract says
//            so, nothing changed (the save blob, the context, the layer, the ready flag, the snapshot):
//            arguments (null ctx, null seq, a foreign seq, count -1, null tokens, budget 0) -> the argument status;
//            count 0 -> OK; Submitted -> BUSY; a chunk open -> BUSY (§4.6's row); a token out of range at rows
//            0, 1, 7, 8, 9 and the last -> SSLM_TOKEN_ID_OUT_OF_RANGE with rows before it committed; the context
//            cap -> SSLM_DEVICE_LOST (the header's "saturated context cap"); a pooled budget limit ->
//            SSLM_GPU_KV_BUDGET_EXCEEDED, the rows before the limit committed and the ready flag set; a guard
//            refusal at row r (pkv_def_refuse, r found at run time on the one-row path, with r mod 8 varying) ->
//            SSLM_SEQUENCE_REJECTED with rows before r committed and the snapshot empty (at least one r >= 9: a later pass).
//            Claim rule: the bridge takes no per-handle decode claim (the claim seam never fires inside it), as
//            today. Every case also prints an `obs` line; the runner diffs those lines against the base binary.
//   height   Ruling 24 (S2): ruling 19a's pass height, pinned through a count that does not depend on speed: the
//            recorder's widened lists (one per slice of a multi-row pass, the MR2 submission seam). For n rows at
//            dispatch budget B: H = min(8, n) when B >= dpl and n >= 2 (else H = 1, the shipped one-row path, which
//            records no widened list); the passes are floor(n / H) of H rows and one of n mod H; each pass of two
//            or more rows makes ceil(L / min(L, floor(B / dpl))) slices. With SSLM_GPU_MULTIROW_PROMPT=0, none.
//            Budgets 1, dpl - 1, dpl, 1.5 dpl, 2 dpl, (L - 1) dpl, L dpl - 1, L dpl and 0xFFFFFFFF, dedicated and
//            pooled; the sweep checks the same count on every call it makes (not under --quick). KE1 and KE1b
//            must turn it red.
//   reuse    Ruling 24 (S1): a guard refusal inside a wide pass on a reused handle: refuse (or a clean prompt),
//            reset, then a call that guard-refuses at row r inside a pass of 8, dedicated and pooled. Status,
//            context r, no snapshot in-process; the save blob (stale K/V rows past the context included) is graded
//            against the shipped bridge by the base leg's obs diff.
//   flags    Ruling 24 (M2): sslm_gpu_seq_last_rows_committed and the ready flag keep their entry values across
//            a bridge call (an entry of 3 from a closed PROMPT chunk), clean at budgets 0xFFFFFFFF and dpl and
//            guard-refused inside a wide pass, dedicated and pooled, equal to the one-row path's. KE5 and KE6 must
//            turn it red.
//   (opt-out, the tail-drop mutant MXE2 and the H = 1 control MXE1 are the runner's legs: see mr3e_wine.sh.)
// The sweep (ruling 24) also runs layer-sliced budgets (dpl, 2 dpl, (L - 1) dpl and 1.5 dpl, one not a multiple)
// and unaligned, page-straddling starts (N3: a prefix call of k = 3, 5, 12, 13 tokens, then n = 7, 8, 9, 13, 20).
//
// --print-obs prints one line per sweep and refusal case. The runner diffs them between the base tree (the head
// before MR3e) and the branch, so "unchanged" is graded against the shipped bridge's real bytes, not this file's
// reading of them.
//
// Usage: mr3e_cells.exe --fixtures=DIR [--cells=sweep,refusal,height,reuse,flags] [--quick] [--print-obs] [--show=N]
//   --quick   the sweep at budget 0xFFFFFFFF only, dedicated only (the mutant and opt-out legs)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "mr2_common.h"
#include "mr2_entry.h"  // the MR2 seams: the recorder's widened-list count (cell height)

// The engine's claim seam (SUPERSLM_GPU_ALLOC_FAULT_INJECTION builds; mr3_entry.h describes it).
void SslmGpuSeqClaimHookForTest(void (*hook)(SslmGpuSequenceHandle* seq, void* user), void* user) noexcept;

namespace {
using S = SslmGpuStatus;
using mr2::Model;
using mr2::World;

constexpr S kArgStatus = S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
constexpr uint32_t kEmbedderBudget = 0xFFFFFFFFu;  // SuperEmbedder's kDispatchBudget (engine_adapter_gpu.cpp)

struct Opt {
	std::string fixtures;
	std::set<std::string> cells;
	int show = 16;
	bool quick = false, print_obs = false;
	bool want(const char* c) const { return cells.empty() || cells.count(c); }
};
Opt g_opt;
std::map<std::string, int> g_fail, g_checks;
std::string g_cell;
void Check(bool ok, const std::string& what) {
	++g_checks[g_cell];
	if (!ok && ++g_fail[g_cell] <= g_opt.show) std::printf("FAIL %s: %s\n", g_cell.c_str(), what.c_str());
}
std::string F(const char* fmt, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(b, sizeof b, fmt, ap);
	va_end(ap);
	return b;
}
const char* N(S s) {
	static thread_local char buf[8][16];
	static thread_local int at = 0;
	char* b = buf[at++ % 8];
	std::snprintf(b, 16, "%u", static_cast<unsigned>(s));
	return b;
}
void StatusIs(S got, S want, const std::string& tag) { Check(got == want, F("%s: returned %s, want %s", tag.c_str(), N(got), N(want))); }
void Obs(const std::string& line) {
	if (g_opt.print_obs) std::printf("obs %s\n", line.c_str());
}

// ------------------------------------------------------------------------------------------------ observation
std::vector<int32_t> Stream(int n, int32_t vocab, uint32_t stream, int from = 0) {
	std::vector<int32_t> v;
	for (int i = from; i < from + n; ++i) v.push_back(mr2::StreamToken(i, vocab, stream));
	return v;
}
S Bridge(World& w, SslmGpuSequenceHandle* s, const int32_t* t, int32_t n, uint32_t budget) {
	return SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, s, t, n, budget);
}
int32_t ReadyFlag(World& w, SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_ready_for_logits(w.ctx, s, &r) == S::SSLM_OK ? r : -1;
}
uint32_t LayerIdx(SslmGpuSequenceHandle* s) { return *SslmGpuSeqHandleLayerIndexForBench(s); }
int64_t CtxLen(SslmGpuSequenceHandle* s) { return *SslmGpuSeqHandleContextLengthForBench(s); }
struct Hidden {
	S st = S::SSLM_DEVICE_LOST;
	std::vector<int8_t> codes;
	int64_t m = 0, e = 0;
	bool operator==(const Hidden& o) const { return st == o.st && codes == o.codes && m == o.m && e == o.e; }
	uint64_t Fnv() const { return mr2::Fnv(std::vector<uint8_t>(codes.begin(), codes.end())); }
	std::string Str() const {
		return F("status %s, %zu codes fnv %016llx, m %lld e %lld", N(st), codes.size(), (unsigned long long)Fnv(), (long long)m, (long long)e);
	}
};
Hidden ReadHidden(World& w, SslmGpuSequenceHandle* s) {
	Hidden h;
	h.codes.assign(SslmGpuSeqHandleHiddenSizeForBench(s), 0);
	size_t req = 0;
	h.st = sslm_gpu_seq_read_prefill_final_hidden(w.ctx, s, h.codes.data(), h.codes.size(), &req, &h.m, &h.e);
	if (h.st != S::SSLM_OK) h.codes.clear(), h.m = h.e = 0;
	else h.codes.resize(req);
	return h;
}
std::vector<uint8_t> Blob(World& w, SslmGpuSequenceHandle* s, S* st = nullptr) {
	std::vector<uint8_t> b;
	S x = S::SSLM_OK;
	if (!mr2::Save(w, s, &b, &x)) b.clear();
	if (st) *st = x;
	return b;
}
// Everything a refusal must leave as it was, and what a case reports.
struct State {
	std::vector<uint8_t> blob;
	S save = S::SSLM_OK;
	Hidden hidden;
	int64_t ctx = -1;
	uint32_t layer = 0;
	int32_t ready = -1;
	bool operator==(const State& o) const {
		return blob == o.blob && save == o.save && hidden == o.hidden && ctx == o.ctx && layer == o.layer && ready == o.ready;
	}
	std::string Str() const {
		return F("save %s blob %zu fnv %016llx ctx %lld layer %u ready %d hidden %s fnv %016llx m %lld e %lld", N(save),
		         blob.size(), (unsigned long long)mr2::Fnv(blob), (long long)ctx, layer, ready, N(hidden.st),
		         (unsigned long long)hidden.Fnv(), (long long)hidden.m, (long long)hidden.e);
	}
};
State Observe(World& w, SslmGpuSequenceHandle* s) {
	State o;
	o.blob = Blob(w, s, &o.save);
	o.hidden = ReadHidden(w, s);
	o.ctx = CtxLen(s);
	o.layer = LayerIdx(s);
	o.ready = ReadyFlag(w, s);
	return o;
}
void Unchanged(World& w, SslmGpuSequenceHandle* s, const State& before, const std::string& tag) {
	const State after = Observe(w, s);
	Check(after == before, F("%s: changed the handle (before %s; after %s)", tag.c_str(), before.Str().c_str(), after.Str().c_str()));
}
void BlobEq(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::string& tag) {
	long d = -1;
	size_t n = 0;
	for (size_t i = 0; i < std::min(got.size(), want.size()); ++i)
		if (got[i] != want[i]) { if (d < 0) d = static_cast<long>(i); ++n; }
	if (d < 0 && got.size() != want.size()) d = static_cast<long>(std::min(got.size(), want.size()));
	Check(d < 0 && !got.empty(), F("%s: blob differs from the one-row path (sizes %zu/%zu, first byte %ld, %zu bytes differ)", tag.c_str(),
	                               got.size(), want.size(), d, n));
}

// ------------------------------------------------------------------------------------------------ pass height
// Ruling 24 (S2): the recorder's widened lists on the context so far (the MR2 submission seam).
uint64_t Widened(World& w) {
	SslmGpuMr2Seams sm{};
	return sslm_gpu_test_only_mr2_seams(w.ctx, &sm) == S::SSLM_OK ? sm.widened_lists : ~0ull;
}
bool OptedOut() {
	const char* e = std::getenv("SSLM_GPU_MULTIROW_PROMPT");
	return e && std::string(e) == "0";
}
// Ruling 19a, as ruling 24 pins it: the widened lists one bridge call of n admitted rows at budget B records.
uint64_t ExpectedWidened(const Model& m, int n, uint32_t budget) {
	const uint32_t dpl = m.dpl(), L = m.layers();
	if (OptedOut() || n < 2 || dpl == 0 || budget < dpl) return 0;
	const uint64_t H = std::min<uint64_t>(8, static_cast<uint64_t>(n));
	const uint64_t passes = static_cast<uint64_t>(n) / H + ((static_cast<uint64_t>(n) % H) >= 2 ? 1 : 0);
	const uint64_t per = std::min<uint64_t>(L, budget / dpl);
	return passes * ((L + per - 1) / per);
}
// The layer-sliced budgets ruling 24 adds: dpl, 2 dpl, (L - 1) dpl and one not a multiple (1.5 dpl), each below
// L dpl where the model allows (deduplicated, in order).
std::vector<uint32_t> SlicedBudgets(const Model& m) {
	const uint32_t d = m.dpl(), L = m.layers();
	std::vector<uint32_t> v;
	for (uint32_t b : {d, 2 * d, (L - 1) * d, d + d / 2})
		if (b >= 1 && std::find(v.begin(), v.end(), b) == v.end()) v.push_back(b);
	return v;
}

// ------------------------------------------------------------------------------------------------ sweep
const std::vector<int> kSizes = [] {
	std::vector<int> v;
	for (int n = 1; n <= 33; ++n) v.push_back(n);
	v.push_back(64), v.push_back(127), v.push_back(500);
	return v;
}();
constexpr int32_t kCap = 512;

// What the one-row path (one bridge call per token) leaves after token n.
struct Ref {
	S st = S::SSLM_OK;
	State state;
};
void Compare(World& w, SslmGpuSequenceHandle* s, S st, const Ref& ref, int n, const std::string& tag) {
	StatusIs(st, ref.st, tag);
	const State got = Observe(w, s);
	Check(got.hidden == ref.state.hidden, F("%s: final hidden %s, the one-row path %s", tag.c_str(), got.hidden.Str().c_str(), ref.state.hidden.Str().c_str()));
	Check(got.ctx == n && got.ctx == ref.state.ctx, F("%s: context %lld, the one-row path %lld (n %d)", tag.c_str(), (long long)got.ctx, (long long)ref.state.ctx, n));
	Check(got.ready == ref.state.ready && got.layer == ref.state.layer,
	      F("%s: ready %d layer %u, the one-row path %d / %u", tag.c_str(), got.ready, got.layer, ref.state.ready, ref.state.layer));
	BlobEq(got.blob, ref.state.blob, tag);
	Obs(F("%s st %s %s", tag.c_str(), N(st), got.Str().c_str()));
}

void SweepConfig(const Model& m, bool pooled, uint32_t budget, uint32_t stream) {
	World w(m);
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuKvPoolHandle* pool = pooled ? w.Pool(2 * (kCap / 16) + 8) : nullptr;
	auto make = [&]() { return pooled ? w.Pooled(pool, kCap) : w.Dedicated(kCap); };
	const int32_t V = m.vocab();
	const std::vector<int32_t> t = Stream(kSizes.back(), V, stream);
	// The one-row path, token by token, snapshotted at every n of the set.
	std::map<int, Ref> ref;
	SslmGpuSequenceHandle* r = make();
	if (!r) { Check(false, "reference handle"); return; }
	S rst = S::SSLM_OK;
	for (int i = 0; i < kSizes.back() && rst == S::SSLM_OK; ++i) {
		rst = Bridge(w, r, &t[i], 1, budget);
		if (std::binary_search(kSizes.begin(), kSizes.end(), i + 1)) ref[i + 1] = Ref{rst, Observe(w, r)};
	}
	Check(rst == S::SSLM_OK, F("%s budget %u: the one-row path itself returned %s", pooled ? "pooled" : "dedicated", budget, N(rst)));
	w.Release(r);
	for (int n : kSizes) {
		const std::string tag = F("%s %s budget %u n %d", m.base().c_str(), pooled ? "pooled" : "dedicated", budget, n);
		SslmGpuSequenceHandle* s = make();
		if (!s) { Check(false, tag + ": handle"); continue; }
		const uint64_t w0 = Widened(w);
		const S st = Bridge(w, s, t.data(), n, budget);
		const uint64_t got = Widened(w) - w0, want = ExpectedWidened(m, n, budget);
		Compare(w, s, st, ref[n], n, tag);
		// Ruling 24 (S2): the pass height, through the recorder's slice count (not under --quick: the H = 1 control
		// MXE1 and the opt-out legs grade bytes only).
		if (!g_opt.quick) Check(got == want, F("%s: %llu widened lists, ruling 19a's H gives %llu", tag.c_str(), (unsigned long long)got, (unsigned long long)want));
		w.Release(s);
	}
}
// N3 (ruling 24): a prefix call of k tokens, then a call of n: passes that start at an unaligned context and, on a
// pooled handle, straddle a page. Against the one-row path (one call per token), as the sweep compares.
void SweepUnaligned(const Model& m, bool pooled, uint32_t stream) {
	World w(m);
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuKvPoolHandle* pool = pooled ? w.Pool(2 * (kCap / 16) + 8) : nullptr;
	auto make = [&]() { return pooled ? w.Pooled(pool, kCap) : w.Dedicated(kCap); };
	for (int k : {3, 5, 12, 13})
		for (int n : {7, 8, 9, 13, 20}) {
			const std::string tag = F("%s %s unaligned k %d n %d", m.base().c_str(), pooled ? "pooled" : "dedicated", k, n);
			const std::vector<int32_t> t = Stream(k + n, m.vocab(), stream);
			SslmGpuSequenceHandle* s = make();
			SslmGpuSequenceHandle* r = make();
			if (!s || !r) { Check(false, tag + ": handles"); continue; }
			S st = Bridge(w, s, t.data(), k, kEmbedderBudget);
			if (st == S::SSLM_OK) st = Bridge(w, s, t.data() + k, n, kEmbedderBudget);
			Ref ref;
			for (int i = 0; i < k + n && ref.st == S::SSLM_OK; ++i) ref.st = Bridge(w, r, &t[i], 1, kEmbedderBudget);
			ref.state = Observe(w, r);
			Compare(w, s, st, ref, k + n, tag);
			w.Release(s);
			w.Release(r);
		}
}
// The embedder's shape: cap exactly n, reset, one call, the read; then a decode-bridge step on both.
void SweepTight(const Model& m, uint32_t stream) {
	World w(m);
	if (!w.ok) { Check(false, "world"); return; }
	const int32_t V = m.vocab();
	for (int n : kSizes) {
		if (n > 160 && g_opt.quick) continue;
		const std::string tag = F("%s embedder cap %d budget 0xFFFFFFFF n %d", m.base().c_str(), n, n);
		const std::vector<int32_t> t = Stream(n, V, stream);
		SslmGpuSequenceHandle* s = w.Dedicated(n);
		SslmGpuSequenceHandle* r = w.Dedicated(n);
		if (!s || !r) { Check(false, tag + ": handles"); continue; }
		sslm_gpu_seq_reset(w.ctx, s);
		sslm_gpu_seq_reset(w.ctx, r);
		const S st = Bridge(w, s, t.data(), n, kEmbedderBudget);
		Ref ref;
		for (int i = 0; i < n && ref.st == S::SSLM_OK; ++i) ref.st = Bridge(w, r, &t[i], 1, kEmbedderBudget);
		ref.state = Observe(w, r);
		Compare(w, s, st, ref, n, tag);
		// The decode bridge's first step (the ready shortcut: no embed, the finish only, since the cap is full).
		int32_t a = -1, b = -1;
		const S sa = SslmGpuSeqDecodeStepForG5Bridge(w.ctx, s, 0, m.full_budget(), &a);
		const S sb = SslmGpuSeqDecodeStepForG5Bridge(w.ctx, r, 0, m.full_budget(), &b);
		Check(sa == sb && a == b, F("%s: the decode bridge after it returned %s token %d, after the one-row path %s token %d", tag.c_str(), N(sa), a, N(sb), b));
		BlobEq(Blob(w, s), Blob(w, r), tag + " then a decode step");
		w.Release(s);
		w.Release(r);
	}
}
void CellSweep(const Model& def, const Model& qk) {
	SweepTight(def, 11);
	SweepConfig(def, false, kEmbedderBudget, 12);
	if (g_opt.quick) return;
	SweepConfig(def, true, kEmbedderBudget, 13);
	SweepConfig(def, false, 1, 14);
	SweepConfig(def, true, 1, 15);
	SweepConfig(def, false, 3 * def.full_budget(), 16);
	SweepConfig(def, true, 3 * def.full_budget(), 17);
	if (qk.layers()) SweepConfig(qk, false, kEmbedderBudget, 18);  // the QK-norm geometry (25 dispatches per layer)
	// Ruling 24 (S2): layer-sliced budgets; (N3) unaligned and page-straddling starts.
	uint32_t stream = 40;
	for (uint32_t b : SlicedBudgets(def)) {
		SweepConfig(def, false, b, stream++);
		SweepConfig(def, true, b, stream++);
	}
	if (qk.layers()) SweepConfig(qk, false, qk.dpl(), stream++);
	SweepUnaligned(def, false, 50);
	SweepUnaligned(def, true, 51);
}

// ------------------------------------------------------------------------------------------------ height
// Ruling 24 (S2): ruling 19a's H at the budget edges, through the recorder's widened-list count alone (fast: the
// mutant leg's KE1 and KE1b run this cell). Status OK and context n on every call.
void CellHeight(const Model& def, const Model& qk) {
	for (const Model* m : {&def, &qk}) {
		if (!m->layers()) continue;
		const uint32_t d = m->dpl(), L = m->layers();
		std::vector<uint32_t> budgets;
		for (uint32_t b : {1u, d - 1, d, d + d / 2, 2 * d, (L - 1) * d, L * d - 1, L * d, kEmbedderBudget})
			if (b >= 1 && std::find(budgets.begin(), budgets.end(), b) == budgets.end()) budgets.push_back(b);
		for (int pooled = 0; pooled < 2; ++pooled) {
			if (m == &qk && pooled) continue;
			World w(*m);
			if (!w.ok) { Check(false, "world"); return; }
			SslmGpuKvPoolHandle* pool = pooled ? w.Pool(2 * (64 / 16) + 8) : nullptr;
			for (uint32_t b : budgets)
				for (int n : {1, 2, 7, 8, 9, 17, 20}) {
					const std::string tag = F("%s %s budget %u n %d", m->base().c_str(), pooled ? "pooled" : "dedicated", b, n);
					SslmGpuSequenceHandle* s = pooled ? w.Pooled(pool, 64) : w.Dedicated(64);
					if (!s) { Check(false, tag + ": handle"); continue; }
					const std::vector<int32_t> t = Stream(n, m->vocab(), 60 + static_cast<uint32_t>(n));
					const uint64_t w0 = Widened(w);
					const S st = Bridge(w, s, t.data(), n, b);
					const uint64_t got = Widened(w) - w0, want = ExpectedWidened(*m, n, b);
					StatusIs(st, S::SSLM_OK, tag);
					Check(CtxLen(s) == n, F("%s: context %lld, want %d", tag.c_str(), (long long)CtxLen(s), n));
					Check(got == want, F("%s: %llu widened lists, ruling 19a's H gives %llu", tag.c_str(), (unsigned long long)got, (unsigned long long)want));
					w.Release(s);
				}
		}
	}
}

// ------------------------------------------------------------------------------------------------ refusal
std::atomic<int> g_claims{0};
void CountClaim(SslmGpuSequenceHandle*, void*) { g_claims.fetch_add(1); }

void CellRefusal(const Model& m, const Model& rf) {
	const int32_t V = m.vocab();
	const uint32_t B = kEmbedderBudget;
	World w(m);
	World other(m);
	if (!w.ok || !other.ok) { Check(false, "worlds"); return; }
	const std::vector<int32_t> head = Stream(5, V, 21), t = Stream(20, V, 22);
	// A handle with a prompt behind it: the ready flag and the snapshot set, so "nothing changed" covers them.
	auto primed = [&]() {
		SslmGpuSequenceHandle* s = w.Dedicated(64);
		if (s) Check(Bridge(w, s, head.data(), 5, B) == S::SSLM_OK, "prime");
		return s;
	};
	// Arguments.
	{
		SslmGpuSequenceHandle* s = primed();
		SslmGpuSequenceHandle* foreign = other.Dedicated(64);
		const State before = Observe(w, s);
		struct Case { const char* name; std::function<S()> call; };
		const std::vector<Case> cases = {
		    {"null ctx", [&] { return SslmGpuSeqPrefillPromptForG5Bridge(nullptr, s, t.data(), 9, B); }},
		    {"null seq", [&] { return SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, nullptr, t.data(), 9, B); }},
		    {"foreign seq", [&] { return SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, foreign, t.data(), 9, B); }},
		    {"count -1", [&] { return Bridge(w, s, t.data(), -1, B); }},
		    {"null tokens", [&] { return Bridge(w, s, nullptr, 9, B); }},
		    {"budget 0", [&] { return Bridge(w, s, t.data(), 9, 0); }},
		};
		for (const Case& c : cases) {
			const S st = c.call();
			StatusIs(st, kArgStatus, c.name);
			Unchanged(w, s, before, c.name);
			Obs(F("refusal %s st %s", c.name, N(st)));
		}
		const S z = Bridge(w, s, t.data(), 0, B);
		StatusIs(z, S::SSLM_OK, "count 0");
		const State after = Observe(w, s);
		Check(after.blob == before.blob && after.ctx == before.ctx && after.ready == before.ready, "count 0: changed the handle");
		Obs(F("refusal count0 st %s %s", N(z), after.Str().c_str()));
	}
	// Submitted, then the same handle drained equals a twin that never saw the call.
	{
		SslmGpuSequenceHandle* s = w.Dedicated(64);
		SslmGpuSequenceHandle* tw = w.Dedicated(64);
		for (SslmGpuSequenceHandle* x : {s, tw}) {
			Check(Bridge(w, x, head.data(), 5, B) == S::SSLM_OK, "submitted: prompt");
			Check(sslm_gpu_seq_embed_token(w.ctx, x, t[0]) == S::SSLM_OK, "submitted: embed");
		}
		Check(sslm_decode_step_gpu(w.ctx, s, nullptr, m.dpl()) == S::SSLM_OK, "submitted: submit");
		const S st = Bridge(w, s, t.data(), 9, B);
		StatusIs(st, S::SSLM_BUSY, "Submitted");
		mr2::Ready(w, s);
		mr2::Slice(w, tw, m.dpl());
		BlobEq(Blob(w, s), Blob(w, tw), "Submitted: drained, against a twin that never made the call");
		Obs(F("refusal submitted st %s", N(st)));
	}
	// A chunk open (§4.6's BUSY row), at layer 0 and at a mid layer.
	for (int mid = 0; mid < 2; ++mid) {
		SslmGpuSequenceHandle* s = primed();
		SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, t.data(), 3, 0};
		int32_t admitted = 0;
		Check(sslm_gpu_seq_embed_rows(w.ctx, s, &spec, &admitted) == S::SSLM_OK && admitted == 3, "chunk open: embed_rows");
		if (mid) mr2::Slice(w, s, m.dpl());
		const State before = Observe(w, s);
		const S st = Bridge(w, s, t.data() + 3, 9, B);
		StatusIs(st, S::SSLM_BUSY, mid ? "chunk open at a mid layer" : "chunk open at layer 0");
		Unchanged(w, s, before, "chunk open");
		sslm_gpu_seq_rows_abandon(w.ctx, s);
		Obs(F("refusal chunk-open%d st %s", mid, N(st)));
	}
	// Mid-token (one layer of a one-row token run): no pinned status in the header; graded by the base diff only.
	{
		SslmGpuSequenceHandle* s = primed();
		sslm_gpu_seq_embed_token(w.ctx, s, t[0]);
		mr2::Slice(w, s, m.dpl());
		const S st = Bridge(w, s, t.data() + 1, 9, B);
		mr2::Ready(w, s);
		Obs(F("refusal mid-token st %s %s", N(st), Observe(w, s).Str().c_str()));
	}
	// A token out of range at row k: the rows before it land, the call returns the embed's status.
	for (int k : {0, 1, 7, 8, 9, 19}) {
		for (int32_t bad : {V, -1}) {
			std::vector<int32_t> p = t;
			p[k] = bad;
			SslmGpuSequenceHandle* s = w.Dedicated(64);
			const S st = Bridge(w, s, p.data(), 20, B);
			const std::string tag = F("token %d at row %d", bad, k);
			StatusIs(st, S::SSLM_TOKEN_ID_OUT_OF_RANGE, tag);
			const State o = Observe(w, s);
			Check(o.ctx == k, F("%s: context %lld, want %d (the rows before it)", tag.c_str(), (long long)o.ctx, k));
			Check(o.hidden.st != S::SSLM_OK, tag + ": the snapshot is readable");
			Obs(F("refusal oov%d-row%d st %s %s", bad, k, N(st), o.Str().c_str()));
			w.Release(s);
		}
	}
	// The context cap: a dedicated handle at cap c given more than c - p0 tokens.
	for (int n : {21, 23, 27, 29}) {
		SslmGpuSequenceHandle* s = w.Dedicated(20);
		const std::vector<int32_t> p = Stream(n, V, 23);
		const S st = Bridge(w, s, p.data(), n, B);
		const std::string tag = F("cap 20 n %d", n);
		StatusIs(st, S::SSLM_DEVICE_LOST, tag);
		const State o = Observe(w, s);
		Check(o.hidden.st != S::SSLM_OK, tag + ": the snapshot is readable");
		Obs(F("refusal cap20-n%d st %s %s", n, N(st), o.Str().c_str()));
		w.Release(s);
	}
	// A pooled budget limit: the rows before it land, the ready flag is set, the snapshot is empty; the bytes equal the
	// one-row path's (20 calls, then the 21st refused).
	for (int n : {21, 23, 27, 29}) {
		SslmGpuKvPoolHandle* pool = w.Pool(16);
		SslmGpuSequenceHandle* s = w.Pooled(pool, 20);
		SslmGpuSequenceHandle* r = w.Pooled(pool, 20);
		const std::vector<int32_t> p = Stream(n, V, 24);
		const S st = Bridge(w, s, p.data(), n, B);
		S rs = S::SSLM_OK;
		for (int i = 0; i < n && rs == S::SSLM_OK; ++i) rs = Bridge(w, r, &p[i], 1, B);
		const std::string tag = F("pooled budget 20 n %d", n);
		StatusIs(st, S::SSLM_GPU_KV_BUDGET_EXCEEDED, tag);
		StatusIs(rs, S::SSLM_GPU_KV_BUDGET_EXCEEDED, tag + " (the one-row path)");
		const State o = Observe(w, s), ro = Observe(w, r);
		Check(o.ctx == 20 && o.ready == 1, F("%s: context %lld ready %d, want 20 and 1", tag.c_str(), (long long)o.ctx, o.ready));
		Check(o.hidden.st != S::SSLM_OK, tag + ": the snapshot is readable");
		BlobEq(o.blob, ro.blob, tag);
		Obs(F("refusal pooled20-n%d st %s %s", n, N(st), o.Str().c_str()));
		w.Release(s);
		w.Release(r);
	}
	// The claim rule: the bridge takes no per-handle decode claim (as today), so the claim seam never fires in it.
	{
		SslmGpuSequenceHandle* s = w.Dedicated(64);
		g_claims = 0;
		SslmGpuSeqClaimHookForTest(&CountClaim, nullptr);
		const S st = Bridge(w, s, t.data(), 20, B);
		SslmGpuSeqClaimHookForTest(nullptr, nullptr);
		StatusIs(st, S::SSLM_OK, "claim: the call");
		Check(g_claims == 0, F("claim: the bridge took a per-handle claim %d times (it takes none today)", g_claims.load()));
		Obs(F("refusal claim st %s claims %d", N(st), g_claims.load()));
	}
	// A guard refusal at row r, found on the one-row path (one call per token) on pkv_def_refuse.
	if (!rf.layers()) { Check(false, "pkv_def_refuse.sslm did not load"); return; }
	World x(rf);
	if (!x.ok) { Check(false, "refuse world"); return; }
	std::set<int> residues;
	int found = 0;
	int late = 0;  // refusals in a later pass (r >= 9)
	for (uint32_t stream = 0; stream < 192 && residues.size() < 16; ++stream) {
		const std::vector<int32_t> p = Stream(48, rf.vocab(), 100 + stream);
		SslmGpuSequenceHandle* r = x.Dedicated(64);
		int at = -1;
		for (int i = 0; i < 44; ++i) {
			const S st = Bridge(x, r, &p[i], 1, B);
			if (st == S::SSLM_SEQUENCE_REJECTED) { at = i; break; }
			if (st != S::SSLM_OK) break;
		}
		x.Release(r);
		const int key = at % 8 + (at >= 9 ? 8 : 0);
		if (at < 1 || residues.count(key)) continue;
		residues.insert(key);
		++found;
		late += at >= 9;
		const int n = at + 4;
		SslmGpuSequenceHandle* s = x.Dedicated(64);
		const S st = Bridge(x, s, p.data(), n, B);
		const std::string tag = F("guard refusal at row %d (stream %u, n %d)", at, 100 + stream, n);
		StatusIs(st, S::SSLM_SEQUENCE_REJECTED, tag);
		const State o = Observe(x, s);
		Check(o.ctx == at, F("%s: context %lld, want %d (the rows before it)", tag.c_str(), (long long)o.ctx, at));
		Check(o.hidden.st != S::SSLM_OK, tag + ": the snapshot is readable");
		Obs(F("refusal guard-row%d st %s %s", at, N(st), o.Str().c_str()));
		// The reset the header requires gives a handle the one-row path then fills equally.
		sslm_gpu_seq_reset(x.ctx, s);
		SslmGpuSequenceHandle* tw = x.Dedicated(64);
		const S a = Bridge(x, s, p.data(), at, B), b = Bridge(x, tw, p.data(), at, B);
		Check(a == S::SSLM_OK && b == S::SSLM_OK, tag + ": after reset");
		BlobEq(Blob(x, s), Blob(x, tw), tag + ": after reset, against a fresh handle");
		x.Release(s);
		x.Release(tw);
	}
	Check(found >= 4 && late >= 1, F("guard-refusal rows on pkv_def_refuse: %d with distinct r mod 8, %d of them at r >= 9 (want 4 and 1)", found, late));
	std::printf("refusal: guard rows (r mod 8, +8 for r >= 9) covered:");
	for (int r : residues) std::printf(" %d", r);
	std::printf("\n");
}

// ------------------------------------------------------------------------------------------------ reuse, flags
// The guard-refusing streams of `rf` (first refusal row `at` >= 1 on the one-row path from a fresh handle), up to
// `want` with distinct at mod 8, and a 12-token stream that does not refuse.
std::vector<std::pair<uint32_t, int>> RefusingStreams(const Model& rf, size_t want, uint32_t* clean) {
	std::vector<std::pair<uint32_t, int>> out;
	std::set<int> residues;
	World x(rf);
	if (!x.ok) return out;
	for (uint32_t stream = 0; stream < 192 && out.size() < want; ++stream) {
		const std::vector<int32_t> p = Stream(48, rf.vocab(), 100 + stream);
		SslmGpuSequenceHandle* r = x.Dedicated(64);
		int at = -1;
		for (int i = 0; i < 44; ++i) {
			const S st = Bridge(x, r, &p[i], 1, kEmbedderBudget);
			if (st == S::SSLM_SEQUENCE_REJECTED) { at = i; break; }
			if (st != S::SSLM_OK) break;
		}
		x.Release(r);
		if (at < 1 || at > 40 || residues.count(at % 8)) continue;
		residues.insert(at % 8);
		out.push_back({100 + stream, at});
	}
	*clean = 0;
	for (uint32_t c = 400; c < 2400 && !*clean; ++c) {
		SslmGpuSequenceHandle* h = x.Dedicated(64);
		const std::vector<int32_t> q = Stream(12, rf.vocab(), c);
		if (Bridge(x, h, q.data(), 12, 1) == S::SSLM_OK) *clean = c;
		x.Release(h);
	}
	return out;
}

// Ruling 24 (S1): refuse (or a clean prompt), reset, then refuse inside a wide pass, dedicated and pooled.
void CellReuse(const Model& rf) {
	if (!rf.layers()) { Check(false, "pkv_def_refuse.sslm did not load"); return; }
	uint32_t clean = 0;
	const auto refusing = RefusingStreams(rf, 6, &clean);
	Check(refusing.size() >= 3 && clean != 0, F("reuse: %zu refusing streams, clean stream %u (want 3 and one)", refusing.size(), clean));
	if (refusing.empty() || !clean) return;
	auto latest = refusing.front();
	for (const auto& r : refusing)
		if (r.second > latest.second) latest = r;
	for (int pooled = 0; pooled < 2; ++pooled) {
		World x(rf);
		if (!x.ok) { Check(false, "world"); return; }
		SslmGpuKvPoolHandle* pool = pooled ? x.Pool(24) : nullptr;
		const char* kind = pooled ? "pooled" : "dedicated";
		for (const auto& [stream, at] : refusing) {
			const std::vector<int32_t> p = Stream(48, rf.vocab(), stream);
			const int n = at + 4;
			for (int setup = 0; setup < 2; ++setup) {
				SslmGpuSequenceHandle* s = pooled ? x.Pooled(pool, 64) : x.Dedicated(64);
				if (!s) { Check(false, "reuse: handle"); continue; }
				std::string what;
				if (setup == 0) {
					const std::vector<int32_t> q = Stream(12, rf.vocab(), clean);
					const S c = Bridge(x, s, q.data(), 12, kEmbedderBudget);
					Check(c == S::SSLM_OK, F("reuse %s: the clean prompt returned %s", kind, N(c)));
					what = F("clean prompt %u", clean);
				} else {
					const std::vector<int32_t> q = Stream(48, rf.vocab(), latest.first);
					const S c = Bridge(x, s, q.data(), latest.second + 4, kEmbedderBudget);
					Check(c == S::SSLM_SEQUENCE_REJECTED, F("reuse %s: the first refusal returned %s", kind, N(c)));
					what = F("refusal of stream %u at %d", latest.first, latest.second);
				}
				sslm_gpu_seq_reset(x.ctx, s);
				const S st = Bridge(x, s, p.data(), n, kEmbedderBudget);
				const std::string tag = F("reuse %s: %s, reset, stream %u refusing at %d (n %d)", kind, what.c_str(), stream, at, n);
				StatusIs(st, S::SSLM_SEQUENCE_REJECTED, tag);
				const State o = Observe(x, s);
				Check(o.ctx == at, F("%s: context %lld, want %d", tag.c_str(), (long long)o.ctx, at));
				Check(o.hidden.st != S::SSLM_OK, tag + ": the snapshot is readable");
				Obs(F("reuse %s setup%d stream %u at %d st %s %s", kind, setup, stream, at, N(st), o.Str().c_str()));
				x.Release(s);
			}
		}
	}
}

// Ruling 24 (M2): the rows query and the ready flag across a bridge call, from an entry of 3 (a closed PROMPT chunk).
bool FlagsEntry(World& w, const Model& m, SslmGpuSequenceHandle* s, const std::vector<int32_t>& e8, int32_t* lrc) {
	if (!s || Bridge(w, s, e8.data(), 5, kEmbedderBudget) != S::SSLM_OK) return false;
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, e8.data() + 5, 3, 0};
	int32_t adm = 0;
	if (sslm_gpu_seq_embed_rows(w.ctx, s, &spec, &adm) != S::SSLM_OK || adm != 3) return false;
	for (int i = 0; i < 64; ++i) {
		if (mr2::Slice(w, s, m.dpl()) != S::SSLM_OK) return false;
		int32_t open = -1;
		sslm_gpu_seq_rows_open(s, &open);
		if (open == 0) break;
	}
	*lrc = -1;
	return sslm_gpu_seq_last_rows_committed(s, lrc) == S::SSLM_OK;
}
int32_t Lrc(SslmGpuSequenceHandle* s) {
	int32_t v = -1;
	return sslm_gpu_seq_last_rows_committed(s, &v) == S::SSLM_OK ? v : -1;
}
void CellFlags(const Model& def, const Model& rf) {
	for (int pooled = 0; pooled < 2; ++pooled) {
		const char* kind = pooled ? "pooled" : "dedicated";
		// Clean: 9 rows (a pass of 8 and a one-row tail) at 0xFFFFFFFF and at dpl, against the one-row path.
		{
			World w(def);
			if (!w.ok) { Check(false, "world"); return; }
			SslmGpuKvPoolHandle* pool = pooled ? w.Pool(24) : nullptr;
			const std::vector<int32_t> t = Stream(17, def.vocab(), 33);
			for (uint32_t b : {kEmbedderBudget, def.dpl()}) {
				const std::string tag = F("flags %s clean budget %u", kind, b);
				SslmGpuSequenceHandle* s = pooled ? w.Pooled(pool, 64) : w.Dedicated(64);
				SslmGpuSequenceHandle* r = pooled ? w.Pooled(pool, 64) : w.Dedicated(64);
				int32_t ls = -1, lr = -1;
				if (!FlagsEntry(w, def, s, t, &ls) || !FlagsEntry(w, def, r, t, &lr)) { Check(false, tag + ": entry"); continue; }
				Check(ls == 3 && lr == 3, F("%s: setup: last_rows_committed %d / %d after a 3-row chunk", tag.c_str(), ls, lr));
				const int32_t ready0 = ReadyFlag(w, s);
				const S st = Bridge(w, s, t.data() + 8, 9, b);
				S rs = S::SSLM_OK;
				for (int i = 0; i < 9 && rs == S::SSLM_OK; ++i) rs = Bridge(w, r, &t[8 + i], 1, b);
				StatusIs(st, rs, tag);
				Check(Lrc(s) == ls, F("%s: last_rows_committed %d after the call, %d at its entry", tag.c_str(), Lrc(s), ls));
				Check(Lrc(s) == Lrc(r) && ReadyFlag(w, s) == ReadyFlag(w, r),
				      F("%s: last_rows_committed %d ready %d; the one-row path %d / %d", tag.c_str(), Lrc(s), ReadyFlag(w, s), Lrc(r), ReadyFlag(w, r)));
				Obs(F("flags %s clean budget %u st %s lrc %d -> %d ready %d -> %d", kind, b, N(st), ls, Lrc(s), ready0, ReadyFlag(w, s)));
				w.Release(s);
				w.Release(r);
			}
		}
		// Guard-refused inside a wide pass, from the same kind of entry on pkv_def_refuse.
		if (!rf.layers()) { Check(false, "pkv_def_refuse.sslm did not load"); return; }
		World x(rf);
		if (!x.ok) { Check(false, "world"); return; }
		SslmGpuKvPoolHandle* pool = pooled ? x.Pool(48) : nullptr;
		uint32_t clean = 0;
		for (uint32_t c = 400; c < 2400 && !clean; ++c) {
			SslmGpuSequenceHandle* h = x.Dedicated(64);
			int32_t l = -1;
			if (FlagsEntry(x, rf, h, Stream(8, rf.vocab(), c), &l) && l == 3) clean = c;
			x.Release(h);
		}
		Check(clean != 0, F("flags %s: no 8-token entry on pkv_def_refuse that does not refuse", kind));
		if (!clean) continue;
		const std::vector<int32_t> e8 = Stream(8, rf.vocab(), clean);
		int found = 0;
		for (uint32_t stream = 0; stream < 192 && found < 3; ++stream) {
			const std::vector<int32_t> p = Stream(40, rf.vocab(), 100 + stream);
			SslmGpuSequenceHandle* r = pooled ? x.Pooled(pool, 64) : x.Dedicated(64);
			int32_t lr = -1;
			if (!FlagsEntry(x, rf, r, e8, &lr)) { x.Release(r); continue; }
			int at = -1;
			S rs = S::SSLM_OK;
			for (int i = 0; i < 36; ++i) {
				rs = Bridge(x, r, &p[i], 1, kEmbedderBudget);
				if (rs != S::SSLM_OK) { at = i; break; }
			}
			if (rs != S::SSLM_SEQUENCE_REJECTED || at < 2) { x.Release(r); continue; }
			++found;
			const std::string tag = F("flags %s refused at row %d (stream %u)", kind, at, 100 + stream);
			SslmGpuSequenceHandle* s = pooled ? x.Pooled(pool, 64) : x.Dedicated(64);
			int32_t ls = -1;
			if (!FlagsEntry(x, rf, s, e8, &ls)) { Check(false, tag + ": entry"); x.Release(r); x.Release(s); continue; }
			const int32_t ready0 = ReadyFlag(x, s);
			const S st = Bridge(x, s, p.data(), at + 4, kEmbedderBudget);
			StatusIs(st, S::SSLM_SEQUENCE_REJECTED, tag);
			Check(CtxLen(s) == 8 + at, F("%s: context %lld, want %d", tag.c_str(), (long long)CtxLen(s), 8 + at));
			Check(Lrc(s) == ls && ls == 3, F("%s: last_rows_committed %d after the call, %d at its entry", tag.c_str(), Lrc(s), ls));
			// A refusal leaves the ready flag as the call found it (docs/api.md; the shipped bridge never sets it on a
			// short count), and the rows query as the one-row path leaves it.
			Check(Lrc(s) == Lrc(r) && ReadyFlag(x, s) == ready0,
			      F("%s: last_rows_committed %d (the one-row path %d), ready %d (at entry %d)", tag.c_str(), Lrc(s), Lrc(r), ReadyFlag(x, s), ready0));
			Obs(F("flags %s refused stream %u at %d st %s lrc %d -> %d ready %d -> %d", kind, 100 + stream, at, N(st), ls, Lrc(s), ready0, ReadyFlag(x, s)));
			x.Release(r);
			x.Release(s);
		}
		Check(found >= 1, F("flags %s: no stream refuses at row >= 2 after the entry", kind));
	}
}

}  // namespace

int main(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--fixtures=", 0) == 0) g_opt.fixtures = a.substr(11);
		else if (a.rfind("--show=", 0) == 0) g_opt.show = std::atoi(a.c_str() + 7);
		else if (a == "--quick") g_opt.quick = true;
		else if (a == "--print-obs") g_opt.print_obs = true;
		else if (a.rfind("--cells=", 0) == 0) {
			std::string l = a.substr(8);
			for (size_t p = 0; p <= l.size();) {
				const size_t q = l.find(',', p);
				g_opt.cells.insert(l.substr(p, q == std::string::npos ? std::string::npos : q - p));
				if (q == std::string::npos) break;
				p = q + 1;
			}
		}
	}
	Model def, qk, rf;
	if (!def.Load(g_opt.fixtures + "/pkv_def.sslm")) return 1;
	if (!g_opt.quick && !qk.Load(g_opt.fixtures + "/pkv_qk.sslm")) qk = Model{};
	const char* env = std::getenv("SSLM_GPU_MULTIROW_PROMPT");
	std::printf("mr3e: SSLM_GPU_MULTIROW_PROMPT=%s\n", env ? env : "(unset)");
	const std::vector<std::pair<const char*, std::function<void()>>> cells = {
	    {"sweep", [&] { CellSweep(def, qk); }},
	    {"refusal", [&] { if (!rf.layers() && !rf.Load(g_opt.fixtures + "/pkv_def_refuse.sslm")) rf = Model{}; CellRefusal(def, rf); }},
	    {"height", [&] { CellHeight(def, qk); }},
	    {"reuse", [&] { if (!rf.layers() && !rf.Load(g_opt.fixtures + "/pkv_def_refuse.sslm")) rf = Model{}; CellReuse(rf); }},
	    {"flags", [&] { if (!rf.layers() && !rf.Load(g_opt.fixtures + "/pkv_def_refuse.sslm")) rf = Model{}; CellFlags(def, rf); }},
	};
	bool all = true;
	for (auto& c : cells) {
		if (!g_opt.want(c.first)) continue;
		g_cell = c.first;
		c.second();
		const int f = g_fail[c.first], n = g_checks[c.first];
		if (n == 0) std::printf("    cell %s: n/a\n", c.first);
		else if (f) std::printf("RED cell %s: %d failures of %d checks\n", c.first, f, n);
		else std::printf("    cell %s: PASS (%d checks)\n", c.first, n);
		all = all && f == 0;
	}
	std::printf("cells: %s\n", all ? "PASS" : "RED");
	return all ? 0 : 1;
}
