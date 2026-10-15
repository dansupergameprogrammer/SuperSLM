// Paged-KV plan (rev 16.2), step G1, group g1b: the cell parts this group owns that are bracketed [G4].
//   2.7/G4   the GPU prefix and adopt hostile population
//   2.11/G4  prefix begin given a pool of another context or model; begin_from given a parent of
//            another context
//   5.8/G4   GPU prefix and adopt rejections (5.4's and 5.5's prefix, begin_from, freeze and adopt
//            cases; adopt onto a Submitted sequence; a prefix prefill stopped mid-span)
//   9.13/G4  the A15 row: adopt onto a dedicated sequence
//
// RED NOW: link-red on the G3 verbs (pool create/destroy, the budgeted create, the GPU mirror peek)
// and the G4 verbs (sslm_gpu_prefix_begin, _begin_from, _set_schema, _prefill, _freeze, _release,
// sslm_gpu_seq_adopt_prefix), all declared at 63e7b0e with no definition.
// GREEN AT G4: every check below on both GPUs, the debug layer clean for every cell that runs an
// adopt or begin_from (G4's exit).
#include "g1b_helpers.h"
#include "g1b_hostile_prefix.h"

namespace g1b {
namespace {

using namespace hostile;

// ---- 2.7 [G4] ------------------------------------------------------------------------------------
// GPU prefix and adopt hostile population. Each case asserts its status; the whole GPU mirror of
// every pool the case involves unchanged across the refused call, read through the GPU mirror peek,
// with the prefix X mid-page at 1,000 so that an adopt's tail copy into S's reserve page would show;
// SslmGpuSeqSetSchemaForG5Bridge on the sequence S still SSLM_OK (a fresh S: a refused adopt changes
// nothing, bind eligibility included); and, after sslm_gpu_ready, the GPU fill probe unchanged on
// every pool involved: the count the test computes from §3.4 for the same construction without the
// refused call (X keeps ceil(1000/16) = 63 pages after freeze, S holds R(64) = 5, and each extra
// holder its own reservation). begin_from takes no pool, so it has no cross-pool case.
//
// Statuses (§4 item 7, rev 6): adopt of an unfrozen prefix, adopt across pools, prefix prefill after
// freeze, set_schema after content and begin_from on an unfrozen parent -> SSLM_GPU_PREFIX_REJECTED;
// adopt with a schema mismatch -> SSLM_GPU_PREFIX_SCHEMA_MISMATCH; adopt across models or contexts,
// and begin_from with a `ctx` other than the parent's -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
void Cell27G4() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	for (const HostileCase& c : kHostile) {
		// The probe runs the whole construction twice per pool (admission and refusal legs); the
		// status, mirror and SetSchema verdicts are asserted on every build.
		size_t n_probes = 0;
		std::vector<uint32_t> want_free;
		auto build = [&](Scene& sc, int leg) -> std::vector<ProbeTarget> {
			HostileScene h = BuildHostile(sc, e, c);
			if (!h.ok) {
				G1B_CHECK_MSG(false, "2.7 %s: construction failed (leg %d)", c.name, leg);
				return {};
			}
			std::vector<std::vector<uint8_t>> before;
			for (size_t i = 0; i < h.probes.size(); ++i) before.push_back(MirrorSnapshot(g, h.probes[i].pool, h.pool_pages[i]));
			const S st = RunHostile(sc, e, c, h);
			G1B_CHECK_STATUS(st, c.want, "2.7/G4 %s (leg %d)", c.name, leg);
			for (size_t i = 0; i < h.probes.size(); ++i) {
				// kills: an adopt that copies X's tail into S's reserve page before a check, or a
				// prefill that writes before its frozen check.
				G1B_CHECK_MSG(MirrorSnapshot(g, h.probes[i].pool, h.pool_pages[i]) == before[i],
				              "2.7/G4 %s: the GPU mirror of pool %zu changed across the refused call", c.name, i);
			}
			// kills: a GPU adopt that clears bind_eligible before a check (the CPU's early revocation).
			G1B_CHECK_STATUS(SslmGpuSeqSetSchemaForG5Bridge(e.ctx1, h.s, 0), S::SSLM_OK, "2.7/G4 %s: SetSchema on the sequence afterwards", c.name);
			n_probes = h.probes.size();
			want_free = h.free_pages;
			return h.probes;
		};
		// The first build fixes how many pools the case involves and their §3.4 counts.
		{
			Scene sc;
			build(sc, 0);
		}
		for (size_t i = 0; i < n_probes; ++i) {
			char label[160];
			std::snprintf(label, sizeof(label), "2.7/G4 %s, pool %zu", c.name, i);
			GpuFillProbe(g, build, i, want_free[i], label);
		}
	}
	// The controls: the same verbs admit the same construction's valid calls, so no refusal above
	// passes because a verb refuses everything.
	Scene sc;
	HostileScene h = BuildHostile(sc, e, kHostile[2]);  // pool A, X and S only
	if (!h.ok) return;
	G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(e.ctx1, h.s, h.x), S::SSLM_OK, "2.7/G4 control: adopt of the frozen X onto S");
	SslmGpuPrefixHandle* child = nullptr;
	const S st = sslm_gpu_prefix_begin_from(e.ctx1, h.x, kSeqBudget, &child);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "2.7/G4 control: begin_from the frozen X on its own context");
	if (st == S::SSLM_OK && child) sc.AddPrefix(e.ctx1, child);
}

// ---- 2.11 [G4] -----------------------------------------------------------------------------------
// sslm_gpu_prefix_begin given a `pool` of another context, then of another model, and
// sslm_gpu_prefix_begin_from given a `parent` of another context: SSLM_SEQUENCE_KV_BUFFER_MISMATCH, a
// null out-handle, and after sslm_gpu_ready the GPU fill probe unchanged on every pool the case
// involves (the foreign pool; for begin_from, the parent's pool, which a begin_from draws from). The
// draw-before-check mutant leaves that pool R(64) = 5 pages short. begin_from of another model cannot
// be built from its signature (the carrier's CLASS-G row).
void Cell211G4() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	constexpr uint32_t kForeign = 40;
	struct Case {
		const char* name;
		SslmGpuContext* pool_ctx;
		SslmGpuModelHandle* pool_model;
	} cases[] = {{"prefix_begin, `pool` of another context", e.ctx2, e.m2},
	             {"prefix_begin, `pool` of another model", e.ctx1, e.m1b}};
	for (const Case& c : cases) {
		auto build = [&](Scene& sc, int) -> std::vector<ProbeTarget> {
			SslmGpuKvPoolHandle* foreign = NewPool(sc, c.pool_ctx, c.pool_model, kForeign);
			if (!foreign) return {};
			SslmGpuPrefixHandle* out = reinterpret_cast<SslmGpuPrefixHandle*>(uintptr_t{0x1});
			const S st = sslm_gpu_prefix_begin(e.ctx1, e.m1, foreign, 64, &out);
			G1B_CHECK_STATUS(st, S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.11/G4 %s", c.name);
			G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "2.11/G4 %s: out-handle not nulled", c.name);
			if (st == S::SSLM_OK && out) sc.AddPrefix(e.ctx1, out);
			return {{c.pool_ctx, c.pool_model, foreign}};
		};
		char label[160];
		std::snprintf(label, sizeof(label), "2.11/G4 %s (foreign pool)", c.name);
		GpuFillProbe(g, build, 0, kForeign, label);
	}
	const std::vector<int32_t> pt = Stream(61, 40, g.vocab);  // parent: 40 tokens, keeps 3 pages
	auto build = [&](Scene& sc, int) -> std::vector<ProbeTarget> {
		SslmGpuKvPoolHandle* pp = NewPool(sc, e.ctx2, e.m2, kForeign);
		SslmGpuPrefixHandle* parent = pp ? FrozenPrefix(sc, g, e.ctx2, e.m2, pp, pt) : nullptr;
		if (!parent) return {};
		SslmGpuPrefixHandle* out = reinterpret_cast<SslmGpuPrefixHandle*>(uintptr_t{0x1});
		const S st = sslm_gpu_prefix_begin_from(e.ctx1, parent, 64, &out);
		G1B_CHECK_STATUS(st, S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.11/G4 begin_from, `parent` of another context");
		G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "2.11/G4 begin_from: out-handle not nulled");
		if (st == S::SSLM_OK && out) sc.AddPrefix(e.ctx1, out);
		return {{e.ctx2, e.m2, pp}};
	};
	GpuFillProbe(g, build, 0, kForeign - g.Frozen(static_cast<int64_t>(pt.size())), "2.11/G4 begin_from, `parent` of another context (the parent's pool)");
	// Controls: each verb admits the matching handles, so no refusal above is a verb that refuses all.
	Scene sc;
	SslmGpuKvPoolHandle* own = NewPool(sc, e.ctx1, e.m1, kForeign);
	SslmGpuPrefixHandle* parent = own ? FrozenPrefix(sc, g, e.ctx1, e.m1, own, pt) : nullptr;
	if (!parent) return;
	SslmGpuPrefixHandle* child = nullptr;
	const S st = sslm_gpu_prefix_begin_from(e.ctx1, parent, 64, &child);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "2.11/G4 control: begin_from a parent of this context");
	if (st == S::SSLM_OK && child) sc.AddPrefix(e.ctx1, child);
}

// ---- 5.8 [G4] ------------------------------------------------------------------------------------
// GPU prefix and adopt rejections. It takes 5.4's and 5.5's prefix, begin_from, freeze and adopt cases
// (rev 4), as tests/paged-kv/c5_dim5_failure.cpp builds them on CPU:
//   (a) an adopt onto a Submitted sequence -> SSLM_BUSY (and, drained, the same adopt is admitted);
//   (b) a prefix prefill whose dispatch_budget stops mid-span gives partial consumption and is
//       resumable, the resumed result byte-equal to one uninterrupted call: the whole GPU mirror of
//       two pools built the same way, and the next tokens of an adopter of each prefix;
//   (c) against an exhausted pool, prefix begin and begin_from are refused
//       SSLM_GPU_KV_POOL_EXHAUSTED with a null out-handle and nothing leaked (the GPU fill probe);
//       adopt against the exhausted pool SUCCEEDS, with the mid-page prefix of 1,000, and the adopter
//       runs its whole budget (b + 1 tokens, §3.4). An aligned prefix has no tail, so there the
//       tail-from-pool mutant is equivalent (5.4's separation); at 1,000 it is refused;
//   (d) 5.5: a create refused before a prefix freeze succeeds after it (freeze returns the unmapped
//       reserve page to the pool).
void Part58Busy(Env& e) {
	const Geometry& g = e.geo;
	Scene sc;
	SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, kPoolA);
	SslmGpuPrefixHandle* x = pool ? FrozenPrefix(sc, g, e.ctx1, e.m1, pool, Stream(51, kPrefixLen, g.vocab)) : nullptr;
	SslmGpuSequenceHandle* s = pool ? NewBudgetSeq(sc, e.ctx1, e.m1, pool, kSeqBudget) : nullptr;
	if (!x || !s) return;
	G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, s, Stream(3, 10, g.vocab)), S::SSLM_OK, "5.8 (a): prefill");
	int32_t tok = -1;
	DecodeOne(g, e.ctx1, s, 1, &tok);  // the ready token
	G1B_CHECK_STATUS(sslm_gpu_seq_embed_token(e.ctx1, s, tok >= 0 ? tok : 1), S::SSLM_OK, "5.8 (a): embed");
	// Submitted until sslm_gpu_ready drains it: the state does not leave Submitted on its own.
	G1B_CHECK_STATUS(sslm_decode_step_gpu(e.ctx1, s, nullptr, g.decode_budget), S::SSLM_OK, "5.8 (a): decode submitted");
	G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(e.ctx1, s, x), S::SSLM_BUSY, "5.8 (a): adopt onto a Submitted sequence");
	Scene::DrainOne(e.ctx1, s);
	G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(e.ctx1, s, x), S::SSLM_OK, "5.8 (a): the same adopt once drained");
}

void Part58MidSpan(Env& e) {
	const Geometry& g = e.geo;
	const std::vector<int32_t> span = Stream(71, 40, g.vocab);
	struct Built {
		std::vector<uint8_t> mirror;
		std::vector<int32_t> tokens;
	};
	auto build = [&](bool interrupted) {
		Built b;
		Scene sc;
		SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, 20);
		SslmGpuPrefixHandle* p = pool ? NewPrefix(sc, e.ctx1, e.m1, pool, 40) : nullptr;
		if (!p) return b;
		if (interrupted) {
			// Dispatches for 7 tokens at full depth: stops inside the 40-token span.
			int32_t consumed = -1;
			const S st = sslm_gpu_prefix_prefill(e.ctx1, p, span.data(), 40, 0, g.decode_budget * 7, &consumed);
			G1B_CHECK_MSG(consumed > 0 && consumed < 40, "5.8 (b): the stopped call consumed %d of 40 (status %s): no partial consumption",
			              consumed, StatusName(st));
			int32_t done = consumed > 0 ? consumed : 0;
			for (int guard = 0; done < 40 && guard < 100; ++guard) {  // resume, one stopped call at a time
				int32_t c = -1;
				sslm_gpu_prefix_prefill(e.ctx1, p, span.data() + done, 40 - done, 0, g.decode_budget * 7, &c);
				if (c <= 0) break;
				done += c;
			}
			G1B_CHECK_MSG(done == 40, "5.8 (b): resumed calls consumed %d of 40", done);
		} else {
			int32_t consumed = -1;
			G1B_CHECK_STATUS(sslm_gpu_prefix_prefill(e.ctx1, p, span.data(), 40, 0, g.decode_budget * 41, &consumed), S::SSLM_OK,
			                 "5.8 (b): one uninterrupted call");
			G1B_CHECK_MSG(consumed == 40, "5.8 (b): the uninterrupted call consumed %d of 40", consumed);
		}
		G1B_CHECK_STATUS(sslm_gpu_prefix_freeze(e.ctx1, p), S::SSLM_OK, "5.8 (b): freeze");
		SslmGpuSequenceHandle* s = NewBudgetSeq(sc, e.ctx1, e.m1, pool, 16);
		if (s && sslm_gpu_seq_adopt_prefix(e.ctx1, s, p) == S::SSLM_OK) b.tokens = DecodeN(g, e.ctx1, s, 8);
		sc.DrainAll();
		b.mirror = MirrorSnapshot(g, pool, 20);
		return b;
	};
	const Built a = build(true), u = build(false);
	G1B_CHECK_MSG(!a.mirror.empty() && a.mirror == u.mirror, "5.8 (b): the resumed prefix's pool mirror differs from one uninterrupted call's");
	G1B_CHECK_MSG(a.tokens.size() == 8 && a.tokens == u.tokens, "5.8 (b): an adopter's next tokens differ (%zu vs %zu tokens)",
	              a.tokens.size(), u.tokens.size());
}

void Part58Exhausted(Env& e) {
	const Geometry& g = e.geo;
	// X (1,000, budget 1,000: R = 64, keeps 63 after freeze) and S (R(64) = 5) fill a pool of 68.
	const uint32_t P = g.Frozen(kPrefixLen) + g.R(kSeqBudget);
	const std::vector<int32_t> xt = Stream(51, kPrefixLen, g.vocab);
	auto build = [&](Scene& sc, int leg) -> std::vector<ProbeTarget> {
		SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, P);
		SslmGpuPrefixHandle* x = pool ? FrozenPrefix(sc, g, e.ctx1, e.m1, pool, xt) : nullptr;
		SslmGpuSequenceHandle* s = x ? NewBudgetSeq(sc, e.ctx1, e.m1, pool, kSeqBudget) : nullptr;
		if (!s) return {};
		SslmGpuPrefixHandle* out = reinterpret_cast<SslmGpuPrefixHandle*>(uintptr_t{0x1});
		S st = sslm_gpu_prefix_begin(e.ctx1, e.m1, pool, 16, &out);  // R(16) = 2 > 0 free
		G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_POOL_EXHAUSTED, "5.8 (c) prefix_begin against the exhausted pool (leg %d)", leg);
		G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "5.8 (c) prefix_begin: out-handle not nulled");
		if (st == S::SSLM_OK && out) sc.AddPrefix(e.ctx1, out);
		out = reinterpret_cast<SslmGpuPrefixHandle*>(uintptr_t{0x1});
		st = sslm_gpu_prefix_begin_from(e.ctx1, x, 16, &out);
		G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_POOL_EXHAUSTED, "5.8 (c) begin_from against the exhausted pool (leg %d)", leg);
		G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "5.8 (c) begin_from: out-handle not nulled");
		if (st == S::SSLM_OK && out) sc.AddPrefix(e.ctx1, out);
		// Adopt against the exhausted pool succeeds (it never draws: §3.4), and runs its whole budget.
		G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(e.ctx1, s, x), S::SSLM_OK, "5.8 (c) adopt of the mid-page 1,000 prefix on the exhausted pool");
		S refusal = S::SSLM_OK;
		const int64_t n = TokensUntilRefusal(g, e.ctx1, s, &refusal);
		G1B_CHECK_MSG(n == kSeqBudget + 1, "5.8 (c) the adopter emitted %lld tokens, want %d", static_cast<long long>(n), kSeqBudget + 1);
		G1B_CHECK_STATUS(refusal, S::SSLM_GPU_KV_BUDGET_EXCEEDED, "5.8 (c) the adopter's refusal at its limit");
		sc.Release(s);  // S's reservation back: exactly R(64) free if nothing leaked
		return {{e.ctx1, e.m1, pool}};
	};
	GpuFillProbe(g, build, 0, g.R(kSeqBudget), "5.8 (c): nothing leaked by the refused begin and begin_from, nor by the adopt");
}

void Part55(Env& e) {
	const Geometry& g = e.geo;
	Scene sc;
	// X begins with R(1000) = 64 pages of a 65-page pool: 1 free, so a 2-page create is refused.
	SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, g.R(kPrefixLen) + 1);
	SslmGpuPrefixHandle* x = pool ? NewPrefix(sc, e.ctx1, e.m1, pool, kPrefixLen) : nullptr;
	if (!x) return;
	G1B_CHECK_STATUS(PrefixPrefillAll(g, e.ctx1, x, Stream(51, kPrefixLen, g.vocab)), S::SSLM_OK, "5.5: prefix content");
	SslmGpuSequenceHandle* s = reinterpret_cast<SslmGpuSequenceHandle*>(uintptr_t{0x1});
	S st = sslm_gpu_seq_create_budgeted(e.ctx1, e.m1, pool, 16, &s);
	G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_POOL_EXHAUSTED, "5.5: a 2-page create before the freeze");
	if (st == S::SSLM_OK && s) sc.seqs.push_back({e.ctx1, s});
	G1B_CHECK_STATUS(sslm_gpu_prefix_freeze(e.ctx1, x), S::SSLM_OK, "5.5: freeze");  // returns 1 page: 2 free
	s = nullptr;
	st = sslm_gpu_seq_create_budgeted(e.ctx1, e.m1, pool, 16, &s);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "5.5: the same create after the freeze");
	if (st == S::SSLM_OK && s) sc.seqs.push_back({e.ctx1, s});
}

void Cell58G4() {
	Env& e = GetEnv();
	if (!e.ok) return;
	Part58Busy(e);
	Part58MidSpan(e);
	Part58Exhausted(e);
	Part55(e);
}

// ---- 9.13 [G4 the A15 row] -------------------------------------------------------------------------
// Adopt onto a dedicated sequence (sslm_gpu_seq_create) -> SSLM_GPU_HOLDER_MODE_MISMATCH, and the
// sequence unchanged: its 'SLM5' re-save equals the save before (dedicated sequences keep writing
// 'SLM5', §4 item 10's writer rule, so this operand exists at G4), and its next tokens equal an unsaved
// twin's that never saw the adopt. Run on a fresh dedicated sequence and on one with content (the
// mid-page prefix X would have to be copied into either).
void Cell913G4() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	Scene sc;
	SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, kPoolB);
	SslmGpuPrefixHandle* x = pool ? FrozenPrefix(sc, g, e.ctx1, e.m1, pool, Stream(51, kPrefixLen, g.vocab)) : nullptr;
	if (!x) return;
	for (int32_t content : {0, 12}) {
		SslmGpuSequenceHandle* d = NewDedicatedSeq(sc, e.ctx1, e.m1, 2048);
		SslmGpuSequenceHandle* twin = NewDedicatedSeq(sc, e.ctx1, e.m1, 2048);
		if (!d || !twin) continue;
		const std::vector<int32_t> t = Stream(81, content > 0 ? content : 12, g.vocab);
		if (content) {
			G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, d, t), S::SSLM_OK, "9.13 A15: prefill");
			G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, twin, t), S::SSLM_OK, "9.13 A15: twin prefill");
		}
		S sv = S::SSLM_OK;
		const std::vector<uint8_t> before = SaveBlob(e.ctx1, d, &sv);
		G1B_CHECK_STATUS(sv, S::SSLM_OK, "9.13 A15: save before (content %d)", content);
		G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(e.ctx1, d, x), S::SSLM_GPU_HOLDER_MODE_MISMATCH,
		                 "9.13 A15: adopt onto a dedicated sequence (content %d)", content);
		const std::vector<uint8_t> after = SaveBlob(e.ctx1, d, &sv);
		G1B_CHECK_MSG(!before.empty() && after == before && ReadAt<uint32_t>(after, 0) == slm6::kMagicV5,
		              "9.13 A15 (content %d): the dedicated sequence's SLM5 re-save differs from the save before", content);
		if (!content) {
			G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, d, t), S::SSLM_OK, "9.13 A15: prefill after the refused adopt");
			G1B_CHECK_STATUS(PrefillPrompt(g, e.ctx1, twin, t), S::SSLM_OK, "9.13 A15: twin prefill");
		}
		const std::vector<int32_t> td = DecodeN(g, e.ctx1, d, 6), tt = DecodeN(g, e.ctx1, twin, 6);
		G1B_CHECK_MSG(td.size() == 6 && td == tt, "9.13 A15 (content %d): next tokens differ from the twin's", content);
	}
}

}  // namespace

G1B_CELL("2.7/G4", "G4", Cell27G4);
G1B_CELL("2.11/G4", "G4", Cell211G4);
G1B_CELL("5.8/G4", "G4", Cell58G4);
G1B_CELL("9.13/G4", "G4", Cell913G4);

}  // namespace g1b
