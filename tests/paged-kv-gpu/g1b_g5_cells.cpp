// Paged-KV plan (rev 16.2), step G1, group g1b: the cell parts this group owns that are bracketed [G5].
//   2.6/G5   hostile 'SLM6' (the cases of 2.3) refused by sslm_gpu_seq_restore_into_pool
//   2.7/G5   the 2.7 population's sequence and prefix re-saved 'SLM6' blobs equal the ones saved before
//   2.11/G5  restore-into-pool given a pool of another context or model (refused, A21), and a prefix of
//            another context or model (a private restore, A13)
//   5.9/G5   restore-into-pool against an exhausted pool: the first draw, and the second draw with a
//            null and with a mismatched handle
//
// RED NOW: link-red on the G3 and G4 verbs these constructions use and on
// sslm_gpu_seq_restore_into_pool (G5); and, at run time before G5, sslm_gpu_seq_save of a pooled
// sequence writes no 'SLM6', so the layout pin check (Slm6LayoutMatches) fails first.
// GREEN AT G5: every check below on both GPUs, the debug layer clean for every cell that runs
// restore-into-pool (G5's exit).
#include "g1b_helpers.h"
#include "g1b_hostile_prefix.h"

namespace g1b {
namespace {

constexpr uint32_t kTarget = 40;

// A pooled sequence's 'SLM6' blob with origin 40 and context length 44, saved from a source pool the
// target pools never see: a budget-64 sequence adopts a frozen 40-token prefix, then decodes 5 tokens
// (the ready token writes no row, the next 4 write rows 40-43). The layout pin is checked here.
struct SourceBlob {
	std::vector<uint8_t> blob;
	std::vector<int32_t> prefix_tokens;
	bool ok = false;
};
SourceBlob MakeSourceBlob(Env& e, Scene& sc) {
	const Geometry& g = e.geo;
	SourceBlob s;
	s.prefix_tokens = Stream(91, 40, g.vocab);
	SslmGpuKvPoolHandle* pool = NewPool(sc, e.ctx1, e.m1, 20);
	SslmGpuPrefixHandle* p = pool ? FrozenPrefix(sc, g, e.ctx1, e.m1, pool, s.prefix_tokens) : nullptr;
	SslmGpuSequenceHandle* q = p ? NewBudgetSeq(sc, e.ctx1, e.m1, pool, 64) : nullptr;
	if (!q) return s;
	G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(e.ctx1, q, p), S::SSLM_OK, "source blob: adopt");
	G1B_CHECK_MSG(DecodeN(g, e.ctx1, q, 5).size() == 5, "source blob: 5 tokens");
	S st = S::SSLM_OK;
	s.blob = SaveBlob(e.ctx1, q, &st);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "source blob: save");
	s.ok = Slm6LayoutMatches(g, s.blob, 64, 40, 44);
	G1B_CHECK_MSG(s.ok, "SLM6 layout pin stale: g1b_helpers.h slm6:: does not match G5's writer (re-anchor it there)");
	return s;
}

S RestoreInto(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool, const std::vector<uint8_t>& blob,
              size_t size, SslmGpuPrefixHandle* prefix, SslmGpuSequenceHandle** out) {
	uint32_t shared = 0xFFFFFFFFu;  // a diagnostic (§7 grading rule): recorded nowhere
	*out = reinterpret_cast<SslmGpuSequenceHandle*>(uintptr_t{0x1});
	return sslm_gpu_seq_restore_into_pool(ctx, model, pool, blob.data(), size, prefix, out, &shared);
}

// ---- 2.6 [G5] ------------------------------------------------------------------------------------
// Hostile 'SLM6', with the cases of 2.3: sslm_gpu_seq_restore_into_pool refuses each with
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH (the status today's GPU restore returns for a malformed blob), with
// a null out-handle and, after sslm_gpu_ready, the GPU fill probe unchanged on the target pool (fresh:
// all kTarget pages free). The cases, each one edit of a real blob (origin 40, context length 44,
// budget 64): budget 0, negative, above cap; a whole_reserve blob (kv_mode 0) with budget != cap;
// origin negative, above context_length; context_length above min(origin + budget, cap) (budget 3:
// limit 43 < 44); kv_mode outside {0, 1}; kv_positions inconsistent (one more, one fewer); truncated
// K/V. Control: the unedited blob is restored (SSLM_OK).
void Cell26G5() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	Scene src;
	const SourceBlob sb = MakeSourceBlob(e, src);
	if (!sb.ok) return;
	struct Case {
		const char* name;
		std::function<void(std::vector<uint8_t>*, size_t*)> edit;
	};
	const size_t kvpos_off = slm6::OffKvPositions(sb.blob);
	const Case cases[] = {
	    {"budget 0", [](std::vector<uint8_t>* b, size_t*) { WriteAt<int32_t>(b, slm6::kOffBudget, 0); }},
	    {"budget negative", [](std::vector<uint8_t>* b, size_t*) { WriteAt<int32_t>(b, slm6::kOffBudget, -1); }},
	    {"budget above cap", [&](std::vector<uint8_t>* b, size_t*) { WriteAt<int32_t>(b, slm6::kOffBudget, static_cast<int32_t>(g.cap + 1)); }},
	    {"whole_reserve blob with budget != cap", [](std::vector<uint8_t>* b, size_t*) { WriteAt<uint32_t>(b, slm6::kOffKvMode, 0u); }},
	    {"origin negative", [](std::vector<uint8_t>* b, size_t*) { WriteAt<int64_t>(b, slm6::kOffOrigin, -1); }},
	    {"origin above context_length", [](std::vector<uint8_t>* b, size_t*) { WriteAt<int64_t>(b, slm6::kOffOrigin, 45); }},
	    {"context_length above min(origin + budget, cap)", [](std::vector<uint8_t>* b, size_t*) { WriteAt<int32_t>(b, slm6::kOffBudget, 3); }},
	    {"kv_mode 2", [](std::vector<uint8_t>* b, size_t*) { WriteAt<uint32_t>(b, slm6::kOffKvMode, 2u); }},
	    {"kv_positions one more", [&](std::vector<uint8_t>* b, size_t*) { WriteAt<uint64_t>(b, kvpos_off, 45); }},
	    {"kv_positions one fewer", [&](std::vector<uint8_t>* b, size_t*) { WriteAt<uint64_t>(b, kvpos_off, 43); }},
	    {"truncated K/V", [](std::vector<uint8_t>*, size_t* size) { *size -= 1; }},
	};
	for (const Case& c : cases) {
		std::vector<uint8_t> blob = sb.blob;
		size_t size = blob.size();
		c.edit(&blob, &size);
		auto build = [&](Scene& sc, int leg) -> std::vector<ProbeTarget> {
			SslmGpuKvPoolHandle* t = NewPool(sc, e.ctx1, e.m1, kTarget);
			if (!t) return {};
			SslmGpuSequenceHandle* out = nullptr;
			const S st = RestoreInto(e.ctx1, e.m1, t, blob, size, nullptr, &out);
			G1B_CHECK_STATUS(st, S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.6/G5 hostile SLM6: %s (leg %d)", c.name, leg);
			G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "2.6/G5 %s: out-handle not nulled", c.name);
			if (st == S::SSLM_OK && out) sc.seqs.push_back({e.ctx1, out});
			return {{e.ctx1, e.m1, t}};
		};
		char label[160];
		std::snprintf(label, sizeof(label), "2.6/G5 hostile SLM6: %s", c.name);
		GpuFillProbe(g, build, 0, kTarget, label);
	}
	Scene sc;
	SslmGpuKvPoolHandle* t = NewPool(sc, e.ctx1, e.m1, kTarget);
	SslmGpuSequenceHandle* out = nullptr;
	const S st = t ? RestoreInto(e.ctx1, e.m1, t, sb.blob, sb.blob.size(), nullptr, &out) : S::SSLM_DEVICE_LOST;
	G1B_CHECK_STATUS(st, S::SSLM_OK, "2.6/G5 control: the unedited blob");
	if (st == S::SSLM_OK && out) sc.seqs.push_back({e.ctx1, out});
}

// ---- 2.7 [G5] ------------------------------------------------------------------------------------
// The sequence's and the prefix's re-saved 'SLM6' blobs equal the ones saved before, for every case
// of 2.7's population (g1b_hostile_prefix.h). The sequence S: its save before the refused call and its
// re-save after. The prefix: no GPU prefix save verb exists among the fourteen (gpu_1p0.h's page-pool
// block), so a prefix's state is saved through an adopter: a fresh budget-16 sequence on the prefix's
// own pool adopts it and is saved. X (the frozen prefix every case holds) is read that way before and
// after the call in the same scene; the case's other prefix, when it has one, after the call, against
// the same read in a twin scene that never made the call (an unfrozen prefix is frozen first, in both
// scenes, since only a frozen prefix can be adopted).
//
// CONSTRUCTION (G5 cells fix): an adopter must carry the prefix's schema binding (§3.9 A18), so when the
// prefix is schema-bound -- "adopt with a schema mismatch"'s other prefix is bound to schema 0 and has
// schema-content progress -- the adopter binds schema 0 before adopting. An unbound adopter of it is
// correctly refused PREFIX_SCHEMA_MISMATCH, which made the read itself fail rather than the graded call.
// The binding is the same in the case scene and the twin scene, so the comparison is unchanged.
std::vector<uint8_t> AdopterSave(Scene& sc, SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                 SslmGpuPrefixHandle* p, const char* what, int32_t schema = -1) {
	SslmGpuSequenceHandle* a = NewBudgetSeq(sc, ctx, model, pool, 16);
	if (!a) return {};
	if (schema >= 0) G1B_CHECK_STATUS(SslmGpuSeqSetSchemaForG5Bridge(ctx, a, schema), S::SSLM_OK, "2.7/G5: adopter schema bind for %s", what);
	G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(ctx, a, p), S::SSLM_OK, "2.7/G5: adopter of %s", what);
	S st = S::SSLM_OK;
	std::vector<uint8_t> b = SaveBlob(ctx, a, &st);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "2.7/G5: adopter save of %s", what);
	sc.Release(a);
	return b;
}
std::vector<uint8_t> OtherPrefixSave(Scene& sc, hostile::HostileScene& h, const hostile::HostileCase& c) {
	if (!h.other) return {};
	sslm_gpu_prefix_freeze(h.other_ctx, h.other);  // idempotent on a frozen prefix (G33)
	// SchemaProgressPrefix binds schema 0 (g1b_hostile_prefix.h); every other case's prefix is unbound.
	const int32_t schema = c.kind == hostile::HostileCase::kAdoptSchemaMismatch ? 0 : -1;
	return AdopterSave(sc, h.other_ctx, h.other_model, h.other_pool, h.other, c.name, schema);
}

void Cell27G5() {
	Env& e = GetEnv();
	if (!e.ok) return;
	for (const hostile::HostileCase& c : hostile::kHostile) {
		std::vector<uint8_t> other_twin;
		{
			Scene twin;
			hostile::HostileScene h = hostile::BuildHostile(twin, e, c);
			if (h.ok) other_twin = OtherPrefixSave(twin, h, c);
		}
		Scene sc;
		hostile::HostileScene h = hostile::BuildHostile(sc, e, c);
		if (!h.ok) {
			G1B_CHECK_MSG(false, "2.7/G5 %s: construction failed", c.name);
			continue;
		}
		S st = S::SSLM_OK;
		const std::vector<uint8_t> s_before = SaveBlob(e.ctx1, h.s, &st);
		G1B_CHECK_STATUS(st, S::SSLM_OK, "2.7/G5 %s: S saved before", c.name);
		const std::vector<uint8_t> x_before = AdopterSave(sc, e.ctx1, e.m1, h.pool_a, h.x, "X before");
		G1B_CHECK_STATUS(hostile::RunHostile(sc, e, c, h), c.want, "2.7/G5 %s", c.name);
		const std::vector<uint8_t> s_after = SaveBlob(e.ctx1, h.s, &st);
		G1B_CHECK_MSG(!s_before.empty() && s_after == s_before && ReadAt<uint32_t>(s_after, 0) == slm6::kMagic,
		              "2.7/G5 %s: S's re-saved SLM6 differs from the save before", c.name);
		const std::vector<uint8_t> x_after = AdopterSave(sc, e.ctx1, e.m1, h.pool_a, h.x, "X after");
		G1B_CHECK_MSG(!x_before.empty() && x_after == x_before, "2.7/G5 %s: X's re-save (through an adopter) differs", c.name);
		if (h.other) {
			const std::vector<uint8_t> other_after = OtherPrefixSave(sc, h, c);
			G1B_CHECK_MSG(!other_twin.empty() && other_after == other_twin,
			              "2.7/G5 %s: the case's prefix re-save (through an adopter) differs from the twin that never saw the call", c.name);
		}
	}
}

// ---- 2.11 [G5] -----------------------------------------------------------------------------------
// sslm_gpu_seq_restore_into_pool given a `pool` of another context, then of another model (§3.9 A21):
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH, a null out-handle, and the GPU fill probe unchanged on the foreign
// pool (all kTarget pages). The draw-before-check mutant leaves it R(64) + E short.
// Given a `prefix_or_null` of another context, then of another model (decision 64, A13), it restores
// privately instead: SSLM_OK; the next tokens equal a restore with a null handle; the fill probe
// unchanged on the foreign prefix's pool (kTarget - ceil(40/16) = 37) and, on the target pool, equal to
// the null-handle restore's: kTarget - R(64) - E(40, 64) = 40 - 5 - 2 = 33 by §3.4, which the
// null-handle construction is probed at too. The foreign prefix holds the blob's own 40 prefix tokens,
// so a build that shared across contexts or models would map its pages: its probes read the
// difference. Control: the same blob restored into a pool of this context and model is admitted.
void Cell211G5() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	Scene src;
	const SourceBlob sb = MakeSourceBlob(e, src);
	if (!sb.ok) return;
	struct PoolCase {
		const char* name;
		SslmGpuContext* ctx;
		SslmGpuModelHandle* model;
	} pool_cases[] = {{"`pool` of another context", e.ctx2, e.m2}, {"`pool` of another model", e.ctx1, e.m1b}};
	for (const PoolCase& c : pool_cases) {
		auto build = [&](Scene& sc, int) -> std::vector<ProbeTarget> {
			SslmGpuKvPoolHandle* foreign = NewPool(sc, c.ctx, c.model, kTarget);
			if (!foreign) return {};
			SslmGpuSequenceHandle* out = nullptr;
			const S st = RestoreInto(e.ctx1, e.m1, foreign, sb.blob, sb.blob.size(), nullptr, &out);
			G1B_CHECK_STATUS(st, S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.11/G5 restore_into_pool, %s", c.name);
			G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "2.11/G5 %s: out-handle not nulled", c.name);
			if (st == S::SSLM_OK && out) sc.seqs.push_back({e.ctx1, out});
			return {{c.ctx, c.model, foreign}};
		};
		char label[160];
		std::snprintf(label, sizeof(label), "2.11/G5 restore_into_pool, %s (foreign pool)", c.name);
		GpuFillProbe(g, build, 0, kTarget, label);
	}
	// The null-handle restore: the oracle for the tokens and the target pool's count.
	std::vector<int32_t> null_tokens;
	{
		Scene sc;
		SslmGpuKvPoolHandle* t = NewPool(sc, e.ctx1, e.m1, kTarget);
		SslmGpuSequenceHandle* out = nullptr;
		const S st = t ? RestoreInto(e.ctx1, e.m1, t, sb.blob, sb.blob.size(), nullptr, &out) : S::SSLM_DEVICE_LOST;
		G1B_CHECK_STATUS(st, S::SSLM_OK, "2.11/G5 control: the null-handle restore into a pool of this context and model");
		if (st == S::SSLM_OK && out) {
			sc.seqs.push_back({e.ctx1, out});
			null_tokens = DecodeN(g, e.ctx1, out, 6);
		}
	}
	const uint32_t target_free = kTarget - g.R(64) - g.E(40, 64);
	auto null_build = [&](Scene& sc, int) -> std::vector<ProbeTarget> {
		SslmGpuKvPoolHandle* t = NewPool(sc, e.ctx1, e.m1, kTarget);
		SslmGpuSequenceHandle* out = nullptr;
		if (t && RestoreInto(e.ctx1, e.m1, t, sb.blob, sb.blob.size(), nullptr, &out) == S::SSLM_OK && out) sc.seqs.push_back({e.ctx1, out});
		return {{e.ctx1, e.m1, t}};
	};
	GpuFillProbe(g, null_build, 0, target_free, "2.11/G5 the null-handle restore's target pool");
	struct PrefixCase {
		const char* name;
		SslmGpuContext* ctx;
		SslmGpuModelHandle* model;
	} prefix_cases[] = {{"`prefix_or_null` of another context", e.ctx2, e.m2}, {"`prefix_or_null` of another model", e.ctx1, e.m1b}};
	for (const PrefixCase& c : prefix_cases) {
		auto build = [&](Scene& sc, int leg) -> std::vector<ProbeTarget> {
			SslmGpuKvPoolHandle* fp = NewPool(sc, c.ctx, c.model, kTarget);
			SslmGpuPrefixHandle* foreign = fp ? FrozenPrefix(sc, g, c.ctx, c.model, fp, sb.prefix_tokens) : nullptr;
			SslmGpuKvPoolHandle* t = NewPool(sc, e.ctx1, e.m1, kTarget);
			if (!foreign || !t) return {};
			SslmGpuSequenceHandle* out = nullptr;
			const S st = RestoreInto(e.ctx1, e.m1, t, sb.blob, sb.blob.size(), foreign, &out);
			G1B_CHECK_STATUS(st, S::SSLM_OK, "2.11/G5 restore_into_pool, %s: a private restore (leg %d)", c.name, leg);
			if (st == S::SSLM_OK && out) {
				sc.seqs.push_back({e.ctx1, out});
				if (leg == 0) {
					const std::vector<int32_t> tok = DecodeN(g, e.ctx1, out, 6);
					G1B_CHECK_MSG(tok.size() == 6 && tok == null_tokens, "2.11/G5 %s: next tokens differ from the null-handle restore's", c.name);
				}
			}
			return {{c.ctx, c.model, fp}, {e.ctx1, e.m1, t}};
		};
		char label[160];
		std::snprintf(label, sizeof(label), "2.11/G5 %s: the foreign prefix's pool", c.name);
		GpuFillProbe(g, build, 0, kTarget - g.Frozen(40), label);
		std::snprintf(label, sizeof(label), "2.11/G5 %s: the target pool, as the null-handle restore's", c.name);
		GpuFillProbe(g, build, 1, target_free, label);
	}
}

// ---- 5.9 [G5] ------------------------------------------------------------------------------------
// Restore-into-pool against an exhausted pool: refused SSLM_GPU_KV_POOL_EXHAUSTED with a null
// out-handle and the GPU fill probe unchanged, for the first draw, and for the second draw with a null
// and with a mismatched handle (the private path, as 2.10). 2.10's construction at B = 16, cap 4096:
// the persona chain X (1,200 tokens, 75 pages frozen) in a pool of 75 + R(512) + E - 1 = 182, and the
// cohort blob (origin 1,200, budget 512): R(512) = 33, E = min(75, 256 - 33) = 75. The first draw (33)
// succeeds and the second (75 of the 74 left) fails, and the fill probe admits exactly the 107 free
// before the call; the skip-rollback mutant leaks R(512) = 33 (it reads 74). With the matching handle
// (X itself) the same restore is admitted, which is why the second-draw cases cannot use one. The
// first-draw case: a pool of 75 + 32 holds X, so 32 < R(512) are free; the probe reads 32.
// The mismatched handle is X', the same 1,200 tokens frozen in the source pool: another pool.
void Cell59G5() {
	Env& e = GetEnv();
	if (!e.ok) return;
	const Geometry& g = e.geo;
	constexpr int32_t kOrigin = 1200, kBudget = 512;
	const std::vector<int32_t> xt = Stream(101, kOrigin, g.vocab);
	const uint32_t R = g.R(kBudget), E = g.E(kOrigin, kBudget), X = g.Frozen(kOrigin);
	G1B_CHECK_MSG(R == 33 && E == 75 && X == 75, "5.9: the §3.4 arithmetic at this geometry (R %u, E %u, X %u)", R, E, X);
	Scene src;
	SslmGpuKvPoolHandle* sp = NewPool(src, e.ctx1, e.m1, X + 1 + R + 4);
	SslmGpuPrefixHandle* xs = sp ? FrozenPrefix(src, g, e.ctx1, e.m1, sp, xt) : nullptr;
	SslmGpuSequenceHandle* q = xs ? NewBudgetSeq(src, e.ctx1, e.m1, sp, kBudget) : nullptr;
	if (!q) return;
	G1B_CHECK_STATUS(sslm_gpu_seq_adopt_prefix(e.ctx1, q, xs), S::SSLM_OK, "5.9: the cohort holder adopts the persona chain");
	G1B_CHECK_MSG(DecodeN(g, e.ctx1, q, 3).size() == 3, "5.9: the cohort holder's 3 tokens");
	S sv = S::SSLM_OK;
	const std::vector<uint8_t> blob = SaveBlob(e.ctx1, q, &sv);
	G1B_CHECK_STATUS(sv, S::SSLM_OK, "5.9: the cohort blob");
	if (!Slm6LayoutMatches(g, blob, kBudget, kOrigin, kOrigin + 2)) {
		G1B_CHECK_MSG(false, "5.9: SLM6 layout pin stale (g1b_helpers.h slm6::)");
		return;
	}
	struct Case {
		const char* name;
		uint32_t pool;                  // X's 75 plus the free pages before the call
		bool mismatched;                // pass X' (another pool) as the handle
	} cases[] = {{"second draw, null handle", X + R + E - 1, false},
	             {"second draw, mismatched handle", X + R + E - 1, true},
	             {"first draw, null handle", X + R - 1, false}};
	for (const Case& c : cases) {
		auto build = [&](Scene& sc, int leg) -> std::vector<ProbeTarget> {
			SslmGpuKvPoolHandle* t = NewPool(sc, e.ctx1, e.m1, c.pool);
			SslmGpuPrefixHandle* x = t ? FrozenPrefix(sc, g, e.ctx1, e.m1, t, xt) : nullptr;
			if (!x) return {};
			SslmGpuSequenceHandle* out = nullptr;
			const S st = RestoreInto(e.ctx1, e.m1, t, blob, blob.size(), c.mismatched ? xs : nullptr, &out);
			G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_POOL_EXHAUSTED, "5.9 %s (leg %d)", c.name, leg);
			G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "5.9 %s: out-handle not nulled", c.name);
			if (st == S::SSLM_OK && out) sc.seqs.push_back({e.ctx1, out});
			return {{e.ctx1, e.m1, t}};
		};
		char label[160];
		std::snprintf(label, sizeof(label), "5.9 %s", c.name);
		GpuFillProbe(g, build, 0, c.pool - X, label);
	}
	// Control: the matching handle admits the same restore into the second-draw pool.
	Scene sc;
	SslmGpuKvPoolHandle* t = NewPool(sc, e.ctx1, e.m1, X + R + E - 1);
	SslmGpuPrefixHandle* x = t ? FrozenPrefix(sc, g, e.ctx1, e.m1, t, xt) : nullptr;
	if (!x) return;
	SslmGpuSequenceHandle* out = nullptr;
	const S st = RestoreInto(e.ctx1, e.m1, t, blob, blob.size(), x, &out);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "5.9 control: the matching handle shares, so the same restore is admitted");
	if (st == S::SSLM_OK && out) sc.seqs.push_back({e.ctx1, out});
}

}  // namespace

G1B_CELL("2.6/G5", "G5", Cell26G5);
G1B_CELL("2.7/G5", "G5", Cell27G5);
G1B_CELL("2.11/G5", "G5", Cell211G5);
G1B_CELL("5.9/G5", "G5", Cell59G5);

}  // namespace g1b
