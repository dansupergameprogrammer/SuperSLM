// Paged-KV GPU slice, step G5 review fix (review-g5 F1/F2): a pooled holder that rests AT ITS CAP saves,
// restores into a pool and continues; and the GPU 'SLM6' K/V section equals the CPU 'SSB6' twin's for the
// same history. Not a §7 plan cell: the regression cell the G5 review asked for (F1's "add a cell"), in
// g1e's style (one executable, the pkv_def fixture, g1e_helpers.h's drivers, g1e_cpu_twin.h's CPU ABI).
//
// The ruling it pins (§3.7's L' rule applied as the CPU twin applies it): L' = context_length +
// (0 < layer_index < num_hidden_layers ? 1 : 0). The GPU rests a full-depth token at layer_index ==
// num_hidden_layers with context_length already advanced, a completed token, so row context_length
// holds no state and is not in the blob. The CPU rests the same history at layer_index 0, so both
// backends carry L' = context_length rows at a rest.
//
//   cap   a budget = cap (4,096) pooled holder prefilled with 4,096 tokens rests at context_length == cap.
//         The size query reports the exact SLM6 size (fixed 148 + residual + 8 + L' x bytes_per_token,
//         L' = cap); the save succeeds; kv_positions == cap; restore_into_pool (null prefix, a fresh
//         pool of R(cap) pages) is SSLM_OK; the restored holder re-saves the blob byte for byte and maps
//         the original's pages; its next turn (tokens and refusal) equals the unsaved original's.
//         Kills: the zero-row reading (L' = cap + 1 > cap: the save fails with need 0, F1).
//   page  a budget-32 holder prefilled with 32 tokens rests at a page boundary (B = 16): the restored
//         holder maps exactly the original's 2 pages (the zero-row reading maps 3), and the same
//         save / restore / re-save / next-turn checks.
//   cpu   for the page and cap histories, and for a mid-token state (the page history in a budget-48
//         holder, then one layer of the next token), the GPU blob's K/V section (kv_positions and the rows [0, L')) equals
//         the CPU SSB6 twin's byte for byte (CPU: sslm_seq_create_budgeted on a CPU page pool, the same
//         prompt through sslm_prefill, sslm_seq_save). Kills: GPU and CPU disagreeing on L' or the
//         row stride at a rest (F2), or on the mid-token zero substitution.
//
// Cost: one 4,096-token GPU prefill (about 10 minutes on lavapipe) and its CPU twin.
//
// Run: g1e_cell_slm6_cap.exe --fixture-dir=DIR
#include "g1e_cpu_twin.h"

using namespace g1e;

namespace {

uint32_t Mapped(SslmGpuSequenceHandle* h) {
	uint32_t page = 0, mapped = 0;
	if (sslm_gpu_test_only_seq_table_entry(h, 0, &page, &mapped) != GS::SSLM_OK) return 0;
	return mapped;
}

// The GPU blob's K/V section: kv_positions u64, then the L' rows (SLM6 ends with it).
std::vector<uint8_t> GpuKvSection(const std::vector<uint8_t>& b, const Slm6View& v) {
	if (!v.ok || v.kv_off < 8) return {};
	return std::vector<uint8_t>(b.begin() + static_cast<std::ptrdiff_t>(v.kv_off - 8), b.end());
}

// The CPU SSB6's K/V section, located by its own header (src/sslm_abi.cpp sslm_seq_save): the fixed part
// is 172 bytes (anti-LM history count u64 at 112), then the residual (hidden_size), the history (4 bytes
// each), kv_positions u64 and the rows to the end.
std::vector<uint8_t> CpuKvSection(const std::vector<uint8_t>& b, const Geo& g) {
	constexpr size_t kFixed = 172, kHistoryCountOff = 112;
	if (!IsMagic(b, "SSB6") || b.size() < kFixed) return {};
	const size_t off = kFixed + g.hidden + 4 * static_cast<size_t>(Le64(b, kHistoryCountOff));
	if (b.size() < off + 8) return {};
	return std::vector<uint8_t>(b.begin() + static_cast<std::ptrdiff_t>(off), b.end());
}

// The CPU twin: a budget holder on a CPU page pool, the same prompt, optionally one layer of the next
// token; its SSB6.
std::vector<uint8_t> CpuBlob(cpu::Model& cm, const Geo& g, int32_t budget, const std::vector<int32_t>& prompt, bool mid) {
	cpu::Pool pool(cm.model, static_cast<uint32_t>(g.R(budget)));
	sslm_seq s = nullptr;
	if (!pool.pool || sslm_seq_create_budgeted(cm.model, &pool.pool, budget, &s) != SSLM_OK) {
		CHECK_MSG(false, "CPU twin: sslm_seq_create_budgeted(%d)", budget);
		return {};
	}
	pool.Seq(s);
	const sslm_status st = cpu::PrefillAll(cm.model, s, prompt);
	CHECK_MSG(st == SSLM_OK, "CPU twin: prefill of %zu tokens: %d", prompt.size(), static_cast<int>(st));
	if (st != SSLM_OK) return {};
	if (mid && !cpu::EnterMidToken(cm.model, s)) return {};
	std::vector<uint8_t> b = cpu::Save(s);
	CHECK_MSG(IsMagic(b, "SSB6"), "CPU twin: a budget holder saves SSB6");
	return b;
}

void CompareKv(const std::vector<uint8_t>& gpu, const std::vector<uint8_t>& cpu, const Geo& g, const char* what) {
	const Slm6View v = Slm6Parse(gpu, g);
	const std::vector<uint8_t> gs = GpuKvSection(gpu, v), cs = CpuKvSection(cpu, g);
	if (gs.empty() || cs.empty()) {
		CHECK_MSG(false, "%s: no K/V section (GPU parse %d, CPU blob %zu bytes)", what, v.ok ? 1 : 0, cpu.size());
		return;
	}
	const uint64_t gpos = Le64(gs, 0), cpos = Le64(cs, 0);
	CHECK_MSG(gpos == cpos && gs.size() == cs.size(),
	          "%s: kv_positions GPU %llu (layer_index %u, context %lld), CPU %llu; sections %zu vs %zu bytes", what,
	          static_cast<unsigned long long>(gpos), v.layer_index, static_cast<long long>(v.context_length),
	          static_cast<unsigned long long>(cpos), gs.size(), cs.size());
	if (gpos != cpos || gs.size() != cs.size()) return;
	size_t first = gs.size();
	for (size_t i = 0; i < gs.size(); ++i)
		if (gs[i] != cs[i]) {
			first = i;
			break;
		}
	CHECK_MSG(first == gs.size(), "%s: the GPU SLM6 K/V rows differ from the CPU SSB6 twin's at section byte %zu of %zu", what,
	          first, gs.size());
	std::printf("%s: K/V section %zu bytes (kv_positions %llu), GPU == CPU: %s\n", what, gs.size(),
	            static_cast<unsigned long long>(gpos), first == gs.size() ? "yes" : "NO");
}

// One case: a fresh pool of R(budget) pages, a budget holder prefilled with `prompt`, saved, restored
// into another fresh pool, both continued. Returns the blob.
std::vector<uint8_t> RoundTrip(const Env& e, int32_t budget, const std::vector<int32_t>& prompt, uint32_t want_mapped,
                               const char* what) {
	const Geo& g = e.geo;
	const uint32_t R = static_cast<uint32_t>(g.R(budget));
	Rig rig(e, R);
	GSeq s = rig.pool ? rig.Budgeted(budget) : GSeq{};
	if (!s.h) return {};
	const GS pst = Prefill(e, s, prompt);
	CHECK_MSG(pst == GS::SSLM_OK, "%s: prefill of %zu tokens: %s", what, prompt.size(), StatusName(pst));
	if (pst != GS::SSLM_OK) return {};
	const int64_t ctx = ContextLength(s.h);
	CHECK_MSG(ctx == static_cast<int64_t>(prompt.size()) && LayerIndex(s.h) == e.fx.layers,
	          "%s: the original rests at context %lld, layer %u (want %zu, %u)", what, static_cast<long long>(ctx),
	          LayerIndex(s.h), prompt.size(), e.fx.layers);
	const uint32_t orig_mapped = Mapped(s.h);
	CHECK_MSG(orig_mapped == want_mapped, "%s: the original maps %u pages (want %u)", what, orig_mapped, want_mapped);
	// The size query: the exact SLM6 size, L' = context_length at a rest (kills F1: need 0).
	size_t need = 0;
	const GS qst = sslm_gpu_seq_save(e.ctx, s.h, nullptr, &need);
	const size_t want = pkvg_slm6::OffRows(g.hidden) + static_cast<size_t>(ctx) * g.BytesPerToken();
	CHECK_MSG(need == want, "%s: the size query reports %zu (status %s), want %zu (L' = %lld)", what, need, StatusName(qst), want,
	          static_cast<long long>(ctx));
	const std::vector<uint8_t> blob = Save(e, s.h);
	if (blob.empty()) return {};
	if (!Slm6SelfCheck(blob, g, budget, 0, ctx, what)) return blob;
	CHECK_MSG(Slm6Parse(blob, g).Lp == ctx, "%s: kv_positions %lld, want %lld", what, static_cast<long long>(Slm6Parse(blob, g).Lp),
	          static_cast<long long>(ctx));
	// Restore into a fresh pool of exactly R(budget) pages (origin 0: E = 0).
	Rig dst(e, R);
	if (!dst.pool) return blob;
	Restored r = RestoreIntoPool(dst, blob, nullptr, s);
	CHECK_MSG(r.status == GS::SSLM_OK, "%s: restore_into_pool: %s", what, StatusName(r.status));
	if (!r.seq.h) return blob;
	CHECK_MSG(Mapped(r.seq.h) == orig_mapped, "%s: the restored holder maps %u pages, the original %u", what, Mapped(r.seq.h),
	          orig_mapped);
	const std::vector<uint8_t> resave = Save(e, r.seq.h);
	CHECK_MSG(resave == blob, "%s: the restored holder's re-save differs from the blob (%zu vs %zu bytes)", what, resave.size(),
	          blob.size());
	const Turn a = NextTurn(e, s);
	GSeq t = r.seq;
	const Turn b = NextTurn(e, t);
	CHECK_MSG(a == b, "%s: the restored holder's next turn (%zu tokens, %s) differs from the original's (%zu, %s)", what,
	          b.tokens.size(), StatusName(b.refusal), a.tokens.size(), StatusName(a.refusal));
	std::printf("%s: next turn %zu tokens, refusal %s (original and restored)\n", what, a.tokens.size(), StatusName(a.refusal));
	return blob;
}

// The mid-token GPU blob for the page history: prefill, then one layer of the next token.
std::vector<uint8_t> MidBlob(const Env& e, int32_t budget, const std::vector<int32_t>& prompt) {
	Rig rig(e, static_cast<uint32_t>(e.geo.R(budget)));
	GSeq s = rig.pool ? rig.Budgeted(budget) : GSeq{};
	if (!s.h || Prefill(e, s, prompt) != GS::SSLM_OK || !EnterMidToken(e, s)) {
		CHECK_MSG(false, "mid-token: construction");
		return {};
	}
	return Save(e, s.h);
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			const Geo& g = e.geo;
			cpu::Model cm(e.fx.bytes);
			const std::vector<int32_t> page_prompt = Stream(131, 32, kVocab);
			const std::vector<int32_t> cap_prompt = Stream(137, static_cast<int32_t>(g.cap), kVocab);
			const std::vector<uint8_t> page = RoundTrip(e, 32, page_prompt, 2, "slm6-cap page (budget 32, rest at 32)");
			const std::vector<uint8_t> mid = MidBlob(e, 48, page_prompt);  // limit 48: room for the next token
			if (cm.ok) {
				if (!page.empty()) CompareKv(page, CpuBlob(cm, g, 32, page_prompt, false), g, "slm6-cap cpu (rest at 32)");
				if (!mid.empty()) CompareKv(mid, CpuBlob(cm, g, 48, page_prompt, true), g, "slm6-cap cpu (mid-token at 32)");
			}
			const std::vector<uint8_t> cap = RoundTrip(e, static_cast<int32_t>(g.cap), cap_prompt,
			                                           static_cast<uint32_t>(g.CapPages()), "slm6-cap cap (budget = cap, rest at cap)");
			if (cm.ok && !cap.empty())
				CompareKv(cap, CpuBlob(cm, g, static_cast<int32_t>(g.cap), cap_prompt, false), g, "slm6-cap cpu (rest at cap)");
		}
	}
	return FinishSuite("g1e_cell_slm6_cap");
}
