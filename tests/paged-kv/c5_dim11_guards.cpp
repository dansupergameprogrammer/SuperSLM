// Paged-KV plan (rev 16.1) step C1: dimension 11's budget-mode guard-vitality cells, owned by C5.
//
//   11.5     the budget guard fires at the token boundary, before embed: no row is written at `limit`
//   11.6/C5  the SSB6 consistency guards of 2.3, budget-mode cases, each mutation-proven: every
//            hostile blob violates exactly one guard, so the build with that guard removed restores it
//
// Rows are read through the page-index peek, addressed by §3.1's formula written in the test; nothing
// drawn is graded by admission in a pool sized to exactly the valid restore.

#include "pkv_budget_c_helpers.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace pkv;
using namespace pkv::bc;

// ================================================================================================
// 11.5: a budget holder resting at its limit with a pending token is refused at the token boundary,
// SSLM_KV_BUDGET_EXCEEDED, out_tokens -1, and the row at `limit` is not written in any layer, K or V.
// The pool buffer is filled with 0x5A, and each limit is mid-page, so row `limit` lies in a page the
// holder already maps: the mutant that checks after the layer loop writes it there, and the peek sees
// the write. Two holders: origin 0 with budget 17, and an adopter at origin 1,000 with budget 512.
// ================================================================================================
void Cell115() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	struct Case {
		int32_t origin;
		int32_t budget;
	};
	for (const Case& c : {Case{0, 17}, Case{1000, 512}}) {
		const int64_t limit = Limit(fx, c.origin, c.budget);
		PKV_CHECK_MSG(limit % fx.B() != 0, "11.5: construction: the limit %lld must be mid-page", static_cast<long long>(limit));
		for (uint32_t layer_budget : {1u, fx.geo.layers}) {
			Rig rig(fx, 300, 0x5A);
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), c.budget, &s), SSLM_OK);
			if (!rig.Seq(s)) return;
			if (c.origin) {
				sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(c.origin)));
				if (!px) return;
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
			} else {
				PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(115, 1, kVocab), 1), SSLM_OK);
			}
			sslm_status st = SSLM_OK;
			const std::vector<int32_t> toks = RunToRefusal(fx.model, s, &st);
			PKV_CHECK_EQ(st, PKV_KV_BUDGET_EXCEEDED);
			const std::vector<uint8_t> before_blob = Save(s);
			PKV_CHECK_EQ(ParseBlobWithHidden(before_blob, HiddenSize(fx)).context_length, limit);
			// the page holding row `limit`, and that row's bytes in every (layer, half, head)
			uint32_t page = 0, mapped = 0;
			PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, static_cast<uint32_t>(limit / fx.B()), &page, &mapped), SSLM_OK);
			auto row_bytes = [&]() {
				const std::vector<uint8_t> pg = PeekPage(fx, *rig.Pool(), page);
				std::vector<uint8_t> out;
				for (uint32_t l = 0; l < fx.geo.layers; ++l)
					for (uint32_t half = 0; half < 2; ++half)
						for (uint32_t h = 0; h < fx.geo.kv_heads; ++h) {
							const uint8_t* r = pg.data() + InPage(fx, l, half, h, limit);
							out.insert(out.end(), r, r + fx.geo.head_dim);
						}
				return out;
			};
			const std::vector<uint8_t> before = row_bytes();
			bool untouched = true;
			for (uint8_t x : before) untouched = untouched && x == 0x5A;
			PKV_CHECK_MSG(untouched, "11.5: construction: row %lld must hold the pool's 0x5A before the call", static_cast<long long>(limit));
			// The refused call, at the layer budget under test.
			sslm_decode_params p{};
			p.layer_budget = static_cast<int32_t>(layer_budget);
			sslm_seq b[1] = {s};
			int32_t tok = 0;
			PKV_CHECK_EQ(sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok), PKV_KV_BUDGET_EXCEEDED);
			PKV_CHECK_EQ(tok, -1);
			// kills: the budget check made after the layer loop (or after embed and layer 0): row `limit`
			// is written before the refusal
			PKV_CHECK_MSG(row_bytes() == before, "11.5 (origin %d, budget %d, layer_budget %u): row %lld was written",
			              c.origin, c.budget, layer_budget, static_cast<long long>(limit));
			// and the holder's state did not move (no layer_index advance, no context_length change)
			PKV_CHECK_MSG(Save(s) == before_blob, "11.5 (origin %d, budget %d): the refused call changed the saved state", c.origin,
			              c.budget);
			(void)toks;
			(void)mapped;
		}
	}
}

// ================================================================================================
// 11.6/C5: the SSB6 consistency guards (§3.7 restore validation), budget mode, one guard per blob.
//
// The base blobs are real budget saves, edited in one field each so that every other guard still
// passes. Each is refused SSLM_INVALID_ARGUMENT with nothing drawn: the pool is sized to exactly the
// valid blob's private restore (prefix pages + R(512) + E), and straight after every refusal the valid
// blob is admitted there. The build with that one guard removed restores the hostile blob, and the
// status check kills it.
//
// Not separable here, and why:
//   - a negative budget: origin + budget < origin <= context_length, so the context guard refuses it
//     too; only budget 0 isolates "1 <= budget" (asserted refused, not claimed as that guard's vitality);
//   - an inconsistent kv_positions isolates its guard only when the exact-size check reads kv_positions;
//     a size check derived from context_length refuses it as well. The case keeps the size consistent
//     with the edited kv_positions, so it separates the guard under the first reading.
// ================================================================================================

constexpr int32_t kBudget = 512;
constexpr int32_t kOrigin = 1000;

std::vector<uint8_t> WithLe32(std::vector<uint8_t> b, size_t at, uint32_t v) {
	PutLe32(b, at, v);
	return b;
}
std::vector<uint8_t> WithLe64(std::vector<uint8_t> b, size_t at, uint64_t v) {
	PutLe64(b, at, v);
	return b;
}

// kv_positions := L' + 1 with the canonical rows re-laid out at L' + 1 (one zero row appended to every
// (layer, half, head) run), so the blob's size agrees with the edited field.
std::vector<uint8_t> WithExtraPosition(const Fixture& fx, const std::vector<uint8_t>& b) {
	const BlobView v = ParseBlobWithHidden(b, HiddenSize(fx));
	std::vector<uint8_t> out(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(v.kv_offset));
	PutLe64(out, v.kv_offset - 8, v.kv_positions + 1);
	const size_t D = fx.geo.head_dim, run = static_cast<size_t>(v.kv_positions) * D;
	for (uint32_t i = 0; i < fx.geo.layers * 2u * fx.geo.kv_heads; ++i) {
		const uint8_t* src = b.data() + v.kv_offset + i * run;
		out.insert(out.end(), src, src + run);
		out.insert(out.end(), D, 0);
	}
	return out;
}

void Cell116Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t cap = fx.geo.context_cap;
	// Base states: an adopter at the origin (L = 1,000, ready), the same 100 tokens on (L = 1,100), and a
	// fresh holder with a 10-token prompt (origin 0, L = 10).
	std::vector<uint8_t> at_origin, on100, fresh10;
	{
		Rig rig(fx, 300);
		sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(kOrigin)));
		sslm_seq s = nullptr, f = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), kBudget, &s), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), kBudget, &f), SSLM_OK);
		if (!px || !rig.Seq(s) || !rig.Seq(f)) return;
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		at_origin = Save(s);
		DecodeN(fx.model, s, 101);
		on100 = Save(s);
		PKV_CHECK_EQ(PrefillAll(fx.model, f, Stream(116, 10, kVocab), kChunk), SSLM_OK);
		fresh10 = Save(f);
	}
	PKV_CHECK_EQ(ParseBlobWithHidden(on100, HiddenSize(fx)).context_length, kOrigin + 100);
	struct Hostile {
		const char* guard;
		std::vector<uint8_t> blob;
		const std::vector<uint8_t>* base;
		bool isolates;  // false: refused, but another guard refuses it too
	};
	const size_t kMode = 156, kBud = 160, kOri = 164;
	std::vector<Hostile> cases = {
	    {"1 <= budget (budget 0 at the origin)", WithLe32(at_origin, kBud, 0), &at_origin, true},
	    {"budget <= cap (cap + 1)", WithLe32(on100, kBud, static_cast<uint32_t>(cap + 1)), &on100, true},
	    {"budget INT32_MAX", WithLe32(on100, kBud, 0x7FFFFFFFu), &on100, true},
	    {"(not isolating) budget -1", WithLe32(at_origin, kBud, 0xFFFFFFFFu), &at_origin, false},
	    {"0 <= origin (origin -1)", WithLe64(fresh10, kOri, ~uint64_t{0}), &fresh10, true},
	    {"origin <= context_length (L + 1)", WithLe64(on100, kOri, kOrigin + 101), &on100, true},
	    {"context_length <= min(origin + budget, cap) (budget 99)", WithLe32(on100, kBud, 99), &on100, true},
	    {"kv_mode in {0, 1} (2)", WithLe32(on100, kMode, 2), &on100, true},
	    {"kv_mode in {0, 1} (0xFFFFFFFF)", WithLe32(on100, kMode, 0xFFFFFFFFu), &on100, true},
	    {"kv_positions == context_length + (layer_index > 0) (L' + 1, size kept consistent)", WithExtraPosition(fx, on100), &on100, true},
	    {"exact size (one byte short)", std::vector<uint8_t>(on100.begin(), on100.end() - 1), &on100, true},
	    {"exact size (one row short)", std::vector<uint8_t>(on100.begin(), on100.end() - static_cast<std::ptrdiff_t>(fx.BytesPerToken())), &on100, true},
	    {"exact size (one byte long)", [&] { std::vector<uint8_t> b = on100; b.push_back(0); return b; }(), &on100, true},
	};
	for (const Hostile& h : cases) {
		const BlobView v = ParseBlobWithHidden(*h.base, HiddenSize(fx));
		const int64_t E = MaterializedPages(fx, v.origin, kBudget);
		const int64_t pages = PrefixPages(fx, kOrigin) + fx.R(kBudget) + E;
		Rig rig(fx, static_cast<uint32_t>(pages));
		sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(kOrigin)));
		if (!px) return;
		for (int with = 0; with < 2; ++with) {
			sslm_seq r = OutSentinel<sslm_seq>();  // non-null; the verb must null it on the refusal
			const sslm_status st = with ? sslm_seq_restore_shared(fx.model, rig.Pool(), h.blob.data(), h.blob.size(), px, &r, nullptr)
			                            : sslm_seq_restore(fx.model, rig.Pool(), h.blob.data(), h.blob.size(), &r);
			// kills: the guard named by the case removed (the restore admits the hostile blob); a refusal
			// that returns before writing *out (the preset left standing)
			PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT && r == nullptr, "11.6 %s (%s): status %d%s", h.guard, with ? "handle" : "no handle",
			              static_cast<int>(st), h.isolates ? "" : " (a second guard refuses this one too)");
			if (r && r != OutSentinel<sslm_seq>()) sslm_seq_release(r);
			// nothing drawn: the valid blob's private restore still fits exactly
			sslm_seq ok = nullptr;
			// kills: a guard placed after the first draw, without the rollback
			PKV_CHECK_MSG(sslm_seq_restore(fx.model, rig.Pool(), h.base->data(), h.base->size(), &ok) == SSLM_OK,
			              "11.6 %s (%s): the valid blob no longer fits after the refusal (pages leaked)", h.guard,
			              with ? "handle" : "no handle");
			if (ok) sslm_seq_release(ok);
		}
	}
}

PKV_CELL("11.5", "C5", Cell115);
PKV_CELL("11.6/C5", "C5", Cell116Budget);

}  // namespace
