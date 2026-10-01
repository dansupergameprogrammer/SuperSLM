// Paged-KV plan (rev 16.1) §7 dimension 11, the legacy holders' guard-vitality cells (step C4):
// 11.2 and 11.6's whole_reserve cases.
//
// Each guard is shown alive by a construction only it can catch, and each refusal is graded by
// its status, by blob or page bytes read through the C4 seams, and by the legacy-create admission
// count (§8); never the stats verbs.

#include "pkv_legacy_b_helpers.h"

#include <climits>
#include <functional>

namespace {

using namespace pkv;
using namespace pkv::legacy_b;

// ---- 11.2 [C4] -------------------------------------------------------------------------------------
//
// The empty-reserve guard at map time (§3.6): an empty reserve below `limit` is unreachable by
// §3.4, and the map still checks it, returning SSLM_KV_POOL_EXHAUSTED with no row written. The
// drain seam moves the holder's whole reserve back to the pool and leaves its table. The holder
// has filled page 0 (16 positions) and emitted its ready token, so its next write, row 16, needs
// a page from the reserve: a whole-token decode is refused SSLM_KV_POOL_EXHAUSTED, and so is a
// one-token prefill on a second holder in the same state. "No row written" is read two ways: the
// re-saved blob's context_length, layer_index and K/V section equal the blob saved before the
// refusal; and through the page-index peek, the holder's page 0 is unchanged and every other page
// of the pool (whose buffer was filled with 0x5A before sslm_kv_pool_create) still reads 0x5A, so
// no row landed in any page. (A reserve page is clean, so the drain returns it unpoisoned, §3.3.)
// Mutant killed: the guard removed -- the map takes a page from an empty reserve and the write
// lands in a page it does not own (the peek), or advances the holder (the blob).

void ExpectNoRowWritten(const Fixture& fx, sslm_kv_pool pool, sslm_seq s, const std::vector<uint8_t>& before,
                        uint32_t own_page, const std::vector<uint8_t>& own_bytes, const char* what) {
	std::vector<uint8_t> after;
	PKV_CHECK(SaveBlob(s, &after));
	const BlobView a = ParseBlobWithHidden(after, HiddenSize(fx)), b = ParseBlobWithHidden(before, HiddenSize(fx));
	PKV_CHECK_MSG(a.ok && b.ok && a.context_length == b.context_length && a.layer_index == b.layer_index,
	              "11.2 %s: the refusal moved context_length or layer_index", what);
	// kills: the guard removed -- a row written into the holder's state
	PKV_CHECK_MSG(a.ok && b.ok && after.size() == before.size() &&
	                  std::memcmp(after.data() + a.kv_offset, before.data() + b.kv_offset, after.size() - a.kv_offset) == 0,
	              "11.2 %s: the K/V section changed across the refusal", what);
	const size_t page_bytes = fx.PageBytes();
	std::vector<uint8_t> page(page_bytes);
	bool own_same = false, others_clean = true;
	for (uint32_t i = 0; i < static_cast<uint32_t>(fx.CapPages()); ++i) {
		if (sslm_pkv_test_only_peek_page_bytes(pool, i, page.data(), page_bytes) != SSLM_OK) {
			others_clean = false;
			continue;
		}
		if (i == own_page) {
			own_same = page == own_bytes;
			continue;
		}
		for (uint8_t v : page) others_clean = others_clean && v == 0x5A;
	}
	PKV_CHECK_MSG(own_same, "11.2 %s: the holder's page 0 changed", what);
	// kills: the guard removed -- a row written into a page the holder does not own
	PKV_CHECK_MSG(others_clean, "11.2 %s: a pool page other than the holder's reads other than 0x5A", what);
}

void Cell112EmptyReserveGuard() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int via_prefill = 0; via_prefill < 2; ++via_prefill) {
		const char* what = via_prefill ? "one-token prefill" : "decode";
		LegacyPool pool(fx.model, 1, 0x5A);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (!s) continue;
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(91, static_cast<int32_t>(fx.B()), fx.vocab), 64), SSLM_OK);
		const int32_t ready = NextToken(fx.model, s);  // the ready token writes no row
		PKV_CHECK(ready >= 0);
		std::vector<uint8_t> before;
		PKV_CHECK(SaveBlob(s, &before));
		uint32_t own = 0, mapped = 0;
		PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, 0, &own, &mapped), SSLM_OK);
		std::vector<uint8_t> own_bytes(fx.PageBytes());
		PKV_CHECK_EQ(sslm_pkv_test_only_peek_page_bytes(pool.pool, own, own_bytes.data(), own_bytes.size()), SSLM_OK);
		PKV_CHECK_EQ(sslm_pkv_test_only_drain_reserve(s), SSLM_OK);
		if (!via_prefill) {
			sslm_decode_params p{};
			p.layer_budget = static_cast<int32_t>(fx.geo.layers);
			sslm_seq b[1] = {s};
			int32_t tok = 0;
			// kills: the guard removed (the decode maps from an empty reserve)
			PKV_CHECK_EQ(sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok), SSLM_KV_POOL_EXHAUSTED);
			PKV_CHECK_EQ(tok, -1);
		} else {
			const int32_t t = 7;
			int32_t consumed = -1;
			// kills: the guard removed (the prefill maps from an empty reserve)
			PKV_CHECK_EQ(sslm_prefill(fx.model, s, &t, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed), SSLM_KV_POOL_EXHAUSTED);
			PKV_CHECK_EQ(consumed, 0);
		}
		ExpectNoRowWritten(fx, pool.pool, s, before, own, own_bytes, what);
		sslm_seq_release(s);
	}
}

// ---- 11.6 [C4 whole_reserve cases] -------------------------------------------------------------------
//
// The SSB6 consistency guards of 2.3 (§3.7 "Restore validation"), each shown alive on a
// hand-built SSB6 whole_reserve blob (rev 16: no legacy holder writes SSB6). The control is a
// legacy adopter of the 1,000-token prefix that has emitted 3 tokens (the ready one writes no
// row, so context_length 1,002), saved resting and again mid-token (L' = 1,003), turned into
// SSB6 with origin 1,000: it restores SSLM_OK and continues as its unsaved twin. Each hostile
// case differs from that control in one field only (or, for kv_positions, in the field and a K/V
// section rebuilt to match it, so the size check cannot catch it), so the guard named is the only
// one that can refuse it, and the mutant with that guard removed accepts. Each is refused SSLM_INVALID_ARGUMENT with a null
// handle and nothing drawn: a 1-block pool still admits its one legacy create afterwards.
//   budget 0, -1, cap + 1, INT32_MAX ("1 <= budget <= cap");
//   whole_reserve with budget cap - 1 ("budget == cap when kv_mode is whole_reserve": with origin
//     1,000 the limit is still the cap, so nothing else rejects it);
//   origin -1 and context_length + 1 ("0 <= origin <= context_length");
//   kv_mode 2 and 0xFFFFFFFF ("kv_mode in {0, 1}");
//   kv_positions L' + 1 and L' - 1, each with a matching section ("kv_positions consistent with
//     context_length and layer_index"; on the mid-token blob L' - 1 is the mid-token row dropped);
//   one byte short, one row short and one trailing byte ("the exact size").
// Not separable on a whole_reserve blob, and stated rather than claimed: "context_length above
// min(origin + budget, cap)" -- with budget = cap the bound is the cap, which the check made
// today already enforces, so the new guard's mutant is equivalent here. The case is still run
// (context_length cap + 1) and must be refused; it separates on budget blobs (11.6's C5 half).

void Cell116WholeReserveGuards() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t cap = fx.geo.context_cap;
	for (int mid = 0; mid < 2; ++mid) {
		const char* base_name = mid ? "mid-token" : "resting";
		LegacyPool home(fx.model, 3);
		sslm_prefix px = nullptr;
		sslm_seq orig = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &home.pool, &px), SSLM_OK);
		if (!px) continue;
		PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, 1000, fx.vocab), 64), SSLM_OK);
		PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &home.pool, &orig), SSLM_OK);
		if (orig) PKV_CHECK_EQ(sslm_seq_adopt_prefix(orig, px), SSLM_OK);
		sslm_prefix_release(px);
		if (!orig) continue;
		NextTokens(fx.model, orig, 3);
		if (mid) EnterMidToken(fx.model, orig);
		std::vector<uint8_t> ssb5;
		PKV_CHECK(SaveBlob(orig, &ssb5));
		const int64_t ctx = static_cast<int64_t>(Le64(ssb5, 60));
		const int64_t lp = BlobLPrime(ssb5);
		PKV_CHECK_EQ(ctx, 1002);
		PKV_CHECK_EQ(lp, mid ? 1003 : 1002);
		const std::vector<uint8_t> control = Ssb6FromSsb5(ssb5, fx, 1000, lp);
		const size_t kvp = Ssb6KvPositionsOffset(control, fx);
		{  // the control: accepted, and the same holder
			LegacyPool pool(fx.model, 1);
			sslm_seq r = nullptr;
			PKV_CHECK_MSG(Restore(fx, &pool.pool, control, &r) == SSLM_OK, "11.6 %s: the control blob is refused", base_name);
			if (r) {
				std::vector<uint8_t> resaved;
				PKV_CHECK(SaveBlob(r, &resaved));
				PKV_CHECK_MSG(resaved == ssb5, "11.6 %s: the control's re-save is not the original's SSB5", base_name);
				PKV_CHECK_MSG(NextTokens(fx.model, r, 4) == NextTokens(fx.model, orig, 4), "11.6 %s: control continuation",
				              base_name);
				sslm_seq_release(r);
			}
		}
		sslm_seq_release(orig);
		struct Case {
			const char* name;
			std::function<std::vector<uint8_t>()> make;
		};
		auto edit = [&](std::function<void(std::vector<uint8_t>&)> f) {
			return [&control, f]() {
				std::vector<uint8_t> b = control;
				f(b);
				return b;
			};
		};
		const std::vector<Case> cases = {
		    {"budget 0", edit([](std::vector<uint8_t>& b) { PutLe32(b, 160, 0); })},
		    {"budget -1", edit([](std::vector<uint8_t>& b) { PutLe32(b, 160, 0xFFFFFFFFu); })},
		    {"budget cap + 1", edit([&](std::vector<uint8_t>& b) { PutLe32(b, 160, static_cast<uint32_t>(cap + 1)); })},
		    {"budget INT32_MAX", edit([](std::vector<uint8_t>& b) { PutLe32(b, 160, static_cast<uint32_t>(INT32_MAX)); })},
		    {"whole_reserve with budget cap - 1",
		     edit([&](std::vector<uint8_t>& b) { PutLe32(b, 160, static_cast<uint32_t>(cap - 1)); })},
		    {"origin -1", edit([](std::vector<uint8_t>& b) { PutLe64(b, 164, ~uint64_t{0}); })},
		    {"origin context_length + 1",
		     edit([&](std::vector<uint8_t>& b) { PutLe64(b, 164, static_cast<uint64_t>(ctx + 1)); })},
		    {"kv_mode 2", edit([](std::vector<uint8_t>& b) { PutLe32(b, 156, 2); })},
		    {"kv_mode 0xFFFFFFFF", edit([](std::vector<uint8_t>& b) { PutLe32(b, 156, 0xFFFFFFFFu); })},
		    {"kv_positions L' + 1, section to match", [&]() { return Ssb6FromSsb5(ssb5, fx, 1000, lp + 1); }},
		    {"kv_positions L' - 1, section to match", [&]() { return Ssb6FromSsb5(ssb5, fx, 1000, lp - 1); }},
		    {"one byte short", edit([](std::vector<uint8_t>& b) { b.pop_back(); })},
		    {"one row short", edit([&](std::vector<uint8_t>& b) { b.resize(b.size() - fx.BytesPerToken()); })},
		    {"one trailing byte", edit([](std::vector<uint8_t>& b) { b.push_back(0); })},
		    {"context_length cap + 1 (equivalent on whole_reserve)",
		     edit([&](std::vector<uint8_t>& b) { PutLe64(b, 60, static_cast<uint64_t>(cap + 1)); })},
		};
		PKV_CHECK_EQ(Le64(control, kvp), lp);
		LegacyPool pool(fx.model, 1);
		for (const Case& c : cases) {
			const std::vector<uint8_t> blob = c.make();
			sslm_seq r = OutSentinel<sslm_seq>();  // non-null; the verb must null it on the refusal
			const sslm_status st = Restore(fx, &pool.pool, blob, &r);
			// kills: the guard that alone refuses this case, removed
			PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "11.6 %s, %s: status %d, want SSLM_INVALID_ARGUMENT", base_name, c.name,
			              static_cast<int>(st));
			// kills: a refusal that returns before writing *out (the caller's value left standing)
			PKV_CHECK_MSG(r == nullptr, "11.6 %s, %s: the out-handle is not null", base_name, c.name);
			if (r && r != OutSentinel<sslm_seq>()) sslm_seq_release(r);
			// nothing drawn: the one block's pages are all still free
			PKV_CHECK_MSG(CountLegacyCreates(fx.model, &pool.pool) == 1, "11.6 %s, %s: the refusal drew pages", base_name,
			              c.name);
		}
	}
}

PKV_CELL("11.2/C4", "C4", Cell112EmptyReserveGuard);
PKV_CELL("11.6/C4", "C4", Cell116WholeReserveGuards);

}  // namespace
