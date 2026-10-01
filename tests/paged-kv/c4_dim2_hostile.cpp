// Paged-KV plan (rev 16.1) §7 dimension 2 (trust boundaries and hostile inputs), the parts step C4
// owns: the SSB6 reader's whole_reserve cases, every older blob restored into a legacy pool, and
// the legacy pool's u32 page-count refusal.
//
//   2.3/C4  hand-built whole_reserve SSB6 blobs, one field edited each -> SSLM_INVALID_ARGUMENT,
//           nothing drawn (graded by the legacy-create admission count)
//   2.4     SSB4/3/2 and 1.9.0's SSB5 restore into a legacy pool and continue as v1.11.0 did; an
//           SSB5 of another model -> SSLM_RESTORE_MODEL_MISMATCH
//   2.9     legacy sslm_kv_pool_create at cap 32768 and block_count 2,097,152 -> refused before
//           the buffer-size check
//
// 11.6's mutation-proving of each consistency guard is a separate cell; these cells grade the
// statuses and "nothing drawn" only.

#include "pkv_legacy_a_helpers.h"

#include <climits>

namespace {

using namespace pkv;
using namespace pkv::legacy_a;

// ---- 2.3 [C4 whole_reserve cases] -------------------------------------------------------------

struct Ssb6Case {
	std::string name;
	std::vector<uint8_t> blob;
};

// The editable fields of a hand-built SSB6 (§3.7; pkv_common.h's layout).
struct Ssb6Fields {
	size_t kv_positions_at = 0;  // the u64 kv_positions
	size_t rows_at = 0;          // the first canonical row byte
};

inline Ssb6Fields LocateSsb6(const std::vector<uint8_t>& b, const Fixture& fx) {
	const BlobView v = ParseBlobWithHidden(b, HiddenSize(fx));
	Ssb6Fields f;
	f.rows_at = v.kv_offset;
	f.kv_positions_at = v.kv_offset - 8;
	return f;
}

// Each 2.3 whole_reserve case, as an edit of the valid blob `base` (L = context_length, L' = the
// stored kv_positions). Where a case can be built with every other field consistent, it is, so
// the edited field is the blob's one defect.
std::vector<Ssb6Case> WholeReserveCases(const std::vector<uint8_t>& base, const Fixture& fx) {
	std::vector<Ssb6Case> cases;
	const Ssb6Fields f = LocateSsb6(base, fx);
	const int64_t L = static_cast<int64_t>(Le64(base, 60));
	const uint64_t Lp = Le64(base, f.kv_positions_at);
	const int64_t cap = fx.geo.context_cap;
	const size_t bpt = fx.BytesPerToken();
	auto with = [&](const char* name, const std::function<void(std::vector<uint8_t>&)>& edit) {
		std::vector<uint8_t> b = base;
		edit(b);
		cases.push_back({name, std::move(b)});
	};
	// Canonical rows are [layer][K|V][head][pos < L'][d] (§3.7), so a row is H_kv * 2 * layers runs of
	// D bytes; changing L' re-lays every run. Rebuild the row section at a new L' from the old one,
	// zero-padding new positions, so the size is exact for the new kv_positions.
	auto relay = [&](std::vector<uint8_t>& b, uint64_t new_lp) {
		const size_t D = fx.geo.head_dim, runs = size_t{fx.geo.layers} * 2u * fx.geo.kv_heads;
		std::vector<uint8_t> rows(runs * new_lp * D, 0);
		for (size_t r = 0; r < runs; ++r)
			for (uint64_t p = 0; p < std::min<uint64_t>(Lp, new_lp); ++p)
				std::memcpy(rows.data() + (r * new_lp + p) * D, base.data() + f.rows_at + (r * Lp + p) * D, D);
		b.resize(f.rows_at);
		b.insert(b.end(), rows.begin(), rows.end());
		PutLe64(b, f.kv_positions_at, new_lp);
	};

	// budget 0, negative or above cap
	with("budget 0", [&](auto& b) { PutLe32(b, 160, 0); });
	with("budget -1", [&](auto& b) { PutLe32(b, 160, static_cast<uint32_t>(-1)); });
	with("budget INT32_MIN", [&](auto& b) { PutLe32(b, 160, static_cast<uint32_t>(INT32_MIN)); });
	with("budget cap + 1", [&](auto& b) { PutLe32(b, 160, static_cast<uint32_t>(cap + 1)); });
	with("budget INT32_MAX", [&](auto& b) { PutLe32(b, 160, static_cast<uint32_t>(INT32_MAX)); });
	// a whole_reserve blob with budget != cap (each otherwise in [1, cap] and with limit >= L)
	with("whole_reserve budget cap - 1", [&](auto& b) { PutLe32(b, 160, static_cast<uint32_t>(cap - 1)); });
	with("whole_reserve budget 512", [&](auto& b) { PutLe32(b, 160, 512u); });
	// origin negative or above context_length
	with("origin -1", [&](auto& b) { PutLe64(b, 164, static_cast<uint64_t>(int64_t{-1})); });
	with("origin INT64_MIN", [&](auto& b) { PutLe64(b, 164, static_cast<uint64_t>(INT64_MIN)); });
	with("origin L + 1", [&](auto& b) { PutLe64(b, 164, static_cast<uint64_t>(L + 1)); });
	with("origin cap + 1", [&](auto& b) { PutLe64(b, 164, static_cast<uint64_t>(cap + 1)); });
	// context_length above min(origin + budget, cap) = cap for whole_reserve; kv_positions and the
	// rows are grown to match, so only the limit is violated
	with("context_length cap + 1", [&](auto& b) {
		PutLe64(b, 60, static_cast<uint64_t>(cap + 1));
		relay(b, static_cast<uint64_t>(cap + 1));
	});
	// kv_mode outside {0, 1}
	with("kv_mode 2", [&](auto& b) { PutLe32(b, 156, 2u); });
	with("kv_mode UINT32_MAX", [&](auto& b) { PutLe32(b, 156, UINT32_MAX); });
	// kv_positions inconsistent with context_length and layer_index (L' = L + (layer_index > 0))
	with("kv_positions L' + 1, rows to match", [&](auto& b) { relay(b, Lp + 1); });
	with("kv_positions L' - 1, rows to match", [&](auto& b) { relay(b, Lp - 1); });
	with("kv_positions 0, no rows", [&](auto& b) { relay(b, 0); });
	with("kv_positions L' + 1, rows unchanged", [&](auto& b) { PutLe64(b, f.kv_positions_at, Lp + 1); });
	with("kv_positions UINT64_MAX", [&](auto& b) { PutLe64(b, f.kv_positions_at, UINT64_MAX); });
	with("layer_index 1 with kv_positions L", [&](auto& b) { PutLe32(b, 68, 1u); });
	// truncated K/V, and the exact-size twin
	with("truncated: last byte", [&](auto& b) { b.pop_back(); });
	with("truncated: last row", [&](auto& b) { b.resize(b.size() - bpt); });
	with("truncated: every row", [&](auto& b) { b.resize(f.rows_at); });
	with("truncated: inside kv_positions", [&](auto& b) { b.resize(f.kv_positions_at + 4); });
	with("one byte past the rows", [&](auto& b) { b.push_back(0); });
	return cases;
}

// 2.3 [C4 whole_reserve cases]. No library writer emits a whole_reserve SSB6 (rev 16), so the
// blob is hand-built from v1.11.0's pinned SSB5 of the lifecycle scenario's adopted state (L = 300,
// origin 300; pkv_common.h's Ssb6WholeReserveFromSsb5). The unedited blob is the must-accept
// control (§3.9 A2: restored, privately, continuing as the reference did). Each edited copy is
// refused SSLM_INVALID_ARGUMENT with no handle, and nothing is drawn: the one-block pool still
// admits its one legacy create, so even a one-page draw left behind shows as 0 admitted (§8).
void Cell23Legacy() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const std::vector<uint8_t> ssb5 = Pin("v1.11.0", "lifecycle", "adopt300");
	const RefRecord* cont = RefLookup(Reference("v1.11.0", fx), "lifecycle", "adopt300+decode8");
	const std::vector<uint8_t> base = Ssb6WholeReserveFromSsb5(ssb5, fx, 300);
	PKV_CHECK_MSG(!base.empty() && cont, "the hand-built whole_reserve SSB6 and its continuation");
	if (base.empty() || !cont) return;

	LegacyPool pool(fx.model, 1);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;

	// The control: the valid blob restores and continues as v1.11.0's live holder did.
	{
		sslm_seq r = nullptr;
		// kills: an SSB6 reader that refuses whole_reserve blobs (or every SSB6), which would make
		// each refusal below vacuous
		PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, base.data(), base.size(), &r), SSLM_OK);
		if (r) {
			PKV_CHECK_MSG(NextTokens(fx.model, r, 8) == cont->tokens, "restored whole_reserve SSB6 continues as v1.11.0");
			PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 0);  // it holds the whole reservation
			PKV_CHECK_EQ(sslm_seq_release(r), SSLM_OK);
		}
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 1);
	}

	for (const Ssb6Case& c : WholeReserveCases(base, fx)) {
		sslm_seq r = nullptr;
		const sslm_status st = sslm_seq_restore(fx.model, &pool.pool, c.blob.data(), c.blob.size(), &r);
		// kills: the reader missing this validation (a restore admitted, or another status)
		PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "%s: restore -> %d, want SSLM_INVALID_ARGUMENT", c.name.c_str(),
		              static_cast<int>(st));
		PKV_CHECK_MSG(r == nullptr, "%s: a refused restore hands back no handle", c.name.c_str());
		if (r) sslm_seq_release(r);
		// kills: validation after the draw (the refused restore leaves pages drawn)
		const int admitted = CountLegacyCreates(fx.model, &pool.pool);
		PKV_CHECK_MSG(admitted == 1, "%s: after the refusal the pool admits %d legacy creates, want 1", c.name.c_str(),
		              admitted);
	}
}

// ---- 2.4 [C4] ---------------------------------------------------------------------------------

// Restores `blob` into a fresh one-block legacy pool of `fx` and checks the holder continues with
// v1.11.0's continuation tokens, re-saves as 1.9.0's SSB5 with the reference's rows (and, when
// `blob_sha` is non-empty, the reference's whole blob), and returns every page on release.
void RestoreAndContinue(const Fixture& fx, const char* what, const std::vector<uint8_t>& blob, const RefRecord* at,
                        const RefRecord* cont, const std::string& blob_sha) {
	PKV_CHECK_MSG(!blob.empty() && at && cont, "%s: blob and reference present", what);
	if (blob.empty() || !at || !cont) return;
	LegacyPool pool(fx.model, 1);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_seq r = nullptr;
	const sslm_status st = sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r);
	// kills: a reader that drops an older magic, or a scatter that cannot fill a one-block pool
	PKV_CHECK_MSG(st == SSLM_OK, "%s: restore into a legacy pool -> %d", what, static_cast<int>(st));
	if (!r) return;
	std::vector<uint8_t> resaved;
	PKV_CHECK(SaveBlob(r, &resaved));
	PKV_CHECK_MSG(IsMagic(resaved, "SSB5"), "%s: a legacy holder re-saves as 1.9.0's SSB5", what);
	std::vector<uint8_t> rows;
	PKV_CHECK(BlobRows(resaved, fx, HiddenSize(fx), at->context_length, &rows));
	// kills: a scatter that misplaces a row across a page boundary, or drops the rows past the
	// first page
	PKV_CHECK_MSG(Le64(resaved, 60) == static_cast<uint64_t>(at->context_length) &&
	                  Sha(rows.data(), rows.size()) == at->rows_sha,
	              "%s: the restored holder's rows equal v1.11.0's", what);
	if (!blob_sha.empty())
		PKV_CHECK_MSG(Sha(resaved.data(), resaved.size()) == blob_sha, "%s: re-save byte-equal to v1.11.0's", what);
	PKV_CHECK_MSG(NextTokens(fx.model, r, 8) == cont->tokens, "%s: continuation equals v1.11.0's", what);
	PKV_CHECK_EQ(sslm_seq_release(r), SSLM_OK);
	PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 1);  // every page back
}

// 2.4 [C4]. SSB4/3/2 and 1.9.0's SSB5 restored into a legacy pool (a page pool from C4 on, §3.7:
// private, whole_reserve, origin 0). The SSB5 and SSB4 blobs are R0's pins (v1.11.0 and v1.8.1, the
// persist scenario's saved state and the saturating scenario's). No SSB3 or SSB2 writer is pinned,
// so those two are cut from the v1.8.1 SSB4 pin the way dim9_persistence_red.cpp R2-R4 cut them from
// a live blob (pkv_legacy_a_helpers.h, LegacyShapeFromSsb4): the saved state rests after a decoded
// token (current_token >= 0, layer_index 0), which SSB3 and SSB2 restore without a residual and
// which is not the residual-lost state those readers refuse. Each continues with v1.11.0's 8
// continuation tokens. Then an SSB5 of another model: the pkv_def pin restored into a pkv_qk pool
// -> SSLM_RESTORE_MODEL_MISMATCH, no handle, nothing drawn.
void Cell24() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const RefFile& ref = Reference("v1.11.0", fx);
	const RefRecord* saved = RefLookup(ref, "persist", "saved");
	const RefRecord* restored = RefLookup(ref, "persist", "restored");
	const RefRecord* cont = RefLookup(ref, "persist", "continuation8");
	const RefRecord* sat = RefLookup(ref, "saturating", "saturated");
	const RefRecord* sat_cont = RefLookup(ref, "saturating", "continuation8");

	const std::vector<uint8_t> ssb5 = Pin("v1.11.0", "persist", "saved");
	const std::vector<uint8_t> ssb4 = Pin("v1.8.1", "persist", "saved");
	PKV_CHECK(IsMagic(ssb5, "SSB5") && IsMagic(ssb4, "SSB4"));
	PKV_CHECK_MSG(Le32(ssb4, 68) == 0 && static_cast<int32_t>(Le32(ssb4, 72)) >= 0,
	              "the saved state rests after a decoded token (layer_index 0, current_token >= 0)");
	const std::vector<uint8_t> ssb3 = LegacyShapeFromSsb4(ssb4, fx, '3');
	const std::vector<uint8_t> ssb2 = LegacyShapeFromSsb4(ssb4, fx, '2');
	PKV_CHECK_MSG(!ssb3.empty() && !ssb2.empty(), "SSB3 and SSB2 cut from the SSB4 pin");

	// v1.11.0's restore-then-save of this very blob is the reference's "persist restored" record,
	// so the SSB5 re-save is compared whole.
	RestoreAndContinue(fx, "1.9.0 SSB5 (persist)", ssb5, saved, cont, restored ? restored->blob_sha : std::string());
	RestoreAndContinue(fx, "SSB4 (persist)", ssb4, saved, cont, "");
	RestoreAndContinue(fx, "SSB3 (persist)", ssb3, saved, cont, "");
	RestoreAndContinue(fx, "SSB2 (persist)", ssb2, saved, cont, "");
	RestoreAndContinue(fx, "1.9.0 SSB5 (saturating)", Pin("v1.11.0", "saturating", "saturated"), sat, sat_cont, "");
	RestoreAndContinue(fx, "SSB4 (saturating)", Pin("v1.8.1", "saturating", "saturated"), sat, sat_cont, "");

	// A legacy holder's 1.9.0 SSB5 from another model.
	const Fixture& qk = GetFixture("pkv_qk");
	if (!qk.ok) return;
	LegacyPool pool(qk.model, 1);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_seq r = nullptr;
	// kills: a reader that skips the model-hash check on the SSB5 path
	PKV_CHECK_EQ(sslm_seq_restore(qk.model, &pool.pool, ssb5.data(), ssb5.size(), &r), SSLM_RESTORE_MODEL_MISMATCH);
	PKV_CHECK(r == nullptr);
	if (r) sslm_seq_release(r);
	PKV_CHECK_EQ(CountLegacyCreates(qk.model, &pool.pool), 1);  // nothing drawn
}

// ---- 2.9 [C4] ---------------------------------------------------------------------------------

// 2.9 [C4]. Legacy sslm_kv_pool_create on the cap-32768 fixture (B = 16, 2,048 pages per block) at
// block_count = 2,097,152: the page count 2^32 exceeds UINT32_MAX while block_count * block_size
// (about 2.6e13 bytes) does not overflow, so the u32 refusal is the only one that can fire. §3.6:
// SSLM_INVALID_ARGUMENT, in the position of the block-product overflow refusal, before the
// buffer-size check, nothing allocated. The buffer is deliberately far too small, so a build that
// reaches the size check reports SSLM_BUFFER_TOO_SMALL instead. One block fewer (the threshold the
// test computes, 2,097,151) is not refused by the page count and reaches the size check.
void Cell29() {
	const Fixture& fx = GetFixture("pkv_32k");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.B(), 16);
	PKV_CHECK_EQ(fx.CapPages(), 2048);
	const uint64_t threshold = uint64_t{UINT32_MAX} / static_cast<uint64_t>(fx.CapPages());  // last admissible count
	PKV_CHECK_EQ(threshold, 2097151);
	const uint32_t over = static_cast<uint32_t>(threshold + 1);
	PKV_CHECK_EQ(over, 2097152);
	const size_t block = sslm_kv_block_size(fx.model);
	PKV_CHECK_MSG(block > 0 && size_t{over} <= SIZE_MAX / block, "block_count * block_size does not overflow");

	AlignedBuf buf(4096, 0x77);
	sslm_kv_pool pool = nullptr;
	const sslm_status st = sslm_kv_pool_create(fx.model, buf.p, buf.n, over, &pool);
	// kills: the page count truncated to u32 (2^32 -> 0 pages) or never checked, which reaches the
	// buffer-size check and reports SSLM_BUFFER_TOO_SMALL; a check placed after the size check
	PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "block_count %u -> %d, want SSLM_INVALID_ARGUMENT", over,
	              static_cast<int>(st));
	PKV_CHECK(pool == nullptr);
	if (pool) sslm_kv_pool_destroy(pool);
	bool untouched = true;
	for (size_t i = 0; i < buf.n; ++i) untouched = untouched && buf.p[i] == 0x77;
	PKV_CHECK_MSG(untouched, "the refused create wrote nothing into the caller's buffer");

	// The threshold, from below: one block fewer is admissible by page count and is refused by the
	// size check, so the u32 refusal sits exactly at UINT32_MAX pages.
	sslm_kv_pool pool2 = nullptr;
	// kills: an off-by-one threshold (>= 2^32 - 2048 pages refused, or a 2^31 cap)
	PKV_CHECK_EQ(sslm_kv_pool_create(fx.model, buf.p, buf.n, static_cast<uint32_t>(threshold), &pool2),
	             SSLM_BUFFER_TOO_SMALL);
	PKV_CHECK(pool2 == nullptr);
	if (pool2) sslm_kv_pool_destroy(pool2);
}

PKV_CELL("2.3/C4", "C4", Cell23Legacy);
PKV_CELL("2.4", "C4", Cell24);
PKV_CELL("2.9", "C4", Cell29);

}  // namespace
