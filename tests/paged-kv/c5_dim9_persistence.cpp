// Paged-KV plan (rev 16.1) step C1: dimension 9's budget-mode cells, owned by C5 (§7, §9.1).
//
//   9.1/C5   save -> restore -> save is byte-identical SSB6 for budget holders, adopted and nested,
//            resting and mid-token; tokens and rows against R0's reference or a legacy twin
//   9.3/C5   the SSB6 side of "old reader, new data": a budget holder's blob is SSB6 (box: v1.11.0
//            rejects it on magic)
//   9.4      an adopted budget sequence saved mid-budget runs to its limit and refuses one past
//   9.5/C5   the separating worst case on a budget holder with budget = cap (carrier row RB16-CELLS)
//   9.9      the strike's falsifier: the fixed case, pages graded by admission, the population
//   9.10     private restores that separate, graded by admission
//   9.11     materialized pages: drawn, returned at the first reset, adopt or release
//   9.14/C5  §3.9's other "keeps" rows, A16-A19 refused, A20's six crossings
//
// Grading (§7, §8): page counts by admission or the two-sided fill probe; bytes by the R0 reference,
// a legacy twin, an unsaved original or a fresh twin; never `sslm_kv_pool_stats`,
// `sslm_seq_kv_stats` or `*out_shared_pages`. Every pool a verdict reads is sized by the test from
// §3.4 / §3.7, written in pkv_budget_c_helpers.h. 9.9 and 9.10 run with the table sentinel on (7.13's
// [C5] part), so a read above `mapped` traps instead of passing.

#include "pkv_budget_c_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using namespace pkv;
using namespace pkv::bc;

// The table sentinel (7.13), on for the life of a cell.
struct SentinelOn {
	SentinelOn() { sslm_pkv_test_only_set_table_sentinel(1); }
	~SentinelOn() { sslm_pkv_test_only_set_table_sentinel(0); }
};

const char* Name(sslm_status s) {
	static char buf[16];
	std::snprintf(buf, sizeof buf, "%d", static_cast<int>(s));
	return buf;
}

// The fields §3.9's "keeps" rows compare straight after a verb, read from the holder's re-saved SSB6.
struct Kept {
	bool ok = false;
	uint32_t mode = 0;
	int32_t budget = 0;
	int64_t origin = 0;
	int64_t context_length = 0;
	uint32_t layer_index = 0;
};
Kept ReadKept(const Fixture& fx, const std::vector<uint8_t>& blob) {
	Kept k;
	const BlobView v = ParseBlobWithHidden(blob, HiddenSize(fx));
	if (!v.ok || !IsMagic(blob, "SSB6")) return k;
	k.ok = true;
	k.mode = v.kv_mode;
	k.budget = v.budget;
	k.origin = v.origin;
	k.context_length = v.context_length;
	k.layer_index = v.layer_index;
	return k;
}

// ---- the fixed cohort case (§3.7, 9.9, 9.10, 9.11) -------------------------------------------------
//
// A budget-512 sequence adopts the persona (origin 1,200, limit 1,712) and takes a 300-token user
// turn, so it rests ready at 1,500 and emits exactly 1 + (1,712 - 1,500) = 213 tokens before the
// refusal (§3.4). (Decoding to 1,500 instead leaves a pending token, which emits 212; the prefill is
// what makes the cell's 213 the §3.4 count.)

constexpr int32_t kCohortBudget = 512;
constexpr int64_t kPersonaLen = 1200;

struct Fixed {
	bool ok = false;
	std::vector<uint8_t> blob;        // SSB6 at 1,500
	Turn cont;                         // the unsaved original's continuation from 1,500
	std::vector<int32_t> readopt;      // its tokens after reset + re-adopt, to the refusal
	sslm_status readopt_refusal = SSLM_OK;
	std::vector<int32_t> next8;        // the first 8 continuation tokens
};

const Fixed& FixedCase(const Fixture& fx) {
	static Fixed f;
	static bool made = false;
	if (made) return f;
	made = true;
	Rig rig(fx, 600);
	rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
	if (!rig.chain.ok) return f;
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), kCohortBudget, &s), SSLM_OK);
	if (!rig.Seq(s)) return f;
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, rig.chain.persona), SSLM_OK);
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(91, 300, kVocab), kChunk), SSLM_OK);
	f.blob = Save(s);
	const Kept k = ReadKept(fx, f.blob);
	PKV_CHECK_MSG(k.ok && k.context_length == 1500 && k.origin == kPersonaLen && k.budget == kCohortBudget,
	              "fixed case: the saved state is not (L 1,500, origin 1,200, budget 512)");
	f.cont = NextTurn(fx.model, s);
	PKV_CHECK_EQ(f.cont.tokens.size(), 1 + (kPersonaLen + kCohortBudget - 1500));  // 213, §3.4
	PKV_CHECK_EQ(f.cont.refusal, PKV_KV_BUDGET_EXCEEDED);
	f.next8.assign(f.cont.tokens.begin(), f.cont.tokens.begin() + std::min<size_t>(8, f.cont.tokens.size()));
	PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, rig.chain.persona), SSLM_OK);
	f.readopt = RunToRefusal(fx.model, s, &f.readopt_refusal);
	PKV_CHECK_EQ(f.readopt.size(), 1 + kCohortBudget);  // 513
	f.ok = !f.blob.empty();
	return f;
}

// ---- the population (9.9, 9.14) ------------------------------------------------------------------
//
// Prefix lengths {0, 1, 15, 16, 17, 1,000, 1,008, 1,200} x budgets {1, 15, 16, 17, 512} x save points
// {at origin, mid, at limit}. The holder is a budget sequence that adopted a budget prefix of that
// length; "mid" is half its budget decoded, "at limit" decoded to the refusal. An empty prefix leaves
// nothing to decode from, so past the origin it first takes a one-token turn.

enum class Sp { kOrigin, kMid, kLimit };
const char* SpName(Sp sp) { return sp == Sp::kOrigin ? "origin" : sp == Sp::kMid ? "mid" : "limit"; }

constexpr int32_t kPopP[] = {0, 1, 15, 16, 17, 1000, 1008, 1200};
constexpr int32_t kPopB[] = {1, 15, 16, 17, 512};
constexpr Sp kPopSp[] = {Sp::kOrigin, Sp::kMid, Sp::kLimit};
constexpr int64_t kSlack = 3;  // pages a test-sized pool leaves free, so every probed count is >= 2

void DriveTo(const Fixture& fx, sslm_seq s, int32_t P, int32_t b, Sp sp) {
	if (sp == Sp::kOrigin) return;
	if (P == 0) {
		const std::vector<int32_t> one = Stream(81, 1, kVocab);
		PKV_CHECK_EQ(PrefillAll(fx.model, s, one, 1), SSLM_OK);
	}
	sslm_status st = SSLM_OK;
	if (sp == Sp::kMid) DecodeN(fx.model, s, b / 2, &st);
	else RunToRefusal(fx.model, s, &st);
}

struct Original {
	bool ok = false;
	std::vector<uint8_t> blob;  // SSB6 at the save point
	Turn turn;                  // the unsaved original's next turn
};

std::string CaseName(int32_t P, int32_t b, Sp sp) {
	return "P=" + std::to_string(P) + " budget=" + std::to_string(b) + " at " + SpName(sp);
}

// The original, in a roomy pool: adopt, drive, save, then its own (unsaved) next turn.
Original MakeOriginal(const Fixture& fx, int32_t P, int32_t b, Sp sp) {
	Original o;
	Rig rig(fx, static_cast<uint32_t>(fx.R(std::max(P, 1)) + fx.R(b) + 8));
	sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(P)));
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), b, &s), SSLM_OK);
	if (!px || !rig.Seq(s)) return o;
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
	DriveTo(fx, s, P, b, sp);
	o.blob = Save(s);
	o.turn = NextTurn(fx.model, s);
	o.ok = !o.blob.empty();
	return o;
}

// How a twin is restored. kHandle: sslm_seq_restore_shared with the matching handle. kNone:
// sslm_seq_restore. kNullShared: sslm_seq_restore_shared(null). kMismatch: restore_shared with a
// prefix of the same length and other content in the same pool.
enum class How { kHandle, kNone, kNullShared, kMismatch };
const char* HowName(How h) {
	return h == How::kHandle ? "matching handle" : h == How::kNone ? "no handle" : h == How::kNullShared ? "null handle" : "mismatched handle";
}

// The test-sized pool for a twin: prefix pages + R(budget) + E (private restores only) + the slack,
// plus the mismatched handle's own pages. After the restore exactly kSlack pages are free.
uint32_t TwinPoolPages(const Fixture& fx, int32_t P, int32_t b, How how) {
	int64_t n = PrefixPages(fx, P) + fx.R(b) + kSlack;
	if (how != How::kHandle) n += MaterializedPages(fx, P, b);
	if (how == How::kMismatch) n += PrefixPages(fx, P);
	return static_cast<uint32_t>(n);
}

struct Twin : Rig {
	sslm_prefix prefix = nullptr;
	sslm_seq seq = nullptr;
	sslm_status restore = SSLM_INVALID_ARGUMENT;
	PageSnap before;  // the prefix's pages before the restore
	Twin(const Fixture& fx, uint32_t pages) : Rig(fx, pages) {}
};

std::unique_ptr<Twin> MakeTwin(const Fixture& fx, int32_t P, int32_t b, const std::vector<uint8_t>& blob, How how,
                               bool reset_readopt, bool snap = false) {
	auto t = std::make_unique<Twin>(fx, TwinPoolPages(fx, P, b, how));
	t->prefix = t->Prefix(BudgetPrefix(fx, t->Pool(), PrefixTokens(P)));
	sslm_prefix other = how == How::kMismatch ? t->Prefix(BudgetPrefix(fx, t->Pool(), PrefixTokens(P, 1))) : nullptr;
	if (!t->prefix) return t;
	if (snap) t->before = SnapPrefix(fx, *t->Pool(), t->prefix);
	switch (how) {
		case How::kHandle:
			t->restore = sslm_seq_restore_shared(fx.model, t->Pool(), blob.data(), blob.size(), t->prefix, &t->seq, nullptr);
			break;
		case How::kNone:
			t->restore = sslm_seq_restore(fx.model, t->Pool(), blob.data(), blob.size(), &t->seq);
			break;
		case How::kNullShared:
			t->restore = sslm_seq_restore_shared(fx.model, t->Pool(), blob.data(), blob.size(), nullptr, &t->seq, nullptr);
			break;
		case How::kMismatch:
			t->restore = sslm_seq_restore_shared(fx.model, t->Pool(), blob.data(), blob.size(), other, &t->seq, nullptr);
			break;
	}
	t->Seq(t->seq);
	if (t->restore == SSLM_OK && reset_readopt) {
		PKV_CHECK_EQ(sslm_seq_reset(t->seq), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(t->seq, t->prefix), SSLM_OK);
	}
	return t;
}

// The free pages after the twin's reset and re-adopt: E (private restores, returned at the first
// reset) plus the slack. The original's reservation is R(budget) by §3.4, the same arithmetic.
int64_t FreeAfterReadopt(const Fixture& fx, int32_t P, int32_t b, How how) {
	return (how == How::kHandle ? 0 : MaterializedPages(fx, P, b)) + kSlack;
}

// ================================================================================================
// 9.1/C5: save -> restore -> save is byte-identical SSB6; resting and mid-token; adopted and nested.
// ================================================================================================

// Restores `blob` with the handle and without, re-saves each, and requires the re-save to equal the
// blob byte for byte; the twins' next `n` tokens equal `expect`.
void RoundTrip(const Fixture& fx, sslm_kv_pool* pool, sslm_prefix handle, const std::vector<uint8_t>& blob,
               const std::vector<int32_t>& expect, const char* what) {
	for (int with = 0; with < 2; ++with) {
		sslm_seq r = nullptr;
		const sslm_status st = with ? sslm_seq_restore_shared(fx.model, pool, blob.data(), blob.size(), handle, &r, nullptr)
		                            : sslm_seq_restore(fx.model, pool, blob.data(), blob.size(), &r);
		PKV_CHECK_MSG(st == SSLM_OK, "9.1 %s (%s): restore status %s", what, with ? "handle" : "no handle", Name(st));
		if (!r) continue;
		const std::vector<uint8_t> again = Save(r);
		// kills: restore drops or rewrites budget / origin / mode (rev 2's budget := limit), the
		// scatter misplaces a row, the save gathers from the wrong page after a shared restore
		PKV_CHECK_MSG(again == blob, "9.1 %s (%s): re-save differs from the restored SSB6 (%zu vs %zu bytes)", what,
		              with ? "handle" : "no handle", again.size(), blob.size());
		if (!expect.empty()) {
			const std::vector<int32_t> next = DecodeN(fx.model, r, static_cast<int64_t>(expect.size()));
			PKV_CHECK_MSG(next == expect, "9.1 %s (%s): tokens after restore differ from the original's", what,
			              with ? "handle" : "no handle");
		}
		sslm_seq_release(r);
	}
}

// The canonical SSB6 row at a mid-token save's context_length, layers >= layer_index, is zero bytes
// (§3.7), read from the blob by the layout formula. The pool was filled with 0x5A, so a gather from
// memory shows (the SSB6 side of 6.4a, run here on the round trip's own blob).
void ExpectMidTokenRowZero(const Fixture& fx, const std::vector<uint8_t>& blob, const char* what) {
	const BlobView v = ParseBlobWithHidden(blob, HiddenSize(fx));
	PKV_CHECK_MSG(v.ok && v.layer_index > 0 && v.kv_positions == static_cast<uint64_t>(v.context_length + 1),
	              "9.1 %s: not a mid-token SSB6 (layer_index %u, L' %llu, L %lld)", what, v.layer_index,
	              static_cast<unsigned long long>(v.kv_positions), static_cast<long long>(v.context_length));
	if (!v.ok || v.layer_index == 0) return;
	bool zero = true;
	for (uint32_t l = v.layer_index; l < fx.geo.layers; ++l)
		for (uint32_t half = 0; half < 2; ++half)
			for (uint32_t h = 0; h < fx.geo.kv_heads; ++h) {
				const uint8_t* r = Ssb6Row(blob, fx, l, half, h, v.context_length);
				for (uint32_t d = 0; d < fx.geo.head_dim; ++d) zero = zero && r[d] == 0;
			}
	// kills: the save gathers the mid-token row's unwritten layers from memory
	PKV_CHECK_MSG(zero, "9.1 %s: mid-token row's layers >= layer_index are not zero in the SSB6", what);
}

// (a) R0's prefix_lengths scenario replayed with budget holders: tokens and rows equal v1.11.0's
// (kTokensAndRows: the blob is SSB6 against the reference's SSB5), and every saved stage round-trips.
void Cell91PrefixLengths(const Fixture& fx) {
	Rig rig(fx, 600);
	std::vector<Record> got;
	for (int32_t len : {0, 1, 15, 16, 17, 1000, 1008}) {
		const std::string tag = "p" + std::to_string(len);
		sslm_prefix px = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, rig.Pool(), std::max(len, 1), &px), SSLM_OK);
		if (!px) return;
		rig.Prefix(px);
		if (len) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, len, kVocab), 64), SSLM_OK);
		PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), 64, &s), SSLM_OK);
		if (!s) return;
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		std::vector<int32_t> pending;
		auto mark = [&](const std::string& stage) {
			Record r;
			r.stage = stage;
			r.tokens.swap(pending);
			const std::vector<uint8_t> blob = Save(s);
			const BlobView v = ParseBlobWithHidden(blob, HiddenSize(fx));
			PKV_CHECK_MSG(v.ok && IsMagic(blob, "SSB6"), "9.1 %s: a budget holder's save is not SSB6", stage.c_str());
			r.context_length = v.context_length;
			r.rows_sha = RowsSha(blob, fx);
			got.push_back(r);
			RoundTrip(fx, rig.Pool(), px, blob, {}, stage.c_str());
		};
		mark(tag + "_adopted");
		if (len) {
			sslm_status st = SSLM_OK;
			pending = DecodeN(fx.model, s, 8, &st);
			PKV_CHECK_EQ(st, SSLM_OK);
			mark(tag + "_decode8");
		}
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(52, 20, kVocab), 64), SSLM_OK);
		const std::vector<int32_t> d4 = DecodeN(fx.model, s, 4);
		pending.insert(pending.end(), d4.begin(), d4.end());
		mark(tag + "_prefill20+decode4");
		sslm_seq_release(s);
	}
	ExpectMatchesReference("v1.11.0", fx, "prefix_lengths", got, Compare::kTokensAndRows);
}

// (b) Nested (the persona chain) and (c) adopted at a mid-page origin: stages resting ready, resting
// with a pending token, and mid-token. Each stage round-trips, and its tokens and rows equal a legacy
// twin's (a whole_reserve holder copy-adopting the same prefix: 1.9.0's SSB5, rows compared).
void Cell91Stages(const Fixture& fx, bool nested) {
	const char* what = nested ? "nested" : "adopted P=1000";
	Rig rig(fx, 700, 0x5A);
	sslm_prefix px = nullptr;
	if (nested) {
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
		px = rig.chain.persona;
	} else {
		px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(1000)));
	}
	if (!px) return;
	sslm_seq s = nullptr, legacy = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), kCohortBudget, &s), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_create(fx.model, rig.Pool(), &legacy), SSLM_OK);
	if (!rig.Seq(s) || !rig.Seq(legacy)) return;
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(legacy, px), SSLM_OK);
	auto stage = [&](const char* name, bool mid_token) {
		const std::vector<uint8_t> b6 = Save(s), b5 = Save(legacy);
		PKV_CHECK_MSG(IsMagic(b6, "SSB6") && IsMagic(b5, "SSB5"), "9.1 %s/%s: magics", what, name);
		const BlobView v6 = ParseBlobWithHidden(b6, HiddenSize(fx)), v5 = ParseBlobWithHidden(b5, HiddenSize(fx));
		PKV_CHECK_MSG(v6.context_length == v5.context_length && v6.layer_index == v5.layer_index,
		              "9.1 %s/%s: the budget holder and its legacy twin rest at different states", what, name);
		// kills: a share adopt or a page gather that changes a row (rows [0, L) of the two formats)
		PKV_CHECK_MSG(RowsSha(b6, fx) == RowsSha(b5, fx), "9.1 %s/%s: rows differ from the legacy twin's", what, name);
		if (mid_token) ExpectMidTokenRowZero(fx, b6, name);
		RoundTrip(fx, rig.Pool(), px, b6, {}, name);
	};
	stage("adopted (ready)", false);
	const std::vector<int32_t> t6 = DecodeN(fx.model, s, 8), t5 = DecodeN(fx.model, legacy, 8);
	PKV_CHECK_MSG(t6.size() == 8 && t6 == t5, "9.1 %s: decode 8 differs from the legacy twin", what);
	stage("decode8 (pending)", false);
	int32_t a = 0, b = 0;
	PKV_CHECK_EQ(LayerStep(fx.model, s, &a), SSLM_OK);
	PKV_CHECK_EQ(LayerStep(fx.model, legacy, &b), SSLM_OK);
	PKV_CHECK_MSG(a == -1 && b == -1, "9.1 %s: one layer of two must leave the token pending", what);
	stage("mid-token", true);
	const std::vector<int32_t> f6 = DecodeN(fx.model, s, 1), f5 = DecodeN(fx.model, legacy, 1);
	PKV_CHECK_MSG(f6 == f5 && f6.size() == 1, "9.1 %s: finishing the mid-token token differs", what);
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(4, 37, kVocab), 7), SSLM_OK);
	PKV_CHECK_EQ(PrefillAll(fx.model, legacy, Stream(4, 37, kVocab), 7), SSLM_OK);
	stage("prefill37 (ready)", false);
	PKV_CHECK_MSG(DecodeN(fx.model, s, 4) == DecodeN(fx.model, legacy, 4), "9.1 %s: decode 4 differs", what);
	stage("prefill37+decode4 (pending)", false);
}

void Cell91Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Cell91PrefixLengths(fx);
	Cell91Stages(fx, true);
	Cell91Stages(fx, false);
}

// ================================================================================================
// 9.3/C5: the SSB6 side. A budget holder's blob is SSB6, whose magic is none of the four v1.11.0's
// reader accepts (SSB5, SSB4, SSB3, SSB2: `kSeqBlobMagicV5`..`V2`, R0's re-anchor note).
//
// Box leg (two binaries): the blob below, written to $SUPERSLM_PAGED_KV_EXPORT_DIR/c5_9.3_ssb6.bin when
// that variable is set, is restored by the pinned v1.11.0 binary on the same fixture and must return
// SSLM_INVALID_ARGUMENT on magic. The cloud has one binary, so it checks the magic and exports the blob.
// ================================================================================================
void Cell93Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Rig rig(fx, 200);
	sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(1000)));
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), kCohortBudget, &s), SSLM_OK);
	if (!px || !rig.Seq(s)) return;
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
	DecodeN(fx.model, s, 100);
	const std::vector<uint8_t> blob = Save(s);
	// kills: a budget holder writing 1.9.0's SSB5 (v1.11.0 would then read it and lose the budget)
	PKV_CHECK_MSG(IsMagic(blob, "SSB6"), "9.3: a budget holder's save is not SSB6");
	for (const char* old : {"SSB5", "SSB4", "SSB3", "SSB2"})
		PKV_CHECK_MSG(!IsMagic(blob, old), "9.3: the SSB6 blob carries v1.11.0's magic %s", old);
	const BlobView v = ParseBlobWithHidden(blob, HiddenSize(fx));
	PKV_CHECK_MSG(v.ok && v.kv_mode == 1 && v.budget == kCohortBudget && v.origin == 1000, "9.3: SSB6 header fields");
	if (const char* dir = std::getenv("SUPERSLM_PAGED_KV_EXPORT_DIR")) {
		const std::string path = std::string(dir) + "/c5_9.3_ssb6.bin";
		FILE* f = std::fopen(path.c_str(), "wb");
		PKV_CHECK_MSG(f && std::fwrite(blob.data(), 1, blob.size(), f) == blob.size(), "9.3: export to %s", path.c_str());
		if (f) std::fclose(f);
		std::printf("9.3: box leg input written to %s (v1.11.0 must refuse it: SSLM_INVALID_ARGUMENT)\n", path.c_str());
	}
}

// ================================================================================================
// 9.4: an adopted budget sequence saved mid-budget and restored runs to its limit with no refusal,
// and refuses one past (round 3 S1). Resting pending (decoded) and resting ready (prefilled), each
// restored with the handle and without; the decode form and the prefill form of "one past".
// ================================================================================================
void Cell94() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int32_t P = 1000, b = kCohortBudget;
	const int64_t limit = Limit(fx, P, b);
	for (int ready = 0; ready < 2; ++ready) {
		Rig rig(fx, 400);
		sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(P)));
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), b, &s), SSLM_OK);
		if (!px || !rig.Seq(s)) return;
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		if (ready) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(92, 256, kVocab), kChunk), SSLM_OK);
		else DecodeN(fx.model, s, 257);  // the ready token, then 256 rows: pending at 1,256
		const std::vector<uint8_t> blob = Save(s);
		const int64_t c = ReadKept(fx, blob).context_length;
		PKV_CHECK_EQ(c, P + 256);
		const int64_t expect = (ready ? 1 : 0) + (limit - c);  // §3.4
		for (int with = 0; with < 3; ++with) {
			// with 0: no handle; 1: the handle; 2: the original itself (unsaved), as the control
			sslm_seq r = s;
			if (with < 2) {
				r = nullptr;
				const sslm_status st = with ? sslm_seq_restore_shared(fx.model, rig.Pool(), blob.data(), blob.size(), px, &r, nullptr)
				                            : sslm_seq_restore(fx.model, rig.Pool(), blob.data(), blob.size(), &r);
				PKV_CHECK_EQ(st, SSLM_OK);
				if (!r) continue;
			}
			sslm_status refusal = SSLM_OK;
			const std::vector<int32_t> toks = RunToRefusal(fx.model, r, &refusal);
			// kills: rev 2's budget := limit (no refusal at 1,512), origin := context_length (runs
			// 256 too far), budget := the remaining budget (refuses early)
			PKV_CHECK_MSG(static_cast<int64_t>(toks.size()) == expect,
			              "9.4 (%s, %s): %zu tokens before the refusal, expected %lld", ready ? "ready" : "pending",
			              with == 0 ? "no handle" : with == 1 ? "handle" : "original", toks.size(), static_cast<long long>(expect));
			PKV_CHECK_EQ(refusal, PKV_KV_BUDGET_EXCEEDED);
			if (with < 2) sslm_seq_release(r);
		}
		// The prefill form, on the ready state: a restored twin takes exactly limit - c tokens and
		// refuses the next one, consuming nothing.
		if (!ready) continue;
		sslm_seq r = nullptr;
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, rig.Pool(), blob.data(), blob.size(), px, &r, nullptr), SSLM_OK);
		if (!rig.Seq(r)) continue;
		int64_t consumed = 0;
		PKV_CHECK_EQ(PrefillAll(fx.model, r, Stream(93, static_cast<int32_t>(limit - c), kVocab), kChunk, &consumed), SSLM_OK);
		PKV_CHECK_EQ(consumed, limit - c);
		const int32_t one = 7;
		int32_t got = -1;
		PKV_CHECK_EQ(sslm_prefill(fx.model, r, &one, 1, 1, SSLM_SPAN_PROMPT, nullptr, &got), PKV_KV_BUDGET_EXCEEDED);
		PKV_CHECK_EQ(got, 0);
	}
}

// ================================================================================================
// 9.5/C5: the separating worst case, on a budget holder with budget = cap saving SSB6 (RB16-CELLS).
// Damped-greedy generation from an empty origin to cap - 1, a mid-token save (L' = cap, history
// cap - 1), then one more whole token: the holder rests at context_length = cap with history cap,
// where the SSB6 is 172 + residual + 4 cap + 8 + cap * bytes_per_token, exactly the +52 bound. Both
// caps (pkv_def 4,096 and pkv_32k 32,768; the 32k leg is ~2 minutes of decode).
// ================================================================================================
void Cell95Budget() {
	for (const char* stem : {"pkv_def", "pkv_32k"}) {
		const Fixture& fx = GetFixture(stem);
		if (!fx.ok) continue;
		const int64_t cap = fx.geo.context_cap;
		const size_t hidden = HiddenSize(fx), bpt = fx.BytesPerToken();
		Rig rig(fx, static_cast<uint32_t>(fx.R(cap)));
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), static_cast<int32_t>(cap), &s), SSLM_OK);
		if (!rig.Seq(s)) continue;
		sslm_decode_params whole{}, one{};
		PKV_CHECK_EQ(sslm_decode_params_init(fx.model, SSLM_DECODE_MODE_DAMPED_GREEDY, static_cast<int32_t>(fx.geo.layers), &whole), SSLM_OK);
		PKV_CHECK_EQ(sslm_decode_params_init(fx.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &one), SSLM_OK);
		const std::vector<int32_t> first = Stream(95, 1, kVocab);
		PKV_CHECK_EQ(PrefillAll(fx.model, s, first, 1), SSLM_OK);
		// After the one-token prompt the holder is ready at 1; each further token writes one row.
		sslm_status st = SSLM_OK;
		for (int64_t emitted = 0; emitted < cap - 1 && st == SSLM_OK; ++emitted) {
			sslm_seq b[1] = {s};
			int32_t tok = -1;
			st = sslm_decode_step_v2(fx.model, b, 1, &whole, nullptr, &tok);
		}
		PKV_CHECK_EQ(st, SSLM_OK);
		{
			sslm_seq b[1] = {s};
			int32_t tok = 0;
			PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &one, nullptr, &tok), SSLM_OK);
			PKV_CHECK_EQ(tok, -1);
		}
		const size_t bound = sslm_seq_state_size(fx.model);
		const std::vector<uint8_t> mid = Save(s);
		const BlobView vm = ParseBlobWithHidden(mid, hidden);
		PKV_CHECK_MSG(vm.ok && vm.context_length == cap - 1 && vm.layer_index == 1 && vm.history_count == static_cast<uint64_t>(cap - 1) &&
		                  vm.kv_positions == static_cast<uint64_t>(cap),
		              "9.5 %s: the mid-token state is not (L cap-1, layer 1, history cap-1, L' cap)", stem);
		PKV_CHECK_EQ(mid.size(), kSsb6Header + hidden + 4 * (cap - 1) + 8 + cap * bpt);
		PKV_CHECK_MSG(mid.size() <= bound, "9.5 %s mid-token: blob %zu > sslm_seq_state_size %zu", stem, mid.size(), bound);
		{
			sslm_seq b[1] = {s};
			int32_t tok = -1;
			PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &one, nullptr, &tok), SSLM_OK);
			PKV_CHECK_MSG(tok >= 0, "9.5 %s: the token writing row cap - 1 did not emit", stem);
		}
		const std::vector<uint8_t> rest = Save(s);
		const BlobView vr = ParseBlobWithHidden(rest, hidden);
		PKV_CHECK_MSG(vr.ok && vr.context_length == cap && vr.layer_index == 0 && vr.history_count == static_cast<uint64_t>(cap),
		              "9.5 %s: the resting state is not (L cap, history cap)", stem);
		PKV_CHECK_EQ(rest.size(), kSsb6Header + hidden + 4 * cap + 8 + cap * bpt);
		// kills: rev 2's +48 to +51 (state_size 4 to 1 bytes short of this blob)
		PKV_CHECK_MSG(rest.size() <= bound, "9.5 %s at rest: blob %zu > sslm_seq_state_size %zu", stem, rest.size(), bound);
		// The plan's margin at this state is 0: the worst case is reached, so the bound is exact here.
		PKV_CHECK_EQ(bound - rest.size(), 0);
		sslm_seq b[1] = {s};
		int32_t tok = 0;
		PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &whole, nullptr, &tok), SSLM_CONTEXT_CAP_EXCEEDED);  // limit == cap (§3.6)
	}
}

// ================================================================================================
// 9.9: the strike's falsifier.
// ================================================================================================
void Cell99Fixed(const Fixture& fx) {
	const Fixed& f = FixedCase(fx);
	if (!f.ok) return;
	const int64_t chain = PersonaChainPages(fx, 200, true);       // 75
	const int64_t R = fx.R(kCohortBudget);                         // 33
	const int64_t E = MaterializedPages(fx, kPersonaLen, kCohortBudget);  // 75
	PKV_CHECK_EQ(chain, 75);
	PKV_CHECK_EQ(chain + R, 108);
	PKV_CHECK_EQ(chain + R + E, 183);
	for (int with = 1; with >= 0; --with) {
		const int64_t N = chain + R + (with ? 0 : E);
		for (int64_t pages : {N, N - 1}) {
			Rig rig(fx, static_cast<uint32_t>(pages));
			rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
			if (!rig.chain.ok) return;
			sslm_seq r = nullptr;
			const sslm_status st = with ? sslm_seq_restore_shared(fx.model, rig.Pool(), f.blob.data(), f.blob.size(), rig.chain.persona, &r, nullptr)
			                            : sslm_seq_restore(fx.model, rig.Pool(), f.blob.data(), f.blob.size(), &r);
			rig.Seq(r);
			if (pages == N - 1) {
				// kills: a restore that draws fewer pages than R (+ E): admitted one page short
				PKV_CHECK_MSG(st == SSLM_KV_POOL_EXHAUSTED, "9.9 fixed (%s): a pool of %lld admitted the restore (status %s)",
				              with ? "handle" : "no handle", static_cast<long long>(pages), Name(st));
				continue;
			}
			// kills: a restore that draws more (with the handle: one that does not share)
			PKV_CHECK_MSG(st == SSLM_OK, "9.9 fixed (%s): a pool of %lld refused the restore (status %s)",
			              with ? "handle" : "no handle", static_cast<long long>(pages), Name(st));
			if (!r) continue;
			const Turn t = NextTurn(fx.model, r);
			PKV_CHECK_EQ(t.tokens.size(), 213);
			PKV_CHECK_MSG(t == f.cont, "9.9 fixed (%s): the continuation differs from the unsaved original's",
			              with ? "handle" : "no handle");
			PKV_CHECK_EQ(sslm_seq_reset(r), SSLM_OK);
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(r, rig.chain.persona), SSLM_OK);
			sslm_status refusal = SSLM_OK;
			const std::vector<int32_t> again = RunToRefusal(fx.model, r, &refusal);
			// kills: rev 2's rule budget := limit (1,713 tokens here, not 513)
			PKV_CHECK_EQ(again.size(), 513);
			PKV_CHECK_EQ(refusal, PKV_KV_BUDGET_EXCEEDED);
			PKV_CHECK_MSG(again == f.readopt, "9.9 fixed (%s): tokens after re-adopt differ from the original's",
			              with ? "handle" : "no handle");
		}
	}
	// After the no-handle twin's reset and re-adopt, exactly the 75 materialized pages came back
	// ("the 33 private pages", measured without sslm_seq_kv_stats).
	// kills: materialized pages sent to the reserve (0 returned), or freed twice (76)
	ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> {
		auto rig = std::make_unique<Rig>(fx, static_cast<uint32_t>(chain + R + E));
		rig->chain = BuildPersonaChain(fx, rig->Pool(), 0, true);
		sslm_seq r = nullptr;
		if (sslm_seq_restore(fx.model, rig->Pool(), f.blob.data(), f.blob.size(), &r) == SSLM_OK) {
			rig->Seq(r);
			NextTurn(fx.model, r);
			sslm_seq_reset(r);
			sslm_seq_adopt_prefix(r, rig->chain.persona);
		}
		return rig;
	}, E);
}

void Cell99Population(const Fixture& fx) {
	int cases = 0;
	for (int32_t P : kPopP)
		for (int32_t b : kPopB)
			for (Sp sp : kPopSp) {
				const Original o = MakeOriginal(fx, P, b, sp);
				if (!o.ok) continue;
				const std::string name = CaseName(P, b, sp);
				// The unsaved original's own reservation after reset and re-adopt: R(budget), graded by the
				// probe in a pool of prefix pages + R + slack (the original's side of "equal").
				ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> {
					auto rig = std::make_unique<Rig>(fx, static_cast<uint32_t>(PrefixPages(fx, P) + fx.R(b) + kSlack));
					sslm_prefix px = rig->Prefix(BudgetPrefix(fx, rig->Pool(), PrefixTokens(P)));
					sslm_seq s = nullptr;
					if (px && sslm_seq_create_budgeted(fx.model, rig->Pool(), b, &s) == SSLM_OK) {
						rig->Seq(s);
						sslm_seq_adopt_prefix(s, px);
						DriveTo(fx, s, P, b, sp);
						sslm_seq_reset(s);
						sslm_seq_adopt_prefix(s, px);
					}
					return rig;
				}, kSlack);
				for (How how : {How::kHandle, How::kNone}) {
					++cases;
					std::unique_ptr<Twin> t = MakeTwin(fx, P, b, o.blob, how, false, true);
					PKV_CHECK_MSG(t->restore == SSLM_OK, "9.9 %s (%s): restore status %s", name.c_str(), HowName(how), Name(t->restore));
					if (!t->seq) continue;
					const Turn turn = NextTurn(fx.model, t->seq);
					// kills: rev 2's budget := limit (the next turn runs past the original's)
					PKV_CHECK_MSG(turn == o.turn, "9.9 %s (%s): next turn %zu tokens (status %s), original %zu (status %s)",
					              name.c_str(), HowName(how), turn.tokens.size(), Name(turn.refusal), o.turn.tokens.size(),
					              Name(o.turn.refusal));
					// kills: the ceil-span mutant, which shares the mid-page tail (63 pages at origin 1,000)
					// and writes the twin's rows into the frozen prefix's page
					const PageSnap after = SnapPrefix(fx, *t->Pool(), t->prefix);
					PKV_CHECK_MSG(after.index == t->before.index && after.bytes == t->before.bytes,
					              "9.9 %s (%s): the prefix's pages changed under the twin's writes", name.c_str(), HowName(how));
					t.reset();
					// kills: materialized pages to the reserve (E not returned), and a restore that keeps a
					// reservation other than R(budget)
					ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> { return MakeTwin(fx, P, b, o.blob, how, true); },
					                 FreeAfterReadopt(fx, P, b, how));
				}
			}
	PKV_CHECK_EQ(cases, 8 * 5 * 3 * 2);
}

void Cell99() {
	SentinelOn sentinel;
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Cell99Fixed(fx);
	Cell99Population(fx);
}

// ================================================================================================
// 9.10: private restores that separate. Each case restores with SSLM_OK and the saved budget kept,
// privately; privacy is graded by admission in a pool sized to exactly handle pages + R(budget) + E.
// ================================================================================================

// After a private budget restore in a pool of exactly (handle pages + R + E), nothing is free: a
// 2-page create is refused. A sharing restore would have left E (= 75) pages free.
void ExpectPrivate(const Fixture& fx, sslm_kv_pool* pool, const char* what) {
	// kills: a restore that shares under a handle failing (a)-(f)
	PKV_CHECK_MSG(!TwoPageCreateAdmitted(fx, pool), "9.10 %s: a 2-page create was admitted after the restore (it shared)", what);
}

// Restores the fixed blob (or `blob`) with `handle` into `pool`, checks status, budget kept, privacy,
// and that the next tokens equal the original's. Returns the restored sequence (caller releases).
sslm_seq RestoreExpectPrivate(const Fixture& fx, sslm_kv_pool* pool, sslm_prefix handle, const std::vector<uint8_t>& blob,
                              const char* what, bool check_private = true) {
	sslm_seq r = nullptr;
	uint32_t shared_diag = 0;
	const sslm_status st = sslm_seq_restore_shared(fx.model, pool, blob.data(), blob.size(), handle, &r, &shared_diag);
	PKV_CHECK_MSG(st == SSLM_OK, "9.10 %s: restore status %s", what, Name(st));
	if (!r) return nullptr;
	const Kept k = ReadKept(fx, Save(r));
	PKV_CHECK_MSG(k.ok && k.budget == kCohortBudget && k.origin == kPersonaLen && k.mode == 1, "9.10 %s: saved budget not kept", what);
	if (check_private) ExpectPrivate(fx, pool, what);
	std::printf("9.10 %s: *out_shared_pages = %u (diagnostic)\n", what, shared_diag);
	return r;
}

void Cell910() {
	SentinelOn sentinel;
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const Fixed& f = FixedCase(fx);
	if (!f.ok) return;
	const int64_t R = fx.R(kCohortBudget), E = MaterializedPages(fx, kPersonaLen, kCohortBudget);
	auto tokens_equal = [&](sslm_seq r, const char* what) {
		const std::vector<int32_t> next = DecodeN(fx.model, r, 8);
		// kills: a share of a one-byte-different prefix, seen in the next tokens
		PKV_CHECK_MSG(next == f.next8, "9.10 %s: tokens after restore differ from the original's", what);
	};

	// (e) a longer handle: the persona prompt plus one token, frozen at 1,201. Its span bytes equal the
	// blob's, so only (e) rejects it.
	{
		std::vector<int32_t> longer = PersonaTokens(0);
		longer.push_back(5);
		Rig rig(fx, static_cast<uint32_t>(PersonaChainPages(fx, 201, true) + R + E));
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true, &longer);
		if (rig.chain.ok) {
			PKV_CHECK_MSG(PrefixSpanEqualsBlob(fx, *rig.Pool(), rig.chain.persona, f.blob, kPersonaLen),
			              "9.10 (e) longer: construction: the span bytes must be equal, so (e) alone rejects");
			sslm_seq r = rig.Seq(RestoreExpectPrivate(fx, rig.Pool(), rig.chain.persona, f.blob, "(e) longer handle"));
			if (r) tokens_equal(r, "(e) longer handle");
		}
	}
	// (e) a shorter handle, the world at 1,000, under the table sentinel: a build that evaluates (f)
	// before (e) reads the world's table to floor(1,200/B) = 75 entries, above its 63 mapped, and traps.
	{
		Rig rig(fx, static_cast<uint32_t>(PrefixPages(fx, 1000) + R + E));
		sslm_prefix world = rig.Prefix(BudgetPrefix(fx, rig.Pool(), WorldTokens()));
		if (world) {
			// kills (by trapping): condition (f) evaluated before (e)
			sslm_seq r = rig.Seq(RestoreExpectPrivate(fx, rig.Pool(), world, f.blob, "(e) shorter handle"));
			if (r) tokens_equal(r, "(e) shorter handle");
		}
	}
	// (f) the blob edited in one byte of layer L-1's V half inside the prefix span. Private, and a
	// re-save equals the edited blob byte for byte.
	{
		std::vector<uint8_t> edited = f.blob;
		const int64_t pos = 500;  // in the span [0, 1,200), in a full page
		const uint8_t* row = Ssb6Row(edited, fx, fx.geo.layers - 1, 1, 0, pos);
		edited[static_cast<size_t>(row - edited.data()) + 3] ^= 0x01;
		Rig rig(fx, static_cast<uint32_t>(PersonaChainPages(fx, 200, true) + R + E));
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
		if (rig.chain.ok) {
			int64_t at = -2;
			PKV_CHECK_MSG(!PrefixSpanEqualsBlob(fx, *rig.Pool(), rig.chain.persona, edited, kPersonaLen, &at) && at == pos,
			              "9.10 (f) edited: construction: the span must differ at position %lld only", static_cast<long long>(pos));
			sslm_seq r = rig.Seq(RestoreExpectPrivate(fx, rig.Pool(), rig.chain.persona, edited, "(f) edited blob"));
			if (r) {
				// kills: the K-only and the layer-0-only compare (they share, and the re-save then carries
				// the prefix's byte, not the blob's)
				PKV_CHECK_MSG(Save(r) == edited, "9.10 (f) edited: the re-save differs from the edited blob");
			}
		}
	}
	// (f) a real producer: a prefix whose prompt differs in its last full-page token (position 1,199).
	{
		std::vector<int32_t> other = PersonaTokens(0);
		other.back() = (other.back() + 1) % kVocab;
		Rig rig(fx, static_cast<uint32_t>(PersonaChainPages(fx, 200, true) + R + E));
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true, &other);
		if (rig.chain.ok) {
			int64_t at = -2;
			PKV_CHECK_MSG(!PrefixSpanEqualsBlob(fx, *rig.Pool(), rig.chain.persona, f.blob, kPersonaLen, &at) &&
			                  at == kPersonaLen - 1,
			              "9.10 (f) producer: construction: the span must first differ at 1,199 (got %lld)", static_cast<long long>(at));
			// kills: a compare that skips the last full page
			sslm_seq r = rig.Seq(RestoreExpectPrivate(fx, rig.Pool(), rig.chain.persona, f.blob, "(f) real producer"));
			if (r) tokens_equal(r, "(f) real producer");
		}
	}
	// (b) an unfrozen handle with equal span bytes: the persona left unfrozen holds R(200) pages.
	{
		Rig rig(fx, static_cast<uint32_t>(SharedPages(fx, 1000) + fx.R(200) + R + E));
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true, nullptr, false);
		if (rig.chain.ok) {
			PKV_CHECK_MSG(PrefixSpanEqualsBlob(fx, *rig.Pool(), rig.chain.persona, f.blob, kPersonaLen),
			              "9.10 (b): construction: the unfrozen handle's span bytes must be equal");
			// kills: (b) not checked
			sslm_seq r = rig.Seq(RestoreExpectPrivate(fx, rig.Pool(), rig.chain.persona, f.blob, "(b) unfrozen handle"));
			if (r) tokens_equal(r, "(b) unfrozen handle");
		}
	}
	// (d) a handle from another pool (same model, equal bytes): the destination pool holds R + E only.
	{
		Rig other(fx, 200);
		other.chain = BuildPersonaChain(fx, other.Pool(), 0, true);
		Rig rig(fx, static_cast<uint32_t>(R + E));
		if (other.chain.ok) {
			// kills: (d) not checked (a cross-pool share would leave E pages free here)
			sslm_seq r = rig.Seq(RestoreExpectPrivate(fx, rig.Pool(), other.chain.persona, f.blob, "(d) handle of another pool"));
			if (r) tokens_equal(r, "(d) handle of another pool");
		}
	}
	// (a) a whole_reserve blob with a handle: E = 0, so privacy is graded after the handle's release:
	// a create of the handle's page count is admitted (a sharing restore would still hold them).
	{
		std::vector<uint8_t> ssb5, ssb6wr;
		std::vector<int32_t> next8;
		{
			Rig src(fx, 400);
			src.chain = BuildPersonaChain(fx, src.Pool(), 0, true);
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_seq_create(fx.model, src.Pool(), &s), SSLM_OK);
			if (src.chain.ok && src.Seq(s)) {
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, src.chain.persona), SSLM_OK);
				PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(91, 300, kVocab), kChunk), SSLM_OK);
				ssb5 = Save(s);
				ssb6wr = Ssb6WholeReserveFromSsb5(ssb5, fx, kPersonaLen);
				next8 = DecodeN(fx.model, s, 8);
			}
		}
		PKV_CHECK_MSG(next8 == f.next8, "9.10 (a): the legacy twin's tokens differ from the budget original's");
		const int64_t chain = PersonaChainPages(fx, 200, true);
		for (const std::vector<uint8_t>* blob : {&ssb6wr, &ssb5}) {
			const char* what = blob == &ssb5 ? "(a) 1.9.0 SSB5 with a handle" : "(a) whole_reserve SSB6 with a handle";
			Rig rig(fx, static_cast<uint32_t>(chain + fx.R(fx.geo.context_cap)));
			rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
			if (!rig.chain.ok || blob->empty()) continue;
			sslm_seq r = nullptr;
			PKV_CHECK_MSG(sslm_seq_restore_shared(fx.model, rig.Pool(), blob->data(), blob->size(), rig.chain.persona, &r, nullptr) == SSLM_OK,
			              "9.10 %s: restore", what);
			if (!rig.Seq(r)) continue;
			PKV_CHECK_MSG(IsMagic(Save(r), "SSB5"), "9.10 %s: the restored holder is not whole_reserve", what);
			PKV_CHECK_MSG(DecodeN(fx.model, r, 8) == next8, "9.10 %s: tokens after restore differ", what);
			rig.chain.Release();
			// kills: the whole_reserve share mutant (the released handle's pages stay held)
			PKV_CHECK_MSG(AdmitsPages(fx, rig.Pool(), chain), "9.10 %s: a %lld-page create was refused after the handle's release",
			              what, static_cast<long long>(chain));
		}
	}
	// (c) the handle's model differs from the destination's: implied by (d), because a pool is bound to
	// one model, so no handle can differ in model while sharing the pool. Stated, not tested (§7).
}

// ================================================================================================
// 9.11: materialized pages, at 9.11's own construction: the persona chain, origin 1,200, budget 512,
// a pool of 183. Drawn: R + E; returned: E at the first reset or adopt, R + E at the first release.
// ================================================================================================
void Cell911() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const Fixed& f = FixedCase(fx);
	if (!f.ok) return;
	const int64_t chain = PersonaChainPages(fx, 200, true), R = fx.R(kCohortBudget);
	const int64_t E = MaterializedPages(fx, kPersonaLen, kCohortBudget);
	const int64_t pool_pages = chain + R + E;
	PKV_CHECK_EQ(pool_pages, 183);
	// Drawn: admitted at exactly R + E free, refused one below.
	for (int64_t pages : {pool_pages, pool_pages - 1}) {
		Rig rig(fx, static_cast<uint32_t>(pages));
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
		sslm_seq r = nullptr;
		const sslm_status st = sslm_seq_restore(fx.model, rig.Pool(), f.blob.data(), f.blob.size(), &r);
		rig.Seq(r);
		// kills: a private restore drawing R only (admitted at 182), or more than R + E (refused at 183)
		PKV_CHECK_EQ(st, pages == pool_pages ? SSLM_OK : SSLM_KV_POOL_EXHAUSTED);
	}
	enum class First { kReset, kAdopt, kRelease };
	for (First first : {First::kReset, First::kAdopt, First::kRelease}) {
		const char* what = first == First::kReset ? "reset" : first == First::kAdopt ? "adopt" : "release";
		auto build = [&]() -> std::unique_ptr<ProbeState> {
			auto rig = std::make_unique<Rig>(fx, static_cast<uint32_t>(pool_pages));
			rig->chain = BuildPersonaChain(fx, rig->Pool(), 0, true);
			sslm_seq r = nullptr;
			if (sslm_seq_restore(fx.model, rig->Pool(), f.blob.data(), f.blob.size(), &r) == SSLM_OK) {
				if (first == First::kReset) sslm_seq_reset(r);
				if (first == First::kAdopt) sslm_seq_adopt_prefix(r, rig->chain.persona);
				if (first == First::kRelease) sslm_seq_release(r);
				else rig->Seq(r);
			}
			return rig;
		};
		// kills: materialized pages sent to the reserve (reset and adopt: 75 against 0); at release that
		// mutant is equivalent (108 against 108), so the release path names its own: "release returns
		// the materialized pages to nothing" (33 against 108)
		PKV_CHECK_MSG(ProbeExactlyFree(fx, build, first == First::kRelease ? R + E : E), "9.11 first %s", what);
	}
	// The freed materialized pages read 0xCD through the peek after the first reset (they were written
	// by the restore's scatter, so they are dirty, §3.3).
	{
		Rig rig(fx, static_cast<uint32_t>(pool_pages), 0x5A);
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
		sslm_seq r = nullptr;
		PKV_CHECK_EQ(sslm_seq_restore(fx.model, rig.Pool(), f.blob.data(), f.blob.size(), &r), SSLM_OK);
		if (rig.Seq(r)) {
			const std::vector<uint32_t> mat = SeqPages(r, static_cast<uint32_t>(SharedPages(fx, kPersonaLen)));
			PKV_CHECK_EQ(mat.size(), SharedPages(fx, kPersonaLen));
			PKV_CHECK_EQ(sslm_seq_reset(r), SSLM_OK);
			for (uint32_t page : mat) {
				const std::vector<uint8_t> b = PeekPage(fx, *rig.Pool(), page);
				bool cd = true;
				for (uint8_t x : b) cd = cd && x == 0xCD;
				// kills: a freed materialized page returned without its poison fill
				PKV_CHECK_MSG(cd, "9.11: freed materialized page %u does not read 0xCD", page);
			}
		}
	}
	// sslm_seq_restore and sslm_seq_restore_shared(null) give byte-identical results; a null
	// out_shared_pages with a matching handle is accepted.
	{
		Rig rig(fx, 600);
		rig.chain = BuildPersonaChain(fx, rig.Pool(), 0, true);
		sslm_seq a = nullptr, b = nullptr, c = nullptr;
		PKV_CHECK_EQ(sslm_seq_restore(fx.model, rig.Pool(), f.blob.data(), f.blob.size(), &a), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, rig.Pool(), f.blob.data(), f.blob.size(), nullptr, &b, nullptr), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, rig.Pool(), f.blob.data(), f.blob.size(), rig.chain.persona, &c, nullptr), SSLM_OK);
		if (rig.Seq(a) && rig.Seq(b) && rig.Seq(c)) {
			PKV_CHECK_MSG(Save(a) == Save(b) && Save(a) == f.blob, "9.11: restore and restore_shared(null) blobs differ");
			const Turn ta = NextTurn(fx.model, a), tb = NextTurn(fx.model, b), tc = NextTurn(fx.model, c);
			PKV_CHECK_MSG(ta == tb && ta == f.cont && tc == f.cont, "9.11: restore and restore_shared(null) tokens differ");
		}
	}
}

// ================================================================================================
// 9.14/C5: §3.9's other "keeps" rows over 9.9's population, A16-A19 refused over it, A20's crossings.
// ================================================================================================

// A prefix with real schema progress (the census's P_PROGRESS): the fixture's schema bound, a 4-token
// prompt, then one SCHEMA_CONTENT token the schema admits from its start state.
sslm_prefix SchemaProgressPrefix(const Fixture& fx, sslm_kv_pool* pool, sslm_schema* out_schema) {
	sslm_schema sc = nullptr;
	PKV_CHECK_EQ(sslm_schema_lookup(fx.model, "g5_minimal_one_field", &sc), SSLM_OK);
	if (out_schema) *out_schema = sc;
	if (!sc) return nullptr;
	const int32_t pp[4] = {0, 1, 2, 3};
	static int32_t admitted = -1;
	if (admitted < 0) {
		LegacyPool lp(fx.model, 1);
		sslm_seq d = nullptr;
		if (sslm_seq_create(fx.model, &lp.pool, &d) == SSLM_OK) {
			sslm_seq_set_schema(d, sc);
			int32_t c = 0;
			sslm_prefill(fx.model, d, pp, 4, 8, SSLM_SPAN_PROMPT, nullptr, &c);
			admitted = Token(fx.model, d);
			sslm_seq_release(d);
		}
	}
	PKV_CHECK_MSG(admitted >= 0, "9.14 A18: no admitted token from the schema's start state");
	sslm_prefix p = nullptr;
	sslm_status st = sslm_prefix_begin_budgeted(fx.model, pool, 8, &p);
	if (st == SSLM_OK) st = sslm_prefix_set_schema(p, sc);
	int32_t c = 0;
	if (st == SSLM_OK) st = sslm_prefix_prefill(fx.model, p, pp, 4, 8, SSLM_SPAN_PROMPT, nullptr, &c);
	if (st == SSLM_OK) st = sslm_prefix_prefill(fx.model, p, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c);
	if (st == SSLM_OK) st = sslm_prefix_freeze(p);
	PKV_CHECK_MSG(st == SSLM_OK, "9.14 A18: schema-progress prefix: status %s", Name(st));
	if (st != SSLM_OK && p) {
		sslm_prefix_release(p);
		p = nullptr;
	}
	return p;
}

// A second mapping of the fixture's bytes: the same weights under another model handle, so its
// prefixes are "a prefix of another model" (A17).
struct OtherModel {
	sslm_model model = nullptr;
	explicit OtherModel(const Fixture& fx) {
		PKV_CHECK_EQ(sslm_model_map(fx.bytes.p, fx.size, &model), SSLM_OK);
	}
	~OtherModel() {
		if (model) sslm_model_unmap(model);
	}
};

// The keeps rows A1, A4, A5 (null and mismatched handle) and A8 over the population.
void Cell914KeepsRestoreAndAdopt(const Fixture& fx) {
	for (int32_t P : kPopP)
		for (int32_t b : kPopB) {
			const Original fresh = MakeOriginal(fx, P, b, Sp::kOrigin);  // a fresh adopter's next turn (A8)
			for (Sp sp : kPopSp) {
				const Original o = sp == Sp::kOrigin ? fresh : MakeOriginal(fx, P, b, sp);
				if (!o.ok) continue;
				const std::string name = CaseName(P, b, sp);
				const Kept want = ReadKept(fx, o.blob);
				PKV_CHECK_MSG(want.ok && want.mode == 1 && want.budget == b && want.origin == P,
				              "9.14 %s: the original's own SSB6 header", name.c_str());
				struct Row {
					const char* id;
					How how;
					bool adopt;
				};
				for (const Row& row : {Row{"A1", How::kNone, false}, Row{"A4", How::kHandle, false}, Row{"A5 null", How::kNullShared, false},
				                       Row{"A5 mismatched", How::kMismatch, false}, Row{"A8", How::kHandle, true}}) {
					std::unique_ptr<Twin> t = MakeTwin(fx, P, b, o.blob, row.how, false);
					PKV_CHECK_MSG(t->restore == SSLM_OK, "9.14 %s %s: restore status %s", row.id, name.c_str(), Name(t->restore));
					if (!t->seq) continue;
					if (row.adopt) PKV_CHECK_EQ(sslm_seq_adopt_prefix(t->seq, t->prefix), SSLM_OK);
					const Kept k = ReadKept(fx, Save(t->seq));
					// kills: rev 2's budget := limit and the pre-plan S1 (budget or origin rewritten by the verb)
					PKV_CHECK_MSG(k.ok && k.mode == 1 && k.budget == b && k.origin == P &&
					                  k.context_length == (row.adopt ? P : want.context_length),
					              "9.14 %s %s: (mode, budget, origin, L) = (%u, %d, %lld, %lld)", row.id, name.c_str(), k.mode, k.budget,
					              static_cast<long long>(k.origin), static_cast<long long>(k.context_length));
					const Turn turn = NextTurn(fx.model, t->seq);
					PKV_CHECK_MSG(turn == (row.adopt ? fresh.turn : o.turn), "9.14 %s %s: next turn differs (%zu vs %zu tokens)", row.id,
					              name.c_str(), turn.tokens.size(), (row.adopt ? fresh.turn : o.turn).tokens.size());
					t.reset();
					// the reservation after reset and re-adopt (and, for A8, after the adopt itself)
					PKV_CHECK_MSG(ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> {
						              std::unique_ptr<Twin> r = MakeTwin(fx, P, b, o.blob, row.how, !row.adopt);
						              if (row.adopt && r->seq) sslm_seq_adopt_prefix(r->seq, r->prefix);
						              return r;
					              }, FreeAfterReadopt(fx, P, b, row.how)),
					              "9.14 %s %s: reservation", row.id, name.c_str());
				}
			}
		}
}

// A6: whole_reserve SSB6, 1.9.0's SSB5 and SSB4 with a matching handle: kept as A2/A3, handle ignored.
void Cell914A6(const Fixture& fx) {
	const int64_t K = kSlack, Rcap = fx.R(fx.geo.context_cap);
	for (int32_t P : kPopP)
		for (Sp sp : {Sp::kOrigin, Sp::kMid}) {  // a whole_reserve holder's limit is the cap; "at limit" is 4,096 tokens away
			std::vector<uint8_t> ssb5;
			std::vector<int32_t> next8;
			{
				Rig src(fx, static_cast<uint32_t>(fx.R(std::max(P, 1)) + Rcap + 4));
				sslm_prefix px = src.Prefix(BudgetPrefix(fx, src.Pool(), PrefixTokens(P)));
				sslm_seq s = nullptr;
				PKV_CHECK_EQ(sslm_seq_create(fx.model, src.Pool(), &s), SSLM_OK);
				if (!px || !src.Seq(s)) continue;
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
				DriveTo(fx, s, P, 16, sp);
				ssb5 = Save(s);
				next8 = DecodeN(fx.model, s, 8);
			}
			std::vector<uint8_t> ssb6wr = Ssb6WholeReserveFromSsb5(ssb5, fx, P);
			for (const std::vector<uint8_t>* blob : {&ssb5, &ssb6wr}) {
				const std::string name = std::string(blob == &ssb5 ? "SSB5" : "whole_reserve SSB6") + " " + CaseName(P, 16, sp);
				auto build = [&](bool release_prefix) {
					auto rig = std::make_unique<Twin>(fx, static_cast<uint32_t>(PrefixPages(fx, P) + Rcap + K));
					rig->prefix = rig->Prefix(BudgetPrefix(fx, rig->Pool(), PrefixTokens(P)));
					rig->restore = sslm_seq_restore_shared(fx.model, rig->Pool(), blob->data(), blob->size(), rig->prefix, &rig->seq, nullptr);
					rig->Seq(rig->seq);
					if (release_prefix && rig->prefix) {
						if (rig->seq) sslm_seq_reset(rig->seq);
						rig->ReleasePrefix(rig->prefix);
						rig->prefix = nullptr;
					}
					return rig;
				};
				{
					std::unique_ptr<Twin> t = build(false);
					PKV_CHECK_MSG(t->restore == SSLM_OK, "9.14 A6 %s: restore status %s", name.c_str(), Name(t->restore));
					if (!t->seq) continue;
					// kept as A2/A3: whole_reserve (its own save is 1.9.0's SSB5), the same state
					PKV_CHECK_MSG(Save(t->seq) == ssb5, "9.14 A6 %s: the re-save is not the legacy holder's SSB5", name.c_str());
					PKV_CHECK_MSG(DecodeN(fx.model, t->seq, 8) == next8, "9.14 A6 %s: next tokens differ", name.c_str());
				}
				// the handle is ignored: after a reset and the handle's release, all its pages are free
				// kills: a whole_reserve restore that shares floor(P/B) pages
				PKV_CHECK_MSG(ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> { return build(true); }, K + PrefixPages(fx, P)),
				              "9.14 A6 %s: reservation", name.c_str());
			}
		}
	// SSB4 (v1.8.1's pins) with a handle: restore, continue as v1.8.1 did, re-save as 1.9.0's SSB5.
	for (const char* sc : {"persist", "saturating"}) {
		const char* stage = std::strcmp(sc, "persist") == 0 ? "saved" : "saturated";
		const std::vector<uint8_t> ssb4 = Pin("v1.8.1", sc, stage);
		const RefRecord* cont = RefLookup(Reference("v1.8.1", fx), sc, "continuation8");
		if (ssb4.empty() || !cont) continue;
		Rig rig(fx, static_cast<uint32_t>(PrefixPages(fx, 1000) + Rcap + K));
		sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(1000)));
		sslm_seq r = nullptr;
		PKV_CHECK_MSG(sslm_seq_restore_shared(fx.model, rig.Pool(), ssb4.data(), ssb4.size(), px, &r, nullptr) == SSLM_OK,
		              "9.14 A6 SSB4 %s: restore", sc);
		if (!rig.Seq(r)) continue;
		PKV_CHECK_MSG(IsMagic(Save(r), "SSB5"), "9.14 A6 SSB4 %s: not whole_reserve after restore", sc);
		PKV_CHECK_MSG(DecodeN(fx.model, r, 8) == cont->tokens, "9.14 A6 SSB4 %s: continuation differs from v1.8.1's", sc);
	}
}

// A20: prefix mode {whole_reserve, budget} x adopter {whole_reserve holder, budget holder, begin_from
// child}, over prefix lengths x budgets x the holder's state before the adopt (fresh, used to mid,
// used to its limit). The adopter's mode decides: a whole_reserve holder copies (the prefix's release
// frees all its pages), a budget holder and a child share (its full pages stay held). A9 is the
// begin_from column: the child keeps its declared budget and starts at the parent's length.
void Cell914A20(const Fixture& fx) {
	const int64_t K = kSlack, Rcap = fx.R(fx.geo.context_cap);
	for (int legacy_prefix = 0; legacy_prefix < 2; ++legacy_prefix)
		for (int32_t P : kPopP)
			for (int32_t b : kPopB) {
				const char* pm = legacy_prefix ? "whole_reserve prefix" : "budget prefix";
				const Original fresh = MakeOriginal(fx, P, b, Sp::kOrigin);
				auto make_prefix = [&](Rig& rig) {
					// a legacy prefix draws ceil(cap/B) until its freeze returns the unmapped ones
					return rig.Prefix(legacy_prefix ? LegacyPrefix(fx, rig.Pool(), PrefixTokens(P))
					                                : BudgetPrefix(fx, rig.Pool(), PrefixTokens(P)));
				};
				const int64_t build_peak = legacy_prefix ? Rcap : fx.R(std::max(P, 1));
				for (Sp before : kPopSp) {
					// a whole_reserve holder's own limit is the cap, so it is driven to "mid" at most
					// (1) budget holder
					auto holder = [&](bool legacy_holder, bool release_prefix) {
						const int64_t own = legacy_holder ? Rcap : fx.R(b);
						auto rig = std::make_unique<Twin>(fx, static_cast<uint32_t>(std::max(build_peak, PrefixPages(fx, P) + own) + own + K));
						sslm_seq s = nullptr;
						const sslm_status cs = legacy_holder ? sslm_seq_create(fx.model, rig->Pool(), &s)
						                                     : sslm_seq_create_budgeted(fx.model, rig->Pool(), b, &s);
						PKV_CHECK_EQ(cs, SSLM_OK);
						rig->seq = rig->Seq(s);
						if (!s) return rig;
						// the holder's own state before the adopt: fresh, or its own turn driven
						if (before != Sp::kOrigin) DriveTo(fx, s, 0, legacy_holder ? 16 : b, before);
						rig->prefix = make_prefix(*rig);
						rig->restore = rig->prefix ? sslm_seq_adopt_prefix(s, rig->prefix) : SSLM_INVALID_ARGUMENT;
						if (release_prefix && rig->prefix) {
							rig->ReleasePrefix(rig->prefix);
							rig->prefix = nullptr;
						}
						return rig;
					};
					for (int legacy_holder = 0; legacy_holder < 2; ++legacy_holder) {
						if (legacy_holder && (b != kPopB[0] || before == Sp::kLimit)) continue;  // no budget axis
						const std::string name = std::string(pm) + " x " + (legacy_holder ? "whole_reserve holder" : "budget holder") +
						                         " (" + SpName(before) + ") " + CaseName(P, b, Sp::kOrigin);
						std::unique_ptr<Twin> t = holder(legacy_holder, false);
						PKV_CHECK_MSG(t->restore == SSLM_OK, "9.14 A20 %s: adopt status %s", name.c_str(), Name(t->restore));
						if (!t->seq || t->restore != SSLM_OK) continue;
						const std::vector<uint8_t> blob = Save(t->seq);
						if (legacy_holder) {
							PKV_CHECK_MSG(IsMagic(blob, "SSB5") && ParseBlobWithHidden(blob, HiddenSize(fx)).context_length == P,
							              "9.14 A20 %s: not a whole_reserve holder at the prefix's length", name.c_str());
							const size_t n = std::min<size_t>(4, fresh.turn.tokens.size());
							PKV_CHECK_MSG(P == 0 || DecodeN(fx.model, t->seq, static_cast<int64_t>(n)) ==
							                            std::vector<int32_t>(fresh.turn.tokens.begin(), fresh.turn.tokens.begin() + n),
							              "9.14 A20 %s: tokens differ from the budget adopter's", name.c_str());
						} else {
							const Kept k = ReadKept(fx, blob);
							PKV_CHECK_MSG(k.ok && k.mode == 1 && k.budget == b && k.origin == P && k.context_length == P,
							              "9.14 A20 %s: (mode, budget, origin) not kept", name.c_str());
							PKV_CHECK_MSG(NextTurn(fx.model, t->seq) == fresh.turn, "9.14 A20 %s: next turn differs", name.c_str());
						}
						t.reset();
						// after the prefix's release: a copy frees every prefix page, a share keeps the full ones
						const int64_t held = legacy_holder ? 0 : SharedPages(fx, P);
						const int64_t own = legacy_holder ? Rcap : fx.R(b);
						const int64_t pool = std::max(build_peak, PrefixPages(fx, P) + own) + own + K;
						// kills: an adopter whose mode does not decide (a legacy holder that shares, or a budget
						// holder that copies)
						PKV_CHECK_MSG(ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> { return holder(legacy_holder, true); },
						                               pool - own - held),
						              "9.14 A20 %s: pages after the prefix's release", name.c_str());
					}
				}
				// (3) the begin_from child (A9 for the budget prefix): keeps the declared budget, origin = P.
				const std::string name = std::string(pm) + " x begin_from child " + CaseName(P, b, Sp::kOrigin);
				auto child = [&](bool release_parent, int64_t* consumed, sslm_status* end) {
					auto rig = std::make_unique<Twin>(fx, static_cast<uint32_t>(std::max(build_peak, PrefixPages(fx, P) + fx.R(b)) + fx.R(b) + K));
					rig->prefix = make_prefix(*rig);
					sslm_prefix c = nullptr;
					rig->restore = rig->prefix ? sslm_prefix_begin_from(rig->prefix, b, &c) : SSLM_INVALID_ARGUMENT;
					rig->Prefix(c);
					if (consumed && c) {
						const std::vector<int32_t> more = Stream(94, b + 1, kVocab);
						size_t at = 0;
						sslm_status st = SSLM_OK;
						while (at < more.size()) {
							int32_t got = 0;
							st = sslm_prefix_prefill(fx.model, c, more.data() + at, static_cast<int32_t>(more.size() - at), kChunk,
							                         SSLM_SPAN_PROMPT, nullptr, &got);
							at += static_cast<size_t>(got > 0 ? got : 0);
							if (st != SSLM_OK || got <= 0) break;
						}
						*consumed = static_cast<int64_t>(at);
						*end = st;
					}
					if (release_parent && rig->prefix) {
						rig->ReleasePrefix(rig->prefix);
						rig->prefix = nullptr;
					}
					return rig;
				};
				int64_t consumed = -1;
				sslm_status end = SSLM_OK;
				{
					std::unique_ptr<Twin> t = child(false, &consumed, &end);
					PKV_CHECK_MSG(t->restore == SSLM_OK, "9.14 A20/A9 %s: begin_from status %s", name.c_str(), Name(t->restore));
				}
				// kills: a child whose limit is not parent length + declared budget (A9)
				PKV_CHECK_MSG(consumed == Limit(fx, P, b) - P && end == PKV_KV_BUDGET_EXCEEDED,
				              "9.14 A20/A9 %s: the child took %lld tokens before %s, expected %lld", name.c_str(),
				              static_cast<long long>(consumed), Name(end), static_cast<long long>(Limit(fx, P, b) - P));
				// kills: a child that copies the parent instead of sharing its full pages
				const int64_t pool = std::max(build_peak, PrefixPages(fx, P) + fx.R(b)) + fx.R(b) + K;
				PKV_CHECK_MSG(ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> { return child(true, nullptr, nullptr); },
				                               pool - fx.R(b) - SharedPages(fx, P)),
				              "9.14 A20/A9 %s: pages after the parent's release", name.c_str());
			}
}

// A16-A19 refused over the population: the status, the holder's re-saved blob unchanged, and the
// fill probe unchanged (the pool keeps exactly its slack free).
void Cell914Refusals(const Fixture& fx) {
	OtherModel other_model(fx);
	enum class Ref { kA16, kA17Unfrozen, kA17Model, kA18, kA19 };
	struct RefRow {
		const char* id;
		Ref r;
		sslm_status want;
		int64_t extra;  // pages the refused input occupies in the holder's pool
	};
	const RefRow rows[] = {
	    {"A16 (prefix of another pool)", Ref::kA16, SSLM_INVALID_ARGUMENT, 0},
	    {"A17 (unfrozen prefix)", Ref::kA17Unfrozen, SSLM_INVALID_ARGUMENT, fx.R(4)},
	    {"A17 (prefix of another model)", Ref::kA17Model, SSLM_INVALID_ARGUMENT, 0},
	    {"A18 (schema progress)", Ref::kA18, SSLM_PREFIX_SCHEMA_MISMATCH, 1},
	    {"A19 (unfrozen parent)", Ref::kA19, SSLM_INVALID_ARGUMENT, fx.R(4)},
	};
	for (int32_t P : kPopP)
		for (int32_t b : kPopB)
			for (Sp sp : kPopSp) {
				const Original o = MakeOriginal(fx, P, b, sp);
				if (!o.ok) continue;
				const std::string name = CaseName(P, b, sp);
				for (const RefRow& row : rows) {
					if (row.r == Ref::kA19 && sp != Sp::kOrigin) continue;  // begin_from takes no holder: once per (P, budget)
					auto build = [&](sslm_status* got, std::vector<uint8_t>* resaved) {
						auto t = std::make_unique<Twin>(fx, TwinPoolPages(fx, P, b, How::kHandle) + static_cast<uint32_t>(row.extra));
						t->prefix = t->Prefix(BudgetPrefix(fx, t->Pool(), PrefixTokens(P)));
						if (!t->prefix) return t;
						sslm_prefix bad = nullptr;
						std::unique_ptr<Rig> elsewhere;
						std::unique_ptr<LegacyPool> other_pool;
						switch (row.r) {
							case Ref::kA16:
								elsewhere = std::make_unique<Rig>(fx, static_cast<uint32_t>(fx.R(std::max(P, 1)) + 2));
								bad = elsewhere->Prefix(BudgetPrefix(fx, elsewhere->Pool(), PrefixTokens(P)));
								break;
							case Ref::kA17Unfrozen:
							case Ref::kA19:
								bad = t->Prefix(BudgetPrefix(fx, t->Pool(), PrefixTokens(4), false));
								break;
							case Ref::kA17Model:
								other_pool = std::make_unique<LegacyPool>(other_model.model, 1);
								PKV_CHECK_EQ(sslm_prefix_begin(other_model.model, &other_pool->pool, &bad), SSLM_OK);
								if (bad) {
									PrefixPrefillAll(other_model.model, bad, PrefixTokens(P > 0 ? P : 1), kChunk);
									sslm_prefix_freeze(bad);
								}
								break;
							case Ref::kA18:
								bad = t->Prefix(SchemaProgressPrefix(fx, t->Pool(), nullptr));
								break;
						}
						t->restore = sslm_seq_restore_shared(fx.model, t->Pool(), o.blob.data(), o.blob.size(), t->prefix, &t->seq, nullptr);
						t->Seq(t->seq);
						if (row.r == Ref::kA19) {
							sslm_prefix c = nullptr;
							*got = bad ? sslm_prefix_begin_from(bad, b, &c) : SSLM_OK;
							if (c) t->Prefix(c);
						} else {
							*got = t->seq && bad ? sslm_seq_adopt_prefix(t->seq, bad) : SSLM_OK;
						}
						if (resaved && t->seq) *resaved = Save(t->seq);
						if (row.r == Ref::kA17Model && bad) sslm_prefix_release(bad);
						if (elsewhere) elsewhere.reset();
						return t;
					};
					sslm_status got = SSLM_OK;
					std::vector<uint8_t> resaved;
					build(&got, &resaved);
					// kills: the refusal moved or dropped (the verb proceeds)
					PKV_CHECK_MSG(got == row.want, "9.14 %s %s: status %s, want %s", row.id, name.c_str(), Name(got), Name(row.want));
					// kills: the check made after a state change (the re-saved blob or the pool moved)
					PKV_CHECK_MSG(resaved == o.blob, "9.14 %s %s: the holder's re-saved blob changed", row.id, name.c_str());
					PKV_CHECK_MSG(ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> {
						              sslm_status g = SSLM_OK;
						              return build(&g, nullptr);
					              }, kSlack),
					              "9.14 %s %s: pages", row.id, name.c_str());
				}
			}
	// A18's revocation, on bind-eligible (fresh) budget holders: after A18's refusal
	// sslm_seq_set_schema returns SSLM_SCHEMA_BIND_REJECTED; after A16 or A17 it returns SSLM_OK.
	Rig rig(fx, 400);
	Rig elsewhere(fx, 100);
	sslm_schema sc = nullptr;
	sslm_prefix progress = rig.Prefix(SchemaProgressPrefix(fx, rig.Pool(), &sc));
	sslm_prefix foreign = elsewhere.Prefix(BudgetPrefix(fx, elsewhere.Pool(), PrefixTokens(17)));
	sslm_prefix unfrozen = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(4), false));
	LegacyPool om_pool(other_model.model, 1);
	sslm_prefix other_model_px = nullptr;
	PKV_CHECK_EQ(sslm_prefix_begin(other_model.model, &om_pool.pool, &other_model_px), SSLM_OK);
	if (other_model_px) PKV_CHECK_EQ(sslm_prefix_freeze(other_model_px), SSLM_OK);
	struct Rev {
		const char* id;
		sslm_prefix p;
		sslm_status adopt;
		sslm_status bind;
	};
	for (const Rev& r : {Rev{"A16", foreign, SSLM_INVALID_ARGUMENT, SSLM_OK}, Rev{"A17 unfrozen", unfrozen, SSLM_INVALID_ARGUMENT, SSLM_OK},
	                     Rev{"A17 model", other_model_px, SSLM_INVALID_ARGUMENT, SSLM_OK},
	                     Rev{"A18", progress, SSLM_PREFIX_SCHEMA_MISMATCH, SSLM_SCHEMA_BIND_REJECTED}}) {
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, rig.Pool(), 64, &s), SSLM_OK);
		if (!rig.Seq(s) || !r.p || !sc) continue;
		PKV_CHECK_MSG(sslm_seq_adopt_prefix(s, r.p) == r.adopt, "9.14 revocation %s: adopt status", r.id);
		// kills: A18's revocation moved or dropped (decision 52 keeps it), or a revocation added to A16/A17
		PKV_CHECK_MSG(sslm_seq_set_schema(s, sc) == r.bind, "9.14 revocation %s: sslm_seq_set_schema after the refusal", r.id);
	}
	if (other_model_px) sslm_prefix_release(other_model_px);
}

void Cell914Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Cell914KeepsRestoreAndAdopt(fx);
	Cell914A6(fx);
	Cell914A20(fx);
	Cell914Refusals(fx);
}

PKV_CELL("9.1/C5", "C5", Cell91Budget);
PKV_CELL("9.3/C5", "C5", Cell93Budget);
PKV_CELL("9.4", "C5", Cell94);
PKV_CELL("9.5/C5", "C5", Cell95Budget);
PKV_CELL("9.9", "C5", Cell99);
PKV_CELL("9.10", "C5", Cell910);
PKV_CELL("9.11", "C5", Cell911);
PKV_CELL("9.14/C5", "C5", Cell914Budget);

}  // namespace
