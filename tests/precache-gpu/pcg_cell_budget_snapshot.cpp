// Precached-prefixes plan §3.1 on GPU [F2]: a frozen prefix is, for every purpose an adopter can observe, the
// state the blob carries. An adopter observes the prefix's prefill snapshot through
// sslm_gpu_seq_read_prefill_final_hidden (adopt copies it), and the restore rebuilds that snapshot from the
// blob's residual, valid whenever the prefix is ready. So a prefix whose last prefill landed tokens and then
// ended SSLM_GPU_KV_BUDGET_EXCEEDED (ready, at full depth, savable) must give its adopters the same snapshot
// read, original and restored alike: the same status, codes and scale, and the same greedy tokens.
//
// RED before the fix: the budget-exceeded exit set ready_for_logits but left the snapshot invalid, so an
// adopter of the original read SSLM_PREFILL_HIDDEN_UNAVAILABLE while an adopter of the restored prefix read
// the residual.
//
// The second leg is the same mismatch on the schema-content path: a prefill whose schema DFA rejects a
// token after earlier tokens landed returns SSLM_SEQUENCE_REJECTED with the prefix ready at full depth.
// RED before the fix: that exit left the snapshot invalid too.
//
// Rev 1.18.3 (code review of the four precache changes, findings 1-3): three legs whose last call is not the
// call that landed tokens. Restore derives the snapshot (valid iff L > 0) and the bind permission (iff L = 0)
// from L alone, so the original must hold the same values whatever call history produced it:
//   (A) a budget stop that lands tokens, then a retry that lands none: the retry's exit skips the refresh;
//   (B) a successful prefill, then a call whose token is outside the vocabulary: the bridge clears the
//       snapshot before the embed check refuses it;
//   (D) an empty prefix given a zero-count prefill, which clears its bind permission; the restored copy at
//       L = 0 has it again, so a begin_from child of each binds a schema differently.
// Each grades only that the original and the restored copy behave the same, not which remedy makes them so.
// RED at d42cec75 on all three; green at ac2fc2c7 (the freeze normalises the snapshot and the bind permission
// from L, as restore does). Mutant freeze-set-removed (apply_gpu_mutant.py, plan §15.22: the freeze-time
// setting removed) is red on A, B and D, since the per-exit refresh went with the fix. (Before the fix the
// mutant exit-refresh-only, every failing exit refreshing and nothing at freeze, was red on D only; it was
// retired with its anchor in ac2fc2c7.)
//
// Run: pcg_cell_budget_snapshot.exe --fixture-dir=DIR (pkv_def).
#include "pcg_helpers.h"

using namespace pcg;

namespace {

struct AdopterView {
	GS read = GS::SSLM_DEVICE_LOST;
	std::vector<int8_t> codes;
	int64_t m = 0, e = 0;
	std::vector<int32_t> tokens;
};

AdopterView Observe(Rig& rig, SslmGpuPrefixHandle* p, int64_t len, int32_t last_token, uint32_t hidden,
                    int32_t schema = -1) {
	AdopterView v;
	GSeq s = rig.Budgeted(256);
	if (!s.h) return v;
	if (schema >= 0) CHECK(SslmGpuSeqSetSchemaForG5Bridge(rig.ctx, s.h, schema) == GS::SSLM_OK);
	const GS st = Adopt(rig, s, p, len);
	CHECK_MSG(st == GS::SSLM_OK, "adopt: %s", StatusName(st));
	if (st == GS::SSLM_OK) {
		v.codes.assign(hidden, 0);
		size_t required = 0;
		v.read = sslm_gpu_seq_read_prefill_final_hidden(rig.ctx, s.h, v.codes.data(), v.codes.size(), &required, &v.m,
		                                                &v.e);
		s.last = last_token;
		v.tokens = DecodeN(rig.e, s, 8);
	}
	rig.ReleaseSeq(s.h);
	return v;
}

void CellBudgetSnapshot(const Env& e) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	const int32_t budget = 20;
	const std::vector<int32_t> t = PrefixTokens(30, 47);
	SslmGpuPrefixHandle* raw = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, budget, &raw);
	CHECK_MSG(st == GS::SSLM_OK, "prefix_begin(%d): %s", budget, StatusName(st));
	if (st != GS::SSLM_OK) return;
	SslmGpuPrefixHandle* p = rig.Prefix(raw);
	// One call that admits every token (full_budget is one token's dispatches), so the budget stops it.
	int32_t consumed = 0;
	st = sslm_gpu_prefix_prefill(rig.ctx, p, t.data(), static_cast<int32_t>(t.size()), 0,
	                             e.full_budget * static_cast<uint32_t>(t.size()), &consumed);
	CHECK_MSG(st == GS::SSLM_GPU_KV_BUDGET_EXCEEDED && consumed > 0,
	          "the last prefill must exceed the budget with tokens landed: %s, consumed %d", StatusName(st), consumed);
	if (consumed <= 0) return;
	CHECK(sslm_gpu_prefix_freeze(rig.ctx, p) == GS::SSLM_OK);
	const std::vector<uint8_t> blob = SavePrefix(e, p);
	if (blob.empty()) return;
	const RestoredPrefix r = RestorePrefix(rig, blob);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "restore: %s", Name(r.status));
	if (!r.p) return;
	uint32_t hidden = 0;
	CHECK(sslm_gpu_model_hidden_size(rig.model, &hidden) == GS::SSLM_OK && hidden > 0);
	const int32_t last = t[static_cast<size_t>(consumed) - 1];
	const AdopterView a = Observe(rig, p, consumed, last, hidden);
	const AdopterView b = Observe(rig, r.p, consumed, last, hidden);
	std::printf("budget snapshot: landed %d of %zu; original adopter read %s, restored adopter read %s\n", consumed,
	            t.size(), Name(a.read), Name(b.read));
	CHECK_MSG(a.read == b.read, "adopters of the original (%s) and of the restored prefix (%s) read the snapshot differently",
	          Name(a.read), Name(b.read));
	CHECK_MSG(a.read == GS::SSLM_OK, "an adopter of a ready budget-exceeded prefix reads its snapshot: %s", Name(a.read));
	if (a.read == GS::SSLM_OK && b.read == GS::SSLM_OK) {
		CHECK_MSG(a.codes == b.codes && a.m == b.m && a.e == b.e, "the two adopters' snapshot codes or scale differ");
	}
	CHECK_MSG(!a.tokens.empty() && a.tokens == b.tokens, "the two adopters' greedy tokens differ");
}

// Schema 0's DFA walked from state 0: up to `n` accepted tokens (each the smallest legal continuation),
// then one token the DFA rejects at the state they reach (the longest such walk, at least one accepted
// token). Empty when the schema cannot supply them.
std::vector<int32_t> AcceptedThenRejected(const Env& e, int32_t n) {
	const superslm::SslmSectionView* sec = e.fx.view.Section(superslm::SslmSectionType::SchemaMasks);
	superslm::SchemaMasksTable table;
	std::string err;
	const uint32_t vocab = e.fx.view.config.vocab_size;
	if (!sec || !superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size, vocab, table, &err)) return {};
	const superslm::SchemaEntry* entry = table.ByIndex(0);
	if (!entry) return {};
	std::vector<int32_t> walk, best;
	uint32_t state = 0;
	for (int32_t i = 0; i < n; ++i) {
		uint32_t next = 0;
		uint32_t tok = 0;
		while (tok < vocab && !table.Transition(*entry, state, tok, &next)) ++tok;
		if (tok >= vocab) break;
		walk.push_back(static_cast<int32_t>(tok));
		state = next;
		uint32_t bad = 0;
		while (bad < vocab && table.Transition(*entry, state, bad, nullptr)) ++bad;
		if (bad < vocab) {
			best = walk;
			best.push_back(static_cast<int32_t>(bad));
		}
	}
	return best;
}

void CellSchemaRejectSnapshot(const Env& e) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	const std::vector<int32_t> t = AcceptedThenRejected(e, 6);
	CHECK_MSG(!t.empty(), "schema 0 supplies accepted tokens and then a rejected one");
	if (t.empty()) return;
	SslmGpuPrefixHandle* raw = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, 64, &raw);
	CHECK_MSG(st == GS::SSLM_OK, "prefix_begin: %s", StatusName(st));
	if (st != GS::SSLM_OK) return;
	SslmGpuPrefixHandle* p = rig.Prefix(raw);
	st = sslm_gpu_prefix_set_schema(rig.ctx, p, 0);
	CHECK_MSG(st == GS::SSLM_OK, "set_schema: %s", StatusName(st));
	if (st != GS::SSLM_OK) return;
	int32_t consumed = 0;
	st = sslm_gpu_prefix_prefill(rig.ctx, p, t.data(), static_cast<int32_t>(t.size()), 1,
	                             e.full_budget * static_cast<uint32_t>(t.size()), &consumed);
	CHECK_MSG(st == GS::SSLM_SEQUENCE_REJECTED && consumed == static_cast<int32_t>(t.size()) - 1,
	          "the schema prefill must reject its last token after the others landed: %s, consumed %d",
	          StatusName(st), consumed);
	if (consumed <= 0) return;
	CHECK(sslm_gpu_prefix_freeze(rig.ctx, p) == GS::SSLM_OK);
	const std::vector<uint8_t> blob = SavePrefix(e, p);
	if (blob.empty()) return;
	const RestoredPrefix r = RestorePrefix(rig, blob);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "restore: %s", Name(r.status));
	if (!r.p) return;
	uint32_t hidden = 0;
	CHECK(sslm_gpu_model_hidden_size(rig.model, &hidden) == GS::SSLM_OK && hidden > 0);
	const int32_t last = t[static_cast<size_t>(consumed) - 1];
	const AdopterView a = Observe(rig, p, consumed, last, hidden, 0);
	const AdopterView b = Observe(rig, r.p, consumed, last, hidden, 0);
	std::printf("schema-reject snapshot: landed %d of %zu; original adopter read %s, restored adopter read %s\n",
	            consumed, t.size(), Name(a.read), Name(b.read));
	CHECK_MSG(a.read == b.read,
	          "schema reject: adopters of the original (%s) and of the restored prefix (%s) read the snapshot differently",
	          Name(a.read), Name(b.read));
	CHECK_MSG(a.read == GS::SSLM_OK, "an adopter of a ready schema-rejected prefix reads its snapshot: %s", Name(a.read));
	if (a.read == GS::SSLM_OK && b.read == GS::SSLM_OK) {
		CHECK_MSG(a.codes == b.codes && a.m == b.m && a.e == b.e, "schema reject: the snapshot codes or scale differ");
	}
	CHECK_MSG(!a.tokens.empty() && a.tokens == b.tokens, "schema reject: the two adopters' greedy tokens differ");
}

// Freeze, save and restore `p`; an adopter of each reads the snapshot and decodes, and the two must agree.
void CompareAdopters(Rig& rig, const Env& e, SslmGpuPrefixHandle* p, int64_t len, int32_t last, const char* leg) {
	CHECK_MSG(sslm_gpu_prefix_freeze(rig.ctx, p) == GS::SSLM_OK, "%s: freeze", leg);
	GS sst = GS::SSLM_DEVICE_LOST;
	const std::vector<uint8_t> blob = SavePrefix(e, p, &sst);
	CHECK_MSG(sst == GS::SSLM_OK && !blob.empty(), "%s: save %s", leg, Name(sst));
	if (blob.empty()) return;
	const RestoredPrefix r = RestorePrefix(rig, blob);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "%s: restore %s", leg, Name(r.status));
	if (!r.p) return;
	uint32_t hidden = 0;
	CHECK(sslm_gpu_model_hidden_size(rig.model, &hidden) == GS::SSLM_OK && hidden > 0);
	const AdopterView a = Observe(rig, p, len, last, hidden);
	const AdopterView b = Observe(rig, r.p, len, last, hidden);
	std::printf("%s: L=%lld; original adopter read %s, restored adopter read %s\n", leg, static_cast<long long>(len),
	            Name(a.read), Name(b.read));
	CHECK_MSG(a.read == b.read, "%s: adopters of the original (%s) and of the restored prefix (%s) read the snapshot differently",
	          leg, Name(a.read), Name(b.read));
	if (a.read == GS::SSLM_OK && b.read == GS::SSLM_OK) {
		CHECK_MSG(a.codes == b.codes && a.m == b.m && a.e == b.e, "%s: the two adopters' snapshot codes or scale differ", leg);
	}
	CHECK_MSG(!a.tokens.empty() && a.tokens == b.tokens, "%s: the two adopters' greedy tokens differ", leg);
}

// (A) A budget stop that lands tokens, then one retry that lands none (the shape of a chunked prefill loop).
void CellBudgetRetrySnapshot(const Env& e) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	const std::vector<int32_t> t = PrefixTokens(30, 47);
	SslmGpuPrefixHandle* raw = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, 20, &raw);
	CHECK_MSG(st == GS::SSLM_OK, "A: prefix_begin(20): %s", StatusName(st));
	if (st != GS::SSLM_OK) return;
	SslmGpuPrefixHandle* p = rig.Prefix(raw);
	int32_t c1 = 0, c2 = 0;
	st = sslm_gpu_prefix_prefill(rig.ctx, p, t.data(), 30, 0, e.full_budget * 30u, &c1);
	CHECK_MSG(st == GS::SSLM_GPU_KV_BUDGET_EXCEEDED && c1 == 20, "A: call 1 must land 20 and stop at the budget: %s, consumed %d",
	          StatusName(st), c1);
	if (c1 <= 0) return;
	st = sslm_gpu_prefix_prefill(rig.ctx, p, t.data() + c1, 1, 0, e.full_budget, &c2);
	CHECK_MSG(st == GS::SSLM_GPU_KV_BUDGET_EXCEEDED && c2 == 0, "A: the retry must land none and stop at the budget: %s, consumed %d",
	          StatusName(st), c2);
	CompareAdopters(rig, e, p, c1, t[static_cast<size_t>(c1) - 1], "A budget retry");
}

// (B) A successful prompt prefill, then a call whose only token is outside the vocabulary.
void CellEmbedRejectSnapshot(const Env& e) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	const std::vector<int32_t> t = PrefixTokens(10, 5);
	SslmGpuPrefixHandle* raw = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, 64, &raw);
	CHECK_MSG(st == GS::SSLM_OK, "B: prefix_begin: %s", StatusName(st));
	if (st != GS::SSLM_OK) return;
	SslmGpuPrefixHandle* p = rig.Prefix(raw);
	int32_t c1 = 0, c2 = 0;
	st = sslm_gpu_prefix_prefill(rig.ctx, p, t.data(), 10, 0, e.full_budget * 10u, &c1);
	CHECK_MSG(st == GS::SSLM_OK && c1 == 10, "B: call 1 must land all 10: %s, consumed %d", StatusName(st), c1);
	if (c1 != 10) return;
	const uint32_t vocab = e.fx.view.config.vocab_size;
	const int32_t bad = static_cast<int32_t>(vocab);
	st = sslm_gpu_prefix_prefill(rig.ctx, p, &bad, 1, 0, e.full_budget, &c2);
	CHECK_MSG(st != GS::SSLM_OK && c2 == 0, "B: token %d (the vocabulary is %u) must be refused with none landed: %s, consumed %d",
	          bad, vocab, StatusName(st), c2);
	std::printf("B embed reject: call 2 (token %d) %s\n", bad, Name(st));
	CompareAdopters(rig, e, p, c1, t[9], "B embed reject");
}

// (D) An empty prefix given a zero-count prefill, frozen, saved and restored at L = 0. A begin_from child of
// the original and one of the restored copy each bind schema 0, and must get the same answer.
void CellBindPermissionHistory(const Env& e) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	SslmGpuPrefixHandle* raw = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, 64, &raw);
	CHECK_MSG(st == GS::SSLM_OK, "D: prefix_begin: %s", StatusName(st));
	if (st != GS::SSLM_OK) return;
	SslmGpuPrefixHandle* p = rig.Prefix(raw);
	const int32_t unused = 0;
	int32_t c = -1;
	st = sslm_gpu_prefix_prefill(rig.ctx, p, &unused, 0, 0, e.full_budget, &c);
	CHECK_MSG(st == GS::SSLM_OK && c == 0, "D: the zero-count prefill: %s, consumed %d", StatusName(st), c);
	CHECK_MSG(sslm_gpu_prefix_freeze(rig.ctx, p) == GS::SSLM_OK, "D: freeze");
	GS sst = GS::SSLM_DEVICE_LOST;
	const std::vector<uint8_t> blob = SavePrefix(e, p, &sst);
	CHECK_MSG(sst == GS::SSLM_OK && !blob.empty(), "D: save %s", Name(sst));
	if (blob.empty()) return;
	const RestoredPrefix r = RestorePrefix(rig, blob);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "D: restore %s", Name(r.status));
	if (!r.p) return;
	SslmGpuPrefixHandle* k1 = nullptr;
	SslmGpuPrefixHandle* k2 = nullptr;
	const GS b1 = sslm_gpu_prefix_begin_from(rig.ctx, p, 16, &k1);
	const GS b2 = sslm_gpu_prefix_begin_from(rig.ctx, r.p, 16, &k2);
	k1 = rig.Prefix(k1);
	k2 = rig.Prefix(k2);
	CHECK_MSG(b1 == GS::SSLM_OK && b2 == GS::SSLM_OK, "D: begin_from: original %s, restored %s", Name(b1), Name(b2));
	if (!k1 || !k2) return;
	const GS s1 = sslm_gpu_prefix_set_schema(rig.ctx, k1, 0);
	const GS s2 = sslm_gpu_prefix_set_schema(rig.ctx, k2, 0);
	std::printf("D bind permission: child of original set_schema %s, child of restored set_schema %s\n", Name(s1), Name(s2));
	CHECK_MSG(s1 == s2, "D: children of the original (%s) and of the restored prefix (%s) bind a schema differently",
	          Name(s1), Name(s2));
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			CellBudgetSnapshot(e);
			CellSchemaRejectSnapshot(e);
			CellBudgetRetrySnapshot(e);
			CellEmbedRejectSnapshot(e);
			CellBindPermissionHistory(e);
		}
	}
	return FinishSuite("pcg_cell_budget_snapshot");
}
