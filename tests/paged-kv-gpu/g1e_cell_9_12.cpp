// Paged-KV plan (rev 16.2) §7 cell 9.12 [G5]: "GPU twins of 9.9-9.11 through
// sslm_gpu_seq_restore_into_pool, on both GPUs", with rev 7's and rev 8's case list for 9.10.
// Group g1e, step G1. Plan sources: §3.7 (restore keeps the holder's contract; re-sharing under (a)-(f),
// (f) on the host mirror; private restore draws R + E, E back at the first reset, adopt or release),
// §3.9 A12/A13, §4 item 10, the CPU cells 9.9-9.11 (tests/paged-kv/c5_dim9_persistence.cpp, whose
// constructions these transcribe verb for verb).
//
// Grading (§7, §8): page counts by the GPU fill probe or an N / N - 1 admission pair (creates through
// sslm_gpu_seq_create_budgeted, read after sslm_gpu_ready); bytes by the mirror peek, the blob, or an
// unsaved GPU original; never sslm_gpu_kv_pool_stats or *out_shared_pages (printed as diagnostics only).
//
// 9.9's twin
//   Fixed case: a budget-512 pooled sequence adopts the persona (1,200) and takes a 300-token turn, so
//   it rests ready at 1,500 (limit 1,712) and saves SLM6. Restored with the persona handle and
//   without: each continues to refusal emitting 213 tokens equal to the unsaved original's; after a
//   reset and a re-adopt each emits 513 (rev 2's budget := limit emits 1,713). Pages by admission:
//   with the handle a pool of 108 admits the restore and 107 refuses it; without, 183 and 182. After
//   the no-handle twin's reset and re-adopt the fill probe finds exactly the 75 returned pages.
//   Population: prefix lengths {0,1,15,16,17,1000,1008,1200} x budgets {1,15,16,17,512} x save points
//   {origin, mid, limit}, each restored with and without the handle (240 cases): the next turn equals
//   the unsaved original's; after the twin writes into its tail the prefix's pages read as before
//   (mirror peek; the ceil-span mutant writes into the frozen prefix's partial page); after the twin's
//   reset and re-adopt the fill probe finds E (no handle) or 0 plus the slack, the original's R(budget).
// 9.10's twin (rev 8's list): each case restores privately with SSLM_OK and the saved budget kept,
//   graded by admission in a pool of exactly handle pages + R + E (a 2-page create is refused; a
//   sharing restore would have left E free), and the next tokens equal the original's.
//   Inherited: (e) the longer handle (1,201; span bytes equal, so only (e) rejects); (f) the edited
//   blob (one byte of layer L-1's V half at position 500), whose re-save is the edited SLM6 byte for
//   byte; (f) the real producer (the persona's last full-page token differs); (b) the unfrozen handle.
//   (c) is implied by (d), as on CPU: stated, not tested.
//   Rewritten: (d) a handle from another pool of the same context is A13's private restore, with the
//   foreign pool untouched (2.11 asserts another context and another model); (e)'s shorter handle (the
//   world at 1,000) asserts the private restore and the next tokens only: the GPU has no table
//   sentinel, so a build that evaluates (f) before (e) is not separated here.
//   Refused, not inherited: (a), a non-budget blob with a handle, is A14: g1e_cell_9_13.cpp asserts it
//   (SSLM_GPU_HOLDER_MODE_MISMATCH, null out-handle, fill probe unchanged), with the persona handle.
// 9.11's twin, at its own construction (persona chain, origin 1,200, budget 512, a pool of 183): a
//   private restore draws R + E (admitted at 183, refused at 182); the fill probe finds E more after the
//   first reset, E more after the first adopt, R + E more after the first release (the mutant that sends
//   materialized pages to the reserve fails at reset and adopt; "release returns them to nothing" fails
//   at release). Two restores with a null prefix re-save byte-identically to the blob and continue
//   alike; a null out_shared_pages with the matching handle is accepted. NOT twinned: 9.11's "freed
//   pages read 0xCD through the peek" (§3.3's CPU poison rule): §4 item 3 states no free-page fill for
//   the GPU pool, so there is nothing to assert (recorded in the g1e notes).
//
// RED NOW: link-red (G3 pool/budgeted create/peek, G4 prefixes/begin_from/adopt, G5 restore-into-pool);
// after G4, red at run until G5. GREEN: every check passes on both GPUs, debug layer clean.
//
// Run: g1e_cell_9_12.exe --fixture-dir=DIR   (the population is long: 240 cases, each with a GPU original,
// a twin and four fill-probe rebuilds)
#include "g1e_helpers.h"

using namespace g1e;

namespace {

constexpr int32_t kCohortBudget = 512;
constexpr int64_t kPersonaLen = 1200;
constexpr int64_t kSlack = 3;

void ResetSeq(const Env& e, GSeq& s) {
	CHECK(sslm_gpu_seq_reset(e.ctx, s.h) == GS::SSLM_OK);
	SslmGpuSequenceHandle* h = s.h;
	s = GSeq{};
	s.h = h;
}

// ---- the fixed cohort case ----------------------------------------------------------------------

struct Fixed {
	bool ok = false;
	std::vector<uint8_t> blob;  // SLM6 at 1,500
	GSeq at_save;               // the original's token state at the save
	Turn cont;                  // the unsaved original's continuation
	std::vector<int32_t> readopt;
	GS readopt_refusal = GS::SSLM_OK;
	std::vector<int32_t> next8;
};

const Fixed& FixedCase(const Env& e) {
	static Fixed f;
	static bool made = false;
	if (made) return f;
	made = true;
	Rig rig(e, 600);
	const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true) : GChain{};
	GSeq s = c.ok ? rig.Budgeted(kCohortBudget) : GSeq{};
	if (!s.h) return f;
	CHECK(Adopt(rig, s, c.persona, kPersonaLen) == GS::SSLM_OK);
	CHECK(Prefill(e, s, Stream(91, 300, kVocab)) == GS::SSLM_OK);
	f.blob = Save(e, s.h);
	f.at_save = s;
	if (!Slm6SelfCheck(f.blob, e.geo, kCohortBudget, kPersonaLen, 1500, "9.12 fixed case")) return f;
	f.cont = NextTurn(e, s);
	CHECK_MSG(f.cont.tokens.size() == 213 && f.cont.refusal == GS::SSLM_GPU_KV_BUDGET_EXCEEDED,
	          "9.12 fixed case: the original emitted %zu tokens and refused with %s (want 213, budget exceeded)",
	          f.cont.tokens.size(), StatusName(f.cont.refusal));
	f.next8.assign(f.cont.tokens.begin(), f.cont.tokens.begin() + std::min<size_t>(8, f.cont.tokens.size()));
	ResetSeq(e, s);
	CHECK(Adopt(rig, s, c.persona, kPersonaLen) == GS::SSLM_OK);
	f.readopt = RunToRefusal(e, s, &f.readopt_refusal);
	CHECK_MSG(f.readopt.size() == 513, "9.12 fixed case: the original emitted %zu tokens after re-adopt (want 513)",
	          f.readopt.size());
	f.ok = !f.blob.empty();
	return f;
}

void Fixed99(const Env& e) {
	const Fixed& f = FixedCase(e);
	if (!f.ok) return;
	const Geo& g = e.geo;
	const int64_t chain = g.PersonaChainPages(200, true), R = g.R(kCohortBudget), E = g.Materialized(kPersonaLen, kCohortBudget);
	CHECK(chain == 75 && chain + R == 108 && chain + R + E == 183);
	for (int with = 1; with >= 0; --with) {
		const char* hw = with ? "handle" : "no handle";
		const int64_t N = chain + R + (with ? 0 : E);
		for (int64_t pages : {N, N - 1}) {
			Rig rig(e, static_cast<uint32_t>(pages));
			const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true) : GChain{};
			if (!c.ok) return;
			uint32_t diag = 0;
			Restored r = RestoreIntoPool(rig, f.blob, with ? c.persona : nullptr, f.at_save, &diag);
			if (pages == N - 1) {
				// kills: a restore drawing fewer than R (+ E): admitted one page short
				CHECK_MSG(r.status == GS::SSLM_GPU_KV_POOL_EXHAUSTED && r.raw_out == nullptr,
				          "9.12 fixed (%s): a pool of %lld admitted the restore (%s)", hw, static_cast<long long>(pages),
				          StatusName(r.status));
				continue;
			}
			// kills: a restore drawing more (with the handle: one that does not share)
			CHECK_MSG(r.status == GS::SSLM_OK, "9.12 fixed (%s): a pool of %lld refused the restore (%s)", hw,
			          static_cast<long long>(pages), StatusName(r.status));
			std::printf("9.12 fixed (%s): *out_shared_pages = %u (diagnostic)\n", hw, diag);
			if (!r.seq.h) continue;
			GSeq t = r.seq;
			const Turn turn = NextTurn(e, t);
			CHECK_MSG(turn == f.cont, "9.12 fixed (%s): the continuation differs from the unsaved original's (%zu tokens)", hw,
			          turn.tokens.size());
			ResetSeq(e, t);
			CHECK(Adopt(rig, t, c.persona, kPersonaLen) == GS::SSLM_OK);
			GS refusal = GS::SSLM_OK;
			const std::vector<int32_t> again = RunToRefusal(e, t, &refusal);
			// kills: rev 2's budget := limit (1,713 tokens, not 513)
			CHECK_MSG(again.size() == 513 && again == f.readopt && refusal == GS::SSLM_GPU_KV_BUDGET_EXCEEDED,
			          "9.12 fixed (%s): after re-adopt %zu tokens (%s), want the original's 513", hw, again.size(),
			          StatusName(refusal));
		}
	}
	// After the no-handle twin's reset and re-adopt, exactly the 75 materialized pages came back.
	// kills: materialized pages sent to the reserve (0 returned), or freed twice (76)
	ProbeExactlyFree(e, [&]() -> std::unique_ptr<ProbeState> {
		auto rig = std::make_unique<Rig>(e, static_cast<uint32_t>(chain + R + E));
		const GChain c = rig->pool ? BuildPersonaChain(*rig, 0, true) : GChain{};
		if (c.ok) {
			Restored r = RestoreIntoPool(*rig, f.blob, nullptr, f.at_save);
			if (r.seq.h) {
				GSeq t = r.seq;
				NextTurn(e, t);
				ResetSeq(e, t);
				Adopt(*rig, t, c.persona, kPersonaLen);
			}
		}
		return rig;
	}, E, "9.12 fixed: pages returned after the no-handle twin's reset and re-adopt");
}

// ---- the population ------------------------------------------------------------------------------

enum class Sp { kOrigin, kMid, kLimit };
const char* SpName(Sp sp) { return sp == Sp::kOrigin ? "origin" : sp == Sp::kMid ? "mid" : "limit"; }
constexpr int32_t kPopP[] = {0, 1, 15, 16, 17, 1000, 1008, 1200};
constexpr int32_t kPopB[] = {1, 15, 16, 17, 512};
constexpr Sp kPopSp[] = {Sp::kOrigin, Sp::kMid, Sp::kLimit};

void DriveTo(const Env& e, GSeq& s, int32_t P, int32_t b, Sp sp) {
	if (sp == Sp::kOrigin) return;
	if (P == 0) CHECK(Prefill(e, s, Stream(81, 1, kVocab)) == GS::SSLM_OK);
	GS st = GS::SSLM_OK;
	if (sp == Sp::kMid) DecodeN(e, s, b / 2, &st);
	else RunToRefusal(e, s, &st);
}

struct Original {
	bool ok = false;
	std::vector<uint8_t> blob;
	GSeq at_save;
	Turn turn;
};

std::string CaseName(int32_t P, int32_t b, Sp sp) {
	return "P=" + std::to_string(P) + " budget=" + std::to_string(b) + " at " + SpName(sp);
}

Original MakeOriginal(const Env& e, int32_t P, int32_t b, Sp sp) {
	Original o;
	Rig rig(e, static_cast<uint32_t>(e.geo.R(std::max(P, 1)) + e.geo.R(b) + 8));
	SslmGpuPrefixHandle* px = rig.pool ? BudgetPrefix(rig, PrefixTokens(P)) : nullptr;
	GSeq s = px ? rig.Budgeted(b) : GSeq{};
	if (!s.h) return o;
	CHECK(Adopt(rig, s, px, P) == GS::SSLM_OK);
	DriveTo(e, s, P, b, sp);
	o.blob = Save(e, s.h);
	o.at_save = s;
	// kills: a pooled save that is not a budget SLM6 carrying the declared budget and origin
	o.ok = Slm6SelfCheck(o.blob, e.geo, b, P, ContextLength(s.h), CaseName(P, b, sp).c_str());
	o.turn = NextTurn(e, s);
	return o;
}

enum class How { kHandle, kNone };

uint32_t TwinPoolPages(const Env& e, int32_t P, int32_t b, How how) {
	int64_t n = e.geo.PrefixPages(P) + e.geo.R(b) + kSlack;
	if (how == How::kNone) n += e.geo.Materialized(P, b);
	return static_cast<uint32_t>(n);
}

struct Twin : Rig {
	SslmGpuPrefixHandle* prefix = nullptr;
	Restored r;
	PrefixSnap before;
	uint32_t pages;
	Twin(const Env& e, uint32_t n) : Rig(e, n), pages(n) {}
};

std::unique_ptr<Twin> MakeTwin(const Env& e, int32_t P, int32_t b, const Original& o, How how, bool reset_readopt,
                               bool snap) {
	auto t = std::make_unique<Twin>(e, TwinPoolPages(e, P, b, how));
	if (!t->pool) return t;
	t->prefix = BudgetPrefix(*t, PrefixTokens(P));
	if (!t->prefix) return t;
	if (snap) t->before = LocatePrefixPages(e, t->pool, t->pages, ExpectedPrefixPages(e, o.blob, P));
	t->r = RestoreIntoPool(*t, o.blob, how == How::kHandle ? t->prefix : nullptr, o.at_save);
	if (t->r.status == GS::SSLM_OK && t->r.seq.h && reset_readopt) {
		ResetSeq(e, t->r.seq);
		CHECK(Adopt(*t, t->r.seq, t->prefix, P) == GS::SSLM_OK);
	}
	return t;
}

void Population99(const Env& e) {
	int cases = 0;
	for (int32_t P : kPopP)
		for (int32_t b : kPopB)
			for (Sp sp : kPopSp) {
				const Original o = MakeOriginal(e, P, b, sp);
				if (!o.ok) continue;
				const std::string name = CaseName(P, b, sp);
				// The unsaved original's own reservation after reset and re-adopt: R(budget), graded in a
				// pool of prefix pages + R + slack (the original's side of "equal").
				ProbeExactlyFree(e, [&]() -> std::unique_ptr<ProbeState> {
					auto rig = std::make_unique<Rig>(e, static_cast<uint32_t>(e.geo.PrefixPages(P) + e.geo.R(b) + kSlack));
					SslmGpuPrefixHandle* px = rig->pool ? BudgetPrefix(*rig, PrefixTokens(P)) : nullptr;
					GSeq s = px ? rig->Budgeted(b) : GSeq{};
					if (s.h) {
						Adopt(*rig, s, px, P);
						DriveTo(e, s, P, b, sp);
						ResetSeq(e, s);
						Adopt(*rig, s, px, P);
					}
					return rig;
				}, kSlack, (name + ": the original's reservation").c_str());
				for (How how : {How::kHandle, How::kNone}) {
					++cases;
					const char* hw = how == How::kHandle ? "matching handle" : "no handle";
					std::unique_ptr<Twin> t = MakeTwin(e, P, b, o, how, false, true);
					CHECK_MSG(t->r.status == GS::SSLM_OK, "9.12 %s (%s): restore_into_pool: %s", name.c_str(), hw,
					          StatusName(t->r.status));
					CHECK_MSG(t->before.complete, "9.12 %s (%s): construction: the prefix's pages were not found in the mirror",
					          name.c_str(), hw);
					if (!t->r.seq.h) continue;
					GSeq tw = t->r.seq;
					const Turn turn = NextTurn(e, tw);
					// kills: rev 2's budget := limit (the next turn runs past the original's)
					CHECK_MSG(turn == o.turn, "9.12 %s (%s): next turn %zu tokens (%s), original %zu (%s)", name.c_str(), hw,
					          turn.tokens.size(), StatusName(turn.refusal), o.turn.tokens.size(), StatusName(o.turn.refusal));
					// kills: the ceil-span mutant (shares the partial page and writes the twin's rows into it)
					CHECK_MSG(PrefixPagesUnchanged(e, t->pool, t->before),
					          "9.12 %s (%s): the prefix's pages changed under the twin's writes", name.c_str(), hw);
					t.reset();
					// kills: materialized pages to the reserve, and a reservation other than R(budget)
					ProbeExactlyFree(e, [&]() -> std::unique_ptr<ProbeState> { return MakeTwin(e, P, b, o, how, true, false); },
					                 (how == How::kNone ? e.geo.Materialized(P, b) : 0) + kSlack,
					                 (name + " (" + hw + "): pages after the twin's reset and re-adopt").c_str());
				}
			}
	CHECK_MSG(cases == 8 * 5 * 3 * 2, "9.12 population: %d cases, want 240", cases);
}

// ---- 9.10's twin -----------------------------------------------------------------------------------

// Restores `blob` with `handle` into `rig`'s pool and checks status, the saved budget kept, privacy
// (a 2-page create refused in a pool sized to exactly handle pages + R + E), and the next 8 tokens.
void ExpectPrivateRestore(const Env& e, Rig& rig, SslmGpuPrefixHandle* handle, const std::vector<uint8_t>& blob,
                          const Fixed& f, const char* what, bool check_tokens = true,
                          const std::vector<uint8_t>* resave_equals = nullptr) {
	uint32_t diag = 0;
	Restored r = RestoreIntoPool(rig, blob, handle, f.at_save, &diag);
	CHECK_MSG(r.status == GS::SSLM_OK && r.seq.h, "9.12 %s: restore_into_pool: %s", what, StatusName(r.status));
	if (!r.seq.h) return;
	std::printf("9.12 %s: *out_shared_pages = %u (diagnostic)\n", what, diag);
	const std::vector<uint8_t> re = Save(e, r.seq.h);
	const Slm6View v = Slm6Parse(re, e.geo);
	CHECK_MSG(v.ok && v.budget == kCohortBudget && v.origin == kPersonaLen && v.kv_mode == 1, "9.12 %s: saved budget not kept", what);
	if (resave_equals) CHECK_MSG(re == *resave_equals, "9.12 %s: the re-save differs from the restored blob", what);
	// kills: a restore that shares under a handle failing (a)-(f) (it would leave E pages free)
	CHECK_MSG(!TwoPageCreateAdmitted(e, rig), "9.12 %s: a 2-page create was admitted after the restore (it shared)", what);
	if (check_tokens) {
		GSeq t = r.seq;
		// kills: a share of a one-byte-different prefix, seen in the next tokens
		CHECK_MSG(DecodeN(e, t, 8) == f.next8, "9.12 %s: tokens after restore differ from the original's", what);
	}
}

// Which of the blob's prefix pages [0, P) the pool holds: the indices of the pages not found.
std::vector<size_t> MissingPrefixPages(const Env& e, Rig& rig, uint32_t pages, const std::vector<uint8_t>& blob, int64_t P) {
	const PrefixSnap s = LocatePrefixPages(e, rig.pool, pages, ExpectedPrefixPages(e, blob, P));
	std::vector<size_t> missing;
	for (size_t i = 0; i < s.index.size(); ++i)
		if (s.index[i] < 0) missing.push_back(i);
	return missing;
}

void Twin910(const Env& e) {
	const Fixed& f = FixedCase(e);
	if (!f.ok) return;
	const Geo& g = e.geo;
	const int64_t R = g.R(kCohortBudget), E = g.Materialized(kPersonaLen, kCohortBudget);
	const int64_t span = g.SharedPages(kPersonaLen);  // 75 full pages, the shared span

	// (e) the longer handle: the persona prompt plus one token, frozen at 1,201; span bytes equal.
	{
		std::vector<int32_t> longer = PersonaTokens(0);
		longer.push_back(5);
		const uint32_t pages = static_cast<uint32_t>(g.PersonaChainPages(201, true) + R + E);
		Rig rig(e, pages);
		const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true, &longer) : GChain{};
		if (c.ok) {
			const std::vector<size_t> miss = MissingPrefixPages(e, rig, pages, f.blob, span * g.B);
			CHECK_MSG(miss.empty(), "9.12 (e) longer: construction: the span bytes must be equal, so (e) alone rejects");
			ExpectPrivateRestore(e, rig, c.persona, f.blob, f, "(e) longer handle");
		}
	}
	// (e) the shorter handle, the world at 1,000 (rewritten: no GPU table sentinel).
	{
		Rig rig(e, static_cast<uint32_t>(g.PrefixPages(1000) + R + E));
		SslmGpuPrefixHandle* world = rig.pool ? BudgetPrefix(rig, WorldTokens()) : nullptr;
		if (world) ExpectPrivateRestore(e, rig, world, f.blob, f, "(e) shorter handle");
	}
	// (f) the edited blob: one byte of layer L-1's V half at position 500. Private; the re-save is the
	// edited SLM6 byte for byte.
	{
		std::vector<uint8_t> edited = f.blob;
		const Slm6View v = Slm6Parse(edited, g);
		const int64_t pos = 500;
		if (v.ok) edited[Slm6RowOff(v, g, g.L - 1, 1, 0, pos) + 3] ^= 0x01;
		const uint32_t pages = static_cast<uint32_t>(g.PersonaChainPages(200, true) + R + E);
		Rig rig(e, pages);
		const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true) : GChain{};
		if (c.ok && v.ok) {
			const std::vector<size_t> miss = MissingPrefixPages(e, rig, pages, edited, span * g.B);
			CHECK_MSG(miss.size() == 1 && miss[0] == static_cast<size_t>(pos / g.B),
			          "9.12 (f) edited: construction: the span must differ in page %lld only", static_cast<long long>(pos / g.B));
			// kills: the K-only and layer-0-only compare (they share, and the re-save carries the prefix's byte)
			ExpectPrivateRestore(e, rig, c.persona, edited, f, "(f) edited blob", false, &edited);
		}
	}
	// (f) the real producer: the persona's last full-page token (position 1,199) differs.
	{
		std::vector<int32_t> other = PersonaTokens(0);
		other.back() = (other.back() + 1) % kVocab;
		const uint32_t pages = static_cast<uint32_t>(g.PersonaChainPages(200, true) + R + E);
		Rig rig(e, pages);
		const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true, &other) : GChain{};
		if (c.ok) {
			const std::vector<size_t> miss = MissingPrefixPages(e, rig, pages, f.blob, span * g.B);
			CHECK_MSG(miss.size() == 1 && miss[0] == static_cast<size_t>((kPersonaLen - 1) / g.B),
			          "9.12 (f) producer: construction: the span must differ in its last full page only");
			// kills: a compare that skips the last full page
			ExpectPrivateRestore(e, rig, c.persona, f.blob, f, "(f) real producer");
		}
	}
	// (b) the unfrozen handle with equal span bytes: the persona left unfrozen holds R(200).
	{
		const uint32_t pages = static_cast<uint32_t>(g.SharedPages(1000) + g.R(200) + R + E);
		Rig rig(e, pages);
		const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true, nullptr, false) : GChain{};
		if (c.ok) {
			const std::vector<size_t> miss = MissingPrefixPages(e, rig, pages, f.blob, span * g.B);
			CHECK_MSG(miss.empty(), "9.12 (b): construction: the unfrozen handle's span bytes must be equal");
			// kills: (b) not checked
			ExpectPrivateRestore(e, rig, c.persona, f.blob, f, "(b) unfrozen handle");
		}
	}
	// (d) rewritten: a handle from another pool of the same context (A13): private, SSLM_OK, and the
	// foreign pool untouched (its free pages as before the call).
	{
		struct Two : ProbeState {
			std::unique_ptr<Rig> other, dest;
			SslmGpuKvPoolHandle* Pool() override { return other->pool; }
			SslmGpuContext* Ctx() override { return other->ctx; }
			SslmGpuModelHandle* Model() override { return other->model; }
			void DrainAll() override {
				other->DrainAll();
				dest->DrainAll();
			}
		};
		auto build = [&](bool check) -> std::unique_ptr<Two> {
			auto t = std::make_unique<Two>();
			t->other = std::make_unique<Rig>(e, 200);
			t->dest = std::make_unique<Rig>(e, static_cast<uint32_t>(R + E));
			const GChain c = t->other->pool ? BuildPersonaChain(*t->other, 0, true) : GChain{};
			if (!c.ok || !t->dest->pool) return t;
			if (check) {
				// kills: (d) not checked (a cross-pool share would leave E pages free in dest)
				ExpectPrivateRestore(e, *t->dest, c.persona, f.blob, f, "(d) handle of another pool");
			} else {
				Restored r = RestoreIntoPool(*t->dest, f.blob, c.persona, f.at_save);
				(void)r;
			}
			return t;
		};
		build(true);
		// kills: a foreign-pool share taking refcounts or pages in the handle's pool
		ProbeExactlyFree(e, [&]() -> std::unique_ptr<ProbeState> { return build(false); }, 200 - g.PersonaChainPages(200, true),
		                 "9.12 (d): the foreign handle's pool");
	}
	// (c): implied by (d); a pool is bound to one model, so no handle can differ in model while sharing
	// the pool. Stated, not tested (§7 9.10, inherited by 9.12).
	// (a): refused by A14, asserted in g1e_cell_9_13.cpp (A14 with the matching prefix handle).
}

// ---- 9.11's twin -----------------------------------------------------------------------------------

void Twin911(const Env& e) {
	const Fixed& f = FixedCase(e);
	if (!f.ok) return;
	const Geo& g = e.geo;
	const int64_t chain = g.PersonaChainPages(200, true), R = g.R(kCohortBudget), E = g.Materialized(kPersonaLen, kCohortBudget);
	const int64_t pool_pages = chain + R + E;
	CHECK(pool_pages == 183);
	for (int64_t pages : {pool_pages, pool_pages - 1}) {
		Rig rig(e, static_cast<uint32_t>(pages));
		const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true) : GChain{};
		if (!c.ok) return;
		Restored r = RestoreIntoPool(rig, f.blob, nullptr, f.at_save);
		// kills: a private restore drawing R only (admitted at 182), or more than R + E (refused at 183)
		CHECK_MSG(r.status == (pages == pool_pages ? GS::SSLM_OK : GS::SSLM_GPU_KV_POOL_EXHAUSTED),
		          "9.11 twin: a pool of %lld: %s", static_cast<long long>(pages), StatusName(r.status));
	}
	enum class First { kReset, kAdopt, kRelease };
	for (First first : {First::kReset, First::kAdopt, First::kRelease}) {
		const char* what = first == First::kReset ? "9.11 twin: first reset" : first == First::kAdopt ? "9.11 twin: first adopt"
		                                                                                               : "9.11 twin: first release";
		auto build = [&]() -> std::unique_ptr<ProbeState> {
			auto rig = std::make_unique<Rig>(e, static_cast<uint32_t>(pool_pages));
			const GChain c = rig->pool ? BuildPersonaChain(*rig, 0, true) : GChain{};
			if (!c.ok) return rig;
			Restored r = RestoreIntoPool(*rig, f.blob, nullptr, f.at_save);
			if (r.seq.h) {
				if (first == First::kReset) ResetSeq(e, r.seq);
				if (first == First::kAdopt) Adopt(*rig, r.seq, c.persona, kPersonaLen);
				if (first == First::kRelease) rig->ReleaseSeq(r.seq.h);
			}
			return rig;
		};
		// kills: materialized pages to the reserve (reset, adopt: 75 against 0); at release that mutant is
		// equivalent (108 against 108), so release names its own: "release returns them to nothing" (33)
		ProbeExactlyFree(e, build, first == First::kRelease ? R + E : E, what);
	}
	// Two restores with a null prefix give byte-identical results; a null out_shared_pages with the
	// matching handle is accepted.
	{
		Rig rig(e, 600);
		const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true) : GChain{};
		if (!c.ok) return;
		Restored a = RestoreIntoPool(rig, f.blob, nullptr, f.at_save);
		Restored b = RestoreIntoPool(rig, f.blob, nullptr, f.at_save);
		Restored m = RestoreIntoPool(rig, f.blob, c.persona, f.at_save, nullptr);
		CHECK_MSG(a.status == GS::SSLM_OK && b.status == GS::SSLM_OK && m.status == GS::SSLM_OK,
		          "9.11 twin: restores %s, %s, %s (the last with a null out_shared_pages)", StatusName(a.status),
		          StatusName(b.status), StatusName(m.status));
		if (a.seq.h && b.seq.h && m.seq.h) {
			CHECK_MSG(Save(e, a.seq.h) == f.blob && Save(e, b.seq.h) == f.blob, "9.11 twin: a null-prefix restore's re-save differs");
			GSeq ta = a.seq, tb = b.seq, tm = m.seq;
			const Turn x = NextTurn(e, ta), y = NextTurn(e, tb), z = NextTurn(e, tm);
			CHECK_MSG(x == f.cont && y == f.cont && z == f.cont, "9.11 twin: the restored holders' tokens differ");
		}
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			Fixed99(e);
			Twin910(e);
			Twin911(e);
			Population99(e);
		}
	}
	return FinishSuite("g1e_cell_9_12");
}
