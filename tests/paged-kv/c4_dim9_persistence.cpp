// Paged-KV plan (rev 16.1) §7 dimension 9, the legacy holders' persistence cells (step C4): 9.1's
// legacy part, 9.2, 9.3, 9.5, 9.7 and 9.14's A2, A3 and A7 rows.
//
// From C4 a legacy (whole_reserve) holder saves 1.9.0's SSB5, gathered from its pages, byte-equal
// to the blob v1.11.0 writes for the same state (§3.7 Writers); readers accept SSB6, 1.9.0's
// SSB5 and SSB4/3/2; SSB5/4/3/2 restore as whole_reserve with origin 0 (§3.7, §3.9 A3). Verdicts
// read the R0 reference and pins, unsaved twins, blob bytes and the legacy-create admission count
// (§8) at pool sizes computed here; never the stats verbs. A legacy holder's origin is visible
// only to the diagnostic sslm_seq_kv_stats (C5, §3.7 "Only the diagnostic origin ... sees this"),
// so no C4 verdict here can, or does, read it.

#include "pkv_legacy_b_helpers.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace pkv;
using namespace pkv::legacy_b;

// ---- 9.1 [C4 legacy] ---------------------------------------------------------------------------
//
// Save -> restore -> save is byte-identical for legacy holders (1.9.0's SSB5), resting and
// mid-token, adopted: fresh (L = 0); prefill 100 + decode 4 (resting); prefill 100 then one layer
// (mid-token); copy-adopted 1,000 (resting ready); copy-adopted 1,008 + 3 tokens, then mid-token;
// reset after use; and a damped-greedy run whose history is in the blob. Each restore lands in a
// pool pre-filled with 0x5A, so a restore or save that touches bytes outside the valid rows shows
// in the second save; the round trip is taken twice, and the restored holder continues as the
// original does.
// Mutants killed: a restore that drops or re-derives a saved field (ready flag, per-site counts,
// history, mid-token residual), and a save that gathers past L' from the pages (the 0x5A bytes).
// The nested and budget halves are C5's.

using Builder = sslm_seq (*)(const Fixture&, sslm_kv_pool*);

sslm_seq NewSeq(const Fixture& fx, sslm_kv_pool* pool) {
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, pool, &s), SSLM_OK);
	return s;
}

sslm_seq AdoptedSeq(const Fixture& fx, sslm_kv_pool* pool, int32_t len) {
	sslm_prefix px = nullptr;
	PKV_CHECK_EQ(sslm_prefix_begin(fx.model, pool, &px), SSLM_OK);
	if (!px) return nullptr;
	if (len > 0) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, len, fx.vocab), 64), SSLM_OK);
	PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
	sslm_seq s = NewSeq(fx, pool);
	if (s) PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
	sslm_prefix_release(px);  // a copy adopter keeps nothing of the prefix (§3.5)
	return s;
}

const struct State91 {
	const char* name;
	Builder build;
} kStates91[] = {
    {"fresh", [](const Fixture& fx, sslm_kv_pool* p) { return NewSeq(fx, p); }},
    {"prefill100+decode4 resting",
     [](const Fixture& fx, sslm_kv_pool* p) {
	     sslm_seq s = NewSeq(fx, p);
	     if (s) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(31, 100, fx.vocab), 64), SSLM_OK);
	     if (s) NextTokens(fx.model, s, 4);
	     return s;
     }},
    {"prefill100 mid-token",
     [](const Fixture& fx, sslm_kv_pool* p) {
	     sslm_seq s = NewSeq(fx, p);
	     if (s) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(31, 100, fx.vocab), 64), SSLM_OK);
	     if (s) EnterMidToken(fx.model, s);
	     return s;
     }},
    {"adopted 1000 resting", [](const Fixture& fx, sslm_kv_pool* p) { return AdoptedSeq(fx, p, 1000); }},
    {"adopted 1008 + 3 tokens, mid-token",
     [](const Fixture& fx, sslm_kv_pool* p) {
	     sslm_seq s = AdoptedSeq(fx, p, 1008);
	     if (s) NextTokens(fx.model, s, 3);
	     if (s) EnterMidToken(fx.model, s);
	     return s;
     }},
    {"reset after use",
     [](const Fixture& fx, sslm_kv_pool* p) {
	     sslm_seq s = NewSeq(fx, p);
	     if (s) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(31, 300, fx.vocab), 64), SSLM_OK);
	     if (s) PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
	     return s;
     }},
    {"damped history, mid-token",
     [](const Fixture& fx, sslm_kv_pool* p) {
	     sslm_seq s = NewSeq(fx, p);
	     if (!s) return s;
	     PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(32, 30, fx.vocab), 64), SSLM_OK);
	     sslm_decode_params dp{};
	     PKV_CHECK_EQ(sslm_decode_params_init(fx.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &dp), SSLM_OK);
	     sslm_seq b[1] = {s};
	     int emitted = 0;
	     for (int guard = 0; guard < 4096 && emitted < 20; ++guard) {
		     int32_t tok = -1;
		     if (sslm_decode_step_v2(fx.model, b, 1, &dp, nullptr, &tok) != SSLM_OK) break;
		     if (tok >= 0) ++emitted;
	     }
	     int32_t tok = -1;
	     PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &dp, nullptr, &tok), SSLM_OK);  // one layer in
	     PKV_CHECK(tok < 0);
	     return s;
     }},
};

void Cell91LegacyRoundTrip() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (const State91& st : kStates91) {
		LegacyPool pool(fx.model, 2);
		sslm_seq orig = st.build(fx, &pool.pool);
		if (!orig) continue;
		std::vector<uint8_t> s1, s2, s3;
		PKV_CHECK(SaveBlob(orig, &s1));
		PKV_CHECK_MSG(IsMagic(s1, "SSB5"), "9.1 %s: a legacy holder saves 1.9.0's SSB5", st.name);
		LegacyPool dirty(fx.model, 1, 0x5A);
		sslm_seq r = nullptr;
		PKV_CHECK_MSG(Restore(fx, &dirty.pool, s1, &r) == SSLM_OK, "9.1 %s: restore", st.name);
		if (!r) {
			sslm_seq_release(orig);
			continue;
		}
		PKV_CHECK(SaveBlob(r, &s2));
		// kills: a restore that drops a field, or a save that gathers past L' from the pages
		PKV_CHECK_MSG(s2 == s1, "9.1 %s: save -> restore -> save is not byte-identical", st.name);
		LegacyPool dirty2(fx.model, 1, 0x5A);
		sslm_seq r2 = nullptr;
		PKV_CHECK_MSG(Restore(fx, &dirty2.pool, s2, &r2) == SSLM_OK, "9.1 %s: second restore", st.name);
		if (r2) {
			PKV_CHECK(SaveBlob(r2, &s3));
			PKV_CHECK_MSG(s3 == s1, "9.1 %s: the second round trip is not byte-identical", st.name);
			sslm_seq_release(r2);
		}
		if (std::strcmp(st.name, "fresh") != 0 && std::strcmp(st.name, "reset after use") != 0)
			PKV_CHECK_MSG(NextTokens(fx.model, r, 4) == NextTokens(fx.model, orig, 4), "9.1 %s: continuation", st.name);
		sslm_seq_release(r);
		sslm_seq_release(orig);
	}
}

// ---- 9.2 [C4] ----------------------------------------------------------------------------------
//
// R0's pinned blobs -- v1.8.1's SSB4 (the last SSB4 writer, standing in for the plan's v1.8.0)
// and v1.11.0's SSB5 (the plan's v1.9.0) -- restore as whole_reserve and continue byte-equal to
// their writer's continuation, which is also v1.11.0's. The pins and records: persist/saved and
// saturating/saturated (each with continuation8), lifecycle/prefill100+decode4 (continue8) and
// lifecycle/adopt300 (adopt300+decode8).
//   whole_reserve: the restored holder holds ceil(cap/B) = 256 pages, graded by the legacy-create
//   admission count: 0 creates in a 1-block pool after the restore, 1 in a 2-block pool;
//   origin 0: unobservable at C4 (see the file header); the restored holder's limit is the cap
//   either way, which 9.14 (A3) grades by its next-turn token count;
//   per-site counts verbatim: the re-save of an SSB5 pin equals the pin byte for byte, and the
//   saturating pin's per-site counts are nonzero, so that equality is not vacuous; an SSB4 pin
//   re-saves as SSB5 with every SSB4 field verbatim, per-site counts 0, and the same residual,
//   history and block.
// Mutants killed: a restore that maps an old blob to a budget holder (fewer pages: the 1-block
// pool admits a create), that zeroes or recomputes per-site counts, or that changes any token.

struct PinCase {
	const char* scenario;
	const char* stage;
	const char* cont;
};
const PinCase kPins[] = {{"persist", "saved", "continuation8"},
                         {"saturating", "saturated", "continuation8"},
                         {"lifecycle", "prefill100+decode4", "continue8"},
                         {"lifecycle", "adopt300", "adopt300+decode8"}};

void Cell92PinnedBlobs() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const RefFile& ref11 = Reference("v1.11.0", fx);
	for (const char* tag : {"v1.8.1", "v1.11.0"}) {
		const bool ssb4 = std::strcmp(tag, "v1.8.1") == 0;
		const RefFile& writer = Reference(tag, fx);
		for (const PinCase& pc : kPins) {
			const std::vector<uint8_t> pin = Pin(tag, pc.scenario, pc.stage);
			const RefRecord* saved = RefLookup(writer, pc.scenario, pc.stage);
			const RefRecord* cont = RefLookup(writer, pc.scenario, pc.cont);
			const RefRecord* cont11 = RefLookup(ref11, pc.scenario, pc.cont);
			if (!saved || !cont || !cont11 || pin.empty()) continue;
			PKV_CHECK_MSG(IsMagic(pin, ssb4 ? "SSB4" : "SSB5"), "9.2 %s %s: pin magic", tag, pc.scenario);
			PKV_CHECK_MSG(BlobSha(pin) == saved->blob_sha, "9.2 %s %s/%s: pin digest", tag, pc.scenario, pc.stage);
			PKV_CHECK_MSG(cont->tokens == cont11->tokens, "9.2 %s %s: the writers' continuations disagree", tag, pc.scenario);
			// whole_reserve, by admission.
			for (uint32_t blocks : {1u, 2u}) {
				LegacyPool pool(fx.model, blocks);
				sslm_seq r = nullptr;
				PKV_CHECK_MSG(Restore(fx, &pool.pool, pin, &r) == SSLM_OK, "9.2 %s %s/%s: restore into %u block(s)", tag,
				              pc.scenario, pc.stage, blocks);
				// kills: an old blob restored as a budget holder holding fewer than 256 pages
				PKV_CHECK_MSG(CountLegacyCreates(fx.model, &pool.pool) == static_cast<int>(blocks) - 1,
				              "9.2 %s %s/%s: the restored holder does not hold exactly one block's pages", tag, pc.scenario,
				              pc.stage);
				if (r) sslm_seq_release(r);
			}
			LegacyPool pool(fx.model, 1);
			sslm_seq r = nullptr;
			PKV_CHECK_EQ(Restore(fx, &pool.pool, pin, &r), SSLM_OK);
			if (!r) continue;
			std::vector<uint8_t> resaved;
			PKV_CHECK(SaveBlob(r, &resaved));
			PKV_CHECK_MSG(IsMagic(resaved, "SSB5"), "9.2 %s %s: re-save magic", tag, pc.scenario);
			if (!ssb4) {
				// kills: per-site counts (or any field) not restored verbatim
				PKV_CHECK_MSG(resaved == pin, "9.2 %s %s/%s: the re-save is not the pin", tag, pc.scenario, pc.stage);
			} else {
				const size_t ssb4_header = 124, tail = pin.size() - ssb4_header;
				bool same = resaved.size() == pin.size() + (kSsb5Header - ssb4_header) &&
				            std::memcmp(resaved.data() + 4, pin.data() + 4, ssb4_header - 4) == 0 &&
				            std::memcmp(resaved.data() + kSsb5Header, pin.data() + ssb4_header, tail) == 0;
				for (size_t at = 124; at < kSsb5Header && same; at += 8) same = Le64(resaved, at) == 0;
				PKV_CHECK_MSG(same, "9.2 %s %s/%s: the SSB4 fields, the zero per-site counts or the tail differ", tag,
				              pc.scenario, pc.stage);
			}
			// kills: any change to the restored state's continuation
			PKV_CHECK_MSG(NextTokens(fx.model, r, static_cast<int>(cont->tokens.size())) == cont->tokens,
			              "9.2 %s %s/%s: continuation differs from the writer's", tag, pc.scenario, pc.stage);
			sslm_seq_release(r);
		}
	}
	// The per-site equality above is not vacuous: v1.11.0's saturating pin carries nonzero counts.
	const std::vector<uint8_t> sat = Pin("v1.11.0", "saturating", "saturated");
	PKV_CHECK_MSG(Le64(sat, 124) + Le64(sat, 132) + Le64(sat, 140) + Le64(sat, 148) > 0,
	              "9.2: the saturating SSB5 pin's per-site counts are all zero");
}

// ---- 9.3 [C4] ----------------------------------------------------------------------------------
//
// Old reader, new data (rev 16): v1.11.0 (the plan's v1.9.0) restores a legacy holder's blob
// saved by the paged build and continues byte-equal. In the cloud the paged build's blob is
// compared byte for byte with the reference blob v1.11.0 wrote for the same state, since equal
// bytes are exactly what v1.11.0's reader reads, and v1.11.0's own restore of those bytes is R0's
// record (persist/restored, continuation8). Graded on the pinned states (byte compare with the pin)
// and on every recorded stage of lifecycle, persist and saturating (blob digest). The box leg runs
// both binaries: with SUPERSLM_PKV_93_OUT set, the paged build's blobs are written there
// (c4_93_<scenario>_<stage>.ssb5) for the v1.11.0 binary to restore and continue against the same
// records. The SSB6-rejection side needs a budget holder's blob, so it is C5's (carrier row
// RB16-CELLS).
// Mutant killed: any byte of a legacy holder's blob that differs from 1.9.0's (a field order, the
// whole-block section, the zero substitution past L').

void Cell93OldReaderNewData() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (const char* sc : {"lifecycle", "persist", "saturating"})
		ExpectMatchesReference("v1.11.0", fx, sc, RunScenario(fx, sc), Compare::kTokensRowsAndBlob);
	const char* out_dir = std::getenv("SUPERSLM_PKV_93_OUT");
	// The pinned states, rebuilt live and compared with v1.11.0's bytes directly.
	Driver d;
	d.model = fx.model;
	d.geo = fx.geo;
	size_t n = 0;
	const Scenario* scs = Scenarios(&n);
	for (const char* sc : {"lifecycle", "persist", "saturating"}) {
		std::vector<Pinned> pins;
		Driver dd = d;
		for (size_t i = 0; i < n; ++i)
			if (std::strcmp(scs[i].name, sc) == 0) scs[i].run(dd, &pins);
		PKV_CHECK_MSG(dd.fail.empty(), "9.3 %s: %s", sc, dd.fail.c_str());
		for (const Pinned& p : pins) {
			const std::vector<uint8_t> ref = Pin("v1.11.0", sc, p.stage.c_str());
			// kills: a legacy blob that is not byte-equal to 1.9.0's for the same state
			PKV_CHECK_MSG(p.blob == ref, "9.3 %s/%s: the paged build's SSB5 differs from v1.11.0's bytes", sc,
			              p.stage.c_str());
			if (out_dir && *out_dir) {
				const std::string path = std::string(out_dir) + "/c4_93_" + sc + "_" + p.stage + ".ssb5";
				if (FILE* f = std::fopen(path.c_str(), "wb")) {
					std::fwrite(p.blob.data(), 1, p.blob.size(), f);
					std::fclose(f);
				} else {
					PKV_CHECK_MSG(false, "9.3: cannot write %s", path.c_str());
				}
			}
		}
	}
}

// ---- 9.5 [C4] ----------------------------------------------------------------------------------
//
// sslm_seq_state_size stays an upper bound at the worst case a legacy holder reaches. Damped
// greedy from an empty origin: one prompt token, then whole tokens until context_length = cap - 1
// (history cap - 1); one layer of the next token, so the save is mid-token with L' = cap and
// history cap - 1; then the token is finished, which writes row cap - 1 and emits, and the holder
// rests at context_length = cap with history cap. Both saves are asserted <= sslm_seq_state_size,
// and the reached state is read off the blob header. At C4 the blob is 1.9.0's SSB5, 16 bytes
// shorter in its header than SSB6, so the bound's margin here is that difference plus whatever
// the size verb adds at this step; the separating worst case (margin 0, +48 to +51 failing) runs
// on a budget holder with budget = cap saving SSB6 (carrier row RB16-CELLS), at C5. Both caps:
// pkv_def (4,096) and pkv_32k (32,768; about a minute and a half on the cloud container).
// Mutant killed: a state-size verb that under-counts the legacy blob (the whole-block section, the
// history or the residual).

void Cell95StateSizeWorstCase() {
	for (const char* stem : {"pkv_def", "pkv_32k"}) {
		const Fixture& fx = GetFixture(stem);
		if (!fx.ok) continue;
		const int64_t cap = fx.geo.context_cap;
		const size_t bound = sslm_seq_state_size(fx.model);
		LegacyPool pool(fx.model, 1);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (!s) continue;
		const int32_t seed = 5;
		int32_t consumed = 0;
		PKV_CHECK_EQ(sslm_prefill(fx.model, s, &seed, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed), SSLM_OK);
		sslm_decode_params p{};
		PKV_CHECK_EQ(sslm_decode_params_init(fx.model, SSLM_DECODE_MODE_DAMPED_GREEDY, static_cast<int32_t>(fx.geo.layers), &p),
		             SSLM_OK);
		sslm_seq b[1] = {s};
		int64_t emitted = 0;
		sslm_status st = SSLM_OK;
		while (emitted < cap - 1) {  // the ready token, then whole tokens: context_length = emitted
			int32_t tok = -1;
			st = sslm_decode_step_v2(fx.model, b, 1, &p, nullptr, &tok);
			if (st != SSLM_OK) break;
			if (tok >= 0) ++emitted;
		}
		PKV_CHECK_MSG(st == SSLM_OK, "9.5 %s: damped decode refused at %lld tokens (%d)", stem, static_cast<long long>(emitted),
		              static_cast<int>(st));
		p.layer_budget = 1;
		int32_t tok = -1;
		PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &p, nullptr, &tok), SSLM_OK);
		PKV_CHECK(tok < 0);
		std::vector<uint8_t> mid;
		PKV_CHECK(SaveBlob(s, &mid));
		PKV_CHECK_EQ(Le64(mid, 60), cap - 1);
		PKV_CHECK_EQ(BlobLPrime(mid), cap);  // L' = cap
		PKV_CHECK_EQ(Le64(mid, 112), cap - 1);  // history cap - 1
		// kills: a state-size verb that under-counts the mid-token worst case
		PKV_CHECK_MSG(mid.size() <= bound, "9.5 %s mid-token: blob %zu > sslm_seq_state_size %zu", stem, mid.size(), bound);
		PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &p, nullptr, &tok), SSLM_OK);  // finishes the token
		PKV_CHECK(tok >= 0);
		std::vector<uint8_t> rest;
		PKV_CHECK(SaveBlob(s, &rest));
		PKV_CHECK_EQ(Le64(rest, 60), cap);
		PKV_CHECK_EQ(Le32(rest, 68), 0);
		PKV_CHECK_EQ(Le64(rest, 112), cap);  // history cap
		// kills: a state-size verb that under-counts the resting-at-cap worst case
		PKV_CHECK_MSG(rest.size() <= bound, "9.5 %s at cap: blob %zu > sslm_seq_state_size %zu", stem, rest.size(), bound);
		PKV_CHECK_MSG(IsMagic(rest, "SSB5") && rest.size() == Ssb5Size(fx, static_cast<uint64_t>(cap)),
		              "9.5 %s at cap: not 1.9.0's SSB5 size", stem);
		sslm_seq_release(s);
	}
}

// ---- 9.7 [C4, after C0] --------------------------------------------------------------------------
//
// Must-accept through real producers (§3.7: no relation between the per-site counts and the
// aggregate is validated).
//   (i) A prefill that forces a landing saturation -- the saturating scenario's prompt, tokens
//       0..11 in one chunk, whose prefill alone saturates on pkv_def (the cloud stand-in for the
//       box-only census fixture of G31; asserted below, aggregate and kv_landing both nonzero) --
//       then save, restore, save, restore. Both restores return SSLM_OK; the two saves are equal;
//       each restored holder continues byte-equal to v1.11.0 (saturating/saturated's 4 tokens,
//       then continuation8).
//  (ii) R0's pinned v1.8.1 SSB4 blob with a nonzero aggregate (saturating/saturated, 333) is
//       restored, saved as SSB5 by the legacy writer (aggregate 333, per-site counts 0), and that
//       SSB5 restored: SSLM_OK, and it continues byte-equal to v1.11.0's continuation8.
// Mutant killed: "restore re-adds the per-site-sum rejection" -- (ii)'s SSB5 carries an aggregate
// of 333 over per-site counts of 0, so the mutant refuses it. (i) is must-accept only: with C0's
// census fix in the base, a prefill's per-site counts sum to its aggregate, so (i) does not
// separate that mutant; it guards the producer path (the plan's census fixture, on the box, is
// the one whose prefill total is 5).

void Cell97MustAccept() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const RefFile& ref = Reference("v1.11.0", fx);
	const RefRecord* sat = RefLookup(ref, "saturating", "saturated");
	const RefRecord* cont = RefLookup(ref, "saturating", "continuation8");
	if (!sat || !cont) return;
	std::vector<int32_t> prompt;
	for (int32_t t = 0; t < fx.vocab && prompt.size() < 12; ++t) prompt.push_back(t);
	{  // (i)
		LegacyPool pool(fx.model, 3);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (s) {
			int32_t consumed = 0;
			PKV_CHECK_EQ(sslm_prefill(fx.model, s, prompt.data(), 12, 12, SSLM_SPAN_PROMPT, nullptr, &consumed), SSLM_OK);
			std::vector<uint8_t> s1, s2;
			PKV_CHECK(SaveBlob(s, &s1));
			PKV_CHECK_MSG(Le64(s1, 92) > 0 && Le64(s1, 124) > 0,
			              "9.7 (i): the prefill did not force a landing saturation (aggregate %llu, kv_landing %llu)",
			              static_cast<unsigned long long>(Le64(s1, 92)), static_cast<unsigned long long>(Le64(s1, 124)));
			sslm_seq r1 = nullptr, r2 = nullptr;
			PKV_CHECK_EQ(Restore(fx, &pool.pool, s1, &r1), SSLM_OK);
			if (r1) PKV_CHECK(SaveBlob(r1, &s2));
			PKV_CHECK_MSG(s2 == s1, "9.7 (i): save -> restore -> save differs");
			PKV_CHECK_EQ(Restore(fx, &pool.pool, s2, &r2), SSLM_OK);
			for (sslm_seq r : {s, r1, r2}) {
				if (!r) continue;
				PKV_CHECK_MSG(NextTokens(fx.model, r, 4) == sat->tokens, "9.7 (i): the 4 tokens after the prefill differ");
				PKV_CHECK_MSG(NextTokens(fx.model, r, 8) == cont->tokens, "9.7 (i): continuation8 differs");
			}
			if (r2) sslm_seq_release(r2);
			if (r1) sslm_seq_release(r1);
			sslm_seq_release(s);
		}
	}
	{  // (ii)
		const std::vector<uint8_t> ssb4 = Pin("v1.8.1", "saturating", "saturated");
		PKV_CHECK_MSG(IsMagic(ssb4, "SSB4") && Le64(ssb4, 92) == 333, "9.7 (ii): the SSB4 pin's aggregate is %llu",
		              static_cast<unsigned long long>(Le64(ssb4, 92)));
		LegacyPool pool(fx.model, 2);
		sslm_seq r1 = nullptr, r2 = nullptr;
		PKV_CHECK_EQ(Restore(fx, &pool.pool, ssb4, &r1), SSLM_OK);
		std::vector<uint8_t> ssb5, again;
		if (r1) PKV_CHECK(SaveBlob(r1, &ssb5));
		PKV_CHECK_MSG(IsMagic(ssb5, "SSB5") && Le64(ssb5, 92) == 333 &&
		                  Le64(ssb5, 124) + Le64(ssb5, 132) + Le64(ssb5, 140) + Le64(ssb5, 148) == 0,
		              "9.7 (ii): the SSB5 re-save is not aggregate 333 over per-site counts of 0");
		// kills: restore re-adds the per-site-sum rejection
		PKV_CHECK_EQ(Restore(fx, &pool.pool, ssb5, &r2), SSLM_OK);
		if (r2) {
			PKV_CHECK(SaveBlob(r2, &again));
			PKV_CHECK_MSG(again == ssb5, "9.7 (ii): the SSB5 round trip differs");
			PKV_CHECK_MSG(NextTokens(fx.model, r2, 8) == cont->tokens, "9.7 (ii): continuation8 after the SSB5 restore");
			sslm_seq_release(r2);
		}
		if (r1) {
			PKV_CHECK_MSG(NextTokens(fx.model, r1, 8) == cont->tokens, "9.7 (ii): continuation8 after the SSB4 restore");
			sslm_seq_release(r1);
		}
	}
}

// ---- 9.14 [C4: the A2, A3 and A7 rows] ------------------------------------------------------------
//
// §3.9's "keeps" rows a legacy holder reaches, over the legacy population: prefix lengths 4.2's
// {0, 1, 15, 16, 17, 1,000, 1,008}, each copy-adopted by a fresh holder, saved at the adopt
// (resting ready), after 3 more tokens (resting), and one layer into the next token (mid-token);
// length 0 has only the first (its adopter has no token to embed).
//   A3: the holder's own 1.9.0 SSB5, restored (origin 0 by design; unobservable at C4);
//   A2: a hand-built SSB6 whole_reserve blob of the same state with the saved origin = the prefix
//       length (rev 16: no library writer emits one);
//   A7: the adopt itself, onto a holder that has already prefilled 50 tokens.
// Each asserts, against the unsaved original (the adopter built fresh in its own pool):
//   - mode whole_reserve and budget cap, straight after the verb and again after a reset, by the
//     legacy-create admission count at two pool sizes (§8, rev 6): P = 3 * ceil(cap/B) = 768
//     pages with the holder live admits 2 creates, and P - 1 = 767 (pkv_legacy_b_helpers.h's
//     SizedLegacyPool) admits 1;
//   - limit cap and the next-turn token count: the holder emits exactly what the original emits
//     until SSLM_CONTEXT_CAP_EXCEEDED, token for token, which for a resting holder is
//     (ready ? 1 : 0) + (cap - context_length), §3.4.
// For A7 the prefix is released before counting: a copy adopter maps no prefix page (§3.5).
// Mutants killed: a one-page deficit (holder holds 257: admits 1 at P) and a one-page excess (255:
// admits 2 at P - 1) in the reservation (*executed* in the plan's probe); the restore that gives a
// whole_reserve blob a budget other than cap (a refusal before the cap, or a smaller reservation);
// legacy adopt that shares (A7: the shared pages stay held after the prefix's release, so P admits
// 1); and the check-after-a-state-change mutant (the holder's tokens or reservation change).
// Rev 2's budget := limit and the pre-plan S1 mutant are equivalent on whole_reserve rows (their
// limit is the cap either way); they separate on the budget rows at C5.

struct PopState {
	int32_t len;
	int point;  // 0 at the adopt, 1 after 3 tokens, 2 mid-token
};

sslm_seq BuildPop(const Fixture& fx, sslm_kv_pool* pool, const PopState& ps) {
	sslm_seq s = AdoptedSeq(fx, pool, ps.len);
	if (!s) return s;
	if (ps.point >= 1) NextTokens(fx.model, s, 3);
	if (ps.point == 2) {
		int32_t tok = -1;
		PKV_CHECK_EQ(StepOneLayer(fx.model, s, &tok), SSLM_OK);
		PKV_CHECK(tok < 0);
	}
	return s;
}

// The admission pair, P and P - 1, for a state `make` puts live into the given pool.
template <typename Make>
void ExpectWholeReserveAdmission(const Fixture& fx, const char* row, const char* what, Make make) {
	for (int minus = 0; minus < 2; ++minus) {
		SizedLegacyPool sp(fx, 3, minus != 0);
		sslm_seq h = make(sp.Pool());
		PKV_CHECK_MSG(h != nullptr, "9.14 %s %s: the verb failed in the %s pool", row, what, minus ? "P-1" : "P");
		if (!h) continue;
		const int want = minus ? 1 : 2;
		// kills: a reservation one page short (at P) or one page over (at P - 1)
		PKV_CHECK_MSG(CountLegacyCreates(fx.model, sp.Pool()) == want, "9.14 %s %s: after the verb, %s admits != %d", row,
		              what, minus ? "P-1" : "P", want);
		PKV_CHECK_EQ(sslm_seq_reset(h), SSLM_OK);
		PKV_CHECK_MSG(CountLegacyCreates(fx.model, sp.Pool()) == want, "9.14 %s %s: after a reset, %s admits != %d", row,
		              what, minus ? "P-1" : "P", want);
		sslm_seq_release(h);
	}
}

void Cell914LegacyRows() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t cap = fx.geo.context_cap;
	PKV_CHECK_EQ(3 * fx.CapPages(), 768);
	std::vector<PopState> pop;
	for (int32_t len : {0, 1, 15, 16, 17, 1000, 1008})
		for (int point = 0; point < (len == 0 ? 1 : 3); ++point) pop.push_back({len, point});
	for (const PopState& ps : pop) {
		char what[64];
		std::snprintf(what, sizeof what, "prefix %d, %s", ps.len,
		              ps.point == 0 ? "at the adopt" : ps.point == 1 ? "after 3 tokens" : "mid-token");
		// The unsaved original, its blob, and its next turn.
		LegacyPool home(fx.model, 2);
		sslm_seq orig = BuildPop(fx, &home.pool, ps);
		if (!orig) continue;
		std::vector<uint8_t> ssb5;
		PKV_CHECK(SaveBlob(orig, &ssb5));
		sslm_status orig_refusal = SSLM_OK;
		const std::vector<int32_t> orig_turn = RunToRefusal(fx.model, orig, &orig_refusal, cap + 1);
		sslm_seq_release(orig);
		if (ps.len > 0) {
			const int64_t c = static_cast<int64_t>(Le64(ssb5, 60));
			const bool ready = Le32(ssb5, 120) != 0;
			PKV_CHECK_MSG(orig_refusal == SSLM_CONTEXT_CAP_EXCEEDED && static_cast<int64_t>(orig_turn.size()) == (ready ? 1 : 0) + cap - c,
			              "9.14 %s: the original emits %zu tokens then %d (want %lld then CONTEXT_CAP_EXCEEDED)", what,
			              orig_turn.size(), static_cast<int>(orig_refusal), static_cast<long long>((ready ? 1 : 0) + cap - c));
		}
		const std::vector<uint8_t> ssb6 = Ssb6WholeReserveFromSsb5(ssb5, fx, ps.len);
		struct Row {
			const char* row;
			const std::vector<uint8_t>* blob;
		};
		for (const Row& rw : {Row{"A3", &ssb5}, Row{"A2", &ssb6}}) {
			ExpectWholeReserveAdmission(fx, rw.row, what, [&](sslm_kv_pool* pool) {
				sslm_seq r = nullptr;
				PKV_CHECK_EQ(Restore(fx, pool, *rw.blob, &r), SSLM_OK);
				return r;
			});
			LegacyPool pool(fx.model, 1);
			sslm_seq r = nullptr;
			PKV_CHECK_MSG(Restore(fx, &pool.pool, *rw.blob, &r) == SSLM_OK, "9.14 %s %s: restore", rw.row, what);
			if (!r) continue;
			sslm_status refusal = SSLM_OK;
			// kills: a restored limit other than the cap, or any change to the restored state
			PKV_CHECK_MSG(RunToRefusal(fx.model, r, &refusal, cap + 1) == orig_turn && refusal == orig_refusal,
			              "9.14 %s %s: the next turn differs from the unsaved original's", rw.row, what);
			sslm_seq_release(r);
		}
		if (ps.point != 0) continue;
		// A7: the adopt itself, onto a used holder, in the admission pools and for the next turn.
		auto adopt_used = [&](sslm_kv_pool* pool) -> sslm_seq {
			sslm_prefix px = nullptr;
			PKV_CHECK_EQ(sslm_prefix_begin(fx.model, pool, &px), SSLM_OK);
			if (!px) return nullptr;
			if (ps.len > 0) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, ps.len, fx.vocab), 64), SSLM_OK);
			PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_seq_create(fx.model, pool, &s), SSLM_OK);
			if (s) {
				PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(9, 50, fx.vocab), 64), SSLM_OK);
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
			}
			sslm_prefix_release(px);
			return s;
		};
		// kills: legacy adopt that shares (the shared pages outlive the prefix's release)
		ExpectWholeReserveAdmission(fx, "A7", what, adopt_used);
		LegacyPool pool(fx.model, 2);
		sslm_seq s = adopt_used(&pool.pool);
		if (!s) continue;
		std::vector<uint8_t> blob;
		PKV_CHECK(SaveBlob(s, &blob));
		PKV_CHECK_MSG(blob == ssb5, "9.14 A7 %s: the used holder's adopt differs from a fresh adopter's", what);
		sslm_status refusal = SSLM_OK;
		PKV_CHECK_MSG(RunToRefusal(fx.model, s, &refusal, cap + 1) == orig_turn && refusal == orig_refusal,
		              "9.14 A7 %s: the next turn differs from the fresh adopter's", what);
		sslm_seq_release(s);
	}
}

PKV_CELL("9.1/C4", "C4", Cell91LegacyRoundTrip);
PKV_CELL("9.2/C4", "C4", Cell92PinnedBlobs);
PKV_CELL("9.3/C4", "C4", Cell93OldReaderNewData);
PKV_CELL("9.5/C4", "C4", Cell95StateSizeWorstCase);
PKV_CELL("9.7/C4", "C4", Cell97MustAccept);
PKV_CELL("9.14/C4", "C4", Cell914LegacyRows);

}  // namespace
