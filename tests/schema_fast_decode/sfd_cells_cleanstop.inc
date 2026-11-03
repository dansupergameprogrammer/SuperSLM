// Engine 1.19, the clean stop at the schema's end: the CPU cells E1-E7 of the design note's §4 (the contract's pins), and E8 (coverage audit gap 8).
// Included by tests/test_schema_fast_decode.cpp inside its anonymous namespace, after the J1-X cells, so it shares
// the suite's F-SFD fixture, rig, blob reader, reference tables and seams, and runs wherever the suite runs (the
// full suite, the forced-tier suites and the scalar-forced standalone).
//
// THE CONTRACT (no ABI change). Every engine path reports the schema's end as the per-sequence sentinel -2 inside
// an SSLM_OK call, with the walk state kept. What the -2 means is read from the accepting query after it
// (sslm_stats' schema_accepting, or sslm_seq_forced_tokens' out_accepting):
//   accepting = 1  the schema is complete: a clean stop. Retryable and idempotent; the sequence stays savable,
//                  restorable, resettable and adapter-swappable.
//   accepting = 0  a dead end: a defect state the caller reports as an error.
//
// Each cell asserts the whole classification: -2 at SSLM_OK AND the accepting value, from both queries; the walk
// unchanged by the -2 call; a retry gives -2 again with the save blob byte-equal (idempotent) and the context
// length unchanged. These cells pass against 1.18's code apart from their accepting reads: they pin the
// behaviour the contract names, they are not a fix's red cells.
//
//   CS.E1  greedy, whole token: POTION and PROMPT_RESULT run to their end.
//   CS.E2  damped greedy: POTION, LONGKEY and UTF8ENUM run to their end.
//   CS.E3  sliced (layer_budget 1 < depth 2): -1, token, ..., -1, -2. The end state equals E1's byte for byte, and
//          the retry adds no K/V row.
//   CS.E4  a batch of two (n = 2): one sequence at its end, one that continues, both orders.
//   CS.E5  jump-forward, then the end: an exact draft of the true tail through the closing token, then decode; and
//          the NULL-options host loop.
//   CS.E6  save and restore across the end (SSB5 from a plain holder, SSB6 from a budget holder).
//   CS.E7  a non-accepting dead end (the degenerate-row seam at a choice state): -2 with accepting 0.
//   CS.E8  jump-forward with a draft that runs past the closing token: the tail kept through it and nothing after.

// Both accepting reads, which must agree; -1 when either call failed, -9 when they disagree.
int32_t CsAccepting(sslm_model m, sslm_seq s) {
	sslm_stats_out so{};
	const sslm_status a = sslm_stats(m, s, &so);
	int32_t count = -1, acc = -1;
	const sslm_status b = sslm_seq_forced_tokens(m, s, nullptr, 0, &count, &acc);
	if (a != SSLM_OK || b != SSLM_OK) return -1;
	return so.schema_accepting == acc ? acc : -9;
}

// A run to the schema's end: `outs` holds every decode's out_tokens value (-1 sliced passes included), `stream`
// the produced tokens. Ends at the first -2 (dead), a non-OK status, or `max` calls.
struct CsRun {
	bool dead = false;
	sslm_status st = SSLM_OK;
	std::vector<int32_t> outs, stream;
	Blob before;                    // the state the -2 call found
	std::vector<uint8_t> end_blob;  // the save blob after the first -2
	int32_t accepting = -1;         // read after the first -2
};
CsRun CsRunToEnd(Rig& rig, sslm_seq s, const sslm_decode_params& p, int max = 800) {
	CsRun r;
	for (int i = 0; i < max; ++i) {
		const Blob b = rig.State(s);
		int32_t t = -100;
		r.st = rig.Decode1(s, p, rig.W(), &t);
		if (r.st != SSLM_OK) return r;
		r.outs.push_back(t);
		if (t == -2) {
			r.dead = true;
			r.before = b;
			r.end_blob = rig.Save(s);
			r.accepting = CsAccepting(rig.m, s);
			return r;
		}
		if (t >= 0) r.stream.push_back(t);
		else if (t != -1) return r;
	}
	return r;
}

// The checks every accepting end shares: the classification, the walk kept, the closing byte, the context length,
// and `retries` further calls that give -2 with the blob byte-equal and the context unchanged.
void CsCheckCleanStop(const char* cell, const char* arm, const Sfd& f, const char* schema, Rig& rig, sslm_seq s,
                      const sslm_decode_params& p, const CsRun& r, size_t prompt_len, int retries = 2) {
	CHECK_MSG(r.dead && r.st == SSLM_OK, "[%s] %s %s: no -2 at SSLM_OK (status %d, %zu calls, %zu tokens)", cell, arm,
	          schema, static_cast<int>(r.st), r.outs.size(), r.stream.size());
	if (!r.dead) return;
	CHECK_MSG(r.accepting == 1, "[%s] %s %s: -2 with accepting %d, want 1 (the schema is complete)", cell, arm, schema,
	          r.accepting);
	const Blob e = ParseBlob(r.end_blob);
	const RefSchema& ref = f.R(schema);
	CHECK_MSG(e.ok && r.before.ok && e.walk == r.before.walk && e.walk < ref.st.size() && ref.st[e.walk].accepting,
	          "[%s] %s %s: walk %u after the -2, %u before it (the reference says accepting %d)", cell, arm, schema, e.walk,
	          r.before.walk, e.walk < ref.st.size() ? int(ref.st[e.walk].accepting) : -1);
	CHECK_MSG(e.ok && e.ready == 1 && e.layer == 0, "[%s] %s %s: rests ready %u layer %d after the -2, want 1 and 0", cell,
	          arm, schema, e.ready, e.layer);
	const std::string text = f.Bytes(r.stream);
	CHECK_MSG(!text.empty() && text.back() == '}', "[%s] %s %s: the document does not end at the closing brace (%zu "
	          "bytes)", cell, arm, schema, text.size());
	CHECK_MSG(e.ctx == int64_t(prompt_len + r.stream.size()), "[%s] %s %s: context %lld at the end, want the prompt %zu "
	          "+ the %zu produced tokens", cell, arm, schema, static_cast<long long>(e.ctx), prompt_len, r.stream.size());
	for (int k = 0; k < retries; ++k) {
		int32_t t = -100;
		const sslm_status st = rig.Decode1(s, p, rig.W(), &t);
		const std::vector<uint8_t> b = rig.Save(s);
		CHECK_MSG(st == SSLM_OK && t == -2 && CsAccepting(rig.m, s) == 1, "[%s] %s %s: retry %d gave status %d out %d "
		          "accepting %d, want OK, -2, 1", cell, arm, schema, k, static_cast<int>(st), t, CsAccepting(rig.m, s));
		CHECK_MSG(b == r.end_blob, "[%s] %s %s: retry %d changed the save blob (context %lld -> %lld)", cell, arm, schema,
		          k, static_cast<long long>(e.ctx), static_cast<long long>(ParseBlob(b).ctx));
	}
}

struct CsCase {
	const char* schema;
	const std::vector<int32_t>* prompt;
};
std::vector<CsCase> CsCases(const Sfd& f) { return {{kPotion, &f.prompt_potion}, {kPromptResult, &f.prompt_pr}}; }

// The greedy whole-token end of `schema`, for the cells that compare with it.
CsRun CsGreedyEnd(const Sfd& f, const CsCase& c) {
	Rig rig(f.m1->m, 1, 16);
	sslm_seq s = rig.NewSeq(c.schema);
	if (!s || !rig.PromptOk(s, *c.prompt)) return {};
	return CsRunToEnd(rig, s, rig.Greedy());
}

// CS.E1 greedy, whole token. The stream is cell 6.X5's pinned Off stream (kXOffGoldens).
void Cell_CS_E1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const CsCase& c : CsCases(f)) {
		Rig rig(f.m1->m, 1, 16);
		sslm_seq s = rig.NewSeq(c.schema);
		if (!s || !rig.PromptOk(s, *c.prompt)) continue;
		const sslm_decode_params g = rig.Greedy();
		const CsRun r = CsRunToEnd(rig, s, g);
		CsCheckCleanStop("CS.E1", "greedy", f, c.schema, rig, s, g, r, c.prompt->size());
		CHECK_MSG(std::count(r.outs.begin(), r.outs.end(), -1) == 0, "[CS.E1] %s: a whole-token run returned -1", c.schema);
		for (const XOffGolden& gd : kXOffGoldens)
			if (std::string(gd.schema) == c.schema)
				CHECK_MSG(r.stream.size() == gd.length && HexDigest(r.stream) == gd.digest, "[CS.E1] %s: stream %zu tokens "
				          "%s, the pinned Off stream %zu %s", c.schema, r.stream.size(), HexDigest(r.stream).c_str(), gd.length,
				          gd.digest);
	}
}

// CS.E2 damped greedy (the default order and top-k): the damped finish's own -2 site. PROMPT_RESULT is not run: its
// free-text string, damped on F-SFD's untrained weights, runs to the 256-token context cap (SSLM_CAPACITY_EXCEEDED)
// before it closes. POTION, LONGKEY and UTF8ENUM end.
void Cell_CS_E2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const std::vector<CsCase> cases = {{kPotion, &f.prompt_potion}, {kLongKey, &f.prompt_potion}, {kUtf8Enum, &f.prompt_potion}};
	for (const CsCase& c : cases) {
		Rig rig(f.m1->m, 1, 16);
		sslm_seq s = rig.NewSeq(c.schema);
		if (!s || !rig.PromptOk(s, *c.prompt)) continue;
		const sslm_decode_params d = rig.Damped();
		const CsRun r = CsRunToEnd(rig, s, d);
		CsCheckCleanStop("CS.E2", "damped", f, c.schema, rig, s, d, r, c.prompt->size());
	}
}

// CS.E3 sliced: layer_budget 1 against F-SFD's depth 2. Every token after the first takes a -1 pass and then its
// finish; the last -1 embeds the closing token and the next call gives -2. The end state is E1's byte for byte, and
// the retry takes the ready shortcut (-2 at once, no -1 and no K/V row).
void Cell_CS_E3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const CsCase& c : CsCases(f)) {
		const CsRun whole = CsGreedyEnd(f, c);
		Rig rig(f.m1->m, 1, 16);
		sslm_seq s = rig.NewSeq(c.schema);
		if (!s || !rig.PromptOk(s, *c.prompt)) continue;
		const sslm_decode_params p1 = rig.Greedy(1);
		const CsRun r = CsRunToEnd(rig, s, p1);
		CsCheckCleanStop("CS.E3", "sliced", f, c.schema, rig, s, p1, r, c.prompt->size());
		CHECK_MSG(r.stream == whole.stream, "[CS.E3] %s: the sliced stream %s, the whole-token stream %s", c.schema,
		          Str(r.stream).c_str(), Str(whole.stream).c_str());
		CHECK_MSG(r.outs.size() >= 2 && r.outs[r.outs.size() - 2] == -1 &&
		              std::count(r.outs.begin(), r.outs.end(), -1) == static_cast<std::ptrdiff_t>(r.stream.size()),
		          "[CS.E3] %s: %zu -1 passes for %zu tokens (want one per embedded token, the closing token's last); "
		          "outcomes end %s", c.schema, static_cast<size_t>(std::count(r.outs.begin(), r.outs.end(), -1)),
		          r.stream.size(), Str(std::vector<int32_t>(r.outs.end() - std::min<size_t>(r.outs.size(), 4), r.outs.end())).c_str());
		CHECK_MSG(!whole.end_blob.empty() && r.end_blob == whole.end_blob, "[CS.E3] %s: the sliced end state differs from "
		          "the whole-token end state (contexts %lld and %lld)", c.schema,
		          static_cast<long long>(ParseBlob(r.end_blob).ctx), static_cast<long long>(ParseBlob(whole.end_blob).ctx));
	}
}

// CS.E4 a batch of two in one sslm_decode_step_v2 call: A rests at the end of the document (the reference path to
// the accepting state with an empty row, as schema content), B at the start (the prompt only). The call is SSLM_OK;
// A gives -2 with accepting 1 and B gives the token it gives alone; both orders. A's state after the batch equals a
// solo -2's, and B's equals its solo decode's.
void Cell_CS_E4() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const uint32_t s_acc = AcceptingState(f, kPotion);
	CHECK_MSG(s_acc != UINT32_MAX, "[CS.E4] POTION has no accepting state with an empty row");
	if (s_acc == UINT32_MAX) return;
	std::vector<uint8_t> solo_a, solo_b;
	int32_t want_b = -100;
	{
		Rig solo(f.m1->m, 1, 8);
		sslm_seq a = ReachState(solo, f, kPotion, s_acc, f.prompt_potion);
		sslm_seq b = solo.NewSeq(kPotion);
		if (!a || !b || !solo.PromptOk(b, f.prompt_potion)) return;
		int32_t ta = -100;
		const sslm_status sa = solo.Decode1(a, solo.Greedy(), solo.W(), &ta);
		const sslm_status sb = solo.Decode1(b, solo.Greedy(), solo.W(), &want_b);
		CHECK_MSG(sa == SSLM_OK && ta == -2 && sb == SSLM_OK && want_b >= 0, "[CS.E4] solo: A %d/%d, B %d/%d",
		          static_cast<int>(sa), ta, static_cast<int>(sb), want_b);
		solo_a = solo.Save(a);
		solo_b = solo.Save(b);
	}
	for (int order = 0; order < 2; ++order) {
		Rig rig(f.m1->m, 1, 8, 3);
		sslm_seq a = ReachState(rig, f, kPotion, s_acc, f.prompt_potion);
		sslm_seq b = rig.NewSeq(kPotion);
		if (!a || !b || !rig.PromptOk(b, f.prompt_potion)) continue;
		sslm_seq batch[2] = {order == 0 ? a : b, order == 0 ? b : a};
		int32_t out[2] = {-100, -100};
		const sslm_decode_params g = rig.Greedy();
		const sslm_status st = sslm_decode_step_v2(rig.m, batch, 2, &g, rig.W(), out);
		const int32_t oa = out[order == 0 ? 0 : 1], ob = out[order == 0 ? 1 : 0];
		CHECK_MSG(st == SSLM_OK, "[CS.E4] order %d: the batch call returned %d, want SSLM_OK", order, static_cast<int>(st));
		CHECK_MSG(oa == -2 && CsAccepting(rig.m, a) == 1, "[CS.E4] order %d: A gave %d accepting %d, want -2 and 1", order,
		          oa, CsAccepting(rig.m, a));
		CHECK_MSG(ob == want_b, "[CS.E4] order %d: B gave %d, %d alone", order, ob, want_b);
		CHECK_MSG(rig.Save(a) == solo_a && rig.Save(b) == solo_b, "[CS.E4] order %d: A's state %s a solo -2's, B's %s its "
		          "solo decode's", order, rig.Save(a) == solo_a ? "equals" : "differs from",
		          rig.Save(b) == solo_b ? "equals" : "differs from");
		// The retry, A alone: -2 again, the state unchanged.
		int32_t t = -100;
		const sslm_status rt = rig.Decode1(a, g, rig.W(), &t);
		CHECK_MSG(rt == SSLM_OK && t == -2 && rig.Save(a) == solo_a, "[CS.E4] order %d: A's retry gave %d/%d", order,
		          static_cast<int>(rt), t);
	}
}

// CS.E5 jump-forward, then the end. (a) From the decode-only run's state after token k (token k - 1 pending), an
// exact draft of the true tail through the closing token: the verb returns SSLM_OK and keeps the whole tail, and
// the next decode is -2 with accepting 1, the context unchanged by it, and the state equal to the decode-only run's
// at its -2. The verb never makes the -2 itself. (b) The NULL-options host loop ends at decode's -2, accepting 1,
// with the decode-only stream and end state.
void Cell_CS_E5() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const CsCase& c : CsCases(f)) {
		const CsRun whole = CsGreedyEnd(f, c);
		const OffRun off = DecodeOnly(f.m1->m, c.schema, *c.prompt);
		CHECK_MSG(off.ok && off.dead_end && off.stream == whole.stream, "[CS.E5] %s: the decode-only run did not end as "
		          "E1's (%zu tokens)", c.schema, off.stream.size());
		if (!off.ok || !off.dead_end || off.stream.empty()) continue;
		const size_t L = off.stream.size();
		for (size_t d : {size_t(1), size_t(2), size_t(5)}) {
			if (d >= L) continue;
			const size_t k = L - d;  // tokens decoded; stream[k-1] pending, the draft stream[k..L)
			Rig rig(f.m1->m, 1, 16);
			sslm_seq s = rig.Restore(off.blob_at[k]);
			if (!s) continue;
			const std::vector<int32_t> draft(off.stream.begin() + static_cast<std::ptrdiff_t>(k), off.stream.end());
			const sslm_decode_params g = rig.Greedy();
			const Jump j = JumpDraft(rig.m, s, static_cast<int32_t>(draft.size()) + 1, &g, rig.W(), draft);
			CHECK_MSG(j.st == SSLM_OK && j.forced == draft, "[CS.E5 (a)] %s, tail of %zu: the verb gave %d, kept %s; want "
			          "OK and the tail %s", c.schema, d, static_cast<int>(j.st), Str(j.forced).c_str(), Str(draft).c_str());
			const Blob mid = rig.State(s);
			CHECK_MSG(mid.ok && mid.ready == 1 && CsAccepting(rig.m, s) == 1, "[CS.E5 (a)] %s, tail of %zu: after the verb "
			          "ready %u accepting %d, want 1 and 1", c.schema, d, mid.ready, CsAccepting(rig.m, s));
			int32_t t = -100;
			const sslm_status st = rig.Decode1(s, g, rig.W(), &t);
			const std::vector<uint8_t> end = rig.Save(s);
			CHECK_MSG(st == SSLM_OK && t == -2 && CsAccepting(rig.m, s) == 1, "[CS.E5 (a)] %s, tail of %zu: the next decode "
			          "gave %d/%d accepting %d, want OK, -2, 1", c.schema, d, static_cast<int>(st), t, CsAccepting(rig.m, s));
			CHECK_MSG(ParseBlob(end).ctx == mid.ctx, "[CS.E5 (a)] %s, tail of %zu: the -2 decode moved the context %lld -> "
			          "%lld", c.schema, d, static_cast<long long>(mid.ctx), static_cast<long long>(ParseBlob(end).ctx));
			CHECK_MSG(end == whole.end_blob, "[CS.E5 (a)] %s, tail of %zu: the end state differs from the decode-only "
			          "run's at its -2", c.schema, d);
		}
		{
			Rig rig(f.m1->m, 1, 16);
			sslm_seq s = rig.NewSeq(c.schema);
			if (!s || !rig.PromptOk(s, *c.prompt)) continue;
			JfOpts o = GreedyOpts(rig);
			const JfRun r = RunJf(rig.m, s, o);
			const bool last_is_decode = !r.steps.empty() && r.steps.back().kind == 1 && r.steps.back().st == SSLM_OK &&
			                            r.steps.back().toks == std::vector<int32_t>{-2};
			CHECK_MSG(r.dead_end && r.end_status == SSLM_OK && last_is_decode && CsAccepting(rig.m, s) == 1,
			          "[CS.E5 (b)] %s: the loop ended dead %d status %d, last step a -2 decode %d, accepting %d", c.schema,
			          r.dead_end, static_cast<int>(r.end_status), last_is_decode, CsAccepting(rig.m, s));
			bool verbs_ok = true;
			for (const JfStep& st : r.steps)
				if (st.kind == 0) verbs_ok = verbs_ok && st.st == SSLM_OK;
			CHECK_MSG(verbs_ok && r.stream == off.stream && rig.Save(s) == whole.end_blob, "[CS.E5 (b)] %s: verbs all OK %d, "
			          "stream %s (decode-only %s), end state %s", c.schema, verbs_ok, Str(r.stream).c_str(),
			          Str(off.stream).c_str(), rig.Save(s) == whole.end_blob ? "equal" : "different");
		}
	}
}

// CS.E6 save and restore across the end. A plain holder saves SSB5 and a budget holder SSB6. The restored sequence
// reads accepting 1, saves the same bytes, decodes -2 again with the bytes unchanged, and takes an adapter and a
// reset; so does the original.
void Cell_CS_E6() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const std::vector<uint8_t> abytes = BuildAdapterBytes(f);
	std::unique_ptr<Aligned> ab;
	sslm_adapter ad = nullptr;
	if (!abytes.empty()) {
		ab = std::make_unique<Aligned>(abytes.size());
		std::memcpy(ab->p, abytes.data(), abytes.size());
		CHECK_MSG(sslm_adapter_map(ab->p, abytes.size(), f.m1->m, &ad) == SSLM_OK, "[CS.E6] the synthetic adapter does not map");
	}
	CHECK_MSG(ad != nullptr, "[CS.E6] no adapter: the adapter-swap arm cannot run");
	for (const CsCase& c : CsCases(f)) {
		const CsRun whole = CsGreedyEnd(f, c);
		for (int budget_holder = 0; budget_holder < 2; ++budget_holder) {
			const char* arm = budget_holder ? "budget holder (SSB6)" : "plain holder (SSB5)";
			Rig rig(f.m1->m, 1, 16);
			const int32_t budget = static_cast<int32_t>(c.prompt->size() + whole.stream.size()) + 4;
			sslm_seq s = budget_holder ? rig.NewBudgeted(budget, c.schema) : rig.NewSeq(c.schema);
			if (!s || !rig.PromptOk(s, *c.prompt)) continue;
			const sslm_decode_params g = rig.Greedy();
			const CsRun r = CsRunToEnd(rig, s, g);
			CsCheckCleanStop("CS.E6", arm, f, c.schema, rig, s, g, r, c.prompt->size(), 1);
			if (!r.dead) continue;
			const std::vector<uint8_t>& blob = r.end_blob;
			const bool magic = blob.size() >= 4 && std::memcmp(blob.data(), budget_holder ? "SSB6" : "SSB5", 4) == 0;
			CHECK_MSG(magic, "[CS.E6] %s %s: the end blob's magic is not %s", arm, c.schema, budget_holder ? "SSB6" : "SSB5");
			sslm_seq q = rig.Restore(blob);
			if (!q) continue;
			CHECK_MSG(CsAccepting(rig.m, q) == 1, "[CS.E6] %s %s: the restored sequence reads accepting %d", arm, c.schema,
			          CsAccepting(rig.m, q));
			CHECK_MSG(rig.Save(q) == blob, "[CS.E6] %s %s: the restored sequence saves other bytes", arm, c.schema);
			int32_t t = -100;
			const sslm_status st = rig.Decode1(q, g, rig.W(), &t);
			CHECK_MSG(st == SSLM_OK && t == -2 && CsAccepting(rig.m, q) == 1 && rig.Save(q) == blob,
			          "[CS.E6] %s %s: the restored decode gave %d/%d accepting %d, bytes %s", arm, c.schema,
			          static_cast<int>(st), t, CsAccepting(rig.m, q), rig.Save(q) == blob ? "unchanged" : "changed");
			for (sslm_seq x : {s, q}) {
				const char* who = x == s ? "original" : "restored";
				if (ad) {
					const sslm_status a1 = sslm_seq_set_adapter(x, ad);
					const sslm_status a0 = sslm_seq_set_adapter(x, nullptr);
					CHECK_MSG(a1 == SSLM_OK && a0 == SSLM_OK, "[CS.E6] %s %s, %s: set_adapter %d then %d after the end, want OK",
					          arm, c.schema, who, static_cast<int>(a1), static_cast<int>(a0));
				}
				const sslm_status rs = sslm_seq_reset(x);
				CHECK_MSG(rs == SSLM_OK, "[CS.E6] %s %s, %s: reset after the end gave %d", arm, c.schema, who,
				          static_cast<int>(rs));
			}
		}
	}
	if (ad) sslm_adapter_release(ad);
}

// CS.E7 a non-accepting dead end: the degenerate-row seam (greedy only) at the first choice state, which is not
// accepting. -2 at SSLM_OK with accepting 0 on both queries, the walk kept, ready re-armed: the defect the caller
// reports as an error. The seam is single-shot, so the next decode gives the token an unarmed sequence gives.
void Cell_CS_E7() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
	for (const CsCase& c : CsCases(f)) {
		const uint32_t s_choice = FirstChoiceState(f, c.schema);
		const RefSchema& ref = f.R(c.schema);
		CHECK_MSG(s_choice != UINT32_MAX && !ref.st[s_choice].accepting, "[CS.E7] %s: no non-accepting choice state",
		          c.schema);
		if (s_choice == UINT32_MAX) continue;
		int32_t want = -100;
		{
			Rig ref_rig(f.m1->m);
			sslm_seq r = ReachState(ref_rig, f, c.schema, s_choice, *c.prompt);
			if (!r || ref_rig.Decode1(r, ref_rig.Greedy(), ref_rig.W(), &want) != SSLM_OK) continue;
		}
		Rig rig(f.m1->m);
		sslm_seq s = ReachState(rig, f, c.schema, s_choice, *c.prompt);
		if (!s) continue;
		const Blob before = rig.State(s);
		ArmCpuFinishDegenerateLogitRowInjection();
		int32_t t = -100;
		const sslm_status st = rig.Decode1(s, rig.Greedy(), rig.W(), &t);
		const Blob after = rig.State(s);
		CHECK_MSG(st == SSLM_OK && t == -2, "[CS.E7] %s: the degenerate row gave %d/%d, want OK and -2", c.schema,
		          static_cast<int>(st), t);
		CHECK_MSG(CsAccepting(rig.m, s) == 0, "[CS.E7] %s: -2 with accepting %d, want 0 (a dead end, not the end)",
		          c.schema, CsAccepting(rig.m, s));
		CHECK_MSG(after.ok && after.walk == s_choice && after.ready == 1 && after.ctx == before.ctx,
		          "[CS.E7] %s: walk %u ready %u context %lld -> %lld; want %u, 1, unchanged", c.schema, after.walk, after.ready,
		          static_cast<long long>(before.ctx), static_cast<long long>(after.ctx), s_choice);
		t = -100;
		const sslm_status nt = rig.Decode1(s, rig.Greedy(), rig.W(), &t);
		CHECK_MSG(nt == SSLM_OK && t == want, "[CS.E7] %s: the decode after (seam spent) gave %d/%d, an unarmed decode %d",
		          c.schema, static_cast<int>(nt), t, want);
	}
#else
	CHECK_MSG(false, "[CS.E7] the suite's library lacks SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION (CMakeLists.txt)");
#endif
}

// CS.E8 jump-forward with a draft that runs past the closing token (coverage audit 1.19, gap 8). From the decode-only
// run's state after token k (token k - 1 pending), an EXACT draft of the true tail through the closing token, then
// the closing token again, then a special token. Nothing is legal after the closing token, so the verb returns
// SSLM_OK and keeps exactly the true tail: the overrun is never committed. The next decode is -2 with accepting 1, the
// context is the prompt plus the decode-only stream (L tokens), and the end state equals the decode-only run's at its
// -2. Tails of 1, 2 and 5 tokens.
void Cell_CS_E8() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	CHECK_MSG(!f.specials.empty(), "[CS.E8] the fixture's tokenizer declares no special token");
	if (f.specials.empty()) return;
	const int32_t special = *f.specials.begin();
	for (const CsCase& c : CsCases(f)) {
		const CsRun whole = CsGreedyEnd(f, c);
		const OffRun off = DecodeOnly(f.m1->m, c.schema, *c.prompt);
		CHECK_MSG(off.ok && off.dead_end && off.stream == whole.stream, "[CS.E8] %s: the decode-only run did not end as "
		          "E1's (%zu tokens)", c.schema, off.stream.size());
		if (!off.ok || !off.dead_end || off.stream.empty()) continue;
		const size_t L = off.stream.size();
		for (size_t d : {size_t(1), size_t(2), size_t(5)}) {
			if (d >= L) continue;
			const size_t k = L - d;  // tokens decoded; stream[k-1] pending, the true tail stream[k..L)
			Rig rig(f.m1->m, 1, 16);
			sslm_seq s = rig.Restore(off.blob_at[k]);
			if (!s) continue;
			const std::vector<int32_t> tail(off.stream.begin() + static_cast<std::ptrdiff_t>(k), off.stream.end());
			std::vector<int32_t> draft = tail;
			draft.push_back(off.stream.back());
			draft.push_back(special);
			const sslm_decode_params g = rig.Greedy();
			const Jump j = JumpDraft(rig.m, s, static_cast<int32_t>(draft.size()) + 1, &g, rig.W(), draft);
			CHECK_MSG(j.st == SSLM_OK && j.forced == tail, "[CS.E8] %s, tail of %zu: the verb gave %d, kept %s from the "
			          "draft %s; want OK and exactly the tail %s", c.schema, d, static_cast<int>(j.st), Str(j.forced).c_str(),
			          Str(draft).c_str(), Str(tail).c_str());
			const Blob mid = rig.State(s);
			CHECK_MSG(mid.ok && mid.ready == 1 && CsAccepting(rig.m, s) == 1 &&
			              mid.ctx == static_cast<int64_t>(c.prompt->size() + L),
			          "[CS.E8] %s, tail of %zu: after the verb ready %u accepting %d context %lld; want 1, 1 and the prompt "
			          "%zu + %zu", c.schema, d, mid.ready, CsAccepting(rig.m, s), static_cast<long long>(mid.ctx),
			          c.prompt->size(), L);
			int32_t t = -100;
			const sslm_status st = rig.Decode1(s, g, rig.W(), &t);
			const std::vector<uint8_t> end = rig.Save(s);
			CHECK_MSG(st == SSLM_OK && t == -2 && CsAccepting(rig.m, s) == 1, "[CS.E8] %s, tail of %zu: the next decode gave "
			          "%d/%d accepting %d, want OK, -2, 1", c.schema, d, static_cast<int>(st), t, CsAccepting(rig.m, s));
			CHECK_MSG(ParseBlob(end).ctx == static_cast<int64_t>(c.prompt->size() + L), "[CS.E8] %s, tail of %zu: context "
			          "%lld at the end, want the prompt %zu + %zu", c.schema, d,
			          static_cast<long long>(ParseBlob(end).ctx), c.prompt->size(), L);
			CHECK_MSG(end == whole.end_blob, "[CS.E8] %s, tail of %zu: the end state differs from the decode-only run's at "
			          "its -2", c.schema, d);
		}
	}
}
