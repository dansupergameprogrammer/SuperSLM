// Schema fast-decode plan rev 1.4.4 §11.2, the verb's cells (slice G1-J): G12.1, G12.2, G12.3, G12.8 and G12.9.
//   G12.1 Fused admission. A pending token and a chain in one call make ceil(R/4) layer submissions (S-GSUB), where
//         the probe's composition (the prompt bridge for the pending token, then the schema-content bridge) makes
//         1 + ceil((R-1)/4). Tokens, walk and the SLM5 and SLM6 blobs equal the composition's; so does
//         sslm_gpu_seq_read_prefill_final_hidden after it (rev 1.4.1, audit G-7).
//   G12.2 The walked embed (Q25 A). The chain fed row by row through sslm_gpu_seq_embed_schema_token and layer
//         slices of sslm_decode_step_gpu, then one finish, gives the verb's token, walk and blob. An illegal token
//         and an unbound sequence are refused with nothing changed. From the Ready shape the walked embed clears the
//         ready flag, so the next decode bridge call embeds and finishes the new row (XG31).
//   G12.3 Entry shapes: every row of §7.3's table, the primed row included; a valid but wrong pending token is
//         embedded as given; after every refusal and after a query a fresh sequence is still bindable, and a walked
//         embed clears bind eligibility; after sslm_gpu_seq_embed_token ("embedded, no layers run") a pending token
//         re-embeds over it.
//   G12.8 The ready getter: 0 with a token in hand and in the primed row, 1 after an admitting verb call and after
//         an SLM5 or SLM6 restore of that state; KV_BUFFER_MISMATCH and BUSY arms; no submission (S-GSUB).
//   G12.9 Exact mode refused (rev 1.4.4): SSLM_JUMP_EXACT with a draft and with NULL, from each of the first three
//         shapes, is SSLM_SEQUENCE_REJECTED with the sequence unchanged (blob, walk, flags, bindability) and no
//         submission; the next canonical call equals a fresh one's (XG37).
// MR5 RE-POINT (multi-row GPU plan, ruling 29 M1 as widened). 1.18 removes Canonical: NULL options are Exact with the
// forced chain as the draft, so every expectation is Exact's, computed without the verb (Off's own next tokens, by
// decode alone on a twin; Exact keeps the draft's leading ids that equal them).
//   G12.1 The fused admission is an Exact pass: the token in hand and a draft of Off's own next 6 tokens in one call
//         (7 rows, one pass). The verb keeps all 6 in 2 submissions (the pass and its device K/V snapshot), against
//         the composition's 1 + ceil(6/4) (the prompt bridge for the pending token, then the schema-content bridge). Tokens, walk and the SLM5
//         and SLM6 blobs equal the composition's, and the next decode gives Off's next token. Re-pointed: the prefill
//         snapshot after the verb equals the one before it (Exact never touches it, XG12), not the composition's,
//         whose prompt bridge rewrites it as 1.17's verb did.
//   G12.2 The walked embed against an Exact pass with a draft of Off's own next 2 tokens (was: the forced chain). The
//         Ready arm's reference is the schema-content bridge for the start chain's first token, then a decode (was:
//         the verb's one-row admission from the ready shape, which only Canonical had: Exact keeps nothing without a
//         pending token).
//   G12.3 Fresh, -1: Exact's no-op (OK, nothing kept, context 0, the sequence as an untouched twin's), which still
//         clears bind eligibility (docs/api.md). Primed, -1: the no-op, sequence unchanged. Retired: "primed -1 equals
//         the ready shape's admission of the chain" (Canonical-only: both are no-ops now). A wrong pending token is
//         embedded as given: the verb equals the composition of the wrong token and Exact's kept prefix of the chain,
//         the prefix computed from Off's next tokens after the wrong token. TokenInHand no longer calls the verb (its
//         -1 call after the prompt is a no-op under Exact).
//   G12.8 unchanged.
//   G12.9 Exact mode is the mode (rev 1.4.4's refusal is gone): SSLM_JUMP_EXACT with NULL draft equals NULL options and
//         keeps Exact's prefix of the chain; with a draft of Off's own next 2 tokens it keeps both and the next decode
//         is Off's; from the ready and fresh shapes both are no-ops (OK, nothing kept, sequence unchanged, no
//         submission), the fresh one no longer bindable, and the next decode equals an untouched twin's (XG37's
//         "nothing left behind", re-pointed from the refusal).
// Box only. Run: sfdg_cell_g12_verb.exe [--fixture-dir=DIR] [--only=G12.N]
#include "sfdg_helpers.h"

using namespace sfdg;

namespace {

constexpr int32_t kExact = 1;  // SSLM_JUMP_EXACT, the CPU enum's value

bool Only(int argc, char** argv, const char* cell) {
	for (int i = 1; i < argc; ++i)
		if (std::strncmp(argv[i], "--only=", 7) == 0) return std::strcmp(argv[i] + 7, cell) == 0;
	return true;
}

SslmGpuKvPoolHandle* g_pool = nullptr;

GSeq NewSeq(GRig& rig, const char* schema, bool pooled) {
	if (!pooled) return rig.New(schema);
	GSeq s;
	const GS st = sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), g_pool, 255, &s.h);
	CHECK_MSG(st == GS::SSLM_OK && s.h, "budgeted create: %s", Name(st));
	if (!s.h) return s;
	rig.seqs.push_back(s.h);
	if (schema) {
		const GS b = SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, rig.Index(schema));
		CHECK_MSG(b == GS::SSLM_OK, "bind %s: %s", schema, Name(b));
	}
	return s;
}

// A sequence with a produced token in hand: the prompt, one decode. Returns the token. (MR5: 1.17 admitted the
// start chain by the verb first; under Exact that -1 call is a no-op, so it is gone.)
int32_t TokenInHand(GRig& rig, GSeq& s, const std::vector<int32_t>& prompt) {
	rig.Prompt(s, prompt);
	GS st = GS::SSLM_OK;
	const int32_t t = Token(rig.env, s, &st);
	CHECK_MSG(st == GS::SSLM_OK && t >= 0, "token in hand: %s %d", Name(st), t);
	return t;
}

// Drive the embedded row's layers in slices of `layers_per_call` layers, drained after each call.
GS DriveSliced(GRig& rig, GSeq& s, uint32_t layers_per_call) {
	const uint32_t L = rig.env.fx.layers;
	const uint32_t budget = rig.env.fx.one_layer_budget * layers_per_call;
	for (int guard = 0; LayerIdx(s.h) < L && guard < 64; ++guard) {
		GS st = sslm_decode_step_gpu(rig.ctx(), s.h, nullptr, budget);
		if (st != GS::SSLM_OK) return st;
		st = Drain(rig.env, s.h);
		if (st != GS::SSLM_OK) return st;
	}
	return LayerIdx(s.h) == L ? GS::SSLM_OK : GS::SSLM_SEQUENCE_REJECTED;
}

// The primed row (G30): the prompt's last row at full depth with the ready flag clear. Every prompt token is
// embedded and driven to depth by hand, as the plugin's prefill does, and the last is never finished. No bridge
// runs: the prompt bridge sets the ready flag, and sslm_gpu_seq_embed_token does not clear it.
void Primed(GRig& rig, GSeq& s, const std::vector<int32_t>& prompt) {
	for (const int32_t t : prompt) {
		if (sslm_gpu_seq_embed_token(rig.ctx(), s.h, t) != GS::SSLM_OK || DriveSliced(rig, s, 4) != GS::SSLM_OK) {
			CHECK_MSG(false, "primed: embed and drive %d", t);
			return;
		}
	}
	s.ready = false;
	s.last = prompt.back();
}

// ---- G12.1 -----------------------------------------------------------------------------------------------------
void Cell_12_1(GRig& rig, const std::vector<int32_t>& prompt) {
	constexpr size_t m = 6;
	for (const bool pooled : {false, true}) {
		const char* arm = pooled ? "SLM6" : "SLM5";
		GSeq a = NewSeq(rig, kPotion, pooled), b = NewSeq(rig, kPotion, pooled), o = NewSeq(rig, kPotion, pooled);
		if (!a.h || !b.h || !o.h) continue;
		const int32_t t = TokenInHand(rig, a, prompt);
		const int32_t tb = TokenInHand(rig, b, prompt);
		const int32_t to = TokenInHand(rig, o, prompt);
		CHECK_MSG(t == tb && t == to, "[G12.1 %s] the twins chose %d, %d and %d", arm, t, tb, to);
		std::vector<int32_t> offn = OffTokensGpu(rig, o, static_cast<int64_t>(m) + 1);
		while (!offn.empty() && offn.back() < 0) offn.pop_back();
		CHECK_MSG(offn.size() == m + 1, "[G12.1 %s] precondition: Off's %zu tokens after the choice %d (%zu)", arm, m + 1, t,
		          offn.size());
		if (offn.size() != m + 1) continue;
		const std::vector<int32_t> draft(offn.begin(), offn.begin() + static_cast<long>(m));
		const uint64_t R = 1 + m;
		const SnapRead snap0 = ReadSnapshot(rig.ctx(), a.h);
		uint64_t n0 = Submissions(rig.ctx());
		const JumpOut j = JumpDraft(rig, a, t, static_cast<int32_t>(R), draft);
		const uint64_t verb_subs = Submissions(rig.ctx()) - n0;
		n0 = Submissions(rig.ctx());
		const GS cs = Compose(rig, b, tb, draft);
		const uint64_t comp_subs = Submissions(rig.ctx()) - n0;
		CHECK_MSG(j.st == GS::SSLM_OK && cs == GS::SSLM_OK && j.forced == draft, "[G12.1 %s] verb %s kept %s (Off's own %s), "
		          "composition %s", arm, Name(j.st), Str(j.forced).c_str(), Str(draft).c_str(), Name(cs));
		// One Exact pass (R <= 8): its rows' layers in one submission, plus the device K/V snapshot of the chunk's first
		// slice (build-mr5.md, decision (b)); no K/V restore (every row kept) and no widened logits stage (F-SFD's tied,
		// bound head finishes on the host). The composition: the prompt bridge's one row, then ceil((R - 1) / 4).
		CHECK_MSG(verb_subs == 2 && comp_subs == 1 + (R - 1 + 3) / 4 && verb_subs < comp_subs,
		          "[G12.1 %s] R = %llu rows: the verb made %llu submissions (want 2: one pass and its K/V snapshot), the "
		          "composition %llu (want %llu)", arm, static_cast<unsigned long long>(R), static_cast<unsigned long long>(verb_subs),
		          static_cast<unsigned long long>(comp_subs), static_cast<unsigned long long>(1 + (R - 1 + 3) / 4));
		const SeqObs oa = Observe(rig, a.h), ob = Observe(rig, b.h);
		CHECK_MSG(oa == ob, "[G12.1 %s] the verb's sequence differs from the composition's: walk %u / %u, ready %d / %d, "
		          "layer %u / %u, context %lld / %lld, mapped %u / %u, blob %s", arm, oa.walk, ob.walk, oa.ready, ob.ready,
		          oa.layer, ob.layer, static_cast<long long>(oa.ctx), static_cast<long long>(ob.ctx), oa.mapped, ob.mapped,
		          oa.blob == ob.blob ? "equal" : "differs");
		CHECK_MSG(ReadSnapshot(rig.ctx(), a.h) == snap0 && snap0.st == GS::SSLM_OK,
		          "[G12.1 %s] the prefill snapshot after the verb differs from the one before it (%s, %s; XG12)", arm,
		          Name(ReadSnapshot(rig.ctx(), a.h).st), Name(snap0.st));
		GS st = GS::SSLM_OK;
		const int32_t na = Token(rig.env, a, &st), nb = Token(rig.env, b, &st);
		CHECK_MSG(na == nb && na == offn[m], "[G12.1 %s] the next tokens %d and %d; Off's %d", arm, na, nb, offn[m]);
		CHECK_MSG(rig.Blob(a.h) == rig.Blob(b.h), "[G12.1 %s] the blobs differ after the next decode", arm);
	}
}

// ---- G12.2 -----------------------------------------------------------------------------------------------------
void Cell_12_2(GRig& rig, const std::vector<int32_t>& prompt) {
	const superslm::SslmModelView& view = rig.env.fx.view;
	GSeq a = rig.New(kPotion), b = rig.New(kPotion), o = rig.New(kPotion);
	if (!a.h || !b.h || !o.h) return;
	const int32_t t = TokenInHand(rig, a, prompt);
	TokenInHand(rig, b, prompt);
	TokenInHand(rig, o, prompt);
	std::vector<int32_t> offn = OffTokensGpu(rig, o, 3);
	while (!offn.empty() && offn.back() < 0) offn.pop_back();
	CHECK_MSG(offn.size() == 3, "[G12.2] precondition: Off's 3 tokens after the choice (%zu)", offn.size());
	if (offn.size() != 3) return;
	GQuery q;
	q.ids.assign(offn.begin(), offn.begin() + 2);  // the draft: Off's own next 2 tokens
	// The verb.
	const JumpOut jv = JumpDraft(rig, a, t, 3, q.ids);
	CHECK_MSG(jv.st == GS::SSLM_OK && jv.forced == q.ids, "[G12.2] the verb: %s kept %s (Off's own %s)", Name(jv.st),
	          Str(jv.forced).c_str(), Str(q.ids).c_str());
	GS st = GS::SSLM_OK;
	const int32_t next_verb = Token(rig.env, a, &st);
	CHECK_MSG(next_verb == offn[2], "[G12.2] the verb's next token %d, Off's %d", next_verb, offn[2]);
	// The sliced path: the pending token by the plain embed (its finish walked it), the chain by the walked embed.
	CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), b.h, t) == GS::SSLM_OK && DriveSliced(rig, b, 4) == GS::SSLM_OK,
	          "[G12.2] the pending row");
	for (int32_t c : q.ids) {
		const GS e = sslm_gpu_seq_embed_schema_token(rig.ctx(), b.h, c);
		CHECK_MSG(e == GS::SSLM_OK && ReadyFlag(rig.ctx(), b.h) == 0, "[G12.2] walked embed %d: %s", c, Name(e));
		CHECK_MSG(DriveSliced(rig, b, 4) == GS::SSLM_OK, "[G12.2] the row of %d", c);
	}
	int32_t next_sliced = -9;
	st = SslmGpuSeqFinishTokenForG5Bridge(rig.ctx(), b.h, &next_sliced);
	CHECK_MSG(st == GS::SSLM_OK && next_sliced == next_verb, "[G12.2] sliced %s token %d, verb %d", Name(st), next_sliced,
	          next_verb);
	CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(a.h) == SslmGpuSeqWalkStateForG5Bridge(b.h) && rig.Blob(a.h) == rig.Blob(b.h),
	          "[G12.2] after the next token the walks are %u and %u, blobs %s", SslmGpuSeqWalkStateForG5Bridge(a.h),
	          SslmGpuSeqWalkStateForG5Bridge(b.h), rig.Blob(a.h) == rig.Blob(b.h) ? "equal" : "differ");
	// Refusals, nothing changed: an illegal token at the walk state, and an unbound sequence.
	{
		const uint32_t w = SslmGpuSeqWalkStateForG5Bridge(b.h);
		int32_t illegal = -1;
		for (int32_t id = 0; id < 4096 && illegal < 0; ++id)
			if (!LegalAt(view, kPotion, w, id)) illegal = id;
		const SeqObs before = Observe(rig, b.h);
		const GS e = sslm_gpu_seq_embed_schema_token(rig.ctx(), b.h, illegal);
		CHECK_MSG(illegal >= 0 && e == GS::SSLM_SEQUENCE_REJECTED && Observe(rig, b.h) == before,
		          "[G12.2] illegal token %d at state %u: %s, sequence %s", illegal, w, Name(e),
		          Observe(rig, b.h) == before ? "unchanged" : "changed");
		GSeq u = rig.New(nullptr);
		if (u.h) {
			rig.Prompt(u, prompt);
			const SeqObs ub = Observe(rig, u.h);
			const GS eu = sslm_gpu_seq_embed_schema_token(rig.ctx(), u.h, prompt[0]);
			CHECK_MSG(eu == GS::SSLM_SEQUENCE_REJECTED && Observe(rig, u.h) == ub, "[G12.2] unbound: %s", Name(eu));
		}
	}
	// The Ready arm (rev 1.4.2, XG31): the walked embed clears the ready flag; the decode bridge then embeds and
	// finishes the new row, as the verb's one-row admission of the same token does.
	GSeq r = rig.New(kPotion), v = rig.New(kPotion);
	if (r.h && v.h) {
		rig.Prompt(r, prompt);
		rig.Prompt(v, prompt);
		const GQuery qs = QueryGpu(rig.ctx(), r.h);
		if (qs.count > 0) {
			const int32_t c0 = qs.ids[0];
			const GS e = sslm_gpu_seq_embed_schema_token(rig.ctx(), r.h, c0);
			CHECK_MSG(e == GS::SSLM_OK && ReadyFlag(rig.ctx(), r.h) == 0, "[G12.2 Ready] walked embed: %s, ready %d", Name(e),
			          ReadyFlag(rig.ctx(), r.h));
			r.last = c0;
			const int32_t tr = Token(rig.env, r, &st);
			const GS cv = rig.Content(v, {c0});
			const int32_t tv = Token(rig.env, v, &st);
			CHECK_MSG(cv == GS::SSLM_OK && tr == tv && tr != -1, "[G12.2 Ready] the decode after the walked embed gave %d; "
			          "after the schema-content bridge's row of %d, %d (%s)", tr, c0, tv, Name(cv));
		}
	}
}

// ---- G12.3 -----------------------------------------------------------------------------------------------------
void Refused(const char* what, GRig& rig, GSeq& s, int32_t pending, GS want, const SslmGpuJumpOptions* o = nullptr) {
	const SeqObs before = Observe(rig, s.h);
	const JumpOut j = Jump(rig, s, pending, 8, o);
	CHECK_MSG(j.st == want && Observe(rig, s.h) == before, "[G12.3] %s: %s (want %s), sequence %s", what, Name(j.st),
	          Name(want), Observe(rig, s.h) == before ? "unchanged" : "changed");
}

void Cell_12_3(GRig& rig, const std::vector<int32_t>& prompt) {
	// Fresh and bound: refusals leave it bindable; a query leaves it bindable; -1 admits the start span.
	{
		GSeq f = rig.New(kPotion);
		if (f.h) {
			Refused("fresh, a token given", rig, f, prompt[0], GS::SSLM_SEQUENCE_REJECTED);
			SslmGpuJumpOptions bad = Opts(0);
			bad.struct_size = 1;
			Refused("fresh, malformed options", rig, f, -1, GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, &bad);
			Refused("fresh, pending out of range", rig, f, 1 << 30, GS::SSLM_TOKEN_ID_OUT_OF_RANGE);
			(void)QueryGpu(rig.ctx(), f.h);
			CHECK_MSG(rig.Bindable(f.h, kPotion), "[G12.3] fresh: not bindable after the refusals and a query");
			GSeq g = rig.New(kPotion);
			const JumpOut j = Jump(rig, f, -1, 64);
			CHECK_MSG(j.st == GS::SSLM_OK && j.count == 0 && CtxLen(f.h) == 0 &&
			              (g.h == nullptr || Observe(rig, f.h) == Observe(rig, g.h)),
			          "[G12.3] fresh, -1: %s, kept %d, context %lld (want Exact's no-op, as an untouched twin)", Name(j.st),
			          j.count, static_cast<long long>(CtxLen(f.h)));
			CHECK_MSG(!rig.Bindable(f.h, kPotion), "[G12.3] fresh: still bindable after a call that passed validation (a no-op "
			          "clears eligibility)");
		}
		GSeq w = rig.New(kPotion);
		if (w.h) {
			const GQuery q = QueryGpu(rig.ctx(), w.h);
			if (q.count > 0) CHECK_MSG(sslm_gpu_seq_embed_schema_token(rig.ctx(), w.h, q.ids[0]) == GS::SSLM_OK, "[G12.3] walked embed");
			CHECK_MSG(!rig.Bindable(w.h, kPotion), "[G12.3] a walked embed left the sequence bindable");
		}
	}
	// Ready: a token given is refused.
	{
		GSeq r = rig.New(kPotion);
		if (r.h) {
			rig.Prompt(r, prompt);
			Refused("ready, a token given", rig, r, prompt[0], GS::SSLM_SEQUENCE_REJECTED);
		}
	}
	// Primed (G30): -1 admits the chain and equals the ready shape's admission; a token given is refused.
	{
		GSeq p = rig.New(kPotion), r = rig.New(kPotion);
		if (p.h && r.h) {
			Primed(rig, p, prompt);
			CHECK_MSG(ReadyFlag(rig.ctx(), p.h) == 0 && LayerIdx(p.h) == rig.env.fx.layers, "[G12.3] primed: ready %d layer %u",
			          ReadyFlag(rig.ctx(), p.h), LayerIdx(p.h));
			Refused("primed, a token given", rig, p, prompt[0], GS::SSLM_SEQUENCE_REJECTED);
			rig.Prompt(r, prompt);
			const SeqObs p0 = Observe(rig, p.h), r0 = Observe(rig, r.h);
			const JumpOut jp = Jump(rig, p, -1, 64), jr = Jump(rig, r, -1, 64);
			CHECK_MSG(jp.st == GS::SSLM_OK && jp.count == 0 && Observe(rig, p.h) == p0 && jr.st == GS::SSLM_OK && jr.count == 0 &&
			              Observe(rig, r.h) == r0,
			          "[G12.3] primed, -1: %s kept %d, sequence %s; ready, -1: %s kept %d, sequence %s (want Exact's no-op)",
			          Name(jp.st), jp.count, Observe(rig, p.h) == p0 ? "unchanged" : "changed", Name(jr.st), jr.count,
			          Observe(rig, r.h) == r0 ? "unchanged" : "changed");
		}
	}
	// A token in hand: -1 refused; a valid but wrong token embedded as given; after sslm_gpu_seq_embed_token the
	// pending token re-embeds over it.
	{
		GSeq a = rig.New(kPotion), b = rig.New(kPotion), c = rig.New(kPotion);
		if (a.h && b.h && c.h) {
			const int32_t t = TokenInHand(rig, a, prompt);
			TokenInHand(rig, b, prompt);
			TokenInHand(rig, c, prompt);
			Refused("token in hand, -1", rig, a, -1, GS::SSLM_SEQUENCE_REJECTED);
			const int32_t wrong = t == prompt[0] ? prompt[1] : prompt[0];
			const GQuery q = QueryGpu(rig.ctx(), b.h);
			GSeq bw = rig.New(kPotion), ow = rig.New(kPotion);
			if (bw.h && ow.h) {
				TokenInHand(rig, bw, prompt);
				TokenInHand(rig, ow, prompt);
				ow.last = wrong;  // Off with the wrong token given: the decode embeds it as given
				const std::vector<int32_t> offw = OffTokensGpu(rig, ow, static_cast<int64_t>(q.ids.size()) + 1);
				const int32_t kept = ExactKeptPrefix(q.ids, offw, 63);
				const std::vector<int32_t> want(q.ids.begin(), q.ids.begin() + kept);
				const JumpOut jw = Jump(rig, b, wrong, 64);
				Compose(rig, bw, wrong, want);
				CHECK_MSG(jw.st == GS::SSLM_OK && jw.forced == want && Observe(rig, b.h) == Observe(rig, bw.h),
				          "[G12.3] a wrong pending token (%d, not %d), embedded as given, did not give Exact's keep: %s, kept %s; "
				          "the chain %s, Off's next after it %s, so Exact keeps %s", wrong, t, Name(jw.st), Str(jw.forced).c_str(), Str(q.ids).c_str(),
				          Str(offw).c_str(), Str(want).c_str());
			}
			CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), c.h, wrong) == GS::SSLM_OK, "[G12.3] embed_token");
			GSeq d = rig.New(kPotion);
			if (d.h) {
				TokenInHand(rig, d, prompt);
				const JumpOut jc = Jump(rig, c, t, 64), jd = Jump(rig, d, t, 64);
				CHECK_MSG(jc.st == GS::SSLM_OK && jc.forced == jd.forced && rig.Blob(c.h) == rig.Blob(d.h),
				          "[G12.3] embedded, no layers run: the pending token did not re-embed over it (%s)", Name(jc.st));
			}
		}
	}
	// Mid-token: refused. Submitted: BUSY.
	{
		GSeq m = rig.New(kPotion);
		if (m.h) {
			const int32_t t = TokenInHand(rig, m, prompt);
			CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), m.h, t) == GS::SSLM_OK, "[G12.3] mid: embed");
			GS st = sslm_decode_step_gpu(rig.ctx(), m.h, nullptr, rig.env.fx.one_layer_budget);
			CHECK_MSG(st == GS::SSLM_OK, "[G12.3] mid: one layer: %s", Name(st));
			const SslmGpuJumpOptions none = Opts(0);
			int32_t out[8], n = -9;
			const GS busy = sslm_gpu_seq_jump_forward(rig.ctx(), m.h, t, 8, &none, out, 8, &n);
			CHECK_MSG(busy == GS::SSLM_BUSY, "[G12.3] Submitted: %s", Name(busy));
			CHECK_MSG(Drain(rig.env, m.h) == GS::SSLM_OK && LayerIdx(m.h) > 0 && LayerIdx(m.h) < rig.env.fx.layers,
			          "[G12.3] mid: layer %u", LayerIdx(m.h));
			Refused("mid-token", rig, m, t, GS::SSLM_SEQUENCE_REJECTED);
			Refused("mid-token, -1", rig, m, -1, GS::SSLM_SEQUENCE_REJECTED);
		}
	}
}

// ---- G12.8 -----------------------------------------------------------------------------------------------------
void Cell_12_8(GRig& rig, const std::vector<int32_t>& prompt) {
	for (const bool pooled : {false, true}) {
		const char* arm = pooled ? "SLM6" : "SLM5";
		GSeq a = NewSeq(rig, kPotion, pooled);
		if (!a.h) continue;
		const int32_t t = TokenInHand(rig, a, prompt);
		uint64_t n0 = Submissions(rig.ctx());
		CHECK_MSG(ReadyFlag(rig.ctx(), a.h) == 0, "[G12.8 %s] token in hand: %d", arm, ReadyFlag(rig.ctx(), a.h));
		Jump(rig, a, t, 64);
		n0 = Submissions(rig.ctx());
		CHECK_MSG(ReadyFlag(rig.ctx(), a.h) == 1, "[G12.8 %s] after an admitting call: %d", arm, ReadyFlag(rig.ctx(), a.h));
		CHECK_MSG(Submissions(rig.ctx()) == n0, "[G12.8 %s] the getter submitted", arm);
		const std::vector<uint8_t> blob = rig.Blob(a.h);
		SslmGpuSequenceHandle* r = nullptr;
		const GS rs = pooled ? sslm_gpu_seq_restore_into_pool(rig.ctx(), rig.model(), g_pool, blob.data(), blob.size(),
		                                                      nullptr, &r, nullptr)
		                     : sslm_gpu_seq_restore(rig.ctx(), rig.model(), blob.data(), blob.size(), &r);
		if (r) rig.seqs.push_back(r);
		CHECK_MSG(rs == GS::SSLM_OK && r && ReadyFlag(rig.ctx(), r) == 1, "[G12.8 %s] after the restore: %s, %d", arm, Name(rs),
		          r ? ReadyFlag(rig.ctx(), r) : -1);
	}
	GSeq p = rig.New(kPotion);
	if (p.h) {
		Primed(rig, p, prompt);
		CHECK_MSG(ReadyFlag(rig.ctx(), p.h) == 0, "[G12.8] primed: %d", ReadyFlag(rig.ctx(), p.h));
		int32_t out = -9;
		CHECK_MSG(sslm_gpu_seq_ready_for_logits(rig.ctx(), p.h, nullptr) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH &&
		              sslm_gpu_seq_ready_for_logits(rig.ctx(), nullptr, &out) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH &&
		              sslm_gpu_seq_ready_for_logits(nullptr, p.h, &out) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && out == -9,
		          "[G12.8] the malformed arms");
		CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), p.h, prompt[0]) == GS::SSLM_OK &&
		              sslm_decode_step_gpu(rig.ctx(), p.h, nullptr, rig.env.fx.one_layer_budget) == GS::SSLM_OK,
		          "[G12.8] submit one layer");
		CHECK_MSG(sslm_gpu_seq_ready_for_logits(rig.ctx(), p.h, &out) == GS::SSLM_BUSY && out == -9, "[G12.8] Submitted: not BUSY");
		Drain(rig.env, p.h);
	}
}

// ---- G12.9 -----------------------------------------------------------------------------------------------------
void Cell_12_9(GRig& rig, const std::vector<int32_t>& prompt) {
	// A token in hand: SSLM_JUMP_EXACT with NULL draft equals NULL options and keeps Exact's prefix of the chain; with a
	// draft of Off's own next 2 tokens it keeps both.
	{
		GSeq s = rig.New(kPotion), twin = rig.New(kPotion), o = rig.New(kPotion), d = rig.New(kPotion);
		if (s.h && twin.h && o.h && d.h) {
			const int32_t p = TokenInHand(rig, s, prompt);
			TokenInHand(rig, twin, prompt);
			TokenInHand(rig, o, prompt);
			TokenInHand(rig, d, prompt);
			const GQuery q = QueryGpu(rig.ctx(), s.h);
			std::vector<int32_t> offn = OffTokensGpu(rig, o, static_cast<int64_t>(std::max<size_t>(q.ids.size(), 2)) + 1);
			const int32_t kept = ExactKeptPrefix(q.ids, offn, 7);
			const std::vector<int32_t> want(q.ids.begin(), q.ids.begin() + kept);
			const SslmGpuJumpOptions ex = Opts(kExact);
			const JumpOut a = Jump(rig, s, p, 8, &ex), b = Jump(rig, twin, p, 8);
			CHECK_MSG(a.st == GS::SSLM_OK && a.forced == want && b.st == GS::SSLM_OK && b.forced == want &&
			              Observe(rig, s.h) == Observe(rig, twin.h),
			          "[G12.9] token in hand, exact NULL: %s kept %s; NULL options %s kept %s; Exact keeps %s (the chain %s, Off's "
			          "next %s)", Name(a.st), Str(a.forced).c_str(), Name(b.st), Str(b.forced).c_str(), Str(want).c_str(),
			          Str(q.ids).c_str(), Str(offn).c_str());
			if (offn.size() >= 3 && offn[0] >= 0 && offn[1] >= 0 && offn[2] >= 0) {
				const std::vector<int32_t> draft(offn.begin(), offn.begin() + 2);
				const JumpOut jd = JumpDraft(rig, d, p, 8, draft);
				GS st = GS::SSLM_OK;
				const int32_t nd = Token(rig.env, d, &st);
				CHECK_MSG(jd.st == GS::SSLM_OK && jd.forced == draft && nd == offn[2], "[G12.9] token in hand, exact with Off's "
				          "own draft %s: %s kept %s, the next decode %d (Off's %d)", Str(draft).c_str(), Name(jd.st),
				          Str(jd.forced).c_str(), nd, offn[2]);
			} else {
				CHECK_MSG(false, "[G12.9] precondition: Off's 3 tokens after the choice (%s)", Str(offn).c_str());
			}
		}
	}
	// Ready and fresh: no pending token, so both forms are no-ops that leave nothing behind (XG37).
	for (int shape = 1; shape < 3; ++shape) {
		const char* name = shape == 1 ? "ready" : "fresh";
		GSeq s = rig.New(kPotion), twin = rig.New(kPotion);
		if (!s.h || !twin.h) continue;
		if (shape == 1) {
			rig.Prompt(s, prompt);
			rig.Prompt(twin, prompt);
		}
		const int32_t draft[1] = {prompt[0]};
		for (int with_draft = 0; with_draft < 2; ++with_draft) {
			const SslmGpuJumpOptions ex = with_draft ? Opts(kExact, draft, 1) : Opts(kExact);
			const SeqObs before = Observe(rig, s.h);
			const uint64_t n0 = Submissions(rig.ctx());
			const JumpOut j = Jump(rig, s, -1, 8, &ex);
			CHECK_MSG(j.st == GS::SSLM_OK && j.count == 0 && Observe(rig, s.h) == before && Submissions(rig.ctx()) == n0,
			          "[G12.9] %s, exact %s: %s, kept %d, sequence %s, %llu submissions (want Exact's no-op)", name,
			          with_draft ? "with a draft" : "NULL", Name(j.st), j.count, Observe(rig, s.h) == before ? "unchanged" : "changed",
			          static_cast<unsigned long long>(Submissions(rig.ctx()) - n0));
		}
		if (shape == 2) {
			CHECK_MSG(!rig.Bindable(s.h, kPotion), "[G12.9] fresh: still bindable after the no-ops (they clear eligibility)");
		} else {
			GS st = GS::SSLM_OK, st2 = GS::SSLM_OK;
			const int32_t a = Token(rig.env, s, &st), b = Token(rig.env, twin, &st2);
			CHECK_MSG(st == st2 && a == b && rig.Blob(s.h) == rig.Blob(twin.h), "[G12.9] ready: the decode after the no-ops %s %d, "
			          "an untouched twin's %s %d", Name(st), a, Name(st2), b);
		}
	}
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		const GS ps = sslm_gpu_kv_pool_create(rig.ctx(), rig.model(), 96, &g_pool);
		CHECK_MSG(ps == GS::SSLM_OK && g_pool, "pool: %s", Name(ps));
		if (Only(argc, argv, "G12.1") && g_pool) Cell_12_1(rig, prompt);
		if (Only(argc, argv, "G12.2")) Cell_12_2(rig, prompt);
		if (Only(argc, argv, "G12.3")) Cell_12_3(rig, prompt);
		if (Only(argc, argv, "G12.8") && g_pool) Cell_12_8(rig, prompt);
		if (Only(argc, argv, "G12.9")) Cell_12_9(rig, prompt);
		rig.ReleaseSeqs();
		if (g_pool) sslm_gpu_kv_pool_destroy(rig.ctx(), g_pool);
	}
	return Finish("sfdg_cell_g12_verb");
}
