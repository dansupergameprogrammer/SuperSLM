// Paged-KV plan (rev 16.2) §7 cell 9.13, its [G5] rows: the GPU refusal rows of §3.9 and the §4.10
// status precedence. (9.13's [G4] row, adopt onto a dedicated sequence (A15), is another group's.)
// Group g1e, step G1. Plan sources: §3.9 A10, A11, A13, A14, A21; §4 item 10 (writer rule, dispositions,
// "Busy, and the status precedence"); §4 item 7 (GPU refusal statuses); cells 2.6 and 2.11.
//
// Each status is SSLM_GPU_HOLDER_MODE_MISMATCH unless stated.
//   A11  sslm_gpu_seq_restore of an SLM6: the status and a null out-handle (a dedicated restore
//        allocates outside any pool, where no fill probe can see it). Kills the mutant that sizes a
//        dedicated buffer to the saved limit (rev 3's text): it returns SSLM_OK.
//   A14  sslm_gpu_seq_restore_into_pool of an SLM5 and of an SLM4, each with a null prefix and with a
//        matching prefix handle (9.12's refused case (a): a non-budget blob with a handle): the
//        status, a null out-handle, and the GPU fill probe unchanged. Also an unknown magic ('SLMX' over
//        the SLM5 operand), "any non-budget blob": the same status, null out, fill probe unchanged.
//   §4.10 precedence, step 3 before step 4: an SLM6 with hostile fields (each of 2.3's) given to the
//        dedicated verb is a mode mismatch, whatever the field; step 4: an SLM6 whose kv_mode is not
//        budget given to the into-pool verb fails validation with SSLM_SEQUENCE_KV_BUFFER_MISMATCH
//        (2.6), null out-handle, fill probe unchanged. Steps 1 and 2 before step 3 (the test author's
//        reading of "the §4.10 precedence", recorded in the g1e notes): a malformed call (null out,
//        another context's model) with an SLM6 or SLM5 returns SSLM_SEQUENCE_KV_BUFFER_MISMATCH, and,
//        while another sequence is Submitted (the G3 wait seam), both verbs return SSLM_BUSY for the
//        blob they would otherwise refuse on magic, with nothing drawn.
//   A21  the into-pool verb given a pool of another context, a pool of another model, and a model of
//        another context: SSLM_SEQUENCE_KV_BUFFER_MISMATCH, a null out-handle, and the GPU fill probe
//        unchanged on every pool involved (2.11). A prefix of another context or model is not refused
//        (A13): it restores privately, asserted in 2.11's G5 part and in 9.12.
//   A10  a dedicated SLM5 round trip, then a reset and a run, behaves as today: the restored holder
//        continues as the unsaved original, and after a reset and a prompt it emits what a fresh
//        dedicated sequence emits for that prompt. A dedicated holder never adopts (A15), so there is
//        no re-adopt.
//
// RED NOW: link-red (G3 pool/budgeted create/wait seam, G4 prefixes, G5 restore-into-pool); after G4,
// red at run until G5 (no SLM6 writer; the dedicated verb does not refuse SLM6 with status 27).
// GREEN: every check passes on both GPUs; debug layer clean (SSLM_GPU_ENABLE_DEBUG_LAYER=1).
//
// Run: g1e_cell_9_13.exe --fixture-dir=DIR
#include "g1e_helpers.h"

using namespace g1e;

namespace {

constexpr int32_t kBudget = 512;
constexpr int64_t kPersonaLen = 1200;

struct Blobs {
	std::vector<uint8_t> slm6;  // a pooled budget holder: persona adopted, 300-token turn (9.9's fixed state)
	std::vector<uint8_t> slm5;  // a dedicated holder of the same history
	std::vector<uint8_t> slm4;
	GSeq like6, like5;
	bool ok = false;
};

Blobs MakeBlobs(const Env& e) {
	Blobs b;
	{
		Rig rig(e, 300);
		const GChain c = rig.pool ? BuildPersonaChain(rig, 0, true) : GChain{};
		GSeq s = c.ok ? rig.Budgeted(kBudget) : GSeq{};
		if (!s.h) return b;
		CHECK(Adopt(rig, s, c.persona, kPersonaLen) == GS::SSLM_OK);
		CHECK(Prefill(e, s, Stream(91, 300, kVocab)) == GS::SSLM_OK);
		b.slm6 = Save(e, s.h);
		b.like6 = s;
		if (!Slm6SelfCheck(b.slm6, e.geo, kBudget, kPersonaLen, 1500, "9.13 the SLM6 operand")) return b;
	}
	{
		SslmGpuSequenceHandle* h = nullptr;
		CHECK(sslm_gpu_seq_create(e.ctx, e.model(), 2048, &h) == GS::SSLM_OK && h);
		if (!h) return b;
		GSeq d;
		d.h = h;
		std::vector<int32_t> hist = WorldTokens();
		const std::vector<int32_t> p = PersonaTokens(0), u = Stream(91, 300, kVocab);
		hist.insert(hist.end(), p.begin(), p.end());
		hist.insert(hist.end(), u.begin(), u.end());
		CHECK(Prefill(e, d, hist) == GS::SSLM_OK);
		b.slm5 = Save(e, h);
		b.like5 = d;
		sslm_gpu_seq_release(e.ctx, h);
	}
	CHECK_MSG(IsMagic(b.slm5, "SLM5"), "9.13: a dedicated holder's save is not 'SLM5'");
	b.slm4 = Slm4FromSlm5(b.slm5);
	b.ok = !b.slm6.empty() && !b.slm4.empty();
	return b;
}

// SLM6 with one hostile field (2.3's cases, in the provisional layout of g1e_helpers.h).
std::vector<std::pair<std::string, std::vector<uint8_t>>> HostileSlm6(const Env& e, const std::vector<uint8_t>& good) {
	std::vector<std::pair<std::string, std::vector<uint8_t>>> out;
	const Slm6View v = Slm6Parse(good, e.geo);
	if (!v.ok) return out;
	auto with = [&](const char* name, const std::function<void(std::vector<uint8_t>&)>& edit) {
		std::vector<uint8_t> b = good;
		edit(b);
		out.emplace_back(name, std::move(b));
	};
	with("budget 0", [&](std::vector<uint8_t>& b) { Put32(b, kSlm6BudgetOff, 0); });
	with("budget -1", [&](std::vector<uint8_t>& b) { Put32(b, kSlm6BudgetOff, 0xFFFFFFFFu); });
	with("budget above cap", [&](std::vector<uint8_t>& b) { Put32(b, kSlm6BudgetOff, static_cast<uint32_t>(e.geo.cap + 1)); });
	with("origin -1", [&](std::vector<uint8_t>& b) { Put64(b, kSlm6OriginOff, ~uint64_t{0}); });
	with("origin above context_length",
	     [&](std::vector<uint8_t>& b) { Put64(b, kSlm6OriginOff, static_cast<uint64_t>(v.context_length + 1)); });
	with("context_length above the limit", [&](std::vector<uint8_t>& b) {
		Put32(b, kSlm6BudgetOff, static_cast<uint32_t>(v.context_length - v.origin - 1));
	});
	with("kv_mode 2", [&](std::vector<uint8_t>& b) { Put32(b, kSlm6KvModeOff, 2); });
	with("kv_mode 0 (whole_reserve)", [&](std::vector<uint8_t>& b) { Put32(b, kSlm6KvModeOff, 0); });
	with("K/V size inconsistent", [&](std::vector<uint8_t>& b) { Put64(b, 48, Le64(b, 48) - e.geo.BytesPerToken()); });
	with("truncated K/V", [&](std::vector<uint8_t>& b) { b.resize(b.size() - 1); });
	return out;
}

// The rebuildable state for the fill probe: a pool holding the persona chain (75 pages), then the
// refused call made on it. `pages` - 75 pages stay free if the call drew nothing.
struct RefusedRig : Rig {
	RefusedRig(const Env& e, uint32_t pages) : Rig(e, pages) {
		if (pool) BuildPersonaChain(*this, 0, true);
	}
};

void A11(const Env& e, const Blobs& b) {
	// kills: the mutant that sizes a dedicated buffer to min(origin + budget, cap) (rev 3), which restores
	SslmGpuSequenceHandle* out = OutSentinel<SslmGpuSequenceHandle>();
	const GS st = sslm_gpu_seq_restore(e.ctx, e.model(), b.slm6.data(), b.slm6.size(), &out);
	CHECK_MSG(st == GS::SSLM_GPU_HOLDER_MODE_MISMATCH, "9.13 A11: sslm_gpu_seq_restore(SLM6): %s", StatusName(st));
	CHECK_MSG(out == nullptr, "9.13 A11: the out-handle is not null");
	if (Real(out)) sslm_gpu_seq_release(e.ctx, out);
	// §4.10 step 3 before step 4: the magic alone decides on the dedicated verb.
	for (const auto& h : HostileSlm6(e, b.slm6)) {
		SslmGpuSequenceHandle* o = OutSentinel<SslmGpuSequenceHandle>();
		const GS s = sslm_gpu_seq_restore(e.ctx, e.model(), h.second.data(), h.second.size(), &o);
		// kills: a dedicated verb that validates the SLM6 body before its magic
		CHECK_MSG(s == GS::SSLM_GPU_HOLDER_MODE_MISMATCH && o == nullptr,
		          "9.13 precedence: sslm_gpu_seq_restore(SLM6, %s): %s (want the mode mismatch, null out)", h.first.c_str(),
		          StatusName(s));
		if (Real(o)) sslm_gpu_seq_release(e.ctx, o);
	}
}

// One refused into-pool call on the persona rig, graded by status, null out and the fill probe.
void RefusedIntoPool(const Env& e, const std::vector<uint8_t>& blob, bool with_prefix, GS want, const std::string& what) {
	const int64_t chain = e.geo.PersonaChainPages(200, true);  // 75
	const uint32_t pages = static_cast<uint32_t>(chain + e.geo.R(kBudget) + e.geo.Materialized(kPersonaLen, kBudget) + 3);
	{
		RefusedRig rig(e, pages);
		if (!rig.chain.ok) return;
		SslmGpuSequenceHandle* out = OutSentinel<SslmGpuSequenceHandle>();
		uint32_t shared = 0xABCDu;
		const GS st = sslm_gpu_seq_restore_into_pool(rig.ctx, rig.model, rig.pool, blob.data(), blob.size(),
		                                             with_prefix ? rig.chain.persona : nullptr, &out, &shared);
		CHECK_MSG(st == want, "%s: %s (want %s)", what.c_str(), StatusName(st), StatusName(want));
		CHECK_MSG(out == nullptr, "%s: the out-handle is not null", what.c_str());
		if (Real(out)) rig.Seq(out);
	}
	// kills: a refusal after a draw (the pool one reservation short), or a refused restore that shares
	ProbeExactlyFree(e, [&]() -> std::unique_ptr<ProbeState> {
		auto rig = std::make_unique<RefusedRig>(e, pages);
		if (rig->chain.ok) {
			SslmGpuSequenceHandle* out = nullptr;
			(void)sslm_gpu_seq_restore_into_pool(rig->ctx, rig->model, rig->pool, blob.data(), blob.size(),
			                                     with_prefix ? rig->chain.persona : nullptr, &out, nullptr);
			if (out) rig->Seq(out);
		}
		return rig;
	}, static_cast<int64_t>(pages) - chain, what.c_str());
}

void A14AndStep4(const Env& e, const Blobs& b) {
	for (int with_prefix = 0; with_prefix <= 1; ++with_prefix) {
		const char* hp = with_prefix ? " with the matching prefix" : " with a null prefix";
		RefusedIntoPool(e, b.slm5, with_prefix != 0, GS::SSLM_GPU_HOLDER_MODE_MISMATCH, std::string("9.13 A14 SLM5") + hp);
		RefusedIntoPool(e, b.slm4, with_prefix != 0, GS::SSLM_GPU_HOLDER_MODE_MISMATCH, std::string("9.13 A14 SLM4") + hp);
		// "Any non-budget blob": an unknown magic is not 'SLM6', so the magic step refuses it (§4.10 step 3,
		// gpu_1p0.h's step (3)); kills a magic check that refuses only the named 'SLM5'/'SLM4'.
		std::vector<uint8_t> unk = b.slm5;
		std::memcpy(unk.data(), "SLMX", 4);
		RefusedIntoPool(e, unk, with_prefix != 0, GS::SSLM_GPU_HOLDER_MODE_MISMATCH, std::string("9.13 A14 unknown magic") + hp);
	}
	// §4.10 step 4: an SLM6 whose kv_mode is not budget fails validation (2.6's status), not the magic.
	std::vector<uint8_t> wr = b.slm6;
	Put32(wr, kSlm6KvModeOff, 0);
	RefusedIntoPool(e, wr, false, GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "9.13 precedence: SLM6 kv_mode 0 into pool");
}

// §4.10 steps 1 and 2 before step 3.
void Steps12(const Env& e, const Blobs& b) {
	// Step 1: malformed calls keep today's status whatever the blob.
	{
		const GS st = sslm_gpu_seq_restore(e.ctx, e.model(), b.slm6.data(), b.slm6.size(), nullptr);
		CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "9.13 step 1: dedicated restore(SLM6, null out): %s", StatusName(st));
		Rig rig(e, 64);
		if (rig.pool) {
			const GS s2 = sslm_gpu_seq_restore_into_pool(e.ctx, e.model(), rig.pool, b.slm5.data(), b.slm5.size(), nullptr,
			                                             nullptr, nullptr);
			CHECK_MSG(s2 == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "9.13 step 1: into-pool(SLM5, null out): %s", StatusName(s2));
		}
	}
	// Step 2: SSLM_BUSY while any sequence in the process is Submitted, before the magic. A dedicated
	// sequence is left Submitted through the G3 wait seam (it waits on the fence without finishing).
	{
		SslmGpuSequenceHandle* busy = nullptr;
		CHECK(sslm_gpu_seq_create(e.ctx, e.model(), 64, &busy) == GS::SSLM_OK && busy);
		Rig rig(e, static_cast<uint32_t>(e.geo.R(kBudget) + e.geo.Materialized(kPersonaLen, kBudget) + 4));
		if (!busy || !rig.pool) {
			if (busy) sslm_gpu_seq_release(e.ctx, busy);
			return;
		}
		GSeq bs;
		bs.h = busy;
		CHECK(Prefill(e, bs, Stream(5, 4, kVocab)) == GS::SSLM_OK);
		CHECK(Token(e, bs) >= 0);
		CHECK(sslm_gpu_seq_embed_token(e.ctx, busy, bs.last) == GS::SSLM_OK);
		CHECK(sslm_decode_step_gpu(e.ctx, busy, nullptr, e.fx.one_layer_budget) == GS::SSLM_OK);
		CHECK(sslm_gpu_test_only_wait_in_flight(e.ctx, busy) == GS::SSLM_OK);
		SslmGpuSequenceHandle* o1 = OutSentinel<SslmGpuSequenceHandle>();
		const GS s1 = sslm_gpu_seq_restore(e.ctx, e.model(), b.slm6.data(), b.slm6.size(), &o1);
		// kills: the magic checked before the process-wide Busy
		CHECK_MSG(s1 == GS::SSLM_BUSY && o1 == nullptr, "9.13 step 2: dedicated restore(SLM6) while Submitted: %s",
		          StatusName(s1));
		SslmGpuSequenceHandle* o2 = OutSentinel<SslmGpuSequenceHandle>();
		const GS s2 = sslm_gpu_seq_restore_into_pool(e.ctx, e.model(), rig.pool, b.slm5.data(), b.slm5.size(), nullptr, &o2,
		                                             nullptr);
		CHECK_MSG(s2 == GS::SSLM_BUSY && o2 == nullptr, "9.13 step 2: into-pool(SLM5) while Submitted: %s", StatusName(s2));
		SslmGpuSequenceHandle* o3 = OutSentinel<SslmGpuSequenceHandle>();
		const GS s3 = sslm_gpu_seq_restore_into_pool(e.ctx, e.model(), rig.pool, b.slm6.data(), b.slm6.size(), nullptr, &o3,
		                                             nullptr);
		CHECK_MSG(s3 == GS::SSLM_BUSY && o3 == nullptr, "9.13 step 2: into-pool(SLM6) while Submitted: %s", StatusName(s3));
		for (SslmGpuSequenceHandle* o : {o1, o2, o3})
			if (Real(o)) rig.Seq(o);
		CHECK(Drain(e, busy) == GS::SSLM_OK);
		// Nothing was drawn: the whole pool is free.
		CHECK_MSG(AdmitsPages(e, rig, static_cast<int64_t>(e.geo.R(kBudget) + e.geo.Materialized(kPersonaLen, kBudget) + 4)),
		          "9.13 step 2: a Busy refusal drew pages");
		sslm_gpu_seq_release(e.ctx, busy);
	}
}

void A21(const Env& e, const Blobs& b) {
	// Another context, with the fixture mapped on it; and a second model on this context.
	SslmGpuContext* ctx2 = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx2) == GS::SSLM_OK && ctx2);
	SslmGpuModelHandle* m2 = nullptr;  // the same artifact, another context
	SslmGpuModelHandle* mb = nullptr;  // the same artifact, this context, another model handle
	if (ctx2) CHECK(sslm_gpu_model_map(ctx2, &e.fx.view, GpuResidencyConfig{}, &m2) == GS::SSLM_OK && m2);
	CHECK(sslm_gpu_model_map(e.ctx, &e.fx.view, GpuResidencyConfig{}, &mb) == GS::SSLM_OK && mb);
	if (!m2 || !mb) {
		if (mb) sslm_gpu_model_unmap(e.ctx, mb);
		if (m2) sslm_gpu_model_unmap(ctx2, m2);
		if (ctx2) sslm_gpu_context_destroy(ctx2);
		return;
	}
	const uint32_t pages = static_cast<uint32_t>(e.geo.R(kBudget) + e.geo.Materialized(kPersonaLen, kBudget) + 3);
	struct Case {
		const char* name;
		int pool_kind;  // 0: this context and model; 1: another context; 2: another model
		int model_kind;  // 0: this context's model; 1: another context's model
	};
	const Case cases[] = {{"a pool of another context", 1, 0}, {"a pool of another model", 2, 0}, {"a model of another context", 0, 1}};
	for (const Case& c : cases) {
		const std::string what = std::string("9.13 A21: ") + c.name;
		auto build = [&]() -> std::unique_ptr<Rig> {
			SslmGpuContext* pc = c.pool_kind == 1 ? ctx2 : e.ctx;
			SslmGpuModelHandle* pm = c.pool_kind == 1 ? m2 : c.pool_kind == 2 ? mb : e.model();
			return std::make_unique<Rig>(e, pages, pc, pm);
		};
		{
			std::unique_ptr<Rig> pool_rig = build();
			if (!pool_rig->pool) continue;
			SslmGpuSequenceHandle* out = OutSentinel<SslmGpuSequenceHandle>();
			SslmGpuModelHandle* model_arg = c.model_kind == 1 ? m2 : e.model();
			const GS st = sslm_gpu_seq_restore_into_pool(e.ctx, model_arg, pool_rig->pool, b.slm6.data(), b.slm6.size(),
			                                             nullptr, &out, nullptr);
			// kills: a restore that draws before it checks the handles (the pool one reservation short)
			CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && out == nullptr, "%s: %s", what.c_str(), StatusName(st));
			if (Real(out)) sslm_gpu_seq_release(e.ctx, out);
		}
		// The pool involved is unchanged: all its pages free.
		ProbeExactlyFree(e, [&]() -> std::unique_ptr<ProbeState> {
			std::unique_ptr<Rig> r = build();
			if (r->pool) {
				SslmGpuSequenceHandle* out = nullptr;
				(void)sslm_gpu_seq_restore_into_pool(e.ctx, c.model_kind == 1 ? m2 : e.model(), r->pool, b.slm6.data(),
				                                     b.slm6.size(), nullptr, &out, nullptr);
				if (out) sslm_gpu_seq_release(e.ctx, out);
			}
			return r;
		}, pages, what.c_str());
	}
	sslm_gpu_model_unmap(e.ctx, mb);
	sslm_gpu_model_unmap(ctx2, m2);
	sslm_gpu_context_destroy(ctx2);
}

void A10(const Env& e, const Blobs& b) {
	SslmGpuSequenceHandle* r = OutSentinel<SslmGpuSequenceHandle>();
	const GS st = sslm_gpu_seq_restore(e.ctx, e.model(), b.slm5.data(), b.slm5.size(), &r);
	CHECK_MSG(st == GS::SSLM_OK && Real(r), "9.13 A10: sslm_gpu_seq_restore(SLM5): %s", StatusName(st));
	if (!Real(r)) return;
	// The unsaved original's continuation: a fresh dedicated twin of the same history.
	std::vector<int32_t> orig_next;
	{
		SslmGpuSequenceHandle* h = nullptr;
		CHECK(sslm_gpu_seq_create(e.ctx, e.model(), 2048, &h) == GS::SSLM_OK && h);
		if (h) {
			GSeq d;
			d.h = h;
			std::vector<int32_t> hist = WorldTokens();
			const std::vector<int32_t> p = PersonaTokens(0), u = Stream(91, 300, kVocab);
			hist.insert(hist.end(), p.begin(), p.end());
			hist.insert(hist.end(), u.begin(), u.end());
			CHECK(Prefill(e, d, hist) == GS::SSLM_OK);
			orig_next = DecodeN(e, d, 12);
			sslm_gpu_seq_release(e.ctx, h);
		}
	}
	GSeq t = b.like5;
	t.h = r;
	CHECK_MSG(DecodeN(e, t, 12) == orig_next && orig_next.size() == 12, "9.13 A10: the restored SLM5 holder's tokens differ");
	// Reset and a run, with no re-adopt: as a fresh dedicated sequence.
	CHECK(sslm_gpu_seq_reset(e.ctx, r) == GS::SSLM_OK);
	t = GSeq{};
	t.h = r;
	const std::vector<int32_t> prompt = Stream(41, 20, kVocab);
	CHECK(Prefill(e, t, prompt) == GS::SSLM_OK);
	const std::vector<int32_t> after_reset = DecodeN(e, t, 12);
	std::vector<int32_t> fresh_run;
	{
		SslmGpuSequenceHandle* h = nullptr;
		CHECK(sslm_gpu_seq_create(e.ctx, e.model(), 2048, &h) == GS::SSLM_OK && h);
		if (h) {
			GSeq f;
			f.h = h;
			CHECK(Prefill(e, f, prompt) == GS::SSLM_OK);
			fresh_run = DecodeN(e, f, 12);
			sslm_gpu_seq_release(e.ctx, h);
		}
	}
	CHECK_MSG(after_reset == fresh_run && fresh_run.size() == 12, "9.13 A10: after reset the restored holder differs from a fresh one");
	sslm_gpu_seq_release(e.ctx, r);
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			const Blobs b = MakeBlobs(e);
			if (b.ok) {
				A11(e, b);
				A14AndStep4(e, b);
				Steps12(e, b);
				A21(e, b);
				A10(e, b);
			} else {
				CHECK_MSG(false, "9.13: the operand blobs could not be made");
			}
		}
	}
	return FinishSuite("g1e_cell_9_13");
}
