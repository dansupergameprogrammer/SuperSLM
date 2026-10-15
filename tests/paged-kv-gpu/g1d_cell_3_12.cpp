// Paged KV GPU slice, plan step G1 -- cell 3.12, THE POOL UPLOAD'S LATCH RULE ON BOTH BRANCHES (plan.md Sec7 3.12;
// Sec4.8; decision 54; carrier row "The latch-rule cell"). Brackets: [G4] the latch-set and set-then-cleared runs;
// [G5] the twin (the same two runs with restore-into-pool as the second upload) and the latch-clear run.
//
// THE RULE (Sec4.8): every dispatch leaves the pool buffer in COPY_SOURCE and sets the latch. The pool upload,
// UploadPoolRegionsSync on harness::GetDevice(), transitions COPY_SOURCE -> COPY_DEST when the latch is set and
// UNORDERED_ACCESS -> COPY_DEST when it is clear, and clears the latch after it. A dispatch between two uploads
// sets the latch and never clears it, so the clear branch is reached only by two uploads with no dispatch
// between them (G4), or by a first upload on a fresh pool (G5).
//
// BUILD: the default GPU build (no fault seams; the queue gate is not used, G1D_NO_QUEUE_GATE). The debug layer
// is REQUIRED (SSLM_GPU_ENABLE_DEBUG_LAYER=1): the verdict is a clean debug layer, which is vacuous without it.
// --model needs a context cap of at least 1,100 (a 1,000-token mid-page prefix plus decodes).
//
// Runs (--run=), each its own process:
//   g4-set      [G4] Latch set: X = prefix_begin(pool, 1000) prefilled with 1,000 tokens (a dispatch) and frozen;
//               pooled A decodes (with its row readback) and sslm_gpu_ready(A); B (budgeted) adopts X (upload 1,
//               latch set); A decodes again and sslm_gpu_ready(A); C = begin_from(X) (upload 2, latch set).
//   g4-cleared  [G4] Set, then cleared: X as above; A decodes and sslm_gpu_ready(A); B adopts X (upload 1, latch
//               set) and C = begin_from(X) (upload 2, the CLEAR branch), with no dispatch between the uploads.
//   g5-set      [G5] g4-set with restore-into-pool (prefix hint X) as the second upload in place of begin_from.
//   g5-cleared  [G5] g4-cleared with restore-into-pool (prefix hint X) as the second upload.
//   g5-clear    [G5] Latch clear: on a fresh pool, the first upload is a restore-into-pool with a null prefix, with
//               no dispatch before it, which reaches the clear branch directly.
// 1,000 is mid-page (B = 16: 62 full pages and an 8-row tail), so both uploads write a tail page; an aligned
// prefix has none and the second upload would not run (rev 10, CA8-N8).
//
// GRADING (Sec7 3.12): each run asserts a clean debug layer -- over the run's subject window (from the first upload
// to the end of the second) and over its grading window -- and the next tokens of every sequence in it byte-equal
// to "a run with a drain between the steps". READING (notes, R-312): that reference is realised as DEDICATED
// twins (sslm_gpu_seq_create, never a pool upload) holding the same tokens, so no mutant of the pool upload can
// reach the reference; a prefix (C) is graded by adopting it onto a fresh budgeted sequence E after a dispatch
// (a latch-set upload) and comparing E's next tokens with the twin. The restored sequences are graded the same
// way against a dedicated twin that ran the same tokens.
//
// THE MUTANTS (Sec7 3.12; each built and shown red on the box, on its run, before 3.12's green counts -- G4 for the
// G4 runs, G5 for the twin): (a) the upload ignores the latch (always UNORDERED_ACCESS -> COPY_DEST), seen on
// g4-set (2) and g4-cleared (1); (b) the inverse (always COPY_SOURCE -> COPY_DEST), seen on g4-cleared (1) and
// g5-clear (1); (c) the upload leaves the latch set, seen only on the set-then-cleared runs. The counts are the
// model's (*executed*); whether the hardware debug layer flags each wrong StateBefore is a hardware fact (D3D12
// buffers decay to COMMON when a list completes), which is why each mutant is commissioned on the box.
//
// RED NOW (63e7b0e): link-red -- sslm_gpu_kv_pool_create, sslm_gpu_kv_pool_destroy, sslm_gpu_seq_create_budgeted
// (G3), sslm_gpu_prefix_begin, _prefill, _freeze, _begin_from, _release, sslm_gpu_seq_adopt_prefix (G4), and in the
// g5 runs sslm_gpu_seq_restore_into_pool (G5). One binary holds all five runs, so the G4 runs are link-red until
// G5 too; the draft build therefore also builds a G4-only binary with -DG1D_3_12_G4_ONLY (see the notes).
// Green: the G4 runs at G4, the g5 runs at G5.
#define G1D_NO_QUEUE_GATE 1
#define G1D_USES_PREFIXES 1
#include "g1d_helpers.h"

namespace {

using g1d::Bench;

constexpr int32_t kPrefixLen = 1000;  // mid-page at B = 16
constexpr int kNext = 4;
constexpr uint32_t kPoolPages = 256;
constexpr int32_t kSmallBudget = 64;
constexpr int64_t kTwinCap = 1100;

struct Run {
	Bench b;
	SslmGpuKvPoolHandle* pool = nullptr;
	SslmGpuPrefixHandle* X = nullptr;
	std::vector<int32_t> prefix_toks;
	int32_t t0 = 0, t1 = 0, q = 0;
	bool ok = true;

	bool Fail(const char* what) {
		CHECK_MSG(false, "SETUP: %s", what);
		ok = false;
		return false;
	}
	bool Open() {
		const std::string model = g1d::Opt("model");
		if (model.empty()) {
			SKIP_MSG("3.12 needs --model=PATH (context cap >= 1100)");
			return ok = false;
		}
		if (!b.Open(model)) return Fail("the model did not load and map");
		if (b.fx.model_cap < kTwinCap) return Fail("the model's context cap is below 1,100");
		if (!g1d::RequireDebugLayer()) return ok = false;
		prefix_toks = g1d::Tokens(b.fx, kPrefixLen, b.fx.vocab / 11);
		t0 = b.fx.SomeToken();
		t1 = b.fx.OtherToken();
		q = (b.fx.vocab / 3 + 5) % b.fx.vocab;
		return true;
	}
	// The run's first dispatch: X's prefill, then freeze (Sec7 3.12: every G4 run starts with a prefill).
	bool MakePrefix(SslmGpuKvPoolHandle* p, SslmGpuPrefixHandle** out) {
		SslmGpuPrefixHandle* x = nullptr;
		if (sslm_gpu_prefix_begin(b.ctx, b.fx.model, p, kPrefixLen + 32, &x) != SSLM_OK || !x)
			return Fail("prefix_begin");
		b.prefixes.push_back(x);
		int32_t consumed = -1;
		const SslmGpuStatus st = sslm_gpu_prefix_prefill(b.ctx, x, prefix_toks.data(), kPrefixLen, /*kind=*/0,
		                                                   b.FullDepthBudget(), &consumed);
		if (st != SSLM_OK || consumed != kPrefixLen) return Fail("prefix_prefill of 1,000 tokens");
		if (sslm_gpu_prefix_freeze(b.ctx, x) != SSLM_OK) return Fail("prefix_freeze");
		*out = x;
		return true;
	}
	bool DecodeReady(SslmGpuSequenceHandle* s, int32_t tok) {
		const SslmGpuStatus d = g1d::SubmitDecode(b, s, tok);
		const SslmGpuStatus r = d == SSLM_OK ? g1d::Ready(b.ctx, s, 1).First() : d;
		CHECK_MSG(r == SSLM_OK, "a pooled decode + sslm_gpu_ready returned %s", StatusName(r));
		return r == SSLM_OK;
	}
	SslmGpuSequenceHandle* Adopt(SslmGpuPrefixHandle* x) {
		SslmGpuSequenceHandle* s = b.Budgeted(pool, kPrefixLen + 64);
		if (!s) {
			Fail("budgeted create for an adopter");
			return nullptr;
		}
		const SslmGpuStatus st = sslm_gpu_seq_adopt_prefix(b.ctx, s, x);
		CHECK_MSG(st == SSLM_OK, "adopt returned %s", StatusName(st));
		return st == SSLM_OK ? s : nullptr;
	}
};

// Twins: dedicated sequences, no pool upload anywhere.
std::vector<int32_t> TwinA(Run& r, bool two_decodes) {
	SslmGpuSequenceHandle* s = r.b.Dedicated(64);
	if (!s || !r.DecodeReady(s, r.t0) || (two_decodes && !r.DecodeReady(s, r.t1))) return {};
	return g1d::NextTokens(r.b, s, r.q, kNext);
}
std::vector<int32_t> TwinPrefix(Run& r, int decodes) {
	SslmGpuSequenceHandle* s = r.b.Dedicated(kTwinCap);
	if (!s || Prefill(r.b.fx, s, r.prefix_toks, 64) != SSLM_OK) return {};
	for (int i = 0; i < decodes; ++i) {
		int32_t tok = -1;
		if (SslmGpuSeqDecodeStepForG5Bridge(r.b.ctx, s, i ? r.t1 : r.t0, r.b.FullDepthBudget(), &tok) != SSLM_OK)
			return {};
	}
	return g1d::NextTokens(r.b, s, r.q, kNext);
}

void Expect(const char* who, const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
	CHECK_MSG(!want.empty() && got == want, "3.12: %s's next tokens [%s] differ from its dedicated twin's [%s]", who,
	          g1d::Join(got).c_str(), g1d::Join(want).c_str());
}

#if !defined(G1D_3_12_G4_ONLY)
// A pooled 'SLM6' blob holding the prefix tokens and two decodes, made in a SECOND context on its own pool, so
// that nothing it runs touches the subject pool's latch.
std::vector<uint8_t> ForeignBlob(Run& r, Bench& b2) {
	if (!b2.Open(r.b.fx.path)) return {};
	SslmGpuKvPoolHandle* p2 = b2.Pool(kPoolPages);
	if (!p2) return {};
	SslmGpuPrefixHandle* x2 = nullptr;
	int32_t consumed = -1;
	if (sslm_gpu_prefix_begin(b2.ctx, b2.fx.model, p2, kPrefixLen + 32, &x2) != SSLM_OK || !x2) return {};
	b2.prefixes.push_back(x2);
	if (sslm_gpu_prefix_prefill(b2.ctx, x2, r.prefix_toks.data(), kPrefixLen, 0, b2.FullDepthBudget(), &consumed) != SSLM_OK ||
	    sslm_gpu_prefix_freeze(b2.ctx, x2) != SSLM_OK)
		return {};
	SslmGpuSequenceHandle* s2 = b2.Budgeted(p2, kPrefixLen + 64);
	if (!s2 || sslm_gpu_seq_adopt_prefix(b2.ctx, s2, x2) != SSLM_OK) return {};
	for (int i = 0; i < 2; ++i) {
		int32_t tok = -1;
		if (SslmGpuSeqDecodeStepForG5Bridge(b2.ctx, s2, i ? r.t1 : r.t0, b2.FullDepthBudget(), &tok) != SSLM_OK) return {};
	}
	return g1d::SaveBlob(b2.ctx, s2);
}
SslmGpuSequenceHandle* RestoreIntoPool(Run& r, const std::vector<uint8_t>& blob, SslmGpuPrefixHandle* hint) {
	SslmGpuSequenceHandle* s = nullptr;
	uint32_t shared = 0;
	const SslmGpuStatus st =
	    sslm_gpu_seq_restore_into_pool(r.b.ctx, r.b.fx.model, r.pool, blob.data(), blob.size(), hint, &s, &shared);
	CHECK_MSG(st == SSLM_OK && s, "restore_into_pool returned %s", StatusName(st));
	if (s) r.b.seqs.push_back(s);
	std::printf("    restore_into_pool: %s, shared pages %u\n", StatusName(st), shared);
	return st == SSLM_OK ? s : nullptr;
}
#endif

void RunG4OrTwin(bool latch_set, bool twin) {
	Run r;
	if (!r.Open()) {
		r.b.Close();
		return;
	}
#if !defined(G1D_3_12_G4_ONLY)
	Bench b2;
	std::vector<uint8_t> blob;
	if (twin) {
		blob = ForeignBlob(r, b2);
		if (blob.empty()) {
			r.Fail("the foreign SLM6 blob (second context)");
			b2.Close();
			r.b.Close();
			return;
		}
	}
#else
	(void)twin;
#endif
	r.pool = r.b.Pool(kPoolPages);
	if (!r.pool || !r.MakePrefix(r.pool, &r.X)) {
		r.b.Close();
		return;
	}
	SslmGpuSequenceHandle* A = r.b.Budgeted(r.pool, kSmallBudget);
	if (!A) {
		r.Fail("pooled A");
		r.b.Close();
		return;
	}
	(void)g1d::DebugMessages();
	g1d::SetPhase("3.12: subject window");
	bool ok = r.DecodeReady(A, r.t0);
	SslmGpuSequenceHandle* B = ok ? r.Adopt(r.X) : nullptr;  // upload 1 (latch set: A's decode dispatched)
	ok = ok && B;
	if (ok && latch_set) ok = r.DecodeReady(A, r.t1);  // the dispatch between the uploads sets the latch
	SslmGpuPrefixHandle* C = nullptr;
	SslmGpuSequenceHandle* R = nullptr;
	if (ok && !twin) {
		const SslmGpuStatus st = sslm_gpu_prefix_begin_from(r.b.ctx, r.X, 32, &C);  // upload 2
		CHECK_MSG(st == SSLM_OK && C, "begin_from returned %s", StatusName(st));
		if (C) r.b.prefixes.push_back(C);
		ok = st == SSLM_OK && C;
	}
#if !defined(G1D_3_12_G4_ONLY)
	if (ok && twin) {
		R = RestoreIntoPool(r, blob, r.X);  // upload 2
		ok = R != nullptr;
	}
#endif
	const size_t subject_msgs = g1d::DebugMessages();
	std::printf("    subject window: %zu debug-layer message(s)\n", subject_msgs);
	CHECK_MSG(subject_msgs == 0, "3.12 %s%s: %zu debug-layer message(s) across the two uploads", twin ? "twin " : "",
	          latch_set ? "latch set" : "set then cleared", subject_msgs);
	if (!ok) {
		r.b.Close();
		return;
	}

	g1d::SetPhase("3.12: grading window");
	// The grading of A first (a dispatch), so E's adopt of C below is a latch-set upload.
	Expect("A", g1d::NextTokens(r.b, A, r.q, kNext), TwinA(r, latch_set));
	Expect("B", g1d::NextTokens(r.b, B, r.q, kNext), TwinPrefix(r, 0));
	if (C) {
		// Sec3.9 A17: only a frozen prefix is adoptable. Freeze submits nothing (no dispatch, no upload), so the
		// adopt below is still the latch-set upload the construction intends.
		const SslmGpuStatus fz = sslm_gpu_prefix_freeze(r.b.ctx, C);
		CHECK_MSG(fz == SSLM_OK, "freeze(C) returned %s", StatusName(fz));
		SslmGpuSequenceHandle* E = fz == SSLM_OK ? r.Adopt(C) : nullptr;
		if (E) Expect("C (through E's adopt)", g1d::NextTokens(r.b, E, r.q, kNext), TwinPrefix(r, 0));
	}
	if (R) Expect("the restored sequence", g1d::NextTokens(r.b, R, r.q, kNext), TwinPrefix(r, 2));
	const size_t grade_msgs = g1d::DebugMessages();
	CHECK_MSG(grade_msgs == 0, "3.12: %zu debug-layer message(s) in the grading window", grade_msgs);
	r.b.Close();
#if !defined(G1D_3_12_G4_ONLY)
	b2.Close();
#endif
}

#if !defined(G1D_3_12_G4_ONLY)
void RunG5Clear() {
	Run r;
	if (!r.Open()) {
		r.b.Close();
		return;
	}
	Bench b2;
	const std::vector<uint8_t> blob = ForeignBlob(r, b2);
	if (blob.empty()) {
		r.Fail("the foreign SLM6 blob (second context)");
		b2.Close();
		r.b.Close();
		return;
	}
	r.pool = r.b.Pool(kPoolPages);  // fresh: no dispatch has touched it
	if (!r.pool) {
		r.Fail("pool");
		b2.Close();
		r.b.Close();
		return;
	}
	(void)g1d::DebugMessages();
	g1d::SetPhase("3.12: latch clear, the first upload is a restore");
	SslmGpuSequenceHandle* R = RestoreIntoPool(r, blob, nullptr);
	const size_t subject_msgs = g1d::DebugMessages();
	std::printf("    subject window: %zu debug-layer message(s)\n", subject_msgs);
	CHECK_MSG(subject_msgs == 0, "3.12 latch clear: %zu debug-layer message(s) on the first upload of a fresh pool",
	          subject_msgs);
	if (R) Expect("the restored sequence", g1d::NextTokens(r.b, R, r.q, kNext), TwinPrefix(r, 2));
	const size_t grade_msgs = g1d::DebugMessages();
	CHECK_MSG(grade_msgs == 0, "3.12: %zu debug-layer message(s) in the grading window", grade_msgs);
	r.b.Close();
	b2.Close();
}
#endif

}  // namespace

int main(int argc, char** argv) {
	g1d::ParseOpts(argc, argv);
	g1d::StartWatchdog(120);
	const std::string run = g1d::Opt("run", "g4-set");
	std::printf("cell 3.12 run=%s\n", run.c_str());
	if (run == "g4-set") RunG4OrTwin(true, false);
	else if (run == "g4-cleared") RunG4OrTwin(false, false);
#if !defined(G1D_3_12_G4_ONLY)
	else if (run == "g5-set") RunG4OrTwin(true, true);
	else if (run == "g5-cleared") RunG4OrTwin(false, true);
	else if (run == "g5-clear") RunG5Clear();
#endif
	else CHECK_MSG(false, "unknown --run=%s (g4-set|g4-cleared|g5-set|g5-cleared|g5-clear)", run.c_str());
	g1d::PhaseDone();
	return FinishSuite("g1d_cell_3_12");
}
