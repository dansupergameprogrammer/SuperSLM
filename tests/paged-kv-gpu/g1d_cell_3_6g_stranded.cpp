// Paged KV GPU slice, plan step G1 -- cell 3.6g's STRANDED and PER-CONTEXT runs, and the NINE STRANDED PINS of
// Sec4.4 *The ledger on the fault paths* (plan.md Sec7 3.6g; Sec4.4 *The submission ledger* `Submit(retry)`, the
// HOOK/HOOK-legs/W5 sentences; decision 73; G3's gate in Sec9: "the stranded pins ... the nine rule-0 legs of
// Sec4.4, each at its device's expectation ... each in a fresh process"). Bracket: [G3]. Carrier rows HOOK,
// HOOK-legs, PINS and W5 land here.
//
// BUILD: the allocation-fault build (cmake -DSUPERSLM_GPU_TEST_SEAMS=ON, i.e. SUPERSLM_GPU_ALLOC_FAULT_INJECTION),
// linked with tests/te425-gpu-alloc-status-red-suite/te425_cells.cpp compiled with /Dmain=te425_cells_main, for
// te425's InstallHooks (te425_cells.cpp:312) -- HOOK-legs: "runs in te425's executable, or in a binary that
// links te425's InstallHooks".
//
// EVERY RUN IS ONE --leg=, ALONE IN A FRESH PROCESS: a stranded ledger is terminal for its process (te425's
// `isolate` flag, the same rule). The runner starts one process per leg.
//
// Legs (--leg=):
//   3.6g.signal     -- Sec7 3.6g stranded: te425's M_SIGNAL hook, first = 1 (the call's first queue Signal),
//                      n = 2 (Submit()'s Signal and its retry), E_FAIL, on the prompt loop. No gate (the text
//                      gates only the SetEvent run; a failed Signal needs none).
//   3.6g.setevent   -- Sec7 3.6g stranded: te425's M_SETEVENT hook, first = 1, E_FAIL, on the prompt loop, with
//                      the queue gate held (without it the fence has already passed and the hook never fires).
//       Both 3.6g runs: the call returns SSLM_DEVICE_LOST; the next three singleton-reaching guarded calls each
//       return SSLM_DEVICE_LOST (terminal, never SSLM_BUSY); an internal-layer reset (RunLayerLoopGpu) throws
//       GpuSubmissionStrandedError. --model must be square (R15) for the internal reset.
//   3.6g.percontext -- Sec7 3.6g per-context: after a clean sequence create (the context's own device resets and
//                      runs), C.signal1.oom's construction strands that device (Submit(no retry) strands at
//                      once); its next reset must throw GpuSubmissionStrandedError. READING (notes, item R-PC):
//                      that reset is reached only through the C ABI (ctx->device is private to gpu_1p0.cpp, and
//                      every *ForTest entry point runs inside InvokeGpuApiBoundary), so the exception is not
//                      observable from a test; the operand is its classified status, SSLM_DEVICE_LOST through
//                      ClassifyHandleFault, on that context's next two device calls (sequence create, model map),
//                      with the singleton unaffected (a second context's prompt loop SSLM_OK and byte-equal).
//   The nine pins, each at its device's expectation (Sec4.4):
//     singleton  -- P.setevent1.oom, P.setevent1.fail, P.setevent2.fail, D.setevent1.fail, R.signal2.oom,
//                   S.setevent1.fail: SSLM_DEVICE_LOST from the stranding call, and from G3 from every later
//                   singleton-reaching guarded call (the cell makes three: a decode step, a prompt prefill, a
//                   restore, each on a second context, each set up before the strand);
//     per-context -- C.signal1.oom, M.signal1.oom, L.setevent1.oom: SSLM_DEVICE_LOST from the stranding call and
//                   from that context's later device calls (sequence create, model map), the singleton not
//                   stranded (a second context's prompt loop SSLM_OK and byte-equal to its pre-strand frame).
//   Routes and artifacts are te425's (te425_cells.cpp:1252-1365): P = U3's prompt loop (5 tokens, 2 sub-chunks),
//   D = embed + full-depth decode step + ready(block=1), R = sslm_gpu_seq_restore of a saved 5-token sequence,
//   S = the schema loop on the G5 artifact ("shopkeeper_intent_extraction"), C = sequence create, M = model map at
//   flags 0, L = the device-logits finish on the R15 artifact mapped HEAD_ON_DEVICE.
//
// HOW EACH HOOK IS ARMED (HOOK-legs, HOOK, W5):
//   - first = 1 is the call's first matching call; the gate is armed only after any trace the leg records.
//   - Ungated legs (the Signal legs, and L.setevent1.oom on a per-context device the gate cannot hold) resolve
//     their ordinal against a clean trace of the very call, as te425 does: the ordinal is the selector's nth,
//     and a clean trace with fewer matching calls is INVALID. L.setevent1.oom is timing-dependent at source
//     (Sec4.4): an INVALID reading counts as a G3 failure, so INVALID here exits non-zero.
//   - Gated singleton SetEvent legs record no trace: with the gate held every wait on singleton work arms its
//     event, so the ordinal is the wait's index in the call.
//   - W5 (P.setevent2.fail, nth = 2): with the gate held the second wait is never reached. READING TAKEN: the
//     gate is released after the first wait is armed (te425's g_calls[M_SETEVENT] rises), and re-armed at once,
//     so the second sub-chunk's submission is gated and its wait arms the hooked SetEvent. This assumes G3's
//     ArmQueueGateForTest() after a release holds the NEXT submission (a fresh gate value). The re-arm races
//     the library's recording of sub-chunk 2 (milliseconds of recording against a microsecond re-arm); a lost
//     race leaves the hook unfired, which is INVALID and fails the run, as for L.setevent1.oom.
//   - A hook that never fires is INVALID and fails the run (te425: "never fired -> INVALID").
//
// THE MUTANTS these runs separate (Sec7 3.6g): rev 8 as written (SSLM_BUSY for ever after a stranding: the
// follow-up calls read SSLM_BUSY), and `unsignaled_` never set (a reset under the stranded list: the internal
// reset does not throw GpuSubmissionStrandedError, and the follow-ups are not refused).
//
// RED NOW (63e7b0e): link-red -- superslm_gpu::ArmQueueGateForTest, ReleaseQueueGateForTest (G3). Beyond the
// link: at the base the follow-up calls after a singleton strand are not terminal (decision 73 is G3's).
// Green at G3.
#define G1D_WITH_TE425_HOOKS 1
#include "g1d_helpers.h"

#include <cstring>

namespace {

using g1d::Bench;
using g1d::QueueGate;
using g1d::Thrown;

enum Route { RT_PROMPT, RT_DECODE, RT_RESTORE, RT_SCHEMA, RT_SEQCREATE, RT_MAP, RT_DEVLOGITS };
enum Dev { DEV_SINGLETON, DEV_CONTEXT };

struct Leg {
	const char* name;
	Route route;
	int method;
	uint32_t nth;
	long hr;
	uint32_t n;
	Dev dev;
	bool gated;
	bool internal_reset;  // 3.6g's "an internal-layer reset throws GpuSubmissionStrandedError"
	const char* artifact;  // which artifact the runner passes as --model
};

constexpr long OOM = E_OUTOFMEMORY;
constexpr long FAIL = E_FAIL;

const Leg kLegs[] = {
    // Sec7 3.6g
    {"3.6g.signal", RT_PROMPT, te425::M_SIGNAL, 1, FAIL, 2, DEV_SINGLETON, false, true, "r15"},
    {"3.6g.setevent", RT_PROMPT, te425::M_SETEVENT, 1, FAIL, 1, DEV_SINGLETON, true, true, "r15"},
    {"3.6g.percontext", RT_SEQCREATE, te425::M_SIGNAL, 1, OOM, 1, DEV_CONTEXT, false, false, "qwen3"},
    // Sec4.4's nine stranded pins (te425_cells.cpp:1289-1364)
    {"P.setevent1.oom", RT_PROMPT, te425::M_SETEVENT, 1, OOM, 1, DEV_SINGLETON, true, false, "qwen3"},
    {"P.setevent1.fail", RT_PROMPT, te425::M_SETEVENT, 1, FAIL, 1, DEV_SINGLETON, true, false, "qwen3"},
    {"P.setevent2.fail", RT_PROMPT, te425::M_SETEVENT, 2, FAIL, 1, DEV_SINGLETON, true, false, "qwen3"},
    {"D.setevent1.fail", RT_DECODE, te425::M_SETEVENT, 1, FAIL, 1, DEV_SINGLETON, true, false, "qwen3"},
    {"R.signal2.oom", RT_RESTORE, te425::M_SIGNAL, 2, OOM, 1, DEV_SINGLETON, false, false, "qwen3"},
    {"S.setevent1.fail", RT_SCHEMA, te425::M_SETEVENT, 1, FAIL, 1, DEV_SINGLETON, true, false, "g5"},
    {"C.signal1.oom", RT_SEQCREATE, te425::M_SIGNAL, 1, OOM, 1, DEV_CONTEXT, false, false, "qwen3"},
    {"M.signal1.oom", RT_MAP, te425::M_SIGNAL, 1, OOM, 1, DEV_CONTEXT, false, false, "qwen3"},
    {"L.setevent1.oom", RT_DEVLOGITS, te425::M_SETEVENT, 1, OOM, 1, DEV_CONTEXT, false, false, "r15"},
};

const char* kSchemaName = "shopkeeper_intent_extraction";

std::vector<int32_t> SchemaChain(const GpuModelFixture& fx, int n) {
	IndependentSchema s;
	std::vector<int32_t> c;
	if (!s.Build(fx.bytes, kSchemaName)) return c;
	uint32_t st = 0;
	for (int k = 0; k < n; ++k) {
		const int32_t t = s.FirstLegal(st);
		uint32_t nx = 0;
		if (t < 0 || !s.Next(st, t, &nx)) break;
		c.push_back(t);
		st = nx;
	}
	return c;
}

// The leg's stack: context 1 (the call under test), context 2 (the follow-ups and the singleton's health).
struct Fx {
	Bench b1, b2;
	SslmGpuSequenceHandle* s1 = nullptr;  // ctx 1's sequence
	SslmGpuSequenceHandle* a2 = nullptr;  // ctx 2: embedded, for the follow-up decode
	SslmGpuSequenceHandle* p2 = nullptr;  // ctx 2: for the follow-up prompt loop
	std::vector<int32_t> toks, chain;
	std::vector<uint8_t> blob1, blob2;
	int32_t schema_idx = -1;
	te425::Frame ref2, f;
	SslmGpuStatus status = SSLM_OK;
	std::string desc;

	bool Open(const Leg& L, const std::string& path) {
		const uint32_t flags = L.route == RT_DEVLOGITS ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u;
		if (!b1.Open(path, flags) || !b2.Open(path)) return false;
		s1 = b1.Dedicated(64);
		a2 = b2.Dedicated(64);
		p2 = b2.Dedicated(64);
		SslmGpuSequenceHandle* c2 = b2.Dedicated(64);
		if (!s1 || !a2 || !p2 || !c2) return false;
		const int32_t vocab = b1.fx.vocab;
		if (vocab > 151643) toks = te425::QwenPrompt(5);
		else
			for (int i = 0; i < 5; ++i) toks.push_back((vocab / 7 + 37 * i) % vocab);
		ref2.Size(static_cast<uint32_t>(b2.fx.hidden));
		f.Size(static_cast<uint32_t>(b2.fx.hidden));
		// ctx 2's pre-strand frame, the follow-up decode's embed, and the follow-up restore's blob.
		if (te425::PromptLoop(b2.ctx, p2, toks, &ref2).First() != SSLM_OK) return false;
		if (sslm_gpu_seq_embed_token(b2.ctx, a2, b2.fx.SomeToken()) != SSLM_OK) return false;
		if (SslmGpuSeqPrefillPromptForG5Bridge(b2.ctx, c2, toks.data(), 5, 64) != SSLM_OK) return false;
		blob2 = g1d::SaveBlob(b2.ctx, c2);
		if (blob2.empty()) return false;
		if (L.route == RT_RESTORE) {
			if (SslmGpuSeqPrefillPromptForG5Bridge(b1.ctx, s1, toks.data(), 5, 64) != SSLM_OK) return false;
			blob1 = g1d::SaveBlob(b1.ctx, s1);
			if (blob1.empty()) return false;
		}
		if (L.route == RT_SCHEMA) {
			chain = SchemaChain(b1.fx, 5);
			schema_idx = SslmGpuSchemaLookupForG5Bridge(b1.fx.model, kSchemaName);
			if (chain.size() != 5 || schema_idx < 0) {
				std::printf("SETUP: the schema chain (%zu) or index (%d) on %s\n", chain.size(), schema_idx, path.c_str());
				return false;
			}
		}
		return true;
	}
	void Close() {
		b2.Close();
		b1.Close();
	}
	bool Pre(Route r) {
		switch (r) {
			case RT_DECODE:
			case RT_DEVLOGITS:
				return sslm_gpu_seq_reset(b1.ctx, s1) == SSLM_OK &&
				       SslmGpuSeqPrefillPromptForG5Bridge(b1.ctx, s1, toks.data(), 5, 64) == SSLM_OK;
			default: return true;
		}
	}
	void Call(Route r) {
		status = SSLM_OK;
		switch (r) {
			case RT_PROMPT: {
				te425::Frame fr;
				fr.Size(static_cast<uint32_t>(b1.fx.hidden));
				status = te425::PromptLoop(b1.ctx, s1, toks, &fr).First();
				break;
			}
			case RT_DECODE: {
				const SslmGpuStatus e = sslm_gpu_seq_embed_token(b1.ctx, s1, 785 % b1.fx.vocab);
				const SslmGpuStatus s = e == SSLM_OK ? sslm_decode_step_gpu(b1.ctx, s1, nullptr, 0xFFFFFFFFu) : e;
				status = s == SSLM_OK ? g1d::Ready(b1.ctx, s1, 1).First() : s;
				break;
			}
			case RT_RESTORE: {
				SslmGpuSequenceHandle* h = nullptr;
				status = sslm_gpu_seq_restore(b1.ctx, b1.fx.model, blob1.data(), blob1.size(), &h);
				if (h) sslm_gpu_seq_release(b1.ctx, h);
				break;
			}
			case RT_SCHEMA: {
				SslmGpuStatus st = sslm_gpu_seq_reset(b1.ctx, s1);
				if (st == SSLM_OK) st = SslmGpuSeqSetSchemaForG5Bridge(b1.ctx, s1, schema_idx);
				int32_t consumed = -1;
				if (st == SSLM_OK)
					st = SslmGpuSeqPrefillSchemaContentForG5Bridge(b1.ctx, s1, chain.data(),
					                                               static_cast<int32_t>(chain.size()), 64, &consumed);
				if (st == SSLM_OK) {
					std::vector<int8_t> codes(b1.fx.hidden);
					size_t req = 0;
					int64_t m = 0, e = 0;
					st = sslm_gpu_seq_read_prefill_final_hidden(b1.ctx, s1, codes.data(), codes.size(), &req, &m, &e);
				}
				status = st;
				break;
			}
			case RT_SEQCREATE: {
				SslmGpuSequenceHandle* h = nullptr;
				status = sslm_gpu_seq_create(b1.ctx, b1.fx.model, 64, &h);
				if (h) sslm_gpu_seq_release(b1.ctx, h);
				break;
			}
			case RT_MAP: {
				SslmGpuModelHandle* h = nullptr;
				status = sslm_gpu_model_map(b1.ctx, &b1.fx.view, GpuResidencyConfig{}, &h);
				if (h) sslm_gpu_model_unmap(b1.ctx, h);
				break;
			}
			case RT_DEVLOGITS: {
				int32_t tok = -1;
				status = SslmGpuSeqFinishTokenForG5Bridge(b1.ctx, s1, &tok);
				break;
			}
		}
	}
};

uint32_t CountMethod(const std::vector<te425::TraceEv>& t, int m) {
	uint32_t c = 0;
	for (const auto& e : t) c += e.method == m;
	return c;
}

void RunLeg(const Leg& L) {
	const std::string path = g1d::Opt("model");
	if (path.empty()) {
		CHECK_MSG(false, "SETUP: --model=PATH (this leg's artifact: %s)", L.artifact);
		return;
	}
	std::string why;
	if (!te425::InstallHooks(&why)) {
		CHECK_MSG(false, "SETUP (INVALID): te425 hooks did not install: %s", why.c_str());
		return;
	}
	Fx fx;
	g1d::SetPhase("setup");
	if (!fx.Open(L, path)) {
		CHECK_MSG(false, "SETUP: the leg's fixture on %s (want the %s artifact)", path.c_str(), L.artifact);
		fx.Close();
		return;
	}
	g1d::InternalLoop il;
	if (L.internal_reset && !il.Build(fx.b1.fx.view, fx.b1.fx.has_qk_norm)) {
		CHECK_MSG(false, "SETUP: the internal-layer fixture (a square model) did not build");
		fx.Close();
		return;
	}
	// 3.6g per-context: "after a clean call a context's device resets and runs".
	g1d::SetPhase("clean pre-call");
	if (!fx.Pre(L.route)) {
		CHECK_MSG(false, "INVALID: the leg's pre-call is not clean");
		fx.Close();
		return;
	}
	uint32_t ordinal = L.nth;
	if (!L.gated) {
		te425::TraceStart();
		fx.Call(L.route);
		const std::vector<te425::TraceEv> t = te425::TraceStop();
		const uint32_t have = CountMethod(t, L.method);
		std::printf("    clean trace: %s; %u matching call(s)\n", StatusName(fx.status), have);
		if (fx.status != SSLM_OK || have < L.nth) {
			CHECK_MSG(false, "INVALID: the clean trace returned %s with %u matching call(s), want SSLM_OK and >= %u%s",
			          StatusName(fx.status), have, L.nth,
			          L.route == RT_DEVLOGITS ? " (L.setevent1.oom is timing-dependent; INVALID is a G3 failure)" : "");
			fx.Close();
			return;
		}
		if (!fx.Pre(L.route)) {
			CHECK_MSG(false, "INVALID: the leg's second pre-call is not clean");
			fx.Close();
			return;
		}
	} else {
		fx.Call(L.route);  // the clean call, ungated, untraced
		CHECK_MSG(fx.status == SSLM_OK, "INVALID: the clean call returned %s", StatusName(fx.status));
		if (fx.status != SSLM_OK || !fx.Pre(L.route)) {
			fx.Close();
			return;
		}
	}

	// The stranding call.
	QueueGate gate;
	g1d::SetPhase("the stranding call");
	if (L.gated) gate.Arm();  // after any trace (HOOK-legs)
	te425::HookArmCall(L.method, ordinal, L.hr, te425::HB_FAIL_NO_CALL, L.n);
	if (L.gated && L.nth >= 2) {
		// W5: release after the first wait is armed, re-arm at once, so the nth wait is gated too.
		const uint32_t c0 = te425::g_calls[L.method].load();
		std::atomic<bool> done{false};
		std::thread call([&] {
			fx.Call(L.route);
			done.store(true);
		});
		uint32_t waits_seen = 0;
		while (!done.load() && waits_seen + 1 < L.nth) {
			if (te425::g_calls[L.method].load() - c0 > waits_seen) {
				++waits_seen;
				gate.Release();
				gate.Arm();
			} else {
				std::this_thread::yield();
			}
		}
		call.join();
	} else {
		fx.Call(L.route);
	}
	const uint32_t fired = te425::g_arm.fired.load();
	te425::HookDisarm();
	const SslmGpuStatus own = fx.status;
	std::printf("    %s: own=%s hook %s#%u fired %u (want %u)\n", L.name, StatusName(own), te425::MethodName(L.method),
	            ordinal, fired, L.n);
	if (fired != L.n) {
		CHECK_MSG(false, "INVALID: the hook fired %u time(s), want %u -- the leg is not a verdict, and INVALID fails "
		                 "the run",
		          fired, L.n);
		fx.Close();
		return;
	}
	CHECK_MSG(own == SSLM_DEVICE_LOST, "%s: the stranding call returned %s, want SSLM_DEVICE_LOST (rule 0)", L.name,
	          StatusName(own));

	if (L.dev == DEV_SINGLETON) {
		// Terminal: the next three singleton-reaching guarded calls, each on context 2.
		g1d::SetPhase("follow-up 1: decode step on a second context");
		const SslmGpuStatus f1 = sslm_decode_step_gpu(fx.b2.ctx, fx.a2, nullptr, 0xFFFFFFFFu);
		g1d::SetPhase("follow-up 2: prompt prefill on a second context");
		const SslmGpuStatus f2 = SslmGpuSeqPrefillPromptForG5Bridge(fx.b2.ctx, fx.p2, fx.toks.data(), 5, 64);
		g1d::SetPhase("follow-up 3: restore on a second context");
		SslmGpuSequenceHandle* h = nullptr;
		const SslmGpuStatus f3 = sslm_gpu_seq_restore(fx.b2.ctx, fx.b2.fx.model, fx.blob2.data(), fx.blob2.size(), &h);
		if (h) sslm_gpu_seq_release(fx.b2.ctx, h);
		std::printf("    follow-ups: decode=%s prefill=%s restore=%s\n", StatusName(f1), StatusName(f2), StatusName(f3));
		CHECK_MSG(f1 == SSLM_DEVICE_LOST && f2 == SSLM_DEVICE_LOST && f3 == SSLM_DEVICE_LOST,
		          "%s: after a singleton strand the next singleton-reaching guarded calls returned decode=%s prefill=%s "
		          "restore=%s, want SSLM_DEVICE_LOST from each (terminal; SSLM_BUSY is rev 8's stranded-Busy mutant)",
		          L.name, StatusName(f1), StatusName(f2), StatusName(f3));
		if (L.internal_reset) {
			g1d::SetPhase("an internal-layer reset");
			std::string what;
			const Thrown t = g1d::CallInternal([&il] { (void)il.RunSync(); }, &what);
			CHECK_MSG(t == Thrown::Stranded,
			          "%s: an internal-layer reset let %s reach the test, want GpuSubmissionStrandedError (%s)", L.name,
			          g1d::ThrownName(t), what.c_str());
		}
	} else {
		// That context's later calls on its device, and the singleton unaffected.
		g1d::SetPhase("per-context: that context's next device calls");
		SslmGpuSequenceHandle* h = nullptr;
		const SslmGpuStatus c1 = sslm_gpu_seq_create(fx.b1.ctx, fx.b1.fx.model, 64, &h);
		if (h) sslm_gpu_seq_release(fx.b1.ctx, h);
		SslmGpuModelHandle* m = nullptr;
		const SslmGpuStatus c2 = sslm_gpu_model_map(fx.b1.ctx, &fx.b1.fx.view, GpuResidencyConfig{}, &m);
		if (m) sslm_gpu_model_unmap(fx.b1.ctx, m);
		std::printf("    stranded context: seq_create=%s model_map=%s\n", StatusName(c1), StatusName(c2));
		CHECK_MSG(c1 == SSLM_DEVICE_LOST && c2 == SSLM_DEVICE_LOST,
		          "%s: the stranded context's next device calls returned seq_create=%s model_map=%s, want "
		          "SSLM_DEVICE_LOST (ClassifyHandleFault on a stranded per-context device)",
		          L.name, StatusName(c1), StatusName(c2));
		g1d::SetPhase("per-context: the singleton is not stranded");
		const SslmGpuStatus s = te425::PromptLoop(fx.b2.ctx, fx.p2, fx.toks, &fx.f).First();
		CHECK_MSG(s == SSLM_OK && fx.f.Same(fx.ref2),
		          "%s: a second context's prompt loop after a per-context strand returned %s (equal=%d), want SSLM_OK "
		          "and byte-equal -- the singleton must not be stranded",
		          L.name, StatusName(s), fx.f.Same(fx.ref2) ? 1 : 0);
	}
	gate.Release();
	fx.Close();
}

}  // namespace

int main(int argc, char** argv) {
	g1d::ParseOpts(argc, argv);
	g1d::StartWatchdog(60);
	const std::string leg = g1d::Opt("leg");
	if (g1d::HasOpt("list")) {
		for (const Leg& L : kLegs)
			std::printf("LEG %s artifact=%s gated=%d dev=%s\n", L.name, L.artifact, L.gated ? 1 : 0,
			            L.dev == DEV_SINGLETON ? "singleton" : "context");
		return 0;
	}
	const Leg* sel = nullptr;
	for (const Leg& L : kLegs)
		if (leg == L.name) sel = &L;
	std::printf("cell 3.6g stranded / stranded pins: leg=%s\n", leg.c_str());
	if (!sel) CHECK_MSG(false, "unknown --leg=%s (--list prints the legs)", leg.c_str());
	else RunLeg(*sel);
	g1d::PhaseDone();
	return FinishSuite("g1d_cell_3_6g_stranded");
}
