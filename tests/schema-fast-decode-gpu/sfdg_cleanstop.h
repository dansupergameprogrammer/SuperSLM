// Engine 1.19, the clean stop at the schema's end: the GPU cells G1-G9 of the design note's §4 (the contract's pins) and G10,
// one cell per executable (sfdg_cell_cs_g1.cpp .. sfdg_cell_cs_g10.cpp), built beside the schema fast-decode GPU cells
// by build_sfd_gpu.bat and build_sfd_gpu_mingw.sh (slice CS), on F-SFD. Box only: the cells need a D3D12 device; the
// cloud compiles and links them.
//
// THE CONTRACT (no ABI change). Every GPU path reports the schema's end as -2 inside an SSLM_OK call: the decode
// bridge, the finish bridge, and sslm_gpu_seq_rows_produced after an EXACT_DRAFT close. The walk is kept, so the
// accepting query read after the -2 classifies it:
//   accepting = 1  the schema is complete: a clean stop. Retryable and idempotent; savable, restorable, resettable.
//   accepting = 0  a dead end: a defect state the caller reports as an error.
// The cells read accepting from both GPU queries (SslmGpuSeqSchemaAcceptingForG5Bridge and sslm_gpu_seq_forced_tokens'
// out_accepting), which must agree.
#ifndef SSLM_SFDG_CLEANSTOP_H
#define SSLM_SFDG_CLEANSTOP_H

#include "sfdg_helpers.h"

namespace sfdg {
namespace cs {

struct Case {
	const char* schema;
	const char* prompt_text;
};
constexpr Case kCases[] = {{kPotion, kPromptPotionText}, {kPromptResult, kPromptPrText}};
constexpr int32_t kExactDraft = 2;    // SSLM_GPU_ROWS_EXACT_DRAFT
constexpr int32_t kSpecial = 151643;  // <|endoftext|>: byteless, never legal under a schema mask

// Both GPU accepting reads; -1 when either call failed, -9 when they disagree.
inline int32_t Accepting(SslmGpuContext* ctx, SslmGpuSequenceHandle* h) {
	int32_t a = -1, count = -1, acc = -1;
	const GS s1 = SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, h, &a);
	const GS s2 = sslm_gpu_seq_forced_tokens(ctx, h, nullptr, 0, &count, &acc);
	if (s1 != GS::SSLM_OK || s2 != GS::SSLM_OK) return -1;
	return a == acc ? a : -9;
}
// Both CPU accepting reads (sslm_stats, sslm_seq_forced_tokens), the same way.
inline int32_t CpuAccepting(sslm_model m, sslm_seq s) {
	sslm_stats_out so{};
	int32_t count = -1, acc = -1;
	const sslm_status a = sslm_stats(m, s, &so);
	const sslm_status b = sslm_seq_forced_tokens(m, s, nullptr, 0, &count, &acc);
	if (a != SSLM_OK || b != SSLM_OK) return -1;
	return so.schema_accepting == acc ? acc : -9;
}

// A GPU run to the schema's end. `outs` is every call's outcome; `stream` the produced tokens. At the first -2:
// the walk and context the -2 call found and left, the ready flag, the blob and accepting after it.
struct Run {
	bool dead = false;
	GS st = GS::SSLM_OK;
	std::vector<int32_t> stream;
	uint32_t walk_before = 0, walk_after = 0;
	int64_t ctx_before = -1, ctx_after = -1;
	int32_t ready_after = -1, accepting = -1;
	int slices = 0;  // the sliced run's sslm_decode_step_gpu submissions
	std::vector<uint8_t> end_blob;
};
inline void RecordEnd(GRig& rig, GSeq& s, Run* r, uint32_t walk0, int64_t ctx0) {
	r->dead = true;
	r->walk_before = walk0;
	r->ctx_before = ctx0;
	r->walk_after = SslmGpuSeqWalkStateForG5Bridge(s.h);
	r->ctx_after = CtxLen(s.h);
	r->ready_after = ReadyFlag(rig.ctx(), s.h);
	r->accepting = Accepting(rig.ctx(), s.h);
	r->end_blob = rig.Blob(s.h);
}
// Whole tokens through the decode bridge (the recommended one-call-per-step entry point).
inline Run RunWhole(GRig& rig, GSeq& s, int max = 400) {
	Run r;
	for (int i = 0; i < max; ++i) {
		const uint32_t walk0 = SslmGpuSeqWalkStateForG5Bridge(s.h);
		const int64_t ctx0 = CtxLen(s.h);
		const int32_t t = Token(rig.env, s, &r.st);
		if (r.st != GS::SSLM_OK) return r;
		if (t == -2) {
			RecordEnd(rig, s, &r, walk0, ctx0);
			return r;
		}
		if (t < 0) return r;
		r.stream.push_back(t);
	}
	return r;
}
// Sliced: each token after the ready one is embedded, driven one layer per sslm_decode_step_gpu submission (drained
// after each), and finished by the finish bridge. The first token (the prompt left the ready flag set) finishes
// through the decode bridge's ready shortcut.
inline Run RunSliced(GRig& rig, GSeq& s, int max = 400) {
	Run r;
	const uint32_t L = rig.env.fx.layers;
	for (int i = 0; i < max; ++i) {
		const uint32_t walk0 = SslmGpuSeqWalkStateForG5Bridge(s.h);
		const int64_t ctx0 = CtxLen(s.h);
		int32_t t = -1;
		GS st = GS::SSLM_OK;
		if (ReadyFlag(rig.ctx(), s.h) == 1) {
			st = SslmGpuSeqDecodeStepForG5Bridge(rig.ctx(), s.h, s.last, rig.env.full_budget, &t);
		} else {
			st = sslm_gpu_seq_embed_token(rig.ctx(), s.h, s.last);
			for (int g = 0; st == GS::SSLM_OK && LayerIdx(s.h) < L && g < 64; ++g) {
				st = sslm_decode_step_gpu(rig.ctx(), s.h, nullptr, rig.env.fx.one_layer_budget);
				if (st == GS::SSLM_OK) st = Drain(rig.env, s.h);
				++r.slices;
			}
			if (st == GS::SSLM_OK) st = SslmGpuSeqFinishTokenForG5Bridge(rig.ctx(), s.h, &t);
		}
		r.st = st;
		if (st != GS::SSLM_OK) return r;
		if (t == -2) {
			RecordEnd(rig, s, &r, walk0, ctx0);
			return r;
		}
		if (t < 0) return r;
		r.stream.push_back(t);
		s.last = t;
		s.ready = s.mid = false;
		s.primed = true;
	}
	return r;
}

// The checks every accepting end shares, and `retries` retries through the decode bridge and the finish bridge
// alternately: -2 at OK each time, accepting 1, the blob and the context unchanged.
inline void CheckCleanStop(const char* cell, const char* arm, const char* schema, GRig& rig, GSeq& s, const Run& r,
                           size_t prompt_len, int retries = 2) {
	CHECK_MSG(r.dead && r.st == GS::SSLM_OK, "[%s] %s %s: no -2 at SSLM_OK (%s, %zu tokens)", cell, arm, schema,
	          Name(r.st), r.stream.size());
	if (!r.dead) return;
	CHECK_MSG(r.accepting == 1, "[%s] %s %s: -2 with accepting %d, want 1 (the schema is complete)", cell, arm, schema,
	          r.accepting);
	CHECK_MSG(r.walk_after == r.walk_before, "[%s] %s %s: the -2 moved the walk %u -> %u", cell, arm, schema,
	          r.walk_before, r.walk_after);
	CHECK_MSG(r.ready_after == 1, "[%s] %s %s: ready %d after the -2, want 1 (re-armed)", cell, arm, schema, r.ready_after);
	CHECK_MSG(r.ctx_after == int64_t(prompt_len + r.stream.size()), "[%s] %s %s: context %lld at the end, want the "
	          "prompt %zu + the %zu produced tokens", cell, arm, schema, static_cast<long long>(r.ctx_after), prompt_len,
	          r.stream.size());
	for (int k = 0; k < retries; ++k) {
		int32_t t = -7;
		const bool finish = (k % 2) == 1;
		const GS st = finish ? SslmGpuSeqFinishTokenForG5Bridge(rig.ctx(), s.h, &t)
		                     : SslmGpuSeqDecodeStepForG5Bridge(rig.ctx(), s.h, s.last, rig.env.full_budget, &t);
		CHECK_MSG(st == GS::SSLM_OK && t == -2 && Accepting(rig.ctx(), s.h) == 1, "[%s] %s %s: retry %d through the %s "
		          "bridge gave %s token %d accepting %d, want OK, -2, 1", cell, arm, schema, k, finish ? "finish" : "decode",
		          Name(st), t, Accepting(rig.ctx(), s.h));
		CHECK_MSG(CtxLen(s.h) == r.ctx_after && rig.Blob(s.h) == r.end_blob, "[%s] %s %s: retry %d changed the sequence "
		          "(context %lld -> %lld)", cell, arm, schema, k, static_cast<long long>(r.ctx_after),
		          static_cast<long long>(CtxLen(s.h)));
	}
}

// The CPU reference: greedy decode to the end, whole tokens (layer budget = the model's depth, CRig::Decode) or one layer
// per call (cpu::Token). Every failure is a CHECK failure: an empty or short reference stream never passes silently.
struct CpuRun {
	bool dead = false;
	std::vector<int32_t> stream;
	int32_t accepting = -1;
	int32_t query_count = -1;
};
inline CpuRun RunCpu(CRig& c, const char* schema, const std::vector<int32_t>& prompt, bool sliced, int max = 400) {
	CpuRun r;
	CHECK_MSG(c.ok, "CPU reference %s: the rig did not open", schema);
	sslm_seq s = c.ok ? c.New(schema) : nullptr;
	const sslm_status ps = s ? c.Prefill(s, prompt, SSLM_SPAN_PROMPT) : SSLM_INVALID_ARGUMENT;
	CHECK_MSG(ps == SSLM_OK, "CPU reference %s: prompt prefill of %zu tokens: %d", schema, prompt.size(),
	          static_cast<int>(ps));
	if (ps != SSLM_OK) return r;
	for (int i = 0; i < max; ++i) {
		sslm_status st = SSLM_OK;
		const int32_t t = sliced ? cpu::Token(c.M(), s, &st) : c.Decode(s);
		CHECK_MSG(!(sliced && st != SSLM_OK) && t >= -2, "CPU reference %s %s: decode %d gave status %d token %d", schema,
		          sliced ? "sliced" : "whole", i, static_cast<int>(st), t);
		if ((sliced && st != SSLM_OK) || t < -2) return r;
		if (t == -2) {
			r.dead = true;
			r.accepting = CpuAccepting(c.M(), s);
			r.query_count = QueryCpu(c.M(), s).count;
			return r;
		}
		CHECK_MSG(t >= 0, "CPU reference %s %s: decode %d gave %d, want a token or -2", schema, sliced ? "sliced" : "whole",
		          i, t);
		if (t < 0) return r;
		r.stream.push_back(t);
	}
	return r;
}

// ---- multi-row (EXACT_DRAFT) ----------------------------------------------------------------------------
inline GS EmbedRows(GRig& rig, SslmGpuSequenceHandle* h, int32_t kind, const std::vector<int32_t>& rows, int32_t* admitted) {
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), kind, rows.data(), static_cast<int32_t>(rows.size()), 0u};
	*admitted = -9;
	return sslm_gpu_seq_embed_rows(rig.ctx(), h, &spec, admitted);
}
inline int32_t RowsOpen(SslmGpuSequenceHandle* h) {
	int32_t r = -1;
	return sslm_gpu_seq_rows_open(h, &r) == GS::SSLM_OK ? r : -1;
}
inline int32_t LastCommitted(SslmGpuSequenceHandle* h) {
	int32_t r = -1;
	return sslm_gpu_seq_last_rows_committed(h, &r) == GS::SSLM_OK ? r : -1;
}
// Slices of `budget` dispatches until the open chunk closes; the closing ready's status. `*slices` counts them.
inline GS CloseChunk(GRig& rig, SslmGpuSequenceHandle* h, uint32_t budget, int* slices = nullptr) {
	for (int i = 0; i < 128; ++i) {
		GS st = sslm_decode_step_gpu(rig.ctx(), h, nullptr, budget);
		if (st == GS::SSLM_OK) st = Drain(rig.env, h);
		if (slices) ++*slices;
		if (st != GS::SSLM_OK) return st;
		if (RowsOpen(h) == 0) return GS::SSLM_OK;
	}
	return GS::SSLM_DEVICE_LOST;
}
struct Exact {
	GS embed = GS::SSLM_OK, close = GS::SSLM_OK, prod = GS::SSLM_OK;
	int32_t admitted = -9, token = -77;
	int slices = 0;
};
inline Exact RunExact(GRig& rig, GSeq& s, const std::vector<int32_t>& rows, uint32_t budget) {
	Exact e;
	e.embed = EmbedRows(rig, s.h, kExactDraft, rows, &e.admitted);
	if (e.embed != GS::SSLM_OK) return e;
	e.close = CloseChunk(rig, s.h, budget, &e.slices);
	e.prod = sslm_gpu_seq_rows_produced(s.h, &e.token);
	if (e.prod == GS::SSLM_OK && e.token == -2) s.ready = true;
	if (e.prod == GS::SSLM_OK && e.token >= 0) {
		s.last = e.token;
		s.ready = s.mid = false;
	}
	return e;
}
// The Off reference on `m`: the decode-bridge stream to its -2 (`dead`: the count of tokens before it), and the blob
// after the -2.
struct Off {
	bool ok = false;
	std::vector<int32_t> toks;
	std::vector<uint8_t> end_blob;
};
inline Off RunOff(GRig& rig, SslmGpuModelHandle* m, const char* schema, const std::vector<int32_t>& prompt) {
	Off o;
	GSeq t = rig.New(schema, m);
	if (!t.h || rig.Prompt(t, prompt) != GS::SSLM_OK) return o;
	const Run r = RunWhole(rig, t);
	o.ok = r.dead && r.st == GS::SSLM_OK;
	o.toks = r.stream;
	o.end_blob = r.end_blob;
	return o;
}
// A sequence on `m` after the prompt and `n` decode-bridge tokens: the last produced token is in hand (pending).
inline GSeq After(GRig& rig, SslmGpuModelHandle* m, const char* schema, const std::vector<int32_t>& prompt, size_t n) {
	GSeq s = rig.New(schema, m);
	if (!s.h || rig.Prompt(s, prompt) != GS::SSLM_OK) return GSeq{};
	for (size_t i = 0; i < n; ++i) {
		GS st = GS::SSLM_OK;
		const int32_t t = Token(rig.env, s, &st);
		if (st != GS::SSLM_OK || t < 0) return GSeq{};
	}
	return s;
}

// G3 and G5's shared body: the EXACT_DRAFT chunk whose last row is the closing token, closed in one slice of the
// full budget (G3) or in one-layer slices (G5).
inline int EndOnLastRow(int argc, char** argv, bool sliced) {
	const char* cell = sliced ? "CS.G5" : "CS.G3";
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		const std::string dir = FixtureDir(argc, argv);
		CRig c(rig.raw);
		Mapped dev;
		if (MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev)) {
			for (const Case& k : kCases) {
				const std::vector<int32_t> prompt = c.Tokenize(k.prompt_text);
				const Off off = RunOff(rig, dev.m, k.schema, prompt);
				CHECK_MSG(off.ok, "[%s] %s: the Off run did not reach its end", cell, k.schema);
				if (!off.ok) continue;
				const size_t e = off.toks.size();
				for (size_t M : {size_t(1), size_t(3), size_t(8)}) {
					if (M > e) continue;
					const size_t j = e - M;
					GSeq s = After(rig, dev.m, k.schema, prompt, j + 1);
					if (!s.h) {
						CHECK_MSG(false, "[%s] %s M%zu: the history did not replay", cell, k.schema, M);
						continue;
					}
					const std::vector<int32_t> rows(off.toks.begin() + static_cast<std::ptrdiff_t>(j), off.toks.end());
					const uint32_t budget = sliced ? rig.env.fx.one_layer_budget : rig.env.full_budget;
					const Exact x = RunExact(rig, s, rows, budget);
					CHECK_MSG(x.embed == GS::SSLM_OK && x.admitted == static_cast<int32_t>(M), "[%s] %s M%zu: embed_rows %s, "
					          "admitted %d", cell, k.schema, M, Name(x.embed), x.admitted);
					CHECK_MSG(x.close == GS::SSLM_OK && x.prod == GS::SSLM_OK && x.token == -2, "[%s] %s M%zu: close %s, "
					          "rows_produced %s token %d (want OK, OK, -2)", cell, k.schema, M, Name(x.close), Name(x.prod),
					          x.token);
					CHECK_MSG(!sliced || x.slices >= static_cast<int>(rig.env.fx.layers), "[%s] %s M%zu: the close took %d "
					          "slice(s) for %u layers", cell, k.schema, M, x.slices, rig.env.fx.layers);
					CHECK_MSG(LastCommitted(s.h) == static_cast<int32_t>(M), "[%s] %s M%zu: last_rows_committed %d, want %zu",
					          cell, k.schema, M, LastCommitted(s.h), M);
					CHECK_MSG(Accepting(rig.ctx(), s.h) == 1 && ReadyFlag(rig.ctx(), s.h) == 1, "[%s] %s M%zu: accepting %d "
					          "ready %d, want 1 and 1", cell, k.schema, M, Accepting(rig.ctx(), s.h), ReadyFlag(rig.ctx(), s.h));
					CHECK_MSG(CtxLen(s.h) == static_cast<int64_t>(prompt.size() + e), "[%s] %s M%zu: context %lld, want %zu",
					          cell, k.schema, M, static_cast<long long>(CtxLen(s.h)), prompt.size() + e);
					const std::vector<uint8_t> b = rig.Blob(s.h);
					CHECK_MSG(b == off.end_blob, "[%s] %s M%zu: the blob differs from the Off run's after its -2", cell,
					          k.schema, M);
					int32_t t = -7;
					const GS st = SslmGpuSeqDecodeStepForG5Bridge(rig.ctx(), s.h, off.toks.back(), rig.env.full_budget, &t);
					CHECK_MSG(st == GS::SSLM_OK && t == -2 && rig.Blob(s.h) == b, "[%s] %s M%zu: the following decode gave %s "
					          "token %d, sequence %s", cell, k.schema, M, Name(st), t, rig.Blob(s.h) == b ? "unchanged" : "changed");
				}
				rig.ReleaseSeqs();
			}
			Unmap(rig, &dev);
		}
	}
	return Finish(sliced ? "sfdg_cell_cs_g5" : "sfdg_cell_cs_g3");
}

}  // namespace cs
}  // namespace sfdg

#endif  // SSLM_SFDG_CLEANSTOP_H
