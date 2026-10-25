// Multi-row GPU plan (rev 1.3.2), slice MR3 (the public prompt verb): the cells of the MR3 red suite.
//
// Drives the verbs plan §4.1-§4.6 publish (mr3_entry.h: the statuses this base returns, §3.7) through the public
// header, with MR2's seams (mr2_entry.h) and the engine's claim and fault seams. Verdicts come only from statuses,
// whole save blobs against the one-row twin (the same stream through embed_token and the decode step, in the same
// process), and the seams, each commissioned by a mutant (plan §7 grading rule). "Per-layer" is
// DispatchesPerLayer: 24, or 25 on pkv_qk. Cells marked [box] run here as a cloud smoke and on the box in MB3.
//
// Cells (plan §7, §9's MR3 row):
//   1.1   1,000 cycles (embed_rows of random M, random slicing, close / abandon at a random layer), plus refusal-then-
//         reset cycles on the refusal fixture and abandon cycles on an adopted (shared-prefix) pooled sequence:
//         device allocations (harness seam: none after warm-up), mapped pages, pool free count and shared pages
//         return to their start values; the model unmaps SSLM_OK          (--cycles=N, default 1000)
//   1.3   reset, release and restore with a chunk open (§4.6): reset equals a reset twin; release frees the row set
//         and the pool pages; a restored handle has no chunk
//   2.1   every bad spec of §4.1 (kind 1 before R-17, kind 2 before MR5, FIRST_PREWALKED with PROMPT), tokens = vocab
//         and -1 at rows 0 and 7, mid-token and chunk-open BUSY, the check order (Submitted + bad spec = BUSY,
//         mid-token + bad spec = BUSY, bad spec + bad token = the argument status), nothing written (save bytes);
//         9.3: a larger struct_size with a 0xAB canary tail is accepted, the canary untouched, the chunk equal
//   2.2   the four MR3 verbs with a null ctx, seq or out pointer and a foreign handle (rows_produced is MR5's)
//   2.4   every verb of §4.6's table at layer 0 after embed and at a mid layer, M 1 and 5, dedicated and pooled:
//         its stated status, rows_open and the layer unchanged, then the chunk completes equal to the twin
//   3.1   [box] the claimed verbs (embed_rows, the decode step, a batch slot, rows_abandon, the decode bridge) raced on
//         one handle: the winner parked inside its claim (claim seam), every loser SSLM_BUSY with nothing changed,
//         then free-running races (--race-iterations=N, default 10000) leave the handle equal to a serial twin
//   3.2   [box] two threads, two sequences, open chunks of M 8 and 3, interleaved (submitting calls serialized by
//         the caller, P1's precondition): each equals its twin
//   3.3w  (the cloud stress leg of 3.3; the TSan leg is mr3_tsan.cpp) sslm_gpu_ready polled on one thread against
//         rows_abandon at the closing layer on another (a documented caller error): another handle's chunk, run
//         beside it, still equals its twin
//   5.1   dedicated headroom: p0 + count > cap admits cap - p0 (to 1 row); p0 = cap returns the cap status
//   5.2   the pooled budget limit by the same rule; zero rows return the pooled limit status
//   5.4   a batch of chunk-open and plain members: one submission per member (the coordinator's ruling on MR-D14),
//         every member equal to its twin; a batch of plain members only keeps the shipped path (one submission per
//         member, statuses, bytes)
//   5.5   budgets 0, 1 and per-layer - 1 on an open chunk (layer 0 and mid): SSLM_DISPATCH_BUDGET_TOO_SMALL, the
//         chunk unchanged, then equal to the twin
//   5.6   device lost at slice k (armed slice fault + removed-device query): SSLM_DEVICE_LOST; then rows_open,
//         rows_abandon and release return SSLM_OK, the pool's free count returns, and a fresh context is equal
//   7.1   embed_rows submits nothing (submission seam) and leaves the handle Idle (a decode step is accepted)
//   7.2   the layer seam advances floor(budget / per-layer) per slice, capped at the chunk's end
//   7.3   count 1 equals embed_token plus a one-row token, on the one-row body (no widened list)
//   7.4   after a clean close the last row rests at full depth; the finish token and blob equal the twin's
//   7.5   abandon at every layer 0..L-1 restores entry (whole blob) and the next feed equals the twin; abandon on a
//         fresh and on a closed handle returns SSLM_OK with no change
//   7.6   save at layer 0 and at every 0 < layer < L returns CHUNK_OPEN with the canary untouched; save succeeds
//         after close and after abandon
//   7.7   an adapter change on a later slice (A -> none, none -> A, A -> A') returns the mid-token status, chunk
//         unchanged; the chunk then completes equal to the twin                       [pkv_def + pkv_def_r4.adp]
//   7.8   bind_adapter returns BUSY while a chunk is open (layer 0 and mid), and is admitted after the close
//   7.13  the prefill snapshot is unchanged across bridge prompt -> embed_rows -> close, abandon or refusal; with no
//         snapshot the read returns SSLM_PREFILL_HIDDEN_UNAVAILABLE before and after
//   7.14  embed_rows clears ready_for_logits (the whole blob after the close equals the twin's) and rollback restores
//         it (abandon after an armed rest gives back the armed blob); a following decode-bridge call equals the twin's
//   7.15  last_rows_committed: M after a clean close, 0 after an abandon, a reset and a restore, j after a refusal
//   7.16  after the close the mirror holds every committed row (M 2..8, dedicated and pooled): blob equal (MX17)
//   8.1   prefix reuse (adopt) then embed_rows equals the twin
//   8.2   shared prefix pages unchanged: the prefix's own save bytes, the other adopter's blob and the whole device
//         pool against the twin world's, M 3, 5 and 8
//   8.6   a batch of plain members beside open chunks at a mid layer: one submission per member, every member and
//         every chunk equal to its twin
//   8.8   embed_rows(PROMPT) on a schema-bound sequence: binding, walk and ready in the SLM5 tail equal the twin's
//   8.9   bridge then embed_rows, and embed_rows then bridge, on one handle, each equal to its twin (see 8.9's note)
//   9.1   save after the close and restore (SLM5 and SLM6): the restored handle re-saves the same bytes, and its
//         continuation equals the twin's restored handle's
//   9.3   in 2.1
//   (9.2 is mr3_92_blobs.cpp, run by the runner against the base tag's binary.)
//
// Usage: mr3_cells.exe --model=PATH [--fixtures=DIR] [--adapter=PATH] [--shaders=DIR] [--cells=..] [--cycles=N]
//                      [--race-iterations=N]
// Prints "cell <id>: PASS" or "RED cell <id>: <n> failures"; exit 0 only when every cell run passes.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <thread>

#include "mr2_common.h"
#include "mr3_entry.h"

void SslmGpuAllocCounterResetForTest() noexcept;
uint32_t SslmGpuAllocCountForTest() noexcept;
#ifdef SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING
SslmGpuStatus SslmGpuSeqPrefillPromptPreBatchingBenchOnly(SslmGpuContext*, SslmGpuSequenceHandle*, const int32_t*,
                                                          int32_t, uint32_t);
#endif

namespace {

using mr2::S;
using mr2::World;

struct Opt {
	std::string model, fixtures, adapter, shaders;
	std::set<std::string> cells;
	int cycles = 1000, race_iterations = 10000;
	bool want(const char* c) const { return cells.empty() || cells.count(c); }
};
Opt g_opt;

// ------------------------------------------------------------------------------------------------ reporting
std::map<std::string, int> g_fail, g_checks;
std::string g_cell;
std::mutex g_report_mu;
void Check(bool ok, const std::string& what) {
	const std::lock_guard<std::mutex> lock(g_report_mu);
	++g_checks[g_cell];
	if (!ok) {
		++g_fail[g_cell];
		if (g_fail[g_cell] <= 14) std::printf("FAIL %s: %s\n", g_cell.c_str(), what.c_str());
	}
}
std::string F(const char* fmt, ...) {
	char b[640];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(b, sizeof b, fmt, ap);
	va_end(ap);
	return b;
}
const char* N(S s) {
	if (s == MR3_CAP_STATUS) return "CAP(provisional)";
	if (s == MR3_CHUNK_OPEN_STATUS) return "CHUNK_OPEN(provisional)";
	static thread_local char buf[8][24];
	static thread_local int at = 0;
	char* b = buf[at++ % 8];
	std::snprintf(b, 24, "%u", static_cast<unsigned>(s));
	return b;
}
void BlobEq(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::string& tag) {
	const long d = mr2::FirstDiff(got, want);
	std::string ctx;
	if (d >= 0) {  // the bytes at the first difference, and how many bytes differ in all
		size_t n = 0;
		for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) n += got[i] != want[i];
		for (long i = d; i < d + 8 && i < static_cast<long>(std::min(got.size(), want.size())); ++i) ctx += F(" %02x/%02x", got[i], want[i]);
		ctx = F(", %zu bytes differ; got/want from there:", n) + ctx;
	}
	Check(d < 0 && !got.empty(), F("%s: blob differs (sizes %zu/%zu, first byte %ld%s)", tag.c_str(), got.size(), want.size(), d, ctx.c_str()));
}
void StatusIs(S got, S want, const std::string& tag) {
	Check(got == want, F("%s: returned %s, want %s", tag.c_str(), N(got), N(want)));
}

// ------------------------------------------------------------------------------------------------ drivers
const char* Shaders() { return g_opt.shaders.empty() ? nullptr : g_opt.shaders.c_str(); }
uint32_t Full(const mr2::Model& m) { return m.full_budget(); }
std::vector<int32_t> Stream(int from, int n, int32_t vocab, uint32_t stream = 0) {
	std::vector<int32_t> v;
	for (int i = 0; i < n; ++i) v.push_back(mr2::StreamToken(from + i, vocab, stream));
	return v;
}
int32_t RowsOpen(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_rows_open(s, &r) == S::SSLM_OK ? r : -1;
}
int32_t LastCommitted(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_last_rows_committed(s, &r) == S::SSLM_OK ? r : -1;
}
uint32_t Layer(SslmGpuSequenceHandle* s) {
	SslmGpuMr2ChunkState cs{};
	sslm_gpu_test_only_mr2_chunk_state(s, &cs);
	return cs.layer;
}
SslmGpuRowsSpec Spec(const std::vector<int32_t>& toks, int32_t kind = SSLM_GPU_ROWS_PROMPT, uint32_t flags = 0) {
	return SslmGpuRowsSpec{sizeof(SslmGpuRowsSpec), kind, toks.data(), static_cast<int32_t>(toks.size()), flags};
}
S Embed(World& w, SslmGpuSequenceHandle* s, const std::vector<int32_t>& toks, int32_t* admitted = nullptr) {
	int32_t a = 0;
	const SslmGpuRowsSpec spec = Spec(toks);
	return sslm_gpu_seq_embed_rows(w.ctx, s, &spec, admitted ? admitted : &a);
}
struct ChunkRun {
	S embed = S::SSLM_OK, close = S::SSLM_OK;
	int32_t admitted = 0;
	int slices = 0;
	bool closed = false;
	std::vector<uint32_t> layers;
};
// Slices of budget(i) on an open chunk until it closes or a call fails.
void Advance(World& w, SslmGpuSequenceHandle* s, ChunkRun& r, const std::function<uint32_t(int)>& budget,
             SslmGpuAdapterHandle* adapter = nullptr) {
	for (int i = 0; i < 64; ++i) {
		S st = sslm_decode_step_gpu(w.ctx, s, adapter, budget(i));
		if (st == S::SSLM_OK) st = mr2::Ready(w, s);
		++r.slices;
		r.layers.push_back(Layer(s));
		if (st != S::SSLM_OK) {
			r.close = st;
			r.closed = RowsOpen(s) == 0;
			return;
		}
		if (RowsOpen(s) == 0) {
			r.closed = true;
			return;
		}
	}
	r.close = S::SSLM_DEVICE_LOST;
}
ChunkRun RunChunk(World& w, SslmGpuSequenceHandle* s, const std::vector<int32_t>& toks,
                  const std::function<uint32_t(int)>& budget, SslmGpuAdapterHandle* adapter = nullptr) {
	ChunkRun r;
	r.embed = Embed(w, s, toks, &r.admitted);
	if (r.embed == S::SSLM_OK) Advance(w, s, r, budget, adapter);
	return r;
}
S Twin(World& w, SslmGpuSequenceHandle* s, const std::vector<int32_t>& toks, uint32_t budget, const mr2::Model& m,
       SslmGpuAdapterHandle* adapter = nullptr) {
	for (int32_t t : toks) {
		const S st = mr2::FeedRow(w, s, t, budget, m.layers(), adapter);
		if (st != S::SSLM_OK) return st;
	}
	return S::SSLM_OK;
}
std::vector<uint8_t> Blob(World& w, SslmGpuSequenceHandle* s, const std::string& tag) {
	std::vector<uint8_t> b;
	S st = S::SSLM_OK;
	if (!mr2::Save(w, s, &b, &st)) Check(false, F("%s: save returned %s", tag.c_str(), N(st)));
	return b;
}
void CleanClose(const ChunkRun& r, SslmGpuSequenceHandle* s, int32_t m, const std::string& tag) {
	Check(r.embed == S::SSLM_OK && r.admitted == m, F("%s: embed_rows %s admitted %d of %d", tag.c_str(), N(r.embed), r.admitted, m));
	Check(r.close == S::SSLM_OK && r.closed, F("%s: chunk %s after %d slices (closed %d)", tag.c_str(), N(r.close), r.slices, r.closed));
	Check(LastCommitted(s) == m, F("%s: last_rows_committed %d, want %d", tag.c_str(), LastCommitted(s), m));
}
void FinishEq(World& wa, SslmGpuSequenceHandle* a, World& wb, SslmGpuSequenceHandle* b, const std::string& tag) {
	int32_t ta = -9, tb = -9;
	const S sa = SslmGpuSeqFinishTokenForG5Bridge(wa.ctx, a, &ta);
	const S sb = SslmGpuSeqFinishTokenForG5Bridge(wb.ctx, b, &tb);
	Check(sa == S::SSLM_OK && sb == S::SSLM_OK && ta == tb, F("%s: finish %s token %d, twin %s token %d", tag.c_str(), N(sa), ta, N(sb), tb));
	BlobEq(Blob(wa, a, tag), Blob(wb, b, tag), tag + " after finish");
}
SslmGpuMr2Seams Seams(World& w) {
	SslmGpuMr2Seams s{};
	Check(sslm_gpu_test_only_mr2_seams(w.ctx, &s) == S::SSLM_OK, "seam read");
	return s;
}
SslmGpuKvPoolStats PoolStats(SslmGpuKvPoolHandle* p) {
	SslmGpuKvPoolStats st{};
	st.struct_size = sizeof st;
	sslm_gpu_kv_pool_stats(p, &st);
	return st;
}
uint32_t HolderMapped(SslmGpuSequenceHandle* s) {
	uint32_t n = 0;
	sslm_gpu_test_only_mr2_holder_mapped(s, &n);
	return n;
}
bool PrefixFill(World& w, SslmGpuPrefixHandle* pre, const std::vector<int32_t>& toks, uint32_t budget) {
	for (size_t done = 0; done < toks.size();) {
		int32_t used = 0;
		if (sslm_gpu_prefix_prefill(w.ctx, pre, toks.data() + done, static_cast<int32_t>(toks.size() - done), 0, budget, &used) !=
		        S::SSLM_OK ||
		    used <= 0)
			return false;
		done += static_cast<size_t>(used);
	}
	return true;
}
// An adopted sequence rests armed (the prefix's prompt fill set ready_for_logits; SLM6 offset 128). embed_rows clears
// the flag (§4.1); embed_token would leave it stale (E25), so a one-row twin of an adopted sequence first consumes it
// through the decode bridge's ready branch, a finish that changes nothing else on a handle with no schema bound.
void DisarmTwin(World& w, SslmGpuSequenceHandle* t) {
	int32_t tok = -1;
	Check(SslmGpuSeqDecodeStepForG5Bridge(w.ctx, t, 0, 1u << 20, &tok) == S::SSLM_OK, "twin: the decode bridge's ready branch");
}
SslmGpuPrefixHandle* FrozenPrefix(World& w, const mr2::Model& m, SslmGpuKvPoolHandle* pool, int n, uint32_t stream) {
	SslmGpuPrefixHandle* pre = nullptr;
	if (sslm_gpu_prefix_begin(w.ctx, w.model, pool, n + 8, &pre) != S::SSLM_OK) return nullptr;
	if (!PrefixFill(w, pre, Stream(0, n, m.vocab(), stream), Full(m)) || sslm_gpu_prefix_freeze(w.ctx, pre) != S::SSLM_OK) {
		sslm_gpu_prefix_release(w.ctx, pre);
		return nullptr;
	}
	return pre;
}
// Save while a chunk is open: CHUNK_OPEN, and a buffer pre-filled with 0xAB stays 0xAB (nothing written).
void SaveRefused(World& w, SslmGpuSequenceHandle* s, const std::string& tag) {
	std::vector<uint8_t> buf(1 << 20, 0xAB);
	size_t sz = buf.size();
	const S st = sslm_gpu_seq_save(w.ctx, s, buf.data(), &sz);
	StatusIs(st, MR3_CHUNK_OPEN_STATUS, tag + ": save with a chunk open");
	Check(std::all_of(buf.begin(), buf.end(), [](uint8_t c) { return c == 0xAB; }), tag + ": the refused save wrote into the buffer");
}

// The pair a cell compares: two handles of the same kind on one world (pooled ones from their own pools).
struct Pair {
	SslmGpuSequenceHandle *a = nullptr, *t = nullptr;
	SslmGpuKvPoolHandle *pa = nullptr, *pt = nullptr;
};
Pair MakePair(World& w, bool pooled, int32_t cap = 64) {
	Pair p;
	if (pooled) {
		p.a = w.Pooled(p.pa = w.Pool(16), cap);
		p.t = w.Pooled(p.pt = w.Pool(16), cap);
	} else {
		p.a = w.Dedicated(cap);
		p.t = w.Dedicated(cap);
	}
	return p;
}

// ------------------------------------------------------------------------------------------------ refusal fixtures
struct RefusalStream {
	std::vector<int32_t> toks;
	bool ok = false;
};
S Probe(const mr2::Model& rm, World& w, const std::vector<int32_t>& toks) {
	SslmGpuSequenceHandle* s = w.Dedicated(64);
	if (!s) return S::SSLM_DEVICE_LOST;
	S st = S::SSLM_OK;
	for (int32_t t : toks)
		if ((st = mr2::FeedRow(w, s, t, Full(rm), rm.layers())) != S::SSLM_OK) break;
	w.Release(s);
	return st;
}
// p0 passing rows, then 8 rows whose row j refuses (rows after j arbitrary).
RefusalStream FindRefusalStream(const mr2::Model& rm, World& w, int p0, int j) {
	RefusalStream r;
	const int32_t V = rm.vocab();
	int cand = 0;
	std::vector<int32_t> v;
	auto next = [&](S want, uint32_t stream, int tries) {
		for (int k = 0; k < tries; ++k) {
			v.push_back(mr2::StreamToken(cand++, V, stream));
			if (Probe(rm, w, v) == want) return true;
			v.pop_back();
		}
		return false;
	};
	for (int i = 0; i < p0 + j; ++i)
		if (!next(S::SSLM_OK, 11, 64)) return r;
	if (!next(S::SSLM_SEQUENCE_REJECTED, 12, 128)) return r;
	for (int i = j + 1; i < 8; ++i) v.push_back(mr2::StreamToken(cand++, V, 14));
	r.toks = v;
	r.ok = true;
	return r;
}
// n rows that pass on the refusal fixture's one-row path.
std::vector<int32_t> PassingRows(const mr2::Model& rm, World& probe, int n, uint32_t stream) {
	std::vector<int32_t> ok;
	for (int c = 0; static_cast<int>(ok.size()) < n && c < 400; ++c) {
		std::vector<int32_t> v = ok;
		v.push_back(mr2::StreamToken(c, rm.vocab(), stream));
		if (Probe(rm, probe, v) == S::SSLM_OK) ok = v;
	}
	return ok;
}
bool LoadRefuse(const mr2::Model& m, mr2::Model* rm) {
	return !g_opt.fixtures.empty() && rm->Load(g_opt.fixtures + "/" + m.base() + "_refuse.sslm");
}
bool LoadAdapter(const mr2::Model& m, std::vector<uint8_t>* bytes, superslm::SslmModelView* view) {
	if (g_opt.adapter.empty() || m.base() != "pkv_def") return false;
	std::string err;
	return mr2::ReadFileBytes(g_opt.adapter, bytes) &&
	       superslm::SslmModel::Load(bytes->data(), bytes->size(), *view, &err) == superslm::SslmModelStatus::Ok;
}

// ------------------------------------------------------------------------------------------------ 1.1
void Cell11(const mr2::Model& m) {
	g_cell = "1.1";
	const int32_t V = m.vocab();
	const uint32_t L = m.layers(), D = m.dpl();
	std::mt19937 rng(1101);
	{
		World w(m, Shaders());
		if (!w.ok) { Check(false, "world"); return; }
		SslmGpuSequenceHandle* a = w.Dedicated(96);
		SslmGpuKvPoolHandle* pool = w.Pool(32);
		const int B = static_cast<int>(PoolStats(pool).page_positions);
		SslmGpuPrefixHandle* pre = FrozenPrefix(w, m, pool, 2 * B, 31);  // the entry on a page boundary (1.1r's lesson)
		SslmGpuSequenceHandle* b = w.Pooled(pool, 4 * B);
		Check(pre && b && sslm_gpu_seq_adopt_prefix(w.ctx, b, pre) == S::SSLM_OK, "adopted sequence");
		// Warm-up: every variant once on each handle.
		int at = 0;
		for (int M : {8, 3, 2, 1}) {
			CleanClose(RunChunk(w, a, Stream(at, M, V, 32), [&](int) { return D; }), a, M, F("warm-up M%d", M));
			at += M;
			const ChunkRun r = RunChunk(w, b, Stream(at, M, V, 33), [&](int) { return D; });
			if (RowsOpen(b) > 0) sslm_gpu_seq_rows_abandon(w.ctx, b);
		}
		sslm_gpu_seq_rows_abandon(w.ctx, b);
		Check(sslm_gpu_seq_reset(w.ctx, a) == S::SSLM_OK, "reset after warm-up");
		// b: roll the warm-up chunks back is not possible after a close; re-adopt on a fresh handle.
		w.Release(b);
		b = w.Pooled(pool, 4 * B);
		Check(b && sslm_gpu_seq_adopt_prefix(w.ctx, b, pre) == S::SSLM_OK, "re-adopt");
		const SslmGpuKvPoolStats p0 = PoolStats(pool);
		const uint32_t mapped0 = HolderMapped(b);
		const std::vector<uint8_t> b_entry = Blob(w, b, "1.1 adopted entry");
		const SslmGpuMr2Seams s0 = Seams(w);
		SslmGpuAllocCounterResetForTest();
		int closes = 0, abandons = 0;
		for (int cyc = 0; cyc < g_opt.cycles; ++cyc) {
			const int M = 1 + static_cast<int>(rng() % 8);
			if (cyc % 4 == 3) {  // the adopted pooled sequence: abandon at a random layer
				int32_t adm = 0;
				Check(Embed(w, b, Stream(cyc, M, V, 34), &adm) == S::SSLM_OK, F("cycle %d adopted embed", cyc));
				const uint32_t stop = rng() % L;
				for (uint32_t l = 0; l < stop; ++l) mr2::Slice(w, b, D);
				Check(sslm_gpu_seq_rows_abandon(w.ctx, b) == S::SSLM_OK, F("cycle %d adopted abandon", cyc));
				++abandons;
				if (cyc % 64 == 3) BlobEq(Blob(w, b, "1.1"), b_entry, F("cycle %d adopted back at entry", cyc));
				continue;
			}
			if (*SslmGpuSeqHandleContextLengthForBench(a) + M > 96) Check(sslm_gpu_seq_reset(w.ctx, a) == S::SSLM_OK, "reset");
			const std::vector<uint8_t> before = (cyc % 16 == 0) ? Blob(w, a, "1.1") : std::vector<uint8_t>();
			int32_t adm = 0;
			Check(Embed(w, a, Stream(cyc, M, V, 35), &adm) == S::SSLM_OK && adm == M, F("cycle %d embed M%d", cyc, M));
			if (rng() % 2) {  // close, at random slicing
				ChunkRun r;
				r.admitted = adm;
				Advance(w, a, r, [&](int) { return D * (1 + rng() % L); });
				CleanClose(r, a, M, F("cycle %d", cyc));
				++closes;
			} else {  // abandon at a random layer
				const uint32_t stop = rng() % L;
				for (uint32_t l = 0; l < stop; ++l) mr2::Slice(w, a, D);
				Check(sslm_gpu_seq_rows_abandon(w.ctx, a) == S::SSLM_OK && RowsOpen(a) == 0, F("cycle %d abandon", cyc));
				++abandons;
				if (!before.empty()) BlobEq(Blob(w, a, "1.1"), before, F("cycle %d back at entry after abandon", cyc));
			}
		}
		const uint32_t allocs = SslmGpuAllocCountForTest();
		const SslmGpuMr2Seams s1 = Seams(w);
		const SslmGpuKvPoolStats p1 = PoolStats(pool);
		Check(allocs == 0, F("%u device allocations over %d cycles after warm-up (harness seam; device bytes must not grow)", allocs, g_opt.cycles));
		Check(s1.row_buffer_sets == s0.row_buffer_sets && s1.row_buffer_free == s1.row_buffer_sets && s1.chunks_open == 0,
		      F("row-buffer sets %u -> %u (free %u), chunks open %u", s0.row_buffer_sets, s1.row_buffer_sets, s1.row_buffer_free, s1.chunks_open));
		Check(p1.free_pages == p0.free_pages && p1.shared_pages == p0.shared_pages,
		      F("pool free %u shared %u, start %u/%u", p1.free_pages, p1.shared_pages, p0.free_pages, p0.shared_pages));
		Check(HolderMapped(b) == mapped0, F("adopted holder maps %u pages, start %u", HolderMapped(b), mapped0));
		BlobEq(Blob(w, b, "1.1"), b_entry, "adopted sequence after every cycle");
		std::printf("1.1: %d closes, %d abandons\n", closes, abandons);
		sslm_gpu_prefix_release(w.ctx, pre);
		// The model unmaps SSLM_OK once every handle is released.
		for (auto& s : w.seqs)
			if (s) { Check(sslm_gpu_seq_release(w.ctx, s) == S::SSLM_OK, "release"); s = nullptr; }
		for (auto& p : w.pools) { Check(sslm_gpu_kv_pool_destroy(w.ctx, p) == S::SSLM_OK, "pool destroy"); }
		w.pools.clear();
		StatusIs(sslm_gpu_model_unmap(w.ctx, w.model), S::SSLM_OK, "model unmap after the cycles");
		w.model = nullptr;
	}
	// Refusal then reset, on the refusal fixture.
	mr2::Model rm;
	if (!LoadRefuse(m, &rm)) { Check(false, "no " + m.base() + "_refuse.sslm under --fixtures"); return; }
	World probe(rm, Shaders());
	World w(rm, Shaders());
	if (!probe.ok || !w.ok) { Check(false, "refusal worlds"); return; }
	SslmGpuKvPoolHandle* pool = w.Pool(16);
	SslmGpuSequenceHandle* a = w.Pooled(pool, 64);
	const SslmGpuKvPoolStats p0 = PoolStats(pool);
	const uint32_t mapped0 = HolderMapped(a);
	const std::vector<uint8_t> fresh = Blob(w, a, "1.1 refusal entry");
	for (int j : {0, 3, 7}) {
		const RefusalStream rs = FindRefusalStream(rm, probe, 0, j);
		if (!rs.ok) { Check(false, F("no refusal stream j%d", j)); continue; }
		for (int rep = 0; rep < 8; ++rep) {
			const ChunkRun r = RunChunk(w, a, rs.toks, [&](int) { return rm.dpl() * (1 + rep % rm.layers()); });
			Check(r.close == S::SSLM_SEQUENCE_REJECTED && LastCommitted(a) == j, F("j%d rep %d: %s, committed %d", j, rep, N(r.close), LastCommitted(a)));
			Check(sslm_gpu_seq_reset(w.ctx, a) == S::SSLM_OK, "reset after the refusal");
			const SslmGpuKvPoolStats p1 = PoolStats(pool);
			Check(p1.free_pages == p0.free_pages && HolderMapped(a) == mapped0,
			      F("j%d rep %d: after refusal + reset pool free %u (start %u), holder maps %u (start %u)", j, rep, p1.free_pages,
			        p0.free_pages, HolderMapped(a), mapped0));
			BlobEq(Blob(w, a, "1.1"), fresh, F("j%d rep %d: reset handle equals the fresh one", j, rep));
		}
	}
}

// ------------------------------------------------------------------------------------------------ 1.3
void Cell13(const mr2::Model& m) {
	g_cell = "1.3";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int mid = 0; mid <= 1; ++mid)
			for (int M : {1, 5}) {
				const std::string tag = F("%s M%d %s %s", m.base().c_str(), M, pooled ? "pooled" : "dedicated", mid ? "mid layer" : "layer 0");
				World w(m, Shaders());
				if (!w.ok) { Check(false, tag); continue; }
				Pair p = MakePair(w, pooled);
				const std::vector<int32_t> pre = Stream(0, 3, V, 40), rows = Stream(3, M, V, 40);
				Check(Twin(w, p.a, pre, Full(m), m) == S::SSLM_OK && Twin(w, p.t, pre, Full(m), m) == S::SSLM_OK, tag + ": prefix");
				// reset: closes the chunk; the handle equals a reset twin.
				Check(Embed(w, p.a, rows) == S::SSLM_OK, tag + ": embed");
				if (mid) Check(mr2::Slice(w, p.a, D) == S::SSLM_OK && (m.layers() < 2 || RowsOpen(p.a) == M), tag + ": one slice");
				const SslmGpuMr2Seams open = Seams(w);
				StatusIs(sslm_gpu_seq_reset(w.ctx, p.a), S::SSLM_OK, tag + ": reset with a chunk open");
				Check(sslm_gpu_seq_reset(w.ctx, p.t) == S::SSLM_OK, tag + ": twin reset");
				Check(RowsOpen(p.a) == 0 && LastCommitted(p.a) == 0, F("%s: after reset rows_open %d last_rows_committed %d", tag.c_str(), RowsOpen(p.a), LastCommitted(p.a)));
				const SslmGpuMr2Seams after = Seams(w);
				Check(after.chunks_open + 1 == open.chunks_open && after.row_buffer_free == after.row_buffer_sets,
				      F("%s: reset left chunks open %u (was %u), row sets %u free %u", tag.c_str(), after.chunks_open, open.chunks_open, after.row_buffer_sets, after.row_buffer_free));
				BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": reset handle vs reset twin");
				CleanClose(RunChunk(w, p.a, rows, [&](int) { return D; }), p.a, M, tag + " after reset");
				Check(Twin(w, p.t, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
				BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": chunk after reset");
				// release: frees the chunk's row set, and a pooled holder's pages.
				const SslmGpuKvPoolStats before_r = pooled ? PoolStats(p.pa) : SslmGpuKvPoolStats{};
				SslmGpuSequenceHandle* r = pooled ? w.Pooled(p.pa, 64) : w.Dedicated(64);
				Check(r && Embed(w, r, rows) == S::SSLM_OK, tag + ": embed on the handle to release");
				if (mid && r) mr2::Slice(w, r, D);
				w.Release(r);
				const SslmGpuMr2Seams rel = Seams(w);
				Check(rel.chunks_open == 0 && rel.row_buffer_free == rel.row_buffer_sets,
				      F("%s: release left chunks open %u, row sets %u free %u", tag.c_str(), rel.chunks_open, rel.row_buffer_sets, rel.row_buffer_free));
				if (pooled) {
					const SslmGpuKvPoolStats now = PoolStats(p.pa);
					Check(now.free_pages == before_r.free_pages,
					      F("%s: pool free %u after releasing a chunk-open handle, %u before its create", tag.c_str(), now.free_pages, before_r.free_pages));
				}
				// restore: a restored handle has no chunk.
				const std::vector<uint8_t> saved = Blob(w, p.a, tag);
				Check(Embed(w, p.a, Stream(20, 2, V, 41)) == S::SSLM_OK, tag + ": open a chunk on the saved handle");
				SslmGpuSequenceHandle* rs = nullptr;
				uint32_t shared = 0;
				const S st = pooled ? sslm_gpu_seq_restore_into_pool(w.ctx, w.model, p.pa, saved.data(), saved.size(), nullptr, &rs, &shared)
				                    : sslm_gpu_seq_restore(w.ctx, w.model, saved.data(), saved.size(), &rs);
				Check(st == S::SSLM_OK && rs, F("%s: restore %s", tag.c_str(), N(st)));
				if (rs) {
					w.seqs.push_back(rs);
					Check(RowsOpen(rs) == 0 && LastCommitted(rs) == 0, F("%s: restored handle rows_open %d last_rows_committed %d", tag.c_str(), RowsOpen(rs), LastCommitted(rs)));
					BlobEq(Blob(w, rs, tag), saved, tag + ": restored handle re-saves its blob");
				}
				sslm_gpu_seq_rows_abandon(w.ctx, p.a);
			}
}

// ------------------------------------------------------------------------------------------------ 2.1 (+9.3)
void Cell21(const mr2::Model& m) {
	g_cell = "2.1";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* a = w.Dedicated(64);
	SslmGpuSequenceHandle* t = w.Dedicated(64);
	const std::vector<int32_t> pre = Stream(0, 3, V, 42), rows = Stream(3, 4, V, 42);
	Check(Twin(w, a, pre, Full(m), m) == S::SSLM_OK && Twin(w, t, pre, Full(m), m) == S::SSLM_OK, "prefix");
	const std::vector<uint8_t> entry = Blob(w, a, "2.1 entry");
	auto refused = [&](const SslmGpuRowsSpec* spec, S want, const std::string& what) {
		int32_t adm = 0;
		const S st = sslm_gpu_seq_embed_rows(w.ctx, a, spec, &adm);
		StatusIs(st, want, what);
		Check(RowsOpen(a) == 0, what + ": a chunk is open after the refusal");
		if (RowsOpen(a) > 0) sslm_gpu_seq_rows_abandon(w.ctx, a);
		BlobEq(Blob(w, a, what), entry, what + ": nothing written");
	};
	const S ARG = MR3_ARG_STATUS;
	refused(nullptr, ARG, "null spec");
	for (uint32_t sz : {0u, 4u, static_cast<uint32_t>(sizeof(SslmGpuRowsSpec)) - 1u}) {
		SslmGpuRowsSpec s = Spec(rows);
		s.struct_size = sz;
		refused(&s, ARG, F("struct_size %u (below v1)", sz));
	}
	for (int32_t c : {0, -1, 9, INT32_MIN, INT32_MAX}) {
		std::vector<int32_t> many = Stream(3, 9, V, 42);
		SslmGpuRowsSpec s = Spec(many);
		s.count = c;
		refused(&s, ARG, F("count %d", c));
	}
	for (int32_t k : {-1, 3, 99}) {
		const SslmGpuRowsSpec s = Spec(rows, k);
		refused(&s, ARG, F("unknown kind %d", k));
	}
	{
		SslmGpuRowsSpec s1 = Spec(rows, SSLM_GPU_ROWS_SCHEMA_FORCED), s2 = Spec(rows, SSLM_GPU_ROWS_EXACT_DRAFT);
		refused(&s1, ARG, "kind SCHEMA_FORCED before R-17");
		refused(&s2, ARG, "kind EXACT_DRAFT before MR5");
		SslmGpuRowsSpec s3 = Spec(rows, SSLM_GPU_ROWS_SCHEMA_FORCED, SSLM_GPU_ROWS_FIRST_PREWALKED);
		refused(&s3, ARG, "SCHEMA_FORCED + FIRST_PREWALKED before R-17");
	}
	for (uint32_t f : {0x2u, 0x80000000u, SSLM_GPU_ROWS_FIRST_PREWALKED | 0x4u}) {
		SslmGpuRowsSpec s = Spec(rows, SSLM_GPU_ROWS_PROMPT, f);
		refused(&s, ARG, F("unknown flag 0x%x", f));
	}
	{
		SslmGpuRowsSpec s = Spec(rows, SSLM_GPU_ROWS_PROMPT, SSLM_GPU_ROWS_FIRST_PREWALKED);
		refused(&s, ARG, "FIRST_PREWALKED with PROMPT");
	}
	{
		// A null token array with count 4: §4.1's list does not name it, so only "refused, nothing written" is graded.
		SslmGpuRowsSpec s = Spec(rows);
		s.tokens = nullptr;
		int32_t adm = 0;
		const S st = sslm_gpu_seq_embed_rows(w.ctx, a, &s, &adm);
		Check(st != S::SSLM_OK, F("null tokens with count 4 returned %s (want a refusal)", N(st)));
		if (RowsOpen(a) > 0) sslm_gpu_seq_rows_abandon(w.ctx, a);
		BlobEq(Blob(w, a, "null tokens"), entry, "null tokens: nothing written");
	}
	for (int row : {0, 7})
		for (int32_t bad : {V, -1}) {
			std::vector<int32_t> eight = Stream(3, 8, V, 42);
			eight[row] = bad;
			SslmGpuRowsSpec s = Spec(eight);
			refused(&s, S::SSLM_TOKEN_ID_OUT_OF_RANGE, F("token %d at row %d", bad, row));
		}
	{  // Order: a bad spec beats a bad token.
		std::vector<int32_t> eight = Stream(3, 8, V, 42);
		eight[0] = V;
		SslmGpuRowsSpec s = Spec(eight, 99);
		refused(&s, ARG, "bad spec (kind 99) + bad token: the argument status first");
	}
	// Chunk open: a second embed_rows returns BUSY, with a valid or a bad spec.
	{
		Check(Embed(w, a, rows) == S::SSLM_OK, "open a chunk");
		int32_t adm = 0;
		const SslmGpuRowsSpec ok = Spec(rows);
		StatusIs(sslm_gpu_seq_embed_rows(w.ctx, a, &ok, &adm), S::SSLM_BUSY, "embed_rows with a chunk open");
		SslmGpuRowsSpec bad = Spec(rows, 99);
		StatusIs(sslm_gpu_seq_embed_rows(w.ctx, a, &bad, &adm), S::SSLM_BUSY, "chunk open + bad spec: BUSY first");
		Check(RowsOpen(a) == 4 && Layer(a) == 0, "the refused calls changed the open chunk");
		Check(sslm_gpu_seq_rows_abandon(w.ctx, a) == S::SSLM_OK, "abandon");
		BlobEq(Blob(w, a, "chunk-open"), entry, "chunk-open refusals then abandon: nothing written");
	}
	if (m.layers() >= 2) {  // Mid-token (0 < layer < L on a one-row token): BUSY, before the spec is read.
		SslmGpuSequenceHandle* x = w.Dedicated(64);
		SslmGpuSequenceHandle* xt = w.Dedicated(64);
		Twin(w, x, pre, Full(m), m);
		Twin(w, xt, pre, Full(m), m);
		Check(sslm_gpu_seq_embed_token(w.ctx, x, rows[0]) == S::SSLM_OK && mr2::Slice(w, x, D) == S::SSLM_OK, "mid-token setup");
		Check(sslm_gpu_seq_embed_token(w.ctx, xt, rows[0]) == S::SSLM_OK && mr2::Slice(w, xt, D) == S::SSLM_OK, "mid-token twin");
		int32_t adm = 0;
		const SslmGpuRowsSpec ok = Spec(rows);
		SslmGpuRowsSpec bad = Spec(rows, 99);
		StatusIs(sslm_gpu_seq_embed_rows(w.ctx, x, &ok, &adm), S::SSLM_BUSY, "embed_rows mid-token");
		StatusIs(sslm_gpu_seq_embed_rows(w.ctx, x, &bad, &adm), S::SSLM_BUSY, "mid-token + bad spec: BUSY first");
		BlobEq(Blob(w, x, "mid-token"), Blob(w, xt, "mid-token"), "mid-token refusals: nothing written");
	}
	{  // Submitted + bad spec: BUSY (check 2 precedes check 4).
		SslmGpuSequenceHandle* x = w.Dedicated(64);
		SslmGpuSequenceHandle* xt = w.Dedicated(64);
		Twin(w, x, pre, Full(m), m);
		Twin(w, xt, pre, Full(m), m);
		Check(sslm_gpu_seq_embed_token(w.ctx, x, rows[0]) == S::SSLM_OK && sslm_decode_step_gpu(w.ctx, x, nullptr, D) == S::SSLM_OK,
		      "Submitted setup");
		int32_t adm = 0;
		SslmGpuRowsSpec bad = Spec(rows, 99);
		const SslmGpuRowsSpec ok = Spec(rows);
		StatusIs(sslm_gpu_seq_embed_rows(w.ctx, x, &bad, &adm), S::SSLM_BUSY, "Submitted + bad spec: BUSY first");
		StatusIs(sslm_gpu_seq_embed_rows(w.ctx, x, &ok, &adm), S::SSLM_BUSY, "embed_rows on a Submitted handle");
		Check(mr2::Ready(w, x) == S::SSLM_OK, "drain");
		Check(sslm_gpu_seq_embed_token(w.ctx, xt, rows[0]) == S::SSLM_OK && mr2::Slice(w, xt, D) == S::SSLM_OK, "Submitted twin");
		BlobEq(Blob(w, x, "submitted"), Blob(w, xt, "submitted"), "Submitted refusals: nothing written");
	}
	// 9.3: a larger struct_size; the tail (a 0xAB canary) is ignored and stays untouched.
	{
		g_cell = "9.3";
		struct Big {
			SslmGpuRowsSpec spec;
			uint8_t tail[24];
		} big{};
		big.spec = Spec(rows);
		big.spec.struct_size = sizeof(Big);
		std::memset(big.tail, 0xAB, sizeof big.tail);
		int32_t adm = 0;
		StatusIs(sslm_gpu_seq_embed_rows(w.ctx, a, &big.spec, &adm), S::SSLM_OK, "struct_size v1 + 24");
		Check(adm == 4, F("admitted %d of 4", adm));
		Check(std::all_of(big.tail, big.tail + sizeof big.tail, [](uint8_t c) { return c == 0xAB; }), "the spec's tail canary was written");
		ChunkRun r;
		r.admitted = adm;
		Advance(w, a, r, [&](int) { return D; });
		CleanClose(r, a, 4, "larger struct_size");
		Check(Twin(w, t, rows, Full(m), m) == S::SSLM_OK, "twin");
		BlobEq(Blob(w, a, "9.3"), Blob(w, t, "9.3"), "larger struct_size chunk vs twin");
		g_cell = "2.1";
	}
}

// ------------------------------------------------------------------------------------------------ 2.2
void Cell22(const mr2::Model& m) {
	g_cell = "2.2";
	const int32_t V = m.vocab();
	World w(m, Shaders()), w2(m, Shaders());
	if (!w.ok || !w2.ok) { Check(false, "worlds"); return; }
	SslmGpuSequenceHandle* a = w.Dedicated(64);
	SslmGpuSequenceHandle* foreign = w2.Dedicated(64);
	const std::vector<int32_t> rows = Stream(0, 4, V, 43);
	const SslmGpuRowsSpec spec = Spec(rows);
	int32_t out = 0;
	const S ARG = MR3_ARG_STATUS;
	const std::vector<uint8_t> e = Blob(w, a, "2.2"), ef = Blob(w2, foreign, "2.2");
	StatusIs(sslm_gpu_seq_embed_rows(nullptr, a, &spec, &out), ARG, "embed_rows null ctx");
	StatusIs(sslm_gpu_seq_embed_rows(w.ctx, nullptr, &spec, &out), ARG, "embed_rows null seq");
	StatusIs(sslm_gpu_seq_embed_rows(w.ctx, a, &spec, nullptr), ARG, "embed_rows null out_admitted");
	StatusIs(sslm_gpu_seq_embed_rows(w.ctx, foreign, &spec, &out), ARG, "embed_rows foreign handle (another context's)");
	StatusIs(sslm_gpu_seq_rows_abandon(nullptr, a), ARG, "rows_abandon null ctx");
	StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, nullptr), ARG, "rows_abandon null seq");
	StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, foreign), ARG, "rows_abandon foreign handle");
	StatusIs(sslm_gpu_seq_rows_open(nullptr, &out), ARG, "rows_open null seq");
	StatusIs(sslm_gpu_seq_rows_open(a, nullptr), ARG, "rows_open null out");
	StatusIs(sslm_gpu_seq_last_rows_committed(nullptr, &out), ARG, "last_rows_committed null seq");
	StatusIs(sslm_gpu_seq_last_rows_committed(a, nullptr), ARG, "last_rows_committed null out");
	Check(RowsOpen(a) == 0 && RowsOpen(foreign) == 0, "a refused call opened a chunk");
	BlobEq(Blob(w, a, "2.2"), e, "the refusals wrote nothing");
	BlobEq(Blob(w2, foreign, "2.2"), ef, "the foreign handle is unchanged");
	// A foreign handle with a chunk open: abandon from the wrong context changes nothing.
	Check(Embed(w2, foreign, rows) == S::SSLM_OK, "open a chunk on the foreign handle");
	StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, foreign), ARG, "rows_abandon of another context's open chunk");
	Check(RowsOpen(foreign) == 4, "the foreign chunk was closed by a refused abandon");
	sslm_gpu_seq_rows_abandon(w2.ctx, foreign);
	// A released handle is a use after free on this ABI (the GPU handles carry no generation; tests/paged-kv's
	// G11 precedent): not driven. See the record.
}

// ------------------------------------------------------------------------------------------------ 2.4
void Cell24(const mr2::Model& m) {
	g_cell = "2.4";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl(), L = m.layers();
	std::vector<uint8_t> adp_bytes;
	superslm::SslmModelView adp_view{};
	const bool have_adp = LoadAdapter(m, &adp_bytes, &adp_view);
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int mid = 0; mid <= (L >= 2 ? 1 : 0); ++mid)
			for (int M : {1, 5}) {
				const std::string tag = F("%s M%d %s %s", m.base().c_str(), M, pooled ? "pooled" : "dedicated", mid ? "mid layer" : "layer 0");
				World w(m, Shaders());
				if (!w.ok) { Check(false, tag); continue; }
				SslmGpuAdapterHandle* adp = nullptr;
				if (have_adp && sslm_gpu_adapter_map(w.ctx, w.model, &adp_view, &adp) == S::SSLM_OK) w.adapters.push_back(adp);
				Pair p = MakePair(w, pooled);
				SslmGpuPrefixHandle* pre = pooled ? FrozenPrefix(w, m, p.pa, 4, 44) : nullptr;
				const std::vector<int32_t> head = Stream(0, 3, V, 45), rows = Stream(3, M, V, 45);
				Check(Twin(w, p.a, head, Full(m), m) == S::SSLM_OK && Twin(w, p.t, head, Full(m), m) == S::SSLM_OK, tag + ": prefix");
				uint32_t hs = 0;
				sslm_gpu_model_hidden_size(w.model, &hs);
				std::vector<int8_t> h0(hs + 16), h1(hs + 16);
				size_t req0 = 0, req1 = 0;
				int64_t sm0 = 0, se0 = 0, sm1 = 0, se1 = 0;
				const S r0 = sslm_gpu_seq_read_prefill_final_hidden(w.ctx, p.a, h0.data(), h0.size(), &req0, &sm0, &se0);
				Check(Embed(w, p.a, rows) == S::SSLM_OK, tag + ": embed");
				if (mid) Check(mr2::Slice(w, p.a, D) == S::SSLM_OK, tag + ": one slice");
				const uint32_t layer = Layer(p.a);
				const int32_t open = RowsOpen(p.a);
				int32_t tok = -1, consumed = 0;
				const int32_t two[2] = {rows[0], rows[0]};
				auto busy = [&](S st, const char* verb) { StatusIs(st, S::SSLM_BUSY, tag + ": " + verb); };
				busy(sslm_gpu_seq_embed_token(w.ctx, p.a, rows[0]), "embed_token");
				busy(SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, p.a, two, 2, Full(m)), "prompt bridge");
				busy(SslmGpuSeqPrefillSchemaContentForG5Bridge(w.ctx, p.a, two, 1, Full(m), &consumed), "schema-content bridge");
#ifdef SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING
				busy(SslmGpuSeqPrefillPromptPreBatchingBenchOnly(w.ctx, p.a, two, 2, Full(m)), "prompt pre-batching bench");
#else
				Check(false, tag + ": the cells were built without SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING (pre-batching not driven)");
#endif
				busy(SslmGpuSeqFinishTokenForG5Bridge(w.ctx, p.a, &tok), "finish bridge");
				busy(SslmGpuSeqDecodeStepForG5Bridge(w.ctx, p.a, rows[0], Full(m), &tok), "decode bridge");
				busy(SslmGpuSeqSetSchemaForG5Bridge(w.ctx, p.a, -1), "set-schema bridge (-1)");
				if (SslmGpuModelHasSchemasForG5Bridge(w.model)) busy(SslmGpuSeqSetSchemaForG5Bridge(w.ctx, p.a, 0), "set-schema bridge (0)");
				if (pre) busy(sslm_gpu_seq_adopt_prefix(w.ctx, p.a, pre), "adopt_prefix");
				busy(sslm_gpu_seq_bind_adapter(w.ctx, p.a, nullptr), "bind_adapter (none)");
				if (adp) busy(sslm_gpu_seq_bind_adapter(w.ctx, p.a, adp), "bind_adapter (r4)");
				// Admitted: the prefill snapshot read (unchanged), sslm_gpu_ready (nothing outstanding), save (CHUNK_OPEN).
				const S r1 = sslm_gpu_seq_read_prefill_final_hidden(w.ctx, p.a, h1.data(), h1.size(), &req1, &sm1, &se1);
				Check(r1 == r0 && req1 == req0 && h1 == h0 && sm1 == sm0 && se1 == se0,
				      F("%s: read_prefill_final_hidden %s (before the chunk %s) or its bytes changed", tag.c_str(), N(r1), N(r0)));
				for (int block = 0; block <= 1; ++block) {
					int32_t ready = 0;
					S out = S::SSLM_DEVICE_LOST;
					const S st = sslm_gpu_ready(w.ctx, p.a, block, &ready, &out);
					Check(st == S::SSLM_OK && ready == 1 && out == S::SSLM_OK,
					      F("%s: sslm_gpu_ready(block %d) %s ready %d out %s on an idle chunk", tag.c_str(), block, N(st), ready, N(out)));
				}
				SaveRefused(w, p.a, tag);
				Check(RowsOpen(p.a) == open && Layer(p.a) == layer,
				      F("%s: rows_open %d layer %u after the table, before %d / %u", tag.c_str(), RowsOpen(p.a), Layer(p.a), open, layer));
				ChunkRun r;
				r.admitted = M;
				Advance(w, p.a, r, [&](int) { return D; });
				CleanClose(r, p.a, M, tag);
				Check(Twin(w, p.t, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
				BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": chunk after the table vs twin");
				if (pre) sslm_gpu_prefix_release(w.ctx, pre);
			}
}

// ------------------------------------------------------------------------------------------------ 3.1
struct Park {
	std::mutex mu;
	std::condition_variable cv;
	SslmGpuSequenceHandle* target = nullptr;
	bool armed = false, entered = false, released = false;
};
void ParkHook(SslmGpuSequenceHandle* seq, void* user) {
	Park* p = static_cast<Park*>(user);
	std::unique_lock<std::mutex> lock(p->mu);
	if (!p->armed || seq != p->target) return;
	p->armed = false;
	p->entered = true;
	p->cv.notify_all();
	p->cv.wait(lock, [&] { return p->released; });
}
void Cell31(const mr2::Model& m) {
	g_cell = "3.1";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	enum Verb { kEmbedRows, kStep, kBatch, kAbandon, kBridge, kVerbs };
	static const char* kName[kVerbs] = {"embed_rows", "decode step", "batch slot", "rows_abandon", "decode bridge"};
	Park park;
	SslmGpuSeqClaimHookForTest(&ParkHook, &park);
	// Each loser verb, made while the winner holds the claim, must return BUSY and change nothing.
	auto lose = [&](World& w, SslmGpuSequenceHandle* s, Verb v, const std::vector<int32_t>& rows) -> S {
		int32_t out = 0;
		switch (v) {
			case kEmbedRows: { const SslmGpuRowsSpec sp = Spec(rows); return sslm_gpu_seq_embed_rows(w.ctx, s, &sp, &out); }
			case kStep: return sslm_decode_step_gpu(w.ctx, s, nullptr, D);
			case kBatch: {
				SslmGpuStatus st = S::SSLM_OK;
				SslmGpuSequenceHandle* one[1] = {s};
				const SslmGpuAdapterHandle* none[1] = {nullptr};
				const S c = sslm_decode_step_batch_gpu(w.ctx, one, none, 1, D, &st);
				return c == S::SSLM_OK ? st : c;
			}
			case kAbandon: return sslm_gpu_seq_rows_abandon(w.ctx, s);
			case kBridge: return SslmGpuSeqDecodeStepForG5Bridge(w.ctx, s, rows[0], Full(m), &out);
			default: return S::SSLM_DEVICE_LOST;
		}
	};
	for (int wv = 0; wv < kVerbs; ++wv) {
		const Verb winner = static_cast<Verb>(wv);
		const std::string tag = F("%s winner %s", m.base().c_str(), kName[wv]);
		World w(m, Shaders());
		if (!w.ok) { Check(false, tag); continue; }
		SslmGpuSequenceHandle* s = w.Dedicated(64);
		SslmGpuSequenceHandle* t = w.Dedicated(64);
		const std::vector<int32_t> head = Stream(0, 3, V, 46), rows = Stream(3, 4, V, 46);
		Twin(w, s, head, Full(m), m);
		Twin(w, t, head, Full(m), m);
		// The winner's precondition: an open chunk for the step, the batch slot and the abandon.
		if (winner == kStep || winner == kBatch || winner == kAbandon) {
			Check(Embed(w, s, rows) == S::SSLM_OK && Embed(w, t, rows) == S::SSLM_OK, tag + ": open chunks");
		}
		{
			std::lock_guard<std::mutex> lock(park.mu);
			park.target = s;
			park.armed = true;
			park.entered = false;
			park.released = false;
		}
		S won = S::SSLM_DEVICE_LOST;
		std::thread th([&] { won = lose(w, s, winner, rows); });
		bool entered = false;
		{
			std::unique_lock<std::mutex> lock(park.mu);
			entered = park.cv.wait_for(lock, std::chrono::seconds(30), [&] { return park.entered; });
		}
		Check(entered, tag + ": the winner returned without taking the claim (claim seam never fired)");
		if (entered) {
			for (int lv = 0; lv < kVerbs; ++lv) {
				const S st = lose(w, s, static_cast<Verb>(lv), rows);
				StatusIs(st, S::SSLM_BUSY, F("%s: loser %s inside the winner's claim", tag.c_str(), kName[lv]));
			}
		}
		{
			std::lock_guard<std::mutex> lock(park.mu);
			park.released = true;
			park.armed = false;
		}
		park.cv.notify_all();
		th.join();
		// The winner alone ran: the same verb serially on the twin gives the same status and bytes. The winner's
		// submission is drained first (the singleton guard refuses a second submission while one is unobserved).
		if (winner == kStep || winner == kBatch) mr2::Ready(w, s);
		const S ts = lose(w, t, winner, rows);
		Check(won == ts, F("%s: winner returned %s, the serial twin %s", tag.c_str(), N(won), N(ts)));
		if (winner == kStep || winner == kBatch) mr2::Ready(w, t);
		sslm_gpu_seq_rows_abandon(w.ctx, s);
		sslm_gpu_seq_rows_abandon(w.ctx, t);
		BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": after the race vs the serial twin");
	}
	SslmGpuSeqClaimHookForTest(nullptr, nullptr);
	// Free-running: two threads hammer the claimed verbs on one open chunk. Every call returns OK or BUSY (a decode
	// step that won is drained by its own thread under the device lock, P1's precondition), and the chunk then
	// completes equal to the twin.
	World w(m, Shaders());
	if (!w.ok) return;
	SslmGpuSequenceHandle* s = w.Dedicated(64);
	SslmGpuSequenceHandle* t = w.Dedicated(64);
	const std::vector<int32_t> rows = Stream(0, 8, V, 47);
	std::mutex device;
	std::atomic<int> odd{0}, busy{0};
	Check(Embed(w, s, rows) == S::SSLM_OK, "free-running: open");
	auto hammer = [&](int seed) {
		std::mt19937 rng(seed);
		for (int i = 0; i < g_opt.race_iterations / 2; ++i) {
			const int v = static_cast<int>(rng() % 3);
			S st;
			if (v == 0) {
				const SslmGpuRowsSpec sp = Spec(rows);
				int32_t out = 0;
				st = sslm_gpu_seq_embed_rows(w.ctx, s, &sp, &out);  // a chunk is open: BUSY, always
				if (st != S::SSLM_BUSY) ++odd;
			} else if (v == 1) {
				// A no-op claimed verb on this handle: a budget below one layer (refused after the claim).
				st = sslm_decode_step_gpu(w.ctx, s, nullptr, 0);
				if (st != S::SSLM_BUSY && st != S::SSLM_DISPATCH_BUDGET_TOO_SMALL) ++odd;
			} else {
				SslmGpuStatus slot = S::SSLM_OK;
				SslmGpuSequenceHandle* one[1] = {s};
				const SslmGpuAdapterHandle* none[1] = {nullptr};
				st = sslm_decode_step_batch_gpu(w.ctx, one, none, 1, 0, &slot);
				(void)slot;
			}
			if (st == S::SSLM_BUSY) ++busy;
		}
	};
	std::thread a(hammer, 1), b(hammer, 2);
	a.join();
	b.join();
	Check(odd == 0, F("free-running: %d calls returned neither the refusal nor BUSY", odd.load()));
	Check(RowsOpen(s) == 8 && Layer(s) == 0, "free-running: the refused calls changed the chunk");
	ChunkRun r;
	r.admitted = 8;
	Advance(w, s, r, [&](int) { return D; });
	CleanClose(r, s, 8, "free-running");
	Twin(w, t, rows, Full(m), m);
	BlobEq(Blob(w, s, "3.1"), Blob(w, t, "3.1"), "free-running: after the races vs the twin");
	std::printf("3.1: free-running %d iterations, %d BUSY\n", g_opt.race_iterations, busy.load());
}

// ------------------------------------------------------------------------------------------------ 3.2
void Cell32(const mr2::Model& m) {
	g_cell = "3.2";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* s[2] = {w.Dedicated(64), w.Dedicated(64)};
	SslmGpuSequenceHandle* t[2] = {w.Dedicated(64), w.Dedicated(64)};
	const int Ms[2] = {8, 3};
	std::vector<int32_t> rows[2] = {Stream(0, 8, V, 48), Stream(0, 3, V, 49)};
	std::mutex device;  // P1's precondition: the caller serializes submitting calls on the device
	S res[2] = {S::SSLM_OK, S::SSLM_OK};
	auto run = [&](int i) {
		for (int round = 0; round < 6; ++round) {
			int32_t adm = 0;
			if (Embed(w, s[i], rows[i], &adm) != S::SSLM_OK) { res[i] = S::SSLM_DEVICE_LOST; return; }
			for (int k = 0; k < 64 && RowsOpen(s[i]) > 0; ++k) {
				std::lock_guard<std::mutex> lock(device);
				const S st = mr2::Slice(w, s[i], D);
				if (st != S::SSLM_OK) { res[i] = st; return; }
			}
			std::this_thread::yield();
		}
	};
	std::thread a(run, 0), b(run, 1);
	a.join();
	b.join();
	for (int i = 0; i < 2; ++i) {
		Check(res[i] == S::SSLM_OK, F("thread %d: %s", i, N(res[i])));
		for (int round = 0; round < 6; ++round) Twin(w, t[i], rows[i], Full(m), m);
		BlobEq(Blob(w, s[i], "3.2"), Blob(w, t[i], "3.2"), F("sequence %d (M %d, 6 chunks)", i, Ms[i]));
	}
}

// ------------------------------------------------------------------------------------------------ 3.3w
void Cell33w(const mr2::Model& m) {
	g_cell = "3.3w";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl(), L = m.layers();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* x = w.Dedicated(256);
	SslmGpuSequenceHandle* y = w.Dedicated(256);
	SslmGpuSequenceHandle* yt = w.Dedicated(256);
	std::mutex device;
	for (int rep = 0; rep < 40; ++rep) {
		const std::vector<int32_t> rx = Stream(rep * 8, 5, V, 50), ry = Stream(rep * 8, 6, V, 51);
		if (*SslmGpuSeqHandleContextLengthForBench(x) + 8 > 256) sslm_gpu_seq_reset(w.ctx, x);
		if (RowsOpen(x) > 0) sslm_gpu_seq_rows_abandon(w.ctx, x);
		Embed(w, x, rx);
		Check(Embed(w, y, ry) == S::SSLM_OK, F("rep %d: y embed", rep));
		// x: up to its closing layer, then its closing slice submitted (not drained).
		for (uint32_t l = 0; l + 1 < L; ++l) {
			std::lock_guard<std::mutex> lock(device);
			mr2::Slice(w, x, D);
		}
		{
			std::lock_guard<std::mutex> lock(device);
			sslm_decode_step_gpu(w.ctx, x, nullptr, D);
		}
		std::atomic<bool> stop{false};
		std::thread poll([&] {  // the caller error: an unclaimed poll racing the abandon on the same handle
			int32_t ready = 0;
			S out = S::SSLM_OK;
			for (int i = 0; i < 2000000 && !stop; ++i) {  // until x's closing slice is observed
				if (sslm_gpu_ready(w.ctx, x, 0, &ready, &out) != S::SSLM_OK || ready == 1) break;
				std::this_thread::yield();
			}
		});
		std::thread abandon([&] {
			for (int i = 0; i < 200; ++i) {
				sslm_gpu_seq_rows_abandon(w.ctx, x);
				std::this_thread::yield();
			}
		});
		// y's chunk runs beside them; it must not be touched.
		// While x's submission is unobserved the singleton guard refuses y's step with SSLM_BUSY, changing nothing
		// (gpu_1p0.h, P2): a retry is the documented response.
		for (int k = 0; k < 1000000 && RowsOpen(y) > 0; ++k) {
			std::lock_guard<std::mutex> lock(device);
			const S st = mr2::Slice(w, y, D);
			if (st == S::SSLM_BUSY) { std::this_thread::yield(); continue; }
			Check(st == S::SSLM_OK, F("rep %d: y slice %s", rep, N(st)));
		}
		Check(RowsOpen(y) == 0, F("rep %d: y never closed", rep));
		abandon.join();
		stop = true;
		poll.join();
		{
			std::lock_guard<std::mutex> lock(device);
			mr2::Ready(w, x);
		}
		Check(Twin(w, yt, ry, Full(m), m) == S::SSLM_OK, "y twin");
	}
	BlobEq(Blob(w, y, "3.3w"), Blob(w, yt, "3.3w"), "the other handle after 40 racing reps");
}

// ------------------------------------------------------------------------------------------------ 5.1 / 5.2
void CellHeadroom(const mr2::Model& m, bool pooled) {
	g_cell = pooled ? "5.2" : "5.1";
	const int32_t V = m.vocab();
	const int32_t cap = 16;
	for (int left : {4, 1, 0}) {
		const std::string tag = F("%s %s %d left of %d, count 8", m.base().c_str(), pooled ? "pooled budget" : "dedicated cap", left, cap);
		World w(m, Shaders());
		if (!w.ok) { Check(false, tag); continue; }
		Pair p = MakePair(w, pooled, cap);
		const std::vector<int32_t> head = Stream(0, cap - left, V, 52), rows = Stream(cap - left, 8, V, 52);
		Check(Twin(w, p.a, head, Full(m), m) == S::SSLM_OK && Twin(w, p.t, head, Full(m), m) == S::SSLM_OK, tag + ": prefix");
		const std::vector<uint8_t> entry = Blob(w, p.a, tag);
		int32_t adm = -1;
		const S st = Embed(w, p.a, rows, &adm);
		if (left == 0) {
			StatusIs(st, pooled ? MR3_POOLED_LIMIT_STATUS : MR3_CAP_STATUS, tag + ": zero rows admitted");
			Check(RowsOpen(p.a) == 0, tag + ": a chunk opened with zero rows");
			BlobEq(Blob(w, p.a, tag), entry, tag + ": nothing written");
			continue;
		}
		Check(st == S::SSLM_OK && adm == left, F("%s: %s admitted %d, want %d", tag.c_str(), N(st), adm, left));
		ChunkRun r;
		r.admitted = adm;
		Advance(w, p.a, r, [&](int) { return m.dpl(); });
		CleanClose(r, p.a, left, tag);
		Check(Twin(w, p.t, std::vector<int32_t>(rows.begin(), rows.begin() + left), Full(m), m) == S::SSLM_OK, tag + ": twin");
		BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag);
	}
}
void Cell51(const mr2::Model& m) { CellHeadroom(m, false); }
void Cell52(const mr2::Model& m) { CellHeadroom(m, true); }

// ------------------------------------------------------------------------------------------------ 5.4 / 8.6
// One batch call; returns the submissions it made (submission seam).
uint64_t Batch(World& w, const std::vector<SslmGpuSequenceHandle*>& seqs, uint32_t budget, std::vector<S>* st, S* call) {
	const SslmGpuMr2Seams s0 = Seams(w);
	std::vector<const SslmGpuAdapterHandle*> none(seqs.size(), nullptr);
	st->assign(seqs.size(), S::SSLM_DEVICE_LOST);
	*call = sslm_decode_step_batch_gpu(w.ctx, seqs.data(), none.data(), static_cast<uint32_t>(seqs.size()), budget, st->data());
	return Seams(w).submissions - s0.submissions;
}
void Cell54(const mr2::Model& m) {
	g_cell = "5.4";
	const int32_t V = m.vocab();
	const uint32_t full = Full(m);
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string tag = F("%s %s", m.base().c_str(), pooled ? "pooled" : "dedicated");
		World w(m, Shaders());
		if (!w.ok) { Check(false, tag); continue; }
		// Members: chunks of M 8, 3 and 1, and two plain one-row tokens.
		const int Ms[3] = {8, 3, 1};
		std::vector<Pair> chunk(3), plain(2);
		for (auto& p : chunk) p = MakePair(w, pooled);
		for (auto& p : plain) p = MakePair(w, pooled);
		std::vector<SslmGpuSequenceHandle*> members;
		for (int i = 0; i < 3; ++i) {
			Twin(w, chunk[i].a, Stream(0, 2, V, 53 + i), full, m);
			Twin(w, chunk[i].t, Stream(0, 2, V, 53 + i), full, m);
			Check(Embed(w, chunk[i].a, Stream(2, Ms[i], V, 53 + i)) == S::SSLM_OK, tag + ": open chunk");
			members.push_back(chunk[i].a);
		}
		for (int i = 0; i < 2; ++i) {
			Twin(w, plain[i].a, Stream(0, 2, V, 57 + i), full, m);
			Twin(w, plain[i].t, Stream(0, 2, V, 57 + i), full, m);
			Check(sslm_gpu_seq_embed_token(w.ctx, plain[i].a, mr2::StreamToken(2, V, 57 + i)) == S::SSLM_OK, tag + ": plain embed");
			members.insert(members.begin() + 2 * i, plain[i].a);  // interleave plain and chunk-open members
		}
		std::vector<S> st;
		S call = S::SSLM_DEVICE_LOST;
		const uint64_t subs = Batch(w, members, static_cast<uint32_t>(members.size()) * full, &st, &call);
		StatusIs(call, S::SSLM_OK, tag + ": batch call");
		for (size_t i = 0; i < st.size(); ++i) StatusIs(st[i], S::SSLM_OK, F("%s: member %zu", tag.c_str(), i));
		Check(subs == members.size(), F("%s: %llu submissions for %zu members (want one per member)", tag.c_str(), (unsigned long long)subs, members.size()));
		for (auto* s : members) Check(mr2::Ready(w, s) == S::SSLM_OK, tag + ": member ready");
		for (int i = 0; i < 3; ++i) {
			Check(RowsOpen(chunk[i].a) == 0 && LastCommitted(chunk[i].a) == Ms[i],
			      F("%s: chunk M%d rows_open %d last_rows_committed %d after the batch", tag.c_str(), Ms[i], RowsOpen(chunk[i].a), LastCommitted(chunk[i].a)));
			Twin(w, chunk[i].t, Stream(2, Ms[i], V, 53 + i), full, m);
			BlobEq(Blob(w, chunk[i].a, tag), Blob(w, chunk[i].t, tag), F("%s: chunk member M%d", tag.c_str(), Ms[i]));
		}
		for (int i = 0; i < 2; ++i) {
			Twin(w, plain[i].t, {mr2::StreamToken(2, V, 57 + i)}, full, m);
			BlobEq(Blob(w, plain[i].a, tag), Blob(w, plain[i].t, tag), F("%s: plain member %d", tag.c_str(), i));
		}
		// The shipped path, plain members only: one submission per member, every member equal (unchanged by MR3).
		std::vector<SslmGpuSequenceHandle*> only;
		for (int i = 0; i < 2; ++i) {
			Check(sslm_gpu_seq_embed_token(w.ctx, plain[i].a, mr2::StreamToken(3, V, 57 + i)) == S::SSLM_OK, tag + ": plain embed 2");
			only.push_back(plain[i].a);
		}
		const uint64_t subs2 = Batch(w, only, 2 * full, &st, &call);
		StatusIs(call, S::SSLM_OK, tag + ": plain-only batch");
		Check(subs2 == only.size() && st[0] == S::SSLM_OK && st[1] == S::SSLM_OK,
		      F("%s: plain-only batch %llu submissions, statuses %s %s (the shipped path: one per member, OK)", tag.c_str(),
		        (unsigned long long)subs2, N(st[0]), N(st[1])));
		for (int i = 0; i < 2; ++i) {
			mr2::Ready(w, plain[i].a);
			Twin(w, plain[i].t, {mr2::StreamToken(3, V, 57 + i)}, full, m);
			BlobEq(Blob(w, plain[i].a, tag), Blob(w, plain[i].t, tag), F("%s: plain-only member %d", tag.c_str(), i));
		}
	}
}
void Cell86(const mr2::Model& m) {
	g_cell = "8.6";
	const int32_t V = m.vocab();
	const uint32_t full = Full(m), D = m.dpl();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	Pair c[2] = {MakePair(w, false), MakePair(w, true)};
	Pair p[2] = {MakePair(w, false), MakePair(w, false)};
	for (int i = 0; i < 2; ++i) {
		Twin(w, c[i].a, Stream(0, 2, V, 60 + i), full, m);
		Twin(w, c[i].t, Stream(0, 2, V, 60 + i), full, m);
		Check(Embed(w, c[i].a, Stream(2, 6, V, 60 + i)) == S::SSLM_OK, "open chunk");
		if (m.layers() >= 2) Check(mr2::Slice(w, c[i].a, D) == S::SSLM_OK, "chunk to a mid layer");
		Check(sslm_gpu_seq_embed_token(w.ctx, p[i].a, mr2::StreamToken(0, V, 62 + i)) == S::SSLM_OK, "plain embed");
	}
	std::vector<S> st;
	S call = S::SSLM_DEVICE_LOST;
	const uint64_t subs = Batch(w, {p[0].a, p[1].a}, 2 * full, &st, &call);
	Check(call == S::SSLM_OK && st[0] == S::SSLM_OK && st[1] == S::SSLM_OK && subs == 2,
	      F("batch beside open chunks: %s, members %s %s, %llu submissions (want 2)", N(call), N(st[0]), N(st[1]), (unsigned long long)subs));
	for (int i = 0; i < 2; ++i) {
		mr2::Ready(w, p[i].a);
		Twin(w, p[i].t, {mr2::StreamToken(0, V, 62 + i)}, full, m);
		BlobEq(Blob(w, p[i].a, "8.6"), Blob(w, p[i].t, "8.6"), F("plain member %d", i));
		Check(RowsOpen(c[i].a) == 6, F("chunk %d changed by the batch beside it", i));
		ChunkRun r;
		r.admitted = 6;
		Advance(w, c[i].a, r, [&](int) { return D; });
		CleanClose(r, c[i].a, 6, F("chunk %d", i));
		Twin(w, c[i].t, Stream(2, 6, V, 60 + i), full, m);
		BlobEq(Blob(w, c[i].a, "8.6"), Blob(w, c[i].t, "8.6"), F("chunk %d (%s)", i, i ? "pooled" : "dedicated"));
	}
}

// ------------------------------------------------------------------------------------------------ 5.5
void Cell55(const mr2::Model& m) {
	g_cell = "5.5";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	for (int mid = 0; mid <= (m.layers() >= 2 ? 1 : 0); ++mid)
		for (int M : {1, 4}) {
			const std::string tag = F("%s M%d %s", m.base().c_str(), M, mid ? "mid layer" : "layer 0");
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			Pair p = MakePair(w, false);
			const std::vector<int32_t> rows = Stream(0, M, V, 64);
			Check(Embed(w, p.a, rows) == S::SSLM_OK, tag + ": embed");
			if (mid) mr2::Slice(w, p.a, D);
			const uint32_t layer = Layer(p.a);
			for (uint32_t b : {0u, 1u, D - 1}) {
				StatusIs(sslm_decode_step_gpu(w.ctx, p.a, nullptr, b), S::SSLM_DISPATCH_BUDGET_TOO_SMALL, F("%s: budget %u", tag.c_str(), b));
				Check(RowsOpen(p.a) == M && Layer(p.a) == layer, F("%s: budget %u changed the chunk", tag.c_str(), b));
			}
			ChunkRun r;
			r.admitted = M;
			Advance(w, p.a, r, [&](int) { return D; });
			CleanClose(r, p.a, M, tag);
			Twin(w, p.t, rows, Full(m), m);
			BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag);
		}
}

// ------------------------------------------------------------------------------------------------ 5.6
void Cell56(const mr2::Model& m) {
	g_cell = "5.6";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl(), L = m.layers();
	for (uint32_t k : {0u, L - 1})
		for (int pooled = 0; pooled <= 1; ++pooled) {
			const std::string tag = F("%s slice %u %s", m.base().c_str(), k, pooled ? "pooled" : "dedicated");
			{
				World w(m, Shaders());
				if (!w.ok) { Check(false, tag); continue; }
				SslmGpuKvPoolHandle* pool = pooled ? w.Pool(16) : nullptr;
				const uint32_t free0 = pooled ? PoolStats(pool).free_pages : 0;
				SslmGpuSequenceHandle* a = pooled ? w.Pooled(pool, 64) : w.Dedicated(64);
				Twin(w, a, Stream(0, 3, V, 65), Full(m), m);
				Check(Embed(w, a, Stream(3, 6, V, 65)) == S::SSLM_OK, tag + ": embed");
				Check(sslm_gpu_test_only_mr2_arm_slice_alloc_failure(w.ctx, static_cast<int32_t>(k)) == S::SSLM_OK, tag + ": arm");
				ArmGpuMapDeviceRemovedQueryInjection();
				ChunkRun r;
				r.admitted = 6;
				Advance(w, a, r, [&](int) { return D; });
				sslm_gpu_test_only_mr2_arm_slice_alloc_failure(w.ctx, -1);
				StatusIs(r.close, S::SSLM_DEVICE_LOST, F("%s: slice %d", tag.c_str(), r.slices - 1));
				Check(r.slices == static_cast<int>(k) + 1, F("%s: failed at slice %d, armed %u", tag.c_str(), r.slices - 1, k));
				int32_t open = -1;
				StatusIs(sslm_gpu_seq_rows_open(a, &open), S::SSLM_OK, tag + ": rows_open after device lost");
				StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, a), S::SSLM_OK, tag + ": rows_abandon after device lost");
				for (auto& s : w.seqs)
					if (s == a) { StatusIs(sslm_gpu_seq_release(w.ctx, a), S::SSLM_OK, tag + ": release after device lost"); s = nullptr; }
				const SslmGpuMr2Seams sm = Seams(w);
				Check(sm.chunks_open == 0 && sm.row_buffer_free == sm.row_buffer_sets, tag + ": the chunk's row set was not freed");
				if (pooled) Check(PoolStats(pool).free_pages == free0, F("%s: pool free %u, start %u", tag.c_str(), PoolStats(pool).free_pages, free0));
			}
			// A fresh context succeeds, and is exact.
			World w(m, Shaders());
			Check(w.ok, tag + ": a fresh context after device lost");
			if (!w.ok) continue;
			Pair p = MakePair(w, pooled);
			const std::vector<int32_t> rows = Stream(3, 6, V, 66);
			CleanClose(RunChunk(w, p.a, rows, [&](int) { return D; }), p.a, 6, tag + " fresh context");
			Twin(w, p.t, rows, Full(m), m);
			BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": fresh context");
		}
}

// ------------------------------------------------------------------------------------------------ 7.1 - 7.4
void Cell71(const mr2::Model& m) {
	g_cell = "7.1";
	const int32_t V = m.vocab();
	for (int M : {1, 3, 8}) {
		World w(m, Shaders());
		if (!w.ok) { Check(false, "world"); continue; }
		SslmGpuSequenceHandle* a = w.Dedicated(64);
		Twin(w, a, Stream(0, 2, V, 67), Full(m), m);
		const SslmGpuMr2Seams s0 = Seams(w);
		Check(Embed(w, a, Stream(2, M, V, 67)) == S::SSLM_OK, F("M%d embed", M));
		const SslmGpuMr2Seams s1 = Seams(w);
		Check(s1.submissions == s0.submissions, F("M%d: embed_rows made %llu submissions (host-only: 0)", M, (unsigned long long)(s1.submissions - s0.submissions)));
		Check(Layer(a) == 0, F("M%d: layer %u after embed_rows", M, Layer(a)));
		int32_t ready = 0;
		S out = S::SSLM_DEVICE_LOST;
		Check(sslm_gpu_ready(w.ctx, a, 0, &ready, &out) == S::SSLM_OK && ready == 1, F("M%d: the handle is not Idle after embed_rows", M));
		StatusIs(mr2::Slice(w, a, m.dpl()), S::SSLM_OK, F("M%d: a decode step on the Idle handle", M));
	}
}
void Cell72(const mr2::Model& m) {
	g_cell = "7.2";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl(), L = m.layers();
	for (uint32_t b : {D, 2 * D, 2 * D - 1, 3 * D + 1, (L + 3) * D})
		for (int M : {1, 4, 8}) {
			World w(m, Shaders());
			if (!w.ok) { Check(false, "world"); continue; }
			SslmGpuSequenceHandle* a = w.Dedicated(64);
			const ChunkRun r = RunChunk(w, a, Stream(0, M, V, 68), [&](int) { return b; });
			CleanClose(r, a, M, F("budget %u M%d", b, M));
			const uint32_t per = b / D;
			uint32_t want = 0;
			for (size_t i = 0; i + 1 < r.layers.size(); ++i) {
				want = std::min(L, want + per);
				Check(r.layers[i] == want, F("budget %u M%d: layer %u after slice %zu, want %u", b, M, r.layers[i], i, want));
			}
			Check(r.slices == static_cast<int>((L + per - 1) / per), F("budget %u M%d: %d slices, want %u", b, M, r.slices, (L + per - 1) / per));
		}
}
void Cell73(const mr2::Model& m) {
	g_cell = "7.3";
	const int32_t V = m.vocab();
	for (uint32_t b : {m.dpl(), Full(m)}) {
		World w(m, Shaders());
		if (!w.ok) { Check(false, "world"); continue; }
		Pair p = MakePair(w, false);
		Twin(w, p.a, Stream(0, 3, V, 69), Full(m), m);
		Twin(w, p.t, Stream(0, 3, V, 69), Full(m), m);
		const std::vector<int32_t> row = Stream(3, 1, V, 69);
		const SslmGpuMr2Seams s0 = Seams(w);
		int32_t adm = 0;
		Check(Embed(w, p.a, row, &adm) == S::SSLM_OK && adm == 1, "count 1 embed");
		SslmGpuMr2ChunkState cs{};
		sslm_gpu_test_only_mr2_chunk_state(p.a, &cs);
		Check(cs.rows == 1 && cs.variant_h == 1, F("count 1: rows %u variant %u (want the one-row body, variant 1)", cs.rows, cs.variant_h));
		ChunkRun r;
		r.admitted = 1;
		Advance(w, p.a, r, [&](int) { return b; });
		CleanClose(r, p.a, 1, F("count 1 budget %u", b));
		Check(Seams(w).widened_lists == s0.widened_lists, "count 1 recorded a widened list");
		Check(mr2::FeedRow(w, p.t, row[0], b, m.layers()) == S::SSLM_OK, "embed_token twin");
		BlobEq(Blob(w, p.a, "7.3"), Blob(w, p.t, "7.3"), F("count 1 vs embed_token (budget %u)", b));
		FinishEq(w, p.a, w, p.t, F("count 1 budget %u", b));
	}
}
void Cell74(const mr2::Model& m) {
	g_cell = "7.4";
	const int32_t V = m.vocab();
	for (int M = 2; M <= 8; ++M) {
		World w(m, Shaders());
		if (!w.ok) { Check(false, "world"); continue; }
		Pair p = MakePair(w, M % 2 == 1);
		const std::vector<int32_t> rows = Stream(0, M, V, 70);
		CleanClose(RunChunk(w, p.a, rows, [&](int) { return m.dpl(); }), p.a, M, F("M%d", M));
		Check(*SslmGpuSeqHandleLayerIndexForBench(p.a) == m.layers(), F("M%d: the last row rests at layer %u, want %u", M, *SslmGpuSeqHandleLayerIndexForBench(p.a), m.layers()));
		Twin(w, p.t, rows, Full(m), m);
		FinishEq(w, p.a, w, p.t, F("M%d", M));
	}
}

// ------------------------------------------------------------------------------------------------ 7.5 - 7.8
void Cell75(const mr2::Model& m) {
	g_cell = "7.5";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl(), L = m.layers();
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {1, 3, 8})
			for (uint32_t at = 0; at < L; ++at) {
				const std::string tag = F("%s M%d abandon at layer %u %s", m.base().c_str(), M, at, pooled ? "pooled" : "dedicated");
				World w(m, Shaders());
				if (!w.ok) { Check(false, tag); continue; }
				Pair p = MakePair(w, pooled);
				const std::vector<int32_t> head = Stream(0, 3, V, 71), rows = Stream(3, M, V, 71);
				Twin(w, p.a, head, Full(m), m);
				Twin(w, p.t, head, Full(m), m);
				const std::vector<uint8_t> entry = Blob(w, p.a, tag);
				Check(Embed(w, p.a, rows) == S::SSLM_OK, tag + ": embed");
				for (uint32_t l = 0; l < at; ++l) Check(mr2::Slice(w, p.a, D) == S::SSLM_OK, tag + ": slice");
				Check(Layer(p.a) == at && RowsOpen(p.a) == M, F("%s: layer %u rows_open %d before the abandon", tag.c_str(), Layer(p.a), RowsOpen(p.a)));
				StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, p.a), S::SSLM_OK, tag);
				Check(RowsOpen(p.a) == 0 && LastCommitted(p.a) == 0, F("%s: rows_open %d last_rows_committed %d", tag.c_str(), RowsOpen(p.a), LastCommitted(p.a)));
				BlobEq(Blob(w, p.a, tag), entry, tag + ": back at entry");
				CleanClose(RunChunk(w, p.a, rows, [&](int) { return D; }), p.a, M, tag + ": next feed");
				Twin(w, p.t, rows, Full(m), m);
				BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": next feed vs twin");
				// Abandon on the closed handle: OK, no change.
				const std::vector<uint8_t> closed = Blob(w, p.a, tag);
				StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, p.a), S::SSLM_OK, tag + ": abandon on a closed handle");
				Check(LastCommitted(p.a) == M, F("%s: abandon on a closed handle changed last_rows_committed to %d", tag.c_str(), LastCommitted(p.a)));
				BlobEq(Blob(w, p.a, tag), closed, tag + ": abandon on a closed handle changed it");
			}
	World w(m, Shaders());
	SslmGpuSequenceHandle* f = w.Dedicated(64);
	const std::vector<uint8_t> fresh = Blob(w, f, "fresh");
	StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, f), S::SSLM_OK, "abandon on a fresh handle");
	BlobEq(Blob(w, f, "fresh"), fresh, "abandon on a fresh handle changed it");
}
void Cell76(const mr2::Model& m) {
	g_cell = "7.6";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl(), L = m.layers();
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {1, 5}) {
			const std::string tag = F("%s M%d %s", m.base().c_str(), M, pooled ? "pooled" : "dedicated");
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			Pair p = MakePair(w, pooled);
			const std::vector<int32_t> rows = Stream(0, M, V, 72);
			const std::vector<uint8_t> entry = Blob(w, p.a, tag);
			Check(Embed(w, p.a, rows) == S::SSLM_OK, tag + ": embed");
			for (uint32_t l = 0; l < L; ++l) {
				SaveRefused(w, p.a, F("%s at layer %u", tag.c_str(), l));
				if (l + 1 < L) Check(mr2::Slice(w, p.a, D) == S::SSLM_OK, tag + ": slice");
			}
			Check(sslm_gpu_seq_rows_abandon(w.ctx, p.a) == S::SSLM_OK, tag + ": abandon");
			BlobEq(Blob(w, p.a, tag), entry, tag + ": save after abandon");
			CleanClose(RunChunk(w, p.a, rows, [&](int) { return D; }), p.a, M, tag);
			Twin(w, p.t, rows, Full(m), m);
			BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": save after close");
		}
}
void Cell77(const mr2::Model& m) {
	g_cell = "7.7";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	if (!LoadAdapter(m, &bytes, &view)) {
		if (m.base() == "pkv_def") Check(false, "no --adapter (pkv_def_r4.adp): 7.7 cannot run");
		else std::printf("7.7: %s has no adapter fixture (cloud adapter legs are pkv_def's); not run here\n", m.base().c_str());
		return;
	}
	if (m.layers() < 2) { Check(false, "7.7 needs L >= 2"); return; }
	enum { kAtoNone, kNoneToA, kAtoA2 };
	for (int c : {kAtoNone, kNoneToA, kAtoA2})
		for (int M : {1, 4}) {
			const char* cn = c == kAtoNone ? "r4 then none" : c == kNoneToA ? "none then r4" : "r4 then another r4 handle";
			const std::string tag = F("M%d %s", M, cn);
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			SslmGpuAdapterHandle *a1 = nullptr, *a2 = nullptr;
			Check(sslm_gpu_adapter_map(w.ctx, w.model, &view, &a1) == S::SSLM_OK && sslm_gpu_adapter_map(w.ctx, w.model, &view, &a2) == S::SSLM_OK,
			      tag + ": adapter maps");
			w.adapters.push_back(a1);
			w.adapters.push_back(a2);
			SslmGpuAdapterHandle* first = c == kNoneToA ? nullptr : a1;
			SslmGpuAdapterHandle* later = c == kAtoNone ? nullptr : c == kNoneToA ? a1 : a2;
			Pair p = MakePair(w, false);
			const std::vector<int32_t> rows = Stream(0, M, V, 73);
			Check(Embed(w, p.a, rows) == S::SSLM_OK, tag + ": embed");
			Check(mr2::Slice(w, p.a, D, first) == S::SSLM_OK, tag + ": layer 0's slice");
			const uint32_t layer = Layer(p.a);
			StatusIs(sslm_decode_step_gpu(w.ctx, p.a, later, D), MR3_MIDTOKEN_ADAPTER_STATUS, tag + ": the changed adapter on a later slice");
			Check(RowsOpen(p.a) == M && Layer(p.a) == layer, tag + ": the refused slice changed the chunk");
			ChunkRun r;
			r.admitted = M;
			Advance(w, p.a, r, [&](int) { return D; }, first);
			CleanClose(r, p.a, M, tag);
			Check(Twin(w, p.t, rows, D, m, first) == S::SSLM_OK, tag + ": twin");
			BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": vs the twin on the first adapter");
		}
}
void Cell78(const mr2::Model& m) {
	g_cell = "7.8";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	const bool have = LoadAdapter(m, &bytes, &view);
	for (int mid = 0; mid <= (m.layers() >= 2 ? 1 : 0); ++mid)
		for (int M : {1, 4}) {
			const std::string tag = F("%s M%d %s", m.base().c_str(), M, mid ? "mid layer" : "layer 0");
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			SslmGpuAdapterHandle* a = nullptr;
			if (have && sslm_gpu_adapter_map(w.ctx, w.model, &view, &a) == S::SSLM_OK) w.adapters.push_back(a);
			Pair p = MakePair(w, false);
			const std::vector<int32_t> rows = Stream(0, M, V, 74);
			Check(Embed(w, p.a, rows) == S::SSLM_OK, tag + ": embed");
			if (mid) mr2::Slice(w, p.a, D);
			StatusIs(sslm_gpu_seq_bind_adapter(w.ctx, p.a, nullptr), S::SSLM_BUSY, tag + ": bind_adapter(none) with a chunk open");
			if (a) StatusIs(sslm_gpu_seq_bind_adapter(w.ctx, p.a, a), S::SSLM_BUSY, tag + ": bind_adapter(r4) with a chunk open");
			ChunkRun r;
			r.admitted = M;
			Advance(w, p.a, r, [&](int) { return D; });
			CleanClose(r, p.a, M, tag);
			Twin(w, p.t, rows, Full(m), m);
			BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag);
			StatusIs(sslm_gpu_seq_bind_adapter(w.ctx, p.a, a), S::SSLM_OK, tag + ": bind_adapter after the close");
		}
}

// ------------------------------------------------------------------------------------------------ 7.13 - 7.16
struct Hidden {
	S st = S::SSLM_OK;
	std::vector<int8_t> codes;
	size_t req = 0;
	int64_t sm = 0, se = 0;
	bool operator==(const Hidden& o) const { return st == o.st && codes == o.codes && req == o.req && sm == o.sm && se == o.se; }
};
Hidden ReadHidden(World& w, SslmGpuSequenceHandle* s) {
	Hidden h;
	uint32_t hs = 0;
	sslm_gpu_model_hidden_size(w.model, &hs);
	h.codes.assign(hs, 0);
	h.st = sslm_gpu_seq_read_prefill_final_hidden(w.ctx, s, h.codes.data(), h.codes.size(), &h.req, &h.sm, &h.se);
	return h;
}
void Cell713(const mr2::Model& m) {
	g_cell = "7.13";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	mr2::Model rm;
	const bool have_refuse = LoadRefuse(m, &rm);
	Check(have_refuse, "no refusal fixture (the refusal leg cannot run)");
	for (int outcome = 0; outcome <= (have_refuse ? 2 : 1); ++outcome)
		for (int bridged = 0; bridged <= 1; ++bridged) {
			const char* on = outcome == 0 ? "close" : outcome == 1 ? "abandon" : "refusal";
			const std::string tag = F("%s %s %s", m.base().c_str(), on, bridged ? "after a bridge prompt" : "no snapshot");
			const mr2::Model& mm = outcome == 2 ? rm : m;
			World probe(mm, Shaders());
			World w(mm, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			SslmGpuSequenceHandle* a = w.Dedicated(64);
			std::vector<int32_t> head = Stream(0, 4, V, 75), rows = Stream(4, 5, V, 75);
			int j = -1;
			if (outcome == 2) {
				const RefusalStream rs = FindRefusalStream(rm, probe, 4, 2);
				if (!rs.ok) { Check(false, tag + ": no refusal stream"); continue; }
				head.assign(rs.toks.begin(), rs.toks.begin() + 4);
				rows.assign(rs.toks.begin() + 4, rs.toks.end());
				j = 2;
			}
			if (bridged) Check(SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, a, head.data(), 4, Full(mm)) == S::SSLM_OK, tag + ": bridge prompt");
			else Twin(w, a, head, Full(mm), mm);
			const Hidden h0 = ReadHidden(w, a);
			Check(bridged ? h0.st == S::SSLM_OK : h0.st == S::SSLM_PREFILL_HIDDEN_UNAVAILABLE, F("%s: the read before returned %s", tag.c_str(), N(h0.st)));
			Check(Embed(w, a, rows) == S::SSLM_OK, tag + ": embed");
			if (outcome == 1) {
				mr2::Slice(w, a, D);
				Check(sslm_gpu_seq_rows_abandon(w.ctx, a) == S::SSLM_OK, tag + ": abandon");
			} else {
				ChunkRun r;
				r.admitted = static_cast<int32_t>(rows.size());
				Advance(w, a, r, [&](int) { return mm.dpl(); });
				StatusIs(r.close, outcome == 2 ? S::SSLM_SEQUENCE_REJECTED : S::SSLM_OK, tag + ": the close");
				if (j >= 0) Check(LastCommitted(a) == j, F("%s: committed %d, want %d", tag.c_str(), LastCommitted(a), j));
			}
			const Hidden h1 = ReadHidden(w, a);
			Check(h1 == h0, F("%s: the snapshot read changed (%s -> %s, or its bytes)", tag.c_str(), N(h0.st), N(h1.st)));
		}
}
void Cell714(const mr2::Model& m) {
	g_cell = "7.14";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	for (int M : {1, 4, 8}) {
		const std::string tag = F("%s M%d", m.base().c_str(), M);
		World w(m, Shaders());
		if (!w.ok) { Check(false, tag); continue; }
		SslmGpuSequenceHandle* a = w.Dedicated(64);
		SslmGpuSequenceHandle* t = w.Dedicated(64);
		const std::vector<int32_t> head = Stream(0, 4, V, 76), rows = Stream(4, M, V, 76);
		// The armed rest: a bridge prompt leaves ready_for_logits set (blob tail ready_word = 1).
		Check(SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, a, head.data(), 4, Full(m)) == S::SSLM_OK, tag + ": bridge prompt");
		const std::vector<uint8_t> armed = Blob(w, a, tag);
		// Rollback restores the flag: abandon after an armed rest gives back the armed blob.
		Check(Embed(w, a, rows) == S::SSLM_OK, tag + ": embed");
		mr2::Slice(w, a, D);
		if (RowsOpen(a) > 0) Check(sslm_gpu_seq_rows_abandon(w.ctx, a) == S::SSLM_OK, tag + ": abandon");
		BlobEq(Blob(w, a, tag), armed, tag + ": abandon after an armed rest restores the armed blob (ready_word 1)");
		// embed_rows clears it: after the close the whole blob, ready_word 0 included, equals the twin that fed the same
		// tokens one row at a time (whose flag was never armed; see the record: embed_token would leave it set, E25).
		CleanClose(RunChunk(w, a, rows, [&](int) { return D; }), a, M, tag);
		Twin(w, t, head, Full(m), m);
		Twin(w, t, rows, Full(m), m);
		BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag + ": after the close (ready_for_logits cleared)");
		// A following decode-bridge call takes the not-armed branch on both (embeds its token), not a stale finish.
		const int32_t nxt = mr2::StreamToken(40, V, 76);
		int32_t ta = -9, tt = -9;
		const S sa = SslmGpuSeqDecodeStepForG5Bridge(w.ctx, a, nxt, Full(m), &ta);
		const S st = SslmGpuSeqDecodeStepForG5Bridge(w.ctx, t, nxt, Full(m), &tt);
		Check(sa == S::SSLM_OK && st == S::SSLM_OK && ta == tt, F("%s: decode bridge %s token %d, twin %s token %d", tag.c_str(), N(sa), ta, N(st), tt));
		BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag + ": after the decode-bridge call");
	}
}
void Cell715(const mr2::Model& m) {
	g_cell = "7.15";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const char* k = pooled ? "pooled" : "dedicated";
		Pair p = MakePair(w, pooled);
		CleanClose(RunChunk(w, p.a, Stream(0, 6, V, 77), [&](int) { return D; }), p.a, 6, F("%s close", k));
		Check(LastCommitted(p.a) == 6, F("%s: %d after a clean close of 6", k, LastCommitted(p.a)));
		const std::vector<uint8_t> saved = Blob(w, p.a, k);
		Check(Embed(w, p.a, Stream(6, 3, V, 77)) == S::SSLM_OK, "embed");
		Check(sslm_gpu_seq_rows_abandon(w.ctx, p.a) == S::SSLM_OK && LastCommitted(p.a) == 0, F("%s: %d after an abandon", k, LastCommitted(p.a)));
		CleanClose(RunChunk(w, p.a, Stream(6, 3, V, 77), [&](int) { return D; }), p.a, 3, F("%s close 2", k));
		Check(sslm_gpu_seq_reset(w.ctx, p.a) == S::SSLM_OK && LastCommitted(p.a) == 0, F("%s: %d after a reset", k, LastCommitted(p.a)));
		SslmGpuSequenceHandle* rs = nullptr;
		uint32_t shared = 0;
		const S st = pooled ? sslm_gpu_seq_restore_into_pool(w.ctx, w.model, p.pa, saved.data(), saved.size(), nullptr, &rs, &shared)
		                    : sslm_gpu_seq_restore(w.ctx, w.model, saved.data(), saved.size(), &rs);
		Check(st == S::SSLM_OK && rs && LastCommitted(rs) == 0, F("%s: restore %s, last_rows_committed %d", k, N(st), rs ? LastCommitted(rs) : -1));
		if (rs) w.seqs.push_back(rs);
	}
	mr2::Model rm;
	if (!LoadRefuse(m, &rm)) { Check(false, "no refusal fixture"); return; }
	World probe(rm, Shaders()), rw(rm, Shaders());
	for (int j : {0, 4}) {
		const RefusalStream rs = FindRefusalStream(rm, probe, 0, j);
		if (!rs.ok) { Check(false, F("no refusal stream j%d", j)); continue; }
		SslmGpuSequenceHandle* a = rw.Dedicated(64);
		const ChunkRun r = RunChunk(rw, a, rs.toks, [&](int) { return rm.dpl(); });
		Check(r.close == S::SSLM_SEQUENCE_REJECTED && LastCommitted(a) == j, F("refusal j%d: %s, last_rows_committed %d", j, N(r.close), LastCommitted(a)));
		Check(RowsOpen(a) == 0, F("refusal j%d: rows_open %d", j, RowsOpen(a)));
	}
}
void Cell716(const mr2::Model& m) {
	g_cell = "7.16";
	const int32_t V = m.vocab();
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int full = 0; full <= 1; ++full)
			for (int M = 2; M <= 8; ++M) {
				const std::string tag = F("%s M%d %s %s", m.base().c_str(), M, pooled ? "pooled" : "dedicated", full ? "full depth" : "one layer per call");
				World w(m, Shaders());
				if (!w.ok) { Check(false, tag); continue; }
				Pair p = MakePair(w, pooled);
				const std::vector<int32_t> head = Stream(0, 3, V, 78), rows = Stream(3, M, V, 78);
				Twin(w, p.a, head, Full(m), m);
				Twin(w, p.t, head, Full(m), m);
				CleanClose(RunChunk(w, p.a, rows, [&](int) { return full ? Full(m) : m.dpl(); }), p.a, M, tag);
				Twin(w, p.t, rows, Full(m), m);
				BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": the mirror after the close (every committed row)");
			}
}

// ------------------------------------------------------------------------------------------------ 8.1 / 8.2
void Cell81(const mr2::Model& m) {
	g_cell = "8.1";
	const int32_t V = m.vocab();
	for (int M : {3, 8}) {
		const std::string tag = F("%s M%d", m.base().c_str(), M);
		World w(m, Shaders());
		if (!w.ok) { Check(false, tag); continue; }
		SslmGpuKvPoolHandle* pool = w.Pool(32);
		const int B = static_cast<int>(PoolStats(pool).page_positions);
		SslmGpuPrefixHandle* pre = FrozenPrefix(w, m, pool, B + 3, 79);
		SslmGpuSequenceHandle* a = w.Pooled(pool, 3 * B);
		SslmGpuSequenceHandle* t = w.Pooled(pool, 3 * B);
		Check(pre && sslm_gpu_seq_adopt_prefix(w.ctx, a, pre) == S::SSLM_OK && sslm_gpu_seq_adopt_prefix(w.ctx, t, pre) == S::SSLM_OK, tag + ": adopt");
		const std::vector<int32_t> rows = Stream(100, M, V, 79);
		CleanClose(RunChunk(w, a, rows, [&](int) { return m.dpl(); }), a, M, tag);
		DisarmTwin(w, t);
		Check(Twin(w, t, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
		BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag + ": adopted then embed_rows vs adopted then one-row");
		FinishEq(w, a, w, t, tag);
		if (pre) sslm_gpu_prefix_release(w.ctx, pre);
	}
}
std::vector<uint8_t> PrefixBlob(World& w, SslmGpuPrefixHandle* p) {
	size_t need = 0;
	sslm_gpu_prefix_save(w.ctx, p, nullptr, &need);
	std::vector<uint8_t> b(need ? need : 1);
	size_t sz = b.size();
	if (sslm_gpu_prefix_save(w.ctx, p, b.data(), &sz) != S::SSLM_OK) return {};
	b.resize(sz);
	return b;
}
void Cell82(const mr2::Model& m) {
	g_cell = "8.2";
	const int32_t V = m.vocab();
	for (int M : {3, 5, 8}) {
		const std::string tag = F("%s M%d", m.base().c_str(), M);
		World wa(m, Shaders()), wb(m, Shaders());
		if (!wa.ok || !wb.ok) { Check(false, tag); continue; }
		struct Side {
			SslmGpuKvPoolHandle* pool = nullptr;
			SslmGpuPrefixHandle* pre = nullptr;
			SslmGpuSequenceHandle *s = nullptr, *other = nullptr;
		} A, Bs;
		auto build = [&](World& w, Side& sd) {
			sd.pool = w.Pool(32);
			const int B = static_cast<int>(PoolStats(sd.pool).page_positions);
			sd.pre = FrozenPrefix(w, m, sd.pool, B + B / 2, 80);
			sd.s = w.Pooled(sd.pool, 3 * B);
			sd.other = w.Pooled(sd.pool, 3 * B);
			return sd.pre && sslm_gpu_seq_adopt_prefix(w.ctx, sd.s, sd.pre) == S::SSLM_OK &&
			       sslm_gpu_seq_adopt_prefix(w.ctx, sd.other, sd.pre) == S::SSLM_OK && Twin(w, sd.other, Stream(0, 3, V, 81), Full(m), m) == S::SSLM_OK;
		};
		if (!build(wa, A) || !build(wb, Bs)) { Check(false, tag + ": construction"); continue; }
		const std::vector<uint8_t> pre_before = PrefixBlob(wa, A.pre), other_before = Blob(wa, A.other, tag);
		const std::vector<int32_t> rows = Stream(200, M, V, 82);
		CleanClose(RunChunk(wa, A.s, rows, [&](int) { return Full(m); }), A.s, M, tag);
		DisarmTwin(wb, Bs.s);
		Check(Twin(wb, Bs.s, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
		BlobEq(PrefixBlob(wa, A.pre), pre_before, tag + ": the prefix's own save bytes");
		BlobEq(Blob(wa, A.other, tag), other_before, tag + ": the other adopter");
		BlobEq(Blob(wa, A.s, tag), Blob(wb, Bs.s, tag), tag + ": the chunk's adopter vs the twin world's");
		std::vector<uint8_t> ka, kb;
		Check(sslm_gpu_test_only_mr2_device_kv(wa.ctx, A.pool, nullptr, &ka) == S::SSLM_OK &&
		          sslm_gpu_test_only_mr2_device_kv(wb.ctx, Bs.pool, nullptr, &kb) == S::SSLM_OK,
		      tag + ": device pool read-back");
		Check(mr2::FirstDiff(ka, kb) < 0, F("%s: device pool differs from the twin world's at byte %ld", tag.c_str(), mr2::FirstDiff(ka, kb)));
		sslm_gpu_prefix_release(wa.ctx, A.pre);
		sslm_gpu_prefix_release(wb.ctx, Bs.pre);
	}
}

// ------------------------------------------------------------------------------------------------ 8.8 / 8.9
void Cell88(const mr2::Model& m) {
	g_cell = "8.8";
	const int32_t V = m.vocab();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	if (!SslmGpuModelHasSchemasForG5Bridge(w.model)) {
		if (m.base() == "pkv_def") Check(false, "pkv_def carries no schema (it should: S8)");
		else std::printf("8.8: %s carries no schema; the cloud leg is pkv_def's\n", m.base().c_str());
		return;
	}
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {1, 4, 8}) {
			const std::string tag = F("M%d %s", M, pooled ? "pooled" : "dedicated");
			Pair p = MakePair(w, pooled);
			Check(SslmGpuSeqSetSchemaForG5Bridge(w.ctx, p.a, 0) == S::SSLM_OK && SslmGpuSeqSetSchemaForG5Bridge(w.ctx, p.t, 0) == S::SSLM_OK, tag + ": bind");
			const std::vector<int32_t> rows = Stream(0, M, V, 83);
			const uint32_t walk0 = SslmGpuSeqWalkStateForG5Bridge(p.a);
			CleanClose(RunChunk(w, p.a, rows, [&](int) { return m.dpl(); }), p.a, M, tag);
			Twin(w, p.t, rows, Full(m), m);
			int32_t bound = -1;
			Check(SslmGpuSeqSchemaBoundForG5Bridge(w.ctx, p.a, &bound) == S::SSLM_OK && bound == 1, tag + ": the binding was dropped");
			Check(SslmGpuSeqWalkStateForG5Bridge(p.a) == walk0 && walk0 == SslmGpuSeqWalkStateForG5Bridge(p.t), tag + ": the walk moved");
			BlobEq(Blob(w, p.a, tag), Blob(w, p.t, tag), tag + ": SLM tail (binding, walk, ready) and all");
		}
}
void Cell89(const mr2::Model& m) {
	g_cell = "8.9";
	const int32_t V = m.vocab();
	const uint32_t D = m.dpl();
	for (int order = 0; order <= 1; ++order)
		for (int M : {1, 5}) {
			const std::string tag = F("%s M%d %s", m.base().c_str(), M, order ? "embed_rows then bridge" : "bridge then embed_rows");
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			SslmGpuSequenceHandle* a = w.Dedicated(64);
			SslmGpuSequenceHandle* t = w.Dedicated(64);
			const std::vector<int32_t> A = Stream(0, 4, V, 84), Bt = Stream(4, M, V, 84);
			if (order == 0) {
				// The bridge's armed flag is what embed_rows clears; embed_token would leave it set (E25), so the twin
				// feeds both segments one row at a time (P4.3: the bridge is byte-equal to one-row feeding).
				Check(SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, a, A.data(), 4, Full(m)) == S::SSLM_OK, tag + ": bridge");
				CleanClose(RunChunk(w, a, Bt, [&](int) { return D; }), a, M, tag);
				Twin(w, t, A, Full(m), m);
				Twin(w, t, Bt, Full(m), m);
			} else {
				CleanClose(RunChunk(w, a, A, [&](int) { return D; }), a, 4, tag);
				Check(SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, a, Bt.data(), M, Full(m)) == S::SSLM_OK, tag + ": bridge after the chunk");
				Twin(w, t, A, Full(m), m);
				Check(SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, t, Bt.data(), M, Full(m)) == S::SSLM_OK, tag + ": twin bridge");
			}
			BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag);
			FinishEq(w, a, w, t, tag);
		}
}

// ------------------------------------------------------------------------------------------------ 9.1
void Cell91(const mr2::Model& m) {
	g_cell = "9.1";
	const int32_t V = m.vocab();
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {3, 8}) {
			const std::string tag = F("%s M%d %s", m.base().c_str(), M, pooled ? "SLM6" : "SLM5");
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			Pair p = MakePair(w, pooled);
			const std::vector<int32_t> head = Stream(0, 5, V, 85), rows = Stream(5, M, V, 85);
			Twin(w, p.a, head, Full(m), m);
			Twin(w, p.t, head, Full(m), m);
			CleanClose(RunChunk(w, p.a, rows, [&](int) { return m.dpl(); }), p.a, M, tag);
			Twin(w, p.t, rows, Full(m), m);
			const std::vector<uint8_t> ba = Blob(w, p.a, tag), bt = Blob(w, p.t, tag);
			BlobEq(ba, bt, tag + ": save after the close");
			SslmGpuSequenceHandle *ra = nullptr, *rt = nullptr;
			uint32_t sh = 0;
			auto restore = [&](const std::vector<uint8_t>& b, SslmGpuKvPoolHandle* pool, SslmGpuSequenceHandle** out) {
				return pooled ? sslm_gpu_seq_restore_into_pool(w.ctx, w.model, pool, b.data(), b.size(), nullptr, out, &sh)
				              : sslm_gpu_seq_restore(w.ctx, w.model, b.data(), b.size(), out);
			};
			Check(restore(ba, p.pa, &ra) == S::SSLM_OK && restore(bt, p.pt, &rt) == S::SSLM_OK && ra && rt, tag + ": restore");
			if (!ra || !rt) continue;
			w.seqs.push_back(ra);
			w.seqs.push_back(rt);
			BlobEq(Blob(w, ra, tag), ba, tag + ": the restored handle re-saves its blob");
			// The continuation: finish, then 8 greedy decode-bridge tokens on each.
			int32_t xa = -1, xt = -1;
			Check(SslmGpuSeqFinishTokenForG5Bridge(w.ctx, ra, &xa) == S::SSLM_OK && SslmGpuSeqFinishTokenForG5Bridge(w.ctx, rt, &xt) == S::SSLM_OK &&
			          xa == xt,
			      F("%s: restored finish %d vs twin %d", tag.c_str(), xa, xt));
			for (int i = 0; i < 8 && xa >= 0; ++i) {
				int32_t oa = -1, ot = -1;
				SslmGpuSeqDecodeStepForG5Bridge(w.ctx, ra, xa, Full(m), &oa);
				SslmGpuSeqDecodeStepForG5Bridge(w.ctx, rt, xt, Full(m), &ot);
				Check(oa == ot, F("%s: continuation token %d: %d vs twin %d", tag.c_str(), i, oa, ot));
				xa = oa;
				xt = ot;
			}
			BlobEq(Blob(w, ra, tag), Blob(w, rt, tag), tag + ": after the continuation");
		}
}

}  // namespace

int main(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		auto val = [&](const char* k) { return s.substr(std::strlen(k)); };
		if (s.rfind("--model=", 0) == 0) g_opt.model = val("--model=");
		else if (s.rfind("--fixtures=", 0) == 0) g_opt.fixtures = val("--fixtures=");
		else if (s.rfind("--adapter=", 0) == 0) g_opt.adapter = val("--adapter=");
		else if (s.rfind("--shaders=", 0) == 0) g_opt.shaders = val("--shaders=");
		else if (s.rfind("--cycles=", 0) == 0) g_opt.cycles = std::atoi(val("--cycles=").c_str());
		else if (s.rfind("--race-iterations=", 0) == 0) g_opt.race_iterations = std::atoi(val("--race-iterations=").c_str());
		else if (s.rfind("--cells=", 0) == 0) {
			const std::string v = val("--cells=");
			for (size_t p = 0; p < v.size();) {
				const size_t q = v.find(',', p);
				g_opt.cells.insert(v.substr(p, q - p));
				p = q == std::string::npos ? v.size() : q + 1;
			}
		} else {
			std::printf("usage: mr3_cells --model=PATH [--fixtures=DIR] [--adapter=PATH] [--shaders=DIR] [--cells=..] [--cycles=N] [--race-iterations=N]\n");
			return 2;
		}
	}
	mr2::Model m;
	if (!m.Load(g_opt.model)) return 1;
#ifdef MR3_UNPUBLISHED
	std::printf("cells: MR3_UNPUBLISHED (gpu_1p0.h publishes neither the verbs nor both provisional statuses): MR2's internal entry is "
	            "driven, and every check of a provisional status is red\n");
#endif
	const std::vector<std::pair<const char*, void (*)(const mr2::Model&)>> cells = {
	    {"2.1", Cell21}, {"2.2", Cell22}, {"7.1", Cell71}, {"7.2", Cell72}, {"7.3", Cell73}, {"7.4", Cell74},
	    {"7.5", Cell75}, {"7.6", Cell76}, {"7.7", Cell77}, {"7.8", Cell78}, {"7.13", Cell713}, {"7.14", Cell714},
	    {"7.15", Cell715}, {"7.16", Cell716}, {"5.1", Cell51}, {"5.2", Cell52}, {"5.5", Cell55}, {"5.4", Cell54},
	    {"8.6", Cell86}, {"8.1", Cell81}, {"8.2", Cell82}, {"8.8", Cell88}, {"8.9", Cell89}, {"9.1", Cell91},
	    {"1.3", Cell13}, {"2.4", Cell24}, {"3.1", Cell31}, {"3.2", Cell32}, {"3.3w", Cell33w}, {"5.6", Cell56},
	    {"1.1", Cell11}};
	bool all = true;
	for (const auto& c : cells) {
		if (!g_opt.want(c.first)) continue;
		g_cell = c.first;
		const auto t0 = std::chrono::steady_clock::now();
		c.second(m);
		const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		// 2.1 also reports 9.3.
		for (const std::string id : {std::string(c.first), std::string(std::strcmp(c.first, "2.1") == 0 ? "9.3" : "")}) {
			if (id.empty()) continue;
			const int f = g_fail[id];
			if (f) std::printf("RED cell %s: %d failures of %d checks (%.0f s)\n", id.c_str(), f, g_checks[id], s);
			else std::printf("cell %s: PASS (%d checks, %.0f s)\n", id.c_str(), g_checks[id], s);
			all = all && f == 0;
		}
	}
	std::printf("cells: %s\n", all ? "PASS" : "RED");
	return all ? 0 : 1;
}
