// Multi-row GPU plan (rev 1.3.2), slice MR2 (the recorder): the cloud cells of the MR2 red suite, on the software GPU.
//
// Drives the recorder through mr2_entry.h (plan §4.1's shapes: embed_rows PROMPT, then sslm_decode_step_gpu slices
// whose budget counts DISPATCHES, the close in the sslm_gpu_ready that finishes the last layer). Verdicts come only
// from statuses, whole save blobs against the one-row twin (the same stream through embed_token and the decode step,
// in the same process), the CPU engine (6.8), and the seams of mr2_entry.h, each commissioned by a mutant
// (plan §7 grading rule). "Per-layer" is DispatchesPerLayer: 24, or 25 on pkv_qk.
//
// Cells (plan §7; [cloud] parts only, box parts are named in the record):
//   1.2   no recorder allocation after warm-up; the row-buffer free list never exceeds the peak of open chunks
//   1.4   three open chunks (M 8, 3, 5) interleaved one layer at a time, a fourth sequence's one-row slice between
//   1.5   one handle: refused chunk (j = 2), reset, M = 8, M = 3, each equal to its twin   [needs *_refuse fixture]
//   4.1   M 1..8 x {dedicated, pooled} x {no adapter, pkv_def_r4.adp on pkv_def}           (cloud matrix)
//   4.2   budget L x per-layer: one full-depth widened list per chunk, 200 chunks of M = 8, then equal to the twin
//   4.6   (cloud leg of the box cell) chunks ending at cap and cap - 1, and straddling a page at row offsets 0..7
//   4.6b  padded rows at the edges (M 3/5/6/7; p0 + M = budget, and on a page boundary) on a pool shared with a
//         second sequence and a shared prefix: the whole device pool equals the twin world's
//   5.3   an allocation failure at slice k: SSLM_GPU_ALLOCATION_FAILED, the handle back at entry, a retry is equal
//   6.1   per-row byte equality: M 1..8 x start offset s 0..7, blob, finish token and post-finish blob (+ 4.10 hashes)
//   6.2   chunk invariance: random partitions of a 512-token stream into chunks of 1..8, random slice budgets in
//         [1, L] x per-layer; the final blob and a 64-token continuation equal the twin's
//   6.3   slice invariance: budgets {1,2,3,4,7,L} x per-layer and 3 x per-layer - 1 (records 2 layers: layer seam)
//   6.4   padded rows (M 3/5/6/7) equal one-at-a-time feeding; the device K/V outside the real rows is unchanged
//         on a canary background; the counters equal the twin's
//   6.6   guard at row j, j 0..7, two slicings, SLM5 and SLM6, fresh context per run: status, last_rows_committed = j,
//         whole blob equal to the twin read after its own refusal; two-refusal and post-refusal-saturation variants
//   6.7   every row saturates without refusing: counters equal the twin's, and every twin row did saturate
//   6.8   CPU parity: a prompt read by chunks then greedy decode equals the CPU engine's tokens
//   1.1r  (MR2 part of MR3's 1.1, MX8's target) rollback paths on a pooled sequence over an adopted prefix: after
//         5.3's failure and after rows_abandon the pool's free and shared counts, the blob and the holder's mapped page
//         count (sslm_gpu_test_only_mr2_holder_mapped; the entry is on a page boundary) return to entry
//   4.13  [box] context create time is printed as a diagnostic only
//
// Usage: mr2_cells.exe --model=PATH [--fixtures=DIR] [--adapter=PATH] [--shaders=DIR] [--cells=6.1,6.6,...]
//                      [--partitions=N]
//   --fixtures  where <model>_refuse.sslm (mr2_fixtures.py) lives; 6.6 and 1.5 run on it
//   --adapter   pkv_def_r4.adp (tools/gen_multirow_adapter_fixture.py) for 4.1's adapter legs
// Prints "cell <id>: PASS" or "RED cell <id>: <n> failures" per cell; exit 0 only when every cell run passes.
#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
#include <new>
#include <random>
#include <set>

#include "mr2_common.h"
#include "mr2_entry.h"
#include "superslm/sslm_abi.h"

namespace {

using mr2::S;
using mr2::World;

struct Opt {
	std::string model, fixtures, adapter, shaders;
	std::set<std::string> cells;
	int partitions = 200;
	bool want(const char* c) const { return cells.empty() || cells.count(c); }
};
Opt g_opt;

// ------------------------------------------------------------------------------------------------ reporting
std::map<std::string, int> g_fail, g_checks;
std::string g_cell;
void Check(bool ok, const std::string& what) {
	++g_checks[g_cell];
	if (!ok) {
		++g_fail[g_cell];
		if (g_fail[g_cell] <= 12) std::printf("FAIL %s: %s\n", g_cell.c_str(), what.c_str());
	}
}
std::string F(const char* fmt, ...) {
	char b[512];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(b, sizeof b, fmt, ap);
	va_end(ap);
	return b;
}
void BlobEq(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::string& tag) {
	const long d = mr2::FirstDiff(got, want);
	Check(d < 0, F("%s: blob differs from the twin's (sizes %zu/%zu, first byte %ld)", tag.c_str(), got.size(), want.size(), d));
}

// ------------------------------------------------------------------------------------------------ drivers
struct ChunkRun {
	S embed = S::SSLM_OK, close = S::SSLM_OK;
	int32_t admitted = 0;
	int slices = 0;
	bool closed = false;
	std::vector<uint32_t> layers;  // the layer seam after each slice
};
int32_t RowsOpen(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_rows_open(s, &r) == S::SSLM_OK ? r : -1;
}
int32_t LastCommitted(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_last_rows_committed(s, &r) == S::SSLM_OK ? r : -1;
}
// prefix_prefill fills one token per call (it reports how many in `used`), so a prefix of n tokens is n calls.
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
S EmbedRows(World& w, SslmGpuSequenceHandle* s, const int32_t* toks, int32_t m, int32_t* admitted) {
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, toks, m, 0u};
	return sslm_gpu_seq_embed_rows(w.ctx, s, &spec, admitted);
}
// embed_rows, then slices of budget(i) dispatches until the chunk closes (rows_open == 0) or a call fails.
ChunkRun RunChunk(World& w, SslmGpuSequenceHandle* s, const std::vector<int32_t>& toks,
                  const std::function<uint32_t(int)>& budget, SslmGpuAdapterHandle* adapter = nullptr) {
	ChunkRun r;
	r.embed = EmbedRows(w, s, toks.data(), static_cast<int32_t>(toks.size()), &r.admitted);
	if (r.embed != S::SSLM_OK) return r;
	for (int i = 0; i < 64; ++i) {
		S st = sslm_decode_step_gpu(w.ctx, s, adapter, budget(i));
		if (st == S::SSLM_OK) st = mr2::Ready(w, s);
		++r.slices;
		SslmGpuMr2ChunkState cs{};
		sslm_gpu_test_only_mr2_chunk_state(s, &cs);
		r.layers.push_back(cs.layer);
		if (st != S::SSLM_OK) {
			r.close = st;
			r.closed = RowsOpen(s) == 0;
			return r;
		}
		if (RowsOpen(s) == 0) {
			r.closed = true;
			return r;
		}
	}
	r.close = S::SSLM_DEVICE_LOST;  // never closed
	return r;
}
uint32_t Full(const mr2::Model& m) { return m.full_budget(); }
// The one-row twin of a stream: FeedRow per token at `budget`.
S Twin(World& w, SslmGpuSequenceHandle* s, const std::vector<int32_t>& toks, uint32_t budget, const mr2::Model& m,
       SslmGpuAdapterHandle* adapter = nullptr) {
	for (int32_t t : toks) {
		const S st = mr2::FeedRow(w, s, t, budget, m.layers(), adapter);
		if (st != S::SSLM_OK) return st;
	}
	return S::SSLM_OK;
}
std::vector<int32_t> Stream(int from, int n, int32_t vocab, uint32_t stream = 0) {
	std::vector<int32_t> v;
	for (int i = 0; i < n; ++i) v.push_back(mr2::StreamToken(from + i, vocab, stream));
	return v;
}
std::vector<uint8_t> Blob(World& w, SslmGpuSequenceHandle* s, const std::string& tag) {
	std::vector<uint8_t> b;
	S st = S::SSLM_OK;
	if (!mr2::Save(w, s, &b, &st)) Check(false, F("%s: save returned %s", tag.c_str(), mr2::Name(st)));
	return b;
}
// A clean chunk's checks: closed OK, all rows admitted and committed.
void CleanClose(const ChunkRun& r, SslmGpuSequenceHandle* s, int32_t m, const std::string& tag) {
	Check(r.embed == S::SSLM_OK && r.admitted == m, F("%s: embed_rows %s admitted %d of %d", tag.c_str(), mr2::Name(r.embed), r.admitted, m));
	Check(r.close == S::SSLM_OK && r.closed, F("%s: chunk %s after %d slices (closed %d)", tag.c_str(), mr2::Name(r.close), r.slices, r.closed));
	Check(LastCommitted(s) == m, F("%s: last_rows_committed %d, want %d", tag.c_str(), LastCommitted(s), m));
}
// Finish both handles' resting rows and compare the token and the blobs after (6.1's logits/token leg).
void FinishEq(World& wa, SslmGpuSequenceHandle* a, World& wb, SslmGpuSequenceHandle* b, const std::string& tag) {
	int32_t ta = -9, tb = -9;
	const S sa = SslmGpuSeqFinishTokenForG5Bridge(wa.ctx, a, &ta);
	const S sb = SslmGpuSeqFinishTokenForG5Bridge(wb.ctx, b, &tb);
	Check(sa == S::SSLM_OK && sb == S::SSLM_OK && ta == tb,
	      F("%s: finish %s token %d, twin %s token %d", tag.c_str(), mr2::Name(sa), ta, mr2::Name(sb), tb));
	BlobEq(Blob(wa, a, tag), Blob(wb, b, tag), tag + " after finish");
}

const char* Shaders() { return g_opt.shaders.empty() ? nullptr : g_opt.shaders.c_str(); }

// ------------------------------------------------------------------------------------------------ 6.1 (+4.10)
void Cell61(const mr2::Model& m) {
	g_cell = "6.1";
	const int32_t V = m.vocab();
	// Every M x every start offset s (p0 mod 8 takes every value), plus MX1's host-half cases (code review N1):
	// p0 = 256 / NQH - 2, where row 0's attention items fit one 256-thread group and a later real row's do not.
	std::vector<std::pair<int, int>> cases;
	for (int M = 1; M <= 8; ++M)
		for (int s = 0; s <= 7; ++s) cases.push_back({M, s});
	for (int M : {4, 5, 8}) cases.push_back({M, static_cast<int>(256 / m.nqh()) - 2});
	for (const auto& [M, s] : cases) {
		{
			const std::string tag = F("%s M%d s%d", m.base().c_str(), M, s);
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag + ": world"); continue; }
			SslmGpuSequenceHandle* a = w.Dedicated(256);
			SslmGpuSequenceHandle* b = w.Dedicated(256);
			const std::vector<int32_t> pre = Stream(0, s, V), rows = Stream(s, M, V);
			Check(Twin(w, a, pre, Full(m), m) == S::SSLM_OK && Twin(w, b, pre, Full(m), m) == S::SSLM_OK, tag + ": prefix");
			const ChunkRun r = RunChunk(w, a, rows, [&](int) { return Full(m); });
			CleanClose(r, a, M, tag);
			Check(Twin(w, b, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
			const std::vector<uint8_t> ba = Blob(w, a, tag), bb = Blob(w, b, tag);
			BlobEq(ba, bb, tag);
			Check(mr2::ReadCounters(a) == mr2::ReadCounters(b),
			      F("%s: counters %s, twin %s", tag.c_str(), mr2::ReadCounters(a).Str().c_str(), mr2::ReadCounters(b).Str().c_str()));
			std::printf("hash 4.10 %s 6.1 M%d s%d %016llx\n", m.base().c_str(), M, s, (unsigned long long)mr2::Fnv(ba));
			FinishEq(w, a, w, b, tag);
		}
	}
}

// ------------------------------------------------------------------------------------------------ 4.1
void Cell41(const mr2::Model& m) {
	g_cell = "4.1";
	const int32_t V = m.vocab();
	std::vector<uint8_t> adp_bytes;
	superslm::SslmModelView adp_view{};
	bool have_adp = false;
	if (!g_opt.adapter.empty() && m.base() == "pkv_def") {
		std::string err;
		have_adp = mr2::ReadFileBytes(g_opt.adapter, &adp_bytes) &&
		           superslm::SslmModel::Load(adp_bytes.data(), adp_bytes.size(), adp_view, &err) == superslm::SslmModelStatus::Ok;
		Check(have_adp, "cannot load the adapter " + g_opt.adapter);
	}
	for (int with_adp = 0; with_adp <= (have_adp ? 1 : 0); ++with_adp)
		for (int pooled = 0; pooled <= 1; ++pooled)
			for (int M = 1; M <= 8; ++M) {
				const std::string tag = F("%s M%d %s %s", m.base().c_str(), M, pooled ? "pooled" : "dedicated", with_adp ? "r4" : "no-adapter");
				World w(m, Shaders());
				if (!w.ok) { Check(false, tag + ": world"); continue; }
				SslmGpuAdapterHandle* adp = nullptr;
				if (with_adp) {
					if (sslm_gpu_adapter_map(w.ctx, w.model, &adp_view, &adp) != S::SSLM_OK) { Check(false, tag + ": adapter map"); continue; }
					w.adapters.push_back(adp);
				}
				SslmGpuSequenceHandle *a, *b;
				if (pooled) {
					a = w.Pooled(w.Pool(16), 64);
					b = w.Pooled(w.Pool(16), 64);
				} else {
					a = w.Dedicated(256);
					b = w.Dedicated(256);
				}
				const std::vector<int32_t> pre = Stream(0, 3, V, 1), rows = Stream(3, M, V, 1);
				Check(Twin(w, a, pre, Full(m), m, adp) == S::SSLM_OK && Twin(w, b, pre, Full(m), m, adp) == S::SSLM_OK, tag + ": prefix");
				CleanClose(RunChunk(w, a, rows, [&](int) { return m.dpl(); }, adp), a, M, tag);
				Check(Twin(w, b, rows, Full(m), m, adp) == S::SSLM_OK, tag + ": twin");
				BlobEq(Blob(w, a, tag), Blob(w, b, tag), tag);
			}
}

// ------------------------------------------------------------------------------------------------ 6.2
void Cell62(const mr2::Model& m) {
	g_cell = "6.2";
	const int32_t V = m.vocab();
	const int N = 512, C = 64;
	const std::vector<int32_t> stream = Stream(0, N, V, 2);
	World tw(m, Shaders());
	if (!tw.ok) { Check(false, "twin world"); return; }
	SslmGpuSequenceHandle* t = tw.Dedicated(N + C + 16);
	Check(Twin(tw, t, stream, Full(m), m) == S::SSLM_OK, "twin stream");
	const std::vector<uint8_t> twin_prompt = Blob(tw, t, "twin");
	std::vector<int32_t> twin_cont;
	int32_t next = mr2::StreamToken(N, V, 2);
	for (int i = 0; i < C; ++i) {
		int32_t o = -1;
		Check(SslmGpuSeqDecodeStepForG5Bridge(tw.ctx, t, next, Full(m), &o) == S::SSLM_OK, "twin continuation");
		twin_cont.push_back(o);
		next = o >= 0 ? o : mr2::StreamToken(i, V);
	}
	const std::vector<uint8_t> twin_final = Blob(tw, t, "twin final");
	for (int p = 0; p < g_opt.partitions; ++p) {
		std::mt19937 rng(9001u + static_cast<uint32_t>(p));
		World w(m, Shaders());
		if (!w.ok) { Check(false, F("partition %d: world", p)); continue; }
		SslmGpuSequenceHandle* a = w.Dedicated(N + C + 16);
		std::string sizes;
		bool ok = true;
		for (int at = 0; at < N && ok;) {
			const int M = std::min<int>(N - at, 1 + static_cast<int>(rng() % 8));
			sizes += std::to_string(M);
			const std::vector<int32_t> rows(stream.begin() + at, stream.begin() + at + M);
			const ChunkRun r = RunChunk(w, a, rows, [&](int) { return m.dpl() * (1 + rng() % m.layers()); });
			ok = r.embed == S::SSLM_OK && r.admitted == M && r.close == S::SSLM_OK && r.closed;
			Check(ok, F("partition %d: chunk at %d (M %d) embed %s close %s", p, at, M, mr2::Name(r.embed), mr2::Name(r.close)));
			at += M;
		}
		if (!ok) continue;
		BlobEq(Blob(w, a, "6.2"), twin_prompt, F("partition %d (%s...)", p, sizes.substr(0, 24).c_str()));
		std::vector<int32_t> cont;
		int32_t nx = mr2::StreamToken(N, V, 2);
		for (int i = 0; i < C; ++i) {
			int32_t o = -1;
			Check(SslmGpuSeqDecodeStepForG5Bridge(w.ctx, a, nx, Full(m), &o) == S::SSLM_OK, F("partition %d continuation", p));
			cont.push_back(o);
			nx = o >= 0 ? o : mr2::StreamToken(i, V);
		}
		Check(cont == twin_cont, F("partition %d: 64-token continuation differs from the twin's", p));
		BlobEq(Blob(w, a, "6.2 final"), twin_final, F("partition %d final", p));
	}
}

// ------------------------------------------------------------------------------------------------ 6.3
void Cell63(const mr2::Model& m) {
	g_cell = "6.3";
	const int32_t V = m.vocab();
	const uint32_t L = m.layers(), D = m.dpl();
	struct B { uint32_t budget; const char* name; };
	const std::vector<B> budgets = {{1 * D, "1x"}, {2 * D, "2x"}, {3 * D, "3x"}, {4 * D, "4x"}, {7 * D, "7x"}, {L * D, "Lx"}, {3 * D - 1, "3x-1"}};
	for (const B& b : budgets) {
		const std::string tag = F("%s budget %s (%u dispatches)", m.base().c_str(), b.name, b.budget);
		World w(m, Shaders());
		if (!w.ok) { Check(false, tag + ": world"); continue; }
		SslmGpuSequenceHandle* a = w.Dedicated(256);
		SslmGpuSequenceHandle* t = w.Dedicated(256);
		const std::vector<int32_t> pre = Stream(0, 3, V, 3), rows = Stream(3, 8, V, 3);
		Check(Twin(w, a, pre, Full(m), m) == S::SSLM_OK && Twin(w, t, pre, Full(m), m) == S::SSLM_OK, tag + ": prefix");
		const ChunkRun r = RunChunk(w, a, rows, [&](int) { return b.budget; });
		CleanClose(r, a, 8, tag);
		// The layer seam: each slice advanced floor(budget / per-layer) layers, capped at the chunk's end.
		const uint32_t per = b.budget / D;
		uint32_t want = 0;
		for (size_t i = 0; i + 1 < r.layers.size(); ++i) {
			want = std::min(L, want + per);
			Check(r.layers[i] == want, F("%s: layer seam after slice %zu is %u, want %u", tag.c_str(), i, r.layers[i], want));
		}
		Check(r.slices == static_cast<int>((L + per - 1) / per), F("%s: %d slices, want %u", tag.c_str(), r.slices, (L + per - 1) / per));
		Check(Twin(w, t, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
		BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag);
	}
	// Below one layer: SSLM_DISPATCH_BUDGET_TOO_SMALL with the chunk unchanged (plan §4.1).
	World w(m, Shaders());
	if (!w.ok) return;
	SslmGpuSequenceHandle* a = w.Dedicated(256);
	int32_t adm = 0;
	const std::vector<int32_t> rows = Stream(0, 4, V, 3);
	Check(EmbedRows(w, a, rows.data(), 4, &adm) == S::SSLM_OK, "embed for the small-budget check");
	SslmGpuMr2ChunkState before{}, after{};
	sslm_gpu_test_only_mr2_chunk_state(a, &before);
	Check(sslm_decode_step_gpu(w.ctx, a, nullptr, D - 1) == S::SSLM_DISPATCH_BUDGET_TOO_SMALL, "budget per-layer - 1 is not refused");
	sslm_gpu_test_only_mr2_chunk_state(a, &after);
	Check(before.layer == after.layer && before.rows == after.rows && RowsOpen(a) == 4, "the refused budget changed the chunk");
}

// ------------------------------------------------------------------------------------------------ 6.4 / 4.6b: device K/V
// A pooled sequence whose budget is first filled with a canary stream and reset (a pooled reset fills nothing), so
// the device rows past the prefix hold non-zero bytes in both worlds.
SslmGpuSequenceHandle* CanarySeq(World& w, const mr2::Model& m, SslmGpuKvPoolHandle* pool, int32_t budget) {
	SslmGpuSequenceHandle* s = w.Pooled(pool, budget);
	if (!s) return nullptr;
	if (Twin(w, s, Stream(0, budget, m.vocab(), 77), Full(m), m) != S::SSLM_OK) return nullptr;
	if (sslm_gpu_seq_reset(w.ctx, s) != S::SSLM_OK) return nullptr;
	return s;
}
std::vector<uint8_t> DeviceKv(World& w, SslmGpuKvPoolHandle* pool, SslmGpuSequenceHandle* s, const std::string& tag) {
	std::vector<uint8_t> kv;
	const S st = sslm_gpu_test_only_mr2_device_kv(w.ctx, pool, s, &kv);
	Check(st == S::SSLM_OK && !kv.empty(), F("%s: device K/V read-back %s (%zu bytes)", tag.c_str(), mr2::Name(st), kv.size()));
	return kv;
}
void Cell64(const mr2::Model& m) {
	g_cell = "6.4";
	const int32_t V = m.vocab();
	for (int M : {3, 5, 6, 7}) {
		const std::string tag = F("%s M%d", m.base().c_str(), M);
		World wa(m, Shaders()), wb(m, Shaders());
		if (!wa.ok || !wb.ok) { Check(false, tag + ": worlds"); continue; }
		SslmGpuKvPoolHandle* pa = wa.Pool(8);
		SslmGpuKvPoolHandle* pb = wb.Pool(8);
		SslmGpuSequenceHandle* a = CanarySeq(wa, m, pa, 32);
		SslmGpuSequenceHandle* b = CanarySeq(wb, m, pb, 32);
		if (!a || !b) { Check(false, tag + ": canary setup"); continue; }
		const std::vector<int32_t> pre = Stream(0, 3, V, 4), rows = Stream(3, M, V, 4);
		Check(Twin(wa, a, pre, Full(m), m) == S::SSLM_OK && Twin(wb, b, pre, Full(m), m) == S::SSLM_OK, tag + ": prefix");
		CleanClose(RunChunk(wa, a, rows, [&](int) { return m.dpl(); }), a, M, tag);
		Check(Twin(wb, b, rows, Full(m), m) == S::SSLM_OK, tag + ": one-at-a-time twin");
		BlobEq(Blob(wa, a, tag), Blob(wb, b, tag), tag);
		Check(mr2::ReadCounters(a) == mr2::ReadCounters(b), tag + ": counters differ from the real rows' (the twin's)");
		const std::vector<uint8_t> ka = DeviceKv(wa, pa, nullptr, tag), kb = DeviceKv(wb, pb, nullptr, tag);
		const long d = mr2::FirstDiff(ka, kb);
		Check(d < 0, F("%s: device pool differs from the twin world's at byte %ld (a padded row wrote K/V)", tag.c_str(), d));
	}
}
void Cell46b(const mr2::Model& m) {
	g_cell = "4.6b";
	const int32_t V = m.vocab();
	for (int edge = 0; edge <= 1; ++edge)
		for (int M : {3, 5, 6, 7}) {
			const std::string tag = F("%s M%d %s", m.base().c_str(), M, edge ? "p0+M on a page boundary" : "p0+M = budget");
			World wa(m, Shaders()), wb(m, Shaders());
			if (!wa.ok || !wb.ok) { Check(false, tag + ": worlds"); continue; }
			SslmGpuKvPoolStats st{};
			st.struct_size = sizeof st;
			auto build = [&](World& w, SslmGpuSequenceHandle** chunk_seq) -> SslmGpuKvPoolHandle* {
				SslmGpuKvPoolHandle* pool = w.Pool(24);
				if (!pool) return nullptr;
				sslm_gpu_kv_pool_stats(pool, &st);
				const int B = static_cast<int>(st.page_positions);
				// A shared prefix of 1.5 pages, a second sequence on it with its own rows, then the chunk's sequence.
				SslmGpuPrefixHandle* pre = nullptr;
				if (sslm_gpu_prefix_begin(w.ctx, w.model, pool, 2 * B, &pre) != S::SSLM_OK) return nullptr;
				const std::vector<int32_t> pt = Stream(0, B + B / 2, V, 5);
				if (!PrefixFill(w, pre, pt, Full(m)) || sslm_gpu_prefix_freeze(w.ctx, pre) != S::SSLM_OK)
					return nullptr;
				// The second sequence holds the shared prefix (adopted) plus rows of its own.
				SslmGpuSequenceHandle* other = w.Pooled(pool, 2 * B);
				if (!other || sslm_gpu_seq_adopt_prefix(w.ctx, other, pre) != S::SSLM_OK ||
				    Twin(w, other, Stream(0, 5, V, 6), Full(m), m) != S::SSLM_OK)
					return nullptr;
				sslm_gpu_prefix_release(w.ctx, pre);
				// p0 + M = budget (edge 0), or p0 + M = 3B, a page boundary, with budget 4B (edge 1).
				const int p0 = edge ? 3 * B - M : 2 * B + 3 - M;
				const int32_t budget = edge ? 4 * B : p0 + M;
				SslmGpuSequenceHandle* s = w.Pooled(pool, budget);
				if (!s || Twin(w, s, Stream(0, p0, V, 7), Full(m), m) != S::SSLM_OK) return nullptr;
				*chunk_seq = s;
				return pool;
			};
			SslmGpuSequenceHandle *a = nullptr, *b = nullptr;
			SslmGpuKvPoolHandle* pa = build(wa, &a);
			SslmGpuKvPoolHandle* pb = build(wb, &b);
			if (!pa || !pb) { Check(false, tag + ": construction"); continue; }
			const std::vector<int32_t> rows = Stream(100, M, V, 8);
			CleanClose(RunChunk(wa, a, rows, [&](int) { return Full(m); }), a, M, tag);
			Check(Twin(wb, b, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
			BlobEq(Blob(wa, a, tag), Blob(wb, b, tag), tag);
			const std::vector<uint8_t> ka = DeviceKv(wa, pa, nullptr, tag), kb = DeviceKv(wb, pb, nullptr, tag);
			const long d = mr2::FirstDiff(ka, kb);
			Check(d < 0, F("%s: device pool differs from the twin world's at byte %ld (page %ld: a padded row reached "
			               "another holder's, the prefix's or a free page)", tag.c_str(), d,
			               d < 0 || ka.empty() ? -1L : static_cast<long>(d / (ka.size() / 24))));
		}
}

// ------------------------------------------------------------------------------------------------ 4.6 (cloud leg)
void Cell46(const mr2::Model& m) {
	g_cell = "4.6";
	const int32_t V = m.vocab();
	const int cap = 40;
	for (int end : {cap, cap - 1}) {
		const std::string tag = F("%s dedicated cap %d, chunk ends at %d", m.base().c_str(), cap, end);
		World w(m, Shaders());
		if (!w.ok) { Check(false, tag); continue; }
		SslmGpuSequenceHandle* a = w.Dedicated(cap);
		SslmGpuSequenceHandle* b = w.Dedicated(cap);
		const std::vector<int32_t> pre = Stream(0, end - 8, V, 9), rows = Stream(end - 8, 8, V, 9);
		Check(Twin(w, a, pre, Full(m), m) == S::SSLM_OK && Twin(w, b, pre, Full(m), m) == S::SSLM_OK, tag + ": prefix");
		CleanClose(RunChunk(w, a, rows, [&](int) { return m.dpl(); }), a, 8, tag);
		Check(Twin(w, b, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
		BlobEq(Blob(w, a, tag), Blob(w, b, tag), tag);
	}
	for (int o = 0; o <= 7; ++o) {
		World w(m, Shaders());
		if (!w.ok) { Check(false, "world"); continue; }
		SslmGpuKvPoolHandle* pool = w.Pool(16);
		SslmGpuKvPoolStats st{};
		st.struct_size = sizeof st;
		sslm_gpu_kv_pool_stats(pool, &st);
		const int B = static_cast<int>(st.page_positions);
		const int p0 = B - o;  // the page boundary falls before row o
		const std::string tag = F("%s pooled, page boundary before row %d (B %d)", m.base().c_str(), o, B);
		SslmGpuSequenceHandle* a = w.Pooled(pool, 2 * B + 8);
		SslmGpuSequenceHandle* b = w.Pooled(w.Pool(16), 2 * B + 8);
		const std::vector<int32_t> pre = Stream(0, p0, V, 10), rows = Stream(p0, 8, V, 10);
		Check(Twin(w, a, pre, Full(m), m) == S::SSLM_OK && Twin(w, b, pre, Full(m), m) == S::SSLM_OK, tag + ": prefix");
		CleanClose(RunChunk(w, a, rows, [&](int) { return Full(m); }), a, 8, tag);
		Check(Twin(w, b, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
		BlobEq(Blob(w, a, tag), Blob(w, b, tag), tag);
	}
}

// ------------------------------------------------------------------------------------------------ refusal fixtures
// Finds, on the one-row path of `rm` (a *_refuse fixture), a stream of p0 + 8 tokens whose rows 0..p0+j-1 do not
// refuse and whose row p0+j refuses (at the last layer); with `second` > j, row p0+second is a token that refused
// in every context tried (so the chunk's row `second` refuses too: the two-refusal fixture). Deterministic.
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
RefusalStream FindRefusalStream(const mr2::Model& rm, World& w, int p0, int j, int second = -1) {
	RefusalStream r;
	const int32_t V = rm.vocab();
	int cand = 0;
	auto next_passing = [&](std::vector<int32_t>& v) {
		for (int tries = 0; tries < 64; ++tries) {
			v.push_back(mr2::StreamToken(cand++, V, 11));
			if (Probe(rm, w, v) == S::SSLM_OK) return true;
			v.pop_back();
		}
		return false;
	};
	auto next_refusing = [&](std::vector<int32_t>& v) {
		for (int tries = 0; tries < 128; ++tries) {
			v.push_back(mr2::StreamToken(cand++, V, 12));
			if (Probe(rm, w, v) == S::SSLM_SEQUENCE_REJECTED) return true;
			v.pop_back();
		}
		return false;
	};
	std::vector<int32_t> v;
	for (int i = 0; i < p0 + j; ++i)
		if (!next_passing(v)) return r;
	if (!next_refusing(v)) return r;
	for (int i = j + 1; i < 8; ++i) {
		if (i == second) {
			// A token that refuses directly after the clean part, and after the clean part plus one row.
			std::vector<int32_t> clean(v.begin(), v.begin() + p0 + j);
			bool found = false;
			for (int tries = 0; tries < 256 && !found; ++tries) {
				const int32_t t = mr2::StreamToken(cand++, V, 13);
				std::vector<int32_t> c1 = clean, c2 = clean;
				c1.push_back(t);
				c2.push_back(mr2::StreamToken(5, V, 11));
				c2.push_back(t);
				if (Probe(rm, w, c1) == S::SSLM_SEQUENCE_REJECTED && Probe(rm, w, c2) == S::SSLM_SEQUENCE_REJECTED) {
					v.push_back(t);
					found = true;
				}
			}
			if (!found) return r;
		} else {
			v.push_back(mr2::StreamToken(cand++, V, 14));
		}
	}
	r.toks = v;
	r.ok = true;
	return r;
}
bool LoadRefuse(const mr2::Model& m, mr2::Model* rm) {
	if (g_opt.fixtures.empty()) return false;
	return rm->Load(g_opt.fixtures + "/" + m.base() + "_refuse.sslm");
}

// One 6.6 run on a fresh context: chunk of 8 after p0 clean rows, against the twin fed rows 0..j with the chunk's
// slicing for row j, read after its own refusal. Whole blob, nothing masked.
void RefusalRun(const mr2::Model& rm, const std::vector<int32_t>& toks, int p0, int j, bool per_layer, bool pooled,
                const std::string& tag) {
	World w(rm, Shaders());
	if (!w.ok) { Check(false, tag + ": world"); return; }
	SslmGpuSequenceHandle *a, *b;
	if (pooled) {  // pools sized so no page is ever recycled in the run
		a = w.Pooled(w.Pool(16), 64);
		b = w.Pooled(w.Pool(16), 64);
	} else {
		a = w.Dedicated(64);
		b = w.Dedicated(64);
	}
	const std::vector<int32_t> pre(toks.begin(), toks.begin() + p0), rows(toks.begin() + p0, toks.end());
	Check(Twin(w, a, pre, Full(rm), rm) == S::SSLM_OK && Twin(w, b, pre, Full(rm), rm) == S::SSLM_OK, tag + ": prefix");
	const uint32_t budget = per_layer ? rm.dpl() : Full(rm);
	const ChunkRun r = RunChunk(w, a, rows, [&](int) { return budget; });
	Check(r.embed == S::SSLM_OK && r.admitted == 8, F("%s: embed_rows %s admitted %d", tag.c_str(), mr2::Name(r.embed), r.admitted));
	Check(r.close == S::SSLM_SEQUENCE_REJECTED && r.closed,
	      F("%s: the closing ready returned %s (closed %d), want SSLM_SEQUENCE_REJECTED", tag.c_str(), mr2::Name(r.close), r.closed));
	Check(LastCommitted(a) == j, F("%s: last_rows_committed %d, want %d", tag.c_str(), LastCommitted(a), j));
	// The twin: rows 0..j-1 to full depth, row j with the chunk's slicing, read after its refusal.
	const std::vector<int32_t> clean(rows.begin(), rows.begin() + j);
	Check(Twin(w, b, clean, Full(rm), rm) == S::SSLM_OK, tag + ": twin rows before j");
	const S tj = mr2::FeedRow(w, b, rows[j], budget, rm.layers());
	Check(tj == S::SSLM_SEQUENCE_REJECTED, F("%s: the twin's row j returned %s, not a refusal (fixture)", tag.c_str(), mr2::Name(tj)));
	BlobEq(Blob(w, a, tag), Blob(w, b, tag), tag + " (whole " + (pooled ? "SLM6" : "SLM5") + ")");
	Check(mr2::ReadCounters(a) == mr2::ReadCounters(b),
	      F("%s: counters %s, twin %s", tag.c_str(), mr2::ReadCounters(a).Str().c_str(), mr2::ReadCounters(b).Str().c_str()));
}

void Cell66(const mr2::Model& m) {
	g_cell = "6.6";
	mr2::Model rm;
	if (!LoadRefuse(m, &rm)) { Check(false, "no " + m.base() + "_refuse.sslm under --fixtures (mr2_fixtures.py)"); return; }
	World probe(rm, Shaders());
	if (!probe.ok) { Check(false, "probe world"); return; }
	const int p0 = 3;
	for (int j = 0; j <= 7; ++j) {
		const RefusalStream rs = FindRefusalStream(rm, probe, p0, j);
		if (!rs.ok) { Check(false, F("%s j%d: no refusal stream found on the one-row path", rm.base().c_str(), j)); continue; }
		for (int per_layer = 1; per_layer >= 0; --per_layer)
			for (int pooled = 0; pooled <= 1; ++pooled)
				RefusalRun(rm, rs.toks, p0, j, per_layer, pooled,
				           F("%s j%d %s %s", rm.base().c_str(), j, per_layer ? "one layer per call" : "whole depth per call",
				             pooled ? "pooled" : "dedicated"));
	}
	// Two-refusal (MX4's fixture): rows j and j + 2 both refuse; the first is the one committed at.
	for (int j : {0, 3, 5}) {
		const RefusalStream rs = FindRefusalStream(rm, probe, p0, j, j + 2);
		if (!rs.ok) { Check(false, F("%s two-refusal j%d: no stream", rm.base().c_str(), j)); continue; }
		RefusalRun(rm, rs.toks, p0, j, true, false, F("%s two-refusal j%d,k%d", rm.base().c_str(), j, j + 2));
	}
	// Post-refusal saturation (MX5's fixture) is every j < 7 above: every row of the pkv fixtures saturates
	// (6.7 checks it), so the rows after j carry deltas a sum over all rows would include.
}

// ------------------------------------------------------------------------------------------------ 6.7
void Cell67(const mr2::Model& m) {
	g_cell = "6.7";
	const int32_t V = m.vocab();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* a = w.Dedicated(64);
	SslmGpuSequenceHandle* b = w.Dedicated(64);
	const std::vector<int32_t> rows = Stream(0, 8, V, 15);
	CleanClose(RunChunk(w, a, rows, [&](int) { return m.dpl(); }), a, 8, m.base());
	for (int r = 0; r < 8; ++r) {
		const mr2::Counters before = mr2::ReadCounters(b);
		Check(mr2::FeedRow(w, b, rows[r], m.dpl(), m.layers()) == S::SSLM_OK, F("twin row %d", r));
		const mr2::Counters d = mr2::ReadCounters(b) - before;
		Check(d.v[0] > 0, F("%s: twin row %d did not saturate (%s): the cell would be vacuous", m.base().c_str(), r, d.Str().c_str()));
	}
	Check(mr2::ReadCounters(a) == mr2::ReadCounters(b),
	      F("%s: counters %s, twin %s", m.base().c_str(), mr2::ReadCounters(a).Str().c_str(), mr2::ReadCounters(b).Str().c_str()));
	BlobEq(Blob(w, a, "6.7"), Blob(w, b, "6.7"), m.base());
}

// ------------------------------------------------------------------------------------------------ 6.8 CPU parity
struct Aligned {
	void* p = nullptr;
	size_t n = 0;
	explicit Aligned(size_t size) : n(size) {
		p = ::operator new(size ? size : 1, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		std::memset(p, 0, size);
	}
	~Aligned() { ::operator delete(p, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)); }
};
std::vector<int32_t> CpuGreedy(const mr2::Model& m, const std::vector<int32_t>& prompt, int n) {
	std::vector<int32_t> out;
	Aligned bytes(m.bytes.size());
	std::memcpy(bytes.p, m.bytes.data(), m.bytes.size());
	sslm_model model = nullptr;
	if (sslm_model_map(bytes.p, bytes.n, &model) != SSLM_OK) { Check(false, "CPU model map"); return out; }
	const uint32_t pages = 16;
	Aligned mem(sslm_kv_page_size(model) * pages + sslm_kv_page_pool_overhead_size(model, pages));
	sslm_kv_pool pool = nullptr;
	sslm_seq seq = nullptr;
	if (sslm_kv_page_pool_create(model, mem.p, mem.n, pages, &pool) != SSLM_OK ||
	    sslm_seq_create_budgeted(model, &pool, 128, &seq) != SSLM_OK) {
		Check(false, "CPU pool/seq");
		if (pool) sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return out;
	}
	size_t at = 0;
	while (at < prompt.size()) {
		int32_t consumed = 0;
		const sslm_status st = sslm_prefill(model, seq, prompt.data() + at, static_cast<int32_t>(prompt.size() - at), 64,
		                                    SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st != SSLM_OK || consumed <= 0) { Check(false, F("CPU prefill %d", static_cast<int>(st))); break; }
		at += static_cast<size_t>(consumed);
	}
	for (int i = 0; i < n; ++i) {
		sslm_decode_params p{};
		p.layer_budget = 1;
		sslm_seq b1[1] = {seq};
		int32_t tok = -1;
		sslm_status st = SSLM_OK;
		for (int guard = 0; tok == -1 && guard < 4096; ++guard)
			if ((st = sslm_decode_step(model, b1, 1, &p, nullptr, &tok)) != SSLM_OK) break;
		if (st != SSLM_OK) { Check(false, F("CPU decode %d", static_cast<int>(st))); break; }
		out.push_back(tok);
	}
	sslm_seq_release(seq);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	return out;
}
void Cell68(const mr2::Model& m) {
	g_cell = "6.8";
	const int32_t V = m.vocab();
	const std::vector<int32_t> prompt = Stream(0, 24, V, 16);
	const std::vector<int32_t> cpu = CpuGreedy(m, prompt, 16);
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* a = w.Dedicated(128);
	int at = 0;
	for (int M : {8, 5, 3, 8}) {
		const std::vector<int32_t> rows(prompt.begin() + at, prompt.begin() + at + M);
		CleanClose(RunChunk(w, a, rows, [&](int i) { return m.dpl() * (1 + i % m.layers()); }), a, M, F("chunk at %d", at));
		at += M;
	}
	std::vector<int32_t> gpu;
	int32_t t = -1;
	Check(SslmGpuSeqFinishTokenForG5Bridge(w.ctx, a, &t) == S::SSLM_OK, "GPU finish");
	gpu.push_back(t);
	for (int i = 1; i < 16; ++i) {
		int32_t o = -1;
		Check(SslmGpuSeqDecodeStepForG5Bridge(w.ctx, a, t, Full(m), &o) == S::SSLM_OK, "GPU decode");
		gpu.push_back(o);
		t = o;
	}
	std::string gs, cs;
	for (int32_t x : gpu) gs += std::to_string(x) + " ";
	for (int32_t x : cpu) cs += std::to_string(x) + " ";
	Check(gpu == cpu, F("%s: GPU tokens [%s] != CPU [%s]", m.base().c_str(), gs.c_str(), cs.c_str()));
}

// ------------------------------------------------------------------------------------------------ 1.2
SslmGpuMr2Seams Seams(World& w) {
	SslmGpuMr2Seams s{};
	Check(sslm_gpu_test_only_mr2_seams(w.ctx, &s) == S::SSLM_OK, "seam read");
	return s;
}
void Cell12(const mr2::Model& m) {
	g_cell = "1.2";
	const int32_t V = m.vocab();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* a = w.Dedicated(512);
	int at = 0;
	CleanClose(RunChunk(w, a, Stream(at, 8, V, 17), [&](int) { return m.dpl(); }), a, 8, "warm-up");
	at += 8;
	const SslmGpuMr2Seams s0 = Seams(w);
	for (int M : {8, 3, 5, 2, 7, 8, 4, 6}) {
		CleanClose(RunChunk(w, a, Stream(at, M, V, 17), [&](int) { return m.dpl(); }), a, M, F("chunk M%d", M));
		at += M;
	}
	const SslmGpuMr2Seams s1 = Seams(w);
	Check(s1.allocations == s0.allocations, F("%llu recorder allocations after warm-up (want 0)",
	                                          (unsigned long long)(s1.allocations - s0.allocations)));
	// Three chunks open at once, then closed: the free list never exceeds the peak.
	SslmGpuSequenceHandle* x[3] = {w.Dedicated(64), w.Dedicated(64), w.Dedicated(64)};
	for (int i = 0; i < 3; ++i) {
		int32_t adm = 0;
		Check(EmbedRows(w, x[i], Stream(0, 4 + i, V, 18 + i).data(), 4 + i, &adm) == S::SSLM_OK, "open chunk");
	}
	for (uint32_t l = 0; l < m.layers(); ++l)
		for (auto* s : x) Check(mr2::Slice(w, s, m.dpl()) == S::SSLM_OK, "interleaved slice");
	const SslmGpuMr2Seams s2 = Seams(w);
	Check(s2.chunks_open == 0 && s2.chunks_open_peak >= 3, F("chunks open %u, peak %u", s2.chunks_open, s2.chunks_open_peak));
	Check(s2.row_buffer_sets <= s2.chunks_open_peak && s2.row_buffer_free == s2.row_buffer_sets,
	      F("row-buffer sets %u (free %u) against a peak of %u open chunks", s2.row_buffer_sets, s2.row_buffer_free, s2.chunks_open_peak));
}

// ------------------------------------------------------------------------------------------------ 1.4
void Cell14(const mr2::Model& m) {
	g_cell = "1.4";
	const int32_t V = m.vocab();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	const int Ms[3] = {8, 3, 5}, pre[3] = {2, 5, 1};
	SslmGpuSequenceHandle *a[3], *t[3];
	for (int i = 0; i < 3; ++i) {
		a[i] = w.Dedicated(64);
		t[i] = w.Dedicated(64);
		Check(Twin(w, a[i], Stream(0, pre[i], V, 20 + i), Full(m), m) == S::SSLM_OK, "prefix");
		Check(Twin(w, t[i], Stream(0, pre[i], V, 20 + i), Full(m), m) == S::SSLM_OK, "twin prefix");
	}
	SslmGpuSequenceHandle* d = w.Dedicated(64);
	SslmGpuSequenceHandle* dt = w.Dedicated(64);
	for (int i = 0; i < 3; ++i) {
		int32_t adm = 0;
		Check(EmbedRows(w, a[i], Stream(pre[i], Ms[i], V, 20 + i).data(), Ms[i], &adm) == S::SSLM_OK && adm == Ms[i], F("open chunk %d", i));
	}
	// One layer at a time across the three, with one one-row slice of the fourth sequence between each.
	int dtok = 0;
	bool d_mid = false;
	for (uint32_t l = 0; l < m.layers(); ++l)
		for (int i = 0; i < 3; ++i) {
			Check(mr2::Slice(w, a[i], m.dpl()) == S::SSLM_OK, F("seq %d layer %u", i, l));
			if (!d_mid) {
				Check(sslm_gpu_seq_embed_token(w.ctx, d, mr2::StreamToken(dtok, V, 23)) == S::SSLM_OK, "fourth embed");
				d_mid = true;
			}
			Check(mr2::Slice(w, d, m.dpl()) == S::SSLM_OK, "fourth slice");
			if (*SslmGpuSeqHandleLayerIndexForBench(d) >= m.layers()) { d_mid = false; ++dtok; }
		}
	for (int i = 0; i < 3; ++i) {
		Check(RowsOpen(a[i]) == 0 && LastCommitted(a[i]) == Ms[i], F("seq %d not closed with %d rows", i, Ms[i]));
		Check(Twin(w, t[i], Stream(pre[i], Ms[i], V, 20 + i), Full(m), m) == S::SSLM_OK, "twin");
		BlobEq(Blob(w, a[i], "1.4"), Blob(w, t[i], "1.4"), F("seq %d (M %d)", i, Ms[i]));
	}
	// The fourth: its twin fed the same tokens one layer at a time without any chunk around.
	for (int k = 0; k < dtok; ++k) Check(mr2::FeedRow(w, dt, mr2::StreamToken(k, V, 23), m.dpl(), m.layers()) == S::SSLM_OK, "fourth twin");
	if (d_mid) {
		Check(sslm_gpu_seq_embed_token(w.ctx, dt, mr2::StreamToken(dtok, V, 23)) == S::SSLM_OK, "fourth twin embed");
		while (*SslmGpuSeqHandleLayerIndexForBench(dt) < *SslmGpuSeqHandleLayerIndexForBench(d))
			Check(mr2::Slice(w, dt, m.dpl()) == S::SSLM_OK, "fourth twin slice");
	}
	BlobEq(Blob(w, d, "1.4"), Blob(w, dt, "1.4"), "the fourth (one-row) sequence");
}

// ------------------------------------------------------------------------------------------------ 1.5
void Cell15(const mr2::Model& m) {
	g_cell = "1.5";
	mr2::Model rm;
	if (!LoadRefuse(m, &rm)) { Check(false, "no " + m.base() + "_refuse.sslm under --fixtures"); return; }
	World probe(rm, Shaders());
	const RefusalStream rs = FindRefusalStream(rm, probe, 3, 2);
	if (!rs.ok) { Check(false, "no refusal stream (j = 2)"); return; }
	World w(rm, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* a = w.Dedicated(64);
	SslmGpuSequenceHandle* b = w.Dedicated(64);
	const std::vector<int32_t> pre(rs.toks.begin(), rs.toks.begin() + 3), rows(rs.toks.begin() + 3, rs.toks.end());
	Check(Twin(w, a, pre, Full(rm), rm) == S::SSLM_OK && Twin(w, b, pre, Full(rm), rm) == S::SSLM_OK, "prefix");
	const ChunkRun r = RunChunk(w, a, rows, [&](int) { return Full(rm); });
	Check(r.close == S::SSLM_SEQUENCE_REJECTED && LastCommitted(a) == 2, F("refused chunk: %s, committed %d", mr2::Name(r.close), LastCommitted(a)));
	Check(Twin(w, b, std::vector<int32_t>(rows.begin(), rows.begin() + 2), Full(rm), rm) == S::SSLM_OK, "twin rows 0..1");
	Check(mr2::FeedRow(w, b, rows[2], Full(rm), rm.layers()) == S::SSLM_SEQUENCE_REJECTED, "twin row 2 refuses");
	BlobEq(Blob(w, a, "1.5"), Blob(w, b, "1.5"), "after the refusal");
	Check(sslm_gpu_seq_reset(w.ctx, a) == S::SSLM_OK && sslm_gpu_seq_reset(w.ctx, b) == S::SSLM_OK, "reset");
	int at = 0;
	for (int M : {8, 3}) {
		// Rows that pass on this fixture: probe them on the one-row path first.
		std::vector<int32_t> ok_rows;
		std::vector<int32_t> hist;
		for (int c = 0; static_cast<int>(ok_rows.size()) < at + M && c < 400; ++c) {
			std::vector<int32_t> v = ok_rows;
			v.push_back(mr2::StreamToken(c, rm.vocab(), 24));
			if (Probe(rm, probe, v) == S::SSLM_OK) ok_rows = v;
		}
		if (static_cast<int>(ok_rows.size()) < at + M) { Check(false, "no passing rows"); return; }
		const std::vector<int32_t> chunk(ok_rows.begin() + at, ok_rows.begin() + at + M);
		CleanClose(RunChunk(w, a, chunk, [&](int) { return rm.dpl(); }), a, M, F("M%d after reset", M));
		Check(Twin(w, b, chunk, Full(rm), rm) == S::SSLM_OK, "twin");
		BlobEq(Blob(w, a, "1.5"), Blob(w, b, "1.5"), F("M%d after reset", M));
		at += M;
	}
}

// ------------------------------------------------------------------------------------------------ 4.2
void Cell42(const mr2::Model& m) {
	g_cell = "4.2";
	const int32_t V = m.vocab();
	const int chunks = 200;
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuSequenceHandle* a = w.Dedicated(chunks * 8 + 16);
	const uint32_t full = m.layers() * m.dpl();
	for (int c = 0; c < chunks; ++c) {
		const SslmGpuMr2Seams s0 = Seams(w);
		const ChunkRun r = RunChunk(w, a, Stream(c * 8, 8, V, 25), [&](int) { return full; });
		const SslmGpuMr2Seams s1 = Seams(w);
		const bool ok = r.close == S::SSLM_OK && r.closed && r.slices == 1;
		Check(ok, F("chunk %d: %s after %d slices", c, mr2::Name(r.close), r.slices));
		Check(s1.widened_lists - s0.widened_lists == 1 && s1.last_widened_list_dispatches == full,
		      F("chunk %d: %llu widened lists, %u dispatches in the last (want one list of %u)", c,
		        (unsigned long long)(s1.widened_lists - s0.widened_lists), s1.last_widened_list_dispatches, full));
		if (!ok) return;
	}
	World tw(m, Shaders());
	SslmGpuSequenceHandle* t = tw.Dedicated(chunks * 8 + 16);
	for (int c = 0; c < chunks; ++c) Twin(tw, t, Stream(c * 8, 8, V, 25), full, m);
	BlobEq(Blob(w, a, "4.2"), Blob(tw, t, "4.2"), F("after %d full-depth chunks", chunks));
}

// ------------------------------------------------------------------------------------------------ 5.3 and 1.1r
void Cell53(const mr2::Model& m) {
	g_cell = "5.3";
	const int32_t V = m.vocab();
	for (uint32_t k = 0; k < m.layers(); ++k)
		for (int pooled = 0; pooled <= 1; ++pooled) {
			const std::string tag = F("%s slice %u %s", m.base().c_str(), k, pooled ? "pooled" : "dedicated");
			World w(m, Shaders());
			if (!w.ok) { Check(false, tag); continue; }
			SslmGpuKvPoolHandle* pool = pooled ? w.Pool(16) : nullptr;
			SslmGpuSequenceHandle* a = pooled ? w.Pooled(pool, 64) : w.Dedicated(64);
			// The twin has a's KV layout: a pooled sequence saves an 'SLM6' budget blob and a dedicated one does not,
			// so a pooled leg's twin is pooled too, from its own pool so a's pool page accounting stays a's alone.
			SslmGpuSequenceHandle* t = pooled ? w.Pooled(w.Pool(16), 64) : w.Dedicated(64);
			Check(Twin(w, a, Stream(0, 3, V, 26), Full(m), m) == S::SSLM_OK, tag + ": prefix");
			const std::vector<uint8_t> entry = Blob(w, a, tag);
			SslmGpuKvPoolStats ps0{}, ps1{};
			ps0.struct_size = ps1.struct_size = sizeof ps0;
			if (pool) sslm_gpu_kv_pool_stats(pool, &ps0);
			Check(sslm_gpu_test_only_mr2_arm_slice_alloc_failure(w.ctx, static_cast<int32_t>(k)) == S::SSLM_OK, tag + ": arm");
			const std::vector<int32_t> rows = Stream(3, 6, V, 26);
			const ChunkRun r = RunChunk(w, a, rows, [&](int) { return m.dpl(); });
			Check(r.close == S::SSLM_GPU_ALLOCATION_FAILED && r.slices == static_cast<int>(k) + 1,
			      F("%s: %s after %d slices, want SSLM_GPU_ALLOCATION_FAILED at slice %u", tag.c_str(), mr2::Name(r.close), r.slices, k));
			Check(RowsOpen(a) == 0 && LastCommitted(a) == 0, F("%s: rows_open %d, last_rows_committed %d after rollback", tag.c_str(), RowsOpen(a), LastCommitted(a)));
			BlobEq(Blob(w, a, tag), entry, tag + ": rolled back to entry");
			if (pool) {
				sslm_gpu_kv_pool_stats(pool, &ps1);
				Check(ps1.free_pages == ps0.free_pages, F("%s: pool free pages %u, entry %u", tag.c_str(), ps1.free_pages, ps0.free_pages));
			}
			sslm_gpu_test_only_mr2_arm_slice_alloc_failure(w.ctx, -1);
			CleanClose(RunChunk(w, a, rows, [&](int) { return m.dpl(); }), a, 6, tag + " retry");
			Check(Twin(w, t, Stream(0, 3, V, 26), Full(m), m) == S::SSLM_OK && Twin(w, t, rows, Full(m), m) == S::SSLM_OK, tag + ": twin");
			BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag + ": retry");
		}
}
void Cell11r(const mr2::Model& m) {
	g_cell = "1.1r";
	const int32_t V = m.vocab();
	World w(m, Shaders());
	if (!w.ok) { Check(false, "world"); return; }
	SslmGpuKvPoolHandle* pool = w.Pool(24);
	SslmGpuKvPoolStats st{};
	st.struct_size = sizeof st;
	sslm_gpu_kv_pool_stats(pool, &st);
	const int B = static_cast<int>(st.page_positions);
	SslmGpuPrefixHandle* pre = nullptr;
	// Two whole pages, so the entry sits at 2B on a page boundary and every chunk's rows map a fresh page that a
	// rollback must unmap (at B + 3 the rows stayed inside the page the entry already held: MX8 was invisible).
	const std::vector<int32_t> pt = Stream(0, 2 * B, V, 27);
	Check(sslm_gpu_prefix_begin(w.ctx, w.model, pool, 2 * B, &pre) == S::SSLM_OK && PrefixFill(w, pre, pt, Full(m)) &&
	      sslm_gpu_prefix_freeze(w.ctx, pre) == S::SSLM_OK, "prefix");
	SslmGpuSequenceHandle* a = w.Pooled(pool, 4 * B);
	Check(a && sslm_gpu_seq_adopt_prefix(w.ctx, a, pre) == S::SSLM_OK, "adopt");
	const std::vector<uint8_t> entry = Blob(w, a, "1.1r");
	SslmGpuKvPoolStats e{};
	e.struct_size = sizeof e;
	sslm_gpu_kv_pool_stats(pool, &e);
	uint32_t mapped_entry = 0, grew = 0;
	Check(sslm_gpu_test_only_mr2_holder_mapped(a, &mapped_entry) == S::SSLM_OK && mapped_entry == 2,
	      F("holder maps %u pages at entry (want the prefix's 2)", mapped_entry));
	std::mt19937 rng(77);
	for (int cyc = 0; cyc < 40; ++cyc) {
		const int M = 1 + static_cast<int>(rng() % 8);
		// Rows in the page past the entry's last, so a rollback must unmap it.
		const std::vector<int32_t> rows = Stream(cyc * 8, M, V, 28);
		int32_t adm = 0;
		const bool fail_path = cyc % 2 == 1;
		if (fail_path) sslm_gpu_test_only_mr2_arm_slice_alloc_failure(w.ctx, static_cast<int32_t>(rng() % m.layers()));
		Check(EmbedRows(w, a, rows.data(), M, &adm) == S::SSLM_OK, F("cycle %d embed", cyc));
		const uint32_t stop = rng() % m.layers();
		S last = S::SSLM_OK;
		for (uint32_t l = 0; l < stop && last == S::SSLM_OK && RowsOpen(a) > 0; ++l) last = mr2::Slice(w, a, m.dpl());
		uint32_t mapped_open = 0;
		sslm_gpu_test_only_mr2_holder_mapped(a, &mapped_open);
		if (mapped_open > mapped_entry) ++grew;
		if (RowsOpen(a) > 0) Check(sslm_gpu_seq_rows_abandon(w.ctx, a) == S::SSLM_OK, F("cycle %d abandon", cyc));
		sslm_gpu_test_only_mr2_arm_slice_alloc_failure(w.ctx, -1);
		SslmGpuKvPoolStats now{};
		now.struct_size = sizeof now;
		sslm_gpu_kv_pool_stats(pool, &now);
		Check(now.free_pages == e.free_pages && now.shared_pages == e.shared_pages,
		      F("cycle %d (%s): free %u shared %u, entry %u/%u", cyc, fail_path ? "failure" : "abandon", now.free_pages,
		        now.shared_pages, e.free_pages, e.shared_pages));
		BlobEq(Blob(w, a, "1.1r"), entry, F("cycle %d back at entry", cyc));
		// MX8: the rollback unmapped the page the chunk mapped (pool counts cannot show it; see mr2_entry.h).
		uint32_t mapped = 0;
		Check(sslm_gpu_test_only_mr2_holder_mapped(a, &mapped) == S::SSLM_OK && mapped == mapped_entry,
		      F("cycle %d (%s): holder maps %u pages after the rollback, entry %u", cyc, fail_path ? "failure" : "abandon",
		        mapped, mapped_entry));
	}
	// Not vacuous: some cycles mapped a page before rolling back.
	Check(grew > 0, "no cycle mapped a page past the entry, so no rollback had one to unmap");
	sslm_gpu_prefix_release(w.ctx, pre);
}

// ------------------------------------------------------------------------------------------------ instruments
// --instrument: the references the cells grade against, checked with no recorder involved (runs under the stub):
//   I1  the one-row twin is deterministic: two handles fed the same 6.1 stream give the same blob
//   I2  the refusal fixtures work: for j 0..7 a refusal stream is found, two one-row twins of it refuse at row j and
//       leave byte-equal blobs (SLM5 and SLM6), so 6.6 cannot fail on a correct build
//   I3  6.8's anchor: the GPU one-row path and the CPU engine give the same greedy tokens on the 6.8 stream
void Instruments(const mr2::Model& m) {
	g_cell = "I1";
	const int32_t V = m.vocab();
	{
		World w(m, Shaders());
		SslmGpuSequenceHandle* a = w.Dedicated(256);
		SslmGpuSequenceHandle* b = w.Dedicated(256);
		Check(Twin(w, a, Stream(0, 15, V), Full(m), m) == S::SSLM_OK && Twin(w, b, Stream(0, 15, V), m.dpl(), m) == S::SSLM_OK, "twins");
		BlobEq(Blob(w, a, "I1"), Blob(w, b, "I1"), "full-depth twin vs one-layer-per-call twin");
	}
	g_cell = "I2";
	mr2::Model rm;
	if (!LoadRefuse(m, &rm)) Check(false, "no refusal fixture");
	else {
		World probe(rm, Shaders());
		for (int j = 0; j <= 7; ++j) {
			const RefusalStream rs = FindRefusalStream(rm, probe, 3, j, j <= 5 ? j + 2 : -1);
			if (!rs.ok) { Check(false, F("j%d: no refusal stream", j)); continue; }
			for (int pooled = 0; pooled <= 1; ++pooled) {
				World w(rm, Shaders());
				SslmGpuSequenceHandle* a = pooled ? w.Pooled(w.Pool(16), 64) : w.Dedicated(64);
				SslmGpuSequenceHandle* b = pooled ? w.Pooled(w.Pool(16), 64) : w.Dedicated(64);
				const std::vector<int32_t> clean(rs.toks.begin(), rs.toks.begin() + 3 + j);
				Check(Twin(w, a, clean, Full(rm), rm) == S::SSLM_OK && Twin(w, b, clean, Full(rm), rm) == S::SSLM_OK, F("j%d clean rows", j));
				Check(mr2::FeedRow(w, a, rs.toks[3 + j], rm.dpl(), rm.layers()) == S::SSLM_SEQUENCE_REJECTED &&
				      mr2::FeedRow(w, b, rs.toks[3 + j], rm.dpl(), rm.layers()) == S::SSLM_SEQUENCE_REJECTED, F("j%d: row j does not refuse", j));
				BlobEq(Blob(w, a, "I2"), Blob(w, b, "I2"), F("j%d %s: two twins after the refusal", j, pooled ? "SLM6" : "SLM5"));
			}
		}
	}
	g_cell = "I3";
	const std::vector<int32_t> prompt = Stream(0, 24, V, 16);
	const std::vector<int32_t> cpu = CpuGreedy(m, prompt, 16);
	World w(m, Shaders());
	SslmGpuSequenceHandle* a = w.Dedicated(128);
	Check(Twin(w, a, prompt, Full(m), m) == S::SSLM_OK, "GPU one-row prompt");
	std::vector<int32_t> gpu;
	int32_t t = -1;
	Check(SslmGpuSeqFinishTokenForG5Bridge(w.ctx, a, &t) == S::SSLM_OK, "GPU finish");
	gpu.push_back(t);
	for (int i = 1; i < 16; ++i) {
		int32_t o = -1;
		Check(SslmGpuSeqDecodeStepForG5Bridge(w.ctx, a, t, Full(m), &o) == S::SSLM_OK, "GPU decode");
		gpu.push_back(o);
		t = o;
	}
	Check(gpu == cpu, F("%s: GPU one-row tokens differ from the CPU engine's", m.base().c_str()));
}

}  // namespace

int main(int argc, char** argv) {
	bool instrument = false;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		auto val = [&](const char* k) { return s.rfind(k, 0) == 0 ? s.substr(std::strlen(k)) : std::string(); };
		if (s.rfind("--model=", 0) == 0) g_opt.model = val("--model=");
		else if (s.rfind("--fixtures=", 0) == 0) g_opt.fixtures = val("--fixtures=");
		else if (s.rfind("--adapter=", 0) == 0) g_opt.adapter = val("--adapter=");
		else if (s.rfind("--shaders=", 0) == 0) g_opt.shaders = val("--shaders=");
		else if (s == "--instrument") instrument = true;
		else if (s.rfind("--partitions=", 0) == 0) g_opt.partitions = std::atoi(val("--partitions=").c_str());
		else if (s.rfind("--cells=", 0) == 0) {
			const std::string v = val("--cells=");
			for (size_t p = 0; p < v.size();) {
				const size_t q = v.find(',', p);
				g_opt.cells.insert(v.substr(p, q - p));
				p = q == std::string::npos ? v.size() : q + 1;
			}
		} else {
			std::printf("usage: mr2_cells --model=PATH [--fixtures=DIR] [--adapter=PATH] [--shaders=DIR] [--cells=..] [--partitions=N]\n");
			return 2;
		}
	}
	const std::vector<std::pair<const char*, void (*)(const mr2::Model&)>> cells = {
	    {"6.1", Cell61}, {"6.3", Cell63}, {"6.4", Cell64}, {"6.7", Cell67}, {"4.1", Cell41}, {"4.6", Cell46},
	    {"4.6b", Cell46b}, {"6.6", Cell66}, {"1.5", Cell15}, {"1.2", Cell12}, {"1.4", Cell14}, {"5.3", Cell53},
	    {"1.1r", Cell11r}, {"6.8", Cell68}, {"4.2", Cell42}, {"6.2", Cell62}};
	if (instrument) {
		mr2::Model im;
		if (!im.Load(g_opt.model)) return 1;
		Instruments(im);
		bool ok = true;
		for (const char* c : {"I1", "I2", "I3"}) {
			std::printf("%s instrument %s (%d checks, %d failures)\n", g_fail[c] ? "RED" : "PASS", c, g_checks[c], g_fail[c]);
			ok = ok && !g_fail[c];
		}
		return ok ? 0 : 1;
	}
#ifdef MR2_STUB
	for (const auto& c : cells)
		if (g_opt.want(c.first)) std::printf("RED cell %s: red-unimplemented (mr2_stub.cpp linked: src/ defines no recorder entry)\n", c.first);
	return 1;
#else
	mr2::Model m;
	if (!m.Load(g_opt.model)) return 1;
	{
		const auto t0 = std::chrono::steady_clock::now();
		SslmGpuContext* ctx = nullptr;
		GpuContextConfig cfg{};
		cfg.shader_dir = Shaders();
		const S st = sslm_gpu_context_create(cfg, &ctx);
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		std::printf("cell 4.13 [box]: cloud context create %.1f ms (%s; diagnostic, filed on the box)\n", ms, mr2::Name(st));
		if (ctx) sslm_gpu_context_destroy(ctx);
	}
	bool all = true;
	for (const auto& c : cells) {
		if (!g_opt.want(c.first)) continue;
		g_cell = c.first;
		const auto t0 = std::chrono::steady_clock::now();
		c.second(m);
		const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		const int f = g_fail[c.first];
		if (f) std::printf("RED cell %s: %d failures of %d checks (%.0f s)\n", c.first, f, g_checks[c.first], s);
		else std::printf("cell %s: PASS (%d checks, %.0f s)\n", c.first, g_checks[c.first], s);
		all = all && f == 0;
	}
	std::printf("cells: %s\n", all ? "PASS" : "RED");
	return all ? 0 : 1;
#endif
}
