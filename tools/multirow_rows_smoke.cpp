// Multi-row GPU plan (rev 1.3.2), slices MR2 and MR3: a builder's smoke of the recorder through the published verbs
// (include/superslm/gpu_1p0.h: sslm_gpu_seq_embed_rows and the decode step and ready that advance and close a
// chunk). A development aid, not the grade: the test author's suites (tests/multirow-gpu) grade the cells. It
// reads plan §8's seams, so the engine is built with -DSUPERSLM_MULTIROW_WIDE.
//
// Every case compares a chunk handle with its one-row twin, byte for byte, through whole save blobs (SLM5 for a
// dedicated sequence, SLM6 for a pooled one), plus the close's status and last_rows_committed:
//   twin   the same p0 prefix, then rows 0..M-1 each through embed_token and decode steps of the case's budget;
//   chunk  the same prefix, then embed_rows(M) and decode steps of the same budget until the close.
// Then one more one-row token on both, compared again (the continuation). Cases:
//   clean    M in 1..8 x p0 in {0, 3, 13} x {dedicated, pooled} x {one layer per call, the whole depth per call}
//            (--quick: p0 = 3, M in {1, 2, 3, 5, 8});
//   refusal  the refusal at row j, the twin fed rows 0..j and read after it refuses row j, at one layer per call
//            and at the whole depth per call: SSLM_SEQUENCE_REJECTED, last_rows_committed = j and equal blobs. With
//            --rope-positions (below): j in {0, 1, 3, 4} for M = 5, j = 0 for M = 1, j = 1 for M = 8. Otherwise, when
//            a refusing row token exists among the first --scan vocabulary tokens at p0 = 3: j in {0, 1, 4}, M = 5;
//   abandon  M = 5 abandoned after one slice: the blob equals the one saved before embed_rows, then a fresh chunk
//            closes equal to its twin;
//   fault    cell 5.3's seam at slice 1 of an M = 4 chunk: SSLM_GPU_ALLOCATION_FAILED, the blob at entry, and a
//            retry equal to its twin;
//   seams    after warm-up no slice allocates, and the free list never holds more sets than chunks were open.
// MR3 (the published verbs, plan §4.1 to §4.6):
//   batch    sslm_decode_step_batch_gpu over a plain member, an M = 5 chunk and an M = 1 chunk. The budget is
//            batch-wide and taken in array order, each member as many whole layers as remain (as the base batch
//            plans a plain member): with the whole depth for two members only, the third reads
//            SSLM_BATCH_BUDGET_EXHAUSTED and its chunk is unchanged; the next call carries it to full depth, and
//            every member equals its twin (fed at the whole depth per call);
//   statuses save with a chunk open (SSLM_GPU_ROWS_CHUNK_OPEN, the buffer and size untouched), admission capped by
//            the context's headroom, and SSLM_GPU_CONTEXT_CAP_EXCEEDED at the cap with nothing written.
//
// MRX (plan erratum E-11, rulings 21, 21b, 22): MR4's public schema-forced kind, its flag and Canonical are removed.
// With --forced-schema=NAME on a model with a forced-next table (tools/gen_schema_fast_decode_fixture.py's fsfd.sslm,
// schema potion_shop_order):
//   rows       kind 1 and flag 0x1, on the very chain the schema forces, are bad specs that write nothing; PROMPT on
//              the same chain is admitted and closes;
//   jump       NULL options, mode 0 (SSLM_JUMP_DEFAULT) and SSLM_JUMP_EXACT, with or without a draft, are refused
//              SSLM_SEQUENCE_REJECTED until MR5, with nothing changed; mode 2, or a draft with mode 0, is malformed.
// Without --forced-schema, kind 1 is a bad spec on a model with no table too.
//
// --refusal-only runs just the refusal cases (with --rope-positions the clean grid's later rows would refuse too).
//
// --rope-positions=N truncates the model's RoPE tables to N positions in this process before it is mapped, so
// every row at position N or later refuses at layer 0's RoPE guard (SSLM_SEQUENCE_REJECTED, the table-extent
// tag): the refusal cases then run with p0 = N - j, whatever the tokens.
//
// Usage: multirow_rows_smoke.exe --model=PATH [--adapter=PATH] [--shaders=DIR] [--quick] [--scan=N] [--refusal-only]
//                                [--rope-positions=N] [--forced-schema=NAME]
// Exit 0 when every case passes.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "multirow_wide.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

#if !defined(SUPERSLM_MULTIROW_WIDE)
#error "multirow_rows_smoke needs the engine built with -DSUPERSLM_MULTIROW_WIDE"
#endif

using enum SslmGpuStatus;

namespace {

bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t n = sz > 0 ? std::fread(out->data(), 1, out->size(), f) : 0;
	std::fclose(f);
	return n == out->size();
}

int32_t TokenAt(int i, int32_t vocab) {
	return static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u) % static_cast<uint32_t>(vocab));
}

struct Env {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuAdapterHandle* adapter = nullptr;
	uint32_t L = 0, dpl = 0;
	int32_t vocab = 0;
};

SslmGpuStatus Drain(const Env& e, SslmGpuSequenceHandle* s) {
	int32_t ready = 0;
	SslmGpuStatus out = SSLM_OK;
	const SslmGpuStatus st = sslm_gpu_ready(e.ctx, s, 1, &ready, &out);
	if (st != SSLM_OK) return st;
	return ready ? out : SSLM_DEVICE_LOST;
}

SslmGpuStatus Step(const Env& e, SslmGpuSequenceHandle* s, uint32_t budget) {
	const SslmGpuStatus st = sslm_decode_step_gpu(e.ctx, s, e.adapter, budget);
	return st != SSLM_OK ? st : Drain(e, s);
}

// One one-row token in slices of `budget` dispatches: the first non-OK status, or OK at full depth.
SslmGpuStatus OneRow(const Env& e, SslmGpuSequenceHandle* s, int32_t tok, uint32_t budget) {
	SslmGpuStatus st = sslm_gpu_seq_embed_token(e.ctx, s, tok);
	if (st != SSLM_OK) return st;
	for (uint32_t g = 0; g <= e.L; ++g) {
		if ((st = Step(e, s, budget)) != SSLM_OK) return st;
		if (*SslmGpuSeqHandleLayerIndexForBench(s) >= e.L) return SSLM_OK;
	}
	return SSLM_DEVICE_LOST;
}

// A chunk of `rows` tokens in slices of `budget` dispatches, through its close: the close's status.
SslmGpuStatus Chunk(const Env& e, SslmGpuSequenceHandle* s, const std::vector<int32_t>& rows, uint32_t budget,
                    int32_t* admitted) {
	SslmGpuRowsSpec spec{};
	spec.struct_size = sizeof spec;
	spec.kind = SSLM_GPU_ROWS_PROMPT;
	spec.tokens = rows.data();
	spec.count = static_cast<int32_t>(rows.size());
	SslmGpuStatus st = sslm_gpu_seq_embed_rows(e.ctx, s, &spec, admitted);
	if (st != SSLM_OK) return st;
	for (uint32_t g = 0; g <= e.L; ++g) {
		st = Step(e, s, budget);
		int32_t open = 0;
		sslm_gpu_seq_rows_open(s, &open);
		if (st != SSLM_OK || open == 0) return st;
	}
	return SSLM_DEVICE_LOST;
}

bool Save(const Env& e, SslmGpuSequenceHandle* s, std::vector<uint8_t>* b) {
	size_t need = 0;
	sslm_gpu_seq_save(e.ctx, s, nullptr, &need);
	b->assign(need ? need : 1, 0);
	size_t sz = b->size();
	if (sslm_gpu_seq_save(e.ctx, s, b->data(), &sz) != SSLM_OK) return false;
	b->resize(sz);
	return true;
}

long FirstDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
	const size_t n = a.size() < b.size() ? a.size() : b.size();
	for (size_t i = 0; i < n; ++i)
		if (a[i] != b[i]) return static_cast<long>(i);
	return a.size() == b.size() ? -1 : static_cast<long>(n);
}

// A pair of handles (twin, chunk) on fresh storage: two dedicated sequences, or two sequences of one fresh pool
// whose pages are never recycled within the case.
struct Pair {
	const Env& e;
	SslmGpuKvPoolHandle* pool = nullptr;
	SslmGpuSequenceHandle* twin = nullptr;
	SslmGpuSequenceHandle* chunk = nullptr;
	bool ok = true;
	Pair(const Env& env, bool pooled) : e(env) {
		if (pooled && sslm_gpu_kv_pool_create(e.ctx, e.model, 16, &pool) != SSLM_OK) ok = false;
		for (SslmGpuSequenceHandle** s : {&twin, &chunk}) {
			const SslmGpuStatus st = pooled ? sslm_gpu_seq_create_budgeted(e.ctx, e.model, pool, 32, s)
			                                : sslm_gpu_seq_create(e.ctx, e.model, 32, s);
			if (!ok || st != SSLM_OK) ok = false;
			else if (e.adapter && sslm_gpu_seq_bind_adapter(e.ctx, *s, e.adapter) != SSLM_OK) ok = false;
		}
	}
	~Pair() {
		if (twin) sslm_gpu_seq_release(e.ctx, twin);
		if (chunk) sslm_gpu_seq_release(e.ctx, chunk);
		if (pool) sslm_gpu_kv_pool_destroy(e.ctx, pool);
	}
	bool Prefix(int p0) {
		for (int i = 0; ok && i < p0; ++i) {
			for (SslmGpuSequenceHandle* s : {twin, chunk}) {
				if (OneRow(e, s, TokenAt(i, e.vocab), e.dpl * e.L) != SSLM_OK) ok = false;
			}
		}
		return ok;
	}
	bool Equal(const char* what, char* why, size_t why_n) {
		std::vector<uint8_t> a, b;
		if (!Save(e, twin, &a) || !Save(e, chunk, &b)) {
			std::snprintf(why, why_n, "%s: save failed", what);
			return false;
		}
		const long d = FirstDiff(a, b);
		if (d >= 0) {
			std::snprintf(why, why_n, "%s: blobs differ at byte %ld of %zu/%zu", what, d, a.size(), b.size());
			return false;
		}
		return true;
	}
};

int g_failed = 0, g_cases = 0;
void Report(bool pass, const std::string& name, const char* why) {
	++g_cases;
	if (!pass) ++g_failed;
	std::printf("%s %s%s%s\n", pass ? "PASS" : "FAIL", name.c_str(), why[0] ? ": " : "", why);
}

// Row tokens that do not refuse on their own (the refusal scan's), when it ran.
std::vector<int32_t> g_clean_tokens;

std::vector<int32_t> RowTokens(int p0, uint32_t m, int32_t vocab, int32_t refuse_tok = -1, int j = -1) {
	std::vector<int32_t> rows(m);
	for (uint32_t r = 0; r < m; ++r) {
		rows[r] = g_clean_tokens.empty() ? TokenAt(p0 + 100 + static_cast<int>(r), vocab)
		                                 : g_clean_tokens[(static_cast<size_t>(p0) + r) % g_clean_tokens.size()];
	}
	if (j >= 0) rows[static_cast<size_t>(j)] = refuse_tok;
	return rows;
}

void CleanCase(const Env& e, uint32_t m, int p0, bool pooled, uint32_t layers_per_call) {
	char why[256] = {};
	const std::string name = "clean M=" + std::to_string(m) + " p0=" + std::to_string(p0) +
	                         (pooled ? " pooled" : " dedicated") + " layers/call=" + std::to_string(layers_per_call) +
	                         (e.adapter ? " +adapter" : "");
	Pair p(e, pooled);
	if (!p.ok || !p.Prefix(p0)) return Report(false, name, "setup");
	const uint32_t budget = e.dpl * layers_per_call;
	const std::vector<int32_t> rows = RowTokens(p0, m, e.vocab);
	for (int32_t t : rows) {
		const SslmGpuStatus st = OneRow(e, p.twin, t, budget);
		if (st != SSLM_OK) {
			std::snprintf(why, sizeof why, "twin row -> %d", static_cast<int>(st));
			return Report(false, name, why);
		}
	}
	int32_t admitted = 0;
	const SslmGpuStatus st = Chunk(e, p.chunk, rows, budget, &admitted);
	int32_t committed = -1, open = -1;
	sslm_gpu_seq_last_rows_committed(p.chunk, &committed);
	sslm_gpu_seq_rows_open(p.chunk, &open);
	if (st != SSLM_OK || admitted != static_cast<int32_t>(m) || committed != static_cast<int32_t>(m) || open != 0) {
		std::snprintf(why, sizeof why, "close -> %d, admitted %d, committed %d, open %d", static_cast<int>(st), admitted,
		              committed, open);
		return Report(false, name, why);
	}
	if (!p.Equal("after the close", why, sizeof why)) return Report(false, name, why);
	const int32_t next = TokenAt(p0 + 99, e.vocab);
	if (OneRow(e, p.twin, next, e.dpl * e.L) != SSLM_OK || OneRow(e, p.chunk, next, e.dpl * e.L) != SSLM_OK) {
		return Report(false, name, "continuation token failed");
	}
	if (!p.Equal("after the continuation", why, sizeof why)) return Report(false, name, why);
	Report(true, name, why);
}

// The tokens among the first `scan` that the one-row path refuses at p0 = 3, one layer per call: the first one
// refusing at the deepest layer seen is returned (-1 if none), and the ones that do not refuse are kept for the
// other rows.
int32_t FindRefusingToken(const Env& e, int scan, int* refused_layer) {
	int32_t found = -1;
	for (int32_t t = 0; t < scan && t < e.vocab; ++t) {
		Pair p(e, false);
		if (!p.ok || !p.Prefix(3)) return -1;
		if (sslm_gpu_seq_embed_token(e.ctx, p.twin, t) != SSLM_OK) return -1;
		bool refused = false;
		for (uint32_t l = 0; l < e.L && !refused; ++l) {
			const SslmGpuStatus st = Step(e, p.twin, e.dpl);
			if (st == SSLM_SEQUENCE_REJECTED) {
				refused = true;
				if (found < 0 || static_cast<int>(l) > *refused_layer) {
					found = t;
					*refused_layer = static_cast<int>(l);
				}
			} else if (st != SSLM_OK) {
				return -1;
			}
		}
		if (!refused) g_clean_tokens.push_back(t);
	}
	return found;
}

void RefusalCase(const Env& e, int32_t refuse_tok, int j, bool pooled, uint32_t layers_per_call, int p0 = 3,
                 uint32_t m = 5) {
	char why[256] = {};
	const std::string name = "refusal M=" + std::to_string(m) + " j=" + std::to_string(j) + " p0=" + std::to_string(p0) +
	                         (pooled ? " pooled" : " dedicated") + " layers/call=" + std::to_string(layers_per_call);
	Pair p(e, pooled);
	if (!p.ok || !p.Prefix(p0)) return Report(false, name, "setup");
	const uint32_t budget = e.dpl * layers_per_call;
	const std::vector<int32_t> rows = refuse_tok >= 0 ? RowTokens(p0, m, e.vocab, refuse_tok, j) : RowTokens(p0, m, e.vocab);
	for (int r = 0; r <= j; ++r) {
		const SslmGpuStatus st = OneRow(e, p.twin, rows[static_cast<size_t>(r)], budget);
		if (st != (r == j ? SSLM_SEQUENCE_REJECTED : SSLM_OK)) {
			std::snprintf(why, sizeof why, "twin row %d -> %d", r, static_cast<int>(st));
			return Report(false, name, why);
		}
	}
	int32_t admitted = 0, committed = -1;
	const SslmGpuStatus st = Chunk(e, p.chunk, rows, budget, &admitted);
	sslm_gpu_seq_last_rows_committed(p.chunk, &committed);
	if (st != SSLM_SEQUENCE_REJECTED || committed != j) {
		std::snprintf(why, sizeof why, "close -> %d, committed %d", static_cast<int>(st), committed);
		return Report(false, name, why);
	}
	if (!p.Equal("after the refusal", why, sizeof why)) return Report(false, name, why);
	Report(true, name, why);
}

void AbandonCase(const Env& e, bool pooled) {
	char why[256] = {};
	const std::string name = std::string("abandon M=5 after one slice") + (pooled ? " pooled" : " dedicated");
	Pair p(e, pooled);
	if (!p.ok || !p.Prefix(3)) return Report(false, name, "setup");
	std::vector<uint8_t> before, after;
	if (!Save(e, p.chunk, &before)) return Report(false, name, "save");
	const std::vector<int32_t> rows = RowTokens(3, 5, e.vocab);
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, rows.data(), 5, 0};
	int32_t admitted = 0;
	if (sslm_gpu_seq_embed_rows(e.ctx, p.chunk, &spec, &admitted) != SSLM_OK || Step(e, p.chunk, e.dpl) != SSLM_OK ||
	    sslm_gpu_seq_rows_abandon(e.ctx, p.chunk) != SSLM_OK || !Save(e, p.chunk, &after)) {
		return Report(false, name, "embed, slice or abandon failed");
	}
	if (FirstDiff(before, after) >= 0) return Report(false, name, "blob after abandon differs from the entry's");
	for (int32_t t : rows) {
		if (OneRow(e, p.twin, t, e.dpl * e.L) != SSLM_OK) return Report(false, name, "twin");
	}
	if (Chunk(e, p.chunk, rows, e.dpl * e.L, &admitted) != SSLM_OK) return Report(false, name, "second chunk");
	if (!p.Equal("after the second chunk", why, sizeof why)) return Report(false, name, why);
	Report(true, name, why);
}

void FaultCase(const Env& e, bool pooled) {
	char why[256] = {};
	const std::string name = std::string("fault (5.3) M=4 at slice 1") + (pooled ? " pooled" : " dedicated");
	Pair p(e, pooled);
	if (!p.ok || !p.Prefix(3)) return Report(false, name, "setup");
	std::vector<uint8_t> before, after;
	if (!Save(e, p.chunk, &before)) return Report(false, name, "save");
	const std::vector<int32_t> rows = RowTokens(3, 4, e.vocab);
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, rows.data(), 4, 0};
	int32_t admitted = 0;
	if (sslm_gpu_test_only_mr2_arm_slice_alloc_failure(e.ctx, 1) != SSLM_OK ||
	    sslm_gpu_seq_embed_rows(e.ctx, p.chunk, &spec, &admitted) != SSLM_OK || Step(e, p.chunk, e.dpl) != SSLM_OK) {
		return Report(false, name, "arm, embed or first slice failed");
	}
	const SslmGpuStatus st = sslm_decode_step_gpu(e.ctx, p.chunk, e.adapter, e.dpl);
	int32_t open = -1;
	sslm_gpu_seq_rows_open(p.chunk, &open);
	if (st != SSLM_GPU_ALLOCATION_FAILED || open != 0 || !Save(e, p.chunk, &after) || FirstDiff(before, after) >= 0) {
		std::snprintf(why, sizeof why, "faulting slice -> %d, open %d, entry blob restored %d", static_cast<int>(st), open,
		              FirstDiff(before, after) < 0);
		return Report(false, name, why);
	}
	for (int32_t t : rows) {
		if (OneRow(e, p.twin, t, e.dpl) != SSLM_OK) return Report(false, name, "twin");
	}
	if (Chunk(e, p.chunk, rows, e.dpl, &admitted) != SSLM_OK) return Report(false, name, "retry");
	if (!p.Equal("after the retry", why, sizeof why)) return Report(false, name, why);
	Report(true, name, why);
}

void BatchCase(const Env& e, bool pooled) {
	char why[256] = {};
	const std::string name = std::string("batch plain + M=5 + M=1") + (pooled ? " pooled" : " dedicated");
	Pair a(e, pooled), b(e, pooled), c(e, pooled);
	if (!a.ok || !b.ok || !c.ok || !a.Prefix(3) || !b.Prefix(3) || !c.Prefix(3)) return Report(false, name, "setup");
	const int32_t tok_a = TokenAt(40, e.vocab);
	const std::vector<int32_t> rows_b = RowTokens(3, 5, e.vocab), rows_c = RowTokens(4, 1, e.vocab);
	const uint32_t depth = e.dpl * e.L;
	if (OneRow(e, a.twin, tok_a, depth) != SSLM_OK) return Report(false, name, "twin a");
	for (int32_t t : rows_b)
		if (OneRow(e, b.twin, t, depth) != SSLM_OK) return Report(false, name, "twin b");
	if (OneRow(e, c.twin, rows_c[0], depth) != SSLM_OK) return Report(false, name, "twin c");
	int32_t adm_b = 0, adm_c = 0;
	SslmGpuRowsSpec sb{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, rows_b.data(), 5, 0};
	SslmGpuRowsSpec sc{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, rows_c.data(), 1, 0};
	if (sslm_gpu_seq_embed_token(e.ctx, a.chunk, tok_a) != SSLM_OK ||
	    sslm_gpu_seq_embed_rows(e.ctx, b.chunk, &sb, &adm_b) != SSLM_OK ||
	    sslm_gpu_seq_embed_rows(e.ctx, c.chunk, &sc, &adm_c) != SSLM_OK) {
		return Report(false, name, "embed");
	}
	SslmGpuSequenceHandle* seqs[3] = {a.chunk, b.chunk, c.chunk};
	const SslmGpuAdapterHandle* ads[3] = {e.adapter, e.adapter, e.adapter};
	SslmGpuStatus st[3] = {};
	// The whole depth for two members: the third (the M = 1 chunk) is exhausted and unchanged.
	int32_t open_c = -1;
	if (sslm_decode_step_batch_gpu(e.ctx, seqs, ads, 3, 2 * depth, st) != SSLM_OK || st[0] != SSLM_OK ||
	    st[1] != SSLM_OK || st[2] != SSLM_BATCH_BUDGET_EXHAUSTED || sslm_gpu_seq_rows_open(c.chunk, &open_c) != SSLM_OK ||
	    open_c != 1 || *SslmGpuSeqHandleLayerIndexForBench(c.chunk) != 0) {
		std::snprintf(why, sizeof why, "two-member budget -> %d %d %d, c open %d", static_cast<int>(st[0]),
		              static_cast<int>(st[1]), static_cast<int>(st[2]), open_c);
		return Report(false, name, why);
	}
	// The next call: a and b are at full depth (a's one-row token rests there, b's chunk closed), so only c runs.
	SslmGpuSequenceHandle* rest[1] = {c.chunk};
	if (sslm_decode_step_batch_gpu(e.ctx, rest, ads, 1, depth, st) != SSLM_OK || st[0] != SSLM_OK) {
		std::snprintf(why, sizeof why, "c's call -> %d", static_cast<int>(st[0]));
		return Report(false, name, why);
	}
	int32_t open_b = -1, committed_b = -1;
	sslm_gpu_seq_rows_open(b.chunk, &open_b);
	sslm_gpu_seq_rows_open(c.chunk, &open_c);
	sslm_gpu_seq_last_rows_committed(b.chunk, &committed_b);
	if (open_b != 0 || open_c != 0 || committed_b != 5) {
		std::snprintf(why, sizeof why, "after L layers: open %d %d, committed %d", open_b, open_c, committed_b);
		return Report(false, name, why);
	}
	if (!a.Equal("plain member", why, sizeof why) || !b.Equal("M=5 member", why, sizeof why) ||
	    !c.Equal("M=1 member", why, sizeof why)) {
		return Report(false, name, why);
	}
	Report(true, name, why);
}

void StatusCase(const Env& e, bool pooled) {
	char why[256] = {};
	const std::string name = std::string("statuses (save CHUNK_OPEN, headroom, ") + (pooled ? "budget limit) pooled" : "cap) dedicated");
	Pair p(e, pooled);
	if (!p.ok || !p.Prefix(29)) return Report(false, name, "setup");
	const std::vector<int32_t> rows = RowTokens(29, 8, e.vocab);
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, rows.data(), 8, 0};
	int32_t admitted = -1;
	if (sslm_gpu_seq_embed_rows(e.ctx, p.chunk, &spec, &admitted) != SSLM_OK || admitted != 3) {
		std::snprintf(why, sizeof why, "headroom 3 of cap 32: admitted %d", admitted);
		return Report(false, name, why);
	}
	std::vector<uint8_t> canary(64, 0xAB);
	size_t sz = canary.size();
	const SslmGpuStatus sv = sslm_gpu_seq_save(e.ctx, p.chunk, canary.data(), &sz);
	bool untouched = sz == canary.size();
	for (uint8_t x : canary) untouched = untouched && x == 0xAB;
	if (sv != SSLM_GPU_ROWS_CHUNK_OPEN || !untouched) {
		std::snprintf(why, sizeof why, "save with a chunk open -> %d, untouched %d", static_cast<int>(sv), untouched);
		return Report(false, name, why);
	}
	for (uint32_t g = 0; g <= e.L; ++g) {
		int32_t open = 0;
		sslm_gpu_seq_rows_open(p.chunk, &open);
		if (open == 0) break;
		if (Step(e, p.chunk, e.dpl) != SSLM_OK) return Report(false, name, "advance");
	}
	std::vector<uint8_t> before, after;
	if (!Save(e, p.chunk, &before)) return Report(false, name, "save after the close");
	admitted = -7;
	const SslmGpuStatus cap = sslm_gpu_seq_embed_rows(e.ctx, p.chunk, &spec, &admitted);
	// A pooled sequence's budget (32) is below the model's cap: its limit binds, with the budget's status.
	const SslmGpuStatus want = pooled ? SSLM_GPU_KV_BUDGET_EXCEEDED : SSLM_GPU_CONTEXT_CAP_EXCEEDED;
	if (cap != want || admitted != -7 || !Save(e, p.chunk, &after) || FirstDiff(before, after) >= 0) {
		std::snprintf(why, sizeof why, "at the cap -> %d, admitted %d", static_cast<int>(cap), admitted);
		return Report(false, name, why);
	}
	Report(true, name, why);
}

}  // namespace

// ---- MRX (E-11, rulings 21, 21b, 22): what remains of MR4's public forced surface: its refusals ----

std::vector<int32_t> Chain(const Env& e, SslmGpuSequenceHandle* s) {
	int32_t ids[64], n = 0, acc = 0;
	if (sslm_gpu_seq_forced_tokens(e.ctx, s, ids, 64, &n, &acc) != SSLM_OK || n < 0) return {};
	return std::vector<int32_t>(ids, ids + n);
}

SslmGpuSequenceHandle* NewBound(const Env& e, int32_t schema, int64_t cap = 64) {
	SslmGpuSequenceHandle* s = nullptr;
	if (sslm_gpu_seq_create(e.ctx, e.model, cap, &s) != SSLM_OK) return nullptr;
	if (schema >= 0 && SslmGpuSeqSetSchemaForG5Bridge(e.ctx, s, schema) != SSLM_OK) {
		sslm_gpu_seq_release(e.ctx, s);
		return nullptr;
	}
	return s;
}

// embed_rows with `kind`/`flags` on a fresh bound sequence must be the bad spec, writing nothing (no admitted
// count, no chunk, the blob unchanged).
void RowsRefused(const Env& e, int32_t schema, const std::vector<int32_t>& rows, int32_t kind, uint32_t flags,
                 const std::string& name) {
	char why[160];
	SslmGpuSequenceHandle* s = NewBound(e, schema);
	if (!s) return Report(false, name, "setup");
	std::vector<uint8_t> before, after;
	(void)Save(e, s, &before);
	const SslmGpuRowsSpec spec{sizeof spec, kind, rows.data(), static_cast<int32_t>(rows.size()), flags};
	int32_t adm = -7, open = -1;
	const SslmGpuStatus st = sslm_gpu_seq_embed_rows(e.ctx, s, &spec, &adm);
	(void)sslm_gpu_seq_rows_open(s, &open);
	(void)Save(e, s, &after);
	std::snprintf(why, sizeof why, "-> %d, admitted %d, open %d, blob %s", static_cast<int>(st), adm, open,
	              before == after ? "equal" : "changed");
	Report(st == SSLM_SEQUENCE_KV_BUFFER_MISMATCH && adm == -7 && open == 0 && before == after, name, why);
	sslm_gpu_seq_release(e.ctx, s);
}

void ForcedCases(const Env& e, const char* schema_name) {
	char why[256];
	const int32_t schema = SslmGpuSchemaLookupForG5Bridge(e.model, schema_name);
	if (schema < 0) {
		Report(false, "forced setup", "schema not found");
		return;
	}
	std::vector<int32_t> chain;
	if (SslmGpuSequenceHandle* s = NewBound(e, schema)) {
		chain = Chain(e, s);
		sslm_gpu_seq_release(e.ctx, s);
	}
	if (chain.size() < 2) return Report(false, "forced setup", "no start chain of 2 or more rows");
	if (chain.size() > SSLM_GPU_MAX_ROWS) chain.resize(SSLM_GPU_MAX_ROWS);
	// The removed kind (1) and flag (0x1): bad specs, even on the very chain the schema forces.
	RowsRefused(e, schema, chain, 1, 0u, "kind 1 (removed schema-forced) on the forced chain");
	RowsRefused(e, schema, chain, 1, 0x1u, "kind 1 + flag 0x1 on the forced chain");
	RowsRefused(e, schema, chain, SSLM_GPU_ROWS_PROMPT, 0x1u, "PROMPT + flag 0x1");
	// PROMPT on the same chain is admitted, and closes.
	{
		SslmGpuSequenceHandle* s = NewBound(e, schema);
		const SslmGpuRowsSpec spec{sizeof spec, SSLM_GPU_ROWS_PROMPT, chain.data(), static_cast<int32_t>(chain.size()), 0};
		int32_t adm = 0, open = -1;
		const SslmGpuStatus st = s ? sslm_gpu_seq_embed_rows(e.ctx, s, &spec, &adm) : SSLM_DEVICE_LOST;
		const SslmGpuStatus cl = st == SSLM_OK ? Step(e, s, e.L * e.dpl) : st;
		if (s) (void)sslm_gpu_seq_rows_open(s, &open);
		std::snprintf(why, sizeof why, "-> %d, admitted %d, close %d, open %d", static_cast<int>(st), adm,
		              static_cast<int>(cl), open);
		Report(st == SSLM_OK && adm == static_cast<int32_t>(chain.size()) && cl == SSLM_OK && open == 0,
		       "PROMPT on the forced chain admitted and closed", why);
		if (s) sslm_gpu_seq_release(e.ctx, s);
	}
	// The jump verb: NULL, SSLM_JUMP_DEFAULT (0) and SSLM_JUMP_EXACT (1) are Exact, refused until MR5 with nothing
	// changed; a mode outside the enum, or a draft with mode 0, is malformed options.
	struct J { bool null_opts; int32_t mode; bool draft; SslmGpuStatus want; const char* what; };
	const J js[] = {{true, 0, false, SSLM_SEQUENCE_REJECTED, "jump, NULL options (Exact until MR5)"},
	                {false, 0, false, SSLM_SEQUENCE_REJECTED, "jump, mode 0 = SSLM_JUMP_DEFAULT (Exact until MR5)"},
	                {false, 1, false, SSLM_SEQUENCE_REJECTED, "jump, SSLM_JUMP_EXACT (until MR5)"},
	                {false, 1, true, SSLM_SEQUENCE_REJECTED, "jump, SSLM_JUMP_EXACT with a draft (until MR5)"},
	                {false, 2, false, SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "jump, mode 2 (outside the enum)"},
	                {false, 0, true, SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "jump, mode 0 with a draft (ruling 21b)"}};
	for (const J& j : js) {
		SslmGpuSequenceHandle* s = NewBound(e, schema);
		if (!s) {
			Report(false, j.what, "setup");
			continue;
		}
		std::vector<uint8_t> before, after;
		(void)Save(e, s, &before);
		const SslmGpuJumpOptions o{sizeof(SslmGpuJumpOptions), j.mode, j.draft ? chain.data() : nullptr,
		                           j.draft ? static_cast<int32_t>(chain.size()) : 0};
		int32_t forced[64], n = -7;
		const SslmGpuStatus st = sslm_gpu_seq_jump_forward(e.ctx, s, -1, 8, j.null_opts ? nullptr : &o, forced, 64, &n);
		(void)Save(e, s, &after);
		std::snprintf(why, sizeof why, "-> %d (want %d), count %d, blob %s", static_cast<int>(st),
		              static_cast<int>(j.want), n, before == after ? "equal" : "changed");
		Report(st == j.want && n == -7 && before == after, j.what, why);
		sslm_gpu_seq_release(e.ctx, s);
	}
}

int main(int argc, char** argv) {
	std::string model_path, adapter_path, shaders;
	bool quick = false, refusal_only = false;
	int scan = 64;
	int rope_positions = 0;
	std::string forced_schema;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--forced-schema=", 0) == 0) {
			forced_schema = a.substr(16);
			continue;
		}
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--adapter=", 0) == 0) adapter_path = a.substr(10);
		else if (a.rfind("--shaders=", 0) == 0) shaders = a.substr(10);
		else if (a.rfind("--scan=", 0) == 0) scan = std::atoi(a.c_str() + 7);
		else if (a.rfind("--rope-positions=", 0) == 0) rope_positions = std::atoi(a.c_str() + 17);
		else if (a == "--quick") quick = true;
		else if (a == "--refusal-only") refusal_only = true;
		else {
			std::printf("usage: multirow_rows_smoke --model=PATH [--adapter=PATH] [--shaders=DIR] [--quick] [--scan=N]\n");
			return 2;
		}
	}
	std::vector<uint8_t> model_bytes, adapter_bytes;
	superslm::SslmModelView view{}, adapter_view{};
	std::string err;
	if (model_path.empty() || !ReadFileBytes(model_path, &model_bytes) ||
	    superslm::SslmModel::Load(model_bytes.data(), model_bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
		std::printf("FAIL cannot load model '%s' %s\n", model_path.c_str(), err.c_str());
		return 1;
	}
	if (!adapter_path.empty() &&
	    (!ReadFileBytes(adapter_path, &adapter_bytes) ||
	     superslm::SslmModel::Load(adapter_bytes.data(), adapter_bytes.size(), adapter_view, &err) !=
	         superslm::SslmModelStatus::Ok)) {
		std::printf("FAIL cannot load adapter '%s' %s\n", adapter_path.c_str(), err.c_str());
		return 1;
	}
	if (rope_positions > 0) {
		for (const char* n : {"cos", "sin"}) {
			auto* t = const_cast<superslm::SslmTensorView*>(view.rope_tables.Tensor(n));
			if (t == nullptr || t->rank != 2 || t->shape[0] < static_cast<uint32_t>(rope_positions)) {
				std::printf("FAIL --rope-positions: no rank-2 '%s' table of at least %d rows\n", n, rope_positions);
				return 1;
			}
			t->shape[0] = static_cast<uint32_t>(rope_positions);
			t->elem_count = uint64_t{t->shape[0]} * t->shape[1];
		}
	}
	Env e;
	e.vocab = static_cast<int32_t>(view.config.vocab_size);
	e.L = view.config.num_hidden_layers;
	bool qk = false;
	for (uint32_t l = 0; l < e.L; ++l)
		if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) qk = true;
	e.dpl = superslm_gpu::DispatchesPerLayer(qk);
	GpuContextConfig cfg{};
	cfg.shader_dir = shaders.empty() ? nullptr : shaders.c_str();
	if (sslm_gpu_context_create(cfg, &e.ctx) != SSLM_OK ||
	    sslm_gpu_model_map(e.ctx, &view, GpuResidencyConfig{}, &e.model) != SSLM_OK) {
		std::printf("FAIL context or model map\n");
		return 1;
	}
	if (!adapter_path.empty() && sslm_gpu_adapter_map(e.ctx, e.model, &adapter_view, &e.adapter) != SSLM_OK) {
		std::printf("FAIL adapter map\n");
		return 1;
	}

	if (!forced_schema.empty()) {
		ForcedCases(e, forced_schema.c_str());
		sslm_gpu_model_unmap(e.ctx, e.model);
		sslm_gpu_context_destroy(e.ctx);
		std::printf("multi-row recorder smoke (MRX forced refusals): %s (%d cases, %d failed)\n", g_failed ? "FAIL" : "PASS",
		            g_cases, g_failed);
		return g_failed ? 1 : 0;
	}
	{
		// MRX: kind 1 (the removed schema-forced kind) is a bad spec, before any table check.
		SslmGpuSequenceHandle* s = nullptr;
		const int32_t tok = 1;
		SslmGpuRowsSpec spec{sizeof spec, 1, &tok, 1, 0};
		int32_t adm = -1;
		SslmGpuStatus st = SSLM_DEVICE_LOST;
		if (sslm_gpu_seq_create(e.ctx, e.model, 8, &s) == SSLM_OK) st = sslm_gpu_seq_embed_rows(e.ctx, s, &spec, &adm);
		char why[96];
		std::snprintf(why, sizeof why, "kind 1 -> %d (admitted %d)", static_cast<int>(st), adm);
		Report(st == SSLM_SEQUENCE_KV_BUFFER_MISMATCH && adm == -1, "kind 1 on a model with no forced-next table", why);
		if (s) sslm_gpu_seq_release(e.ctx, s);
	}
	const std::vector<uint32_t> ms = quick ? std::vector<uint32_t>{1, 2, 3, 5, 8} : std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8};
	const std::vector<int> p0s = quick ? std::vector<int>{3} : std::vector<int>{0, 3, 13};
	for (int pooled = 0; pooled < 2 && !refusal_only; ++pooled) {
		for (uint32_t layers_per_call : {1u, e.L}) {
			for (int p0 : p0s) {
				for (uint32_t m : ms) CleanCase(e, m, p0, pooled != 0, layers_per_call);
			}
		}
	}
	int refused_layer = -1;
	const int32_t refuse_tok = rope_positions > 0 ? -1 : FindRefusingToken(e, scan, &refused_layer);
	if (rope_positions > 0) {
		std::printf("NOTE refusal: every row at position %d or later refuses (RoPE tables truncated)\n", rope_positions);
		for (int pooled = 0; pooled < 2; ++pooled) {
			for (int j : {0, 1, 3, 4}) {
				RefusalCase(e, -1, j, pooled != 0, 1, rope_positions - j);
				RefusalCase(e, -1, j, pooled != 0, e.L, rope_positions - j);
			}
			RefusalCase(e, -1, 0, pooled != 0, 1, rope_positions, 1);  // M = 1: the one-row body's refusal
			RefusalCase(e, -1, 1, pooled != 0, e.L, rope_positions - 1, 8);
		}
	} else if (refuse_tok < 0) {
		std::printf("NOTE refusal: no token among the first %d refuses at p0 = 3 on this model; refusal cases not run\n",
		            scan);
	} else {
		std::printf("NOTE refusal: token %d refuses at layer %d; %zu of the first %d tokens do not refuse\n", refuse_tok,
		            refused_layer, g_clean_tokens.size(), scan);
		for (int pooled = 0; pooled < 2; ++pooled) {
			for (int j : {0, 1, 4}) {
				RefusalCase(e, refuse_tok, j, pooled != 0, 1);
				RefusalCase(e, refuse_tok, j, pooled != 0, e.L);
			}
		}
	}
	g_clean_tokens.clear();  // the cases below use the model's own stream again
	SslmGpuMr2Seams warm{};
	sslm_gpu_test_only_mr2_seams(e.ctx, &warm);
	for (int pooled = 0; pooled < 2 && !refusal_only; ++pooled) {
		AbandonCase(e, pooled != 0);
		FaultCase(e, pooled != 0);
		BatchCase(e, pooled != 0);
		StatusCase(e, pooled != 0);
	}
	SslmGpuMr2Seams end{};
	sslm_gpu_test_only_mr2_seams(e.ctx, &end);
	{
		char why[256];
		std::snprintf(why, sizeof why,
		              "allocations %llu -> %llu after warm-up, sets %u (free %u), chunks open %u, peak %u, widened lists %llu, "
		              "dispatches per list max %u",
		              static_cast<unsigned long long>(warm.allocations), static_cast<unsigned long long>(end.allocations),
		              end.row_buffer_sets, end.row_buffer_free, end.chunks_open, end.chunks_open_peak,
		              static_cast<unsigned long long>(end.widened_lists), end.max_widened_list_dispatches);
		const bool pass = end.allocations == warm.allocations && end.row_buffer_sets <= end.chunks_open_peak &&
		                  end.row_buffer_free == end.row_buffer_sets && end.chunks_open == 0 &&
		                  end.max_widened_list_dispatches <= e.L * e.dpl;
		Report(pass, "seams", why);
	}
	if (e.adapter) sslm_gpu_adapter_unmap(e.ctx, e.adapter);
	sslm_gpu_model_unmap(e.ctx, e.model);
	sslm_gpu_context_destroy(e.ctx);
	std::printf("multi-row recorder smoke: %s (%d cases, %d failed)\n", g_failed ? "FAIL" : "PASS", g_cases, g_failed);
	return g_failed ? 1 : 0;
}
