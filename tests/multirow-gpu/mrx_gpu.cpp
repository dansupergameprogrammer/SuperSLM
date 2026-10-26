// Multi-row GPU plan (rev 1.3.2), slice MRX (plan §16 erratum E-11, ruling 20): Canonical removal, the GPU cells.
// Built by mrx_wine.sh under mingw and run under Wine on the software GPU. Public verbs and the G5 bridge only (the
// shape of mr4_83_canon.cpp), so the same file is the RECORDER for 1.17's GPU saves: built from the v1.17.0 tree with
// -DMRX_V117 (which leaves out the multi-row verbs 1.17 does not have), `--record=DIR` writes a blob saved right after
// 1.17's Canonical jump (opts NULL) per schema, with the decode-only stream that follows it. The GPU jump verb first
// shipped in 1.17.0, so 1.17 is the only release with GPU Canonical saves (the CPU's 1.15-1.17 are mrx_cpu's x.5).
//
// Cells (on F-SFD, tools/gen_schema_fast_decode_fixture.py's fsfd.sslm; three schemas):
//   g.1  The old Canonical value (mode 0) is refused by sslm_gpu_seq_jump_forward through the malformed-options status
//        (SSLM_SEQUENCE_KV_BUFFER_MISMATCH), with nothing changed: the count unwritten, out_forced untouched, the save
//        blob, walk state, context length and ready flag equal, and a fresh bound sequence still bindable. Controls:
//        mode 2 and -1 stay refused the same way.
//   g.2  NULL options equal explicit Exact (mode 1, no draft) byte for byte: status, count, out_forced, the save blob,
//        then four decode-bridge tokens and the blob after them. Budgets 1, 4 and 8, after 0 and 2 decodes. Red until
//        MR5 (GPU Exact).
//   g.3  sslm_gpu_seq_embed_rows refuses kind 1 (the removed SCHEMA_FORCED) as a bad spec
//        (SSLM_SEQUENCE_KV_BUFFER_MISMATCH), with flags 0 and with 0x1 (FIRST_PREWALKED), the rows being the chain
//        the schema forces (so the kind is the only thing wrong): out_admitted unwritten, no chunk open, the save blob
//        equal. Control: kind 2 (EXACT_DRAFT) stays a bad spec; kind 0 (PROMPT) on the same tokens is admitted.
//   g.4a 1.17's Canonical saves restore: SSLM_OK, the restored handle saves the same bytes, and decode-only it gives
//        the recorder's own next 8 tokens.
//   g.4b ...and continue under the mode asked for (E-11 (d)): an explicit Exact jump, then decodes, gives the same 8
//        tokens, and NULL options give what explicit Exact gives (tokens and blob). Red until MR5.
//
// The literal values are used, never SSLM_JUMP_CANONICAL or SSLM_GPU_ROWS_SCHEMA_FORCED: both names leave the
// headers at MRX, and this file has to compile there and in 1.17's tree.
//
// Usage: mrx_gpu.exe --fixtures=DIR [--legacy=DIR] [--cells=g.1,g.2,g.3,g.4]   |   mrx_gpu.exe --fixtures=DIR --record=DIR
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

namespace {
using S = SslmGpuStatus;
constexpr int32_t kModeCanonicalOld = 0;  // SSLM_JUMP_CANONICAL before MRX; reserved, never reused (E-11 (a))
constexpr int32_t kModeExact = 1;         // SSLM_JUMP_EXACT
constexpr int32_t kRowsSchemaForcedOld = 1;  // SSLM_GPU_ROWS_SCHEMA_FORCED before MRX (E-11 (c))
constexpr uint32_t kRowsFirstPrewalked = 0x1u;
constexpr int kNext = 8;
const char* kSchemas[] = {"potion_shop_order", "prompt_result", "longkey"};

std::string g_fixtures, g_legacy, g_record, g_cell;
std::set<std::string> g_cells;
std::map<std::string, int> g_fail, g_checks;
void Check(bool ok, const std::string& what) {
	++g_checks[g_cell];
	if (!ok && ++g_fail[g_cell] <= 16) std::printf("FAIL %s: %s\n", g_cell.c_str(), what.c_str());
}
std::string F(const char* fmt, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(b, sizeof b, fmt, ap);
	va_end(ap);
	return b;
}
bool ReadAll(const std::string& p, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(p.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long n = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(n > 0 ? static_cast<size_t>(n) : 0);
	const bool ok = n > 0 && std::fread(out->data(), 1, out->size(), f) == out->size();
	std::fclose(f);
	return ok;
}
bool WriteAll(const std::string& p, const void* d, size_t n) {
	std::FILE* f = std::fopen(p.c_str(), "wb");
	if (!f) return false;
	const bool ok = std::fwrite(d, 1, n, f) == n;
	std::fclose(f);
	return ok;
}

struct Engine {
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	uint32_t full = 0;
	bool ok = false;
	Engine() {
		std::string err;
		if (!ReadAll(g_fixtures + "/fsfd.sslm", &bytes) ||
		    superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
			std::printf("FAIL load fsfd.sslm %s\n", err.c_str());
			return;
		}
		full = superslm_gpu::DispatchesPerLayer(false) * view.config.num_hidden_layers;  // F-SFD has no QK norm
		if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != S::SSLM_OK ||
		    sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != S::SSLM_OK) {
			std::printf("FAIL context\n");
			return;
		}
		ok = true;
	}
	~Engine() {
		if (model) sslm_gpu_model_unmap(ctx, model);
		if (ctx) sslm_gpu_context_destroy(ctx);
	}
	std::vector<int32_t> Prompt(uint32_t stream) const {
		std::vector<int32_t> v;
		for (int i = 0; i < 12; ++i)
			v.push_back(static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u + 7919u * stream) % 20000u + 200u));
		return v;
	}
	std::vector<uint8_t> Save(SslmGpuSequenceHandle* s) const {
		size_t need = 0;
		sslm_gpu_seq_save(ctx, s, nullptr, &need);
		std::vector<uint8_t> b(need ? need : 1, 0);
		size_t sz = b.size();
		if (sslm_gpu_seq_save(ctx, s, b.data(), &sz) != S::SSLM_OK) return {};
		b.resize(sz);
		return b;
	}
	int32_t Ready(SslmGpuSequenceHandle* s) const {
		int32_t r = -1;
		sslm_gpu_seq_ready_for_logits(ctx, s, &r);
		return r;
	}
	int32_t Chain(SslmGpuSequenceHandle* s, std::vector<int32_t>* ids = nullptr) const {
		int32_t buf[64] = {}, count = 0, acc = 0;
		if (sslm_gpu_seq_forced_tokens(ctx, s, buf, 64, &count, &acc) != S::SSLM_OK) return -1;
		if (ids) ids->assign(buf, buf + (count < 64 ? count : 64));
		return count;
	}
	// A dedicated sequence bound to `schema` with the stream's prompt prefilled (the ready shape), or null.
	SslmGpuSequenceHandle* Bound(const char* schema, uint32_t stream, int32_t* last) const {
		SslmGpuSequenceHandle* s = nullptr;
		if (sslm_gpu_seq_create(ctx, model, 240, &s) != S::SSLM_OK) return nullptr;
		const int32_t idx = SslmGpuSchemaLookupForG5Bridge(model, schema);
		const std::vector<int32_t> p = Prompt(stream);
		if (SslmGpuSeqSetSchemaForG5Bridge(ctx, s, idx) != S::SSLM_OK ||
		    SslmGpuSeqPrefillPromptForG5Bridge(ctx, s, p.data(), static_cast<int32_t>(p.size()), full) != S::SSLM_OK) {
			sslm_gpu_seq_release(ctx, s);
			return nullptr;
		}
		*last = p.back();
		return s;
	}
	// One decode-bridge token. `pending` is the produced token in hand (or -1); on return it holds the new token.
	S Decode(SslmGpuSequenceHandle* s, int32_t* pending, int32_t* last, int32_t* tok) const {
		*tok = -1;
		const S st = SslmGpuSeqDecodeStepForG5Bridge(ctx, s, *pending >= 0 ? *pending : *last, full, tok);
		if (st == S::SSLM_OK && *tok >= 0) *pending = *last = *tok;
		return st;
	}
	// Bound, then `decodes` decode-bridge tokens: the entry the jump cells start from.
	SslmGpuSequenceHandle* Pending(const char* schema, uint32_t stream, int decodes, int32_t* pending, int32_t* last) const {
		SslmGpuSequenceHandle* s = Bound(schema, stream, last);
		*pending = -1;
		for (int d = 0; s && d < decodes; ++d) {
			int32_t t = -1;
			if (Decode(s, pending, last, &t) != S::SSLM_OK || t < 0) {
				sslm_gpu_seq_release(ctx, s);
				return nullptr;
			}
		}
		return s;
	}
};

struct Jump {
	S st = S::SSLM_OK;
	int32_t count = -7;
	std::vector<int32_t> out;
	bool operator==(const Jump& o) const { return st == o.st && count == o.count && out == o.out; }
};
// mode < -100: NULL options.
Jump DoJump(const Engine& e, SslmGpuSequenceHandle* s, int32_t pending, int32_t budget, int32_t mode) {
	Jump j;
	j.out.assign(16, -7);
	SslmGpuJumpOptions o{};
	o.struct_size = sizeof o;
	o.mode = mode;
	j.st = sslm_gpu_seq_jump_forward(e.ctx, s, pending, budget, mode < -100 ? nullptr : &o, j.out.data(),
	                                 static_cast<int32_t>(j.out.size()), &j.count);
	return j;
}
constexpr int32_t kNull = -1000;

// After an admitting jump the ready flag is set and the pending token is consumed.
void AfterJump(const Engine& e, SslmGpuSequenceHandle* s, const Jump& j, int32_t* pending, int32_t* last) {
	if (j.st != S::SSLM_OK) return;
	if (j.count > 0) *last = j.out[static_cast<size_t>(j.count) - 1];
	if (e.Ready(s) == 1) *pending = -1;
}

std::string Tag(const char* schema, int a, int b) { return F("%s/%d/%d", schema, a, b); }

// ------------------------------------------------------------------------------------------------ g.1
void CellG1(Engine& e) {
	int bit = 0;
	for (const char* schema : kSchemas)
		for (int32_t mode : {kModeCanonicalOld, 2, -1})
			for (int decodes : {0, 2}) {
				const std::string tag = F("%s mode %d after %d decodes", schema, mode, decodes);
				int32_t pending = -1, last = -1;
				SslmGpuSequenceHandle* s = e.Pending(schema, 11u, decodes, &pending, &last);
				if (!s) { Check(false, tag + ": setup"); continue; }
				const int32_t chain = e.Chain(s);
				const std::vector<uint8_t> before = e.Save(s);
				const uint32_t walk = SslmGpuSeqWalkStateForG5Bridge(s);
				const int64_t len = *SslmGpuSeqHandleContextLengthForBench(s);
				const int32_t ready = e.Ready(s);
				const Jump j = DoJump(e, s, pending, 4, mode);
				Check(j.st == S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
				      F("%s: returned %u, want SSLM_SEQUENCE_KV_BUFFER_MISMATCH (%u)%s", tag.c_str(), (unsigned)j.st,
				        (unsigned)S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, chain > 0 ? "" : " (no chain here)"));
				Check(j.count == -7, F("%s: the count was written (%d)", tag.c_str(), j.count));
				bool untouched = true;
				for (int32_t x : j.out) untouched = untouched && x == -7;
				Check(untouched, tag + ": out_forced was written");
				Check(!before.empty() && e.Save(s) == before, tag + ": the save blob changed");
				Check(SslmGpuSeqWalkStateForG5Bridge(s) == walk && *SslmGpuSeqHandleContextLengthForBench(s) == len && e.Ready(s) == ready,
				      tag + ": the walk, context length or ready flag changed");
				if (chain > 0) ++bit;
				sslm_gpu_seq_release(e.ctx, s);
				// Bindability: a fresh bound sequence refused the same way is still bindable (a rebind succeeds).
				SslmGpuSequenceHandle* f = nullptr;
				if (sslm_gpu_seq_create(e.ctx, e.model, 240, &f) != S::SSLM_OK) { Check(false, tag + ": fresh"); continue; }
				const int32_t idx = SslmGpuSchemaLookupForG5Bridge(e.model, schema);
				S b = SslmGpuSeqSetSchemaForG5Bridge(e.ctx, f, idx);
				const Jump jf = DoJump(e, f, -1, 4, mode);
				if (b == S::SSLM_OK) b = SslmGpuSeqSetSchemaForG5Bridge(e.ctx, f, idx);
				Check(jf.st == S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && b == S::SSLM_OK,
				      F("%s: fresh bound: jump %u, rebind %u (want 7, 0: refused and still bindable)", tag.c_str(), (unsigned)jf.st, (unsigned)b));
				sslm_gpu_seq_release(e.ctx, f);
			}
	Check(bit >= 6, F("only %d mode cases had a forced chain at the call (want 6)", bit));
}

// ------------------------------------------------------------------------------------------------ g.2
void CellG2(Engine& e) {
	int compared = 0;
	for (const char* schema : kSchemas)
		for (int32_t budget : {1, 4, 8})
			for (int decodes : {0, 2}) {
				const std::string tag = Tag(schema, budget, decodes);
				int32_t pa = -1, la = -1, pb = -1, lb = -1;
				SslmGpuSequenceHandle* a = e.Pending(schema, 23u, decodes, &pa, &la);
				SslmGpuSequenceHandle* b = e.Pending(schema, 23u, decodes, &pb, &lb);
				if (!a || !b) {
					Check(false, tag + ": setup");
					if (a) sslm_gpu_seq_release(e.ctx, a);
					if (b) sslm_gpu_seq_release(e.ctx, b);
					continue;
				}
				if (e.Chain(a) > 0) ++compared;
				const Jump ja = DoJump(e, a, pa, budget, kNull), jb = DoJump(e, b, pb, budget, kModeExact);
				Check(ja == jb, F("%s: NULL options returned %u count %d, explicit Exact %u count %d", tag.c_str(), (unsigned)ja.st, ja.count,
				                  (unsigned)jb.st, jb.count));
				Check(e.Save(a) == e.Save(b), tag + ": the save blob after NULL differs from explicit Exact's");
				AfterJump(e, a, ja, &pa, &la);
				AfterJump(e, b, jb, &pb, &lb);
				std::vector<int32_t> ta, tb;
				for (int k = 0; k < 4; ++k) {
					int32_t x = -1, y = -1;
					if (e.Decode(a, &pa, &la, &x) == S::SSLM_OK) ta.push_back(x);
					if (e.Decode(b, &pb, &lb, &y) == S::SSLM_OK) tb.push_back(y);
				}
				Check(ta == tb && e.Save(a) == e.Save(b), tag + ": the continuation after NULL differs from explicit Exact's");
				sslm_gpu_seq_release(e.ctx, a);
				sslm_gpu_seq_release(e.ctx, b);
			}
	Check(compared >= 6, F("only %d cases had a forced chain at the jump (want 6)", compared));
}

// ------------------------------------------------------------------------------------------------ g.3
#ifndef MRX_V117
void CellG3(Engine& e) {
	int bit = 0;
	for (const char* schema : kSchemas)
		for (int decodes : {0, 2}) {
			int32_t pending = -1, last = -1;
			SslmGpuSequenceHandle* s = e.Pending(schema, 37u, decodes, &pending, &last);
			if (!s) { Check(false, F("%s/%d: setup", schema, decodes)); continue; }
			std::vector<int32_t> chain;
			e.Chain(s, &chain);
			if (!chain.empty()) ++bit;
			else chain.push_back(last);  // no chain here: the kind must be refused whatever the rows are
			struct Case { int32_t kind; uint32_t flags; bool admitted_ok; const char* what; };
			const Case cases[] = {{kRowsSchemaForcedOld, 0, false, "kind 1 (SCHEMA_FORCED)"},
			                      {kRowsSchemaForcedOld, kRowsFirstPrewalked, false, "kind 1 + FIRST_PREWALKED"},
			                      {2, 0, false, "kind 2 (EXACT_DRAFT, control)"},
			                      {0, 0, true, "kind 0 (PROMPT, control)"}};
			for (const Case& c : cases) {
				const std::string tag = F("%s/%d %s", schema, decodes, c.what);
				std::vector<int32_t> rows;
				if (c.flags & kRowsFirstPrewalked) rows.push_back(last);
				for (int32_t x : chain) if (rows.size() < 4) rows.push_back(x);
				const std::vector<uint8_t> before = e.Save(s);
				SslmGpuRowsSpec spec{};
				spec.struct_size = sizeof spec;
				spec.kind = c.kind;
				spec.tokens = rows.data();
				spec.count = static_cast<int32_t>(rows.size());
				spec.flags = c.flags;
				int32_t admitted = -7, open = -1;
				const S st = sslm_gpu_seq_embed_rows(e.ctx, s, &spec, &admitted);
				sslm_gpu_seq_rows_open(s, &open);
				if (c.admitted_ok) {
					Check(st == S::SSLM_OK && admitted > 0, F("%s: returned %u admitted %d (the control must be admitted)", tag.c_str(), (unsigned)st, admitted));
				} else {
					Check(st == S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
					      F("%s: returned %u, want a bad spec SSLM_SEQUENCE_KV_BUFFER_MISMATCH (%u)", tag.c_str(), (unsigned)st,
					        (unsigned)S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH));
					Check(admitted == -7, F("%s: out_admitted was written (%d)", tag.c_str(), admitted));
					Check(open == 0, F("%s: a chunk is open (%d rows)", tag.c_str(), open));
				}
				if (open > 0) sslm_gpu_seq_rows_abandon(e.ctx, s);
				if (!c.admitted_ok) Check(!before.empty() && e.Save(s) == before, tag + ": the save blob changed");
			}
			sslm_gpu_seq_release(e.ctx, s);
		}
	Check(bit >= 3, F("only %d entries had a forced chain to offer as rows (want 3)", bit));
}
#endif

// ------------------------------------------------------------------------------------------------ recorder
// Per schema: decode-bridge tokens until a NULL-options jump (budget 4) admits a chain, then the save, then kNext
// decode-only tokens. `<schema>.blob` and `<schema>.next` ("last t1 .. t8").
int Record(Engine& e) {
	int fails = 0;
	for (const char* schema : kSchemas) {
		int32_t pending = -1, last = -1;
		SslmGpuSequenceHandle* s = e.Bound(schema, 51u, &last);
		if (!s) { std::printf("FAIL record %s: setup\n", schema); ++fails; continue; }
		bool jumped = false;
		for (int d = 0; d < 40 && !jumped; ++d) {
			const Jump j = DoJump(e, s, pending, 4, kNull);
			if (j.st != S::SSLM_OK) break;
			AfterJump(e, s, j, &pending, &last);
			if (j.count > 0) { jumped = true; break; }
			int32_t t = -1;
			if (e.Decode(s, &pending, &last, &t) != S::SSLM_OK || t < 0) break;
		}
		const std::vector<uint8_t> blob = e.Save(s);
		if (!jumped || blob.empty()) { std::printf("FAIL record %s: jumped %d blob %zu\n", schema, jumped, blob.size()); ++fails; sslm_gpu_seq_release(e.ctx, s); continue; }
		std::string next = F("%d", last);
		int32_t p2 = pending, l2 = last;
		for (int k = 0; k < kNext; ++k) {
			int32_t t = -1;
			const S st = e.Decode(s, &p2, &l2, &t);
			next += F(" %d", st == S::SSLM_OK ? t : -100 - static_cast<int>(st));
		}
		next += F(" pending %d\n", pending);
		if (!WriteAll(g_record + "/" + schema + ".blob", blob.data(), blob.size()) ||
		    !WriteAll(g_record + "/" + schema + ".next", next.data(), next.size())) {
			std::printf("FAIL record %s: write\n", schema);
			++fails;
		} else {
			std::printf("recorded %s: %zu bytes, next %s", schema, blob.size(), next.c_str());
		}
		sslm_gpu_seq_release(e.ctx, s);
	}
	return fails ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ g.4
struct Recorded {
	std::vector<uint8_t> blob;
	int32_t last = -1, pending = -1;
	std::vector<int32_t> next;
	bool Load(const std::string& base) {
		std::vector<uint8_t> txt;
		if (!ReadAll(base + ".blob", &blob) || !ReadAll(base + ".next", &txt)) return false;
		txt.push_back(0);
		const char* p = reinterpret_cast<const char*>(txt.data());
		int n = 0, v[kNext + 1] = {};
		for (int i = 0; i <= kNext; ++i) {
			int used = 0;
			if (std::sscanf(p, "%d%n", &v[i], &used) != 1) return false;
			p += used;
			++n;
		}
		if (std::sscanf(p, " pending %d", &pending) != 1) return false;
		last = v[0];
		next.assign(v + 1, v + n);
		return true;
	}
};
void CellG4(Engine& e, bool continue_part) {
	int found = 0;
	for (const char* schema : kSchemas) {
		const std::string tag = F("1.17 %s", schema);
		Recorded r;
		if (!r.Load(g_legacy + "/" + schema)) { Check(false, tag + ": no recorded blob under " + g_legacy); continue; }
		++found;
		if (!continue_part) {
			SslmGpuSequenceHandle* s = nullptr;
			const S st = sslm_gpu_seq_restore(e.ctx, e.model, r.blob.data(), r.blob.size(), &s);
			Check(st == S::SSLM_OK && s, F("%s: sslm_gpu_seq_restore returned %u", tag.c_str(), (unsigned)st));
			if (!s) continue;
			Check(e.Save(s) == r.blob, tag + ": the restored handle saves different bytes");
			int32_t pending = r.pending, last = r.last;
			std::vector<int32_t> got;
			for (int k = 0; k < kNext; ++k) {
				int32_t t = -1;
				const S d = e.Decode(s, &pending, &last, &t);
				got.push_back(d == S::SSLM_OK ? t : -100 - static_cast<int>(d));
			}
			Check(got == r.next, F("%s: the decode-only continuation differs from 1.17's own (first %d vs %d)", tag.c_str(), got[0], r.next[0]));
			sslm_gpu_seq_release(e.ctx, s);
			continue;
		}
		// g.4b: restore, then a jump under the mode asked for (explicit Exact, then NULL), then decodes to kNext tokens.
		std::vector<int32_t> by_mode[2];
		std::vector<uint8_t> blob_mode[2];
		for (int k = 0; k < 2; ++k) {
			SslmGpuSequenceHandle* s = nullptr;
			if (sslm_gpu_seq_restore(e.ctx, e.model, r.blob.data(), r.blob.size(), &s) != S::SSLM_OK || !s) { Check(false, tag + ": restore"); continue; }
			int32_t pending = r.pending, last = r.last;
			const Jump j = DoJump(e, s, pending, 8, k ? kNull : kModeExact);
			Check(j.st == S::SSLM_OK, F("%s: the %s jump after the restore returned %u", tag.c_str(), k ? "NULL" : "Exact", (unsigned)j.st));
			std::vector<int32_t>& got = by_mode[k];
			if (j.st == S::SSLM_OK) {
				got.assign(j.out.begin(), j.out.begin() + j.count);
				AfterJump(e, s, j, &pending, &last);
			}
			while (static_cast<int>(got.size()) < kNext) {
				int32_t t = -1;
				const S d = e.Decode(s, &pending, &last, &t);
				got.push_back(d == S::SSLM_OK ? t : -100 - static_cast<int>(d));
				if (d != S::SSLM_OK) break;
			}
			got.resize(kNext, -1);
			blob_mode[k] = e.Save(s);
			sslm_gpu_seq_release(e.ctx, s);
		}
		Check(by_mode[0] == r.next, F("%s: explicit Exact jump + decodes differs from 1.17's decode-only stream", tag.c_str()));
		Check(by_mode[1] == by_mode[0] && blob_mode[1] == blob_mode[0], tag + ": NULL options after the restore differ from explicit Exact");
	}
	Check(found == 3, F("%d of 3 recorded blobs found under %s", found, g_legacy.c_str()));
}
void CellG4a(Engine& e) { CellG4(e, false); }
void CellG4b(Engine& e) { CellG4(e, true); }
}  // namespace

int main(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--fixtures=", 0) == 0) g_fixtures = a.substr(11);
		else if (a.rfind("--legacy=", 0) == 0) g_legacy = a.substr(9);
		else if (a.rfind("--record=", 0) == 0) g_record = a.substr(9);
		else if (a.rfind("--cells=", 0) == 0) {
			std::string l = a.substr(8);
			for (size_t p = 0; p <= l.size();) {
				const size_t q = l.find(',', p);
				g_cells.insert(l.substr(p, q == std::string::npos ? std::string::npos : q - p));
				if (q == std::string::npos) break;
				p = q + 1;
			}
		}
	}
	Engine e;
	if (!e.ok) return 2;
	if (!g_record.empty()) return Record(e);
#ifdef MRX_V117
	std::printf("FAIL the 1.17 build is a recorder only (--record=DIR)\n");
	return 2;
#else
	bool all = true;
	const std::pair<const char*, void (*)(Engine&)> cells[] = {
	    {"g.1", CellG1}, {"g.2", CellG2}, {"g.3", CellG3}, {"g.4a", CellG4a}, {"g.4b", CellG4b}};
	for (auto& c : cells) {
		const std::string name = c.first;
		const std::string group = name.substr(0, 3);
		if (!g_cells.empty() && !g_cells.count(name) && !g_cells.count(group)) continue;
		if (group == "g.4" && g_legacy.empty()) continue;
		g_cell = name;
		c.second(e);
		const int f = g_fail[name], n = g_checks[name];
		if (f) std::printf("RED cell %s: %d failures of %d checks\n", c.first, f, n);
		else std::printf("    cell %s: PASS (%d checks)\n", c.first, n);
		all = all && f == 0;
	}
	std::printf("cells: %s\n", all ? "PASS" : "RED");
	return all ? 0 : 1;
#endif
}
