// Multi-row GPU plan (rev 1.3.2), slice MRX (plan §16 erratum E-11, ruling 20): Canonical removal, the CPU cells.
// Native Linux (no Wine): mrx_cpu.sh builds the CPU library from a tree's CMakeLists core sources and links this file.
// The same file is the RECORDER for 1.15-1.17's saves: built against a release tag's tree, `--record=DIR` writes blobs
// made under that release's Canonical jump (opts NULL), with the decode-only stream that follows each.
//
// Cells (on F-SFD, tools/gen_schema_fast_decode_fixture.py's fsfd.sslm):
//   x.1  The old Canonical value (mode 0) is refused by sslm_seq_jump_forward through the malformed-options status
//        (SSLM_INVALID_ARGUMENT) with nothing changed: the count unwritten, out_forced untouched, the save blob
//        equal, and a fresh bound sequence still bindable. Controls: mode 2 and -1 stay refused.
//   x.3  NULL options equal explicit Exact (mode 1, no draft) byte for byte: status, out_forced, count, the save blob,
//        then six decode steps (tokens and blob). Three schemas, budgets 1, 4 and 8.
//   x.5  Saves made by 1.15, 1.16 and 1.17 under Canonical restore: SSLM_OK, and the resumed sequence continues under
//        the mode the caller asks for. Decode-only, it gives the recorder's own next 8 tokens. One decode then an Exact
//        jump (explicit mode 1), then decodes, gives the same 8. NULL options give the same as explicit Exact.
//   (x.2 header and x.6 CHANGELOG are mrx_cpu.sh's static legs.)
//
// The literal mode values are used, never SSLM_JUMP_CANONICAL: the name leaves the headers at MRX, and this file
// still has to compile there and in the 1.15-1.17 trees.
//
// Usage: mrx_cpu --fixtures=DIR [--legacy=DIR] [--cells=x.1,x.3,x.5]   |   mrx_cpu --fixtures=DIR --record=DIR
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "superslm/sslm_abi.h"

namespace {
constexpr int32_t kModeCanonicalOld = 0;  // SSLM_JUMP_CANONICAL before MRX; reserved, never reused (E-11 (a))
constexpr int32_t kModeExact = 1;         // SSLM_JUMP_EXACT

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
std::vector<uint8_t> ReadFile(const std::string& p) {
	std::ifstream in(p, std::ios::binary);
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}
bool WriteFile(const std::string& p, const void* d, size_t n) {
	std::FILE* f = std::fopen(p.c_str(), "wb");
	if (!f) return false;
	const bool ok = std::fwrite(d, 1, n, f) == n;
	std::fclose(f);
	return ok;
}
struct Buf {
	void* p = nullptr;
	explicit Buf(size_t n) { p = std::aligned_alloc(64, (n + 63) / 64 * 64); std::memset(p, 0, (n + 63) / 64 * 64); }
	~Buf() { std::free(p); }
	Buf(const Buf&) = delete;
	Buf& operator=(const Buf&) = delete;
};

const char* kSchemas[] = {"potion_shop_order", "prompt_result", "longkey"};

struct Engine {
	std::vector<uint8_t> file;
	Buf* bytes = nullptr;
	Buf* pool_buf = nullptr;
	Buf* ws_buf = nullptr;
	sslm_model m = nullptr;
	sslm_kv_pool pool = nullptr;
	sslm_workspace ws = nullptr;
	bool ok = false;
	Engine() {
		file = ReadFile(g_fixtures + "/fsfd.sslm");
		if (file.empty()) { std::printf("FAIL cannot read %s/fsfd.sslm\n", g_fixtures.c_str()); return; }
		bytes = new Buf(file.size());
		std::memcpy(bytes->p, file.data(), file.size());
		const sslm_status ms = sslm_model_map(bytes->p, file.size(), &m);
		if (ms != SSLM_OK) { std::printf("FAIL sslm_model_map %d\n", static_cast<int>(ms)); return; }
		const uint32_t blocks = 24;
		const size_t size = blocks * sslm_kv_block_size(m) + sslm_kv_pool_overhead_size(m, blocks);
		pool_buf = new Buf(size);
		if (sslm_kv_pool_create(m, pool_buf->p, size, blocks, &pool) != SSLM_OK) { std::printf("FAIL pool\n"); return; }
		const sslm_config cfg{1, 64, 2, 0u};
		const size_t wsz = sslm_workspace_size(m, &cfg);
		ws_buf = new Buf(wsz);
		if (sslm_workspace_create(m, &cfg, ws_buf->p, wsz, &ws) != SSLM_OK) { std::printf("FAIL workspace\n"); return; }
		ok = true;
	}
	~Engine() {
		if (ws) sslm_workspace_destroy(ws);
		if (pool) sslm_kv_pool_destroy(pool);
		if (m) sslm_model_unmap(m);
		delete ws_buf;
		delete pool_buf;
		delete bytes;
	}
	std::vector<int32_t> Prompt(uint32_t stream) const {
		std::vector<int32_t> v;
		for (int i = 0; i < 10; ++i) v.push_back(static_cast<int32_t>((1103515245u * (i + 7) + 12345u + 7919u * stream) % 20000u + 200u));
		return v;
	}
	sslm_seq Bound(const char* schema, bool prompt, uint32_t stream) {
		sslm_seq s = nullptr;
		if (sslm_seq_create(m, &pool, &s) != SSLM_OK || !s) return nullptr;
		sslm_schema h = nullptr;
		if (sslm_schema_lookup(m, schema, &h) != SSLM_OK || sslm_seq_set_schema(s, h) != SSLM_OK) { sslm_seq_release(s); return nullptr; }
		if (prompt) {
			const std::vector<int32_t> p = Prompt(stream);
			int32_t consumed = 0;
			if (sslm_prefill(m, s, p.data(), int32_t(p.size()), int32_t(p.size()), SSLM_SPAN_PROMPT, ws, &consumed) != SSLM_OK) {
				sslm_seq_release(s);
				return nullptr;
			}
		}
		return s;
	}
	sslm_status Decode(sslm_seq s, int32_t* tok) {
		sslm_decode_params p{};
		sslm_decode_params_init(m, SSLM_DECODE_MODE_GREEDY, 2, &p);
		*tok = -9;
		return sslm_decode_step_v2(m, &s, 1, &p, ws, tok);
	}
	std::vector<uint8_t> Save(sslm_seq s) {
		std::vector<uint8_t> b(sslm_seq_state_size(m) + 65536);
		size_t n = b.size();
		if (sslm_seq_save(s, b.data(), &n) != SSLM_OK) return {};
		b.resize(n);
		return b;
	}
	int32_t Chain(sslm_seq s) {
		int32_t n = 0, acc = 0;
		return sslm_seq_forced_tokens(m, s, nullptr, 0, &n, &acc) == SSLM_OK ? n : -1;
	}
	// A bound sequence after the prompt and `decodes` decode steps (the last produced token pending), or null.
	sslm_seq Pending(const char* schema, uint32_t stream, int decodes) {
		sslm_seq s = Bound(schema, true, stream);
		for (int i = 0; s && i < decodes; ++i) {
			int32_t t = -1;
			if (Decode(s, &t) != SSLM_OK || t < 0) { sslm_seq_release(s); return nullptr; }
		}
		return s;
	}
};

struct Jump {
	sslm_status st = SSLM_OK;
	int32_t count = -7;
	std::vector<int32_t> out;
	bool operator==(const Jump& o) const { return st == o.st && count == o.count && out == o.out; }
};
Jump DoJump(Engine& e, sslm_seq s, int32_t budget, const sslm_jump_options* opts) {
	Jump j;
	j.out.assign(16, -7);
	j.st = sslm_seq_jump_forward(e.m, s, budget, nullptr, opts, e.ws, j.out.data(), 16, &j.count);
	return j;
}
sslm_jump_options Opts(int32_t mode) { return sslm_jump_options{sizeof(sslm_jump_options), mode, nullptr, 0}; }

// ------------------------------------------------------------------------------------------------ x.1
void CellX1(Engine& e) {
	for (const char* schema : kSchemas)
		for (int32_t mode : {kModeCanonicalOld, 2, -1}) {
			const std::string tag = F("%s mode %d", schema, mode);
			sslm_seq s = e.Pending(schema, 3, 1);
			if (!s) { Check(false, tag + ": setup"); continue; }
			const std::vector<uint8_t> before = e.Save(s);
			const sslm_jump_options o = Opts(mode);
			Jump j = DoJump(e, s, 4, &o);
			Check(j.st == SSLM_INVALID_ARGUMENT, F("%s: returned %d, want SSLM_INVALID_ARGUMENT (%d)", tag.c_str(), j.st, SSLM_INVALID_ARGUMENT));
			Check(j.count == -7, F("%s: the count was written (%d)", tag.c_str(), j.count));
			bool untouched = true;
			for (int32_t v : j.out) untouched = untouched && v == -7;
			Check(untouched, tag + ": out_forced was written");
			Check(e.Save(s) == before && !before.empty(), tag + ": the save blob changed");
			sslm_seq_release(s);
			// Bindability: a refused call leaves a fresh bound sequence bindable.
			sslm_seq f = e.Bound(schema, false, 3);
			if (!f) { Check(false, tag + ": fresh"); continue; }
			Jump jf = DoJump(e, f, 4, &o);
			sslm_schema h = nullptr;
			sslm_schema_lookup(e.m, schema, &h);
			Check(jf.st == SSLM_INVALID_ARGUMENT && sslm_seq_set_schema(f, h) == SSLM_OK,
			      F("%s on a fresh sequence: returned %d; the sequence must stay bindable", tag.c_str(), jf.st));
			sslm_seq_release(f);
		}
}

// ------------------------------------------------------------------------------------------------ x.3
void CellX3(Engine& e) {
	int compared = 0;
	for (const char* schema : kSchemas)
		for (int32_t budget : {1, 4, 8})
			for (int decodes = 1; decodes <= 4; ++decodes) {
				const std::string tag = F("%s budget %d after %d decodes", schema, budget, decodes);
				sslm_seq a = e.Pending(schema, 5, decodes), b = e.Pending(schema, 5, decodes);
				if (!a || !b) { Check(false, tag + ": setup"); if (a) sslm_seq_release(a); if (b) sslm_seq_release(b); continue; }
				const bool chain = e.Chain(a) > 0;
				const sslm_jump_options ex = Opts(kModeExact);
				const Jump ja = DoJump(e, a, budget, nullptr), jb = DoJump(e, b, budget, &ex);
				Check(ja == jb, F("%s: NULL options returned %d count %d, explicit Exact %d count %d", tag.c_str(), ja.st, ja.count, jb.st, jb.count));
				Check(e.Save(a) == e.Save(b), tag + ": the save blob after NULL differs from explicit Exact's");
				std::vector<int32_t> ta, tb;
				for (int i = 0; i < 6; ++i) {
					int32_t x = -1, y = -1;
					e.Decode(a, &x), e.Decode(b, &y);
					ta.push_back(x), tb.push_back(y);
				}
				Check(ta == tb && e.Save(a) == e.Save(b), tag + ": the continuation after NULL differs from explicit Exact's");
				compared += chain;
				sslm_seq_release(a);
				sslm_seq_release(b);
			}
	Check(compared >= 6, F("only %d cases had a forced chain at the jump (want 6)", compared));
}

// ------------------------------------------------------------------------------------------------ record / x.5
// The recorder: per schema, the prompt, decodes and Canonical jumps (opts NULL in 1.15-1.17) until a jump admitted
// forced tokens; then the save, and the next 8 decode-only tokens.
int Record(Engine& e) {
	int written = 0;
	for (const char* schema : kSchemas) {
		sslm_seq s = e.Bound(schema, true, 9);
		if (!s) { std::printf("record %s: setup failed\n", schema); continue; }
		bool jumped = false;
		for (int step = 0; step < 24 && !jumped; ++step) {
			int32_t t = -1;
			if (e.Decode(s, &t) != SSLM_OK || t < 0) break;
			const Jump j = DoJump(e, s, 4, nullptr);  // NULL: Canonical on 1.15-1.17
			if (j.st != SSLM_OK) break;
			jumped = j.count > 0;
		}
		if (!jumped) { std::printf("record %s: no canonical jump admitted forced tokens\n", schema); sslm_seq_release(s); continue; }
		const std::vector<uint8_t> blob = e.Save(s);
		std::vector<int32_t> next;
		for (int i = 0; i < 8; ++i) {
			int32_t t = -1;
			e.Decode(s, &t);
			next.push_back(t);
		}
		const std::string base = g_record + "/" + schema;
		if (!blob.empty() && WriteFile(base + ".blob", blob.data(), blob.size()) && WriteFile(base + ".next", next.data(), next.size() * 4)) {
			std::printf("record %s: %zu-byte blob, next %d %d %d ...\n", schema, blob.size(), next[0], next[1], next[2]);
			++written;
		}
		sslm_seq_release(s);
	}
	return written == 3 ? 0 : 1;
}

void CellX5(Engine& e) {
	int found = 0;
	for (const char* release : {"v1.15.0", "v1.16.0", "v1.17.0"})
		for (const char* schema : kSchemas) {
			const std::string base = g_legacy + "/" + release + "/" + schema, tag = F("%s %s", release, schema);
			const std::vector<uint8_t> blob = ReadFile(base + ".blob"), nb = ReadFile(base + ".next");
			if (blob.empty() || nb.size() != 32) { Check(false, tag + ": no recorded blob"); continue; }
			++found;
			std::vector<int32_t> want(8);
			std::memcpy(want.data(), nb.data(), 32);
			auto restore = [&]() {
				sslm_seq s = nullptr;
				const sslm_status st = sslm_seq_restore(e.m, &e.pool, blob.data(), blob.size(), &s);
				Check(st == SSLM_OK && s, F("%s: sslm_seq_restore returned %d", tag.c_str(), st));
				return st == SSLM_OK ? s : nullptr;
			};
			// Decode only.
			if (sslm_seq s = restore()) {
				std::vector<int32_t> got;
				for (int i = 0; i < 8; ++i) { int32_t t = -1; e.Decode(s, &t); got.push_back(t); }
				Check(got == want, F("%s: the decode-only continuation differs from %s's own (first %d vs %d)", tag.c_str(), release, got[0], want[0]));
				sslm_seq_release(s);
			}
			// One decode, then a jump under the mode asked for (explicit Exact; and NULL, which is Exact), then decodes.
			std::vector<int32_t> by_mode[2];
			std::vector<uint8_t> blob_mode[2];
			for (int k = 0; k < 2; ++k) {
				sslm_seq s = restore();
				if (!s) continue;
				int32_t t = -1;
				e.Decode(s, &t);
				by_mode[k].push_back(t);
				const sslm_jump_options ex = Opts(kModeExact);
				const Jump j = DoJump(e, s, 4, k == 0 ? &ex : nullptr);
				Check(j.st == SSLM_OK, F("%s: the %s jump after the restore returned %d", tag.c_str(), k ? "NULL" : "Exact", j.st));
				for (int32_t i = 0; j.st == SSLM_OK && i < j.count && by_mode[k].size() < 8; ++i) by_mode[k].push_back(j.out[i]);
				while (by_mode[k].size() < 8) { e.Decode(s, &t); by_mode[k].push_back(t); }
				blob_mode[k] = e.Save(s);
				sslm_seq_release(s);
			}
			Check(by_mode[0] == want, F("%s: decode + explicit Exact jump + decodes differs from %s's decode-only stream", tag.c_str(), release));
			Check(by_mode[1] == by_mode[0] && blob_mode[1] == blob_mode[0], tag + ": NULL options after the restore differ from explicit Exact");
		}
	Check(found == 9, F("%d of 9 recorded blobs found under %s", found, g_legacy.c_str()));
}
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
	bool all = true;
	const std::pair<const char*, void (*)(Engine&)> cells[] = {{"x.1", CellX1}, {"x.3", CellX3}, {"x.5", CellX5}};
	for (auto& c : cells) {
		if (!g_cells.empty() && !g_cells.count(c.first)) continue;
		if (std::string(c.first) == "x.5" && g_legacy.empty()) continue;
		g_cell = c.first;
		c.second(e);
		const int f = g_fail[c.first], n = g_checks[c.first];
		if (f) std::printf("RED cell %s: %d failures of %d checks\n", c.first, f, n);
		else std::printf("    cell %s: PASS (%d checks)\n", c.first, n);
		all = all && f == 0;
	}
	std::printf("cells: %s\n", all ? "PASS" : "RED");
	return all ? 0 : 1;
}
