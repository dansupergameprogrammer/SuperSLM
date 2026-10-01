// Paged-KV plan (rev 16.1) step C1: the red suite's shared harness.
//
// Every paged-KV cell includes this file. It carries:
//   - the cell registry (PKV_CELL), the CHECK macros and the runner (pkv_main.cpp);
//   - fixture loading from SUPERSLM_PAGED_KV_FIXTURE_DIR (tools/gen_paged_kv_fixture.py's output);
//     a missing directory or file fails the cell, never skips it;
//   - the byte-equality oracle's reference (tests/paged-kv/reference, step R0): the .ref records,
//     the pinned blobs, and the scenario scripts run on the build under test;
//   - blob readers for 1.9.0's SSB5 and the plan's SSB6 (§3.7), with the row extractor written
//     here from the layout formulas, never by calling the engine;
//   - the test-side instruments of §8: the two-sided fill probe, its one-state form, and the
//     legacy-create admission count. Each takes no reading from the code under test's own stats.
//
// Helpers that call a verb a later step defines are inline and emitted only where a cell calls
// them, so each per-step executable links against exactly the symbols its own cells use.

#ifndef SUPERSLM_TESTS_PKV_COMMON_H
#define SUPERSLM_TESTS_PKV_COMMON_H

#include "pkv_abi.h"
#include "reference/pkv_scenarios.h"
#include "reference/pkv_zrl.h"

#include "superslm/sslm_abi.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

#ifdef SSLM_HAS_PAGED_KV_ABI
static_assert(SSLM_KV_BUDGET_EXCEEDED == PKV_KV_BUDGET_EXCEEDED, "§3.6: SSLM_KV_BUDGET_EXCEEDED is ordinal 29");
#endif

namespace pkv {

// ---- the registry and the checks -------------------------------------------------------------

struct CellEntry {
	const char* id;    // the §7 cell id, with its part when split ("1.13/C4")
	const char* step;  // the owning step: C2, C3, C4, C5 or C6
	void (*fn)();
};

inline std::vector<CellEntry>& Registry() {
	static std::vector<CellEntry> r;
	return r;
}

struct Registrar {
	Registrar(const char* id, const char* step, void (*fn)()) { Registry().push_back({id, step, fn}); }
};

#define PKV_CAT2(a, b) a##b
#define PKV_CAT(a, b) PKV_CAT2(a, b)
// PKV_CELL("6.1/C4", "C4", Cell61Legacy) registers a function defined elsewhere in the TU.
#define PKV_CELL(id, step, fn) static ::pkv::Registrar PKV_CAT(pkv_registrar_, __LINE__)(id, step, fn)

struct RunState {
	const char* cell = "(none)";
	int checks = 0;
	int failures = 0;
	int cell_failures = 0;
};

inline RunState& State() {
	static RunState s;
	return s;
}

inline void Fail(const char* file, int line, const char* fmt, ...) {
	RunState& s = State();
	++s.failures;
	++s.cell_failures;
	std::fprintf(stderr, "FAIL [%s] %s:%d: ", s.cell, file, line);
	va_list ap;
	va_start(ap, fmt);
	std::vfprintf(stderr, fmt, ap);
	va_end(ap);
	std::fputc('\n', stderr);
}

#define PKV_CHECK(cond)                                                    \
	do {                                                                   \
		++::pkv::State().checks;                                           \
		if (!(cond)) ::pkv::Fail(__FILE__, __LINE__, "%s", #cond);         \
	} while (0)
#define PKV_CHECK_MSG(cond, ...)                                           \
	do {                                                                   \
		++::pkv::State().checks;                                           \
		if (!(cond)) ::pkv::Fail(__FILE__, __LINE__, __VA_ARGS__);         \
	} while (0)
#define PKV_CHECK_EQ(a, b)                                                                              \
	do {                                                                                                \
		++::pkv::State().checks;                                                                        \
		const long long pkv_a_ = static_cast<long long>(a), pkv_b_ = static_cast<long long>(b);         \
		if (pkv_a_ != pkv_b_) ::pkv::Fail(__FILE__, __LINE__, "%s == %s (%lld vs %lld)", #a, #b, pkv_a_, pkv_b_); \
	} while (0)

// ---- memory ---------------------------------------------------------------------------------

struct AlignedBuf {
	uint8_t* p = nullptr;
	size_t n = 0;
	AlignedBuf() = default;
	explicit AlignedBuf(size_t size, uint8_t fill = 0) { Reset(size, fill); }
	AlignedBuf(const AlignedBuf&) = delete;
	AlignedBuf& operator=(const AlignedBuf&) = delete;
	~AlignedBuf() { Free(); }
	void Reset(size_t size, uint8_t fill = 0) {
		Free();
		n = (size + SSLM_ABI_ALIGNMENT_BYTES - 1) / SSLM_ABI_ALIGNMENT_BYTES * SSLM_ABI_ALIGNMENT_BYTES;
		if (n == 0) n = SSLM_ABI_ALIGNMENT_BYTES;
		p = static_cast<uint8_t*>(::operator new(n, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
		std::memset(p, fill, n);
	}
	void Free() {
		if (p) ::operator delete(p, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		p = nullptr;
		n = 0;
	}
};

// ---- fixtures -------------------------------------------------------------------------------

// The four fixtures of tools/gen_paged_kv_fixture.py, by stem: "pkv_def", "pkv_qk", "pkv_odd",
// "pkv_32k". All have 2 layers, 2 KV heads, vocabulary 256 and kv_block_size 16.
struct Fixture {
	std::string stem;
	bool ok = false;
	AlignedBuf bytes;
	size_t size = 0;
	sslm_model model = nullptr;
	Geometry geo{};
	std::string sha;  // the file's SHA-256, matched against each .ref header
	int32_t vocab = 256;
	int64_t B() const { return geo.context_cap % 16 == 0 ? 16 : geo.context_cap; }  // §3.1, written here
	int64_t CapPages() const { return (geo.context_cap + B() - 1) / B(); }
	// §3.4's R(budget), written here: min(ceil(budget / B) + 1, ceil(cap / B)).
	int64_t R(int64_t budget) const {
		const int64_t r = (budget + B() - 1) / B() + 1;
		return r < CapPages() ? r : CapPages();
	}
	// §3.1's page_bytes, written here: L * 2 * H_kv * B * D (int8 K/V).
	size_t PageBytes() const { return size_t{geo.layers} * 2u * geo.kv_heads * static_cast<size_t>(B()) * geo.head_dim; }
	size_t BytesPerToken() const { return size_t{geo.layers} * 2u * geo.kv_heads * geo.head_dim; }
	~Fixture() {
		if (model) sslm_model_unmap(model);
	}
};

inline bool GeometryForStem(const std::string& stem, Geometry* g) {
	if (stem == "pkv_def") *g = {2, 2, 48, 4096};
	else if (stem == "pkv_qk") *g = {2, 2, 128, 4096};
	else if (stem == "pkv_odd") *g = {2, 2, 48, 4100};
	else if (stem == "pkv_32k") *g = {2, 2, 48, 32768};
	else return false;
	return true;
}

inline bool ReadAll(const std::string& path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	out->assign(std::istreambuf_iterator<char>(f), {});
	return true;
}

// Loads (once) a fixture. A missing directory variable or file is a failed check.
inline const Fixture& GetFixture(const char* stem) {
	static std::map<std::string, std::unique_ptr<Fixture>> cache;
	auto it = cache.find(stem);
	if (it != cache.end()) {
		PKV_CHECK_MSG(it->second->ok, "fixture %s did not load (see its first failure)", stem);
		return *it->second;
	}
	auto f = std::make_unique<Fixture>();
	f->stem = stem;
	const char* dir = std::getenv("SUPERSLM_PAGED_KV_FIXTURE_DIR");
	PKV_CHECK_MSG(dir && *dir, "SUPERSLM_PAGED_KV_FIXTURE_DIR is not set; run tools/gen_paged_kv_fixture.py DIR");
	PKV_CHECK_MSG(GeometryForStem(stem, &f->geo), "unknown fixture %s", stem);
	std::vector<uint8_t> raw;
	if (dir && *dir && ReadAll(std::string(dir) + "/" + stem + ".sslm", &raw) && !raw.empty()) {
		f->size = raw.size();
		f->bytes.Reset(raw.size());
		std::memcpy(f->bytes.p, raw.data(), raw.size());
		f->sha = Sha(raw.data(), raw.size());
		const sslm_status st = sslm_model_map(f->bytes.p, f->size, &f->model);
		PKV_CHECK_MSG(st == SSLM_OK, "sslm_model_map(%s) == %d", stem, static_cast<int>(st));
		f->ok = st == SSLM_OK;
	} else {
		PKV_CHECK_MSG(false, "fixture %s/%s.sslm not found or empty", dir ? dir : "(unset)", stem);
	}
	const Fixture& ref = *f;
	cache[stem] = std::move(f);
	return ref;
}

// ---- the reference (R0) ---------------------------------------------------------------------

#ifndef PKV_REFERENCE_DIR
#error "PKV_REFERENCE_DIR must name tests/paged-kv/reference (the CMake target defines it)"
#endif

// One parsed .ref file: `<tag>_<stem>.ref`, keyed "<scenario> <stage>".
struct RefRecord {
	int64_t context_length = -1;
	int64_t saturation = -1;
	std::string blob_sha, rows_sha;
	std::vector<int32_t> tokens;
};
struct RefFile {
	bool ok = false;
	std::string fixture_sha;
	std::map<std::string, RefRecord> records;
};

inline const RefFile& Reference(const char* tag, const Fixture& fx) {
	static std::map<std::string, std::unique_ptr<RefFile>> cache;
	const std::string key = std::string(tag) + "_" + fx.stem;
	auto it = cache.find(key);
	if (it != cache.end()) return *it->second;
	auto r = std::make_unique<RefFile>();
	std::ifstream in(std::string(PKV_REFERENCE_DIR) + "/" + key + ".ref");
	PKV_CHECK_MSG(static_cast<bool>(in), "reference %s.ref missing", key.c_str());
	std::string line;
	while (in && std::getline(in, line)) {
		if (line.rfind("# ", 0) == 0) {
			const size_t at = line.find("sha256=");
			if (at != std::string::npos) r->fixture_sha = line.substr(at + 7);
			continue;
		}
		// <scenario> <stage> L=<n> sat=<n> blob=<..> rows=<..> tokens=<..>
		char sc[64], st[96], blob[80], rows[80];
		long long L = 0, sat = 0;
		const size_t tpos = line.find(" tokens=");
		if (tpos == std::string::npos) continue;
		if (std::sscanf(line.c_str(), "%63s %95s L=%lld sat=%lld blob=%79s rows=%79s", sc, st, &L, &sat, blob, rows) != 6)
			continue;
		RefRecord rec;
		rec.context_length = L;
		rec.saturation = sat;
		rec.blob_sha = blob;
		rec.rows_sha = rows;
		const std::string toks = line.substr(tpos + 8);
		size_t p = 0;
		while (p < toks.size()) {
			const size_t c = toks.find(',', p);
			rec.tokens.push_back(std::atoi(toks.substr(p, c == std::string::npos ? std::string::npos : c - p).c_str()));
			if (c == std::string::npos) break;
			p = c + 1;
		}
		r->records[std::string(sc) + " " + st] = rec;
	}
	r->ok = !r->records.empty();
	PKV_CHECK_MSG(r->ok, "reference %s.ref is empty", key.c_str());
	// pkv_qk's calibration follows numpy's float64 exp (gen_paged_kv_fixture.py), so its reference
	// holds only for the fixture it was recorded against; a mismatch is reported, never ignored.
	PKV_CHECK_MSG(!fx.ok || r->fixture_sha == fx.sha,
	              "reference %s.ref was recorded against fixture %s, this host generated %s; rerun "
	              "tools/build_paged_kv_reference.sh on this host",
	              key.c_str(), r->fixture_sha.c_str(), fx.sha.c_str());
	const RefFile& out = *r;
	cache[key] = std::move(r);
	return out;
}

// The record for "<scenario> <stage>", or a failed check.
inline const RefRecord* RefLookup(const RefFile& ref, const std::string& scenario, const std::string& stage) {
	auto it = ref.records.find(scenario + " " + stage);
	PKV_CHECK_MSG(it != ref.records.end(), "reference has no record '%s %s'", scenario.c_str(), stage.c_str());
	return it == ref.records.end() ? nullptr : &it->second;
}

// A pinned blob: reference/pins/<tag>_pkv_def_<scenario>_<stage>.zrl, decoded.
inline std::vector<uint8_t> Pin(const char* tag, const char* scenario, const char* stage) {
	std::vector<uint8_t> enc, out;
	const std::string path = std::string(PKV_REFERENCE_DIR) + "/pins/" + tag + "_pkv_def_" + scenario + "_" + stage + ".zrl";
	const bool ok = ReadAll(path, &enc) && ZrlDecode(enc, &out);
	PKV_CHECK_MSG(ok, "pinned blob %s missing or malformed", path.c_str());
	return out;
}

// Runs one of pkv_scenarios.h's scenarios on the build under test (legacy verbs only) and returns
// its records. A scenario failure is a failed check.
inline std::vector<Record> RunScenario(const Fixture& fx, const char* scenario) {
	Driver d;
	d.model = fx.model;
	d.geo = fx.geo;
	size_t n = 0;
	const Scenario* sc = Scenarios(&n);
	for (size_t i = 0; i < n; ++i)
		if (std::strcmp(sc[i].name, scenario) == 0) sc[i].run(d, nullptr);
	PKV_CHECK_MSG(d.fail.empty(), "scenario %s on %s: %s", scenario, fx.stem.c_str(), d.fail.c_str());
	return d.records;
}

enum class Compare { kTokensRowsAndBlob, kTokensAndRows };

// Compares a scenario run against the reference, record for record. kTokensAndRows is for a run
// whose blob format differs from the reference's (an SSB6 save against 1.9.0's SSB5).
inline void ExpectMatchesReference(const char* tag, const Fixture& fx, const char* scenario,
                                   const std::vector<Record>& got, Compare mode) {
	const RefFile& ref = Reference(tag, fx);
	size_t expected = 0;
	for (const auto& kv : ref.records)
		if (kv.first.rfind(std::string(scenario) + " ", 0) == 0) ++expected;
	PKV_CHECK_MSG(got.size() == expected, "%s on %s: %zu records, reference has %zu", scenario, fx.stem.c_str(),
	              got.size(), expected);
	for (const Record& r : got) {
		const RefRecord* e = RefLookup(ref, scenario, r.stage);
		if (!e) continue;
		PKV_CHECK_MSG(r.tokens == e->tokens, "%s/%s on %s: tokens differ", scenario, r.stage.c_str(), fx.stem.c_str());
		PKV_CHECK_MSG(r.context_length == e->context_length, "%s/%s: L %lld vs %lld", scenario, r.stage.c_str(),
		              static_cast<long long>(r.context_length), static_cast<long long>(e->context_length));
		PKV_CHECK_MSG(r.rows_sha == e->rows_sha, "%s/%s on %s: K/V rows differ", scenario, r.stage.c_str(),
		              fx.stem.c_str());
		if (mode == Compare::kTokensRowsAndBlob)
			PKV_CHECK_MSG(r.blob_sha == e->blob_sha, "%s/%s on %s: blob differs", scenario, r.stage.c_str(),
			              fx.stem.c_str());
	}
}

// ---- blobs (§3.7) ---------------------------------------------------------------------------
//
// 1.9.0's SSB5 (R0 note r0-reanchor-v1.11.0.md): a 156-byte fixed header -- magic 0, model_hash 4,
// kv_precision 36, schema_name_hash 40, dfa_walk_state 48, adapter_binding_id 52, context_length
// 60, layer_index 68, current_token 72, hidden_scale 76 (m) and 84 (e), kv_saturation_count 92,
// forced_token_count 100, damped_greedy_order 108, history_count 112, ready_for_logits 120, the four
// per-site counts 124/132/140/148 -- then the residual (hidden_size bytes), the history (4 bytes
// each), kv_block_count (u32, always 1) and the whole block.
// SSB6: SSB5's 156 bytes, then kv_mode u32 at 156, budget i32 at 160, origin i64 at 164 (172 in
// all); the residual; the history; kv_positions u64 = L'; the rows of [0, L') in canonical layout
// (the flat layout with context_cap replaced by L').

constexpr size_t kSsb5Header = 156;
constexpr size_t kSsb6Header = 172;

inline uint64_t Le64(const std::vector<uint8_t>& b, size_t at) {
	uint64_t v = 0;
	if (at + 8 <= b.size()) std::memcpy(&v, b.data() + at, 8);
	return v;
}
inline uint32_t Le32(const std::vector<uint8_t>& b, size_t at) {
	uint32_t v = 0;
	if (at + 4 <= b.size()) std::memcpy(&v, b.data() + at, 4);
	return v;
}
inline void PutLe64(std::vector<uint8_t>& b, size_t at, uint64_t v) { std::memcpy(b.data() + at, &v, 8); }
inline void PutLe32(std::vector<uint8_t>& b, size_t at, uint32_t v) { std::memcpy(b.data() + at, &v, 4); }

inline bool IsMagic(const std::vector<uint8_t>& b, const char* m) { return b.size() >= 4 && std::memcmp(b.data(), m, 4) == 0; }

struct BlobView {
	bool ok = false;
	char magic[5] = {0};
	int64_t context_length = 0;
	uint32_t layer_index = 0;
	uint64_t history_count = 0;
	uint32_t kv_mode = 0;   // SSB6 only
	int32_t budget = 0;     // SSB6 only
	int64_t origin = 0;     // SSB6 only
	uint64_t kv_positions = 0;  // SSB6: the stored L'; 0 for SSB5 (its block spans the whole cap)
	size_t kv_offset = 0;   // where the rows (SSB6) or the block (SSB5) start
};

inline BlobView ParseBlobWithHidden(const std::vector<uint8_t>& b, size_t hidden_size) {
	BlobView v;
	if (b.size() < kSsb5Header) return v;
	std::memcpy(v.magic, b.data(), 4);
	v.context_length = static_cast<int64_t>(Le64(b, 60));
	v.layer_index = Le32(b, 68);
	v.history_count = Le64(b, 112);
	if (IsMagic(b, "SSB6")) {
		if (b.size() < kSsb6Header) return v;
		v.kv_mode = Le32(b, 156);
		v.budget = static_cast<int32_t>(Le32(b, 160));
		v.origin = static_cast<int64_t>(Le64(b, 164));
		const size_t at = kSsb6Header + hidden_size + 4 * static_cast<size_t>(v.history_count);
		if (at + 8 > b.size()) return v;
		v.kv_positions = Le64(b, at);
		v.kv_offset = at + 8;
	} else if (IsMagic(b, "SSB5")) {
		const size_t at = kSsb5Header + hidden_size + 4 * static_cast<size_t>(v.history_count);
		if (at + 4 > b.size()) return v;
		v.kv_offset = at + 4;
	} else {
		return v;
	}
	v.ok = true;
	return v;
}

// The rows of positions [0, L) in pkv_scenarios.h's position-major order, from either format, so
// a digest compares with the reference's rows_sha whatever the save's magic.
inline bool BlobRows(const std::vector<uint8_t>& b, const Fixture& fx, size_t hidden_size, int64_t L,
                     std::vector<uint8_t>* rows) {
	const BlobView v = ParseBlobWithHidden(b, hidden_size);
	if (!v.ok) return false;
	if (IsMagic(b, "SSB6")) {
		Geometry g = fx.geo;
		g.context_cap = static_cast<int64_t>(v.kv_positions);  // canonical layout: cap -> L'
		if (L > g.context_cap || v.kv_offset + v.kv_positions * fx.BytesPerToken() > b.size()) return false;
		*rows = RowsFromBlock(b.data() + v.kv_offset, g, L);
		return true;
	}
	const size_t block = sslm_kv_block_size(fx.model);
	if (v.kv_offset + block > b.size()) return false;
	*rows = RowsFromBlock(b.data() + v.kv_offset, fx.geo, L);
	return true;
}

// The residual's length: every pkv fixture has hidden_size 192 (gen_decode_threading_fixture.config).
// Real artifacts differ; box cells that parse past the fixed header must use the artifact's own.
inline size_t HiddenSize(const Fixture&) { return 192; }

inline bool SaveBlob(sslm_seq s, std::vector<uint8_t>* out) {
	size_t need = 0;
	if (sslm_seq_save(s, nullptr, &need) != SSLM_BUFFER_TOO_SMALL || need == 0) return false;
	out->assign(need, 0);
	size_t wrote = need;
	if (sslm_seq_save(s, out->data(), &wrote) != SSLM_OK) return false;
	out->resize(wrote);
	return true;
}

// A hand-built SSB6 whole_reserve blob from a legacy holder's SSB5 (no library writer emits one,
// rev 16): SSB5's header, then kv_mode 0, budget = cap, `origin`, the residual and history verbatim,
// kv_positions = L' and the canonical rows [0, L'). The fields are then the cell's to edit.
inline std::vector<uint8_t> Ssb6WholeReserveFromSsb5(const std::vector<uint8_t>& ssb5, const Fixture& fx,
                                                     int64_t origin) {
	const size_t hidden = HiddenSize(fx);
	const BlobView v = ParseBlobWithHidden(ssb5, hidden);
	std::vector<uint8_t> out;
	if (!v.ok || !IsMagic(ssb5, "SSB5")) return out;
	const int64_t Lp = v.context_length + (v.layer_index > 0 ? 1 : 0);
	out.assign(ssb5.begin(), ssb5.begin() + kSsb5Header);
	std::memcpy(out.data(), "SSB6", 4);
	out.resize(kSsb6Header);
	PutLe32(out, 156, 0);
	PutLe32(out, 160, static_cast<uint32_t>(fx.geo.context_cap));
	PutLe64(out, 164, static_cast<uint64_t>(origin));
	const size_t tail = hidden + 4 * static_cast<size_t>(v.history_count);
	out.insert(out.end(), ssb5.begin() + kSsb5Header, ssb5.begin() + kSsb5Header + static_cast<std::ptrdiff_t>(tail));
	out.resize(out.size() + 8);
	PutLe64(out, out.size() - 8, static_cast<uint64_t>(Lp));
	// canonical layout [layer][K|V][head][pos < L'][d] from the block's [layer][K|V][head][pos < cap][d]
	const uint8_t* block = ssb5.data() + v.kv_offset;
	const size_t D = fx.geo.head_dim, cap = static_cast<size_t>(fx.geo.context_cap);
	for (uint32_t l = 0; l < fx.geo.layers; ++l)
		for (uint32_t half = 0; half < 2; ++half)
			for (uint32_t h = 0; h < fx.geo.kv_heads; ++h) {
				const uint8_t* src = block + ((size_t{l} * 2 + half) * fx.geo.kv_heads + h) * cap * D;
				out.insert(out.end(), src, src + static_cast<size_t>(Lp) * D);
			}
	return out;
}

// ---- pools and handles ------------------------------------------------------------------------

// A legacy pool of `blocks` blocks (sslm_kv_pool_create), over memory pre-filled with `fill`.
struct LegacyPool {
	AlignedBuf mem;
	sslm_kv_pool pool = nullptr;
	sslm_status status = SSLM_INVALID_ARGUMENT;
	LegacyPool(sslm_model model, uint32_t blocks, uint8_t fill = 0) {
		const size_t size = sslm_kv_block_size(model) * blocks + sslm_kv_pool_overhead_size(model, blocks);
		mem.Reset(size, fill);
		status = sslm_kv_pool_create(model, mem.p, mem.n, blocks, &pool);
	}
	~LegacyPool() {
		if (pool) sslm_kv_pool_destroy(pool);
	}
};

// A page pool of `pages` pages (sslm_kv_page_pool_create, C5), over memory pre-filled with `fill`.
struct PagePool {
	AlignedBuf mem;
	sslm_kv_pool pool = nullptr;
	sslm_status status = SSLM_INVALID_ARGUMENT;
	PagePool(sslm_model model, uint32_t pages, uint8_t fill = 0) {
		const size_t size = sslm_kv_page_size(model) * pages + sslm_kv_page_pool_overhead_size(model, pages);
		mem.Reset(size, fill);
		status = sslm_kv_page_pool_create(model, mem.p, mem.n, pages, &pool);
	}
	~PagePool() {
		if (pool) sslm_kv_pool_destroy(pool);
	}
};

// Owning handle lists, released in reverse order (sequences first, then prefixes).
struct Handles {
	std::vector<sslm_seq> seqs;
	std::vector<sslm_prefix> prefixes;
	void ReleaseAll() {
		for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) sslm_seq_release(*it);
		for (auto it = prefixes.rbegin(); it != prefixes.rend(); ++it) sslm_prefix_release(*it);
		seqs.clear();
		prefixes.clear();
	}
	~Handles() { ReleaseAll(); }
};

// ---- the instruments (§8) ------------------------------------------------------------------

// The legacy-create admission count (§8, rev 5): legacy creates admitted until the first refusal,
// all released again before returning. It resolves ceil(cap/B) pages; a cell grading an exact page
// count runs it at two pool sizes (P and P - 1).
inline int CountLegacyCreates(sslm_model model, sslm_kv_pool* pool, sslm_status* refusal = nullptr) {
	std::vector<sslm_seq> made;
	sslm_status st = SSLM_OK;
	for (int guard = 0; guard < 1 << 20; ++guard) {
		sslm_seq s = nullptr;
		st = sslm_seq_create(model, pool, &s);
		if (st != SSLM_OK) break;
		made.push_back(s);
	}
	if (refusal) *refusal = st;
	for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_seq_release(*it);
	return static_cast<int>(made.size());
}

// The fill probe's create sizes (§8): creates of 2 to ceil(cap/B) pages, totalling `k` (k >= 2).
// A create of budget (r - 1) * B reserves exactly r pages, by §3.4's formula written here.
inline std::vector<int64_t> ProbeSplit(const Fixture& fx, int64_t k) {
	std::vector<int64_t> parts;
	const int64_t M = fx.CapPages();
	// The probe needs creates of at least 2 pages, so a one-page-per-cap geometry (B = cap) and
	// k < 2 cannot be probed: a failed check, never a loop.
	PKV_CHECK_MSG(M >= 2 && k >= 2, "fill probe: k = %lld on a %lld-page cap cannot be probed",
	              static_cast<long long>(k), static_cast<long long>(M));
	if (M < 2 || k < 2) return parts;
	while (k > 0) {
		int64_t r = k < M ? k : M;
		if (k - r == 1) r -= 1;  // never leave a 1-page remainder
		parts.push_back(r);
		k -= r;
	}
	return parts;
}

// Budgeted creates totalling `k` pages (C5). Returns true when every create is admitted; the
// admitted sequences are appended to `out` (the caller releases them).
inline bool AdmitPages(const Fixture& fx, sslm_kv_pool* pool, int64_t k, std::vector<sslm_seq>* out) {
	bool all = true;
	for (int64_t r : ProbeSplit(fx, k)) {
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_create_budgeted(fx.model, pool, static_cast<int32_t>((r - 1) * fx.B()), &s);
		if (st == SSLM_OK) out->push_back(s);
		else all = false;
	}
	return all;
}

// A state the two-sided probe can rebuild: the builder makes a fresh pool and state, and returns an
// object that keeps them alive (pool, handles) and exposes the pool.
struct ProbeState {
	virtual ~ProbeState() = default;
	virtual sslm_kv_pool* Pool() = 0;
};

// The two-sided fill probe (§8, rev 5): exactly `k` pages are free in the state `build` makes iff
// creates totalling k are all admitted on one build and creates totalling k + 1 are not all
// admitted on a fresh rebuild. k >= 2. Graded by admission only.
inline bool ProbeExactlyFree(const Fixture& fx, const std::function<std::unique_ptr<ProbeState>()>& build, int64_t k) {
	bool admits_k = false, refuses_k1 = false;
	{
		std::unique_ptr<ProbeState> s = build();
		std::vector<sslm_seq> made;
		admits_k = s && AdmitPages(fx, s->Pool(), k, &made);
		for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_seq_release(*it);
	}
	{
		std::unique_ptr<ProbeState> s = build();
		std::vector<sslm_seq> made;
		refuses_k1 = s && !AdmitPages(fx, s->Pool(), k + 1, &made);
		for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_seq_release(*it);
	}
	PKV_CHECK_MSG(admits_k, "fill probe: creates totalling %lld pages were not all admitted", static_cast<long long>(k));
	PKV_CHECK_MSG(refuses_k1, "fill probe: creates totalling %lld pages were all admitted", static_cast<long long>(k + 1));
	return admits_k && refuses_k1;
}

// The one-state form (§8, rev 6), for a state that cannot be rebuilt (a race): on the one live
// pool, for k >= 3: creates totalling k - 1 are admitted; a 2-page create is refused; one admitted
// create of r pages is released and creates totalling r + 1 are admitted. Leaves the pool as found.
inline bool ProbeExactlyFreeOneState(const Fixture& fx, sslm_kv_pool* pool, int64_t k) {
	std::vector<sslm_seq> made;
	const bool step1 = AdmitPages(fx, pool, k - 1, &made);
	sslm_seq extra = nullptr;
	const bool step2 = sslm_seq_create_budgeted(fx.model, pool, static_cast<int32_t>(fx.B()), &extra) != SSLM_OK;
	if (extra) made.push_back(extra);
	bool step3 = false;
	if (step1 && !made.empty()) {
		const int64_t r = ProbeSplit(fx, k - 1).back();
		sslm_seq_release(made.back());
		made.pop_back();
		step3 = AdmitPages(fx, pool, r + 1, &made);
	}
	for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_seq_release(*it);
	PKV_CHECK_MSG(step1 && step2 && step3, "one-state fill probe at k=%lld: admit k-1 %d, refuse +2 %d, re-admit r+1 %d",
	              static_cast<long long>(k), step1, step2, step3);
	return step1 && step2 && step3;
}

// ---- decode helpers --------------------------------------------------------------------------

// The next greedy token, one layer per call; -1 when the step reports a status (returned in *st).
inline int32_t NextToken(sslm_model model, sslm_seq seq, sslm_status* st = nullptr) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {seq};
	int32_t tok = -1;
	sslm_status s = SSLM_OK;
	for (int guard = 0; tok < 0 && guard < 4096; ++guard) {
		s = sslm_decode_step(model, b, 1, &p, nullptr, &tok);
		if (s != SSLM_OK) break;
	}
	if (st) *st = s;
	return s == SSLM_OK ? tok : -1;
}

// Prefill all of `t` in chunks of `chunk`; returns the first non-OK status (and how many were
// consumed in *consumed_total).
inline sslm_status PrefillAll(sslm_model model, sslm_seq s, const std::vector<int32_t>& t, int32_t chunk,
                              int64_t* consumed_total = nullptr) {
	size_t at = 0;
	sslm_status st = SSLM_OK;
	while (at < t.size()) {
		int32_t consumed = 0;
		st = sslm_prefill(model, s, t.data() + at, static_cast<int32_t>(t.size() - at), chunk, SSLM_SPAN_PROMPT,
		                  nullptr, &consumed);
		at += static_cast<size_t>(consumed > 0 ? consumed : 0);
		if (st != SSLM_OK || consumed <= 0) break;
	}
	if (consumed_total) *consumed_total = static_cast<int64_t>(at);
	return st;
}

inline sslm_status PrefixPrefillAll(sslm_model model, sslm_prefix p, const std::vector<int32_t>& t, int32_t chunk) {
	size_t at = 0;
	sslm_status st = SSLM_OK;
	while (at < t.size()) {
		int32_t consumed = 0;
		st = sslm_prefix_prefill(model, p, t.data() + at, static_cast<int32_t>(t.size() - at), chunk, SSLM_SPAN_PROMPT,
		                         nullptr, &consumed);
		at += static_cast<size_t>(consumed > 0 ? consumed : 0);
		if (st != SSLM_OK || consumed <= 0) break;
	}
	return st;
}

}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_COMMON_H
