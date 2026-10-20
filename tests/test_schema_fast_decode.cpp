// Schema fast-decode plan (rev 1.2.1), slices M1 (masked finish) and J1 (jump-forward): the red
// suite of §11.1 (the engine cells) and §5.9 (the feature oracle, cell 10.1), authored before either
// slice is built (§9). Every cell is tagged with its plan ID and slice; the cell numbers are §11.1's.
// GPU cells (§11.2, G1) and plugin cells (§11.3, P1/P2) are not here. Cell 7 is the coverage
// auditor's re-score of §12 and has no code; 10.3 and 10.4 are box steps (B2, B5).
//
// Own translation unit because it calls the C ABI (test_slm18x_saturation_census.cpp explains why
// test_main.cpp cannot include sslm_abi.h on Windows). test_main.cpp calls RunSchemaFastDecodeCells;
// the scalar-forced standalone (schema_fast_decode_standalone_main.cpp) calls it too, so cells 4.2 and
// 6.6 run on the scalar tier as well as on the SSE2, AVX2 and AVX-512 forced suites.
//
// The fixture is F-SFD (§11.0), from tools/gen_schema_fast_decode_fixture.py, run by every CI leg
// that runs a suite, into the directory SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR names: fsfd.sslm and
// the reference tables fsfd_ref.txt. A missing fixture fails every cell that needs it; nothing here
// skips. F-HAND and the other variants are built in memory from F-SFD's sections (the test_slm19x
// writer pattern).
//
// "Reference" is main's behaviour (the full finish, mapped with S-BOUND = 0; plain prefill and decode)
// or the independent reference rule (fsfd_ref.txt, computed outside the engine from the compiler's
// character DFA and a Python Encode over bytes). It is never the build under test's new path.
//
// Until M1 and J1 land, the surface these cells call is the red stubs of
// tests/schema_fast_decode/sfd_red_stubs.cpp (sfd_api.h states the convention).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "schema_fast_decode/sfd_api.h"
#include "sslm_fixtures.h"
#include "sslm_model_hostile_fixtures.h"
#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"
#include "superslm/parallel_for.h"
#include "superslm/schema_masks.h"
#include "superslm/sha256.h"
#include "superslm/sslm_abi.h"
#include "support/bad_alloc_injection.h"

#if !defined(SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS)
#error "test_schema_fast_decode.cpp needs a library built with SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS"
#endif

#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
extern "C" void ArmCpuFinishDegenerateLogitRowInjection();
#endif

// Atomic: cells 3.2 and 3.3 check from worker threads.
static std::atomic<int> GChecks{0};
static std::atomic<int> GFailures{0};

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

namespace {

namespace sfd = superslm::test::sfd;

constexpr int32_t kV = 151936;        // F-SFD's CFG1 vocab
constexpr int32_t kTokVocab = 151665; // TOK1 entries; ids at and above have no bytes (G12)
constexpr int32_t kCap = 256;         // F-SFD's context cap
constexpr int32_t kHidden = 64;
constexpr int32_t kLayers = 2;
constexpr int32_t kReplacementToken = 5691;  // U+FFFD, EF BF BD (G12)

const char* kPotion = "potion_shop_order";
const char* kPromptResult = "prompt_result";
const char* kLongKey = "longkey";
const char* kUtf8Enum = "utf8enum";

// ---- golden values (cell 6.6) ---------------------------------------------------------------------
// The forced-table digests: SHA-256 over, for every state s in order, LE32 next_forced(s) (-1 when
// unforced) then LE32 |F(s)|. Computed by tools/gen_schema_fast_decode_fixture.py from the reference
// rule (it prints them); the cell checks the engine's table, read through the table seam, against
// these and against the digest line the generator writes into fsfd_ref.txt.
struct GoldenTable {
	const char* schema;
	const char* digest;
};
constexpr GoldenTable kGoldenTables[] = {
    {"potion_shop_order", "7024d46353a94c1421ef9f6b57c162ea04671c58416c05bf52f3eae004bc337c"},
    {"prompt_result", "8fd3adfaf374de598b8bffd76fc74f1f19e746e573eb97006158afabdb69d8fd"},
    {"longkey", "aa2c5c6b6e226720e06bf6d1c13c35dbc912cbd2e40e785e4acfe8c068530b3e"},
    {"utf8enum", "80cc3d912cbfad5daa498608bf488b245a91fef60f44b02db2b10db3dd7414d3"},
};
// The jump-forward stream digests: SHA-256 over the stream's LE32 ids (output tokens only, the
// prompt excluded), greedy, chunk budget 64, from the cell's fixed prompts. Computed on main by the
// reference replay below (ReferenceReplay: a fresh sequence per choice that prefills the prompt and
// then the reference history, then decodes), which is main's prefill and decode only.
struct GoldenStream {
	const char* schema;
	size_t length;
	const char* digest;
};
constexpr GoldenStream kGoldenStreams[] = {
    {"potion_shop_order", 24, "5498cef23900bfbfad790d7174b6badc04bddc58afdca629daf8e2d0b48488bd"},
    {"prompt_result", 42, "16072100850158b6b6314415875d694601b47c80de8335be6d276230c65b08fb"},
};

// ---- small utilities ------------------------------------------------------------------------------

std::string Str(const std::vector<int32_t>& v, size_t limit = 40) {
	std::string s = "{";
	for (size_t i = 0; i < v.size() && i < limit; ++i) s += (i ? "," : "") + std::to_string(v[i]);
	if (v.size() > limit) s += ",...(" + std::to_string(v.size()) + ")";
	return s + "}";
}

std::string HexDigest(const std::vector<int32_t>& ids) {
	superslm::Sha256 h;
	for (int32_t x : ids) {
		uint8_t b[4];
		const uint32_t u = static_cast<uint32_t>(x);
		b[0] = static_cast<uint8_t>(u);
		b[1] = static_cast<uint8_t>(u >> 8);
		b[2] = static_cast<uint8_t>(u >> 16);
		b[3] = static_cast<uint8_t>(u >> 24);
		h.Update(b, 4);
	}
	uint8_t out[32];
	h.Final(out);
	return superslm::ToHex(out);
}

uint32_t Le32(const uint8_t* p) {
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t Le64(const uint8_t* p) { return uint64_t(Le32(p)) | (uint64_t(Le32(p + 4)) << 32); }

struct Aligned {
	uint8_t* p = nullptr;
	size_t n = 0;
	explicit Aligned(size_t bytes) : n(bytes) {
		p = static_cast<uint8_t*>(::operator new(bytes ? bytes : 1, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
	}
	~Aligned() { ::operator delete(p, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)); }
	Aligned(const Aligned&) = delete;
	Aligned& operator=(const Aligned&) = delete;
};

// Runs `fn` on its own thread and waits up to `seconds`. A timeout is a failure (2.1: "a timeout counts
// as red"); the thread is then left detached, so a hung build cannot hang the suite.
bool RunWithTimeout(std::function<void()> fn, int seconds) {
	auto done = std::make_shared<std::promise<void>>();
	std::future<void> f = done->get_future();
	std::thread t([fn, done] {
		fn();
		done->set_value();
	});
	if (f.wait_for(std::chrono::seconds(seconds)) != std::future_status::ready) {
		t.detach();
		return false;
	}
	t.join();
	return true;
}

// ---- artifacts ------------------------------------------------------------------------------------

struct RawSec {
	uint32_t type = 0;
	uint32_t dtype = 0;
	uint32_t alignment = 64;
	std::vector<uint8_t> data;
};

bool ReadSections(const std::vector<uint8_t>& bytes, std::vector<RawSec>* out, uint32_t* flags) {
	if (bytes.size() < superslm::kHeaderBytes) return false;
	const uint32_t count = Le32(bytes.data() + 12);
	*flags = Le32(bytes.data() + 16);
	out->clear();
	for (uint32_t i = 0; i < count; ++i) {
		const size_t row = superslm::kHeaderBytes + size_t{i} * superslm::kSectionDescBytes;
		if (row + superslm::kSectionDescBytes > bytes.size()) return false;
		RawSec s;
		s.type = Le32(bytes.data() + row);
		s.dtype = Le32(bytes.data() + row + 4);
		const uint64_t off = Le64(bytes.data() + row + 8);
		const uint64_t len = Le64(bytes.data() + row + 16);
		s.alignment = Le32(bytes.data() + row + 32);
		if (off + len > bytes.size()) return false;
		s.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(off),
		              bytes.begin() + static_cast<std::ptrdiff_t>(off + len));
		out->push_back(std::move(s));
	}
	return true;
}

std::vector<uint8_t> Rebuild(const std::vector<RawSec>& secs, uint32_t flags) {
	std::vector<superslm_test::FixtureSection> fs;
	for (const RawSec& r : secs) {
		superslm_test::FixtureSection f;
		f.type = r.type;
		f.dtype = r.dtype;
		f.alignment = r.alignment;
		f.data = r.data;
		fs.push_back(std::move(f));
	}
	superslm_test::BuiltArtifact built = superslm_test::BuildArtifact(fs);
	superslm_test::PutU32(built.bytes, 16, flags);
	superslm_test::RecomputeIntegrityHash(built.bytes);
	return built.bytes;
}

constexpr uint32_t kSecTokenizer = static_cast<uint32_t>(superslm::SslmSectionType::Tokenizer);
constexpr uint32_t kSecUnicode = static_cast<uint32_t>(superslm::SslmSectionType::UnicodeTables);
constexpr uint32_t kSecSchemas = static_cast<uint32_t>(superslm::SslmSectionType::SchemaMasks);
constexpr uint32_t kSecDgc1 = static_cast<uint32_t>(superslm::SslmSectionType::DampedGreedyConstants);

// A mapped model with its own aligned copy of the bytes. `bound` set: S-BOUND holds that bound while
// sslm_model_map runs (and is reset after), so the model's exactness flag is computed against it.
constexpr size_t kNoBound = SIZE_MAX;
struct Mapped {
	std::unique_ptr<Aligned> bytes;
	sslm_model m = nullptr;
	sslm_status st = SSLM_OK;
	Mapped() = default;
	~Mapped() {
		if (m) sslm_model_unmap(m);
	}
	Mapped(const Mapped&) = delete;
	Mapped& operator=(const Mapped&) = delete;
};

std::unique_ptr<Mapped> Map(const std::vector<uint8_t>& raw, size_t bound = kNoBound) {
	auto r = std::make_unique<Mapped>();
	r->bytes = std::make_unique<Aligned>(raw.size());
	if (!raw.empty()) std::memcpy(r->bytes->p, raw.data(), raw.size());
	if (bound != kNoBound) sfd::SetExactnessHiddenBound(bound);
	r->st = sslm_model_map(r->bytes->p, raw.size(), &r->m);
	if (bound != kNoBound) sfd::ResetExactnessHiddenBound();
	if (r->st != SSLM_OK) r->m = nullptr;
	return r;
}

// ---- SCM1 writer (independent of SchemaMasksTable::Parse; test_slm19x's layout) -------------------

struct SchemaSpec {
	std::string name;
	uint32_t state_count = 0;
	std::vector<uint32_t> accepting;
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> rows;  // ascending (token, next)
};

std::vector<uint8_t> WriteScm1(const std::vector<SchemaSpec>& schemas, uint32_t vocab) {
	using superslm_test::PutU32;
	using superslm_test::PutU64;
	const uint32_t page = (vocab + 7) / 8;
	std::string blob;
	std::vector<uint32_t> name_off;
	for (const SchemaSpec& s : schemas) {
		name_off.push_back(static_cast<uint32_t>(blob.size()));
		blob += s.name;
	}
	uint64_t cursor = 24 + 56 * schemas.size() + blob.size();
	struct Blocks {
		std::vector<uint8_t> mask, offs, acc, trans;
		uint64_t mask_off = 0, offs_off = 0, acc_off = 0, trans_off = 0;
	};
	std::vector<Blocks> blocks;
	for (const SchemaSpec& s : schemas) {
		Blocks b;
		b.mask.assign(size_t{s.state_count} * page, 0);
		b.offs.assign((size_t{s.state_count} + 1) * 4, 0);
		uint32_t total = 0;
		for (uint32_t st = 0; st < s.state_count; ++st) {
			PutU32(b.offs, size_t{st} * 4, total);
			for (const auto& [tok, next] : s.rows[st]) {
				b.mask[size_t{st} * page + (tok >> 3)] |= static_cast<uint8_t>(1u << (tok & 7));
				const size_t at = b.trans.size();
				b.trans.resize(at + 8);
				PutU32(b.trans, at, tok);
				PutU32(b.trans, at + 4, next);
				++total;
			}
		}
		PutU32(b.offs, size_t{s.state_count} * 4, total);
		b.acc.assign(s.accepting.size() * 4, 0);
		for (size_t a = 0; a < s.accepting.size(); ++a) PutU32(b.acc, a * 4, s.accepting[a]);
		b.mask_off = cursor;
		cursor += b.mask.size();
		b.offs_off = cursor;
		cursor += b.offs.size();
		b.acc_off = cursor;
		cursor += b.acc.size();
		b.trans_off = cursor;
		cursor += b.trans.size();
		blocks.push_back(std::move(b));
	}
	std::vector<uint8_t> out(static_cast<size_t>(cursor), 0);
	std::memcpy(out.data(), "SCM1", 4);
	PutU32(out, 4, 1);
	PutU32(out, 8, static_cast<uint32_t>(schemas.size()));
	PutU32(out, 12, vocab);
	PutU32(out, 16, static_cast<uint32_t>(blob.size()));
	PutU32(out, 20, 0);
	for (size_t i = 0; i < schemas.size(); ++i) {
		const size_t d = 24 + 56 * i;
		uint32_t tc = 0;
		for (const auto& r : schemas[i].rows) tc += static_cast<uint32_t>(r.size());
		PutU32(out, d + 0, name_off[i]);
		PutU32(out, d + 4, static_cast<uint32_t>(schemas[i].name.size()));
		PutU32(out, d + 8, schemas[i].state_count);
		PutU32(out, d + 12, static_cast<uint32_t>(schemas[i].accepting.size()));
		PutU32(out, d + 16, tc);
		PutU32(out, d + 20, 0);
		PutU64(out, d + 24, blocks[i].mask_off);
		PutU64(out, d + 32, blocks[i].offs_off);
		PutU64(out, d + 40, blocks[i].acc_off);
		PutU64(out, d + 48, blocks[i].trans_off);
	}
	std::memcpy(out.data() + 24 + 56 * schemas.size(), blob.data(), blob.size());
	for (const Blocks& b : blocks) {
		std::memcpy(out.data() + b.mask_off, b.mask.data(), b.mask.size());
		std::memcpy(out.data() + b.offs_off, b.offs.data(), b.offs.size());
		if (!b.acc.empty()) std::memcpy(out.data() + b.acc_off, b.acc.data(), b.acc.size());
		if (!b.trans.empty()) std::memcpy(out.data() + b.trans_off, b.trans.data(), b.trans.size());
	}
	return out;
}

// ---- the fixture: F-SFD, its reference tables, its tokenizer bytes ---------------------------------

struct RefState {
	bool accepting = false;
	bool forced = false;
	uint32_t flen = 0;
	std::string F;  // bytes
	std::vector<int32_t> chain;
	std::vector<int32_t> path;
};

struct RefSchema {
	std::string name;
	uint32_t start = 0;
	std::string digest;
	std::vector<RefState> st;
};

struct Sfd {
	bool ok = false;
	std::vector<uint8_t> raw;
	std::vector<RawSec> secs;
	uint32_t flags = 0;
	std::unique_ptr<Mapped> m1;    // default map: the build under test's finish
	std::unique_ptr<Mapped> full;  // mapped with S-BOUND = 0: the full finish (the reference)
	std::map<std::string, RefSchema> ref;
	std::vector<uint8_t> scm_bytes;
	superslm::SchemaMasksTable scm;
	std::vector<std::string> vocab;  // TOK1 id -> bytes, padded to kV with ""
	int32_t byte_tok[256] = {};
	std::set<int32_t> specials;
	std::vector<int32_t> prompt_potion, prompt_pr;

	const superslm::SchemaEntry* Entry(const char* name, size_t* index = nullptr) const {
		return scm.ByName(name, index);
	}
	int32_t Index(const char* name) const {
		size_t i = 0;
		return scm.ByName(name, &i) ? static_cast<int32_t>(i) : -1;
	}
	bool Step(const char* name, uint32_t s, int32_t tok, uint32_t* next) const {
		const superslm::SchemaEntry* e = Entry(name);
		return e && tok >= 0 && scm.Transition(*e, s, static_cast<uint32_t>(tok), next);
	}
	std::vector<std::pair<uint32_t, uint32_t>> Row(const char* name, uint32_t s) const {
		std::vector<std::pair<uint32_t, uint32_t>> row;
		const superslm::SchemaEntry* e = Entry(name);
		if (!e || s >= e->state_count) return row;
		const uint32_t b = Le32(e->state_offsets_le + size_t{s} * 4);
		const uint32_t en = Le32(e->state_offsets_le + size_t{s} * 4 + 4);
		for (uint32_t i = b; i < en; ++i)
			row.push_back({Le32(e->transitions_le + size_t{i} * 8), Le32(e->transitions_le + size_t{i} * 8 + 4)});
		return row;
	}
	std::vector<uint32_t> LegalIds(const char* name, uint32_t s) const {
		std::vector<uint32_t> ids;
		for (const auto& [t, n] : Row(name, s)) ids.push_back(t);
		return ids;
	}
	const RefSchema& R(const char* name) const {
		static const RefSchema empty;
		auto it = ref.find(name);
		return it == ref.end() ? empty : it->second;
	}
	std::string Bytes(const std::vector<int32_t>& ids) const {
		std::string s;
		for (int32_t t : ids)
			if (t >= 0 && t < static_cast<int32_t>(vocab.size())) s += vocab[static_cast<size_t>(t)];
		return s;
	}
	int32_t TokenOf(const std::string& bytes) const {
		for (size_t i = 0; i < vocab.size(); ++i)
			if (vocab[i] == bytes && !specials.count(static_cast<int32_t>(i))) return static_cast<int32_t>(i);
		return -1;
	}
};

bool ParseTok1(const std::vector<uint8_t>& b, Sfd* f) {
	if (b.size() < 24 + 1024 || std::memcmp(b.data(), "TOK1", 4) != 0) return false;
	const uint32_t vc = Le32(b.data() + 8), mc = Le32(b.data() + 12), sc = Le32(b.data() + 16);
	size_t p = 24;
	for (int i = 0; i < 256; ++i) f->byte_tok[i] = static_cast<int32_t>(Le32(b.data() + p + 4 * size_t(i)));
	p += 1024;
	if (p + 4 * (size_t{vc} + 1) + 4 > b.size()) return false;
	std::vector<uint32_t> offs(size_t{vc} + 1);
	for (size_t i = 0; i <= vc; ++i) offs[i] = Le32(b.data() + p + 4 * i);
	p += 4 * (size_t{vc} + 1);
	const uint32_t bl = Le32(b.data() + p);
	p += 4;
	if (p + bl > b.size()) return false;
	f->vocab.assign(kV, std::string());
	for (size_t i = 0; i < vc && i < size_t(kV); ++i)
		f->vocab[i].assign(reinterpret_cast<const char*>(b.data() + p + offs[i]), offs[i + 1] - offs[i]);
	p += bl;
	p += 12 * size_t{mc};
	if (p + 4 * size_t{sc} > b.size()) return false;
	for (size_t i = 0; i < sc; ++i) f->specials.insert(static_cast<int32_t>(Le32(b.data() + p + 4 * i)));
	return true;
}

bool ParseRef(const std::string& text, Sfd* f) {
	std::istringstream in(text);
	std::string word;
	in >> word;
	if (word != "SFDREF") return false;
	int version = 0;
	in >> version;
	RefSchema* cur = nullptr;
	while (in >> word) {
		if (word == "end") return true;
		if (word == "schema") {
			std::string name, digest;
			uint32_t n = 0, start = 0;
			in >> name >> n >> start >> digest;
			RefSchema& r = f->ref[name];
			r.name = name;
			r.start = start;
			r.digest = digest;
			r.st.assign(n, RefState{});
			cur = &r;
		} else if (word == "state") {
			if (!cur) return false;
			uint32_t s = 0;
			int acc = 0, forced = 0;
			RefState rs;
			std::string fhex;
			size_t nc = 0, np = 0;
			in >> s >> acc >> forced >> rs.flen >> fhex >> nc;
			rs.chain.resize(nc);
			for (size_t i = 0; i < nc; ++i) in >> rs.chain[i];
			in >> np;
			rs.path.resize(np);
			for (size_t i = 0; i < np; ++i) in >> rs.path[i];
			rs.accepting = acc != 0;
			rs.forced = forced != 0;
			if (fhex != "-")
				for (size_t i = 0; i + 1 < fhex.size(); i += 2)
					rs.F.push_back(static_cast<char>(std::stoi(fhex.substr(i, 2), nullptr, 16)));
			if (s >= cur->st.size()) return false;
			cur->st[s] = std::move(rs);
		} else {
			return false;
		}
	}
	return false;
}

std::vector<int32_t> Tokenize(sslm_model m, const char* text) {
	std::vector<int32_t> ids(512);
	int32_t n = static_cast<int32_t>(ids.size());
	const sslm_status st = sslm_tokenize(m, text, ids.data(), &n);
	CHECK_MSG(st == SSLM_OK, "sslm_tokenize(\"%s\") == %d", text, static_cast<int>(st));
	ids.resize(st == SSLM_OK ? static_cast<size_t>(n) : 0);
	return ids;
}

const Sfd& GetSfd() {
	static std::unique_ptr<Sfd> cached;
	if (cached) {
		CHECK_MSG(cached->ok, "F-SFD did not load (see the first failure)");
		return *cached;
	}
	cached = std::make_unique<Sfd>();
	Sfd& f = *cached;
	const char* dir = std::getenv("SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR");
	CHECK_MSG(dir != nullptr && *dir != '\0',
	          "SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR is not set; run tools/gen_schema_fast_decode_fixture.py "
	          "and point it at the output (a missing fixture fails the cell, §11.0)");
	if (dir == nullptr || *dir == '\0') return f;
	const std::string base = std::string(dir) + "/";
	const bool read = superslm_marshal::ReadFile((base + "fsfd.sslm").c_str(), f.raw) && !f.raw.empty();
	CHECK_MSG(read, "fixture %sfsfd.sslm not found or empty", base.c_str());
	std::vector<uint8_t> reftext;
	const bool ref_read = superslm_marshal::ReadFile((base + "fsfd_ref.txt").c_str(), reftext) && !reftext.empty();
	CHECK_MSG(ref_read, "fixture %sfsfd_ref.txt not found or empty", base.c_str());
	if (!read || !ref_read) return f;
	const bool ref_ok = ParseRef(std::string(reftext.begin(), reftext.end()), &f);
	CHECK_MSG(ref_ok && f.ref.size() == 4, "fsfd_ref.txt does not parse (%zu schemas)", f.ref.size());
	const bool secs_ok = ReadSections(f.raw, &f.secs, &f.flags);
	CHECK_MSG(secs_ok, "fsfd.sslm: the section table does not read");
	bool tok_ok = false, scm_ok = false;
	for (const RawSec& s : f.secs) {
		if (s.type == kSecTokenizer) tok_ok = ParseTok1(s.data, &f);
		if (s.type == kSecSchemas) f.scm_bytes = s.data;
	}
	std::string err;
	scm_ok = !f.scm_bytes.empty() &&
	         superslm::SchemaMasksTable::Parse(f.scm_bytes.data(), f.scm_bytes.size(), kV, f.scm, &err);
	CHECK_MSG(tok_ok, "fsfd.sslm: TOK1 does not parse");
	CHECK_MSG(scm_ok, "fsfd.sslm: SCM1 does not parse: %s", err.c_str());
	f.m1 = Map(f.raw);
	CHECK_MSG(f.m1->st == SSLM_OK, "sslm_model_map(fsfd.sslm) == %d", static_cast<int>(f.m1->st));
	f.full = Map(f.raw, 0);
	CHECK_MSG(f.full->st == SSLM_OK, "sslm_model_map(fsfd.sslm, S-BOUND 0) == %d", static_cast<int>(f.full->st));
	f.ok = ref_ok && secs_ok && tok_ok && scm_ok && f.m1->m && f.full->m;
	if (f.ok) {
		f.prompt_potion = Tokenize(f.m1->m, "Customer: I would like to buy a health potion, please.\nOrder:");
		// PROMPT_RESULT's string is free text, and on F-SFD's untrained weights most prompts run it to the
		// 256-token cap. This one closes the string after 37 decodes on main (a dead end inside the cap),
		// which the feature oracle's finish count (10.1 (c), "+1 the final dead end") needs.
		f.prompt_pr = Tokenize(f.m1->m, "Summarize the request in one line.\nNotes:");
		f.ok = !f.prompt_potion.empty() && !f.prompt_pr.empty();
	}
	return f;
}

// The reference tables' own self-consistency against the artifact's SCM1 (the generator and the
// artifact come from one run, but a stale fsfd_ref.txt beside a fresh fsfd.sslm must not pass).
void CheckReferenceMatchesArtifact(const Sfd& f) {
	for (const auto& [name, r] : f.ref) {
		const superslm::SchemaEntry* e = f.Entry(name.c_str());
		CHECK_MSG(e != nullptr && e->state_count == r.st.size(), "[ref] %s: SCM1 has %u states, reference %zu",
		          name.c_str(), e ? e->state_count : 0u, r.st.size());
		if (!e) continue;
		for (size_t s = 0; s < r.st.size(); ++s) {
			uint32_t w = r.start;
			bool walk_ok = true;
			for (int32_t t : r.st[s].path) walk_ok = walk_ok && f.Step(name.c_str(), w, t, &w);
			CHECK_MSG(walk_ok && w == s, "[ref] %s: the reference path does not reach state %zu", name.c_str(), s);
		}
	}
}

// ---- the save blob ---------------------------------------------------------------------------------

struct Blob {
	bool ok = false;
	bool v6 = false;
	uint32_t walk = 0;
	int64_t ctx = 0;
	int32_t layer = 0;
	int32_t current = 0;
	int64_t forced = 0;
	int32_t order = 0;
	int64_t hist_n = 0;
	uint32_t ready = 0;
	std::vector<int32_t> hist;
	std::vector<uint8_t> kv;  // the K/V region (SSB5 flat block, or SSB6 rows)
};

Blob ParseBlob(const std::vector<uint8_t>& b) {
	Blob o;
	if (b.size() < 156) return o;
	o.v6 = std::memcmp(b.data(), "SSB6", 4) == 0;
	if (!o.v6 && std::memcmp(b.data(), "SSB5", 4) != 0) return o;
	o.walk = Le32(b.data() + 48);
	o.ctx = static_cast<int64_t>(Le64(b.data() + 60));
	o.layer = static_cast<int32_t>(Le32(b.data() + 68));
	o.current = static_cast<int32_t>(Le32(b.data() + 72));
	o.forced = static_cast<int64_t>(Le64(b.data() + 100));
	o.order = static_cast<int32_t>(Le32(b.data() + 108));
	o.hist_n = static_cast<int64_t>(Le64(b.data() + 112));
	o.ready = Le32(b.data() + 120);
	size_t off = (o.v6 ? 172 : 156) + size_t{kHidden};
	if (o.hist_n < 0 || off + 4 * static_cast<size_t>(o.hist_n) > b.size()) return o;
	for (int64_t i = 0; i < o.hist_n; ++i) o.hist.push_back(static_cast<int32_t>(Le32(b.data() + off + 4 * size_t(i))));
	off += 4 * static_cast<size_t>(o.hist_n) + (o.v6 ? 8 : 4);
	if (off > b.size()) return o;
	o.kv.assign(b.begin() + static_cast<std::ptrdiff_t>(off), b.end());
	o.ok = true;
	return o;
}

// ---- the ABI rig -----------------------------------------------------------------------------------

struct Rig {
	sslm_model m = nullptr;
	std::unique_ptr<Aligned> pool_buf;
	sslm_kv_pool pool = nullptr;
	std::vector<std::unique_ptr<Aligned>> ws_bufs;
	std::vector<sslm_workspace> ws;
	std::vector<sslm_seq> seqs;
	bool ok = false;

	explicit Rig(sslm_model model, int workspaces = 1, uint32_t blocks = 8, int32_t max_batch = 3) : m(model) {
		if (!m) return;
		const size_t size = blocks * sslm_kv_block_size(m) + sslm_kv_pool_overhead_size(m, blocks);
		pool_buf = std::make_unique<Aligned>(size);
		const sslm_status ps = sslm_kv_pool_create(m, pool_buf->p, size, blocks, &pool);
		CHECK_MSG(ps == SSLM_OK, "sslm_kv_pool_create == %d", static_cast<int>(ps));
		ok = ps == SSLM_OK;
		const sslm_config cfg{max_batch, 64, kLayers, 0u};
		for (int i = 0; i < workspaces && ok; ++i) {
			const size_t wsz = sslm_workspace_size(m, &cfg);
			ws_bufs.push_back(std::make_unique<Aligned>(wsz));
			sslm_workspace w = nullptr;
			const sslm_status wst = sslm_workspace_create(m, &cfg, ws_bufs.back()->p, wsz, &w);
			CHECK_MSG(wst == SSLM_OK, "sslm_workspace_create == %d", static_cast<int>(wst));
			ok = wst == SSLM_OK;
			ws.push_back(w);
		}
	}
	~Rig() {
		for (sslm_seq s : seqs) sslm_seq_release(s);
		for (sslm_workspace w : ws)
			if (w) sslm_workspace_destroy(w);
		if (pool) sslm_kv_pool_destroy(pool);
	}
	Rig(const Rig&) = delete;
	Rig& operator=(const Rig&) = delete;

	sslm_workspace W(int i = 0) const { return i >= 0 && i < static_cast<int>(ws.size()) ? ws[size_t(i)] : nullptr; }
	void Poison(int i, uint32_t seed) {
		if (i < 0 || i >= static_cast<int>(ws_bufs.size())) return;
		std::mt19937 g(seed);
		uint8_t* p = ws_bufs[size_t(i)]->p;
		for (size_t k = 0; k < ws_bufs[size_t(i)]->n; ++k) p[k] = static_cast<uint8_t>(g());
	}
	bool Bind(sslm_seq s, const char* schema) {
		if (!schema) return true;
		sslm_schema h = nullptr;
		const sslm_status ls = sslm_schema_lookup(m, schema, &h);
		CHECK_MSG(ls == SSLM_OK, "sslm_schema_lookup(%s) == %d", schema, static_cast<int>(ls));
		if (ls != SSLM_OK) return false;
		const sslm_status bs = sslm_seq_set_schema(s, h);
		CHECK_MSG(bs == SSLM_OK, "sslm_seq_set_schema(%s) == %d", schema, static_cast<int>(bs));
		return bs == SSLM_OK;
	}
	sslm_seq NewSeq(const char* schema = nullptr, sslm_adapter adapter = nullptr) {
		if (!ok) return nullptr;
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_create(m, &pool, &s);
		CHECK_MSG(st == SSLM_OK, "sslm_seq_create == %d", static_cast<int>(st));
		if (!s) return nullptr;
		seqs.push_back(s);
		if (adapter) CHECK_MSG(sslm_seq_set_adapter(s, adapter) == SSLM_OK, "sslm_seq_set_adapter");
		if (!Bind(s, schema)) return nullptr;
		return s;
	}
	sslm_seq NewBudgeted(int32_t budget, const char* schema) {
		if (!ok) return nullptr;
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_create_budgeted(m, &pool, budget, &s);
		CHECK_MSG(st == SSLM_OK, "sslm_seq_create_budgeted(%d) == %d", budget, static_cast<int>(st));
		if (!s) return nullptr;
		seqs.push_back(s);
		if (!Bind(s, schema)) return nullptr;
		return s;
	}
	sslm_seq Restore(const std::vector<uint8_t>& blob) {
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_restore(m, &pool, blob.data(), blob.size(), &s);
		CHECK_MSG(st == SSLM_OK, "sslm_seq_restore == %d", static_cast<int>(st));
		if (s) seqs.push_back(s);
		return s;
	}
	void Forget(sslm_seq s) { seqs.erase(std::remove(seqs.begin(), seqs.end(), s), seqs.end()); }
	void Release(sslm_seq s) {
		Forget(s);
		if (s) sslm_seq_release(s);
	}
	sslm_status Prefill(sslm_seq s, const std::vector<int32_t>& toks, sslm_span_kind kind, int32_t chunk = 64,
	                    sslm_workspace w = nullptr, int32_t* consumed_total = nullptr) {
		int32_t total = 0;
		sslm_status st = SSLM_OK;
		for (size_t at = 0; at < toks.size();) {
			const int32_t n = static_cast<int32_t>(std::min<size_t>(size_t(chunk), toks.size() - at));
			int32_t c = 0;
			st = sslm_prefill(m, s, toks.data() + at, n, n, kind, w, &c);
			total += c;
			if (st != SSLM_OK) break;
			at += size_t(n);
		}
		if (consumed_total) *consumed_total = total;
		return st;
	}
	bool PromptOk(sslm_seq s, const std::vector<int32_t>& p) {
		const sslm_status st = Prefill(s, p, SSLM_SPAN_PROMPT);
		CHECK_MSG(st == SSLM_OK, "prompt prefill == %d", static_cast<int>(st));
		return st == SSLM_OK;
	}
	bool ContentOk(sslm_seq s, const std::vector<int32_t>& t, int32_t chunk = 64) {
		if (t.empty()) return true;
		const sslm_status st = Prefill(s, t, SSLM_SPAN_SCHEMA_CONTENT, chunk);
		CHECK_MSG(st == SSLM_OK, "schema-content prefill of %zu tokens == %d", t.size(), static_cast<int>(st));
		return st == SSLM_OK;
	}
	sslm_decode_params Greedy(int32_t layer_budget = kLayers) const {
		sslm_decode_params p{};
		const sslm_status st = sslm_decode_params_init(m, SSLM_DECODE_MODE_GREEDY, layer_budget, &p);
		CHECK_MSG(st == SSLM_OK, "sslm_decode_params_init(greedy) == %d", static_cast<int>(st));
		return p;
	}
	sslm_decode_params Damped(int32_t order = SSLM_DAMPED_GREEDY_DEFAULT_ANTI_LM_ORDER,
	                          int32_t top_k = SSLM_DAMPED_GREEDY_DEFAULT_TOP_K) const {
		sslm_decode_params p{};
		const sslm_status st = sslm_decode_params_init(m, SSLM_DECODE_MODE_DAMPED_GREEDY, kLayers, &p);
		CHECK_MSG(st == SSLM_OK, "sslm_decode_params_init(damped) == %d", static_cast<int>(st));
		p.anti_lm_max_order = order;
		p.top_k = top_k;
		return p;
	}
	sslm_status Decode1(sslm_seq s, const sslm_decode_params& p, sslm_workspace w, int32_t* out) {
		*out = -100;
		return sslm_decode_step_v2(m, &s, 1, &p, w, out);
	}
	std::vector<uint8_t> Save(sslm_seq s) {
		std::vector<uint8_t> blob(sslm_seq_state_size(m) + 4096);
		size_t n = blob.size();
		const sslm_status st = sslm_seq_save(s, blob.data(), &n);
		CHECK_MSG(st == SSLM_OK, "sslm_seq_save == %d", static_cast<int>(st));
		blob.resize(st == SSLM_OK ? n : 0);
		return blob;
	}
	Blob State(sslm_seq s) { return ParseBlob(Save(s)); }
	sslm_status TryBind(sslm_seq s, const char* schema) {
		sslm_schema h = SSLM_SCHEMA_NONE;
		if (schema) sslm_schema_lookup(m, schema, &h);
		return sslm_seq_set_schema(s, h);
	}
};

// ---- the verbs, wrapped ---------------------------------------------------------------------------

struct Query {
	sslm_status st = SSLM_OK;
	int32_t count = -1;
	int32_t accepting = -1;
	std::vector<int32_t> ids;
};

Query QueryForced(sslm_model m, sslm_seq s, int32_t cap = 512) {
	Query q;
	q.ids.assign(size_t(cap > 0 ? cap : 1), -7);
	q.st = sslm_seq_forced_tokens(m, s, cap > 0 ? q.ids.data() : nullptr, cap, &q.count, &q.accepting);
	if (q.st == SSLM_OK && q.count >= 0) q.ids.resize(size_t(std::min(q.count, cap)));
	else q.ids.clear();
	return q;
}

struct Jump {
	sslm_status st = SSLM_OK;
	int32_t count = -1;
	std::vector<int32_t> forced;
};

Jump JumpForward(sslm_model m, sslm_seq s, int32_t budget, const sslm_decode_params* params, sslm_workspace w,
                 int32_t cap = -1) {
	Jump j;
	const int32_t c = cap < 0 ? budget : cap;
	std::vector<int32_t> buf(size_t(c > 0 ? c : 1), -7);
	j.st = sslm_seq_jump_forward(m, s, budget, params, w, buf.data(), c, &j.count);
	if (j.st == SSLM_OK || (j.count > 0 && j.count <= c)) {
		if (j.count > 0 && j.count <= c) j.forced.assign(buf.begin(), buf.begin() + j.count);
	}
	return j;
}

// ---- the §5.6 host loop -----------------------------------------------------------------------------

struct JfStep {
	int kind = 0;  // 0: a verb call; 1: a decode call
	sslm_status st = SSLM_OK;
	std::vector<int32_t> toks;  // verb: the forced ids it admitted; decode: {out}
	bool pending_before = false;
};

struct JfRun {
	bool dead_end = false;  // ended at decode's -2, every status OK
	sslm_status end_status = SSLM_OK;
	std::vector<int32_t> stream;
	std::vector<int32_t> choices;
	std::vector<JfStep> steps;
	int verb_calls = 0;
	int decodes = 0;
	int choice_choice = 0;  // decodes that embedded a pending token (choice -> choice)
	bool stuck = false;     // a verb admitted nothing while the query reported a chain
};

struct JfOpts {
	int32_t budget = 64;
	const sslm_decode_params* vparams = nullptr;  // the verb's params (NULL = greedy)
	sslm_decode_params dparams{};                 // decode's params
	sslm_workspace w = nullptr;
	int max_decodes = 400;
	// Called after every verb and decode call; may replace the sequence (save/restore between spans).
	std::function<void(sslm_seq&, const JfStep&)> after;
};

// The §5.6 loop: call the verb until the query reports no chain, then decode; repeat until decode's -2
// or a non-OK status. The prompt has already been prefilled.
JfRun RunJf(sslm_model m, sslm_seq& s, const JfOpts& o) {
	JfRun r;
	bool pending = false;  // a decode produced a token not yet admitted
	for (int d = 0; d < o.max_decodes; ++d) {
		bool admitted_forced = false;
		for (int guard = 0; guard < 1000; ++guard) {
			JfStep step;
			step.kind = 0;
			step.pending_before = pending;
			const Jump j = JumpForward(m, s, o.budget, o.vparams, o.w);
			step.st = j.st;
			step.toks = j.forced;
			r.steps.push_back(step);
			++r.verb_calls;
			r.stream.insert(r.stream.end(), j.forced.begin(), j.forced.end());
			if (!j.forced.empty()) admitted_forced = true;
			if (j.st == SSLM_OK && (pending || !j.forced.empty())) pending = false;
			if (o.after) o.after(s, step);
			if (j.st != SSLM_OK) {
				r.end_status = j.st;
				return r;
			}
			const Query q = QueryForced(m, s, 0);
			if (q.st != SSLM_OK) {
				r.end_status = q.st;
				return r;
			}
			if (q.count == 0) break;
			if (j.forced.empty()) {
				r.stuck = true;
				r.end_status = SSLM_INVALID_ARGUMENT;
				return r;
			}
		}
		JfStep step;
		step.kind = 1;
		step.pending_before = pending;
		int32_t out = -100;
		step.st = sslm_decode_step_v2(m, &s, 1, &o.dparams, o.w, &out);
		step.toks = {out};
		r.steps.push_back(step);
		++r.decodes;
		if (pending && !admitted_forced) ++r.choice_choice;
		if (o.after) o.after(s, step);
		if (step.st != SSLM_OK) {
			r.end_status = step.st;
			return r;
		}
		if (out == -2) {
			r.dead_end = true;
			return r;
		}
		if (out < 0) {
			r.end_status = SSLM_INVALID_ARGUMENT;
			return r;
		}
		r.stream.push_back(out);
		r.choices.push_back(out);
		pending = true;
	}
	return r;
}

// The stream the reference rule gives for these choices: the start chain, then for each choice the
// choice and the chain from the state it leads to. False when a choice is not legal where it lands.
bool RefStream(const Sfd& f, const char* schema, const std::vector<int32_t>& choices, std::vector<int32_t>* out,
               uint32_t* end_state = nullptr) {
	const RefSchema& r = f.R(schema);
	out->clear();
	if (r.st.empty()) return false;
	uint32_t s = r.start;
	auto chain = [&](uint32_t from) {
		for (int32_t t : r.st[from].chain) {
			out->push_back(t);
			if (!f.Step(schema, s, t, &s)) return false;
		}
		return true;
	};
	if (!chain(s)) return false;
	for (int32_t c : choices) {
		out->push_back(c);
		if (!f.Step(schema, s, c, &s)) return false;
		if (!chain(s)) return false;
	}
	if (end_state) *end_state = s;
	return true;
}

// The reference replay, main's paths only: for each choice, a fresh sequence prefills the prompt and
// then the reference history (start chain, choices, chains) as schema content, and decodes once. Its
// choice is appended with the chain it leads to, until a dead end (-2) or the cap.
struct Replay {
	bool ok = false;
	bool dead_end = false;
	bool capped = false;
	std::vector<int32_t> stream;
	std::vector<int32_t> choices;
	std::vector<std::vector<uint8_t>> kv_after;  // the K/V bytes after each choice's decode
	std::vector<int64_t> ctx_after;
};

Replay ReferenceReplay(sslm_model m, const Sfd& f, const char* schema, const std::vector<int32_t>& prompt,
                       sslm_adapter adapter = nullptr, int max_choices = 400) {
	Replay rp;
	const RefSchema& r = f.R(schema);
	if (r.st.empty()) return rp;
	Rig rig(m, 1, 2);
	if (!rig.ok) return rp;
	const sslm_decode_params g = rig.Greedy();
	uint32_t s = r.start;
	auto add_chain = [&]() {
		for (int32_t t : r.st[s].chain) {
			rp.stream.push_back(t);
			if (!f.Step(schema, s, t, &s)) return false;
		}
		return true;
	};
	if (!add_chain()) return rp;
	for (int i = 0; i < max_choices; ++i) {
		if (prompt.size() + rp.stream.size() >= size_t(kCap)) {
			rp.capped = true;
			rp.ok = true;
			return rp;
		}
		sslm_seq q = rig.NewSeq(schema, adapter);
		if (!q || !rig.PromptOk(q, prompt) || !rig.ContentOk(q, rp.stream)) return rp;
		int32_t c = -100;
		const sslm_status st = rig.Decode1(q, g, rig.W(), &c);
		CHECK_MSG(st == SSLM_OK, "[replay %s] decode at choice %d == %d", schema, i, static_cast<int>(st));
		if (st != SSLM_OK) return rp;
		const Blob b = rig.State(q);
		rig.Release(q);
		if (c == -2) {
			rp.dead_end = true;
			rp.ok = true;
			return rp;
		}
		rp.kv_after.push_back(b.kv);
		rp.ctx_after.push_back(b.ctx);
		rp.choices.push_back(c);
		rp.stream.push_back(c);
		if (!f.Step(schema, s, c, &s)) return rp;
		if (!add_chain()) return rp;
	}
	return rp;
}

// Fresh bound sequence, prompt prefilled, then the reference path to `state` as schema content: the
// sequence rests ready for logits at that state.
sslm_seq ReachState(Rig& rig, const Sfd& f, const char* schema, uint32_t state, const std::vector<int32_t>& prompt) {
	const RefSchema& r = f.R(schema);
	if (state >= r.st.size()) return nullptr;
	sslm_seq s = rig.NewSeq(schema);
	if (!s || !rig.PromptOk(s, prompt) || !rig.ContentOk(s, r.st[state].path)) return nullptr;
	return s;
}

// The first unforced, non-accepting state of `schema` whose row is small (a "choice" state), in
// reference order; and its legal row.
uint32_t FirstChoiceState(const Sfd& f, const char* schema, size_t max_row = 64) {
	const RefSchema& r = f.R(schema);
	for (uint32_t s = 0; s < r.st.size(); ++s) {
		if (r.st[s].forced || r.st[s].accepting) continue;
		const size_t n = f.Row(schema, s).size();
		if (n >= 2 && n <= max_row) return s;
	}
	return UINT32_MAX;
}

uint32_t AcceptingState(const Sfd& f, const char* schema) {
	const RefSchema& r = f.R(schema);
	for (uint32_t s = 0; s < r.st.size(); ++s)
		if (r.st[s].accepting && f.Row(schema, s).empty()) return s;
	return UINT32_MAX;
}

uint32_t LargestRowState(const Sfd& f, const char* schema, size_t* rows) {
	const RefSchema& r = f.R(schema);
	uint32_t best = UINT32_MAX;
	*rows = 0;
	for (uint32_t s = 0; s < r.st.size(); ++s) {
		const size_t n = f.Row(schema, s).size();
		if (n > *rows) {
			*rows = n;
			best = s;
		}
	}
	return best;
}

std::vector<int32_t> SRow() {
	std::vector<int32_t> row(size_t(kV), 0);
	const size_t n = sfd::LastFinishRow(row.data(), row.size());
	row.resize(n == size_t(kV) ? row.size() : 0);
	return row;
}

uint64_t LastRows() {
	const size_t n = sfd::RowsLogCount();
	return n ? sfd::RowsLogAt(n - 1) : UINT64_MAX;
}

// The expected legal-list split (§4.2: SplitColumns over n at alignment 64): rows per task =
// roundup(ceil(n / max_tasks), 64), tasks = ceil(n / rows). 0 when no `run` call is due.
int32_t ExpectedTasks(size_t n, int32_t max_tasks) {
	if (max_tasks < 2 || n == 0) return 0;
	size_t rows = (n + size_t(max_tasks) - 1) / size_t(max_tasks);
	rows = (rows + 63) / 64 * 64;
	return static_cast<int32_t>((n + rows - 1) / rows);
}

// ---- hooks ------------------------------------------------------------------------------------------

enum class Hostile { kNone, kDuplicate, kOmit, kMinusOne, kTaskCount };

struct CountingHook {
	int32_t max_tasks = 4;
	Hostile hostile = Hostile::kNone;
	int hostile_call = 0;
	std::mutex mu;
	std::vector<int32_t> counts;
	sslm_parallel_for Pf(uint32_t bits) { return sslm_parallel_for{&Run, this, max_tasks, bits}; }
	void Clear() {
		std::lock_guard<std::mutex> l(mu);
		counts.clear();
	}
	std::vector<int32_t> Counts() {
		std::lock_guard<std::mutex> l(mu);
		return counts;
	}
	static void Run(void* ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		CountingHook& h = *static_cast<CountingHook*>(ctx);
		int call = 0;
		{
			std::lock_guard<std::mutex> l(h.mu);
			call = static_cast<int>(h.counts.size());
			h.counts.push_back(task_count);
		}
		if (h.hostile != Hostile::kNone && call == h.hostile_call) {
			switch (h.hostile) {
			case Hostile::kDuplicate:
				for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
				task(task_ctx, 0);
				return;
			case Hostile::kOmit:
				for (int32_t i = 0; i + 1 < task_count; ++i) task(task_ctx, i);
				return;
			case Hostile::kMinusOne:
				task(task_ctx, -1);
				for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
				return;
			case Hostile::kTaskCount:
				for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
				task(task_ctx, task_count);
				return;
			case Hostile::kNone:
				break;
			}
		}
		for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
	}
};

bool Install(sslm_workspace w, CountingHook* h, uint32_t bits) {
	if (!h) return sslm_workspace_set_parallel_for(w, nullptr) == SSLM_OK;
	const sslm_parallel_for pf = h->Pf(bits);
	const sslm_status st = sslm_workspace_set_parallel_for(w, &pf);
	CHECK_MSG(st == SSLM_OK, "installing a hook (max_tasks %d, bits %#x) == %d", h->max_tasks, bits,
	          static_cast<int>(st));
	return st == SSLM_OK;
}

// ---- F-HAND (§11.0): hand-written SCM1 over F-SFD's own sections -------------------------------------

struct Hand {
	bool ok = false;
	std::vector<uint8_t> bytes;
	std::unique_ptr<Mapped> m1, full;
	int32_t tok_a = -1, tok_b = -1, tok_brace = -1, tok_lt = -1, tok_ab = -1, tok_ac = -1;
	int32_t special = -1;
	int32_t byteless = 151700;
	int32_t single = 12345;
};

const Hand& GetHand() {
	static std::unique_ptr<Hand> cached;
	if (cached) return *cached;
	cached = std::make_unique<Hand>();
	Hand& h = *cached;
	const Sfd& f = GetSfd();
	if (!f.ok) return h;
	h.tok_a = f.byte_tok['a'];
	h.tok_b = f.byte_tok['b'];
	h.tok_brace = f.byte_tok['{'];
	h.tok_lt = f.byte_tok['<'];
	h.tok_ab = f.TokenOf("ab");
	h.tok_ac = f.TokenOf("ac");
	for (int32_t sid : f.specials)
		if (sid < kTokVocab && !f.vocab[size_t(sid)].empty() && f.vocab[size_t(sid)][0] == '<') {
			h.special = sid;
			break;
		}
	CHECK_MSG(h.tok_ab >= 0 && h.tok_ac >= 0 && h.special >= 0 && f.vocab[size_t(h.byteless)].empty(),
	          "[F-HAND] token lookup: ab %d, ac %d, special %d", h.tok_ab, h.tok_ac, h.special);
	using Row = std::vector<std::pair<uint32_t, uint32_t>>;
	auto u = [](int32_t t) { return static_cast<uint32_t>(t); };
	auto sorted = [](Row r) {
		std::sort(r.begin(), r.end());
		return r;
	};
	std::vector<SchemaSpec> specs;
	// 2.1: a true forced cycle. 0 --{--> 1 --a--> 2 --b--> 1. State 0's F is "{" (its successor is on
	// the cycle, so unforced); states 1 and 2 are unforced.
	specs.push_back({"h_cycle", 3, {}, {{{u(h.tok_brace), 1}}, {{u(h.tok_a), 2}}, {{u(h.tok_b), 1}}}});
	// 2.5: a shared first byte with no legal byte token ("ab", "ac"; "a" itself is not legal).
	specs.push_back({"h_nobyte", 2, {1}, {sorted({{u(h.tok_ab), 1}, {u(h.tok_ac), 1}}), {}}});
	// 2.5: a forced-shaped row containing a byteless id (>= 151,665).
	specs.push_back({"h_byteless", 2, {1}, {sorted({{u(h.tok_a), 1}, {u(h.byteless), 1}}), {}}});
	// 2.5: a forced-shaped row containing a special id whose bytes start with the same byte.
	specs.push_back({"h_special", 2, {1}, {sorted({{u(h.tok_lt), 1}, {u(h.special), 1}}), {}}});
	// X8's state: accepting, with a non-empty row that shares one first byte and holds the byte token.
	specs.push_back({"h_accepting", 2, {0, 1}, {{{u(h.tok_a), 1}}, {}}});
	// 4.1: legal rows at id 0, at V - 1, a single interior row, and all V rows.
	specs.push_back({"h_row0", 2, {1}, {{{0u, 1}}, {}}});
	specs.push_back({"h_rowlast", 2, {1}, {{{u(kV - 1), 1}}, {}}});
	specs.push_back({"h_rowsingle", 2, {1}, {{{u(h.single), 1}}, {}}});
	Row all;
	all.reserve(size_t(kV));
	for (int32_t t = 0; t < kV; ++t) all.push_back({u(t), 1});
	specs.push_back({"h_rowall", 2, {1}, {all, {}}});
	std::vector<RawSec> secs = f.secs;
	for (RawSec& s : secs)
		if (s.type == kSecSchemas) s.data = WriteScm1(specs, kV);
	h.bytes = Rebuild(secs, f.flags);
	h.m1 = Map(h.bytes);
	h.full = Map(h.bytes, 0);
	CHECK_MSG(h.m1->st == SSLM_OK && h.full->st == SSLM_OK, "[F-HAND] map == %d / %d", static_cast<int>(h.m1->st),
	          static_cast<int>(h.full->st));
	h.ok = h.m1->m && h.full->m;
	return h;
}

std::vector<uint8_t> VariantWithout(const Sfd& f, std::set<uint32_t> drop, uint32_t clear_flags) {
	std::vector<RawSec> secs;
	for (const RawSec& s : f.secs)
		if (!drop.count(s.type)) secs.push_back(s);
	return Rebuild(secs, f.flags & ~clear_flags);
}

// ---- a synthetic adapter for F-SFD (8.5): the paged-kv budget suite's rank-1 LoRA construction -------

std::vector<uint8_t> BuildAdapterBytes(const Sfd& f) {
	using namespace superslm_test;
	superslm::SslmArtifact art;
	superslm::SslmError err;
	if (superslm::SslmArtifact::OpenFromMemory(f.m1->bytes->p, f.raw.size(), art, &err) != superslm::SslmStatus::Ok)
		return {};
	const superslm::SslmSectionView* cfg = art.Section(superslm::SslmSectionType::Config);
	const superslm::SslmSectionView* sil = art.Section(superslm::SslmSectionType::SigmoidLut);
	if (!cfg || !sil) return {};
	std::vector<uint8_t> adp = {'A', 'D', 'P', '1'};
	WriteU32LE(adp, 1);
	WriteU32LE(adp, 1);
	WriteU32LE(adp, 0x1u | 0x40u);  // q_proj, v_proj
	const auto hash = art.RawIntegrityHash();
	adp.insert(adp.end(), hash.begin(), hash.end());
	const double alpha = 8.0;
	uint64_t bits = 0;
	std::memcpy(&bits, &alpha, 8);
	WriteU64LE(adp, bits);
	WriteU32LE(adp, 0);
	WriteU32LE(adp, 0);
	const std::string name = "sfd-8.5";
	WriteU32LE(adp, static_cast<uint32_t>(name.size()));
	WriteBytes(adp, name);
	struct Proj {
		std::string name;
		uint32_t in, out;
	};
	const std::vector<Proj> projs = {{"layer0.q_proj", uint32_t(kHidden), 64u}, {"layer1.v_proj", uint32_t(kHidden), 32u}};
	std::vector<ManifestTensorSpec> wt, dt, ut;
	for (const Proj& p : projs) {
		wt.push_back({p.name + ".lora_A", {1, p.in}});
		wt.push_back({p.name + ".lora_B", {p.out, 1}});
		dt.push_back({p.name, {p.out * 3}});
		ut.push_back({p.name, {3}});
	}
	BuiltManifest w = BuildManifest(reinterpret_cast<const uint8_t*>("WGT1"), 1, wt);
	BuiltManifest d = BuildManifest(superslm::kDeltaFoldScalesMagic, 4, dt);
	BuiltManifest uu = BuildManifest(superslm::kUFoldScalesMagic, 4, ut);
	for (BuiltManifest* m : {&d, &uu})
		for (size_t t = 0; t < m->tensor_data_off.size(); ++t)
			for (uint64_t r = 0; r < m->tensor_elem_count[t] / 3; ++r) {
				const int32_t triple[3] = {1, 0, 0};
				std::memcpy(m->bytes.data() + m->tensor_data_off[t] + r * 12, triple, 12);
			}
	auto copy = [](superslm::SslmSectionType t, const superslm::SslmSectionView* v) {
		return MakeSection(t, v->dtype, std::vector<uint8_t>(v->data, v->data + v->byte_size));
	};
	BuiltArtifact a = BuildArtifact({copy(superslm::SslmSectionType::Config, cfg),
	                                 copy(superslm::SslmSectionType::SigmoidLut, sil),
	                                 MakeSection(superslm::SslmSectionType::Provenance, superslm::SslmDtype::Raw, adp),
	                                 MakeSection(superslm::SslmSectionType::Weights, superslm::SslmDtype::Int8, w.bytes),
	                                 MakeSection(superslm::SslmSectionType::DeltaFoldScales, superslm::SslmDtype::Int32, d.bytes),
	                                 MakeSection(superslm::SslmSectionType::UFoldScales, superslm::SslmDtype::Int32, uu.bytes)});
	return a.bytes;
}

// ===================================================================================================
// M1 cells
// ===================================================================================================

// 2.4 [M1] The predicate as a pure function: set at 131,071, clear at 131,072.
void Cell_2_4() {
	CHECK_MSG(superslm::LegalRowFinishExact(131071), "[2.4] the predicate is clear at hidden 131,071");
	CHECK_MSG(!superslm::LegalRowFinishExact(131072), "[2.4] the predicate is set at hidden 131,072");
	CHECK_MSG(superslm::LegalRowFinishExact(kHidden), "[2.4] the predicate is clear at F-SFD's hidden 64");
	CHECK_MSG(superslm::LegalRowFinishExact(896), "[2.4] the predicate is clear at the 0.5B's hidden 896");
}

struct PureRig {
	std::vector<int8_t> head;
	std::vector<int8_t> codes;
	std::vector<int64_t> wide_ref;
	std::vector<int32_t> row_ref;
	bool ok = false;
	PureRig() : head(size_t(kV) * kHidden), codes(kHidden), wide_ref(size_t(kV)), row_ref(size_t(kV)) {
		std::mt19937 g(0x5fd);
		for (int8_t& x : head) x = static_cast<int8_t>(static_cast<int>(g() % 255) - 127);
		for (int8_t& x : codes) x = static_cast<int8_t>(static_cast<int>(g() % 255) - 127);
		ok = superslm::LogitsSite(codes.data(), kHidden, head.data(), size_t(kV), wide_ref.data(), row_ref.data()) ==
		     superslm::SslmForwardStatus::Ok;
		CHECK_MSG(ok, "[pure] the reference LogitsSite failed");
	}
};

// Calls LogitsSiteLegalRows over `ids` on poisoned outputs; checks every legal position equals the
// reference full row and every other position is untouched.
void CheckLegalRowsPure(const char* cell, PureRig& p, const std::vector<uint32_t>& ids, const sslm_parallel_for* pf) {
	std::vector<int64_t> wide(size_t(kV), INT64_C(0x5a5a5a5a5a5a5a5a));
	std::vector<int32_t> out(size_t(kV), 0x13579bdf);
	const superslm::SslmForwardStatus st = superslm::LogitsSiteLegalRows(
	    p.codes.data(), kHidden, p.head.data(), size_t(kV), ids.data(), ids.size(), wide.data(), out.data(), pf);
	CHECK_MSG(st == superslm::SslmForwardStatus::Ok, "[%s] LogitsSiteLegalRows over %zu ids == %d", cell, ids.size(),
	          static_cast<int>(st));
	if (st != superslm::SslmForwardStatus::Ok) return;
	std::vector<uint8_t> legal(size_t(kV), 0);
	for (uint32_t id : ids) legal[id] = 1;
	size_t bad_legal = 0, bad_other = 0;
	for (size_t t = 0; t < size_t(kV); ++t) {
		if (legal[t]) {
			if (out[t] != p.row_ref[t] || wide[t] != p.wide_ref[t]) ++bad_legal;
		} else if (out[t] != 0x13579bdf || wide[t] != INT64_C(0x5a5a5a5a5a5a5a5a)) {
			++bad_other;
		}
	}
	CHECK_MSG(bad_legal == 0, "[%s] %zu of %zu legal rows differ from LogitsSite", cell, bad_legal, ids.size());
	CHECK_MSG(bad_other == 0, "[%s] %zu positions outside the legal list were written", cell, bad_other);
}

// 3.1 [M1] A counting hook at max_tasks 1, 2, 4 and 8: run is called iff max_tasks >= 2 and n >= 1, the
// task count is the legal-list split's, and the output equals serial. Pure function, then through the
// decode step at a POTION choice state, a PROMPT_RESULT 146,994-row state and an accepting state (n = 0).
void Cell_3_1() {
	PureRig p;
	const Sfd& f = GetSfd();
	if (!p.ok || !f.ok) return;
	size_t big = 0;
	const uint32_t s_big = LargestRowState(f, kPromptResult, &big);
	const uint32_t s_choice = FirstChoiceState(f, kPotion);
	const uint32_t s_acc = AcceptingState(f, kPotion);
	struct Case {
		const char* schema;
		uint32_t state;
	};
	const Case cases[] = {{kPotion, s_choice}, {kPromptResult, s_big}, {kPotion, s_acc}};
	for (const Case& c : cases) {
		const std::vector<uint32_t> ids = f.LegalIds(c.schema, c.state);
		for (int32_t mt : {1, 2, 4, 8}) {
			CountingHook h;
			h.max_tasks = mt;
			const sslm_parallel_for pf = h.Pf(0);
			CheckLegalRowsPure("3.1 pure", p, ids, &pf);
			const std::vector<int32_t> counts = h.Counts();
			const int32_t want = ExpectedTasks(ids.size(), mt);
			CHECK_MSG(want == 0 ? counts.empty() : (counts.size() == 1 && counts[0] == want),
			          "[3.1 pure] %s state %u, n %zu, max_tasks %d: run calls %s, want %s", c.schema, c.state,
			          ids.size(), mt, Str(counts).c_str(), want ? ("{" + std::to_string(want) + "}").c_str() : "none");
		}
	}
	// Through the decode step.
	for (const Case& c : cases) {
		const size_t n = f.Row(c.schema, c.state).size();
		Rig ref(f.m1->m);
		sslm_seq r = ReachState(ref, f, c.schema, c.state, f.prompt_potion);
		int32_t want_tok = -100;
		if (!r || ref.Decode1(r, ref.Greedy(), nullptr, &want_tok) != SSLM_OK) {
			CHECK_MSG(false, "[3.1] the serial decode at %s state %u failed", c.schema, c.state);
			continue;
		}
		for (int32_t mt : {1, 2, 4, 8}) {
			Rig rig(f.m1->m);
			CountingHook h;
			h.max_tasks = mt;
			if (!rig.ok || !Install(rig.W(), &h, 0)) continue;
			sslm_seq s = ReachState(rig, f, c.schema, c.state, f.prompt_potion);
			if (!s) continue;
			h.Clear();
			int32_t tok = -100;
			const sslm_status st = rig.Decode1(s, rig.Greedy(), rig.W(), &tok);
			const std::vector<int32_t> counts = h.Counts();
			const int32_t want = ExpectedTasks(n, mt);
			CHECK_MSG(st == SSLM_OK && tok == want_tok, "[3.1] %s state %u max_tasks %d: status %d token %d, serial %d",
			          c.schema, c.state, mt, static_cast<int>(st), tok, want_tok);
			CHECK_MSG(want == 0 ? counts.empty() : (counts.size() == 1 && counts[0] == want),
			          "[3.1] %s state %u (n %zu) max_tasks %d: run calls %s, want %d tasks once (0 = none)", c.schema,
			          c.state, n, mt, Str(counts).c_str(), want);
		}
	}
}

// 4.1 [M1] Legal rows at id 0, at V - 1, a single row, the real-schema maximum (PROMPT_RESULT's
// 146,994-row state) and all V rows: each equals the reference in token and S-ROW bytes. Pure function
// first, then through the decode step against the full finish.
void Cell_4_1() {
	PureRig p;
	const Sfd& f = GetSfd();
	const Hand& h = GetHand();
	if (!p.ok || !f.ok || !h.ok) return;
	size_t big = 0;
	const uint32_t s_big = LargestRowState(f, kPromptResult, &big);
	CHECK_MSG(big == 146994, "[4.1] PROMPT_RESULT's largest row has %zu legal tokens, the plan measured 146,994", big);
	std::vector<uint32_t> all(static_cast<size_t>(kV));
	for (int32_t t = 0; t < kV; ++t) all[size_t(t)] = static_cast<uint32_t>(t);
	CheckLegalRowsPure("4.1 id 0", p, {0u}, nullptr);
	CheckLegalRowsPure("4.1 id V-1", p, {static_cast<uint32_t>(kV - 1)}, nullptr);
	CheckLegalRowsPure("4.1 single", p, {static_cast<uint32_t>(h.single)}, nullptr);
	CheckLegalRowsPure("4.1 146,994", p, f.LegalIds(kPromptResult, s_big), nullptr);
	CheckLegalRowsPure("4.1 all V", p, all, nullptr);
	// Through the decode step: F-HAND's row schemas at state 0, and PROMPT_RESULT's largest state.
	struct Case {
		sslm_model m1, full;
		const char* schema;
		uint32_t state;
		size_t n;
	};
	const Case cases[] = {{h.m1->m, h.full->m, "h_row0", 0, 1},
	                      {h.m1->m, h.full->m, "h_rowlast", 0, 1},
	                      {h.m1->m, h.full->m, "h_rowsingle", 0, 1},
	                      {h.m1->m, h.full->m, "h_rowall", 0, size_t(kV)},
	                      {f.m1->m, f.full->m, kPromptResult, s_big, big}};
	for (const Case& c : cases) {
		std::vector<int32_t> rows[2];
		int32_t toks[2] = {-100, -100};
		uint64_t srows = UINT64_MAX;
		for (int arm = 0; arm < 2; ++arm) {
			Rig rig(arm == 0 ? c.full : c.m1);
			sslm_seq s = rig.NewSeq(c.schema);
			if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
			if (c.state != 0 && !rig.ContentOk(s, f.R(c.schema).st[c.state].path)) continue;
			sfd::RowsLogReset();
			sfd::LastFinishRowReset();
			const sslm_status st = rig.Decode1(s, rig.Greedy(), rig.W(), &toks[arm]);
			CHECK_MSG(st == SSLM_OK, "[4.1] %s: decode == %d", c.schema, static_cast<int>(st));
			rows[arm] = SRow();
			if (arm == 1) srows = LastRows();
		}
		CHECK_MSG(toks[0] == toks[1], "[4.1] %s: token %d, the full finish's %d", c.schema, toks[1], toks[0]);
		CHECK_MSG(!rows[0].empty() && rows[0] == rows[1], "[4.1] %s: the S-ROW bytes differ from the full finish's "
		          "(rows read: %zu / %zu)", c.schema, rows[0].size(), rows[1].size());
		CHECK_MSG(srows == c.n, "[4.1] %s: the legal-row finish computed %llu rows, want n = %zu", c.schema,
		          static_cast<unsigned long long>(srows), c.n);
	}
}

// 5.4 [M1] A dead end with n = 0: -2, ready_for_logits re-armed, token 0 (an all-masked row), as today.
// The degenerate-row seam still gives -2 under M1 (audit N4).
void Cell_5_4() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const uint32_t s_acc = AcceptingState(f, kPotion);
	Rig rig(f.m1->m);
	sslm_seq s = ReachState(rig, f, kPotion, s_acc, f.prompt_potion);
	if (!s) return;
	for (int rep = 0; rep < 2; ++rep) {
		sfd::RowsLogReset();
		sfd::LastFinishRowReset();
		int32_t tok = -100;
		const sslm_status st = rig.Decode1(s, rig.Greedy(), rig.W(), &tok);
		const Blob b = rig.State(s);
		CHECK_MSG(st == SSLM_OK && tok == -2, "[5.4] accepting state, decode %d: status %d out %d, want -2", rep,
		          static_cast<int>(st), tok);
		CHECK_MSG(b.ok && b.ready == 1 && b.layer == 0 && b.walk == s_acc,
		          "[5.4] after the dead end: ready %u, layer %d, walk %u (want 1, 0, %u)", b.ready, b.layer, b.walk, s_acc);
		CHECK_MSG(sfd::RowsLogCount() == 1 && LastRows() == 0, "[5.4] the n = 0 finish logged %zu finishes, last %llu "
		          "rows; want one finish of 0 rows", sfd::RowsLogCount(), static_cast<unsigned long long>(LastRows()));
		const std::vector<int32_t> row = SRow();
		const bool all_masked = !row.empty() && std::all_of(row.begin(), row.end(), [](int32_t v) { return v == INT32_MIN; });
		CHECK_MSG(all_masked, "[5.4] the dead end's S-ROW is not all INT32_MIN (token 0); rows read %zu", row.size());
	}
#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
	const uint32_t s_choice = FirstChoiceState(f, kPotion);
	sslm_seq c = ReachState(rig, f, kPotion, s_choice, f.prompt_potion);
	if (!c) return;
	sfd::RowsLogReset();
	ArmCpuFinishDegenerateLogitRowInjection();
	int32_t tok = -100;
	const sslm_status st = rig.Decode1(c, rig.Greedy(), rig.W(), &tok);
	const Blob b = rig.State(c);
	CHECK_MSG(st == SSLM_OK && tok == -2 && b.walk == s_choice && b.ready == 1,
	          "[5.4] degenerate row at the choice state: status %d out %d walk %u ready %u", static_cast<int>(st), tok,
	          b.walk, b.ready);
	CHECK_MSG(LastRows() == f.Row(kPotion, s_choice).size(), "[5.4] the degenerate-row finish ran %llu rows, want the "
	          "legal-row finish's n = %zu", static_cast<unsigned long long>(LastRows()), f.Row(kPotion, s_choice).size());
#else
	CHECK_MSG(false, "[5.4] the suite's library lacks SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION (CMakeLists.txt)");
#endif
}

// 5.5 [M1] A hostile hook on the legal-list split gives ParallelForIncomplete (SSLM_INVALID_ARGUMENT at
// the ABI). The sequence rests as today, and a retry through a correct hook gives the same token.
void Cell_5_5() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const uint32_t s_choice = FirstChoiceState(f, kPotion);
	const size_t n = f.Row(kPotion, s_choice).size();
	int32_t want_tok = -100;
	{
		Rig ref(f.m1->m);
		sslm_seq r = ReachState(ref, f, kPotion, s_choice, f.prompt_potion);
		if (!r || ref.Decode1(r, ref.Greedy(), nullptr, &want_tok) != SSLM_OK) return;
	}
	for (Hostile hk : {Hostile::kDuplicate, Hostile::kOmit, Hostile::kMinusOne, Hostile::kTaskCount}) {
		Rig rig(f.m1->m);
		CountingHook h;
		h.max_tasks = 4;
		h.hostile = hk;
		if (!rig.ok || !Install(rig.W(), &h, 0)) continue;
		sslm_seq s = ReachState(rig, f, kPotion, s_choice, f.prompt_potion);
		if (!s) continue;
		const std::vector<uint8_t> before = rig.Save(s);
		h.Clear();
		int32_t tok = -100;
		const sslm_status st = rig.Decode1(s, rig.Greedy(), rig.W(), &tok);
		const std::vector<int32_t> counts = h.Counts();
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "[5.5] hostile %d: status %d, want SSLM_INVALID_ARGUMENT",
		          static_cast<int>(hk), static_cast<int>(st));
		CHECK_MSG(counts.size() == 1 && counts[0] == ExpectedTasks(n, 4),
		          "[5.5] hostile %d: the split had task counts %s; the legal-list split of n = %zu is %d task(s)",
		          static_cast<int>(hk), Str(counts).c_str(), n, ExpectedTasks(n, 4));
		const std::vector<uint8_t> after = rig.Save(s);
		CHECK_MSG(before == after, "[5.5] hostile %d: the sequence did not rest as it was", static_cast<int>(hk));
		h.hostile = Hostile::kNone;
		const sslm_status rt = rig.Decode1(s, rig.Greedy(), rig.W(), &tok);
		CHECK_MSG(rt == SSLM_OK && tok == want_tok, "[5.5] hostile %d: the retry gave status %d token %d, want %d",
		          static_cast<int>(hk), static_cast<int>(rt), tok, want_tok);
	}
}

// 6.1 [M1, F-SFD] Over every state of every schema, with a workspace and without one, the legal-row
// finish equals the full finish in produced token, S-ROW bytes and SSB blob. One arm poisons the
// workspace (and so logit_row) with random bytes first. Each arm first shows it ran the legal-row
// finish (S-ROWS = n), so a build that never takes the new path cannot pass.
void Cell_6_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	int states = 0, bad_tok = 0, bad_row = 0, bad_blob = 0, bad_rows = 0;
	for (const auto& [name, r] : f.ref) {
		for (uint32_t s = 0; s < r.st.size(); ++s) {
			++states;
			const size_t n = f.Row(name.c_str(), s).size();
			Rig full(f.full->m), m1(f.m1->m);
			sslm_seq a = ReachState(full, f, name.c_str(), s, f.prompt_potion);
			sslm_seq b = ReachState(m1, f, name.c_str(), s, f.prompt_potion);
			if (!a || !b) continue;
			const std::vector<uint8_t> blob_b = m1.Save(b);
			int32_t ta = -100;
			sfd::LastFinishRowReset();
			full.Decode1(a, full.Greedy(), full.W(), &ta);
			const std::vector<int32_t> row_a = SRow();
			const std::vector<uint8_t> after_a = full.Save(a);
			for (int arm = 0; arm < 3; ++arm) {
				sslm_seq q = arm == 0 ? b : m1.Restore(blob_b);
				if (!q) continue;
				if (arm == 2) m1.Poison(0, 0x61u + s);
				sfd::RowsLogReset();
				sfd::LastFinishRowReset();
				int32_t tb = -100;
				m1.Decode1(q, m1.Greedy(), arm == 1 ? nullptr : m1.W(), &tb);
				const std::vector<int32_t> row_b = SRow();
				if (tb != ta) ++bad_tok;
				if (row_a.empty() || row_b != row_a) ++bad_row;
				if (m1.Save(q) != after_a) ++bad_blob;
				if (!(sfd::RowsLogCount() == 1 && LastRows() == n)) ++bad_rows;
			}
		}
	}
	CHECK_MSG(states == 266, "[6.1] %d states visited, F-SFD has 266", states);
	CHECK_MSG(bad_rows == 0, "[6.1] %d of %d arms did not run the legal-row finish (S-ROWS != n)", bad_rows, states * 3);
	CHECK_MSG(bad_tok == 0, "[6.1] %d arms produced a different token", bad_tok);
	CHECK_MSG(bad_row == 0, "[6.1] %d arms differ in S-ROW bytes (or no row was read)", bad_row);
	CHECK_MSG(bad_blob == 0, "[6.1] %d arms differ in the SSB blob", bad_blob);
}

// 6.3 [M1] Damped greedy at top_k greater than the legal count, over 64 steps: produced token, anti-LM
// state and SSB blob equal the reference (the full finish). The WithScratch path reads no wide_logits
// element it did not write: the legal-row arm poisons the workspace before every step.
void Cell_6_3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const char* schema : {kPotion, kLongKey}) {
		Rig full(f.full->m), m1(f.m1->m);
		sslm_seq a = full.NewSeq(schema);
		sslm_seq b = m1.NewSeq(schema);
		if (!a || !b || !full.PromptOk(a, f.prompt_potion) || !m1.PromptOk(b, f.prompt_potion)) continue;
		const sslm_decode_params pa = full.Damped(2, 32);
		const sslm_decode_params pb = m1.Damped(2, 32);
		int bad = 0, bad_rows = 0, legal_ok = 0;
		for (int step = 0; step < 64; ++step) {
			const uint32_t walk = m1.State(b).walk;
			const size_t n = f.Row(schema, walk).size();
			if (n < 32) ++legal_ok;
			int32_t ta = -100, tb = -100;
			const sslm_status sa = full.Decode1(a, pa, full.W(), &ta);
			m1.Poison(0, 0x63u + uint32_t(step));
			sfd::RowsLogReset();
			const sslm_status sb = m1.Decode1(b, pb, m1.W(), &tb);
			if (!(sfd::RowsLogCount() == 1 && LastRows() == n)) ++bad_rows;
			if (sa != sb || ta != tb || full.Save(a) != m1.Save(b)) ++bad;
			if (sa != SSLM_OK) break;
		}
		CHECK_MSG(legal_ok > 0, "[6.3] %s: no step had fewer legal tokens than top_k 32", schema);
		CHECK_MSG(bad_rows == 0, "[6.3] %s: %d of 64 steps did not run the legal-row finish (S-ROWS != n)", schema, bad_rows);
		CHECK_MSG(bad == 0, "[6.3] %s: %d steps differ from the full finish in token, anti-LM state or blob", schema, bad);
	}
}

// 8.4 [M1] A decode batch of n = 3: POTION, PROMPT_RESULT and an unbound sequence. Each sequence's tokens
// equal its solo run, and the batch's finishes ran n, n and V rows.
void Cell_8_4() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const char* schemas[3] = {kPotion, kPromptResult, nullptr};
	constexpr int kSteps = 24;
	std::vector<int32_t> solo[3];
	for (int i = 0; i < 3; ++i) {
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(schemas[i]);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
		for (int k = 0; k < kSteps; ++k) {
			int32_t t = -100;
			if (rig.Decode1(s, rig.Greedy(), rig.W(), &t) != SSLM_OK) break;
			solo[i].push_back(t);
		}
	}
	Rig rig(f.m1->m, 1, 8, 3);
	sslm_seq s[3];
	for (int i = 0; i < 3; ++i) {
		s[i] = rig.NewSeq(schemas[i]);
		if (!s[i] || !rig.PromptOk(s[i], f.prompt_potion)) return;
	}
	std::vector<int32_t> got[3];
	int bad_rows = 0;
	for (int k = 0; k < kSteps; ++k) {
		uint64_t want_rows[3];
		for (int i = 0; i < 3; ++i) want_rows[i] = schemas[i] ? f.Row(schemas[i], rig.State(s[i]).walk).size() : uint64_t(kV);
		int32_t out[3] = {-100, -100, -100};
		const sslm_decode_params g = rig.Greedy();
		sfd::RowsLogReset();
		const sslm_status st = sslm_decode_step_v2(rig.m, s, 3, &g, rig.W(), out);
		CHECK_MSG(st == SSLM_OK, "[8.4] batch step %d == %d", k, static_cast<int>(st));
		if (st != SSLM_OK) break;
		size_t logged = 0;
		for (int i = 0; i < 3; ++i) {
			got[i].push_back(out[i]);
			if (out[i] == -2) continue;  // a dead end still finishes: n = 0 rows
		}
		logged = sfd::RowsLogCount();
		if (logged != 3) ++bad_rows;
		else
			for (int i = 0; i < 3; ++i)
				if (sfd::RowsLogAt(size_t(i)) != want_rows[i]) ++bad_rows;
	}
	for (int i = 0; i < 3; ++i)
		CHECK_MSG(got[i] == solo[i], "[8.4] sequence %d (%s): batch %s, solo %s", i, schemas[i] ? schemas[i] : "unbound",
		          Str(got[i]).c_str(), Str(solo[i]).c_str());
	CHECK_MSG(bad_rows == 0, "[8.4] %d batch steps did not log finishes of n, n and V rows in order", bad_rows);
}

// 11.1 [M1] S-BOUND set below the model's hidden size gives S-ROWS = V; with it unset, S-ROWS = n.
void Cell_11_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const uint32_t s_choice = FirstChoiceState(f, kPotion);
	const size_t n = f.Row(kPotion, s_choice).size();
	std::unique_ptr<Mapped> below = Map(f.raw, size_t(kHidden - 1));
	std::unique_ptr<Mapped> at = Map(f.raw, size_t(kHidden));
	struct Case {
		sslm_model m;
		uint64_t want;
		const char* what;
	};
	const Case cases[] = {{below->m, uint64_t(kV), "S-BOUND 63 (below hidden 64)"},
	                      {at->m, uint64_t(n), "S-BOUND 64 (= hidden)"},
	                      {f.m1->m, uint64_t(n), "S-BOUND unset"}};
	for (const Case& c : cases) {
		Rig rig(c.m);
		sslm_seq s = ReachState(rig, f, kPotion, s_choice, f.prompt_potion);
		if (!s) continue;
		sfd::RowsLogReset();
		int32_t t = -100;
		rig.Decode1(s, rig.Greedy(), rig.W(), &t);
		CHECK_MSG(sfd::RowsLogCount() == 1 && LastRows() == c.want, "[11.1] %s: S-ROWS logged %zu finishes, last %llu "
		          "rows; want %llu", c.what, sfd::RowsLogCount(), static_cast<unsigned long long>(LastRows()),
		          static_cast<unsigned long long>(c.want));
	}
}

// 11.4 [M1] Damped greedy with the flag set: S-ROWS = n at every step, with a workspace and without one.
void Cell_11_4() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (int with_ws = 0; with_ws < 2; ++with_ws) {
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
		const sslm_decode_params p = rig.Damped();
		int bad = 0, steps = 0;
		for (int k = 0; k < 20; ++k) {
			const size_t n = f.Row(kPotion, rig.State(s).walk).size();
			sfd::RowsLogReset();
			int32_t t = -100;
			if (rig.Decode1(s, p, with_ws ? rig.W() : nullptr, &t) != SSLM_OK) break;
			++steps;
			if (!(sfd::RowsLogCount() == 1 && LastRows() == n)) ++bad;
		}
		CHECK_MSG(steps == 20 && bad == 0, "[11.4] damped greedy %s a workspace: %d of %d steps did not log S-ROWS = n",
		          with_ws ? "with" : "without", bad, steps);
	}
}

// ===================================================================================================
// J1 cells
// ===================================================================================================

JfOpts GreedyOpts(Rig& rig, int32_t budget = 64) {
	JfOpts o;
	o.budget = budget;
	o.dparams = rig.Greedy();
	o.w = rig.W();
	return o;
}

// A fresh bound sequence, prompt prefilled, the §5.6 loop to its end.
JfRun FreshRun(Rig& rig, const Sfd& /*f*/, const char* schema, const std::vector<int32_t>& prompt, const JfOpts& o,
               sslm_seq* out_seq = nullptr) {
	sslm_seq s = rig.NewSeq(schema);
	if (!s || !rig.PromptOk(s, prompt)) return JfRun{};
	JfRun r = RunJf(rig.m, s, o);
	if (out_seq) *out_seq = s;
	return r;
}

// Everything 10.1 (a) asks of a run: it ended at a dead end, and its stream is the reference rule's
// for the choices it made.
void CheckStreamIsReference(const char* cell, const Sfd& f, const char* schema, const JfRun& r) {
	CHECK_MSG(r.dead_end && r.end_status == SSLM_OK, "[%s] %s: the loop ended with status %d (dead end %d, stuck %d) "
	          "after %d verb calls and %d decodes", cell, schema, static_cast<int>(r.end_status), r.dead_end, r.stuck,
	          r.verb_calls, r.decodes);
	std::vector<int32_t> want;
	const bool legal = RefStream(f, schema, r.choices, &want);
	CHECK_MSG(legal && r.stream == want, "[%s] %s: stream %s, reference %s", cell, schema, Str(r.stream).c_str(),
	          Str(want).c_str());
}

// 10.1 [J1] The feature oracle, (a) through (e) (§5.9), with forced_token_count asserted.
void Cell_10_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	struct Case {
		const char* schema;
		const std::vector<int32_t>* prompt;
	};
	const Case cases[] = {{kPotion, &f.prompt_potion}, {kPromptResult, &f.prompt_pr}};
	for (const Case& c : cases) {
		const Replay rp = ReferenceReplay(f.m1->m, f, c.schema, *c.prompt);
		CHECK_MSG(rp.ok && rp.dead_end, "[10.1] %s: the reference replay did not end at a dead end (ok %d, capped %d)",
		          c.schema, rp.ok, rp.capped);
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(c.schema);
		if (!s || !rig.PromptOk(s, *c.prompt)) continue;
		// (e) is checked at each decode: the run's choice and K/V bytes equal the replay's.
		size_t choice_i = 0;
		int bad_e = 0;
		JfOpts o = GreedyOpts(rig);
		o.after = [&](sslm_seq& q, const JfStep& st) {
			if (st.kind != 1 || st.st != SSLM_OK || st.toks[0] < 0) return;
			const Blob b = rig.State(q);
			if (choice_i >= rp.choices.size() || st.toks[0] != rp.choices[choice_i] || b.kv != rp.kv_after[choice_i] ||
			    b.ctx != rp.ctx_after[choice_i])
				++bad_e;
			++choice_i;
		};
		sfd::LayerRowsLogReset();
		sfd::RowsLogReset();
		const JfRun r = RunJf(rig.m, s, o);
		// (a)
		CheckStreamIsReference("10.1 (a)", f, c.schema, r);
		CHECK_MSG(r.stream == rp.stream, "[10.1 (a)] %s: stream %s, replay %s", c.schema, Str(r.stream).c_str(),
		          Str(rp.stream).c_str());
		// (b) token rows
		size_t batched = 0, decode_rows = 0, over_budget = 0, entries = sfd::LayerRowsLogCount();
		uint64_t batched_rows = 0;
		for (size_t i = 0; i < entries; ++i) {
			const sfd::LayerRowsEntry e = sfd::LayerRowsLogAt(i);
			if (e.kind == sfd::kLayerRowsBatched) {
				++batched;
				batched_rows += e.rows;
				if (e.rows > uint64_t(o.budget)) ++over_budget;
			} else if (e.kind == sfd::kLayerRowsDecode) {
				++decode_rows;
			}
		}
		CHECK_MSG(batched > 0 && over_budget == 0, "[10.1 (b)] %s: %zu batched passes, %zu over the budget", c.schema,
		          batched, over_budget);
		CHECK_MSG(batched_rows + decode_rows == r.stream.size(), "[10.1 (b)] %s: %llu batched rows + %zu decode rows, "
		          "the stream has %zu tokens", c.schema, static_cast<unsigned long long>(batched_rows), decode_rows,
		          r.stream.size());
		CHECK_MSG(decode_rows == size_t(r.choice_choice), "[10.1 (b)] %s: %zu layer-running decodes, %d choice->choice "
		          "transitions", c.schema, decode_rows, r.choice_choice);
		// (c) finishes = choices + 1
		CHECK_MSG(r.decodes == int(r.choices.size()) + 1 && sfd::RowsLogCount() == r.choices.size() + 1,
		          "[10.1 (c)] %s: %d decodes, %zu finishes logged, %zu choices", c.schema, r.decodes, sfd::RowsLogCount(),
		          r.choices.size());
		// (e)
		CHECK_MSG(choice_i == rp.choices.size() && bad_e == 0, "[10.1 (e)] %s: %d of %zu choices differ from a fresh "
		          "sequence's (choice or K/V), %zu replay choices", c.schema, bad_e, choice_i, rp.choices.size());
		// forced_token_count: every token the verbs admitted, never a choice.
		const Blob b = rig.State(s);
		CHECK_MSG(b.ok && b.forced == int64_t(r.stream.size() - r.choices.size()), "[10.1] %s: forced_token_count %lld, "
		          "want %zu", c.schema, static_cast<long long>(b.forced), r.stream.size() - r.choices.size());
	}
	// (d) per-state chain: on every state of every schema, the engine's chain equals the reference rule's.
	int bad_d = 0, states = 0;
	for (const auto& [name, r] : f.ref) {
		Rig rig(f.m1->m);
		for (uint32_t st = 0; st < r.st.size(); ++st) {
			++states;
			sslm_seq s = ReachState(rig, f, name.c_str(), st, f.prompt_potion);
			if (!s) {
				++bad_d;
				continue;
			}
			const Query q = QueryForced(rig.m, s, 512);
			const RefState& rs = r.st[st];
			if (q.st != SSLM_OK || q.count != int32_t(rs.chain.size()) || q.ids != rs.chain ||
			    q.accepting != (rs.accepting ? 1 : 0))
				++bad_d;
			rig.Release(s);
		}
	}
	CHECK_MSG(states == 266 && bad_d == 0, "[10.1 (d)] %d of %d states: the engine's chain differs from the reference "
	          "rule's", bad_d, states);
}

// 1.1 [J1] One sequence: prompt, the loop to completion, reset, bind another schema, the loop again.
// The outputs equal two fresh sequences'.
void Cell_1_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	Rig rig(f.m1->m);
	const JfRun fa = FreshRun(rig, f, kPotion, f.prompt_potion, GreedyOpts(rig));
	const JfRun fb = FreshRun(rig, f, kLongKey, f.prompt_potion, GreedyOpts(rig));
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const JfRun a = RunJf(rig.m, s, GreedyOpts(rig));
	CHECK_MSG(sslm_seq_reset(s) == SSLM_OK, "[1.1] reset");
	if (!rig.Bind(s, kLongKey) || !rig.PromptOk(s, f.prompt_potion)) return;
	const JfRun b = RunJf(rig.m, s, GreedyOpts(rig));
	CHECK_MSG(fa.dead_end && fb.dead_end, "[1.1] the fresh runs did not complete (status %d, %d)",
	          static_cast<int>(fa.end_status), static_cast<int>(fb.end_status));
	CHECK_MSG(a.stream == fa.stream && !a.stream.empty(), "[1.1] first run %s, fresh %s", Str(a.stream).c_str(),
	          Str(fa.stream).c_str());
	CHECK_MSG(b.stream == fb.stream && !b.stream.empty(), "[1.1] after reset %s, fresh %s", Str(b.stream).c_str(),
	          Str(fb.stream).c_str());
}

// 1.2 [J1] After a span, set_schema(NONE) and set_schema(other) each return BIND_REJECTED and change
// nothing. A verb call on a fresh sequence (including a no-op call) clears bind_eligible. The read-only
// query does not.
void Cell_1_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	Rig rig(f.m1->m);
	{
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
		const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
		CHECK_MSG(j.st == SSLM_OK && j.count > 0, "[1.2] the start span: status %d count %d", static_cast<int>(j.st), j.count);
		const std::vector<uint8_t> before = rig.Save(s);
		const sslm_status a = rig.TryBind(s, nullptr);
		const sslm_status b = rig.TryBind(s, kLongKey);
		CHECK_MSG(a == SSLM_SCHEMA_BIND_REJECTED && b == SSLM_SCHEMA_BIND_REJECTED, "[1.2] after a span: unbind %d, "
		          "rebind %d", static_cast<int>(a), static_cast<int>(b));
		CHECK_MSG(rig.Save(s) == before, "[1.2] a rejected bind changed the sequence");
	}
	{
		// A no-op verb call (unbound, so no chain) on a fresh sequence clears eligibility.
		sslm_seq s = rig.NewSeq(nullptr);
		if (!s) return;
		const Jump j = JumpForward(rig.m, s, 8, nullptr, rig.W());
		CHECK_MSG(j.st == SSLM_OK && j.count == 0, "[1.2] a no-op call: status %d count %d", static_cast<int>(j.st), j.count);
		CHECK_MSG(rig.TryBind(s, kPotion) == SSLM_SCHEMA_BIND_REJECTED, "[1.2] a bind after a no-op verb call was accepted");
	}
	{
		// The same on a fresh bound sequence at its forced start (no prompt): the start span is admitted.
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s) return;
		const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
		CHECK_MSG(j.st == SSLM_OK && j.forced == f.R(kPotion).st[f.R(kPotion).start].chain,
		          "[1.2] fresh bound, no prompt: status %d forced %s", static_cast<int>(j.st), Str(j.forced).c_str());
		CHECK_MSG(rig.TryBind(s, kLongKey) == SSLM_SCHEMA_BIND_REJECTED, "[1.2] a bind after a verb call was accepted");
	}
	{
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s) return;
		const Query q = QueryForced(rig.m, s);
		CHECK_MSG(q.st == SSLM_OK && q.count == int32_t(f.R(kPotion).st[0].chain.size()), "[1.2] query: status %d count %d",
		          static_cast<int>(q.st), q.count);
		CHECK_MSG(rig.TryBind(s, kLongKey) == SSLM_OK, "[1.2] the read-only query cleared bind eligibility");
	}
}

// 1.3 [J1] Model unmap frees the table (LSan reads the leak; the cell proves the table existed).
void Cell_1_3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (int i = 0; i < 3; ++i) {
		std::unique_ptr<Mapped> m = Map(f.raw);
		if (!m->m) continue;
		int32_t nf = -9;
		uint32_t fl = 0;
		const sslm_status st = sfd::ForcedTableEntry(m->m, f.Index(kPotion), 0, &nf, &fl);
		CHECK_MSG(st == SSLM_OK && nf >= 0 && fl > 0, "[1.3] map %d: no forced table (status %d, next %d, |F| %u)", i,
		          static_cast<int>(st), nf, fl);
	}
}

// 2.1 [J1, F-HAND] A forced cycle. The build terminates (a timeout counts as red). The cycle's states are
// unforced, the verb is a no-op there, and the loop stops at decode.
void Cell_2_1() {
	const Sfd& f = GetSfd();
	const Hand& h = GetHand();
	if (!f.ok || h.bytes.empty()) return;
	std::unique_ptr<Mapped> m;
	const bool done = RunWithTimeout([&] { m = Map(h.bytes); }, 60);
	CHECK_MSG(done, "[2.1] mapping the forced-cycle artifact did not finish in 60 s");
	if (!done || !m->m) return;
	// h_cycle's index in F-HAND's SCM1 is its position in the spec list: 0.
	int32_t nf[3] = {-9, -9, -9};
	uint32_t fl[3] = {9, 9, 9};
	for (uint32_t s = 0; s < 3; ++s) sfd::ForcedTableEntry(m->m, 0, s, &nf[s], &fl[s]);
	CHECK_MSG(nf[0] == h.tok_brace && fl[0] == 1, "[2.1] state 0: next %d |F| %u, want '{' (%d) and 1", nf[0], fl[0],
	          h.tok_brace);
	CHECK_MSG(nf[1] == -1 && fl[1] == 0 && nf[2] == -1 && fl[2] == 0, "[2.1] the cycle's states are forced: (%d, %u), "
	          "(%d, %u)", nf[1], fl[1], nf[2], fl[2]);
	Rig rig(m->m);
	sslm_seq s = rig.NewSeq("h_cycle");
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	JfOpts o = GreedyOpts(rig);
	o.max_decodes = 1;
	const JfRun r = RunJf(rig.m, s, o);
	CHECK_MSG(r.stream.size() == 2 && !r.stream.empty() && r.stream[0] == h.tok_brace && r.decodes == 1,
	          "[2.1] the loop: stream %s after %d decodes; want '{' then one decode", Str(r.stream).c_str(), r.decodes);
	// The decode's choice is pending: this call admits it (h = 1) and nothing forced follows on the cycle.
	(void)JumpForward(rig.m, s, 8, nullptr, rig.W());
	const Blob b = rig.State(s);
	const Jump j2 = JumpForward(rig.m, s, 8, nullptr, rig.W());
	CHECK_MSG(j2.st == SSLM_OK && j2.count == 0 && rig.State(s).walk == b.walk, "[2.1] on the cycle the verb admitted "
	          "%d (status %d)", j2.count, static_cast<int>(j2.st));
}

// 2.2 [J1] Refused, with no state change: chunk_budget 0 or negative; cap < chunk_budget; a null
// out_forced_count. A refused call leaves the sequence bindable.
void Cell_2_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const std::vector<uint8_t> before = rig.Save(s);
	int32_t buf[64];
	int32_t count = -9;
	struct Case {
		int32_t budget, cap;
		bool null_count;
		const char* what;
	};
	const Case cases[] = {{0, 64, false, "budget 0"}, {-1, 64, false, "budget -1"}, {8, 7, false, "cap 7 < budget 8"},
	                      {8, 8, true, "null out_forced_count"}};
	for (const Case& c : cases) {
		count = -9;
		const sslm_status st =
		    sslm_seq_jump_forward(rig.m, s, c.budget, nullptr, rig.W(), buf, c.cap, c.null_count ? nullptr : &count);
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "[2.2] %s: status %d, want SSLM_INVALID_ARGUMENT", c.what, static_cast<int>(st));
		CHECK_MSG(rig.Save(s) == before, "[2.2] %s changed the sequence", c.what);
	}
	sslm_seq fresh = rig.NewSeq(kPotion);
	if (!fresh) return;
	const sslm_status st = sslm_seq_jump_forward(rig.m, fresh, 0, nullptr, rig.W(), buf, 64, &count);
	CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "[2.2] budget 0 on a fresh sequence: status %d", static_cast<int>(st));
	CHECK_MSG(rig.TryBind(fresh, kLongKey) == SSLM_OK, "[2.2] a refused call cleared bind eligibility");
}

// 2.3 [J1] SCM1 without TOK1: the map succeeds, the verbs return SSLM_ARTIFACT_REJECTED, decode works.
void Cell_2_3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	std::unique_ptr<Mapped> m = Map(VariantWithout(f, {kSecTokenizer, kSecUnicode}, 0));
	CHECK_MSG(m->st == SSLM_OK, "[2.3] map without TOK1 == %d", static_cast<int>(m->st));
	if (!m->m) return;
	Rig rig(m->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const std::vector<uint8_t> before = rig.Save(s);
	const Query q = QueryForced(rig.m, s);
	const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(q.st == SSLM_ARTIFACT_REJECTED, "[2.3] query: status %d", static_cast<int>(q.st));
	CHECK_MSG(j.st == SSLM_ARTIFACT_REJECTED, "[2.3] verb: status %d", static_cast<int>(j.st));
	CHECK_MSG(rig.Save(s) == before, "[2.3] the refused verb changed the sequence");
	int32_t t = -100;
	const sslm_status ds = rig.Decode1(s, rig.Greedy(), rig.W(), &t);
	CHECK_MSG(ds == SSLM_OK && t >= 0, "[2.3] decode: status %d token %d", static_cast<int>(ds), t);
	CHECK_MSG(sfd::ForcedTableEntry(m->m, 0, 0, nullptr, nullptr) != SSLM_OK, "[2.3] the table seam answered");
	int32_t nf = 0;
	uint32_t fl = 0;
	CHECK_MSG(sfd::ForcedTableEntry(m->m, 0, 0, &nf, &fl) == SSLM_ARTIFACT_REJECTED, "[2.3] the table seam does not "
	          "report a missing table");
}

// 2.5 [J1, F-HAND] No byte token; a byteless id or a special id in the row: each state is unforced, with
// no crash. And X8's state: accepting with a forced-shaped row is unforced, out_accepting is 1, the verb is
// a no-op.
void Cell_2_5() {
	const Sfd& f = GetSfd();
	const Hand& h = GetHand();
	if (!f.ok || !h.ok) return;
	const char* names[] = {"h_nobyte", "h_byteless", "h_special", "h_accepting"};
	for (int i = 0; i < 4; ++i) {
		int32_t nf = -9;
		uint32_t fl = 9;
		const sslm_status st = sfd::ForcedTableEntry(h.m1->m, 1 + i, 0, &nf, &fl);
		CHECK_MSG(st == SSLM_OK && nf == -1 && fl == 0, "[2.5] %s state 0: status %d next %d |F| %u, want unforced",
		          names[i], static_cast<int>(st), nf, fl);
		Rig rig(h.m1->m);
		sslm_seq s = rig.NewSeq(names[i]);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
		const Query q = QueryForced(rig.m, s);
		CHECK_MSG(q.st == SSLM_OK && q.count == 0 && q.accepting == (i == 3 ? 1 : 0), "[2.5] %s: query status %d count %d "
		          "accepting %d", names[i], static_cast<int>(q.st), q.count, q.accepting);
		const std::vector<uint8_t> before = rig.Save(s);
		const Jump j = JumpForward(rig.m, s, 8, nullptr, rig.W());
		CHECK_MSG(j.st == SSLM_OK && j.count == 0 && rig.State(s).ctx == ParseBlob(before).ctx, "[2.5] %s: the verb "
		          "admitted %d (status %d)", names[i], j.count, static_cast<int>(j.st));
	}
}

// 2.6 [J1, F-SFD UTF8ENUM] S-CHOICE chooses the 0xC3 byte token. The chain's bytes equal F, the
// replacement character's token 5691 never appears, and the chain equals the reference rule's.
void Cell_2_6() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kUtf8Enum);
	const int32_t c3 = f.byte_tok[0xC3];
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kUtf8Enum);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const Jump start = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(start.st == SSLM_OK && start.forced == r.st[r.start].chain, "[2.6] the start span: status %d forced %s",
	          static_cast<int>(start.st), Str(start.forced).c_str());
	// The choice state is the reference walk's over the start span (not the engine's, so a verb that
	// admitted nothing fails above and here, not at the S-CHOICE legality check).
	uint32_t choice = r.start;
	bool walked = true;
	for (int32_t tok : r.st[r.start].chain) walked = walked && f.Step(kUtf8Enum, choice, tok, &choice);
	CHECK_MSG(walked && rig.State(s).walk == choice, "[2.6] after the start span the walk is at %u, the reference's %u",
	          rig.State(s).walk, choice);
	uint32_t next = 0;
	CHECK_MSG(walked && f.Step(kUtf8Enum, choice, c3, &next), "[2.6] 0xC3 (%d) is not legal at state %u", c3, choice);
	sfd::ArmChoice(c3);
	int32_t t = -100;
	const sslm_status ds = rig.Decode1(s, rig.Greedy(), rig.W(), &t);
	sfd::DisarmChoice();
	CHECK_MSG(ds == SSLM_OK && t == c3, "[2.6] S-CHOICE: decode status %d token %d, want %d", static_cast<int>(ds), t, c3);
	const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
	const RefState& rs = next < r.st.size() ? r.st[next] : r.st[0];
	CHECK_MSG(j.st == SSLM_OK && j.forced == rs.chain, "[2.6] chain after 0xC3: status %d %s, reference %s",
	          static_cast<int>(j.st), Str(j.forced).c_str(), Str(rs.chain).c_str());
	CHECK_MSG(f.Bytes(j.forced) == rs.F && !rs.F.empty(), "[2.6] the chain's bytes are not F");
	CHECK_MSG(std::find(j.forced.begin(), j.forced.end(), kReplacementToken) == j.forced.end(),
	          "[2.6] the chain holds the replacement character's token 5691");
}

// 3.2 [J1] Two sequences on two threads share one table (TSan). No race, and the outputs equal serial.
void Cell_3_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	JfRun serial[2], threaded[2];
	const char* schemas[2] = {kPotion, kPromptResult};
	const std::vector<int32_t>* prompts[2] = {&f.prompt_potion, &f.prompt_pr};
	{
		Rig rig(f.m1->m);
		for (int i = 0; i < 2; ++i) serial[i] = FreshRun(rig, f, schemas[i], *prompts[i], GreedyOpts(rig));
	}
	Rig rig(f.m1->m, 2);
	sslm_seq s[2] = {rig.NewSeq(schemas[0]), rig.NewSeq(schemas[1])};
	for (int i = 0; i < 2; ++i)
		if (!s[i] || !rig.PromptOk(s[i], *prompts[i])) return;
	std::thread th[2];
	for (int i = 0; i < 2; ++i) {
		th[i] = std::thread([&, i] {
			JfOpts o;
			o.dparams = rig.Greedy();
			o.w = rig.W(i);
			threaded[i] = RunJf(rig.m, s[i], o);
		});
	}
	for (std::thread& t : th) t.join();
	for (int i = 0; i < 2; ++i) {
		CHECK_MSG(threaded[i].stream == serial[i].stream && serial[i].dead_end, "[3.2] %s: threaded %s, serial %s",
		          schemas[i], Str(threaded[i].stream).c_str(), Str(serial[i].stream).c_str());
	}
}

// A hook whose first `run` blocks until released (or 5 s pass), so a second thread can act while the
// verb is inside its batched pass.
struct BlockingHook {
	std::mutex mu;
	std::condition_variable cv;
	bool entered = false;
	bool released = false;
	int calls = 0;
	sslm_parallel_for Pf() { return sslm_parallel_for{&Run, this, 4, SSLM_PARALLEL_FOR_MATVEC}; }
	static void Run(void* ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		BlockingHook& h = *static_cast<BlockingHook*>(ctx);
		{
			std::unique_lock<std::mutex> l(h.mu);
			if (h.calls++ == 0) {
				h.entered = true;
				h.cv.notify_all();
				h.cv.wait_for(l, std::chrono::seconds(5), [&] { return h.released; });
			}
		}
		for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
	}
	bool WaitEntered(int seconds) {
		std::unique_lock<std::mutex> l(mu);
		return cv.wait_for(l, std::chrono::seconds(seconds), [&] { return entered; });
	}
	void Release() {
		std::lock_guard<std::mutex> l(mu);
		released = true;
		cv.notify_all();
	}
};

// 3.3 [J1] Under TSan: a release during the verb, and a decode_step on the same sequence during the verb.
// No race, and the surviving call's outputs are unchanged. The verb is held inside a one-token admission
// (budget 1, threaded under bit 1) by a hook that blocks in `run`.
void Cell_3_3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const std::vector<int32_t>& start = f.R(kPotion).st[f.R(kPotion).start].chain;
	// The serial reference for the decode arm: verb (budget 1), then decode.
	int32_t ref_tok = -100;
	{
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
		JumpForward(rig.m, s, 1, nullptr, rig.W());
		rig.Decode1(s, rig.Greedy(), rig.W(), &ref_tok);
	}
	for (int arm = 0; arm < 2; ++arm) {
		Rig rig(f.m1->m);
		BlockingHook h;
		const sslm_parallel_for pf = h.Pf();
		if (!rig.ok || sslm_workspace_set_parallel_for(rig.W(), &pf) != SSLM_OK) continue;
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
		Jump j;
		std::thread verb([&] { j = JumpForward(rig.m, s, 1, nullptr, rig.W()); });
		const bool inside = h.WaitEntered(5);
		CHECK_MSG(inside, "[3.3] arm %d: the verb never entered the hook's run (a one-token admission under bit 1)", arm);
		std::thread other;
		sslm_status other_st = SSLM_OK;
		int32_t other_tok = -100;
		if (inside) {
			if (arm == 0) {
				rig.Forget(s);
				other = std::thread([&] { other_st = sslm_seq_release(s); });
			} else {
				Rig* r = &rig;
				other = std::thread([&, r] {
					const sslm_decode_params g = r->Greedy();
					other_st = sslm_decode_step_v2(r->m, &s, 1, &g, nullptr, &other_tok);
				});
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
		h.Release();
		verb.join();
		if (other.joinable()) other.join();
		CHECK_MSG(j.st == SSLM_OK && j.forced.size() == 1 && !start.empty() && j.forced[0] == start[0],
		          "[3.3] arm %d: the verb gave status %d forced %s, want {%d}", arm, static_cast<int>(j.st),
		          Str(j.forced).c_str(), start.empty() ? -1 : start[0]);
		if (arm == 0) {
			CHECK_MSG(other_st == SSLM_OK, "[3.3] the concurrent release == %d", static_cast<int>(other_st));
		} else {
			CHECK_MSG(other_st == SSLM_OK && other_tok == ref_tok, "[3.3] the concurrent decode: status %d token %d, "
			          "serial %d", static_cast<int>(other_st), other_tok, ref_tok);
		}
	}
}

// 4.2 [J1] Spans of 1..8 tokens and 9+ (LONGKEY): K/V equals token-by-token prefill (on every tier the
// suite runs on: scalar, SSE2, AVX2, AVX-512).
void Cell_4_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const std::vector<int32_t>& chain = f.R(kLongKey).st[f.R(kLongKey).start].chain;
	CHECK_MSG(chain.size() >= 9, "[4.2] LONGKEY's start chain has %zu tokens; 9+ needed", chain.size());
	for (int32_t m : {1, 2, 3, 4, 5, 6, 7, 8, 26}) {
		const size_t k = std::min<size_t>(size_t(m), chain.size());
		Rig rig(f.m1->m);
		sslm_seq a = rig.NewSeq(kLongKey);
		sslm_seq b = rig.NewSeq(kLongKey);
		if (!a || !b || !rig.PromptOk(a, f.prompt_potion) || !rig.PromptOk(b, f.prompt_potion)) continue;
		const Jump j = JumpForward(rig.m, a, m, nullptr, rig.W());
		CHECK_MSG(j.st == SSLM_OK && j.forced.size() == k, "[4.2] span of %d: status %d, admitted %zu", m,
		          static_cast<int>(j.st), j.forced.size());
		rig.ContentOk(b, std::vector<int32_t>(chain.begin(), chain.begin() + std::ptrdiff_t(k)), 1);
		const Blob ba = rig.State(a), bb = rig.State(b);
		CHECK_MSG(ba.ok && bb.ok && ba.kv == bb.kv && ba.ctx == bb.ctx, "[4.2] span of %d: K/V differs from token-by-token "
		          "prefill (context %lld vs %lld)", m, static_cast<long long>(ba.ctx), static_cast<long long>(bb.ctx));
	}
}

// 5.1 [J1] §5.5's table, row by row, on a cap holder and on a KV-budget holder. Each case is followed by
// one decode_step, in which context_length must not rise.
void Cell_5_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kPotion);
	const size_t start_n = r.st[r.start].chain.size();  // 3
	for (int holder = 0; holder < 2; ++holder) {
		const sslm_status limit_st = holder == 0 ? SSLM_CONTEXT_CAP_EXCEEDED : SSLM_KV_BUDGET_EXCEEDED;
		const char* hname = holder == 0 ? "cap holder" : "KV-budget holder";
		for (int h = 0; h <= 4; ++h) {  // the headroom at the pending choice; h = 4 is the "h >= k" row for small k
			Rig rig(f.m1->m);
			std::vector<int32_t> prompt;
			int32_t budget = 0;
			if (holder == 0) {
				const size_t len = size_t(kCap) - start_n - size_t(h);
				for (size_t i = 0; i < len; ++i) prompt.push_back(f.prompt_potion[i % f.prompt_potion.size()]);
			} else {
				prompt = f.prompt_potion;
				budget = static_cast<int32_t>(prompt.size() + start_n) + h;
			}
			sslm_seq s = holder == 0 ? rig.NewSeq(kPotion) : rig.NewBudgeted(budget, kPotion);
			if (!s || !rig.PromptOk(s, prompt)) continue;
			const Jump st0 = JumpForward(rig.m, s, 64, nullptr, rig.W());
			CHECK_MSG(st0.st == SSLM_OK && st0.forced.size() == start_n, "[5.1 %s h %d] the start span: status %d, %zu",
			          hname, h, static_cast<int>(st0.st), st0.forced.size());
			int32_t c = -100;
			if (rig.Decode1(s, rig.Greedy(), rig.W(), &c) != SSLM_OK || c < 0) continue;
			const Blob before = rig.State(s);
			const RefState& next = r.st[before.walk < r.st.size() ? before.walk : 0];
			const size_t k = 1 + next.chain.size();
			const std::vector<uint8_t> blob_before = rig.Save(s);
			const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
			const Blob after = rig.State(s);
			if (h == 0) {
				CHECK_MSG(j.st == limit_st && rig.Save(s) == blob_before, "[5.1 %s h 0] status %d; want the limit status "
				          "and nothing changed", hname, static_cast<int>(j.st));
				int32_t t = -100;
				const sslm_status ds = rig.Decode1(s, rig.Greedy(), rig.W(), &t);
				CHECK_MSG(ds == limit_st, "[5.1 %s h 0] the next decode: status %d, want the limit status", hname,
				          static_cast<int>(ds));
				continue;
			}
			if (size_t(h) < k) {
				const size_t forced = size_t(h) - 1;  // the pending token takes one row
				uint32_t walk = before.walk;
				for (size_t i = 0; i < forced; ++i) f.Step(kPotion, walk, next.chain[i], &walk);
				CHECK_MSG(j.st == limit_st, "[5.1 %s h %d < k %zu] status %d, want the limit status", hname, h, k,
				          static_cast<int>(j.st));
				CHECK_MSG(j.forced.size() == forced && after.ready == 1 && after.walk == walk &&
				              after.forced == before.forced + int64_t(forced) && after.ctx == before.ctx + h,
				          "[5.1 %s h %d] admitted %zu forced (want %zu), ready %u, walk %u (want %u), forced_token_count "
				          "%lld (want %lld), context %lld (want %lld)", hname, h, j.forced.size(), forced, after.ready,
				          after.walk, walk, static_cast<long long>(after.forced),
				          static_cast<long long>(before.forced + int64_t(forced)), static_cast<long long>(after.ctx),
				          static_cast<long long>(before.ctx + h));
			} else {
				CHECK_MSG(j.st == SSLM_OK && j.forced == next.chain && after.ctx == before.ctx + int64_t(k),
				          "[5.1 %s h %d >= k %zu] status %d, forced %s", hname, h, k, static_cast<int>(j.st),
				          Str(j.forced).c_str());
			}
			const int64_t ctx = rig.State(s).ctx;
			int32_t t = -100;
			rig.Decode1(s, rig.Greedy(), rig.W(), &t);
			CHECK_MSG(rig.State(s).ctx == ctx, "[5.1 %s h %d] the next decode raised context_length %lld -> %lld", hname, h,
			          static_cast<long long>(ctx), static_cast<long long>(rig.State(s).ctx));
		}
	}
}

// 5.2 [J1] Allocation failure inside the batched pass: no state change, and a retry gives the same result.
void Cell_5_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const std::vector<uint8_t> before = rig.Save(s);
	superslm_test::g_inject_jump_forward_pass_alloc_failure = true;
	const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
	const bool consumed = !superslm_test::g_inject_jump_forward_pass_alloc_failure;
	superslm_test::g_inject_jump_forward_pass_alloc_failure = false;
	CHECK_MSG(consumed && j.st == SSLM_ALLOCATION_FAILED, "[5.2] status %d (slot consumed %d), want SSLM_ALLOCATION_FAILED",
	          static_cast<int>(j.st), consumed);
	CHECK_MSG(rig.Save(s) == before, "[5.2] the failed call changed the sequence");
	const Jump retry = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(retry.st == SSLM_OK && retry.forced == f.R(kPotion).st[0].chain, "[5.2] the retry: status %d forced %s",
	          static_cast<int>(retry.st), Str(retry.forced).c_str());
}

// 5.3 [J1] A verb call with layer_index != 0 is refused and changes nothing.
void Cell_5_3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion) || !rig.ContentOk(s, f.R(kPotion).st[0].chain)) return;
	int32_t c = -100;
	rig.Decode1(s, rig.Greedy(), rig.W(), &c);  // a choice, now pending
	int32_t mid = -100;
	const sslm_status ms = rig.Decode1(s, rig.Greedy(1), rig.W(), &mid);
	const Blob b = rig.State(s);
	CHECK_MSG(ms == SSLM_OK && mid == -1 && b.layer == 1, "[5.3] setup: a budget-1 decode left layer %d (status %d, out %d)",
	          b.layer, static_cast<int>(ms), mid);
	const std::vector<uint8_t> before = rig.Save(s);
	const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(j.st == SSLM_INVALID_ARGUMENT, "[5.3] mid-layer verb: status %d", static_cast<int>(j.st));
	CHECK_MSG(rig.Save(s) == before, "[5.3] the refused call changed the sequence");
}

// 5.6 [J1] Allocation failure during the table build: the map fails with SSLM_ALLOCATION_FAILED.
void Cell_5_6() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	superslm_test::g_inject_forced_table_build_alloc_failure = true;
	std::unique_ptr<Mapped> m = Map(f.raw);
	const bool consumed = !superslm_test::g_inject_forced_table_build_alloc_failure;
	superslm_test::g_inject_forced_table_build_alloc_failure = false;
	CHECK_MSG(consumed && m->st == SSLM_ALLOCATION_FAILED && m->m == nullptr, "[5.6] map: status %d (slot consumed %d)",
	          static_cast<int>(m->st), consumed);
	std::unique_ptr<Mapped> again = Map(f.raw);
	CHECK_MSG(again->st == SSLM_OK, "[5.6] the next map == %d", static_cast<int>(again->st));
}

// 6.2 [J1] Under the §5.6 loop, chunk budgets 1, 2, 3, 7 and 64 give identical tokens and an identical SSB blob.
void Cell_6_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const char* schema : {kPotion, kLongKey}) {
		std::vector<int32_t> streams[5];
		std::vector<uint8_t> blobs[5];
		const int32_t budgets[5] = {64, 1, 2, 3, 7};
		for (int i = 0; i < 5; ++i) {
			Rig rig(f.m1->m);
			sslm_seq s = nullptr;
			const JfRun r = FreshRun(rig, f, schema, f.prompt_potion, GreedyOpts(rig, budgets[i]), &s);
			CHECK_MSG(r.dead_end, "[6.2] %s budget %d: the loop ended with status %d", schema, budgets[i],
			          static_cast<int>(r.end_status));
			streams[i] = r.stream;
			if (s) blobs[i] = rig.Save(s);
		}
		for (int i = 1; i < 5; ++i) {
			CHECK_MSG(streams[i] == streams[0] && !streams[0].empty(), "[6.2] %s budget %d: %s, budget 64: %s", schema,
			          budgets[i], Str(streams[i]).c_str(), Str(streams[0]).c_str());
			CHECK_MSG(blobs[i] == blobs[0], "[6.2] %s budget %d: the SSB blob differs from budget 64's", schema, budgets[i]);
		}
	}
}

// 6.4 [J1] Repeat runs are byte-equal.
void Cell_6_4() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	std::vector<int32_t> st[2];
	std::vector<uint8_t> bl[2];
	for (int i = 0; i < 2; ++i) {
		Rig rig(f.m1->m);
		sslm_seq s = nullptr;
		const JfRun r = FreshRun(rig, f, kPromptResult, f.prompt_pr, GreedyOpts(rig), &s);
		st[i] = r.stream;
		if (s) bl[i] = rig.Save(s);
	}
	CHECK_MSG(st[0] == st[1] && !st[0].empty() && bl[0] == bl[1] && !bl[0].empty(), "[6.4] repeat runs differ (%zu / %zu "
	          "tokens)", st[0].size(), st[1].size());
}

// 6.5 [J1] Contract violation: a decode_step at a forced state after a partial span. The output is
// schema-valid and deterministic, and it differs from 6.2's stream. This documents why §5.6 binds.
void Cell_6_5() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	std::vector<int32_t> lawful;
	{
		Rig rig(f.m1->m);
		lawful = FreshRun(rig, f, kPotion, f.prompt_potion, GreedyOpts(rig)).stream;
	}
	std::vector<int32_t> broken[2];
	for (int rep = 0; rep < 2; ++rep) {
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
		const Jump j = JumpForward(rig.m, s, 1, nullptr, rig.W());  // a partial span: one of the start chain's tokens
		CHECK_MSG(j.st == SSLM_OK && j.count == 1, "[6.5] the partial span: status %d count %d", static_cast<int>(j.st), j.count);
		broken[rep] = j.forced;
		int32_t t = -100;
		const sslm_status ds = rig.Decode1(s, rig.Greedy(), rig.W(), &t);  // at a forced state: the violation
		if (ds != SSLM_OK || t < 0) continue;
		broken[rep].push_back(t);
		const JfRun rest = RunJf(rig.m, s, GreedyOpts(rig));
		broken[rep].insert(broken[rep].end(), rest.stream.begin(), rest.stream.end());
		uint32_t w = f.R(kPotion).start;
		bool valid = true;
		for (int32_t x : broken[rep]) valid = valid && f.Step(kPotion, w, x, &w);
		CHECK_MSG(valid && rest.dead_end, "[6.5] the contract-breaking stream is not schema-valid or did not end");
	}
	CHECK_MSG(broken[0] == broken[1] && !broken[0].empty(), "[6.5] the contract-breaking stream is not deterministic");
	CHECK_MSG(broken[0] != lawful && !lawful.empty(), "[6.5] the contract-breaking stream equals the lawful one; the cell "
	          "shows nothing");
}

// 6.6 [J1] Golden hashes of (next_forced, |F|) for every F-SFD schema, and of the jump-forward stream.
void Cell_6_6() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const GoldenTable& g : kGoldenTables) {
		const RefSchema& r = f.R(g.schema);
		CHECK_MSG(r.digest == g.digest, "[6.6] %s: fsfd_ref.txt's digest %s differs from the pinned %s (the generator "
		          "or the tokenizer moved)", g.schema, r.digest.c_str(), g.digest);
		std::vector<int32_t> words;
		bool ok = true;
		for (uint32_t s = 0; s < r.st.size(); ++s) {
			int32_t nf = -9;
			uint32_t fl = 0;
			ok = ok && sfd::ForcedTableEntry(f.m1->m, f.Index(g.schema), s, &nf, &fl) == SSLM_OK;
			words.push_back(nf);
			words.push_back(static_cast<int32_t>(fl));
		}
		const std::string got = HexDigest(words);
		CHECK_MSG(ok && got == g.digest, "[6.6] %s: the engine's table digest %s, pinned %s", g.schema, got.c_str(), g.digest);
	}
	for (const GoldenStream& g : kGoldenStreams) {
		Rig rig(f.m1->m);
		const JfRun r = FreshRun(rig, f, g.schema, std::string(g.schema) == kPotion ? f.prompt_potion : f.prompt_pr,
		                         GreedyOpts(rig));
		const std::string got = HexDigest(r.stream);
		CHECK_MSG(r.stream.size() == g.length && got == g.digest, "[6.6] %s: the stream digest %s (%zu tokens), pinned %s "
		          "(%zu)", g.schema, got.c_str(), r.stream.size(), g.digest, g.length);
	}
}

// 8.1 [J1] Jump-forward with prefix adoption, and with sslm_seq_save/restore between spans: the outputs
// equal an uninterrupted run. (After the rebase onto 1.14, also with sslm_prefix_save/restore: not in
// this build, so not here.)
void Cell_8_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	std::vector<int32_t> plain;
	{
		Rig rig(f.m1->m);
		plain = FreshRun(rig, f, kPotion, f.prompt_potion, GreedyOpts(rig)).stream;
	}
	{
		Rig rig(f.m1->m);
		sslm_prefix p = nullptr;
		sslm_schema h = nullptr;
		sslm_schema_lookup(rig.m, kPotion, &h);
		CHECK_MSG(sslm_prefix_begin(rig.m, &rig.pool, &p) == SSLM_OK && p, "[8.1] prefix begin");
		if (!p) return;
		CHECK_MSG(sslm_prefix_set_schema(p, h) == SSLM_OK, "[8.1] prefix schema");
		int32_t consumed = 0;
		CHECK_MSG(sslm_prefix_prefill(rig.m, p, f.prompt_potion.data(), int32_t(f.prompt_potion.size()), 64,
		                              SSLM_SPAN_PROMPT, rig.W(), &consumed) == SSLM_OK,
		          "[8.1] prefix prefill");
		CHECK_MSG(sslm_prefix_freeze(p) == SSLM_OK, "[8.1] prefix freeze");
		sslm_seq s = rig.NewSeq(kPotion);
		if (s) {
			CHECK_MSG(sslm_seq_adopt_prefix(s, p) == SSLM_OK, "[8.1] adopt");
			const JfRun r = RunJf(rig.m, s, GreedyOpts(rig));
			CHECK_MSG(r.stream == plain && !plain.empty(), "[8.1] adopted prefix: %s, uninterrupted %s", Str(r.stream).c_str(),
			          Str(plain).c_str());
			rig.Release(s);
		}
		sslm_prefix_release(p);
	}
	{
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
		JfOpts o = GreedyOpts(rig);
		o.after = [&](sslm_seq& q, const JfStep& st) {
			if (st.st != SSLM_OK) return;
			const std::vector<uint8_t> blob = rig.Save(q);
			sslm_seq fresh = rig.Restore(blob);
			if (!fresh) return;
			rig.Release(q);
			q = fresh;
		};
		const JfRun r = RunJf(rig.m, s, o);
		CHECK_MSG(r.stream == plain && !plain.empty(), "[8.1] save/restore after every call: %s, uninterrupted %s",
		          Str(r.stream).c_str(), Str(plain).c_str());
	}
}

// 8.2 [J1] Jump-forward with the hook and bit 1, for a one-token admission (threaded) and for M >= 2
// (serial). The outputs equal the run without a hook.
void Cell_8_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (int32_t budget : {1, 64}) {
		std::vector<int32_t> plain;
		{
			Rig rig(f.m1->m);
			plain = FreshRun(rig, f, kPotion, f.prompt_potion, GreedyOpts(rig, budget)).stream;
		}
		Rig rig(f.m1->m);
		CountingHook h;
		h.max_tasks = 4;
		if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) continue;
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
		int one_row = 0, one_row_threaded = 0, multi_row = 0, multi_row_threaded = 0;
		JfOpts o = GreedyOpts(rig, budget);
		size_t calls_before = 0;
		o.after = [&](sslm_seq&, const JfStep& st) {
			const size_t now = h.Counts().size();
			if (st.kind == 0 && st.st == SSLM_OK) {
				const size_t m = st.toks.size() + (st.pending_before ? 1 : 0);
				if (m == 1) {
					++one_row;
					if (now > calls_before) ++one_row_threaded;
				} else if (m >= 2) {
					++multi_row;
					if (now > calls_before) ++multi_row_threaded;
				}
			}
			calls_before = now;
		};
		h.Clear();
		const JfRun r = RunJf(rig.m, s, o);
		CHECK_MSG(r.stream == plain && !plain.empty(), "[8.2] budget %d with the hook: %s, without %s", budget,
		          Str(r.stream).c_str(), Str(plain).c_str());
		if (budget == 1) {
			CHECK_MSG(one_row > 0 && one_row_threaded == one_row, "[8.2] %d of %d one-token admissions called run", one_row_threaded,
			          one_row);
		} else {
			CHECK_MSG(multi_row > 0 && multi_row_threaded == 0, "[8.2] %d of %d admissions of M >= 2 called run",
			          multi_row_threaded, multi_row);
		}
	}
}

// 8.3 [J1] Today's explicit SCHEMA_CONTENT prefill from a ready seed is unchanged: it admits, walks and
// counts as it does on main, and its K/V equals the verb's admission of the same tokens.
void Cell_8_3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kPotion);
	const std::vector<int32_t>& chain = r.st[r.start].chain;
	Rig rig(f.m1->m);
	sslm_seq a = rig.NewSeq(kPotion);
	sslm_seq b = rig.NewSeq(kPotion);
	if (!a || !b || !rig.PromptOk(a, f.prompt_potion) || !rig.PromptOk(b, f.prompt_potion)) return;
	CHECK_MSG(rig.ContentOk(a, chain), "[8.3] the explicit prefill");
	uint32_t walk = r.start;
	for (int32_t t : chain) f.Step(kPotion, walk, t, &walk);
	const Blob ba = rig.State(a);
	CHECK_MSG(ba.walk == walk && ba.forced == int64_t(chain.size()) && ba.ready == 1, "[8.3] explicit prefill: walk %u "
	          "(want %u), forced %lld, ready %u", ba.walk, walk, static_cast<long long>(ba.forced), ba.ready);
	const Jump j = JumpForward(rig.m, b, 64, nullptr, rig.W());
	const Blob bb = rig.State(b);
	CHECK_MSG(j.st == SSLM_OK && j.forced == chain && bb.kv == ba.kv && bb.walk == ba.walk && bb.forced == ba.forced,
	          "[8.3] the verb's admission differs from the explicit prefill (status %d)", static_cast<int>(j.st));
}

// 8.5 [J1] An adapter-bound sequence. Oracle (e) runs against a fresh sequence with the same adapter.
void Cell_8_5() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const std::vector<uint8_t> bytes = BuildAdapterBytes(f);
	CHECK_MSG(!bytes.empty(), "[8.5] the synthetic adapter did not build");
	if (bytes.empty()) return;
	std::unique_ptr<Aligned> ab = std::make_unique<Aligned>(bytes.size());
	std::memcpy(ab->p, bytes.data(), bytes.size());
	sslm_adapter ad = nullptr;
	const sslm_status as = sslm_adapter_map(ab->p, bytes.size(), f.m1->m, &ad);
	CHECK_MSG(as == SSLM_OK, "[8.5] sslm_adapter_map == %d", static_cast<int>(as));
	if (!ad) return;
	{
		const Replay rp = ReferenceReplay(f.m1->m, f, kPotion, f.prompt_potion, ad);
		const Replay base = ReferenceReplay(f.m1->m, f, kPotion, f.prompt_potion);
		CHECK_MSG(rp.ok && rp.dead_end, "[8.5] the adapter replay did not complete");
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion, ad);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			size_t ci = 0;
			int bad = 0;
			JfOpts o = GreedyOpts(rig);
			o.after = [&](sslm_seq& q, const JfStep& st) {
				if (st.kind != 1 || st.st != SSLM_OK || st.toks[0] < 0) return;
				const Blob b = rig.State(q);
				if (ci >= rp.choices.size() || st.toks[0] != rp.choices[ci] || b.kv != rp.kv_after[ci]) ++bad;
				++ci;
			};
			const JfRun r = RunJf(rig.m, s, o);
			CHECK_MSG(r.dead_end && r.stream == rp.stream && bad == 0 && ci == rp.choices.size(),
			          "[8.5] adapter-bound: stream %s, adapter replay %s, %d choices differ", Str(r.stream).c_str(),
			          Str(rp.stream).c_str(), bad);
			CHECK_MSG(rp.kv_after != base.kv_after, "[8.5] the adapter does not move the K/V rows; the cell shows nothing");
			rig.Release(s);
		}
	}
	sslm_adapter_release(ad);
}

// The SSB blob's anti-LM history and order, for 8.6.
struct AntiLm {
	int32_t order = 0;
	std::vector<int32_t> hist;
};
AntiLm ReadAntiLm(Rig& rig, sslm_seq s) {
	const Blob b = rig.State(s);
	return AntiLm{b.order, b.hist};
}

// 8.6 (rev 1.2) [J1, AL, F-SFD POTION] The anti-LM rule, read through the SSB blob's history.
void Cell_8_6() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kPotion);
	const std::vector<int32_t>& start = r.st[r.start].chain;
	// (i) Start span, damped order 3.
	{
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			const sslm_decode_params p = rig.Damped(3);
			const Jump j = JumpForward(rig.m, s, 64, &p, rig.W());
			const AntiLm a = ReadAntiLm(rig, s);
			CHECK_MSG(j.st == SSLM_OK && a.order == 3 && a.hist == start && start.size() == 3, "[8.6 (i)] status %d, order "
			          "%d, history %s, want the start span %s", static_cast<int>(j.st), a.order, Str(a.hist).c_str(),
			          Str(start).c_str());
		}
	}
	// (ii) Whole run: the same params throughout; the history equals the output so far after every call.
	{
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			const sslm_decode_params p = rig.Damped(3);
			JfOpts o;
			o.vparams = &p;
			o.dparams = p;
			o.w = rig.W();
			std::vector<int32_t> so_far;
			int bad = 0, calls = 0;
			o.after = [&](sslm_seq& q, const JfStep& st) {
				if (st.st != SSLM_OK) return;
				if (st.kind == 0) so_far.insert(so_far.end(), st.toks.begin(), st.toks.end());
				else if (st.toks[0] >= 0) so_far.push_back(st.toks[0]);
				++calls;
				if (ReadAntiLm(rig, q).hist != so_far) ++bad;
			};
			const JfRun run = RunJf(rig.m, s, o);
			CHECK_MSG(run.dead_end && bad == 0 && calls > 0 && ReadAntiLm(rig, s).hist == run.stream,
			          "[8.6 (ii)] %d of %d calls left a history that is not the output so far (end status %d)", bad, calls,
			          static_cast<int>(run.end_status));
		}
	}
	// (iii) Greedy: NULL params, then greedy params: no anti-LM state exists after the run.
	for (int arm = 0; arm < 2; ++arm) {
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
		const sslm_decode_params g = rig.Greedy();
		JfOpts o = GreedyOpts(rig);
		o.vparams = arm == 0 ? nullptr : &g;
		const JfRun run = RunJf(rig.m, s, o);
		const AntiLm a = ReadAntiLm(rig, s);
		CHECK_MSG(run.dead_end && a.order == 0 && a.hist.empty(), "[8.6 (iii)] %s params: order %d, history %zu tokens "
		          "(end status %d)", arm == 0 ? "NULL" : "greedy", a.order, a.hist.size(), static_cast<int>(run.end_status));
	}
	// (iv) Stale state: a sequence that decoded damped, then the verb with greedy params: history unchanged.
	{
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			int32_t t = -100;
			rig.Decode1(s, rig.Damped(3), rig.W(), &t);  // at the forced start: a damped decode makes a state
			const AntiLm before = ReadAntiLm(rig, s);
			const sslm_decode_params g = rig.Greedy();
			const Jump j = JumpForward(rig.m, s, 64, &g, rig.W());
			const AntiLm after = ReadAntiLm(rig, s);
			CHECK_MSG(before.order == 3 && before.hist.size() == 1, "[8.6 (iv)] setup: order %d history %zu", before.order,
			          before.hist.size());
			CHECK_MSG(j.st == SSLM_OK && j.count > 0 && after.hist == before.hist && after.order == before.order,
			          "[8.6 (iv)] the greedy verb: status %d count %d, history %s -> %s", static_cast<int>(j.st), j.count,
			          Str(before.hist).c_str(), Str(after.hist).c_str());
		}
	}
	// (v) Order change: verb at order 2, then decode at order 3. With jump-forward off the same calls
	// re-key: the decode at a new order destroys the history and starts one at its own token.
	{
		std::vector<int32_t> ref_hist;
		int32_t ref_order = 0;
		{
			Rig rig(f.m1->m);
			sslm_seq s = rig.NewSeq(kPotion);
			if (s && rig.PromptOk(s, f.prompt_potion)) {
				int32_t t = -100;
				for (size_t i = 0; i < start.size(); ++i) rig.Decode1(s, rig.Damped(2), rig.W(), &t);
				rig.Decode1(s, rig.Damped(3), rig.W(), &t);
				const AntiLm a = ReadAntiLm(rig, s);
				ref_hist = a.hist;
				ref_order = a.order;
			}
		}
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			const sslm_decode_params p2 = rig.Damped(2);
			const Jump j = JumpForward(rig.m, s, 64, &p2, rig.W());
			int32_t t = -100;
			rig.Decode1(s, rig.Damped(3), rig.W(), &t);
			const AntiLm a = ReadAntiLm(rig, s);
			CHECK_MSG(j.st == SSLM_OK && a.order == 3 && ref_order == 3 && a.hist.size() == ref_hist.size() &&
			              a.hist.size() == 1 && a.hist[0] == t,
			          "[8.6 (v)] verb@2 then decode@3: status %d, order %d, history %s; jump-forward off re-keys to %s",
			          static_cast<int>(j.st), a.order, Str(a.hist).c_str(), Str(ref_hist).c_str());
		}
	}
	// (vi) Allocation failure at the create: SSLM_ALLOCATION_FAILED, nothing admitted.
	{
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			const std::vector<uint8_t> before = rig.Save(s);
			const sslm_decode_params p = rig.Damped(3);
			superslm_test::g_inject_anti_lm_create_alloc_failure = true;
			const Jump j = JumpForward(rig.m, s, 64, &p, rig.W());
			const bool consumed = !superslm_test::g_inject_anti_lm_create_alloc_failure;
			superslm_test::g_inject_anti_lm_create_alloc_failure = false;
			CHECK_MSG(consumed && j.st == SSLM_ALLOCATION_FAILED && rig.Save(s) == before, "[8.6 (vi)] status %d (slot "
			          "consumed %d); want SSLM_ALLOCATION_FAILED and nothing admitted", static_cast<int>(j.st), consumed);
		}
	}
	// (vii) Refusal parity with sslm_decode_step_v2's parameter validation.
	{
		std::unique_ptr<Mapped> nodgc = Map(VariantWithout(f, {kSecDgc1}, superslm::kDampedGreedyArtifactConstantsFlag));
		CHECK_MSG(nodgc->st == SSLM_OK, "[8.6 (vii)] the no-DGC1 variant == %d", static_cast<int>(nodgc->st));
		Rig rig(f.m1->m);
		const sslm_decode_params good = rig.Damped(2);
		struct Bad {
			const char* what;
			sslm_decode_params p;
			bool on_nodgc;
		};
		std::vector<Bad> bads;
		auto add = [&](const char* w, std::function<void(sslm_decode_params&)> mut, bool nd = false) {
			sslm_decode_params p = good;
			mut(p);
			bads.push_back({w, p, nd});
		};
		add("struct_size", [](sslm_decode_params& p) { p.struct_size -= 4; });
		add("layer_budget 0", [](sslm_decode_params& p) { p.layer_budget = 0; });
		add("layer_budget 3", [](sslm_decode_params& p) { p.layer_budget = kLayers + 1; });
		add("mode 7", [](sslm_decode_params& p) { p.mode = 7; });
		add("alpha -1", [](sslm_decode_params& p) { p.alpha_q15 = -1; });
		add("anti_lm_max_order 0", [](sslm_decode_params& p) { p.anti_lm_max_order = 0; });
		add("top_k 0", [](sslm_decode_params& p) { p.top_k = 0; });
		add("top_k V+1", [](sslm_decode_params& p) { p.top_k = kV + 1; });
		add("damped on no DGC1", [](sslm_decode_params&) {}, true);
		// The softmax width check: a q_c the check refuses at this top_k (asserted below).
		add("softmax width", [](sslm_decode_params& p) { p.q_c = INT64_MAX / 4; p.q_b = INT64_MAX / 4; });
		add("peak (q_b = q_c = 0)", [](sslm_decode_params& p) { p.q_b = 0; p.q_c = 0; });
		add("peak (all 0)", [](sslm_decode_params& p) { p.q_ln2 = 0; p.q_b = 0; p.q_c = 0; });
		CHECK_MSG(superslm::CheckSoftmaxRowWidthDomain(INT64_MAX / 4, INT64_MAX / 4, size_t(good.top_k)) !=
		              superslm::SslmForwardStatus::Ok,
		          "[8.6 (vii)] the width arm's q is accepted by the width check; the arm would not reach it");
		for (const Bad& b : bads) {
			sslm_model m = b.on_nodgc ? nodgc->m : f.m1->m;
			if (!m) continue;
			Rig r2(m);
			sslm_seq d = r2.NewSeq(kPotion);
			sslm_seq v = r2.NewSeq(kPotion);
			if (!d || !v || !r2.PromptOk(d, f.prompt_potion) || !r2.PromptOk(v, f.prompt_potion)) continue;
			int32_t t = -100;
			const sslm_status ds = sslm_decode_step_v2(m, &d, 1, &b.p, r2.W(), &t);
			const std::vector<uint8_t> before = r2.Save(v);
			const Jump j = JumpForward(m, v, 64, &b.p, r2.W());
			CHECK_MSG(ds != SSLM_OK && j.st == ds, "[8.6 (vii)] %s: decode %d, verb %d", b.what, static_cast<int>(ds),
			          static_cast<int>(j.st));
			const Blob after = ParseBlob(r2.Save(v));
			CHECK_MSG(r2.Save(v) == before && after.hist_n == 0 && after.order == 0, "[8.6 (vii)] %s: the refused verb "
			          "changed the sequence or made an anti-LM state", b.what);
		}
	}
	// (viii) A refused call keeps the history: a sequence with an order-3 history, then a damped order-2
	// verb call (a) mid-layer, (b) at h = 0, (c) with invalid params.
	{
		for (int arm = 0; arm < 3; ++arm) {
			Rig rig(f.m1->m);
			const int32_t budget = static_cast<int32_t>(f.prompt_potion.size() + start.size());
			sslm_seq s = arm == 1 ? rig.NewBudgeted(budget, kPotion) : rig.NewSeq(kPotion);
			if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
			const sslm_decode_params p3 = rig.Damped(3);
			const Jump j0 = JumpForward(rig.m, s, 64, &p3, rig.W());
			int32_t t = -100;
			rig.Decode1(s, p3, rig.W(), &t);  // a choice, pending; history = start span + choice
			if (arm == 0) {
				// (a) The pending token is embedded by a budget-1 decode at the same order: mid-layer, no finish,
				// so the history is unchanged by it.
				sslm_decode_params p1 = p3;
				p1.layer_budget = 1;
				int32_t mid = -100;
				const sslm_status ms = rig.Decode1(s, p1, rig.W(), &mid);
				const Blob mb = rig.State(s);
				CHECK_MSG(ms == SSLM_OK && mid == -1 && mb.layer == 1, "[8.6 (viii) 0] setup: a budget-1 decode left "
				          "layer %d (status %d, out %d)", mb.layer, static_cast<int>(ms), mid);
			}
			const std::vector<uint8_t> before = rig.Save(s);
			const AntiLm a0 = ReadAntiLm(rig, s);
			sslm_decode_params p2 = rig.Damped(2);
			if (arm == 2) p2.struct_size = 0;
			const Jump j = JumpForward(rig.m, s, 64, &p2, rig.W());
			const sslm_status want = arm == 1 ? SSLM_KV_BUDGET_EXCEEDED : SSLM_INVALID_ARGUMENT;
			CHECK_MSG(j0.st == SSLM_OK && a0.order == 3 && !a0.hist.empty(), "[8.6 (viii) %d] setup: status %d, order %d",
			          arm, static_cast<int>(j0.st), a0.order);
			CHECK_MSG(j.st == want && rig.Save(s) == before, "[8.6 (viii) %d] status %d (want %d), blob unchanged %d", arm,
			          static_cast<int>(j.st), static_cast<int>(want), rig.Save(s) == before);
		}
	}
}

// 8.7 [J1] Choice followed by choice (POTION 't', forced with S-CHOICE). The verb is a no-op, the pending
// token is decoded normally, and nothing else changes.
void Cell_8_7() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const int32_t t_tok = f.TokenOf("t");
	const RefSchema& r = f.R(kPotion);
	uint32_t s_q = UINT32_MAX, s_t = UINT32_MAX;
	for (uint32_t s = 0; s < r.st.size() && s_q == UINT32_MAX; ++s) {
		uint32_t n = 0;
		if (!r.st[s].forced && !r.st[s].accepting && f.Step(kPotion, s, t_tok, &n) && !r.st[n].forced &&
		    f.Row(kPotion, n).size() == 4) {
			s_q = s;
			s_t = n;
		}
	}
	CHECK_MSG(s_q != UINT32_MAX, "[8.7] no POTION state where 't' leads to a 4-row choice state");
	if (s_q == UINT32_MAX) return;
	Rig rig(f.m1->m);
	sslm_seq s = ReachState(rig, f, kPotion, s_q, f.prompt_potion);
	if (!s) return;
	sfd::ArmChoice(t_tok);
	int32_t c = -100;
	rig.Decode1(s, rig.Greedy(), rig.W(), &c);
	sfd::DisarmChoice();
	CHECK_MSG(c == t_tok, "[8.7] S-CHOICE: decode produced %d, want 't' (%d)", c, t_tok);
	const std::vector<uint8_t> before = rig.Save(s);
	const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(j.st == SSLM_OK && j.count == 0 && rig.Save(s) == before, "[8.7] the verb at a choice->choice: status %d "
	          "count %d, blob unchanged %d", static_cast<int>(j.st), j.count, rig.Save(s) == before);
	const Blob b = ParseBlob(before);
	CHECK_MSG(b.current == t_tok && b.ready == 0 && b.walk == s_t, "[8.7] pending: current %d ready %u walk %u", b.current,
	          b.ready, b.walk);
	sfd::LayerRowsLogReset();
	int32_t next = -100;
	const sslm_status ds = rig.Decode1(s, rig.Greedy(), rig.W(), &next);
	uint32_t n = 0;
	CHECK_MSG(ds == SSLM_OK && f.Step(kPotion, s_t, next, &n) && rig.State(s).ctx == b.ctx + 1,
	          "[8.7] the next decode: status %d token %d, context %lld -> %lld", static_cast<int>(ds), next,
	          static_cast<long long>(b.ctx), static_cast<long long>(rig.State(s).ctx));
	CHECK_MSG(sfd::LayerRowsLogCount() == 1 && sfd::LayerRowsLogAt(0).kind == sfd::kLayerRowsDecode,
	          "[8.7] the pending token was not embedded by the decode's own layer loop (S-LAYERROWS %zu entries)",
	          sfd::LayerRowsLogCount());
}

// 8.8 [J1] Jump-forward from a ready seed (after the prompt), after a dead end, and after a
// ParallelForIncomplete. Nothing is embedded twice: context_length and S-LAYERROWS both show it.
void Cell_8_8() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kPotion);
	// A forced state other than the start, with a chain of two or more tokens.
	uint32_t s_f = UINT32_MAX;
	for (uint32_t s = 0; s < r.st.size(); ++s)
		if (s != r.start && r.st[s].forced && r.st[s].chain.size() >= 2) {
			s_f = s;
			break;
		}
	for (int arm = 0; arm < 3; ++arm) {
		Rig rig(f.m1->m);
		CountingHook h;
		h.max_tasks = 4;
		h.hostile = Hostile::kOmit;
		sslm_seq s = nullptr;
		uint32_t at = r.start;
		if (arm == 0) {
			s = rig.NewSeq(kPotion);
			if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
		} else {
			at = s_f;
			s = ReachState(rig, f, kPotion, s_f, f.prompt_potion);
			if (!s) continue;
			int32_t t = -100;
			if (arm == 1) {
#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
				ArmCpuFinishDegenerateLogitRowInjection();
				rig.Decode1(s, rig.Greedy(), rig.W(), &t);
				CHECK_MSG(t == -2, "[8.8] setup: the degenerate row did not dead-end (%d)", t);
#else
				CHECK_MSG(false, "[8.8] the suite's library lacks SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION");
#endif
			} else {
				Install(rig.W(), &h, 0);
				const sslm_status st = rig.Decode1(s, rig.Greedy(), rig.W(), &t);
				CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "[8.8] setup: the hostile finish gave %d", static_cast<int>(st));
				Install(rig.W(), nullptr, 0);
			}
		}
		const Blob before = rig.State(s);
		sfd::LayerRowsLogReset();
		const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
		const Blob after = rig.State(s);
		const size_t k = r.st[at].chain.size();
		CHECK_MSG(before.ready == 1, "[8.8 arm %d] setup: not resting ready for logits", arm);
		CHECK_MSG(j.st == SSLM_OK && j.forced == r.st[at].chain && after.ctx == before.ctx + int64_t(k),
		          "[8.8 arm %d] status %d, admitted %zu, context %lld -> %lld (want +%zu)", arm, static_cast<int>(j.st),
		          j.forced.size(), static_cast<long long>(before.ctx), static_cast<long long>(after.ctx), k);
		CHECK_MSG(sfd::LayerRowsLogCount() == 1 && sfd::LayerRowsLogAt(0).kind == sfd::kLayerRowsBatched &&
		              sfd::LayerRowsLogAt(0).rows == k,
		          "[8.8 arm %d] S-LAYERROWS: %zu entries, first %llu rows; want one batched pass of %zu", arm,
		          sfd::LayerRowsLogCount(), static_cast<unsigned long long>(sfd::LayerRowsLogAt(0).rows), k);
	}
}

// 9.1 [J1] Restore each resting shape (§5.7), both ways between main's paths and the J1 verbs.
void Cell_9_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kPotion);
	const std::vector<int32_t>& start = r.st[r.start].chain;
	Rig rig(f.m1->m, 1, 12);  // five originals and five restores
	// (a) A pending embed saved by main mid-generation (prefill + decode) continues under jump-forward.
	{
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion) && rig.ContentOk(s, start)) {
			int32_t c = -100;
			rig.Decode1(s, rig.Greedy(), rig.W(), &c);
			const Blob b = rig.State(s);
			sslm_seq q = rig.Restore(rig.Save(s));
			if (q) {
				const Jump j = JumpForward(rig.m, q, 64, nullptr, rig.W());
				CHECK_MSG(b.current == c && b.ready == 0 && j.st == SSLM_OK && j.forced == r.st[b.walk].chain,
				          "[9.1 pending] status %d forced %s, reference %s", static_cast<int>(j.st), Str(j.forced).c_str(),
				          Str(r.st[b.walk].chain).c_str());
			}
		}
	}
	// (b) Ready for logits at a forced state (a partial explicit prefill).
	{
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion) && rig.ContentOk(s, {start[0]})) {
			const Blob b = rig.State(s);
			sslm_seq q = rig.Restore(rig.Save(s));
			if (q) {
				const Jump j = JumpForward(rig.m, q, 64, nullptr, rig.W());
				CHECK_MSG(j.st == SSLM_OK && j.forced == r.st[b.walk].chain, "[9.1 ready, forced] status %d forced %s",
				          static_cast<int>(j.st), Str(j.forced).c_str());
			}
		}
	}
	// (c) Ready for logits at an unforced state: the verb is a no-op and decode continues as without restore.
	{
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion) && rig.ContentOk(s, start)) {
			sslm_seq q = rig.Restore(rig.Save(s));
			if (q) {
				const Jump j = JumpForward(rig.m, q, 64, nullptr, rig.W());
				int32_t a = -100, b = -100;
				rig.Decode1(s, rig.Greedy(), rig.W(), &a);
				rig.Decode1(q, rig.Greedy(), rig.W(), &b);
				CHECK_MSG(j.st == SSLM_OK && j.count == 0 && a == b, "[9.1 ready, unforced] status %d count %d, decode %d vs %d",
				          static_cast<int>(j.st), j.count, a, b);
			}
		}
	}
	// (d) Mid-layer: the verb refuses.
	{
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion) && rig.ContentOk(s, start)) {
			int32_t c = -100, mid = -100;
			rig.Decode1(s, rig.Greedy(), rig.W(), &c);
			rig.Decode1(s, rig.Greedy(1), rig.W(), &mid);
			sslm_seq q = rig.Restore(rig.Save(s));
			if (q) {
				const Jump j = JumpForward(rig.m, q, 64, nullptr, rig.W());
				CHECK_MSG(mid == -1 && j.st == SSLM_INVALID_ARGUMENT, "[9.1 mid-layer] setup out %d, verb status %d", mid,
				          static_cast<int>(j.st));
			}
		}
	}
	// (e) A blob saved right after a verb call restores and continues by plain decode, as the original does.
	{
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
			CHECK_MSG(j.st == SSLM_OK && j.forced == start, "[9.1 after a verb] status %d", static_cast<int>(j.st));
			sslm_seq q = rig.Restore(rig.Save(s));
			if (q) {
				std::vector<int32_t> a, b;
				for (int i = 0; i < 6; ++i) {
					int32_t x = -100, y = -100;
					rig.Decode1(s, rig.Greedy(), rig.W(), &x);
					rig.Decode1(q, rig.Greedy(), rig.W(), &y);
					a.push_back(x);
					b.push_back(y);
				}
				CHECK_MSG(a == b, "[9.1 after a verb] restored %s, original %s", Str(b).c_str(), Str(a).c_str());
			}
		}
	}
}

std::string ReadRepoFile(const char* rel) {
	static const char* kPrefixes[] = {"", "../", "../../", "../../../"};
	for (const char* p : kPrefixes) {
		std::vector<uint8_t> b;
		if (superslm_marshal::ReadFile((std::string(p) + rel).c_str(), b)) return std::string(b.begin(), b.end());
	}
#if defined(SFD_SOURCE_DIR)
	std::vector<uint8_t> b;
	if (superslm_marshal::ReadFile((std::string(SFD_SOURCE_DIR) + "/" + rel).c_str(), b))
		return std::string(b.begin(), b.end());
#endif
	return {};
}

// 9.2 [J1] The ABI inventory and allowlist carry the new verbs, the version is bumped, and the existing
// ordinals are unchanged.
void Cell_9_2() {
	const std::string inc = ReadRepoFile("include/superslm/sslm_abi_functions.inc");
	const std::string verbs = ReadRepoFile("tests/t2807-api-slot/c_abi_verbs.txt");
	CHECK_MSG(!inc.empty() && !verbs.empty(), "[9.2] the inventory files were not found from the working directory");
	for (const char* v : {"sslm_seq_forced_tokens", "sslm_seq_jump_forward"}) {
		CHECK_MSG(inc.find(std::string(v) + "(") != std::string::npos, "[9.2] sslm_abi_functions.inc does not declare %s", v);
		CHECK_MSG(verbs.find(std::string("\n") + v + "\n") != std::string::npos, "[9.2] c_abi_verbs.txt does not list %s", v);
	}
	// The version is CMakeLists.txt's project() VERSION, read from the file (no build defines it for the
	// suite, so CMake and build.bat agree). Main is 1.13.1; the plan names no target, only a bump.
	const std::string cml = ReadRepoFile("CMakeLists.txt");
	const size_t at = cml.find("project(superslm VERSION ");
	std::string version;
	if (at != std::string::npos) {
		const size_t b = at + std::strlen("project(superslm VERSION ");
		version = cml.substr(b, cml.find_first_of(" )", b) - b);
	}
	CHECK_MSG(!version.empty() && version != "1.13.1", "[9.2] the project version is still %s",
	          version.empty() ? "(CMakeLists.txt not found)" : version.c_str());
	// No new status (§5.3): the closed enum's last ordinal is unchanged.
	CHECK_MSG(static_cast<int>(SSLM_KV_BUDGET_EXCEEDED) == 29 && static_cast<int>(SSLM_ALLOCATION_FAILED) == 25 &&
	              static_cast<int>(SSLM_SCHEMA_SPAN_UNREACHABLE) == 23 && static_cast<int>(SSLM_ARTIFACT_REJECTED) == 4,
	          "[9.2] a status ordinal moved");
}

// 10.2 [J1] The read-only query (audit F8).
void Cell_10_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kLongKey);
	const std::vector<int32_t>& chain = r.st[r.start].chain;
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kLongKey);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const std::vector<uint8_t> before = rig.Save(s);
	const Query q5 = QueryForced(rig.m, s, 5);
	CHECK_MSG(q5.st == SSLM_OK && q5.count == int32_t(chain.size()) &&
	              q5.ids == std::vector<int32_t>(chain.begin(), chain.begin() + 5),
	          "[10.2] cap 5: status %d count %d ids %s", static_cast<int>(q5.st), q5.count, Str(q5.ids).c_str());
	int32_t count = -9, acc = -9;
	const sslm_status z = sslm_seq_forced_tokens(rig.m, s, nullptr, 0, &count, &acc);
	CHECK_MSG(z == SSLM_OK && count == int32_t(chain.size()) && acc == 0, "[10.2] cap 0, null ids: status %d count %d "
	          "accepting %d", static_cast<int>(z), count, acc);
	CHECK_MSG(rig.Save(s) == before, "[10.2] the query changed the save blob");
	CHECK_MSG(rig.TryBind(s, kPotion) == SSLM_OK, "[10.2] the query cleared bind eligibility");
	// Unbound: 0.
	sslm_seq u = rig.NewSeq(nullptr);
	if (u) {
		const Query qu = QueryForced(rig.m, u);
		CHECK_MSG(qu.st == SSLM_OK && qu.count == 0 && qu.accepting == 0, "[10.2] unbound: status %d count %d accepting %d",
		          static_cast<int>(qu.st), qu.count, qu.accepting);
	}
	// Unforced (a choice) and accepting.
	const uint32_t s_choice = FirstChoiceState(f, kPotion);
	const uint32_t s_acc = AcceptingState(f, kPotion);
	for (uint32_t st : {s_choice, s_acc}) {
		sslm_seq q = ReachState(rig, f, kPotion, st, f.prompt_potion);
		if (!q) continue;
		const Query qq = QueryForced(rig.m, q);
		CHECK_MSG(qq.st == SSLM_OK && qq.count == 0 && qq.accepting == (st == s_acc ? 1 : 0), "[10.2] state %u: status %d "
		          "count %d accepting %d", st, static_cast<int>(qq.st), qq.count, qq.accepting);
		rig.Release(q);
	}
}

// 11.2 [J1] The pre-scan still rejects an illegal forced token when the table is corrupted through a seam.
void Cell_11_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	std::unique_ptr<Mapped> m = Map(f.raw);  // its own copy: the corruption must not reach other cells
	if (!m->m) return;
	const RefSchema& r = f.R(kPotion);
	const int32_t illegal = f.byte_tok['!'];
	uint32_t n = 0;
	CHECK_MSG(!f.Step(kPotion, r.start, illegal, &n), "[11.2] '!' is legal at POTION's start; pick another token");
	const sslm_status cs = sfd::CorruptForcedTableEntry(m->m, f.Index(kPotion), r.start, illegal);
	CHECK_MSG(cs == SSLM_OK, "[11.2] the corruption seam == %d", static_cast<int>(cs));
	Rig rig(m->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const std::vector<uint8_t> before = rig.Save(s);
	const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(j.st == SSLM_SCHEMA_SPAN_UNREACHABLE && j.count == 0 && rig.Save(s) == before,
	          "[11.2] a corrupted chain: status %d count %d (want SSLM_SCHEMA_SPAN_UNREACHABLE, nothing admitted)",
	          static_cast<int>(j.st), j.count);
}

// 11.3 [J1] With prewalked_prefix = 1, S-WALK records zero walk steps for the pending token.
void Cell_11_3() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kPotion);
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	sfd::WalkCountReset();
	const Jump j0 = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(j0.st == SSLM_OK && sfd::WalkCount() == r.st[r.start].chain.size(), "[11.3] the start span (no pending): "
	          "%llu walk steps, want %zu", static_cast<unsigned long long>(sfd::WalkCount()), r.st[r.start].chain.size());
	int32_t c = -100;
	rig.Decode1(s, rig.Greedy(), rig.W(), &c);
	const uint32_t w = rig.State(s).walk;
	sfd::WalkCountReset();
	const Jump j1 = JumpForward(rig.m, s, 64, nullptr, rig.W());
	CHECK_MSG(j1.st == SSLM_OK && j1.count > 0 && sfd::WalkCount() == uint64_t(j1.count) &&
	              j1.forced == r.st[w < r.st.size() ? w : 0].chain,
	          "[11.3] pending + %d forced: %llu walk steps, want %d (none for the pending token)", j1.count,
	          static_cast<unsigned long long>(sfd::WalkCount()), j1.count);
}

// Prints the values cell 6.6 pins, computed by the reference replay on main (SUPERSLM_SFD_PRINT_GOLDENS=1).
void PrintGoldens() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const char* schema : {kPotion, kPromptResult}) {
		const Replay rp = ReferenceReplay(f.m1->m, f, schema, schema == kPotion ? f.prompt_potion : f.prompt_pr);
		std::printf("SFD GOLDEN stream %s: ok %d dead_end %d capped %d length %zu choices %zu digest %s\n  stream %s\n", schema,
		            rp.ok, rp.dead_end, rp.capped, rp.stream.size(), rp.choices.size(), HexDigest(rp.stream).c_str(),
		            Str(rp.stream, 400).c_str());
	}
}

}  // namespace

void RunSchemaFastDecodeCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	const char* only = std::getenv("SUPERSLM_SFD_ONLY");  // a cell id, e.g. "10.1" (diagnostic runs only)
	auto run = [&](const char* id, void (*fn)()) {
		if (only && *only && std::string(only) != id) return;
		const int before = GFailures;
		const auto t0 = std::chrono::steady_clock::now();
		fn();
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		std::printf("[schema fast-decode] cell %-5s %s (%.0f ms)\n", id, GFailures == before ? "green" : "RED", ms);
	};
	const char* print = std::getenv("SUPERSLM_SFD_PRINT_GOLDENS");
	if (print && *print == '1') PrintGoldens();
	{
		const Sfd& f = GetSfd();
		if (f.ok) CheckReferenceMatchesArtifact(f);
	}
	// M1
	run("2.4", Cell_2_4);
	run("3.1", Cell_3_1);
	run("4.1", Cell_4_1);
	run("5.4", Cell_5_4);
	run("5.5", Cell_5_5);
	run("6.1", Cell_6_1);
	run("6.3", Cell_6_3);
	run("8.4", Cell_8_4);
	run("11.1", Cell_11_1);
	run("11.4", Cell_11_4);
	// J1
	run("1.1", Cell_1_1);
	run("1.2", Cell_1_2);
	run("1.3", Cell_1_3);
	run("2.1", Cell_2_1);
	run("2.2", Cell_2_2);
	run("2.3", Cell_2_3);
	run("2.5", Cell_2_5);
	run("2.6", Cell_2_6);
	run("3.2", Cell_3_2);
	run("3.3", Cell_3_3);
	run("4.2", Cell_4_2);
	run("5.1", Cell_5_1);
	run("5.2", Cell_5_2);
	run("5.3", Cell_5_3);
	run("5.6", Cell_5_6);
	run("6.2", Cell_6_2);
	run("6.4", Cell_6_4);
	run("6.5", Cell_6_5);
	run("6.6", Cell_6_6);
	run("8.1", Cell_8_1);
	run("8.2", Cell_8_2);
	run("8.3", Cell_8_3);
	run("8.5", Cell_8_5);
	run("8.6", Cell_8_6);
	run("8.7", Cell_8_7);
	run("8.8", Cell_8_8);
	run("9.1", Cell_9_1);
	run("9.2", Cell_9_2);
	run("10.1", Cell_10_1);
	run("10.2", Cell_10_2);
	run("11.2", Cell_11_2);
	run("11.3", Cell_11_3);
	std::printf("schema fast-decode (M1, J1) cells: %d checks, %d failures\n", GChecks.load(), GFailures.load());
	checks += GChecks;
	failures += GFailures;
}
