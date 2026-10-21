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
// The surface these cells call is declared in tests/schema_fast_decode/sfd_api.h and defined by M1 and
// J1 in the library (the red round's stubs were removed when the slices landed).

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
#include "../src/forward/parallel_split.h"  // the minimum-work seam (cells 3.3, 8.2)

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>  // VirtualAlloc: cell 2.X2's guarded draft
#else
#include <sys/mman.h>  // mmap: cell 2.X2's guarded draft
#endif

#if !defined(SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS)
#error "test_schema_fast_decode.cpp needs a library built with SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS"
#endif

#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
extern "C" void ArmCpuFinishDegenerateLogitRowInjection();
#endif

// The paged-KV test seams the J1-X cells read (src/sslm_abi.cpp; tests/paged-kv/pkv_abi.h declares the same).
// The pin exists only where the library defines the paged-KV seam macro (build.bat's list does not; the
// J1-X cells then run their page-size-2 arms at F-SFD's own page size and say so, as test_slm18x does).
#if defined(SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS)
extern "C" sslm_status KvPagesTestPinPagePositions(sslm_model model, int64_t page_positions);
#endif
extern "C" sslm_status sslm_pkv_test_only_seq_table_entry(sslm_seq seq, uint32_t i, uint32_t* out_page,
                                                         uint32_t* out_mapped);
extern "C" sslm_status sslm_pkv_test_only_peek_page_bytes(sslm_kv_pool pool, uint32_t page_index, uint8_t* out_buf,
                                                          size_t n);
extern "C" sslm_status sslm_pkv_test_only_drain_reserve(sslm_seq seq);

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

// Cell 9.1 (fix round 1, audit F-9): the SSB blob `main` saves after the POTION prompt, the start chain by
// explicit SCHEMA_CONTENT prefill, and one greedy decode (a pending choice), on F-SFD. Computed on main
// (SUPERSLM_SFD_PRINT_GOLDENS=1); the J1 build's blob for the same calls must equal it byte for byte, which
// is constraint C8 (the save format does not change) with a cell that can fail inside one binary.
// Cell 6.5 (rev 1.2.2, X14's kill): the contract-breaking POTION stream (one start-chain token, then a
// decode at the forced state it leaves, then the §5.6 loop), from main's reference replay with that head.
constexpr size_t kBrokenStreamLength = 23;
constexpr const char* kBrokenStreamDigest = "ad0a0c70c0e1d0ffd8e8093a7855b9c81e859185a98839278f8ceb852e7e97d3";
constexpr size_t kMainBlobSize = 32992;
constexpr const char* kMainBlobDigest = "e03fa1092d1f5d99b63535ac96022071828b2a446672f66c83169573242c10d8";

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

std::string BytesDigest(const std::vector<uint8_t>& b) {
	superslm::Sha256 h;
	if (!b.empty()) h.Update(b.data(), b.size());
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

	explicit Rig(sslm_model model, int workspaces = 1, uint32_t blocks = 8, int32_t max_batch = 3,
	             int32_t max_chunk = 64)
	    : m(model) {
		if (!m) return;
		const size_t size = blocks * sslm_kv_block_size(m) + sslm_kv_pool_overhead_size(m, blocks);
		pool_buf = std::make_unique<Aligned>(size);
		const sslm_status ps = sslm_kv_pool_create(m, pool_buf->p, size, blocks, &pool);
		CHECK_MSG(ps == SSLM_OK, "sslm_kv_pool_create == %d", static_cast<int>(ps));
		ok = ps == SSLM_OK;
		const sslm_config cfg{max_batch, max_chunk, kLayers, 0u};
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
	j.st = sfd::JumpCanonical(m, s, budget, params, w, buf.data(), c, &j.count);
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
	bool embedded_pending = false;  // verb: this call embedded the pending token (read from the blob)
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

// The sequence's resting shape, read from its SSB blob: a pending token (decoded, not yet embedded) is
// layer 0, not ready for logits, current_token >= 0 (§5.4). Fix round 1 (audit F-1, F-2): the loop reads
// this from the engine instead of tracking it, because a no-op verb call at an unforced state leaves the
// pending token pending (§5.2, §5.3; cell 8.7), and a budget-1 call may admit the pending token alone.
struct Shape {
	bool ok = false;
	bool pending = false;
	int64_t ctx = 0;
};
Shape ReadShape(sslm_model m, sslm_seq s) {
	std::vector<uint8_t> blob(sslm_seq_state_size(m) + 4096);
	size_t n = blob.size();
	Shape sh;
	if (sslm_seq_save(s, blob.data(), &n) != SSLM_OK) return sh;
	blob.resize(n);
	const Blob b = ParseBlob(blob);
	sh.ok = b.ok;
	sh.pending = b.ok && b.layer == 0 && b.ready == 0 && b.current >= 0;
	sh.ctx = b.ctx;
	return sh;
}

// The §5.6 loop: call the verb until the query reports no chain, then decode; repeat until decode's -2
// or a non-OK status. The prompt has already been prefilled. A verb call made progress when it admitted
// a forced token or embedded the pending token (the blob's pending flag cleared, context_length rose);
// a call that made none while the query still reports a chain is "stuck". A decode is choice -> choice
// when the pending token was still pending when it ran (it embedded that token itself).
JfRun RunJf(sslm_model m, sslm_seq& s, const JfOpts& o) {
	JfRun r;
	for (int d = 0; d < o.max_decodes; ++d) {
		for (int guard = 0; guard < 1000; ++guard) {
			const Shape before = ReadShape(m, s);
			JfStep step;
			step.kind = 0;
			step.pending_before = before.pending;
			const Jump j = JumpForward(m, s, o.budget, o.vparams, o.w);
			step.st = j.st;
			step.toks = j.forced;
			const Shape after = ReadShape(m, s);
			const bool embedded_pending = before.pending && !after.pending && after.ctx > before.ctx;
			step.embedded_pending = embedded_pending;
			r.steps.push_back(step);
			++r.verb_calls;
			r.stream.insert(r.stream.end(), j.forced.begin(), j.forced.end());
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
			if (j.forced.empty() && !embedded_pending) {
				r.stuck = true;
				r.end_status = SSLM_INVALID_ARGUMENT;
				return r;
			}
		}
		JfStep step;
		step.kind = 1;
		step.pending_before = ReadShape(m, s).pending;
		int32_t out = -100;
		step.st = sslm_decode_step_v2(m, &s, 1, &o.dparams, o.w, &out);
		step.toks = {out};
		r.steps.push_back(step);
		++r.decodes;
		if (step.pending_before) ++r.choice_choice;
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
                       sslm_adapter adapter = nullptr, int max_choices = 400,
                       const std::vector<int32_t>& head = {}) {
	// `head` (cell 6.5): tokens admitted as content before the first decode, in place of the start chain;
	// the first decode then runs wherever they leave the walk (a forced state, for the contract violation).
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
	if (head.empty()) {
		if (!add_chain()) return rp;
	} else {
		for (int32_t t : head) {
			rp.stream.push_back(t);
			if (!f.Step(schema, s, t, &s)) return rp;
		}
	}
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
	bool hostile_persistent = false;  // hostile on every call from hostile_call on (J1-X cell 5.X3)
	bool threads = false;  // run each honest call's tasks on their own threads (the Q6C cells, under TSan)
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
		if (h.hostile != Hostile::kNone && (h.hostile_persistent ? call >= h.hostile_call : call == h.hostile_call)) {
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
		if (h.threads) {
			std::vector<std::thread> workers;
			for (int32_t i = 0; i < task_count; ++i) workers.emplace_back([task, task_ctx, i] { task(task_ctx, i); });
			for (std::thread& t : workers) t.join();
			return;
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
	CHECK_MSG(superslm::LegalRowFinishExact(131071), "[2.4] the predicate is not set at hidden 131,071");
	CHECK_MSG(!superslm::LegalRowFinishExact(131072), "[2.4] the predicate is set at hidden 131,072 (want clear)");
	CHECK_MSG(superslm::LegalRowFinishExact(kHidden), "[2.4] the predicate is not set at F-SFD's hidden 64");
	CHECK_MSG(superslm::LegalRowFinishExact(896), "[2.4] the predicate is not set at the 0.5B's hidden 896");
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
		// Fix round 1 (audit F-4, reading 4 rejected for damped greedy): the full-finish arm is the same build,
		// so a defect in the shared damped masker moves both arms together. Two anchors outside the build:
		// (1) at every step whose reference state is not accepting, the produced token is legal there by the
		// reference table (an unmasked or poisoned row picks an illegal candidate, and the walk misses: -2);
		// (2) the post-mask rows (S-ROW) of the poisoned legal-row arm and the full arm are byte-equal, so a
		// masked position left holding workspace poison, rather than INT32_MIN, differs from the full row.
		int bad = 0, bad_rows = 0, legal_ok = 0, bad_srow = 0, bad_legal = 0, informative = 0;
		for (int step = 0; step < 64; ++step) {
			const uint32_t walk = m1.State(b).walk;
			const size_t n = f.Row(schema, walk).size();
			const bool accepting = walk < f.R(schema).st.size() && f.R(schema).st[walk].accepting;
			if (n < 32) ++legal_ok;
			int32_t ta = -100, tb = -100;
			sfd::LastFinishRowReset();
			const sslm_status sa = full.Decode1(a, pa, full.W(), &ta);
			const std::vector<int32_t> row_a = SRow();
			m1.Poison(0, 0x63u + uint32_t(step));
			sfd::RowsLogReset();
			sfd::LastFinishRowReset();
			const sslm_status sb = m1.Decode1(b, pb, m1.W(), &tb);
			const std::vector<int32_t> row_b = SRow();
			if (!(sfd::RowsLogCount() == 1 && LastRows() == n)) ++bad_rows;
			if (row_a.empty() || row_a != row_b) ++bad_srow;
			if (!accepting) {
				++informative;
				uint32_t nx = 0;
				if (sa != SSLM_OK || ta < 0 || !f.Step(schema, walk, ta, &nx)) ++bad_legal;
				if (sb != SSLM_OK || tb < 0 || !f.Step(schema, walk, tb, &nx)) ++bad_legal;
			}
			if (sa != sb || ta != tb || full.Save(a) != m1.Save(b)) ++bad;
			if (sa != SSLM_OK) break;
		}
		CHECK_MSG(legal_ok > 0, "[6.3] %s: no step had fewer legal tokens than top_k 32", schema);
		CHECK_MSG(informative > 0, "[6.3] %s: no step was at a non-accepting state", schema);
		CHECK_MSG(bad_rows == 0, "[6.3] %s: %d of 64 steps did not run the legal-row finish (S-ROWS != n)", schema, bad_rows);
		CHECK_MSG(bad_srow == 0, "[6.3] %s: %d of 64 steps: the post-mask row (S-ROW) of the poisoned legal-row arm differs "
		          "from the full finish's (or no row was read)", schema, bad_srow);
		CHECK_MSG(bad_legal == 0, "[6.3] %s: %d tokens at non-accepting states are not legal by the reference table "
		          "(%d informative steps)", schema, bad_legal, informative);
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

// 10.1 [J1] The feature oracle, (a) through (e) (§5.9), with forced_token_count asserted, and (f) (rev 1.2.2) the
// verb's admission at every forced state (X19 with (a)).
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
	// (f) (rev 1.2.2; audit F-10) the verb's own admission at every forced state of every schema, the 4
	// fallback states included: ids equal the reference chain, and context_length rises by its length.
	int bad_f = 0, forced_states = 0;
	for (const auto& [name, r] : f.ref) {
		Rig rig(f.m1->m);
		for (uint32_t st = 0; st < r.st.size(); ++st) {
			const RefState& rs = r.st[st];
			if (!rs.forced || rs.chain.empty()) continue;
			++forced_states;
			sslm_seq s = ReachState(rig, f, name.c_str(), st, f.prompt_potion);
			if (!s) {
				++bad_f;
				continue;
			}
			const int64_t ctx0 = rig.State(s).ctx;
			const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
			const Blob b = rig.State(s);
			if (j.st != SSLM_OK || j.forced != rs.chain || b.ctx != ctx0 + int64_t(rs.chain.size())) ++bad_f;
			rig.Release(s);
		}
	}
	CHECK_MSG(forced_states > 0 && bad_f == 0, "[10.1 (f)] %d of %d forced states: the verb's admission differs from the "
	          "reference chain (ids or context_length)", bad_f, forced_states);
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
	CHECK_MSG(nf[1] == -1 && fl[1] == 0 && nf[2] == -1 && fl[2] == 0, "[2.1] the cycle's states are not both unforced: "
	          "(%d, %u), (%d, %u), want (-1, 0) twice", nf[1], fl[1], nf[2], fl[2]);
	Rig rig(m->m);
	sslm_seq s = rig.NewSeq("h_cycle");
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	JfOpts o = GreedyOpts(rig);
	o.max_decodes = 1;
	const JfRun r = RunJf(rig.m, s, o);
	CHECK_MSG(r.stream.size() == 2 && !r.stream.empty() && r.stream[0] == h.tok_brace && r.decodes == 1,
	          "[2.1] the loop: stream %s after %d decodes; want '{' then one decode", Str(r.stream).c_str(), r.decodes);
	// The decode's choice is pending, but the cycle's states are unforced, so this call is a no-op (§5.3: an
	// unforced state admits nothing and the pending token stays for decode_step); so is the next one.
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
		    sfd::JumpCanonical(rig.m, s, c.budget, nullptr, rig.W(), buf, c.cap, c.null_count ? nullptr : &count);
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "[2.2] %s: status %d, want SSLM_INVALID_ARGUMENT", c.what, static_cast<int>(st));
		CHECK_MSG(rig.Save(s) == before, "[2.2] %s changed the sequence", c.what);
	}
	sslm_seq fresh = rig.NewSeq(kPotion);
	if (!fresh) return;
	const sslm_status st = sfd::JumpCanonical(rig.m, fresh, 0, nullptr, rig.W(), buf, 64, &count);
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

// 2.7 (rev 1.2.2) [J1] The read-only query's refusals (§5.3): NULL model, NULL seq, NULL out_count, NULL
// out_accepting, cap < 0, NULL ids with cap > 0 and a released seq give SSLM_INVALID_ARGUMENT; a model with
// no table gives SSLM_ARTIFACT_REJECTED. The output buffers (pre-filled with a sentinel) are unwritten, and
// the SSB blob is byte-equal before and after.
constexpr int32_t kSent = -0x5F5F;  // cell 2.7's output sentinel

void Cell_2_7() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	struct Out {
		std::vector<int32_t> ids = std::vector<int32_t>(8, kSent);
		int32_t count = kSent, accepting = kSent;
		bool Untouched() const {
			for (int32_t x : ids)
				if (x != kSent) return false;
			return count == kSent && accepting == kSent;
		}
	};
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const std::vector<uint8_t> before = rig.Save(s);
	struct Case {
		const char* what;
		int which;
	};
	const Case cases[] = {{"NULL model", 0}, {"NULL seq", 1},      {"NULL out_count", 2},      {"NULL out_accepting", 3},
	                      {"cap < 0", 4},    {"NULL ids, cap 8", 5}};
	for (const Case& c : cases) {
		Out o;
		sslm_status st = SSLM_OK;
		switch (c.which) {
			case 0: st = sslm_seq_forced_tokens(nullptr, s, o.ids.data(), 8, &o.count, &o.accepting); break;
			case 1: st = sslm_seq_forced_tokens(rig.m, nullptr, o.ids.data(), 8, &o.count, &o.accepting); break;
			case 2: st = sslm_seq_forced_tokens(rig.m, s, o.ids.data(), 8, nullptr, &o.accepting); break;
			case 3: st = sslm_seq_forced_tokens(rig.m, s, o.ids.data(), 8, &o.count, nullptr); break;
			case 4: st = sslm_seq_forced_tokens(rig.m, s, o.ids.data(), -1, &o.count, &o.accepting); break;
			default: st = sslm_seq_forced_tokens(rig.m, s, nullptr, 8, &o.count, &o.accepting); break;
		}
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT && o.Untouched() && rig.Save(s) == before,
		          "[2.7] %s: status %d (want SSLM_INVALID_ARGUMENT), outputs untouched %d, blob unchanged %d", c.what,
		          static_cast<int>(st), o.Untouched(), rig.Save(s) == before);
	}
	// A released seq: not live in the registry. The handle is never dereferenced by a conforming build.
	{
		sslm_seq dead = rig.NewSeq(kPotion);
		if (dead) {
			rig.Release(dead);
			Out o;
			const sslm_status st = sslm_seq_forced_tokens(rig.m, dead, o.ids.data(), 8, &o.count, &o.accepting);
			CHECK_MSG(st == SSLM_INVALID_ARGUMENT && o.Untouched(), "[2.7] a released seq: status %d, outputs untouched %d",
			          static_cast<int>(st), o.Untouched());
		}
	}
	// A model with no table (no TOK1).
	{
		std::unique_ptr<Mapped> m = Map(VariantWithout(f, {kSecTokenizer, kSecUnicode}, 0));
		CHECK_MSG(m->st == SSLM_OK, "[2.7] map without TOK1 == %d", static_cast<int>(m->st));
		if (m->m) {
			Rig r2(m->m);
			sslm_seq q = r2.NewSeq(kPotion);
			if (q && r2.PromptOk(q, f.prompt_potion)) {
				const std::vector<uint8_t> b0 = r2.Save(q);
				Out o;
				const sslm_status st = sslm_seq_forced_tokens(r2.m, q, o.ids.data(), 8, &o.count, &o.accepting);
				CHECK_MSG(st == SSLM_ARTIFACT_REJECTED && o.Untouched() && r2.Save(q) == b0, "[2.7] no table: status %d "
				          "(want SSLM_ARTIFACT_REJECTED), outputs untouched %d", static_cast<int>(st), o.Untouched());
			}
		}
	}
}

// 3.4 (rev 1.2.2) [M1, TSan leg] M1's legal-list split on real worker threads: a hook that runs each
// call's tasks concurrently (one per thread, the caller taking task 0), at max_tasks 4 and 8, at
// PROMPT_RESULT's 146,994-row state and at POTION's 16-row choice state. The produced token and the S-ROW
// bytes equal serial's; on the TSan leg an overlapping or mis-bounded split is a reported race.
struct ThreadHook {
	int32_t max_tasks = 4;
	std::atomic<int> calls{0};
	sslm_parallel_for Pf() { return sslm_parallel_for{&Run, this, max_tasks, 0u}; }
	static void Run(void* ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		ThreadHook& h = *static_cast<ThreadHook*>(ctx);
		h.calls.fetch_add(1);
		std::vector<std::thread> workers;
		for (int32_t i = 1; i < task_count; ++i) workers.emplace_back([task, task_ctx, i] { task(task_ctx, i); });
		if (task_count > 0) task(task_ctx, 0);
		for (std::thread& t : workers) t.join();
	}
};

void Cell_3_4() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	size_t big = 0;
	const uint32_t s_big = LargestRowState(f, kPromptResult, &big);
	const uint32_t s_choice = FirstChoiceState(f, kPotion);
	struct Case {
		const char* schema;
		uint32_t state;
	};
	const Case cases[] = {{kPromptResult, s_big}, {kPotion, s_choice}};
	for (const Case& c : cases) {
		const size_t n = f.Row(c.schema, c.state).size();
		int32_t want_tok = -100;
		std::vector<int32_t> want_row;
		{
			Rig ref(f.m1->m);
			sslm_seq r = ReachState(ref, f, c.schema, c.state, f.prompt_potion);
			sfd::LastFinishRowReset();
			if (!r || ref.Decode1(r, ref.Greedy(), ref.W(), &want_tok) != SSLM_OK) {
				CHECK_MSG(false, "[3.4] the serial decode at %s state %u failed", c.schema, c.state);
				continue;
			}
			want_row = SRow();
		}
		for (int32_t mt : {4, 8}) {
			Rig rig(f.m1->m);
			ThreadHook h;
			h.max_tasks = mt;
			const sslm_parallel_for pf = h.Pf();
			if (!rig.ok || sslm_workspace_set_parallel_for(rig.W(), &pf) != SSLM_OK) continue;
			sslm_seq s = ReachState(rig, f, c.schema, c.state, f.prompt_potion);
			if (!s) continue;
			sfd::RowsLogReset();
			sfd::LastFinishRowReset();
			int32_t tok = -100;
			const sslm_status st = rig.Decode1(s, rig.Greedy(), rig.W(), &tok);
			const std::vector<int32_t> row = SRow();
			CHECK_MSG(st == SSLM_OK && tok == want_tok, "[3.4] %s state %u (n %zu) max_tasks %d: status %d token %d, "
			          "serial %d", c.schema, c.state, n, mt, static_cast<int>(st), tok, want_tok);
			CHECK_MSG(sfd::RowsLogCount() == 1 && LastRows() == n, "[3.4] %s state %u max_tasks %d: the threaded finish "
			          "was not the legal-row finish (S-ROWS %llu, want n = %zu)", c.schema, c.state, mt,
			          static_cast<unsigned long long>(LastRows()), n);
			CHECK_MSG(!row.empty() && row == want_row, "[3.4] %s state %u max_tasks %d: the threaded S-ROW differs from "
			          "serial's (rows read %zu / %zu)", c.schema, c.state, mt, row.size(), want_row.size());
			CHECK_MSG(h.calls.load() >= 1, "[3.4] %s state %u max_tasks %d: the hook's run was never called", c.schema,
			          c.state, mt);
		}
	}
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
// The decode-threading minimum-work seam (src/forward/parallel_split.h), lowered for one scope. Fix round 2:
// on F-SFD's small projections (hidden 64), a one-token admission never reaches the default 256 KiB per
// task, so the M = 1 matvec split never calls `run` (plan §5.3: a one-token prefill threads exactly as a
// decode step does). Cells 3.3 and 8.2 observe that split, so they lower the minimum inside their own
// scope only; every other cell keeps the default.
struct MinWorkScope {
	explicit MinWorkScope(size_t bytes) { superslm::test::SetMinRowBytesPerTask(bytes); }
	~MinWorkScope() { superslm::test::ResetMinRowBytesPerTask(); }
	MinWorkScope(const MinWorkScope&) = delete;
	MinWorkScope& operator=(const MinWorkScope&) = delete;
};
constexpr size_t kSmallMinWork = 1024;  // at most 4 KiB: every F-SFD M = 1 projection then threads

void Cell_3_3() {
	const MinWorkScope min_work(kSmallMinWork);
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
		// Fix round 1 (audit N-6): explicit SCHEMA_CONTENT prefill also adds to forced_token_count, so the
		// whole blob compares, not the K/V region alone.
		CHECK_MSG(rig.Save(a) == rig.Save(b), "[4.2] span of %d: the blob differs from token-by-token prefill's", m);
	}
	// Fix round 1 (audit F-6; §5.3): the two lawful call shapes no other cell makes. The LONGKEY start span
	// (26 tokens) admitted (a) with ws NULL and (b) at budget 65 on a workspace whose max_chunk_budget is 8
	// (the heap fallback; rev 1's refusal is withdrawn) gives status OK, the reference chain, and a blob
	// byte-equal to the in-workspace budget-64 admission's.
	{
		std::vector<uint8_t> want;
		{
			Rig rig(f.m1->m);
			sslm_seq s = rig.NewSeq(kLongKey);
			if (s && rig.PromptOk(s, f.prompt_potion)) {
				const Jump j = JumpForward(rig.m, s, 64, nullptr, rig.W());
				if (j.st == SSLM_OK && j.forced == chain) want = rig.Save(s);
			}
		}
		CHECK_MSG(!want.empty(), "[4.2 reference] the in-workspace budget-64 admission of the LONGKEY start span failed");
		{
			Rig rig(f.m1->m);
			sslm_seq s = rig.NewSeq(kLongKey);
			if (s && rig.PromptOk(s, f.prompt_potion)) {
				const Jump j = JumpForward(rig.m, s, 64, nullptr, nullptr);
				CHECK_MSG(j.st == SSLM_OK && j.forced == chain && !want.empty() && rig.Save(s) == want,
				          "[4.2 ws NULL] status %d, admitted %zu (want %zu), blob equal %d", static_cast<int>(j.st),
				          j.forced.size(), chain.size(), !want.empty() && rig.Save(s) == want);
			}
		}
		{
			Rig rig(f.m1->m, 1, 8, 3, 8);
			sslm_seq s = rig.NewSeq(kLongKey);
			if (rig.ok && s && rig.PromptOk(s, f.prompt_potion)) {
				const Jump j = JumpForward(rig.m, s, 65, nullptr, rig.W());
				CHECK_MSG(j.st == SSLM_OK && j.forced == chain && !want.empty() && rig.Save(s) == want,
				          "[4.2 budget 65 > max_chunk_budget 8] status %d, admitted %zu (want %zu), blob equal %d",
				          static_cast<int>(j.st), j.forced.size(), chain.size(), !want.empty() && rig.Save(s) == want);
			}
		}
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
		// Fix round 1 (audit F-5): k is derived from the run, through the reference walk (the start chain,
		// then the decode's choice), never from the engine's walk. The sweep runs h = 0 .. 10, which spans
		// k - 1, k and k + 1 for every k F-SFD's POTION gives here (6 on main); each h is classified by its
		// own k, and the rows h = k - 1, h = k and h = k + 1 must all have run on both holders.
		bool saw_below = false, saw_exact = false, saw_above = false;
		for (int h = 0; h <= 10; ++h) {  // the headroom at the pending choice
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
			uint32_t ref_walk = r.start;
			bool walked = true;
			for (int32_t tok : r.st[r.start].chain) walked = walked && f.Step(kPotion, ref_walk, tok, &ref_walk);
			walked = walked && f.Step(kPotion, ref_walk, c, &ref_walk);
			CHECK_MSG(walked && ref_walk < r.st.size(), "[5.1 %s h %d] the choice %d is not legal on the reference walk",
			          hname, h, c);
			if (!walked || ref_walk >= r.st.size()) continue;
			const RefState& next = r.st[ref_walk];
			const size_t k = 1 + next.chain.size();
			if (h > 0 && size_t(h) + 1 == k) saw_below = true;
			if (size_t(h) == k) saw_exact = true;
			if (size_t(h) == k + 1) saw_above = true;
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
				uint32_t walk = ref_walk;
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
		CHECK_MSG(saw_below && saw_exact && saw_above, "[5.1 %s] the sweep missed a boundary row: h = k - 1 %d, h = k %d, "
		          "h = k + 1 %d", hname, saw_below, saw_exact, saw_above);
	}
	// Fix round 3 (code review round 1, finding 4): h = 0 with NO pending token. A first verb call with one
	// token of headroom admits the start chain's first forced token (a partial case, 1 <= h < k), so the
	// sequence rests ready for logits with h = 0 and nothing pending. A second verb call admits nothing,
	// returns the limit status and changes nothing. §5.5's partial rows apply ("the next decode_step runs the
	// finish only, with no re-embed"): the finish needs no row, so the next decode returns SSLM_OK with a
	// token and context_length does not rise. (§5.5's h = 0 row says the next decode returns "today's status
	// for that limit"; with no pending token today's decode at a ready sequence is SSLM_OK. docs/api.md's
	// "returns the same" limit status holds only when a token is pending.)
	for (int holder = 0; holder < 2; ++holder) {
		const sslm_status limit_st = holder == 0 ? SSLM_CONTEXT_CAP_EXCEEDED : SSLM_KV_BUDGET_EXCEEDED;
		const char* hname = holder == 0 ? "cap holder" : "KV-budget holder";
		Rig rig(f.m1->m);
		std::vector<int32_t> prompt;
		int32_t budget = 0;
		if (holder == 0) {
			for (size_t i = 0; i + 1 < size_t(kCap); ++i) prompt.push_back(f.prompt_potion[i % f.prompt_potion.size()]);
		} else {
			prompt = f.prompt_potion;
			budget = static_cast<int32_t>(prompt.size()) + 1;
		}
		sslm_seq s = holder == 0 ? rig.NewSeq(kPotion) : rig.NewBudgeted(budget, kPotion);
		if (!s || !rig.PromptOk(s, prompt)) continue;
		const Jump j1 = JumpForward(rig.m, s, 64, nullptr, rig.W());
		const Blob b1 = rig.State(s);
		CHECK_MSG(j1.st == limit_st && j1.forced.size() == 1 && j1.forced[0] == r.st[r.start].chain[0] && b1.ready == 1,
		          "[5.1 %s h 0, no pending] setup: status %d, forced %s, ready %u; want the limit status, the start "
		          "chain's first token and ready for logits", hname, static_cast<int>(j1.st), Str(j1.forced).c_str(),
		          b1.ready);
		const std::vector<uint8_t> blob_before = rig.Save(s);
		const Jump j2 = JumpForward(rig.m, s, 64, nullptr, rig.W());
		CHECK_MSG(j2.st == limit_st && j2.count == 0 && rig.Save(s) == blob_before, "[5.1 %s h 0, no pending] status %d "
		          "count %d; want the limit status, nothing admitted and nothing changed", hname, static_cast<int>(j2.st),
		          j2.count);
		int32_t t = -100;
		const sslm_status ds = rig.Decode1(s, rig.Greedy(), rig.W(), &t);
		const Blob b2 = rig.State(s);
		CHECK_MSG(ds == SSLM_OK && t >= 0 && t < kV && b2.ctx == b1.ctx, "[5.1 %s h 0, no pending] the next decode: "
		          "status %d token %d, context_length %lld -> %lld; want SSLM_OK, a token, and no rise", hname,
		          static_cast<int>(ds), t, static_cast<long long>(b1.ctx), static_cast<long long>(b2.ctx));
	}
}

// Fix round 3 (code review round 1, findings 1-3): a verb failure AFTER the anti-LM re-key. The slots of the
// first arms (cells 5.2 and 8.6 (vi) below) fire before EnsureAntiLm touches the state, and on a sequence
// with no prior state, so they cannot see a re-key that a later failure leaves behind. Here the sequence
// first holds a damped state at order 3 with a one-token history (the verb at order 3, budget 1, admits the
// start chain's first token); the failing call is the verb at damped order 2, budget 1, which re-keys
// before its pass. The failure fires at one of:
//   - the cell's own slot (kPassSlot for 5.2, kCreateSlot for 8.6 (vi)), now on the prior-state sequence;
//   - the layer-site seam inside the batched pass (src/forward/parallel_split.h): a std::bad_alloc at the q
//     projection of layer 0 and at the down projection of the last layer (after K/V landing), and a domain
//     rejection at the gate projection of the last layer. These are the named equivalents of the
//     allocations the review's probe failed (#27, #31, #40, all inside the pass after the re-key);
//   - a hostile hook that omits a task (kOmit) under the lowered minimum-work seam, so the pass's one-row
//     split reaches the hook and fails with SSLM_INVALID_ARGUMENT.
// Plan §5.8 and cell 5.2: a failed call changes nothing (order, history and the whole save blob), and a
// retry gives the same result as a sequence that never failed. The real allocation failure inside
// AntiLmCreate (finding 2, a dangling pointer) needs a replaced global operator new, so it is swept in its
// own executable, tests/schema_fast_decode/sfd_alloc_main.cpp; the save and release that follow every arm
// here are what LSan/ASan read.
enum class Rekey { kPassSlot, kCreateSlot, kQBadAlloc, kDownBadAlloc, kGateUpStatus, kHostileHook };

const char* RekeyName(Rekey k) {
	switch (k) {
	case Rekey::kPassSlot: return "pass slot";
	case Rekey::kCreateSlot: return "create slot";
	case Rekey::kQBadAlloc: return "bad_alloc at layer 0 q";
	case Rekey::kDownBadAlloc: return "bad_alloc at the last layer's down";
	case Rekey::kGateUpStatus: return "domain rejection at the last layer's gate";
	case Rekey::kHostileHook: return "hostile hook (omit)";
	}
	return "?";
}

void PriorStateFailure(const char* cell, Rekey fault) {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const char* name = RekeyName(fault);
	// The reference: the same two calls with no failure.
	std::vector<uint8_t> ref_after;
	{
		Rig ref(f.m1->m);
		sslm_seq s = ref.NewSeq(kPotion);
		if (!s || !ref.PromptOk(s, f.prompt_potion)) return;
		const sslm_decode_params p3 = ref.Damped(3), p2 = ref.Damped(2);
		const Jump a = JumpForward(ref.m, s, 1, &p3, ref.W());
		const Jump b = JumpForward(ref.m, s, 1, &p2, ref.W());
		CHECK_MSG(a.st == SSLM_OK && a.count == 1 && b.st == SSLM_OK && b.count == 1, "[%s, %s] the reference: verb@3 "
		          "status %d count %d, verb@2 status %d count %d", cell, name, static_cast<int>(a.st), a.count,
		          static_cast<int>(b.st), b.count);
		ref_after = ref.Save(s);
	}
	Rig rig(f.m1->m);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion)) return;
	const sslm_decode_params p3 = rig.Damped(3), p2 = rig.Damped(2);
	const Jump setup = JumpForward(rig.m, s, 1, &p3, rig.W());
	const std::vector<uint8_t> before = rig.Save(s);
	const Blob b0 = ParseBlob(before);
	CHECK_MSG(setup.st == SSLM_OK && setup.count == 1 && b0.order == 3 && b0.hist.size() == 1, "[%s, %s] setup: verb@3 "
	          "status %d count %d, order %d, history %s", cell, name, static_cast<int>(setup.st), setup.count, b0.order,
	          Str(b0.hist).c_str());
	if (setup.st != SSLM_OK || b0.order != 3 || b0.hist.size() != 1) return;
	sslm_status want = SSLM_ALLOCATION_FAILED;
	CountingHook h;
	h.max_tasks = 4;
	h.hostile = Hostile::kOmit;
	std::unique_ptr<MinWorkScope> min_work;
	switch (fault) {
	case Rekey::kPassSlot: superslm_test::g_inject_jump_forward_pass_alloc_failure = true; break;
	case Rekey::kCreateSlot: superslm_test::g_inject_anti_lm_create_alloc_failure = true; break;
	case Rekey::kQBadAlloc:
		superslm::test::ArmLayerSiteFault(superslm::MatvecFaultSite::kQ, 0, superslm::MatvecFaultKind::kBadAlloc);
		break;
	case Rekey::kDownBadAlloc:
		superslm::test::ArmLayerSiteFault(superslm::MatvecFaultSite::kDown, static_cast<uint32_t>(kLayers - 1),
		                                  superslm::MatvecFaultKind::kBadAlloc);
		break;
	case Rekey::kGateUpStatus:
		superslm::test::ArmLayerSiteFault(superslm::MatvecFaultSite::kGateUp, static_cast<uint32_t>(kLayers - 1),
		                                  superslm::MatvecFaultKind::kNonPfiStatus);
		want = SSLM_ARTIFACT_REJECTED;
		break;
	case Rekey::kHostileHook:
		min_work = std::make_unique<MinWorkScope>(kSmallMinWork);
		if (!Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) return;
		want = SSLM_INVALID_ARGUMENT;
		break;
	}
	const Jump j = JumpForward(rig.m, s, 1, &p2, rig.W());
	bool fired = true;
	switch (fault) {
	case Rekey::kPassSlot:
		fired = !superslm_test::g_inject_jump_forward_pass_alloc_failure;
		superslm_test::g_inject_jump_forward_pass_alloc_failure = false;
		break;
	case Rekey::kCreateSlot:
		fired = !superslm_test::g_inject_anti_lm_create_alloc_failure;
		superslm_test::g_inject_anti_lm_create_alloc_failure = false;
		break;
	case Rekey::kQBadAlloc:
	case Rekey::kDownBadAlloc:
	case Rekey::kGateUpStatus: superslm::test::DisarmLayerSiteFault(); break;
	case Rekey::kHostileHook:
		fired = !h.Counts().empty();
		Install(rig.W(), nullptr, 0);
		min_work.reset();
		break;
	}
	CHECK_MSG(fired && j.st == want && j.count == 0, "[%s, %s] verb@2 on an order-3 state: status %d count %d (fired %d), "
	          "want status %d and nothing admitted", cell, name, static_cast<int>(j.st), j.count, fired,
	          static_cast<int>(want));
	const std::vector<uint8_t> after = rig.Save(s);
	const Blob b1 = ParseBlob(after);
	CHECK_MSG(b1.order == b0.order, "[%s, %s] the failed call changed the anti-LM order %d -> %d", cell, name, b0.order,
	          b1.order);
	CHECK_MSG(b1.hist == b0.hist, "[%s, %s] the failed call changed the anti-LM history %s -> %s", cell, name,
	          Str(b0.hist).c_str(), Str(b1.hist).c_str());
	CHECK_MSG(after == before, "[%s, %s] the failed call changed the save blob", cell, name);
	const Jump retry = JumpForward(rig.m, s, 1, &p2, rig.W());
	CHECK_MSG(retry.st == SSLM_OK && retry.count == 1 && rig.Save(s) == ref_after, "[%s, %s] the retry: status %d count "
	          "%d, blob %s the never-failed sequence's", cell, name, static_cast<int>(retry.st), retry.count,
	          rig.Save(s) == ref_after ? "equals" : "differs from");
	(void)rig.Save(s);
	rig.Release(s);
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
	// Fix round 3 (review finding 3): the same failure after the re-key, on a prior damped state.
	for (Rekey k : {Rekey::kPassSlot, Rekey::kQBadAlloc, Rekey::kDownBadAlloc, Rekey::kGateUpStatus, Rekey::kHostileHook})
		PriorStateFailure("5.2", k);
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
	// Rev 1.2.2: the pinned stream. The violation moves the walk mid-chain, so a verb that carried a chain
	// remainder across calls (X14) admits an illegal token next and cannot produce this stream.
	CHECK_MSG(broken[0].size() == kBrokenStreamLength && HexDigest(broken[0]) == kBrokenStreamDigest,
	          "[6.5] the contract-breaking stream %s (%zu tokens, digest %s); main's replay pins %zu, %s",
	          Str(broken[0]).c_str(), broken[0].size(), HexDigest(broken[0]).c_str(), kBrokenStreamLength,
	          kBrokenStreamDigest);
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
// §16 Q6 C (k = 7, box rule 3.10): an admission of M = 2..7 rows now runs as one-row threaded passes, so
// only M >= 8 stays serial; the budget-64 arm asserts both halves (the Q6C cells cover the rule itself).
void Cell_8_2() {
	const MinWorkScope min_work(kSmallMinWork);
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (int32_t budget : {1, 64}) {
		std::vector<int32_t> plain;
		{
			Rig rig(f.m1->m);
			const JfRun pr = FreshRun(rig, f, kPotion, f.prompt_potion, GreedyOpts(rig, budget));
			plain = pr.stream;
			// Fix round 1 (audit F-2): the run covers the whole stream, to the dead end, at every budget.
			CHECK_MSG(pr.dead_end, "[8.2] budget %d without the hook: the run ended with status %d (stuck %d) after %zu "
			          "tokens", budget, static_cast<int>(pr.end_status), pr.stuck, pr.stream.size());
		}
		Rig rig(f.m1->m);
		CountingHook h;
		h.max_tasks = 4;
		if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) continue;
		sslm_seq s = rig.NewSeq(kPotion);
		if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
		int one_row = 0, one_row_threaded = 0, multi_row = 0, multi_row_threaded = 0, wide_row = 0, wide_row_threaded = 0;
		JfOpts o = GreedyOpts(rig, budget);
		size_t calls_before = 0;
		o.after = [&](sslm_seq&, const JfStep& st) {
			const size_t now = h.Counts().size();
			if (st.kind == 0 && st.st == SSLM_OK) {
				const size_t m = st.toks.size() + (st.embedded_pending ? 1 : 0);
				if (m == 1) {
					++one_row;
					if (now > calls_before) ++one_row_threaded;
				} else if (m >= 2 && m <= 7) {
					++multi_row;
					if (now > calls_before) ++multi_row_threaded;
				} else if (m >= 8) {
					++wide_row;
					if (now > calls_before) ++wide_row_threaded;
				}
			}
			calls_before = now;
		};
		h.Clear();
		const JfRun r = RunJf(rig.m, s, o);
		CHECK_MSG(r.dead_end, "[8.2] budget %d with the hook: the run ended with status %d (stuck %d)", budget,
		          static_cast<int>(r.end_status), r.stuck);
		CHECK_MSG(r.stream == plain && !plain.empty(), "[8.2] budget %d with the hook: %s, without %s", budget,
		          Str(r.stream).c_str(), Str(plain).c_str());
		if (budget == 1) {
			CHECK_MSG(one_row > 0 && one_row_threaded == one_row, "[8.2] %d of %d one-token admissions called run", one_row_threaded,
			          one_row);
		} else {
			CHECK_MSG(multi_row > 0 && multi_row_threaded == multi_row, "[8.2] %d of %d admissions of M = 2..7 called run "
			          "(Q6 C: all of them)", multi_row_threaded, multi_row);
			CHECK_MSG(wide_row_threaded == 0, "[8.2] %d of %d admissions of M >= 8 called run", wide_row_threaded, wide_row);
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
	// Fix round 1 (audit N-6): both admissions add to forced_token_count, so the whole blob compares.
	CHECK_MSG(rig.Save(a) == rig.Save(b), "[8.3] the blob after the verb differs from the explicit prefill's");
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

// A POTION choice -> choice construction (cells 8.6 (viii) e, 8.7): a non-accepting unforced state s_q
// where 't' leads to an unforced 4-row choice state s_t. Returns 't''s id; s_q stays UINT32_MAX if none.
int32_t FindChoiceChoice(const Sfd& f, uint32_t* s_q, uint32_t* s_t) {
	const int32_t t_tok = f.TokenOf("t");
	const RefSchema& r = f.R(kPotion);
	for (uint32_t s = 0; s < r.st.size() && *s_q == UINT32_MAX; ++s) {
		uint32_t n = 0;
		if (!r.st[s].forced && !r.st[s].accepting && f.Step(kPotion, s, t_tok, &n) && !r.st[n].forced &&
		    f.Row(kPotion, n).size() == 4) {
			*s_q = s;
			*s_t = n;
		}
	}
	return t_tok;
}

// 8.6 (rev 1.2, arms through (ix) at rev 1.2.2) [J1, AL, F-SFD POTION] The anti-LM rule, read through the SSB
// blob's history. X23d maps to (ix) with LSan.
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
	// (vi), fix round 3 (review findings 2-3): on a sequence that already holds a damped state at order 3,
	// the create slot and the failures after the re-key. A real failing AntiLmCreate (the dangling pointer of
	// finding 2) is swept by tests/schema_fast_decode/sfd_alloc_main.cpp.
	for (Rekey k : {Rekey::kCreateSlot, Rekey::kQBadAlloc, Rekey::kDownBadAlloc}) PriorStateFailure("8.6 (vi)", k);
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
	// (viii) (d), (e) (fix round 1, audit F-8; §5.8 "Order"): the two admitting-or-validated calls that must
	// never touch the history either. On an order-3 history, a damped order-2 call that (d) admits only the
	// pending token (h = 1 on a KV-budget holder: the limit status), or (e) is a no-op (choice -> choice at
	// an unforced state, cell 8.7's construction). Order and history are byte-equal to before.
	{
		Rig rig(f.m1->m);
		const int32_t budget = static_cast<int32_t>(f.prompt_potion.size() + start.size()) + 1;
		sslm_seq s = rig.NewBudgeted(budget, kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			const sslm_decode_params p3 = rig.Damped(3);
			const Jump j0 = JumpForward(rig.m, s, 64, &p3, rig.W());
			int32_t t = -100;
			rig.Decode1(s, p3, rig.W(), &t);  // a choice, pending, with one token of headroom
			const AntiLm a0 = ReadAntiLm(rig, s);
			const Blob b0 = rig.State(s);
			const sslm_decode_params p2 = rig.Damped(2);
			const Jump j = JumpForward(rig.m, s, 64, &p2, rig.W());
			const AntiLm a1 = ReadAntiLm(rig, s);
			const Blob b1 = rig.State(s);
			CHECK_MSG(j0.st == SSLM_OK && a0.order == 3 && !a0.hist.empty(), "[8.6 (viii) d] setup: status %d, order %d",
			          static_cast<int>(j0.st), a0.order);
			CHECK_MSG(j.st == SSLM_KV_BUDGET_EXCEEDED && j.count == 0 && b1.ctx == b0.ctx + 1 && b1.ready == 1,
			          "[8.6 (viii) d] h = 1: status %d count %d, context %lld -> %lld, ready %u; want the limit status and "
			          "the pending token alone admitted", static_cast<int>(j.st), j.count, static_cast<long long>(b0.ctx),
			          static_cast<long long>(b1.ctx), b1.ready);
			CHECK_MSG(a1.order == a0.order && a1.hist == a0.hist, "[8.6 (viii) d] the pending-only call changed the "
			          "history: order %d -> %d, %s -> %s", a0.order, a1.order, Str(a0.hist).c_str(), Str(a1.hist).c_str());
		}
	}
	{
		uint32_t s_q = UINT32_MAX, s_t = UINT32_MAX;
		const int32_t t_tok = FindChoiceChoice(f, &s_q, &s_t);
		CHECK_MSG(s_q != UINT32_MAX, "[8.6 (viii) e] no POTION choice -> choice construction");
		Rig rig(f.m1->m);
		sslm_seq s = s_q == UINT32_MAX ? nullptr : ReachState(rig, f, kPotion, s_q, f.prompt_potion);
		if (s) {
			sfd::ArmChoice(t_tok);
			int32_t c = -100;
			rig.Decode1(s, rig.Damped(3), rig.W(), &c);  // creates the order-3 history; 't' pending
			sfd::DisarmChoice();
			const AntiLm a0 = ReadAntiLm(rig, s);
			const std::vector<uint8_t> before = rig.Save(s);
			const sslm_decode_params p2 = rig.Damped(2);
			const Jump j = JumpForward(rig.m, s, 64, &p2, rig.W());
			const AntiLm a1 = ReadAntiLm(rig, s);
			CHECK_MSG(c == t_tok && a0.order == 3 && !a0.hist.empty(), "[8.6 (viii) e] setup: S-CHOICE gave %d (want %d), "
			          "order %d", c, t_tok, a0.order);
			CHECK_MSG(j.st == SSLM_OK && j.count == 0 && a1.order == a0.order && a1.hist == a0.hist &&
			              rig.Save(s) == before,
			          "[8.6 (viii) e] the no-op call: status %d count %d, order %d -> %d, blob unchanged %d",
			          static_cast<int>(j.st), j.count, a0.order, a1.order, rig.Save(s) == before);
		}
	}
	// (ix) (rev 1.2.2; audit F-7) The verb's own re-key: a damped decode at order 3 creates a history (the
	// start span's first token, now pending), then the verb at damped order 2 admits the pending token and
	// the chain. The state is at order 2 and the history is exactly the verb's forced tokens. Reference:
	// the same calls with the verb replaced by order-2 decodes, which re-key the same way (main's code).
	// LSan on the ASan leg catches an old state re-keyed without being destroyed (X23d).
	{
		uint32_t walk = r.start;
		int32_t first = -1;
		std::vector<int32_t> want;
		{
			Rig ref(f.m1->m);
			sslm_seq s = ref.NewSeq(kPotion);
			if (s && ref.PromptOk(s, f.prompt_potion)) {
				ref.Decode1(s, ref.Damped(3), ref.W(), &first);
				if (first >= 0 && f.Step(kPotion, walk, first, &walk) && walk < r.st.size()) want = r.st[walk].chain;
			}
		}
		std::vector<int32_t> ref_hist;
		int32_t ref_order = 0;
		{
			Rig ref(f.m1->m);
			sslm_seq s = ref.NewSeq(kPotion);
			if (s && ref.PromptOk(s, f.prompt_potion)) {
				int32_t t = -100;
				ref.Decode1(s, ref.Damped(3), ref.W(), &t);
				for (size_t i = 0; i < want.size(); ++i) ref.Decode1(s, ref.Damped(2), ref.W(), &t);
				const AntiLm a = ReadAntiLm(ref, s);
				ref_hist = a.hist;
				ref_order = a.order;
			}
		}
		CHECK_MSG(!want.empty() && ref_order == 2 && ref_hist.size() == want.size(), "[8.6 (ix)] reference: chain %s, "
		          "jump-forward off re-keys to order %d history %s", Str(want).c_str(), ref_order, Str(ref_hist).c_str());
		Rig rig(f.m1->m);
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && rig.PromptOk(s, f.prompt_potion)) {
			int32_t t = -100;
			rig.Decode1(s, rig.Damped(3), rig.W(), &t);
			const AntiLm a0 = ReadAntiLm(rig, s);
			const sslm_decode_params p2 = rig.Damped(2);
			const Jump j = JumpForward(rig.m, s, 64, &p2, rig.W());
			const AntiLm a = ReadAntiLm(rig, s);
			CHECK_MSG(t == first && a0.order == 3 && a0.hist.size() == 1, "[8.6 (ix)] setup: decode@3 gave %d (want %d), "
			          "order %d history %s", t, first, a0.order, Str(a0.hist).c_str());
			CHECK_MSG(j.st == SSLM_OK && j.forced == want && a.order == 2 && a.hist == j.forced &&
			              a.order == ref_order && a.hist.size() == ref_hist.size(),
			          "[8.6 (ix)] verb@2 after decode@3: status %d forced %s, order %d, history %s; want order 2 and the "
			          "history = the forced tokens %s", static_cast<int>(j.st), Str(j.forced).c_str(), a.order,
			          Str(a.hist).c_str(), Str(want).c_str());
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
// ParallelForIncomplete. Nothing is embedded twice: context_length and S-LAYERROWS both show it. Rev 1.2.2: the
// next decode is finish-only (X13 with 10.1 (b)).
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
		// Rev 1.2.2 (X13's kill): after the admission the sequence rests ready for logits, so the next decode
		// runs a finish only: no S-LAYERROWS decode entry and no context_length rise.
		CHECK_MSG(after.ready == 1, "[8.8 arm %d] after the admission: ready_for_logits %u", arm, after.ready);
		sfd::LayerRowsLogReset();
		int32_t nt = -100;
		const sslm_status ns = rig.Decode1(s, rig.Greedy(), rig.W(), &nt);
		CHECK_MSG(ns == SSLM_OK && sfd::LayerRowsLogCount() == 0 && rig.State(s).ctx == after.ctx,
		          "[8.8 arm %d] the next decode: status %d, %zu S-LAYERROWS entries, context %lld -> %lld; want a "
		          "finish only", arm, static_cast<int>(ns), sfd::LayerRowsLogCount(), static_cast<long long>(after.ctx),
		          static_cast<long long>(rig.State(s).ctx));
	}
}

// The calls whose blob kMainBlobDigest pins: prompt, start chain by explicit prefill, one greedy decode.
std::vector<uint8_t> MainShapeBlob(Rig& rig, const Sfd& f) {
	const RefSchema& r = f.R(kPotion);
	sslm_seq s = rig.NewSeq(kPotion);
	if (!s || !rig.PromptOk(s, f.prompt_potion) || !rig.ContentOk(s, r.st[r.start].chain)) return {};
	int32_t c = -100;
	if (rig.Decode1(s, rig.Greedy(), rig.W(), &c) != SSLM_OK || c < 0) return {};
	std::vector<uint8_t> b = rig.Save(s);
	rig.Release(s);
	return b;
}

// 9.1 [J1] Restore each resting shape (§5.7), both ways between main's paths and the J1 verbs.
void Cell_9_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kPotion);
	const std::vector<int32_t>& start = r.st[r.start].chain;
	Rig rig(f.m1->m, 1, 14);  // five originals, five restores, and the (f)/(g) arms
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
	// (f) Fix round 1 (audit F-9): main's blob, pinned. The same calls on this build save the same bytes.
	{
		const std::vector<uint8_t> b = MainShapeBlob(rig, f);
		CHECK_MSG(!b.empty() && b.size() == kMainBlobSize && BytesDigest(b) == kMainBlobDigest, "[9.1 main blob] %zu bytes, "
		          "digest %s; main saved %zu bytes, %s", b.size(), BytesDigest(b).c_str(), kMainBlobSize, kMainBlobDigest);
	}
	// (g) A blob saved right after a verb call equals the blob of the same tokens admitted by explicit prefill
	// (both add to forced_token_count, so the whole blob compares), which main can restore.
	{
		sslm_seq a = rig.NewSeq(kPotion);
		sslm_seq b = rig.NewSeq(kPotion);
		if (a && b && rig.PromptOk(a, f.prompt_potion) && rig.PromptOk(b, f.prompt_potion) && rig.ContentOk(b, start)) {
			const Jump j = JumpForward(rig.m, a, 64, nullptr, rig.W());
			CHECK_MSG(j.st == SSLM_OK && j.forced == start && rig.Save(a) == rig.Save(b), "[9.1 verb blob] status %d forced "
			          "%s; the blob after the verb differs from explicit prefill's", static_cast<int>(j.st),
			          Str(j.forced).c_str());
		}
		if (a) rig.Release(a);
		if (b) rig.Release(b);
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

// A repository text file, read for the inventory cells (9.2, 9.X2), with every '\r' removed: the cells search it
// line by line ("\n" + verb + "\n", a line compared whole), and a CRLF checkout left each line ending in '\r' so
// no verb ever matched (MSVC ctest, 2026-10-21). Every caller only searches the text; none hashes its bytes.
static std::string RepoText(const std::vector<uint8_t>& b) {
	std::string s;
	s.reserve(b.size());
	for (const uint8_t c : b)
		if (c != '\r') s.push_back(static_cast<char>(c));
	return s;
}

std::string ReadRepoFile(const char* rel) {
	static const char* kPrefixes[] = {"", "../", "../../", "../../../"};
	for (const char* p : kPrefixes) {
		std::vector<uint8_t> b;
		if (superslm_marshal::ReadFile((std::string(p) + rel).c_str(), b)) return RepoText(b);
	}
#if defined(SFD_SOURCE_DIR)
	std::vector<uint8_t> b;
	if (superslm_marshal::ReadFile((std::string(SFD_SOURCE_DIR) + "/" + rel).c_str(), b)) return RepoText(b);
#endif
	return {};
}

// 9.2 [J1] The ABI inventory and allowlist carry the new verbs, the version header and CMake agree on a released
// version, and the existing ordinals are unchanged.
void Cell_9_2() {
	const std::string inc = ReadRepoFile("include/superslm/sslm_abi_functions.inc");
	const std::string verbs = ReadRepoFile("tests/t2807-api-slot/c_abi_verbs.txt");
	CHECK_MSG(!inc.empty() && !verbs.empty(), "[9.2] the inventory files were not found from the working directory");
	for (const char* v : {"sslm_seq_forced_tokens", "sslm_seq_jump_forward"}) {
		CHECK_MSG(inc.find(std::string(v) + "(") != std::string::npos, "[9.2] sslm_abi_functions.inc does not declare %s", v);
		CHECK_MSG(verbs.find(std::string("\n") + v + "\n") != std::string::npos, "[9.2] c_abi_verbs.txt does not list %s", v);
	}
	// The version (conductor ruling): a development build carries the last released version until its release
	// commit, and the bump is the release commit's check, not this cell's. So 9.2 asserts what
	// tests/ci/check_version_header.py enforces -- sslm_version.h's MAJOR/MINOR/PATCH equal CMakeLists.txt's
	// project() VERSION -- and that the value is a released one: CHANGELOG.md carries a dated "## [x.y.z] - "
	// heading for it.
	const std::string cml = ReadRepoFile("CMakeLists.txt");
	const size_t at = cml.find("project(superslm VERSION ");
	std::string version;
	if (at != std::string::npos) {
		const size_t b = at + std::strlen("project(superslm VERSION ");
		version = cml.substr(b, cml.find_first_of(" )", b) - b);
	}
	CHECK_MSG(!version.empty(), "[9.2] CMakeLists.txt's project(superslm VERSION ...) was not found");
	{
		const std::string hdr = ReadRepoFile("include/superslm/sslm_version.h");
		std::string from_header;
		for (const char* part : {"MAJOR", "MINOR", "PATCH"}) {
			const std::string key = std::string("#define SUPERSLM_VERSION_") + part + " ";
			const size_t k = hdr.find(key);
			std::string n;
			if (k != std::string::npos)
				for (size_t i = k + key.size(); i < hdr.size() && hdr[i] >= '0' && hdr[i] <= '9'; ++i) n += hdr[i];
			from_header += (from_header.empty() ? "" : ".") + (n.empty() ? std::string("?") : n);
		}
		CHECK_MSG(!hdr.empty() && from_header == version, "[9.2] sslm_version.h says %s, CMakeLists.txt says %s",
		          hdr.empty() ? "(not found)" : from_header.c_str(), version.c_str());
		const std::string log = ReadRepoFile("CHANGELOG.md");
		CHECK_MSG(!version.empty() && log.find("\n## [" + version + "] - ") != std::string::npos,
		          "[9.2] the version %s is not a released one: CHANGELOG.md has no dated \"## [%s] - \" heading",
		          version.c_str(), version.c_str());
	}
	// No new status (§5.3; rev 1.2.2, audit M-2): the status list's entry count equals the base this line
	// builds on, and the existing ordinals are unchanged. That base is main's 30 plus the one status the
	// merged precached-prefixes line appends (SSLM_RESTORE_PRODUCER_MISMATCH, ordinal 30): 31. Schema
	// fast-decode itself still adds none.
	{
		int n = 0;
#define SFD_COUNT_STATUS(name) ++n;
		SSLM_STATUS_ENUM_LIST(SFD_COUNT_STATUS)
#undef SFD_COUNT_STATUS
		CHECK_MSG(n == 31, "[9.2] the status list has %d entries; main plus precached prefixes has 31", n);
	}
	CHECK_MSG(static_cast<int>(SSLM_RESTORE_PRODUCER_MISMATCH) == 30, "[9.2] the precache status ordinal moved");
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
	// Bind eligibility (fix round 1, audit F-3): the prompt prefill already cleared it on `s`, so the
	// query's "changes nothing, including bind eligibility" is checked on a sequence that is eligible:
	// prefilled, then reset (which re-arms eligibility and keeps the binding, cell 1.1), then queried,
	// then rebound.
	{
		sslm_seq e = rig.NewSeq(kLongKey);
		if (e && rig.PromptOk(e, f.prompt_potion)) {
			CHECK_MSG(sslm_seq_reset(e) == SSLM_OK, "[10.2] reset");
			const Query qe = QueryForced(rig.m, e, 5);
			CHECK_MSG(qe.st == SSLM_OK && qe.count == int32_t(chain.size()), "[10.2] after the reset: status %d count %d",
			          static_cast<int>(qe.st), qe.count);
			CHECK_MSG(rig.TryBind(e, kPotion) == SSLM_OK, "[10.2] the query cleared bind eligibility");
		}
	}
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

// ===================================================================================================
// §16 Q6 C: short spans under bit 1 (k = 7, box scorecard 2026-10-20, rule 3.10)
// ===================================================================================================
//
// Q6 C, chosen: under bit 1 a verb call whose pass admits M rows with 2 <= M <= k runs them as M one-row
// passes, each through the threaded one-token split; k = 7 by rule 3.10 (M = 2..7 lose 5.4-8.6 ms per span
// to one-token threaded admissions, M = 8 wins). M = 1, M >= 8 and every call without bit 1 keep the one
// batched pass. S-LAYERROWS shows the shape (M entries of one row, or one entry of M rows), the hook's
// `run` calls show the threading, and everything observable equals the batched pass's: status, the forced
// ids, the count, and the whole save blob (K/V, residual, walk, forced_token_count, anti-LM order and
// history, the saturation census). These cells run with the minimum-work seam lowered (kSmallMinWork), so
// a one-row pass on F-SFD's small projections reaches `run`.

constexpr size_t kQ6cSplitMax = 7;

// The S-LAYERROWS shape the rule asks of one verb call that admits `m` rows.
bool Q6cShapeIs(const std::vector<sfd::LayerRowsEntry>& passes, size_t m, bool bit1) {
	const bool split = bit1 && m >= 2 && m <= kQ6cSplitMax;
	if (passes.size() != (split ? m : 1)) return false;
	for (const sfd::LayerRowsEntry& e : passes)
		if (e.kind != sfd::kLayerRowsBatched || e.rows != (split ? 1 : m)) return false;
	return true;
}

std::string Q6cShape(const std::vector<sfd::LayerRowsEntry>& passes) {
	std::string s = "{";
	for (size_t i = 0; i < passes.size(); ++i)
		s += (i ? "," : "") + std::to_string(passes[i].kind) + ":" + std::to_string(passes[i].rows);
	return s + "}";
}

std::vector<sfd::LayerRowsEntry> Q6cPasses() {
	std::vector<sfd::LayerRowsEntry> v;
	for (size_t i = 0; i < sfd::LayerRowsLogCount(); ++i) v.push_back(sfd::LayerRowsLogAt(i));
	return v;
}

enum class Q6cArm { kNoHook, kBit1, kBit0 };

struct SpanRun {
	bool ok = false;
	Jump j;
	std::vector<uint8_t> blob;
	std::vector<sfd::LayerRowsEntry> passes;
	size_t run_calls = 0;
};

// A LONGKEY sequence resting ready for logits at `state` (the prompt, then the reference path as content,
// with no workspace and so no hook), then one verb call at `budget` under `arm`. `kv_room` >= 0 makes the
// holder a budget holder with that many positions past the path; `threads` runs the hook's tasks on their
// own threads; `max_tasks` and `null_run` shape the hook (Q6C.5).
SpanRun Q6cSpan(const Sfd& f, uint32_t state, int32_t budget, Q6cArm arm, const sslm_decode_params* params = nullptr,
                int32_t kv_room = -1, bool threads = false, int32_t max_tasks = 4, bool null_run = false) {
	SpanRun o;
	const RefSchema& r = f.R(kLongKey);
	if (state >= r.st.size()) return o;
	Rig rig(f.m1->m);
	CountingHook h;
	h.max_tasks = max_tasks;
	h.threads = threads;
	if (!rig.ok) return o;
	if (arm != Q6cArm::kNoHook) {
		sslm_parallel_for pf = h.Pf(arm == Q6cArm::kBit1 ? SSLM_PARALLEL_FOR_MATVEC : 0u);
		if (null_run) pf.run = nullptr;
		const sslm_status st = sslm_workspace_set_parallel_for(rig.W(), &pf);
		CHECK_MSG(st == SSLM_OK, "[Q6C] installing a hook (max_tasks %d, run %s) == %d", max_tasks,
		          null_run ? "null" : "set", static_cast<int>(st));
		if (st != SSLM_OK) return o;
	}
	sslm_seq s = kv_room < 0 ? rig.NewSeq(kLongKey)
	                         : rig.NewBudgeted(int32_t(f.prompt_potion.size() + r.st[state].path.size()) + kv_room, kLongKey);
	if (!s || !rig.PromptOk(s, f.prompt_potion) || !rig.ContentOk(s, r.st[state].path)) return o;
	h.Clear();
	sfd::LayerRowsLogReset();
	o.j = JumpForward(rig.m, s, budget, params, rig.W());
	o.passes = Q6cPasses();
	o.run_calls = h.Counts().size();
	o.blob = rig.Save(s);
	o.ok = !o.blob.empty();
	return o;
}

// The first LONGKEY state whose forced chain has exactly `len` tokens (UINT32_MAX if none), and the state
// with the longest chain.
uint32_t Q6cChainState(const Sfd& f, size_t len) {
	const RefSchema& r = f.R(kLongKey);
	for (uint32_t s = 0; s < r.st.size(); ++s)
		if (r.st[s].forced && r.st[s].chain.size() == len) return s;
	return UINT32_MAX;
}
uint32_t Q6cLongestState(const Sfd& f) {
	const RefSchema& r = f.R(kLongKey);
	uint32_t best = UINT32_MAX;
	for (uint32_t s = 0; s < r.st.size(); ++s)
		if (r.st[s].forced && (best == UINT32_MAX || r.st[s].chain.size() > r.st[best].chain.size())) best = s;
	return best;
}

// Q6C.1 [Q6 C] One verb call at M = 1, 2, 7, 8 and 9 rows, both as a whole span (budget 64 at a state whose
// chain is M) and as a chunk_budget-bounded piece (budget M at the longest chain). Under bit 1 the status,
// forced ids and whole blob equal the batched pass's (no hook, and bit 0) and the token-by-token reference
// (the same tokens as explicit content, one per prefill call); 7 splits into seven threaded one-row passes,
// 8 stays one serial pass of eight. Bit 0 keeps one pass at every M and never calls `run`. The bit-1 arm's
// hook runs its tasks on real threads, so the cell is also the split path's TSan cell.
void Cell_Q6C_1() {
	const MinWorkScope min_work(kSmallMinWork);
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kLongKey);
	const uint32_t longest = Q6cLongestState(f);
	CHECK_MSG(longest != UINT32_MAX && r.st[longest].chain.size() >= 9, "[Q6C.1] LONGKEY has no chain of 9 or more");
	if (longest == UINT32_MAX || r.st[longest].chain.size() < 9) return;
	for (size_t m : {size_t{1}, size_t{2}, size_t{7}, size_t{8}, size_t{9}}) {
		for (int piece = 0; piece < 2; ++piece) {
			const uint32_t at = piece ? longest : Q6cChainState(f, m);
			const int32_t budget = piece ? int32_t(m) : 64;
			CHECK_MSG(at != UINT32_MAX, "[Q6C.1 M %zu] LONGKEY has no chain of exactly %zu", m, m);
			if (at == UINT32_MAX) continue;
			const std::vector<int32_t> want(r.st[at].chain.begin(), r.st[at].chain.begin() + std::ptrdiff_t(m));
			const char* how = piece ? "piece" : "span";
			const SpanRun bit1 = Q6cSpan(f, at, budget, Q6cArm::kBit1, nullptr, -1, /*threads=*/true);
			const SpanRun plain = Q6cSpan(f, at, budget, Q6cArm::kNoHook);
			const SpanRun bit0 = Q6cSpan(f, at, budget, Q6cArm::kBit0);
			for (const SpanRun* x : {&bit1, &plain, &bit0}) {
				const char* arm = x == &bit1 ? "bit 1" : x == &plain ? "no hook" : "bit 0";
				CHECK_MSG(x->ok && x->j.st == SSLM_OK && x->j.count == int32_t(m) && x->j.forced == want,
				          "[Q6C.1 M %zu %s, %s] status %d count %d forced %s; want %s", m, how, arm,
				          static_cast<int>(x->j.st), x->j.count, Str(x->j.forced).c_str(), Str(want).c_str());
			}
			CHECK_MSG(bit1.blob == plain.blob && bit0.blob == plain.blob, "[Q6C.1 M %zu %s] the save blob: bit 1 %s, bit 0 "
			          "%s the no-hook batched pass's", m, how, bit1.blob == plain.blob ? "equals" : "differs from",
			          bit0.blob == plain.blob ? "equals" : "differs from");
			// The token-by-token reference: main's explicit content prefill, one token per call.
			{
				Rig rig(f.m1->m);
				sslm_seq s = ReachState(rig, f, kLongKey, at, f.prompt_potion);
				if (s && rig.ContentOk(s, want, 1)) {
					CHECK_MSG(rig.Save(s) == bit1.blob, "[Q6C.1 M %zu %s] bit 1's blob differs from the token-by-token "
					          "reference", m, how);
				}
			}
			// The boundary.
			CHECK_MSG(Q6cShapeIs(bit1.passes, m, true), "[Q6C.1 M %zu %s, bit 1] S-LAYERROWS %s; want %s", m, how,
			          Q6cShape(bit1.passes).c_str(),
			          (m >= 2 && m <= kQ6cSplitMax) ? "one-row passes, one per row" : "one batched pass of M rows");
			CHECK_MSG((m <= kQ6cSplitMax) == (bit1.run_calls > 0), "[Q6C.1 M %zu %s, bit 1] %zu run calls; want %s", m,
			          how, bit1.run_calls, m <= kQ6cSplitMax ? "some (threaded one-row passes)" : "none (one serial pass)");
			CHECK_MSG(Q6cShapeIs(bit0.passes, m, false) && bit0.run_calls == 0, "[Q6C.1 M %zu %s, bit 0] S-LAYERROWS %s, "
			          "%zu run calls; want one pass of M rows and none", m, how, Q6cShape(bit0.passes).c_str(),
			          bit0.run_calls);
			CHECK_MSG(Q6cShapeIs(plain.passes, m, false), "[Q6C.1 M %zu %s, no hook] S-LAYERROWS %s", m, how,
			          Q6cShape(plain.passes).c_str());
		}
	}
}

// Q6C.2 [Q6 C] The §5.6 host loop under bit 1 at chunk_budget 1, 2, 3, 7, 8, 9 and 64, on LONGKEY (chains
// of every length to 26, so most spans are cut by the budget) and on POTION (a pending token joins each
// span): the stream and the final blob equal the no-hook run's at budget 64, the stream is the reference
// rule's, every verb call's passes have the rule's shape for the rows it admitted, and no call runs more
// rows than its budget (C11). Then a span cut by the holder's K/V budget (a partial admission, 5 of 9):
// the limit status, count and blob equal the batched pass's, and the 5 admitted rows ran split.
void Cell_Q6C_2() {
	const MinWorkScope min_work(kSmallMinWork);
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const char* schema : {kLongKey, kPotion}) {
		std::vector<int32_t> plain;
		std::vector<uint8_t> plain_blob;
		{
			Rig rig(f.m1->m);
			sslm_seq s = nullptr;
			const JfRun r = FreshRun(rig, f, schema, f.prompt_potion, GreedyOpts(rig), &s);
			CHECK_MSG(r.dead_end, "[Q6C.2] %s without the hook: status %d", schema, static_cast<int>(r.end_status));
			plain = r.stream;
			if (s) plain_blob = rig.Save(s);
		}
		for (int32_t budget : {1, 2, 3, 7, 8, 9, 64}) {
			Rig rig(f.m1->m);
			CountingHook h;
			h.max_tasks = 4;
			if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) continue;
			JfOpts o = GreedyOpts(rig, budget);
			int bad_shape = 0, over_budget = 0, split_calls = 0;
			std::string first_bad;
			o.after = [&](sslm_seq&, const JfStep& st) {
				if (st.kind == 0 && st.st == SSLM_OK) {
					const size_t m = st.toks.size() + (st.embedded_pending ? 1 : 0);
					const std::vector<sfd::LayerRowsEntry> passes = Q6cPasses();
					uint64_t rows = 0;
					for (const sfd::LayerRowsEntry& e : passes) rows += e.rows;
					if (rows > uint64_t(budget)) ++over_budget;
					if (m >= 1 && !Q6cShapeIs(passes, m, true)) {
						if (bad_shape++ == 0) first_bad = "M " + std::to_string(m) + ": " + Q6cShape(passes);
					}
					if (m >= 2 && m <= kQ6cSplitMax) ++split_calls;
				}
				sfd::LayerRowsLogReset();
			};
			sslm_seq s = rig.NewSeq(schema);
			if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
			sfd::LayerRowsLogReset();
			const JfRun r = RunJf(rig.m, s, o);
			CheckStreamIsReference("Q6C.2", f, schema, r);
			CHECK_MSG(r.stream == plain && !plain.empty() && rig.Save(s) == plain_blob, "[Q6C.2] %s budget %d under bit 1: "
			          "stream %s, no hook %s; the blob %s", schema, budget, Str(r.stream).c_str(), Str(plain).c_str(),
			          rig.Save(s) == plain_blob ? "equal" : "differs");
			CHECK_MSG(over_budget == 0, "[Q6C.2] %s budget %d: %d verb calls ran more rows than the budget", schema, budget,
			          over_budget);
			CHECK_MSG(bad_shape == 0, "[Q6C.2] %s budget %d: %d verb calls do not have the rule's passes (first: %s)",
			          schema, budget, bad_shape, first_bad.c_str());
			if (budget >= 2) {
				CHECK_MSG(split_calls > 0, "[Q6C.2] %s budget %d: no verb call admitted 2..7 rows; the arm shows nothing",
				          schema, budget);
			}
		}
	}
	// The K/V budget cuts a 9-token span at 5.
	const uint32_t at = Q6cChainState(f, 9);
	CHECK_MSG(at != UINT32_MAX, "[Q6C.2] LONGKEY has no chain of exactly 9");
	if (at == UINT32_MAX) return;
	const SpanRun bit1 = Q6cSpan(f, at, 64, Q6cArm::kBit1, nullptr, 5);
	const SpanRun plain = Q6cSpan(f, at, 64, Q6cArm::kNoHook, nullptr, 5);
	CHECK_MSG(plain.ok && plain.j.st == SSLM_KV_BUDGET_EXCEEDED && plain.j.count == 5, "[Q6C.2 K/V budget] no hook: status "
	          "%d count %d; want SSLM_KV_BUDGET_EXCEEDED and 5", static_cast<int>(plain.j.st), plain.j.count);
	CHECK_MSG(bit1.ok && bit1.j.st == plain.j.st && bit1.j.count == plain.j.count && bit1.j.forced == plain.j.forced &&
	              bit1.blob == plain.blob,
	          "[Q6C.2 K/V budget] bit 1: status %d count %d, the blob %s; want the batched pass's", static_cast<int>(bit1.j.st),
	          bit1.j.count, bit1.blob == plain.blob ? "equal" : "differs");
	CHECK_MSG(Q6cShapeIs(bit1.passes, 5, true) && bit1.run_calls > 0, "[Q6C.2 K/V budget] bit 1: S-LAYERROWS %s, %zu run "
	          "calls; want five threaded one-row passes", Q6cShape(bit1.passes).c_str(), bit1.run_calls);
}

// Q6C.3 [Q6 C] Damped greedy under bit 1: the anti-LM history and save/restore across split spans.
// (a) The host loop at budgets 3 and 64, with a save and a restore after every call: the stream and the final
// blob (the anti-LM order and history included) equal the uninterrupted no-hook run's. (b) One 9-token span
// as three budget-3 calls (three splits), with a save and a restore between them: the blob equals one
// no-hook call admitting all nine, and its history holds the nine forced tokens in order.
void Cell_Q6C_3() {
	const MinWorkScope min_work(kSmallMinWork);
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const char* schema : {kPotion, kLongKey}) {
		std::vector<int32_t> plain;
		std::vector<uint8_t> plain_blob;
		{
			Rig rig(f.m1->m);
			const sslm_decode_params p = rig.Damped();
			JfOpts o = GreedyOpts(rig);
			o.vparams = &p;
			o.dparams = p;
			sslm_seq s = nullptr;
			const JfRun r = FreshRun(rig, f, schema, f.prompt_potion, o, &s);
			CHECK_MSG(r.dead_end, "[Q6C.3 (a)] %s damped, no hook: status %d", schema, static_cast<int>(r.end_status));
			plain = r.stream;
			if (s) plain_blob = rig.Save(s);
			CHECK_MSG(!ParseBlob(plain_blob).hist.empty(), "[Q6C.3 (a)] %s: the damped run has no anti-LM history",
			          schema);
		}
		for (int32_t budget : {3, 64}) {
			Rig rig(f.m1->m);
			CountingHook h;
			h.max_tasks = 4;
			if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) continue;
			const sslm_decode_params p = rig.Damped();
			JfOpts o = GreedyOpts(rig, budget);
			o.vparams = &p;
			o.dparams = p;
			int splits = 0;
			o.after = [&](sslm_seq& q, const JfStep& st) {
				if (st.kind == 0 && st.st == SSLM_OK) {
					const size_t m = st.toks.size() + (st.embedded_pending ? 1 : 0);
					if (m >= 2 && m <= kQ6cSplitMax && Q6cShapeIs(Q6cPasses(), m, true)) ++splits;
				}
				sfd::LayerRowsLogReset();
				if (st.st != SSLM_OK) return;
				const std::vector<uint8_t> blob = rig.Save(q);
				sslm_seq fresh = rig.Restore(blob);
				if (!fresh) return;
				rig.Release(q);
				q = fresh;
			};
			sslm_seq s = rig.NewSeq(schema);
			if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
			sfd::LayerRowsLogReset();
			const JfRun r = RunJf(rig.m, s, o);
			const std::vector<uint8_t> blob = rig.Save(s);
			CHECK_MSG(r.dead_end && r.stream == plain && !plain.empty(), "[Q6C.3 (a)] %s budget %d, bit 1, save/restore "
			          "after every call: stream %s, uninterrupted no hook %s", schema, budget, Str(r.stream).c_str(),
			          Str(plain).c_str());
			const Blob b = ParseBlob(blob), pb = ParseBlob(plain_blob);
			CHECK_MSG(blob == plain_blob, "[Q6C.3 (a)] %s budget %d: the final blob differs (order %d / %d, history %s / %s)",
			          schema, budget, b.order, pb.order, Str(b.hist).c_str(), Str(pb.hist).c_str());
			CHECK_MSG(splits > 0, "[Q6C.3 (a)] %s budget %d: no verb call ran split", schema, budget);
		}
	}
	// (b)
	const uint32_t at = Q6cChainState(f, 9);
	const RefSchema& r = f.R(kLongKey);
	CHECK_MSG(at != UINT32_MAX, "[Q6C.3 (b)] LONGKEY has no chain of exactly 9");
	if (at == UINT32_MAX) return;
	std::vector<uint8_t> one;
	{
		Rig rig(f.m1->m);
		const sslm_decode_params p = rig.Damped();
		sslm_seq s = ReachState(rig, f, kLongKey, at, f.prompt_potion);
		if (!s) return;
		const Jump j = JumpForward(rig.m, s, 64, &p, rig.W());
		CHECK_MSG(j.st == SSLM_OK && j.forced == r.st[at].chain, "[Q6C.3 (b)] one no-hook call: status %d, %s",
		          static_cast<int>(j.st), Str(j.forced).c_str());
		one = rig.Save(s);
	}
	Rig rig(f.m1->m);
	CountingHook h;
	h.max_tasks = 4;
	if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) return;
	const sslm_decode_params p = rig.Damped();
	sslm_seq s = ReachState(rig, f, kLongKey, at, f.prompt_potion);
	if (!s) return;
	std::vector<int32_t> got;
	int split_calls = 0;
	for (int call = 0; call < 3; ++call) {
		sfd::LayerRowsLogReset();
		const Jump j = JumpForward(rig.m, s, 3, &p, rig.W());
		if (j.st == SSLM_OK && j.count == 3 && Q6cShapeIs(Q6cPasses(), 3, true)) ++split_calls;
		got.insert(got.end(), j.forced.begin(), j.forced.end());
		if (call < 2) {
			sslm_seq fresh = rig.Restore(rig.Save(s));
			if (!fresh) return;
			rig.Release(s);
			s = fresh;
		}
	}
	const std::vector<uint8_t> three = rig.Save(s);
	const Blob b = ParseBlob(three);
	CHECK_MSG(got == r.st[at].chain && three == one, "[Q6C.3 (b)] three budget-3 calls with save/restore: %s, the blob %s "
	          "one no-hook call's", Str(got).c_str(), three == one ? "equals" : "differs from");
	CHECK_MSG(b.ok && b.hist == r.st[at].chain, "[Q6C.3 (b)] the anti-LM history %s; want the nine forced tokens %s",
	          Str(b.hist).c_str(), Str(r.st[at].chain).c_str());
	CHECK_MSG(split_calls == 3, "[Q6C.3 (b)] %d of 3 calls ran as three threaded one-row passes", split_calls);
}

// Q6C.4 [Q6 C] A failure inside a split span. A hostile hook (omit, then duplicate) breaks exactly-once on
// one `run` call of a 3-row split under damped params: the first call (row 1), a call in the middle of row
// 2, and the last call (row 3). Review round 1, finding 2: a build that checks only the last pass's status
// passes the last-call arm, so the first two arms are the ones that kill it. In every arm the call returns
// SSLM_INVALID_ARGUMENT with nothing admitted, the anti-LM order and history and the whole save blob are
// unchanged (the K/V rows past context_length are not in it, the saturation census is restored), and a
// retry with an honest hook gives the blob of a call that never failed.
void Cell_Q6C_4() {
	const MinWorkScope min_work(kSmallMinWork);
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const uint32_t at = Q6cChainState(f, 9);
	CHECK_MSG(at != UINT32_MAX, "[Q6C.4] LONGKEY has no chain of exactly 9");
	if (at == UINT32_MAX) return;
	// The never-failed call, and the number of run calls it makes.
	std::vector<uint8_t> ref_after;
	size_t calls = 0;
	{
		Rig rig(f.m1->m);
		CountingHook h;
		h.max_tasks = 4;
		if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) return;
		const sslm_decode_params p = rig.Damped(2);
		sslm_seq s = ReachState(rig, f, kLongKey, at, f.prompt_potion);
		if (!s) return;
		h.Clear();
		sfd::LayerRowsLogReset();
		const Jump j = JumpForward(rig.m, s, 3, &p, rig.W());
		calls = h.Counts().size();
		CHECK_MSG(j.st == SSLM_OK && j.count == 3 && Q6cShapeIs(Q6cPasses(), 3, true) && calls >= 3 && calls % 3 == 0,
		          "[Q6C.4] the never-failed call: status %d count %d, S-LAYERROWS %s, %zu run calls; want three "
		          "threaded one-row passes with the same run calls each", static_cast<int>(j.st), j.count,
		          Q6cShape(Q6cPasses()).c_str(), calls);
		ref_after = rig.Save(s);
	}
	if (calls < 3 || calls % 3 != 0) return;
	const size_t per_row = calls / 3;
	struct Where {
		const char* name;
		size_t call;
	};
	const Where wheres[] = {{"first call (row 1)", 0}, {"middle of row 2", per_row + per_row / 2}, {"last call (row 3)", calls - 1}};
	for (Hostile kind : {Hostile::kOmit, Hostile::kDuplicate}) {
		const char* kname = kind == Hostile::kOmit ? "omit" : "duplicate";
		for (const Where& w : wheres) {
			Rig rig(f.m1->m);
			CountingHook h;
			h.max_tasks = 4;
			h.hostile = kind;
			h.hostile_call = static_cast<int>(w.call);
			if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) continue;
			const sslm_decode_params p = rig.Damped(2);
			sslm_seq s = ReachState(rig, f, kLongKey, at, f.prompt_potion);
			if (!s) continue;
			const std::vector<uint8_t> before = rig.Save(s);
			h.Clear();
			const Jump j = JumpForward(rig.m, s, 3, &p, rig.W());
			CHECK_MSG(h.Counts().size() > w.call && j.st == SSLM_INVALID_ARGUMENT && j.count == 0, "[Q6C.4 %s, %s] %zu "
			          "run calls (the hostile one is %zu), status %d count %d; want SSLM_INVALID_ARGUMENT and nothing "
			          "admitted", kname, w.name, h.Counts().size(), w.call, static_cast<int>(j.st), j.count);
			const std::vector<uint8_t> after = rig.Save(s);
			const Blob b0 = ParseBlob(before), b1 = ParseBlob(after);
			CHECK_MSG(after == before, "[Q6C.4 %s, %s] the failed call changed the save blob (order %d -> %d, history %s "
			          "-> %s, context %lld -> %lld)", kname, w.name, b0.order, b1.order, Str(b0.hist).c_str(),
			          Str(b1.hist).c_str(), static_cast<long long>(b0.ctx), static_cast<long long>(b1.ctx));
			h.hostile = Hostile::kNone;
			const Jump retry = JumpForward(rig.m, s, 3, &p, rig.W());
			CHECK_MSG(retry.st == SSLM_OK && retry.count == 3 && rig.Save(s) == ref_after, "[Q6C.4 %s, %s] the retry: "
			          "status %d count %d, the blob %s the never-failed call's", kname, w.name, static_cast<int>(retry.st),
			          retry.count, rig.Save(s) == ref_after ? "equals" : "differs from");
		}
	}
}

// Q6C.5 [Q6 C] Review round 1, finding 1: bit 1 on a host where a one-row pass cannot thread keeps the one
// batched pass, because M serial one-row passes cost more than one batched pass (r(M) < M). The split keys
// on GemmDispatch's own rule (MatvecGroupSplit's task_count >= 2 over the largest one-row projection), not
// on the bit. Three such hosts, at M = 2 and 7: max_tasks 1, max_tasks 0 with a null `run` (both with the
// minimum work lowered), and max_tasks 4 at the shipped minimum work, below which F-SFD's projections fall.
// Each gives one pass of M rows, no `run` call, and the no-hook blob. The control (max_tasks 4, minimum work
// lowered) splits.
void Cell_Q6C_5() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	struct Host {
		const char* name;
		int32_t max_tasks;
		bool null_run;
		bool small_min_work;
		bool splits;
	};
	const Host hosts[] = {{"max_tasks 1", 1, false, true, false},
	                      {"max_tasks 0, run null", 0, true, true, false},
	                      {"max_tasks 4, shipped minimum work", 4, false, false, false},
	                      {"max_tasks 4, minimum work lowered (control)", 4, false, true, true}};
	for (size_t m : {size_t{2}, size_t{7}}) {
		const uint32_t at = Q6cChainState(f, m);
		CHECK_MSG(at != UINT32_MAX, "[Q6C.5] LONGKEY has no chain of exactly %zu", m);
		if (at == UINT32_MAX) continue;
		const SpanRun plain = Q6cSpan(f, at, 64, Q6cArm::kNoHook);
		for (const Host& h : hosts) {
			std::unique_ptr<MinWorkScope> min_work;
			if (h.small_min_work) min_work = std::make_unique<MinWorkScope>(kSmallMinWork);
			const SpanRun x = Q6cSpan(f, at, 64, Q6cArm::kBit1, nullptr, -1, false, h.max_tasks, h.null_run);
			CHECK_MSG(x.ok && x.j.st == SSLM_OK && x.j.count == int32_t(m) && x.blob == plain.blob, "[Q6C.5 M %zu, %s] "
			          "status %d count %d, the blob %s the no-hook pass's", m, h.name, static_cast<int>(x.j.st),
			          x.j.count, x.blob == plain.blob ? "equals" : "differs from");
			if (h.splits) {
				CHECK_MSG(Q6cShapeIs(x.passes, m, true) && x.run_calls > 0, "[Q6C.5 M %zu, %s] S-LAYERROWS %s, %zu run "
				          "calls; want %zu threaded one-row passes", m, h.name, Q6cShape(x.passes).c_str(), x.run_calls, m);
			} else {
				CHECK_MSG(Q6cShapeIs(x.passes, m, false) && x.run_calls == 0, "[Q6C.5 M %zu, %s] S-LAYERROWS %s, %zu run "
				          "calls; want one batched pass of %zu rows and none", m, h.name, Q6cShape(x.passes).c_str(),
				          x.run_calls, m);
			}
		}
	}
}

// Q6C.6 [Q6 C] Review round 1, finding 3: the split is the verb's alone. Under bit 1 on a host whose one-row
// passes thread (max_tasks 4, minimum work lowered), sslm_prefill (schema content and prompt) and
// sslm_prefix_prefill of M = 2..7 tokens each run one pass of M rows and never call `run`, and the prefill's
// blob equals the no-hook prefill's.
void Cell_Q6C_6() {
	const MinWorkScope min_work(kSmallMinWork);
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	const RefSchema& r = f.R(kLongKey);
	const uint32_t at = Q6cLongestState(f);
	if (at == UINT32_MAX || r.st[at].chain.size() < kQ6cSplitMax) return;
	for (size_t m = 2; m <= kQ6cSplitMax; ++m) {
		const std::vector<int32_t> content(r.st[at].chain.begin(), r.st[at].chain.begin() + std::ptrdiff_t(m));
		const std::vector<int32_t> prompt(f.prompt_potion.begin(), f.prompt_potion.begin() + std::ptrdiff_t(m));
		for (sslm_span_kind kind : {SSLM_SPAN_SCHEMA_CONTENT, SSLM_SPAN_PROMPT}) {
			const char* kname = kind == SSLM_SPAN_PROMPT ? "prompt" : "schema content";
			const std::vector<int32_t>& toks = kind == SSLM_SPAN_PROMPT ? prompt : content;
			std::vector<uint8_t> blobs[2];
			for (int hooked = 0; hooked < 2; ++hooked) {
				Rig rig(f.m1->m);
				CountingHook h;
				h.max_tasks = 4;
				if (!rig.ok || (hooked && !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC))) continue;
				sslm_seq s = kind == SSLM_SPAN_PROMPT ? rig.NewSeq(kLongKey) : ReachState(rig, f, kLongKey, at, f.prompt_potion);
				if (!s) continue;
				h.Clear();
				sfd::LayerRowsLogReset();
				int32_t consumed = 0;
				const sslm_status st = sslm_prefill(rig.m, s, toks.data(), int32_t(m), int32_t(m), kind, rig.W(), &consumed);
				const std::vector<sfd::LayerRowsEntry> passes = Q6cPasses();
				if (hooked) {
					CHECK_MSG(st == SSLM_OK && consumed == int32_t(m) && Q6cShapeIs(passes, m, false) && h.Counts().empty(),
					          "[Q6C.6 sslm_prefill %s, M %zu, bit 1] status %d consumed %d, S-LAYERROWS %s, %zu run calls; "
					          "want one pass of M rows and none", kname, m, static_cast<int>(st), consumed,
					          Q6cShape(passes).c_str(), h.Counts().size());
				}
				blobs[hooked] = rig.Save(s);
			}
			CHECK_MSG(!blobs[0].empty() && blobs[0] == blobs[1], "[Q6C.6 sslm_prefill %s, M %zu] the blob under bit 1 %s "
			          "the no-hook prefill's", kname, m, blobs[0] == blobs[1] ? "equals" : "differs from");
		}
		Rig rig(f.m1->m);
		CountingHook h;
		h.max_tasks = 4;
		if (!rig.ok || !Install(rig.W(), &h, SSLM_PARALLEL_FOR_MATVEC)) continue;
		sslm_prefix p = nullptr;
		CHECK_MSG(sslm_prefix_begin(rig.m, &rig.pool, &p) == SSLM_OK && p, "[Q6C.6] prefix begin");
		if (!p) continue;
		h.Clear();
		sfd::LayerRowsLogReset();
		int32_t consumed = 0;
		const sslm_status st =
		    sslm_prefix_prefill(rig.m, p, prompt.data(), int32_t(m), int32_t(m), SSLM_SPAN_PROMPT, rig.W(), &consumed);
		const std::vector<sfd::LayerRowsEntry> passes = Q6cPasses();
		CHECK_MSG(st == SSLM_OK && consumed == int32_t(m) && Q6cShapeIs(passes, m, false) && h.Counts().empty(),
		          "[Q6C.6 sslm_prefix_prefill, M %zu, bit 1] status %d consumed %d, S-LAYERROWS %s, %zu run calls; want "
		          "one pass of M rows and none", m, static_cast<int>(st), consumed, Q6cShape(passes).c_str(),
		          h.Counts().size());
		sslm_prefix_release(p);
	}
}

// ===================================================================================================
// Plan rev 1.3.3, slice TF (§4.5, C16, §11.5): the decode step's finish as a transaction. TF.1 and TF.3-TF.7
// are allocation-failure cells and live in the allocation sweep (sfd_alloc_main.cpp), which can fail
// allocation k of one call; TF.2 and TF.4 (b) are here. TF.8 ("unchanged behaviour") is no new cell: it is
// the decode-threading suite's ParallelForIncomplete cells and every M1 and J1 success-path cell above,
// run unchanged.
//
// The arms (§11.5), on POTION from the suite's prompt, at greedy and at damped order 2, with a workspace
// and without one:
//   (a)  the layer branch: N decodes, then the target decode embeds t_{N-1}, runs the layers and finishes;
//   (b)  the ready branch, off: the first decode after the prompt;
//   (c)  canonical J1 after a span: the §5.6 drain, a choice, the drain again, then the target decode;
//   (c') canonical at budget 1 after N decodes: the verb admits the pending t_{N-1} alone (exact mode's
//        j = 0 commit shape), and the target decode runs the ready branch at (a)'s output position. Its
//        blob before the target decode is (a)'s finish entry, which (a) cannot observe inside its call.
// Exact mode's arms, (d) after a j = 0 and a j >= 2 commit, are J1-X cells (TF.2d, TF.4 (b) X) and land with
// that slice. N is the first position >= 3 whose state after t_{N-1} is forced with a canonical first
// token other than t_N (so the budget-1 verb in (c') admits t_{N-1} and nothing else).
// ===================================================================================================

enum class TfArm { kLayer, kReady, kCanonSpan, kCanonPending };
const char* TfArmName(TfArm a) {
	switch (a) {
	case TfArm::kLayer: return "(a) layer branch";
	case TfArm::kReady: return "(b) first token after the prompt";
	case TfArm::kCanonSpan: return "(c) canonical after a span";
	case TfArm::kCanonPending: return "(c') canonical pending-only at N";
	}
	return "?";
}
constexpr TfArm kTfArms[] = {TfArm::kLayer, TfArm::kReady, TfArm::kCanonSpan, TfArm::kCanonPending};

struct TfCfg {
	bool damped = false;
	bool ws = false;
};
std::string TfCfgName(const TfCfg& c) {
	return std::string(c.damped ? "damped@2" : "greedy") + (c.ws ? ", workspace" : ", no workspace");
}

struct TfRig {
	Rig rig;
	const Sfd& f;
	TfCfg c;
	std::vector<int32_t> stream;  // off's never-failed stream from the prompt
	int n = -1;
	bool ok = false;

	TfRig(const Sfd& fsfd, const TfCfg& cfg) : rig(fsfd.m1->m, 1, 16), f(fsfd), c(cfg) {}
	sslm_workspace W() const { return c.ws ? rig.W() : nullptr; }
	sslm_decode_params P(int32_t order = 2) const { return c.damped ? rig.Damped(order) : rig.Greedy(); }
	sslm_status Decode(sslm_seq s, int32_t order, int32_t* out) { return rig.Decode1(s, P(order), W(), out); }
	// The canonical verb with the decode's own params (NULL under greedy, §5.8).
	Jump Verb(sslm_seq s, int32_t budget) {
		const sslm_decode_params p = P(2);
		return JumpForward(rig.m, s, budget, c.damped ? &p : nullptr, W());
	}
	bool Drain(sslm_seq s) {
		for (int guard = 0; guard < 16; ++guard) {
			const Query q = QueryForced(rig.m, s, 0);
			if (q.st != SSLM_OK) return false;
			if (q.count == 0) return true;
			if (Verb(s, 64).st != SSLM_OK) return false;
		}
		return false;
	}
	sslm_seq Fresh() {
		sslm_seq s = rig.NewSeq(kPotion);
		if (s && !rig.PromptOk(s, f.prompt_potion)) {
			rig.Release(s);
			return nullptr;
		}
		return s;
	}
	// The arm's sequence, resting at the target decode's entry; null on a setup failure. `at` overrides N.
	sslm_seq Build(TfArm arm, int at = -1) {
		const int k = at >= 0 ? at : n;
		sslm_seq s = Fresh();
		if (!s) return nullptr;
		bool good = true;
		int32_t t = -100;
		switch (arm) {
		case TfArm::kReady: break;
		case TfArm::kLayer:
		case TfArm::kCanonPending:
			for (int i = 0; i < k && good; ++i) good = Decode(s, 2, &t) == SSLM_OK && t >= 0;
			if (good && arm == TfArm::kCanonPending) {
				const Jump j = Verb(s, 1);
				good = j.st == SSLM_OK && j.count == 0 && !ReadShape(rig.m, s).pending;
			}
			break;
		case TfArm::kCanonSpan:
			good = Drain(s) && Decode(s, 2, &t) == SSLM_OK && t >= 0 && Drain(s);
			break;
		}
		if (!good) {
			rig.Release(s);
			return nullptr;
		}
		return s;
	}
	// Status, token and blob of `count` decodes at `order`.
	struct Run {
		std::vector<int> st;
		std::vector<int32_t> tok;
		std::vector<std::vector<uint8_t>> blob;
		bool operator==(const Run& o) const { return st == o.st && tok == o.tok && blob == o.blob; }
	};
	Run Continue(sslm_seq s, int32_t order, int count) {
		Run r;
		for (int i = 0; i < count; ++i) {
			int32_t t = -100;
			r.st.push_back(static_cast<int>(Decode(s, order, &t)));
			r.tok.push_back(t);
			r.blob.push_back(rig.Save(s));
		}
		return r;
	}
	bool Setup() {
		if (!rig.ok) return false;
		sslm_seq s = Fresh();
		if (!s) return false;
		for (int i = 0; i < 64; ++i) {
			int32_t t = -100;
			if (Decode(s, 2, &t) != SSLM_OK || t < 0) break;
			stream.push_back(t);
		}
		rig.Release(s);
		for (int k = 3; k + 1 < static_cast<int>(stream.size()) && n < 0; ++k) {
			sslm_seq q = Build(TfArm::kLayer, k);
			if (!q) continue;
			const Query fq = QueryForced(rig.m, q, 1);
			if (fq.st == SSLM_OK && fq.count > 0 && !fq.ids.empty() && fq.ids[0] != stream[size_t(k)]) n = k;
			rig.Release(q);
		}
		ok = n >= 3;
		CHECK_MSG(ok, "[TF %s] no target position: off's stream has %zu tokens and no position n >= 3 whose state is "
		          "forced with a canonical first token other than the model's", TfCfgName(c).c_str(), stream.size());
		return ok;
	}
};
std::string Str(const TfRig::Run& r) {
	std::string s = "{";
	for (size_t i = 0; i < r.st.size(); ++i) s += (i ? "," : "") + std::to_string(r.st[i]) + ":" + std::to_string(r.tok[i]);
	return s + "}";
}
std::string TfFields(const std::vector<uint8_t>& b) {
	const Blob o = ParseBlob(b);
	if (!o.ok) return "<no blob>";
	char buf[200];
	std::snprintf(buf, sizeof buf, "walk %u ctx %lld layer %d current %d ready %u order %d history %zu", o.walk,
	              static_cast<long long>(o.ctx), o.layer, o.current, o.ready, o.order, o.hist.size());
	return buf;
}

const char* TfExitName(sfd::FinishExit x) {
	switch (x) {
	case sfd::FinishExit::kFinalNorm: return "final norm";
	case sfd::FinishExit::kLogitsReject: return "logits row reject";
	case sfd::FinishExit::kDampedNotOk: return "damped !ok";
	case sfd::FinishExit::kDampedRefused: return "damped refused";
	case sfd::FinishExit::kNone: break;
	}
	return "none";
}

// TF.2 [TF, F-SFD POTION; XF1] The finish's status exits, through S-FINISHEXIT (none is reachable by
// construction on F-SFD): the final norm's status and a logits row reject in every cfg, damped !ok and
// damped refused in the damped cfgs, in each arm. The failed decode rests as TF.1 does ((b), (c), (c'):
// byte-equal to the blob before the call; (a): byte-equal to (c')'s blob before its target decode, the
// layer branch's finish entry); the deterministic refusal returns the same status on two retries with the
// blob unchanged; S-FINISHEXIT failed all three finishes; and with the exit disarmed the next three decodes
// equal the never-failed run of the same arm (statuses, tokens, blobs).
void Cell_TF_2() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const TfCfg c : {TfCfg{false, false}, TfCfg{false, true}, TfCfg{true, false}, TfCfg{true, true}}) {
		TfRig x(f, c);
		if (!x.Setup()) continue;
		const std::string cn = TfCfgName(c);
		std::vector<uint8_t> twin_before;
		if (sslm_seq t = x.Build(TfArm::kCanonPending)) {
			twin_before = x.rig.Save(t);
			x.rig.Release(t);
		}
		CHECK_MSG(!twin_before.empty(), "[TF.2 %s] setup: arm (c') did not build at N %d", cn.c_str(), x.n);
		std::vector<sfd::FinishExit> exits = {sfd::FinishExit::kFinalNorm, sfd::FinishExit::kLogitsReject};
		if (c.damped) {
			exits.push_back(sfd::FinishExit::kDampedNotOk);
			exits.push_back(sfd::FinishExit::kDampedRefused);
		}
		for (const sfd::FinishExit ex : exits) {
			for (const TfArm arm : kTfArms) {
				const char* an = TfArmName(arm);
				const char* xn = TfExitName(ex);
				TfRig::Run ref;
				if (sslm_seq r = x.Build(arm)) {
					ref = x.Continue(r, 2, 3);
					x.rig.Release(r);
				}
				sslm_seq s = x.Build(arm);
				CHECK_MSG(s != nullptr && ref.st.size() == 3 && ref.st[0] == SSLM_OK,
				          "[TF.2 %s, %s, %s] setup: the arm did not build, or its never-failed decode failed", cn.c_str(),
				          xn, an);
				if (!s || ref.st.size() != 3) {
					if (s) x.rig.Release(s);
					continue;
				}
				const std::vector<uint8_t> before = x.rig.Save(s);
				sfd::ArmFinishExit(ex, 0);
				int32_t t = -100;
				const sslm_status st = x.Decode(s, 2, &t);
				const std::vector<uint8_t> rest = x.rig.Save(s);
				const uint64_t fired = sfd::FinishExitFired();
				if (st == SSLM_OK || fired == 0) {
					sfd::DisarmFinishExit();
					CHECK_MSG(false, "[TF.2 %s, %s, %s] the exit was not taken: status %d token %d, S-FINISHEXIT fired "
					          "%llu (the seam is unimplemented, or the finish never reached this exit)", cn.c_str(), xn,
					          an, static_cast<int>(st), t, static_cast<unsigned long long>(fired));
					x.rig.Release(s);
					continue;
				}
				const std::vector<uint8_t>& entry = arm == TfArm::kLayer ? twin_before : before;
				CHECK_MSG(rest == entry, "[TF.2 %s, %s, %s] status %d; the failed finish does not rest at its entry: %s, "
				          "entry %s", cn.c_str(), xn, an, static_cast<int>(st), TfFields(rest).c_str(),
				          TfFields(entry).c_str());
				int32_t t2 = -100, t3 = -100;
				const sslm_status st2 = x.Decode(s, 2, &t2);
				const std::vector<uint8_t> rest2 = x.rig.Save(s);
				const sslm_status st3 = x.Decode(s, 2, &t3);
				const std::vector<uint8_t> rest3 = x.rig.Save(s);
				const uint64_t fired3 = sfd::FinishExitFired();
				sfd::DisarmFinishExit();
				CHECK_MSG(st2 == st && st3 == st && rest2 == rest && rest3 == rest && fired3 == 3,
				          "[TF.2 %s, %s, %s] the two retries: status %d, %d (first %d), blob unchanged %d, %d, S-FINISHEXIT "
				          "fired %llu (want 3)", cn.c_str(), xn, an, static_cast<int>(st2), static_cast<int>(st3),
				          static_cast<int>(st), rest2 == rest, rest3 == rest, static_cast<unsigned long long>(fired3));
				const TfRig::Run cont = x.Continue(s, 2, 3);
				CHECK_MSG(cont == ref, "[TF.2 %s, %s, %s] disarmed, the next three decodes %s; never-failed %s (blobs %s)",
				          cn.c_str(), xn, an, Str(cont).c_str(), Str(ref).c_str(),
				          cont.blob == ref.blob ? "equal" : "differ");
				x.rig.Release(s);
			}
		}
	}
}

// TF.4 (b) [TF, F-SFD POTION; XF4's dead-end half] (rev 1.3.3) At a dead end the prepared re-key is
// installed, as today: the target decode names order 3 on a sequence whose anti-LM state is at order 2
// ((b): no state yet, so the prepared state is a create), and its finish dead-ends (S-CHOICE armed with a
// token illegal at the walk state takes the dead-end path, §11.0). The decode returns OK with -2 and rests
// ready at layer 0 with the walk unchanged, its anti-LM state at order 3 with an empty history, the pin
// consumed; the next decode at order 3 emits a legal token. Off's half; exact mode's equality with it is a
// J1-X cell (TF.4 (b) X). Green on the tip by design: it pins today's behaviour so TF keeps it.
void Cell_TF_4b() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const TfCfg c : {TfCfg{true, false}, TfCfg{true, true}}) {
		TfRig x(f, c);
		if (!x.Setup()) continue;
		const std::string cn = TfCfgName(c);
		for (const TfArm arm : kTfArms) {
			const char* an = TfArmName(arm);
			sslm_seq s = x.Build(arm);
			CHECK_MSG(s != nullptr, "[TF.4 (b) %s, %s] setup: the arm did not build", cn.c_str(), an);
			if (!s) continue;
			const std::vector<uint8_t> bb = x.rig.Save(s);
			const Blob b = ParseBlob(bb);
			int32_t illegal = -1;
			for (int32_t t = 0; t < kTokVocab && illegal < 0; ++t) {
				uint32_t nx = 0;
				if (!f.Step(kPotion, b.walk, t, &nx)) illegal = t;
			}
			CHECK_MSG(b.ok && illegal >= 0, "[TF.4 (b) %s, %s] setup: no illegal token at walk %u", cn.c_str(), an, b.walk);
			if (!b.ok || illegal < 0) {
				x.rig.Release(s);
				continue;
			}
			sfd::ArmChoice(illegal);
			int32_t t = -100;
			const sslm_status st = x.Decode(s, 3, &t);
			const bool pin_left = sfd::ChoiceArmed();
			sfd::DisarmChoice();
			const Blob a = x.rig.State(s);
			const int64_t ctx = b.ctx + (arm == TfArm::kLayer ? 1 : 0);
			CHECK_MSG(st == SSLM_OK && t == -2 && !pin_left, "[TF.4 (b) %s, %s] the dead end: status %d token %d, pin left "
			          "%d", cn.c_str(), an, static_cast<int>(st), t, pin_left);
			CHECK_MSG(a.ok && a.order == 3 && a.hist.empty() && a.ready == 1 && a.layer == 0 && a.walk == b.walk &&
			              a.ctx == ctx,
			          "[TF.4 (b) %s, %s] after the -2 decode: %s; want order 3, empty history, ready, layer 0, walk %u, "
			          "ctx %lld (before %s)", cn.c_str(), an, TfFields(x.rig.Save(s)).c_str(), b.walk,
			          static_cast<long long>(ctx), TfFields(bb).c_str());
			int32_t nt = -100;
			uint32_t nx = 0;
			const sslm_status ns = x.Decode(s, 3, &nt);
			CHECK_MSG(ns == SSLM_OK && nt >= 0 && f.Step(kPotion, b.walk, nt, &nx) && x.rig.State(s).order == 3,
			          "[TF.4 (b) %s, %s] the next decode at order 3: status %d token %d", cn.c_str(), an,
			          static_cast<int>(ns), nt);
			x.rig.Release(s);
		}
	}
}

// J1-X (plan rev 1.3.3 §5.10, §11.4): the exact jump-forward cells.
#include "schema_fast_decode/sfd_cells_x.inc"

// Q6C.7 [Q6 C] One thread, no hook (the box's 1-thread jump-forward-on arm; A/B 2026-10-20): the §5.6 host
// loop on POTION and LONGKEY at budget 64, with the workspace and with ws = NULL. Every verb call that admits
// M >= 1 rows makes exactly one S-LAYERROWS entry, a batched pass of M rows, and the stream is the
// reference rule's.
void Cell_Q6C_7() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const char* schema : {kPotion, kLongKey}) {
		for (int with_ws = 0; with_ws < 2; ++with_ws) {
			Rig rig(f.m1->m);
			JfOpts o = GreedyOpts(rig);
			if (!with_ws) o.w = nullptr;
			int bad = 0, calls = 0, multi = 0;
			std::string first_bad;
			o.after = [&](sslm_seq&, const JfStep& st) {
				if (st.kind == 0 && st.st == SSLM_OK) {
					const size_t m = st.toks.size() + (st.embedded_pending ? 1 : 0);
					const std::vector<sfd::LayerRowsEntry> passes = Q6cPasses();
					if (m >= 1) {
						++calls;
						if (m >= 2) ++multi;
						if (!Q6cShapeIs(passes, m, false) && bad++ == 0)
							first_bad = "M " + std::to_string(m) + ": " + Q6cShape(passes);
					}
				}
				sfd::LayerRowsLogReset();
			};
			sslm_seq s = rig.NewSeq(schema);
			if (!s || !rig.PromptOk(s, f.prompt_potion)) continue;
			sfd::LayerRowsLogReset();
			const JfRun r = RunJf(rig.m, s, o);
			CheckStreamIsReference("Q6C.7", f, schema, r);
			CHECK_MSG(calls > 0 && multi > 0 && bad == 0, "[Q6C.7] %s, %s, no hook: %d of %d verb calls (%d of M >= 2) "
			          "did not make exactly one batched pass of M rows (first: %s)", schema, with_ws ? "workspace" : "ws NULL",
			          bad, calls, multi, first_bad.c_str());
		}
	}
}

// Prints the values cell 6.6 pins, computed by the reference replay on main (SUPERSLM_SFD_PRINT_GOLDENS=1).
void PrintGoldens() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	{
		Rig rig(f.m1->m);
		const std::vector<uint8_t> b = MainShapeBlob(rig, f);
		std::printf("SFD GOLDEN main blob: %zu bytes, digest %s\n", b.size(), BytesDigest(b).c_str());
	}
	{
		const RefSchema& r = f.R(kPotion);
		const std::vector<int32_t> head = {r.st[r.start].chain.empty() ? -1 : r.st[r.start].chain[0]};
		const Replay rp = ReferenceReplay(f.m1->m, f, kPotion, f.prompt_potion, nullptr, 400, head);
		std::printf("SFD GOLDEN broken stream: ok %d dead_end %d length %zu digest %s\n  stream %s\n", rp.ok, rp.dead_end,
		            rp.stream.size(), HexDigest(rp.stream).c_str(), Str(rp.stream, 400).c_str());
	}
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
		std::printf("[schema fast-decode] cell %-6s %s (%.0f ms)\n", id, GFailures == before ? "green" : "RED", ms);
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
	run("2.7", Cell_2_7);
	run("3.2", Cell_3_2);
	run("3.3", Cell_3_3);
	run("3.4", Cell_3_4);
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
	// §16 Q6 C (k = 7)
	run("Q6C.1", Cell_Q6C_1);
	run("Q6C.2", Cell_Q6C_2);
	run("Q6C.3", Cell_Q6C_3);
	run("Q6C.4", Cell_Q6C_4);
	run("Q6C.5", Cell_Q6C_5);
	run("Q6C.6", Cell_Q6C_6);
	run("Q6C.7", Cell_Q6C_7);
	// TF (plan rev 1.3.3 §11.5; TF.1, TF.3-TF.7 are in the allocation sweep, TF.8 is the cells above)
	run("TF.2", Cell_TF_2);
	run("TF.4b", Cell_TF_4b);
	// J1-X (plan rev 1.3.3 §5.10, §11.4; red until exact mode lands: sfd_red_stubs_x.cpp)
	run("6.X1", Cell_X_6_1);
	run("6.X2", Cell_X_6_2);
	run("6.X3", Cell_X_6_3);
	run("6.X5", Cell_X_6_5);
	run("6.X6", Cell_X_6_6);
	run("6.X7", Cell_X_6_7);
	run("1.X1", Cell_X_1_1);
	run("1.X2", Cell_X_1_2);
	run("1.X3", Cell_X_1_3);
	run("1.X4", Cell_X_1_4);
	run("2.X1", Cell_X_2_1);
	run("2.X2", Cell_X_2_2);
	run("2.X3", Cell_X_2_3);
	run("3.X1", Cell_X_3_1);
	run("3.X2", Cell_X_3_2);
	run("3.X3", Cell_X_3_3);
	run("4.X1", Cell_X_4_1);
	run("4.X2", Cell_X_4_2);
	run("5.X1", Cell_X_5_1);
	run("5.X2", Cell_X_5_2);
	run("5.X3", Cell_X_5_3);
	run("5.X4", Cell_X_5_4);
	run("5.X5", Cell_X_5_5);
	run("5.X6", Cell_X_5_6);
	run("5.X7", Cell_X_5_7);
	run("8.X1", Cell_X_8_1);
	run("8.X2", Cell_X_8_2);
	run("8.X3", Cell_X_8_3);
	run("8.X4", Cell_X_8_4);
	run("8.X5", Cell_X_8_5);
	run("8.X6", Cell_X_8_6);
	run("8.X7", Cell_X_8_7);
	run("9.X1", Cell_X_9_1);
	run("9.X2", Cell_X_9_2);
	run("10.X1", Cell_X_10_1);
	run("11.X1", Cell_X_11_1);
	run("11.X2", Cell_X_11_2);
	run("11.X3", Cell_X_11_3);
	run("TF.2d", Cell_X_TF_2d);
	run("TF.4bX", Cell_X_TF_4b);
	std::printf("schema fast-decode (M1, J1, TF, J1-X) cells: %d checks, %d failures\n", GChecks.load(), GFailures.load());
	checks += GChecks;
	failures += GFailures;
}
