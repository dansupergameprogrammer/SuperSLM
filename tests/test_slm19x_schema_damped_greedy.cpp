// 1.9.x C-ABI cells for schema-constrained decoding (include/superslm/schema_masks.h) and damped
// greedy decoding (src/damped_greedy_antilm.cpp, src/damped_greedy_topk.cpp,
// src/damped_greedy_phaseD.cpp). Every cell reaches those files the way a caller does: by mapping
// an artifact, binding a schema, prefilling and decoding through the C ABI, and it asserts what
// that caller observes -- the tokens returned, the statuses, the stats -- never only that code ran.
//
// Fixtures. The model is the committed tests/fixtures/t2572_arm_c_non_qknorm_fixture.sslm (vocab 32,
// 2 layers, context cap 16), the same artifact the 1.8.x saturation census cells map. Its variants
// are built in memory by re-emitting its sections through tests/sslm_fixtures.h's BuildArtifact and
// appending:
//   - a SchemaMasks (SCM1) section written by this file's own writer from a hand-stated automaton,
//     independent of SchemaMasksTable::Parse, so the automaton the cells replay is the one stated
//     here and not one read back from the code under test;
//   - a DampedGreedyConstants (DGC1) section carrying tools/convert_model.py's own production pair
//     (DAMPED_GREEDY_SCALE_M = 2883584, DAMPED_GREEDY_SCALE_E = -36) with the header flag bit, i.e.
//     exactly what `convert_model.py --enable-damped-greedy` writes.
// No generated binary is committed and the CI coverage job needs no generator step.
//
// A separate translation unit for the same reason as tests/test_slm18x_saturation_census.cpp: the C
// ABI's unscoped sslm_status enumerators collide with test_main.cpp's `using enum SslmGpuStatus;` on
// Windows. test_main.cpp calls RunSlm19xSchemaDampedGreedyCells and adds this unit's counts.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <new>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/intmath.h"
#include "superslm/layer_marshal.h"
#include "superslm/sslm_abi.h"
#include "sslm_fixtures.h"
#include "sslm_tokenizer_fixtures.h"

using superslm_test::BuildArtifact;
using superslm_test::FixtureSection;
using superslm_test::GetU32;
using superslm_test::GetU64;
using superslm_test::PutU32;
using superslm_test::PutU64;
using superslm_test::RecomputeIntegrityHash;
using superslm_test::ResolveFixturePath;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

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

// ---- the base fixture and its in-memory variants -------------------------------------------

constexpr int32_t kVocab = 32;            // CFG1 vocab_size of the t2572 fixture
constexpr int32_t kLayers = 2;            // CFG1 num_hidden_layers

// tools/convert_model.py's production DGC1 pair and the triple its own comment states the
// certified i-exp derivation produces from it: "q=(493, 964, 487361)".
constexpr int64_t kConverterScaleM = 2883584;
constexpr int32_t kConverterScaleE = -36;
constexpr int64_t kConverterQLn2 = 493;
constexpr int64_t kConverterQB = 964;
constexpr int64_t kConverterQC = 487361;

constexpr int32_t kAlphaMax = (1 << 20) - 1;  // the ABI's documented [0, 2^20) ceiling, inclusive top

struct RawSec {
	uint32_t type;
	uint32_t dtype;
	uint32_t alignment;
	std::vector<uint8_t> data;
};

// Reads the committed fixture's section table directly (docs/sslm_format.md byte layout).
bool LoadBaseSections(std::vector<RawSec>* out, uint32_t* out_flags) {
	const std::string path = ResolveFixturePath("t2572_arm_c_non_qknorm_fixture.sslm");
	CHECK_MSG(!path.empty(), "fixture t2572_arm_c_non_qknorm_fixture.sslm not found");
	if (path.empty()) return false;
	std::vector<uint8_t> bytes;
	CHECK_MSG(superslm_marshal::ReadFile(path.c_str(), bytes), "failed to read %s", path.c_str());
	if (bytes.size() < superslm::kHeaderBytes) return false;
	const uint32_t count = GetU32(bytes, 12);
	*out_flags = GetU32(bytes, 16);
	out->clear();
	for (uint32_t i = 0; i < count; ++i) {
		const size_t row = superslm::kHeaderBytes + static_cast<size_t>(i) * superslm::kSectionDescBytes;
		RawSec s;
		s.type = GetU32(bytes, row + 0);
		s.dtype = GetU32(bytes, row + 4);
		const uint64_t off = GetU64(bytes, row + 8);
		const uint64_t len = GetU64(bytes, row + 16);
		s.alignment = GetU32(bytes, row + 32);
		if (off + len > bytes.size()) return false;
		s.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(off),
		              bytes.begin() + static_cast<std::ptrdiff_t>(off + len));
		out->push_back(std::move(s));
	}
	return true;
}

// Re-emits the base fixture's sections plus `extra`, with header flags `flags`.
std::vector<uint8_t> BuildVariant(const std::vector<FixtureSection>& extra, uint32_t extra_flags) {
	std::vector<RawSec> base;
	uint32_t base_flags = 0;
	if (!LoadBaseSections(&base, &base_flags)) return {};
	std::vector<FixtureSection> secs;
	for (const RawSec& r : base) {
		FixtureSection f;
		f.type = r.type;
		f.dtype = r.dtype;
		f.alignment = r.alignment;
		f.data = r.data;
		secs.push_back(std::move(f));
	}
	for (const FixtureSection& e : extra) secs.push_back(e);
	auto built = BuildArtifact(secs);
	PutU32(built.bytes, 16, base_flags | extra_flags);
	RecomputeIntegrityHash(built.bytes);
	return built.bytes;
}

FixtureSection RawSection(superslm::SslmSectionType type, std::vector<uint8_t> data) {
	FixtureSection f;
	f.type = static_cast<uint32_t>(type);
	f.dtype = static_cast<uint32_t>(superslm::SslmDtype::Raw);
	f.alignment = 8;
	f.data = std::move(data);
	return f;
}

std::vector<uint8_t> Dgc1Bytes(int64_t m, int32_t e) {
	std::vector<uint8_t> b(12, 0);
	PutU64(b, 0, static_cast<uint64_t>(m));
	PutU32(b, 8, static_cast<uint32_t>(e));
	return b;
}

// ---- SCM1 writer (independent of SchemaMasksTable::Parse) -----------------------------------

struct SchemaSpec {
	std::string name;
	uint32_t state_count;
	std::vector<uint32_t> accepting;
	// rows[state] = ascending (token, next_state) pairs
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> rows;
};

struct Scm1Layout {
	std::vector<size_t> desc_off;       // descriptor row start, per schema
	uint64_t name_blob_off = 0;
	std::vector<uint64_t> mask_off, offsets_off, accepting_off, transitions_off;
};

// Header (24) | descriptors (56 each) | name blob | per-schema blocks: mask pages, CSR state
// offsets, accepting states, transitions -- every multi-byte field little-endian.
std::vector<uint8_t> WriteScm1(const std::vector<SchemaSpec>& schemas, uint32_t vocab,
                               Scm1Layout* layout) {
	const uint32_t page = (vocab + 7) / 8;
	std::string blob;
	std::vector<uint32_t> name_off;
	for (const SchemaSpec& s : schemas) {
		name_off.push_back(static_cast<uint32_t>(blob.size()));
		blob += s.name;
	}
	const uint64_t manifest_end = 24 + 56 * schemas.size() + blob.size();
	uint64_t cursor = manifest_end;
	*layout = Scm1Layout{};
	layout->name_blob_off = 24 + 56 * schemas.size();
	struct Blocks {
		std::vector<uint8_t> mask, offs, acc, trans;
	};
	std::vector<Blocks> blocks;
	for (const SchemaSpec& s : schemas) {
		Blocks b;
		b.mask.assign(static_cast<size_t>(s.state_count) * page, 0);
		b.offs.assign((static_cast<size_t>(s.state_count) + 1) * 4, 0);
		uint32_t total = 0;
		for (uint32_t st = 0; st < s.state_count; ++st) {
			PutU32(b.offs, static_cast<size_t>(st) * 4, total);
			for (const auto& [tok, next] : s.rows[st]) {
				b.mask[static_cast<size_t>(st) * page + (tok >> 3)] |= static_cast<uint8_t>(1u << (tok & 7));
				std::vector<uint8_t> t(8);
				PutU32(t, 0, tok);
				PutU32(t, 4, next);
				b.trans.insert(b.trans.end(), t.begin(), t.end());
				++total;
			}
		}
		PutU32(b.offs, static_cast<size_t>(s.state_count) * 4, total);
		b.acc.assign(s.accepting.size() * 4, 0);
		for (size_t a = 0; a < s.accepting.size(); ++a) PutU32(b.acc, a * 4, s.accepting[a]);
		layout->mask_off.push_back(cursor);
		cursor += b.mask.size();
		layout->offsets_off.push_back(cursor);
		cursor += b.offs.size();
		layout->accepting_off.push_back(cursor);
		cursor += b.acc.size();
		layout->transitions_off.push_back(cursor);
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
		layout->desc_off.push_back(d);
		const SchemaSpec& s = schemas[i];
		uint32_t tc = 0;
		for (const auto& r : s.rows) tc += static_cast<uint32_t>(r.size());
		PutU32(out, d + 0, name_off[i]);
		PutU32(out, d + 4, static_cast<uint32_t>(s.name.size()));
		PutU32(out, d + 8, s.state_count);
		PutU32(out, d + 12, static_cast<uint32_t>(s.accepting.size()));
		PutU32(out, d + 16, tc);
		PutU32(out, d + 20, 0);
		PutU64(out, d + 24, layout->mask_off[i]);
		PutU64(out, d + 32, layout->offsets_off[i]);
		PutU64(out, d + 40, layout->accepting_off[i]);
		PutU64(out, d + 48, layout->transitions_off[i]);
	}
	std::memcpy(out.data() + layout->name_blob_off, blob.data(), blob.size());
	for (size_t i = 0; i < schemas.size(); ++i) {
		const Blocks& b = blocks[i];
		std::memcpy(out.data() + layout->mask_off[i], b.mask.data(), b.mask.size());
		std::memcpy(out.data() + layout->offsets_off[i], b.offs.data(), b.offs.size());
		if (!b.acc.empty()) std::memcpy(out.data() + layout->accepting_off[i], b.acc.data(), b.acc.size());
		if (!b.trans.empty())
			std::memcpy(out.data() + layout->transitions_off[i], b.trans.data(), b.trans.size());
	}
	return out;
}

// The two schemas every schema cell uses. Tokens are chosen so the unconstrained model does not
// emit them on its own at the same positions (each cell checks that precondition, so a forced
// outcome cannot be a coincidence of the logits).
//   s_ab : 0 --kForcedA--> 1 --kForcedB--> 2 ; accepting {2}; state 2 has no continuation.
//   s_any: 0 {1->1, 2->1, 3->0}, 1 {1->0, 4->0}; accepting {0, 1}.
constexpr uint32_t kForcedA = 5;
constexpr uint32_t kForcedB = 7;

SchemaSpec SchemaAB() {
	return SchemaSpec{"s_ab", 3, {2}, {{{kForcedA, 1}}, {{kForcedB, 2}}, {}}};
}
SchemaSpec SchemaAny() {
	return SchemaSpec{"s_any", 2, {0, 1}, {{{1, 1}, {2, 1}, {3, 0}}, {{1, 0}, {4, 0}}}};
}

// Independent walk replay over a stated automaton: true and *next set iff `tok` continues `state`.
bool SpecTransition(const SchemaSpec& s, uint32_t state, int32_t tok, uint32_t* next) {
	if (state >= s.state_count || tok < 0) return false;
	for (const auto& [t, n] : s.rows[state]) {
		if (t == static_cast<uint32_t>(tok)) {
			*next = n;
			return true;
		}
	}
	return false;
}
bool SpecAccepting(const SchemaSpec& s, uint32_t state) {
	for (uint32_t a : s.accepting)
		if (a == state) return true;
	return false;
}

std::vector<uint8_t> StandardScm1() {
	Scm1Layout layout;
	return WriteScm1({SchemaAB(), SchemaAny()}, kVocab, &layout);
}

// The fully-featured variant: both schemas and the converter's DGC1 section + flag.
std::vector<uint8_t> BuildFullVariant() {
	return BuildVariant({RawSection(superslm::SslmSectionType::SchemaMasks, StandardScm1()),
	                     RawSection(superslm::SslmSectionType::DampedGreedyConstants,
	                                Dgc1Bytes(kConverterScaleM, kConverterScaleE))},
	                    superslm::kDampedGreedyArtifactConstantsFlag);
}

// ---- mapped model + pool --------------------------------------------------------------------

struct Mapped {
	std::vector<uint8_t> bytes;  // must outlive `model`
	sslm_model model = nullptr;
	void* pool_buf = nullptr;
	sslm_kv_pool pool = nullptr;
	sslm_status map_status = SSLM_INVALID_ARGUMENT;

	bool Open(std::vector<uint8_t> b, uint32_t blocks = 4) {
		bytes = std::move(b);
		if (bytes.empty()) return false;
		map_status = sslm_model_map(bytes.data(), bytes.size(), &model);
		if (map_status != SSLM_OK) return false;
		const size_t size =
		    blocks * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, blocks);
		pool_buf = ::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		const sslm_status ps = sslm_kv_pool_create(model, pool_buf, size, blocks, &pool);
		CHECK_MSG(ps == SSLM_OK, "sslm_kv_pool_create == %d", static_cast<int>(ps));
		return ps == SSLM_OK;
	}
	~Mapped() {
		if (pool) CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
		if (pool_buf) ::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (model) CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
};

// A caller-owned workspace (sslm_workspace_create), for the cells that compare workspace and
// no-workspace decoding.
struct Workspace {
	void* buf = nullptr;
	sslm_workspace ws = nullptr;
	bool Create(sslm_model model) {
		sslm_config cfg{};
		cfg.max_batch = 1;
		cfg.max_chunk_budget = 16;
		cfg.max_layer_budget = kLayers;
		const size_t size = sslm_workspace_size(model, &cfg);
		if (size == 0) return false;
		buf = ::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		return sslm_workspace_create(model, &cfg, buf, size, &ws) == SSLM_OK;
	}
	~Workspace() {
		if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		if (buf) ::operator delete(buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
};

sslm_seq NewSeq(Mapped& m, sslm_schema schema = SSLM_SCHEMA_NONE) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(m.model, &m.pool, &seq) == SSLM_OK);
	if (seq && schema) CHECK(sslm_seq_set_schema(seq, schema) == SSLM_OK);
	return seq;
}

bool PromptPrefill(Mapped& m, sslm_seq seq, int32_t token = 0) {
	int32_t consumed = 0;
	const sslm_status st =
	    sslm_prefill(m.model, seq, &token, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed);
	CHECK_MSG(st == SSLM_OK && consumed == 1, "prompt prefill status %d consumed %d",
	          static_cast<int>(st), consumed);
	return st == SSLM_OK;
}

sslm_decode_params GreedyParams() {
	sslm_decode_params p{};
	p.struct_size = sizeof(p);
	p.layer_budget = kLayers;
	p.mode = SSLM_DECODE_MODE_GREEDY;
	return p;
}

sslm_decode_params DampedParams(int32_t alpha, int32_t order, int32_t top_k) {
	sslm_decode_params p = GreedyParams();
	p.mode = SSLM_DECODE_MODE_DAMPED_GREEDY;
	p.alpha_q15 = alpha;
	p.anti_lm_max_order = order;
	p.top_k = top_k;
	p.q_ln2 = kConverterQLn2;
	p.q_b = kConverterQB;
	p.q_c = kConverterQC;
	return p;
}

// Decodes `steps` tokens; every call must return SSLM_OK.
std::vector<int32_t> DecodeN(Mapped& m, sslm_seq seq, const sslm_decode_params& p, int steps,
                             sslm_workspace ws = nullptr) {
	std::vector<int32_t> out;
	for (int i = 0; i < steps; ++i) {
		int32_t tok = -100;
		const sslm_status st = sslm_decode_step_v2(m.model, &seq, 1, &p, ws, &tok);
		CHECK_MSG(st == SSLM_OK, "decode step %d status %d", i, static_cast<int>(st));
		if (st != SSLM_OK) break;
		out.push_back(tok);
	}
	return out;
}

std::string Join(const std::vector<int32_t>& v) {
	std::string s;
	for (int32_t t : v) s += std::to_string(t) + " ";
	return s;
}

bool HasDuplicate(const std::vector<int32_t>& v) {
	std::set<int32_t> seen;
	for (int32_t t : v)
		if (!seen.insert(t).second) return true;
	return false;
}

int32_t SchemaAccepting(Mapped& m, sslm_seq seq) {
	sslm_stats_out s{};
	CHECK(sslm_stats(m.model, seq, &s) == SSLM_OK);
	return s.schema_accepting;
}

// The unconstrained greedy continuation of the one-token prompt, used as the precondition every
// forcing cell checks against.
std::vector<int32_t> UnboundGreedy(Mapped& m, int steps) {
	sslm_seq seq = NewSeq(m);
	if (!seq) return {};
	std::vector<int32_t> toks;
	if (PromptPrefill(m, seq)) toks = DecodeN(m, seq, GreedyParams(), steps);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	return toks;
}

constexpr int kRunSteps = 12;  // 1 prompt position + 12 decoded tokens stays under context cap 16

}  // namespace

// ============================== schema_masks.h =================================================

// A model's compiled schemas are enumerable by index and resolvable by exact name; a model with no
// SchemaMasks section exposes none.
static void TestSlm19x_SchemaSetEnumeratesAndResolvesByExactName() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) {
		CHECK_MSG(false, "full variant failed to map: %d", static_cast<int>(m.map_status));
		return;
	}
	CHECK(sslm_schema_count(m.model) == 2);
	const char* want[2] = {"s_ab", "s_any"};
	for (int32_t i = 0; i < 2; ++i) {
		char buf[16] = {};
		size_t n = sizeof(buf);
		CHECK(sslm_schema_name(m.model, i, buf, &n) == SSLM_OK);
		CHECK_MSG(n == std::strlen(want[i]) && std::memcmp(buf, want[i], n) == 0,
		          "schema %d name '%.*s'", i, static_cast<int>(n), buf);
	}
	sslm_schema ab = nullptr, any = nullptr, miss = reinterpret_cast<sslm_schema>(1);
	CHECK(sslm_schema_lookup(m.model, "s_ab", &ab) == SSLM_OK && ab != nullptr);
	CHECK(sslm_schema_lookup(m.model, "s_any", &any) == SSLM_OK && any != nullptr);
	CHECK(ab != any);
	CHECK(sslm_schema_lookup(m.model, "s_a", &miss) == SSLM_SCHEMA_NOT_FOUND && miss == nullptr);
	CHECK(sslm_schema_lookup(m.model, "S_AB", &miss) == SSLM_SCHEMA_NOT_FOUND);
	CHECK(sslm_schema_lookup(m.model, "s_any_", &miss) == SSLM_SCHEMA_NOT_FOUND);

	Mapped plain;
	if (!plain.Open(BuildVariant({}, 0))) return;
	CHECK(sslm_schema_count(plain.model) == 0);
	CHECK(sslm_schema_lookup(plain.model, "s_ab", &miss) == SSLM_SCHEMA_NOT_FOUND);
}

// Greedy decoding under a bound schema emits exactly the schema's language: the two forced tokens,
// then -- at the accepting state with no continuation -- the -2 dead end, repeatably.
// schema_accepting reports the automaton's own accept set at every step.
static void TestSlm19x_GreedyDecodeFollowsSchemaToItsDeadEnd() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	const std::vector<int32_t> free_run = UnboundGreedy(m, 2);
	CHECK_MSG(free_run.size() == 2 && !(free_run[0] == static_cast<int32_t>(kForcedA) &&
	                                     free_run[1] == static_cast<int32_t>(kForcedB)),
	          "unconstrained greedy already emits the forced pair (%s); the cell would not "
	          "discriminate",
	          Join(free_run).c_str());
	sslm_schema ab = nullptr;
	CHECK(sslm_schema_lookup(m.model, "s_ab", &ab) == SSLM_OK);
	sslm_seq seq = NewSeq(m, ab);
	if (!seq) return;
	int32_t bound = 0;
	CHECK(sslm_seq_schema_bound(seq, &bound) == SSLM_OK && bound == 1);
	if (PromptPrefill(m, seq)) {
		CHECK(SchemaAccepting(m, seq) == 0);
		const std::vector<int32_t> toks = DecodeN(m, seq, GreedyParams(), 4);
		CHECK_MSG(toks == (std::vector<int32_t>{static_cast<int32_t>(kForcedA),
		                                        static_cast<int32_t>(kForcedB), -2, -2}),
		          "bound greedy emitted %s, want %u %u -2 -2", Join(toks).c_str(), kForcedA, kForcedB);
		CHECK(SchemaAccepting(m, seq) == 1);
		// The dead end rests the sequence so it can be reset and reused (the header's contract).
		CHECK(sslm_seq_reset(seq) == SSLM_OK);
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// Every token greedy decoding emits under s_any is a legal continuation of the automaton stated in
// this file, replayed independently step by step -- while the unconstrained model, over the same
// run, emits tokens outside that language (so the constraint is doing the work).
static void TestSlm19x_GreedyDecodeStaysInsideTheSchemaLanguage() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	const std::vector<int32_t> free_run = UnboundGreedy(m, kRunSteps);
	bool free_leaves_language = false;
	for (int32_t t : free_run)
		if (t < 1 || t > 4) free_leaves_language = true;
	CHECK_MSG(free_leaves_language, "unconstrained greedy never leaves {1,2,3,4} (%s)",
	          Join(free_run).c_str());

	const SchemaSpec spec = SchemaAny();
	sslm_schema any = nullptr;
	CHECK(sslm_schema_lookup(m.model, "s_any", &any) == SSLM_OK);
	sslm_seq seq = NewSeq(m, any);
	if (!seq) return;
	if (PromptPrefill(m, seq)) {
		uint32_t state = 0;
		for (int i = 0; i < kRunSteps; ++i) {
			const std::vector<int32_t> one = DecodeN(m, seq, GreedyParams(), 1);
			if (one.size() != 1) break;
			uint32_t next = 0;
			CHECK_MSG(SpecTransition(spec, state, one[0], &next),
			          "step %d: token %d is not a continuation of state %u", i, one[0], state);
			state = next;
			CHECK(SchemaAccepting(m, seq) == (SpecAccepting(spec, state) ? 1 : 0));
		}
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// A schema-content span is admitted exactly as far as the automaton reaches: a legal span is
// consumed whole and counted as forced; a span with an unreachable token is admitted up to it and
// refused there with SSLM_SCHEMA_SPAN_UNREACHABLE, and decoding resumes from the admitted state.
static void TestSlm19x_SchemaContentPrefillAdmitsOnlyReachableSpans() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	sslm_schema ab = nullptr;
	CHECK(sslm_schema_lookup(m.model, "s_ab", &ab) == SSLM_OK);

	sslm_seq a = NewSeq(m, ab);
	if (a && PromptPrefill(m, a)) {
		const int32_t span[2] = {static_cast<int32_t>(kForcedA), static_cast<int32_t>(kForcedB)};
		int32_t consumed = -1;
		CHECK(sslm_prefill(m.model, a, span, 2, 2, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) ==
		      SSLM_OK);
		CHECK(consumed == 2);
		sslm_stats_out s{};
		CHECK(sslm_stats(m.model, a, &s) == SSLM_OK);
		CHECK(s.forced_token_count == 2);
		CHECK(s.schema_accepting == 1);
		CHECK(DecodeN(m, a, GreedyParams(), 1) == std::vector<int32_t>{-2});
	}
	if (a) CHECK(sslm_seq_release(a) == SSLM_OK);

	sslm_seq b = NewSeq(m, ab);
	if (b && PromptPrefill(m, b)) {
		const int32_t span[3] = {static_cast<int32_t>(kForcedA), 9, static_cast<int32_t>(kForcedB)};
		int32_t consumed = -1;
		CHECK(sslm_prefill(m.model, b, span, 3, 3, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) ==
		      SSLM_SCHEMA_SPAN_UNREACHABLE);
		CHECK_MSG(consumed == 1, "consumed %d, want 1 (the token before the unreachable one)",
		          consumed);
		CHECK(SchemaAccepting(m, b) == 0);
		CHECK_MSG(DecodeN(m, b, GreedyParams(), 1) ==
		              std::vector<int32_t>{static_cast<int32_t>(kForcedB)},
		          "decode after a partial span does not resume at the admitted state");
	}
	if (b) CHECK(sslm_seq_release(b) == SSLM_OK);
}

// A bound sequence saved mid-generation restores bound to the same schema at the same walk state:
// the restored sequence and the original continue with identical, legal tokens. A blob whose saved
// schema name hash names no schema of this model is refused.
static void TestSlm19x_SchemaBindingSurvivesSaveRestore() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	const SchemaSpec spec = SchemaAny();
	sslm_schema any = nullptr;
	CHECK(sslm_schema_lookup(m.model, "s_any", &any) == SSLM_OK);
	sslm_seq orig = NewSeq(m, any);
	if (!orig || !PromptPrefill(m, orig)) return;
	const std::vector<int32_t> head = DecodeN(m, orig, GreedyParams(), 3);
	uint32_t state = 0;
	for (int32_t t : head) {
		uint32_t next = 0;
		CHECK(SpecTransition(spec, state, t, &next));
		state = next;
	}
	std::vector<uint8_t> blob(sslm_seq_state_size(m.model));
	size_t n = blob.size();
	CHECK(sslm_seq_save(orig, blob.data(), &n) == SSLM_OK);
	blob.resize(n);

	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(m.model, &m.pool, blob.data(), blob.size(), &restored) == SSLM_OK);
	if (restored) {
		int32_t bound = 0;
		CHECK(sslm_seq_schema_bound(restored, &bound) == SSLM_OK && bound == 1);
		const std::vector<int32_t> tail_o = DecodeN(m, orig, GreedyParams(), 4);
		const std::vector<int32_t> tail_r = DecodeN(m, restored, GreedyParams(), 4);
		CHECK_MSG(tail_o == tail_r, "original continued %s, restored %s", Join(tail_o).c_str(),
		          Join(tail_r).c_str());
		uint32_t s2 = state;
		for (int32_t t : tail_r) {
			uint32_t next = 0;
			CHECK_MSG(SpecTransition(spec, s2, t, &next), "restored emitted illegal token %d", t);
			s2 = next;
		}
		CHECK(sslm_seq_release(restored) == SSLM_OK);
	}

	// Offset 40 of the blob is its saved schema-name hash (src/sslm_abi.cpp, sslm_seq_restore).
	std::vector<uint8_t> tampered = blob;
	PutU64(tampered, 40, GetU64(tampered, 40) ^ 0x1ull);
	sslm_seq none = reinterpret_cast<sslm_seq>(1);
	CHECK(sslm_seq_restore(m.model, &m.pool, tampered.data(), tampered.size(), &none) ==
	      SSLM_RESTORE_SCHEMA_MISMATCH);
	CHECK(sslm_seq_release(orig) == SSLM_OK);
}

// Every structural and cross-check defect a SchemaMasks section can carry is refused at map time
// with SSLM_ARTIFACT_REJECTED and no model handle -- never a partially-usable model. The unmutated
// section maps (the must-accept half), so each refusal is the mutation's.
static void TestSlm19x_MalformedSchemaSectionIsRefusedAtMap() {
	Scm1Layout L;
	const std::vector<uint8_t> good = WriteScm1({SchemaAB(), SchemaAny()}, kVocab, &L);
	auto map_scm1 = [&](const std::vector<uint8_t>& scm1, sslm_model* out) {
		static std::vector<uint8_t> keep;  // artifact bytes must outlive a successful map
		keep = BuildVariant({RawSection(superslm::SslmSectionType::SchemaMasks, scm1)}, 0);
		*out = nullptr;
		return sslm_model_map(keep.data(), keep.size(), out);
	};
	{
		sslm_model ok = nullptr;
		CHECK(map_scm1(good, &ok) == SSLM_OK);
		if (ok) {
			CHECK(sslm_schema_count(ok) == 2);
			CHECK(sslm_model_unmap(ok) == SSLM_OK);
		}
	}
	const size_t d0 = L.desc_off[0], d1 = L.desc_off[1];
	const uint64_t size = good.size();
	struct Mut {
		const char* name;
		std::function<void(std::vector<uint8_t>&)> apply;
	};
	const std::vector<Mut> muts = {
	    {"truncated below the fixed header", [](auto& b) { b.resize(20); }},
	    {"bad magic", [](auto& b) { b[0] = 'X'; }},
	    {"version 2", [](auto& b) { PutU32(b, 4, 2); }},
	    {"header reserved != 0", [](auto& b) { PutU32(b, 20, 1); }},
	    {"schema_count 0", [](auto& b) { PutU32(b, 8, 0); }},
	    {"schema_count above the 65536 bound", [](auto& b) { PutU32(b, 8, 65537); }},
	    {"vocab_size 0", [](auto& b) { PutU32(b, 12, 0); }},
	    {"vocab_size disagrees with the model config", [](auto& b) { PutU32(b, 12, kVocab + 1); }},
	    {"name blob runs past the section", [](auto& b) { PutU32(b, 16, 0xFFFFFFF0u); }},
	    {"descriptor reserved != 0", [&](auto& b) { PutU32(b, d0 + 20, 1); }},
	    {"empty name", [&](auto& b) { PutU32(b, d0 + 4, 0); }},
	    {"name runs past the name blob", [&](auto& b) { PutU32(b, d1 + 4, 64); }},
	    {"duplicate name", [&](auto& b) { PutU32(b, d1 + 0, GetU32(b, d0 + 0)); PutU32(b, d1 + 4, GetU32(b, d0 + 4)); }},
	    {"state_count 0", [&](auto& b) { PutU32(b, d0 + 8, 0); }},
	    {"state_count above the 2^20 bound", [&](auto& b) { PutU32(b, d0 + 8, 1048577); }},
	    {"more accepting states than states", [&](auto& b) { PutU32(b, d0 + 12, 4); }},
	    {"transition_count above the 2^24 bound", [&](auto& b) { PutU32(b, d0 + 16, 16777217); }},
	    {"block placed inside the manifest", [&](auto& b) { PutU64(b, d0 + 24, 0); }},
	    {"block ends past the section", [&](auto& b) { PutU64(b, d0 + 24, size); }},
	    {"block offset + length overflows", [&](auto& b) { PutU64(b, d0 + 48, UINT64_MAX - 3); }},
	    {"two blocks of one schema overlap", [&](auto& b) { PutU64(b, d0 + 40, L.offsets_off[0]); }},
	    {"blocks of two schemas overlap", [&](auto& b) { PutU64(b, d1 + 24, L.mask_off[0]); }},
	    {"state offsets do not start at 0", [&](auto& b) { PutU32(b, L.offsets_off[0], 1); }},
	    {"state offsets do not end at transition_count", [&](auto& b) { PutU32(b, L.offsets_off[0] + 12, 1); }},
	    {"state offsets decrease", [&](auto& b) { PutU32(b, L.offsets_off[0] + 4, 2); PutU32(b, L.offsets_off[0] + 8, 1); }},
	    {"accepting state out of range", [&](auto& b) { PutU32(b, L.accepting_off[0], 3); }},
	    {"accepting states not ascending", [&](auto& b) { PutU32(b, L.accepting_off[1], 1); PutU32(b, L.accepting_off[1] + 4, 0); }},
	    {"transition token outside the vocabulary", [&](auto& b) { PutU32(b, L.transitions_off[0], kVocab); }},
	    {"transition target outside the states", [&](auto& b) { PutU32(b, L.transitions_off[0] + 4, 3); }},
	    {"row tokens not ascending", [&](auto& b) { PutU32(b, L.transitions_off[1], 2); PutU32(b, L.transitions_off[1] + 8, 1); }},
	    {"mask bit set with no transition", [&](auto& b) { b[L.mask_off[0] + 1] |= 0x02; }},
	    {"transition with no mask bit", [&](auto& b) { b[L.mask_off[0] + (kForcedA >> 3)] &= static_cast<uint8_t>(~(1u << (kForcedA & 7))); }},
	};
	for (const Mut& mut : muts) {
		std::vector<uint8_t> bad = good;
		mut.apply(bad);
		sslm_model out = nullptr;
		const sslm_status st = map_scm1(bad, &out);
		CHECK_MSG(st == SSLM_ARTIFACT_REJECTED && out == nullptr, "%s: map returned %d", mut.name,
		          static_cast<int>(st));
		if (out) sslm_model_unmap(out);
	}
}

// ============================== damped greedy =================================================

// Damped greedy is available exactly when the artifact carries a well-formed DGC1 section and its
// flag: sslm_decode_params_init then fills the documented defaults and the scale triple the
// converter's pair derives. Every other combination either refuses the mode or refuses the artifact.
static void TestSlm19x_DampedGreedyAvailabilityFollowsTheArtifact() {
	{
		Mapped full;
		if (full.Open(BuildFullVariant())) {
			sslm_decode_params p{};
			CHECK(sslm_decode_params_init(full.model, SSLM_DECODE_MODE_DAMPED_GREEDY, kLayers, &p) ==
			      SSLM_OK);
			CHECK(p.struct_size == sizeof(p) && p.layer_budget == kLayers &&
			      p.mode == SSLM_DECODE_MODE_DAMPED_GREEDY);
			CHECK(p.alpha_q15 == SSLM_DAMPED_GREEDY_DEFAULT_ALPHA_Q15);
			CHECK(p.anti_lm_max_order == SSLM_DAMPED_GREEDY_DEFAULT_ANTI_LM_ORDER);
			CHECK(p.top_k == SSLM_DAMPED_GREEDY_DEFAULT_TOP_K);
			CHECK_MSG(p.q_ln2 == kConverterQLn2 && p.q_b == kConverterQB && p.q_c == kConverterQC,
			          "q = (%lld, %lld, %lld), want (493, 964, 487361)",
			          static_cast<long long>(p.q_ln2), static_cast<long long>(p.q_b),
			          static_cast<long long>(p.q_c));
			// The initialized params decode.
			sslm_seq seq = NewSeq(full);
			if (seq && PromptPrefill(full, seq)) {
				const std::vector<int32_t> t = DecodeN(full, seq, p, 3);
				CHECK(t.size() == 3);
				for (int32_t x : t) CHECK(x >= 0 && x < kVocab);
			}
			if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
		}
	}
	{
		// No DGC1 at all: the mode is refused both at init and at decode.
		Mapped plain;
		if (plain.Open(BuildVariant({}, 0))) {
			sslm_decode_params p{};
			CHECK(sslm_decode_params_init(plain.model, SSLM_DECODE_MODE_DAMPED_GREEDY, kLayers, &p) ==
			      SSLM_ARTIFACT_REJECTED);
			CHECK(sslm_decode_params_init(plain.model, SSLM_DECODE_MODE_GREEDY, kLayers, &p) == SSLM_OK);
			CHECK(p.mode == SSLM_DECODE_MODE_GREEDY);
			sslm_seq seq = NewSeq(plain);
			if (seq && PromptPrefill(plain, seq)) {
				const sslm_decode_params dp = DampedParams(65536, 2, 6);
				int32_t tok = -100;
				CHECK(sslm_decode_step_v2(plain.model, &seq, 1, &dp, nullptr, &tok) ==
				      SSLM_ARTIFACT_REJECTED);
			}
			if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
		}
	}
	{
		// The section without its flag bit: the artifact maps, the mode stays unavailable.
		Mapped noflag;
		if (noflag.Open(BuildVariant({RawSection(superslm::SslmSectionType::DampedGreedyConstants,
		                                         Dgc1Bytes(kConverterScaleM, kConverterScaleE))},
		                             0))) {
			sslm_decode_params p{};
			CHECK(sslm_decode_params_init(noflag.model, SSLM_DECODE_MODE_DAMPED_GREEDY, kLayers, &p) ==
			      SSLM_ARTIFACT_REJECTED);
		} else {
			CHECK_MSG(false, "DGC1 section without its flag failed to map: %d",
			          static_cast<int>(noflag.map_status));
		}
	}
	struct Bad {
		const char* name;
		std::vector<FixtureSection> extra;
	};
	const std::vector<Bad> bad = {
	    {"flag set, no DGC1 section", {}},
	    {"DGC1 of 16 bytes", {RawSection(superslm::SslmSectionType::DampedGreedyConstants,
	                                     std::vector<uint8_t>(16, 0))}},
	    // e = -200 drives every i-exp derivation shift negative (IExpScaleConstants' kNegativeShift).
	    {"DGC1 scale outside the i-exp domain",
	     {RawSection(superslm::SslmSectionType::DampedGreedyConstants,
	                 Dgc1Bytes(kConverterScaleM, -200))}},
	    // (m=0, e=0) derives q = (0, 0, 0): IExpScaleConstants and the width check both accept it,
	    // but no decode can use it (IExpConstruct refuses q_ln2 = 0). An out-of-domain DGC1 is a
	    // defined rejection at read (damped_greedy_phaseD.cpp), which sslm_model_map reports as
	    // SSLM_ARTIFACT_REJECTED -- never a model that advertises damped greedy and then refuses
	    // every damped decode made with the params sslm_decode_params_init hands back.
	    {"DGC1 scale whose derived constants no decode accepts (m=0, e=0)",
	     {RawSection(superslm::SslmSectionType::DampedGreedyConstants, Dgc1Bytes(0, 0))}},
	};
	for (const Bad& b : bad) {
		std::vector<uint8_t> bytes = BuildVariant(b.extra, superslm::kDampedGreedyArtifactConstantsFlag);
		sslm_model out = nullptr;
		const sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &out);
		CHECK_MSG(st == SSLM_ARTIFACT_REJECTED && out == nullptr, "%s: map returned %d", b.name,
		          static_cast<int>(st));
		if (out) sslm_model_unmap(out);
	}
}

// Caller-supplied scale constants that the softmax width check accepts but whose i-exp peak is
// unusable are refused by decode itself with SSLM_INVALID_ARGUMENT, before any sequence is
// touched: the out token is not written and the sequence then decodes exactly as a pristine twin.
// (q_ln2 = 493, q_b = q_c = 0): M = 0 passes the width check, the i-exp construction succeeds, and
// the peak evaluates to 0. (q_ln2 = q_b = q_c = 0): the construction refuses q_ln2 = 0. Each triple's
// width acceptance is asserted first, so a pass cannot come from the earlier width refusal.
static void TestSlm19x_DampedGreedyUnusableScaleConstantsAreRefusedAtDecode() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	sslm_seq twin = NewSeq(m);
	sslm_seq seq = NewSeq(m);
	if (!twin || !seq || !PromptPrefill(m, twin) || !PromptPrefill(m, seq)) return;
	const sslm_decode_params good = DampedParams(65536, 2, 6);
	const int64_t triples[2][3] = {{kConverterQLn2, 0, 0}, {0, 0, 0}};
	for (const auto& q : triples) {
		sslm_decode_params p = good;
		p.q_ln2 = q[0];
		p.q_b = q[1];
		p.q_c = q[2];
		CHECK_MSG(superslm::CheckSoftmaxRowWidthDomain(p.q_b, p.q_c, static_cast<size_t>(p.top_k)) ==
		              superslm::SslmForwardStatus::Ok,
		          "q = (%lld, %lld, %lld) is refused by the width check; the cell would not reach "
		          "the peak check",
		          static_cast<long long>(q[0]), static_cast<long long>(q[1]),
		          static_cast<long long>(q[2]));
		int32_t tok = -100;
		const sslm_status st = sslm_decode_step_v2(m.model, &seq, 1, &p, nullptr, &tok);
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "q = (%lld, %lld, %lld): status %d",
		          static_cast<long long>(q[0]), static_cast<long long>(q[1]),
		          static_cast<long long>(q[2]), static_cast<int>(st));
		CHECK_MSG(tok == -100, "q = (%lld, %lld, %lld): out token written (%d)",
		          static_cast<long long>(q[0]), static_cast<long long>(q[1]),
		          static_cast<long long>(q[2]), tok);
	}
	const std::vector<int32_t> a = DecodeN(m, seq, good, 3);
	const std::vector<int32_t> b = DecodeN(m, twin, good, 3);
	CHECK_MSG(a == b, "after peak refusals %s, pristine twin %s", Join(a).c_str(), Join(b).c_str());
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_seq_release(twin) == SSLM_OK);
}

// Out-of-domain damped-greedy fields are refused with SSLM_INVALID_ARGUMENT and leave the sequence
// exactly as it was: after every refusal it decodes the same token a pristine twin decodes. The
// domain's inclusive edges are accepted. Greedy mode ignores the damped-only fields entirely.
static void TestSlm19x_DampedGreedyParamsOutsideTheirDomainAreRefusedHarmlessly() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	sslm_seq twin = NewSeq(m);
	sslm_seq seq = NewSeq(m);
	if (!twin || !seq || !PromptPrefill(m, twin) || !PromptPrefill(m, seq)) return;
	const sslm_decode_params good = DampedParams(65536, 2, 6);

	struct Case {
		const char* name;
		sslm_decode_params p;
	};
	std::vector<Case> refused;
	auto with = [&](const char* name, auto edit) {
		sslm_decode_params p = good;
		edit(p);
		refused.push_back({name, p});
	};
	with("unknown mode", [](sslm_decode_params& p) { p.mode = 7; });
	with("negative alpha", [](sslm_decode_params& p) { p.alpha_q15 = -1; });
	with("alpha at 2^20", [](sslm_decode_params& p) { p.alpha_q15 = 1 << 20; });
	with("anti-LM order 0", [](sslm_decode_params& p) { p.anti_lm_max_order = 0; });
	with("anti-LM order 83", [](sslm_decode_params& p) { p.anti_lm_max_order = 83; });
	with("top_k 0", [](sslm_decode_params& p) { p.top_k = 0; });
	with("top_k above the vocabulary", [](sslm_decode_params& p) { p.top_k = kVocab + 1; });
	for (const Case& c : refused) {
		int32_t tok = -100;
		const sslm_status st = sslm_decode_step_v2(m.model, &seq, 1, &c.p, nullptr, &tok);
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "%s: status %d", c.name, static_cast<int>(st));
		CHECK_MSG(tok == -100, "%s: out token written (%d)", c.name, tok);
	}
	const std::vector<int32_t> a = DecodeN(m, seq, good, 2);
	const std::vector<int32_t> b = DecodeN(m, twin, good, 2);
	CHECK_MSG(a == b, "after refusals %s, pristine twin %s", Join(a).c_str(), Join(b).c_str());

	// Inclusive edges decode.
	for (const sslm_decode_params& p :
	     {DampedParams(kAlphaMax, 2, 6), DampedParams(65536, 82, 6), DampedParams(65536, 2, kVocab),
	      DampedParams(0, 1, 1)}) {
		int32_t tok = -100;
		CHECK(sslm_decode_step_v2(m.model, &seq, 1, &p, nullptr, &tok) == SSLM_OK);
		CHECK(tok >= 0 && tok < kVocab);
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_seq_release(twin) == SSLM_OK);

	// Greedy mode with garbage in every damped-only field decodes exactly as legacy greedy.
	sslm_seq g1 = NewSeq(m);
	sslm_seq g2 = NewSeq(m);
	if (g1 && g2 && PromptPrefill(m, g1) && PromptPrefill(m, g2)) {
		sslm_decode_params junk = GreedyParams();
		junk.alpha_q15 = -5;
		junk.anti_lm_max_order = -9;
		junk.top_k = 0;
		const std::vector<int32_t> x = DecodeN(m, g1, junk, 3);
		std::vector<int32_t> y;
		for (int i = 0; i < 3; ++i) {
			sslm_decode_params legacy{};
			legacy.layer_budget = kLayers;
			int32_t tok = -100;
			CHECK(sslm_decode_step(m.model, &g2, 1, &legacy, nullptr, &tok) == SSLM_OK);
			y.push_back(tok);
		}
		CHECK_MSG(x == y, "greedy with damped junk %s, legacy greedy %s", Join(x).c_str(),
		          Join(y).c_str());
	}
	if (g1) CHECK(sslm_seq_release(g1) == SSLM_OK);
	if (g2) CHECK(sslm_seq_release(g2) == SSLM_OK);
}

// alpha = 0 removes the anti-LM term, and the most probable token is always among the top k, so
// damped greedy selects what greedy selects at every k: the token streams agree at k = 1 (the
// argmax alone), a small k, and the whole vocabulary. (Measured on this fixture at
// q = (493, 964, 487361).)
static void TestSlm19x_DampedGreedyAlphaZeroSelectsWhatGreedySelects() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	const std::vector<int32_t> greedy = UnboundGreedy(m, kRunSteps);
	for (int32_t k : {1, 6, kVocab}) {
		sslm_seq seq = NewSeq(m);
		if (seq && PromptPrefill(m, seq)) {
			const std::vector<int32_t> damped = DecodeN(m, seq, DampedParams(0, 2, k), kRunSteps);
			CHECK_MSG(damped == greedy, "alpha=0 k=%d damped %s, greedy %s", k,
			          Join(damped).c_str(), Join(greedy).c_str());
		}
		if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
	}
}

// The anti-LM's promise, from its definition: with an order-1 anti-LM, p_omega of a token already
// emitted once in a t-token history is floor(2^15 / t); at alpha = 2^20 - 1 and t <= 15 its penalty
// is at least ((2^20 - 1) * 2184) >> 15 = 69887, above any renormalized q_theta (<= 2^15), while an
// unseen token's score is >= 0. So over the whole vocabulary no token is ever emitted twice in a
// run shorter than the vocabulary -- where greedy, on the same prompt, repeats.
static void TestSlm19x_DampedGreedyAntiLmStopsRepetition() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	const std::vector<int32_t> greedy = UnboundGreedy(m, kRunSteps);
	CHECK_MSG(HasDuplicate(greedy), "greedy never repeats on this prompt (%s); the cell would not "
	          "discriminate", Join(greedy).c_str());
	sslm_seq seq = NewSeq(m);
	if (seq && PromptPrefill(m, seq)) {
		const std::vector<int32_t> damped =
		    DecodeN(m, seq, DampedParams(kAlphaMax, 1, kVocab), kRunSteps);
		CHECK(damped.size() == static_cast<size_t>(kRunSteps));
		CHECK_MSG(!HasDuplicate(damped), "damped greedy repeated a token: %s", Join(damped).c_str());
		CHECK(damped != greedy);
	}
	if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// Decoding is a function of the sequence and the params alone: the caller's workspace (which
// selects the scratch-buffer scoring path) and a second independent sequence change no token, for
// the default operating point, a higher-order anti-LM, and a small top_k.
static void TestSlm19x_DampedGreedyTokensDoNotDependOnWorkspaceOrRun() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	Workspace ws;
	CHECK(ws.Create(m.model));
	for (const sslm_decode_params& p : {DampedParams(65536, 2, 6), DampedParams(kAlphaMax, 3, 6),
	                                    DampedParams(kAlphaMax, 5, 2)}) {
		sslm_seq a = NewSeq(m), b = NewSeq(m), c = NewSeq(m);
		if (a && b && c && PromptPrefill(m, a) && PromptPrefill(m, b) && PromptPrefill(m, c)) {
			const std::vector<int32_t> ta = DecodeN(m, a, p, kRunSteps, nullptr);
			const std::vector<int32_t> tb = DecodeN(m, b, p, kRunSteps, ws.ws);
			const std::vector<int32_t> tc = DecodeN(m, c, p, kRunSteps, nullptr);
			CHECK_MSG(ta == tb, "order %d k %d: no-workspace %s, workspace %s", p.anti_lm_max_order,
			          p.top_k, Join(ta).c_str(), Join(tb).c_str());
			CHECK(ta == tc);
			for (int32_t t : ta) CHECK(t >= 0 && t < kVocab);
		}
		for (sslm_seq s : {a, b, c})
			if (s) CHECK(sslm_seq_release(s) == SSLM_OK);
	}
}

// The anti-LM's history travels with a saved sequence: the restored sequence continues with exactly
// the original's tokens, and -- under the no-repeat operating point -- never re-emits a token from
// before the save, which a restore that dropped the history would.
static void TestSlm19x_DampedGreedyAntiLmHistorySurvivesSaveRestore() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	const sslm_decode_params p = DampedParams(kAlphaMax, 1, kVocab);
	sslm_seq orig = NewSeq(m);
	if (!orig || !PromptPrefill(m, orig)) return;
	const std::vector<int32_t> head = DecodeN(m, orig, p, 4);
	std::vector<uint8_t> blob(sslm_seq_state_size(m.model));
	size_t n = blob.size();
	CHECK(sslm_seq_save(orig, blob.data(), &n) == SSLM_OK);
	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(m.model, &m.pool, blob.data(), n, &restored) == SSLM_OK);
	if (restored) {
		const std::vector<int32_t> tail_o = DecodeN(m, orig, p, 6);
		const std::vector<int32_t> tail_r = DecodeN(m, restored, p, 6);
		CHECK_MSG(tail_o == tail_r, "original continued %s, restored %s", Join(tail_o).c_str(),
		          Join(tail_r).c_str());
		std::vector<int32_t> all = head;
		all.insert(all.end(), tail_r.begin(), tail_r.end());
		CHECK_MSG(!HasDuplicate(all), "restored run re-emitted a pre-save token: %s | %s",
		          Join(head).c_str(), Join(tail_r).c_str());
		CHECK(sslm_seq_release(restored) == SSLM_OK);
	}
	CHECK(sslm_seq_release(orig) == SSLM_OK);
}

// Damped greedy scores the schema-masked row: under s_any every token is a legal continuation of the
// stated automaton, at the whole vocabulary and at a top_k smaller than the legal set; under s_ab it
// emits the forced pair and then the -2 dead end, repeatably.
static void TestSlm19x_DampedGreedyObeysTheBoundSchema() {
	Mapped m;
	if (!m.Open(BuildFullVariant())) return;
	const SchemaSpec spec = SchemaAny();
	sslm_schema any = nullptr, ab = nullptr;
	CHECK(sslm_schema_lookup(m.model, "s_any", &any) == SSLM_OK);
	CHECK(sslm_schema_lookup(m.model, "s_ab", &ab) == SSLM_OK);
	for (const sslm_decode_params& p :
	     {DampedParams(kAlphaMax, 1, kVocab), DampedParams(65536, 2, 2), DampedParams(65536, 2, 6)}) {
		sslm_seq seq = NewSeq(m, any);
		if (seq && PromptPrefill(m, seq)) {
			const std::vector<int32_t> toks = DecodeN(m, seq, p, kRunSteps);
			CHECK(toks.size() == static_cast<size_t>(kRunSteps));
			uint32_t state = 0;
			for (int32_t t : toks) {
				uint32_t next = 0;
				CHECK_MSG(SpecTransition(spec, state, t, &next),
				          "k %d: token %d is not a continuation of state %u (%s)", p.top_k, t, state,
				          Join(toks).c_str());
				state = next;
			}
		}
		if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
	}
	sslm_seq seq = NewSeq(m, ab);
	if (seq && PromptPrefill(m, seq)) {
		const std::vector<int32_t> toks = DecodeN(m, seq, DampedParams(65536, 2, 6), 4);
		CHECK_MSG(toks == (std::vector<int32_t>{static_cast<int32_t>(kForcedA),
		                                        static_cast<int32_t>(kForcedB), -2, -2}),
		          "bound damped greedy emitted %s", Join(toks).c_str());
		CHECK(SchemaAccepting(m, seq) == 1);
	}
	if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
}

void RunSlm19xSchemaDampedGreedyCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestSlm19x_SchemaSetEnumeratesAndResolvesByExactName();
	TestSlm19x_GreedyDecodeFollowsSchemaToItsDeadEnd();
	TestSlm19x_GreedyDecodeStaysInsideTheSchemaLanguage();
	TestSlm19x_SchemaContentPrefillAdmitsOnlyReachableSpans();
	TestSlm19x_SchemaBindingSurvivesSaveRestore();
	TestSlm19x_MalformedSchemaSectionIsRefusedAtMap();
	TestSlm19x_DampedGreedyAvailabilityFollowsTheArtifact();
	TestSlm19x_DampedGreedyParamsOutsideTheirDomainAreRefusedHarmlessly();
	TestSlm19x_DampedGreedyUnusableScaleConstantsAreRefusedAtDecode();
	TestSlm19x_DampedGreedyAlphaZeroSelectsWhatGreedySelects();
	TestSlm19x_DampedGreedyAntiLmStopsRepetition();
	TestSlm19x_DampedGreedyTokensDoNotDependOnWorkspaceOrRun();
	TestSlm19x_DampedGreedyAntiLmHistorySurvivesSaveRestore();
	TestSlm19x_DampedGreedyObeysTheBoundSchema();
	checks += GChecks;
	failures += GFailures;
}
