// T-2899 (Curie) -- plan `Claude/Plans/te266-gpu-path.md` Sec3.10.7 (`sslm_seq_adopt_prefix`'s
// stale-walk state on adoption) and Sec3.5 step 1's own owed §3.10.7 cell, PLUS the cell the
// TE-364 adversary re-strike owes (`Claude/Loki/te364-gpu-plan-restrike-2026-09-21.md` Sec6):
// an adopt_prefix row where the reused sequence has already had schema content prefilled,
// killing the deletion of the `forced_token_count = 0` reset (that deletion fails 12 of 60
// rows).
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED. This cell is TE-364's own instrument
// (`Claude/Loki/te364-probe/te364_census.cpp`), filed here as the formal red suite cell it was
// always meant to become -- TE-364 §6 names it "a ready red candidate for the test author."
// TE-364's own population is a strict superset of T-2898's own 16-row census (the population
// this plan's Sec3.5 step 1 names as "authored as a formal red cell"): 10 adopting-sequence
// resting states x 3 prefix kinds x {live, restored} = 60 rows, against 8 states x
// {live,restored} = 16. Adopting the wider census as the ONE formal cell realizes both the
// plan's own Sec3.10.7 obligation and the gap TE-364 found in one file, rather than filing two
// cells that overlap on 16 of 60 rows.
//
// FIXTURE. --model=PATH, a real artifact carrying the `g5_minimal_one_field` schema (the
// project's own C39 synthetic, `tests/t2199-damped-greedy-red-suite`'s own
// `t2199_s8_fixture.sslm`, SHA-256 `a8f88ab9e82f220bf5fdffa68d4fac4cf248d0158a4ae9dc557eeedd
// 51df8097` -- 8 layers, vocabulary 128) or a production-scale artifact carrying
// `shopkeeper_intent_extraction` (`--schema=shopkeeper_intent_extraction`, the G5 suite's own
// argv artifact, `t2132_g5_fixture_1p5b.sslm`). NOTE ON REPRODUCTION: at authoring time,
// re-running `tools/_t2199_s8_synthetic_full_model_fixture.py` from this checkout does NOT
// reproduce the pinned `a8f88ab9...8097` byte-for-byte -- it instead produces a
// deterministic-but-different `2ea564f7...4930` (verified: two independent local runs of the
// unmodified, unchanged-since-2026-09-15 script agree with each other, but not with the
// hash every Loki/Vitruvius probe since T-2866 has pinned). This is a pre-existing generator
// reproducibility gap, not introduced by this cell, and it is not this ticket's charge to
// close (T-2899 is the Sec3.5 step 1 red suite, not the fixture generator's own audit) -- filed
// here as a plain fact for whoever next depends on this fixture reproducing from a clean
// checkout. The verified `a8f88ab9...8097` bytes are what every existing pinned Sec3.10 probe
// output (`Claude/Vitruvius/t2866-probe/` onward) was captured against, so this cell targets
// the SAME bytes; if only the freshly-generated `2ea564f7...4930` fixture is available, this
// cell still runs (the schema and its topology are unaffected -- TE-364's own census ran the
// same cross product against a copy of the same pinned bytes), it simply will not reproduce a
// cited SHA-256 mismatch as a SETUP failure below.
//
// GRADED PER ROW, against a FRESH sequence with the same schema binding adopting the same
// prefix: adoption status; the whole `sslm_seq_save` blob; `sslm_stats`'s
// `forced_token_count` and `schema_accepting`; the next 4 greedy tokens. A row is MATCH only
// if every quantity is equal.
//
// RED AT a3f89cb (pristine Stage 1, no fix landed): 40 of 60 rows MATCH, 20 MISMATCH --
// 16 stale-walk rows (the withdrawn-guard defect Sec3.10.7 fixes) plus 4 more the wider
// prefix-kind sweep reaches that T-2898's own narrower census did not exercise (all under
// `P_PROMPT`/`P_BOUND_START`, `B_ADVANCED`/`B_FORCED`/`B_DEADEND`/`B_FORCED_DEAD`). Confirmed
// at authoring time against both fixtures (D:/_t2899/te364-adopt-prefix-run/, this seat's own
// re-run of the identical program from this filed location).
//
// GUARD VITALITY, two independent mutants (both against Sec3.10.7's own reference fix,
// `Claude/Vitruvius/t2898-probe/sslm_abi_cpu_fixed_v5.cpp`):
//   MUTANT 1 (Sec3.10.3 row 11's own named mutant, T-2898): revert ONLY the `dfa_walk_state`
//     reset at `sslm_seq_adopt_prefix`'s `prefix_has_real_progress == false` branch to a no-op.
//     Must turn exactly the 4 BOUND_ADVANCED rows (2 prefix kinds with no progress x
//     live/restored) red on the `next4`/`schema_accepting` fields, no other row disturbed.
//   MUTANT 2 (TE-364's own finding, Sec6 -- this cell's OWN reason for existing): revert ONLY
//     the `forced_token_count = 0` line the SAME function sets. Must turn exactly the 12
//     B_FORCED/B_FORCED_DEAD rows (3 prefix kinds x live/restored x 2 states) red on
//     `stats.forced_token_count`, no other row disturbed.
//
// Run: cell_adopt_prefix_census.exe --model=PATH [--schema=NAME] [--pk=0|1|2]
#include "sslm_phaseD_stub.h"
#include "sslm_phaseD_fixture.h"
#include "sslm_fixtures.h"
#include "sslm_damped_greedy.h"
#include "superslm/decode_digest.h"
#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/schema_masks.h"
#include "superslm/sha256.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond)                                                                    \
	do {                                                                                \
		++GChecks;                                                                     \
		if (!(cond)) {                                                                 \
			++GFailures;                                                               \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
			std::fflush(stdout);                                                       \
		}                                                                               \
	} while (0)

using namespace superslm;
using namespace superslm_test;
using namespace superslm_test_phaseD;

static std::string g_model_path;

static bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	if (sz > 0) {
		const size_t n = std::fread(out->data(), 1, static_cast<size_t>(sz), f);
		std::fclose(f);
		if (n != static_cast<size_t>(sz)) return false;
	} else {
		std::fclose(f);
	}
	return true;
}

struct AlignedBuffer {
	explicit AlignedBuffer(size_t n)
	    : bytes_(n), storage_(n > 0 ? ::operator new(n, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)) : nullptr) {}
	~AlignedBuffer() {
		if (storage_) ::operator delete(storage_, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer& operator=(const AlignedBuffer&) = delete;
	void* data() { return storage_; }
	size_t size() const { return bytes_; }

  private:
	size_t bytes_;
	void* storage_;
};

struct RealModelFixture {
	std::vector<uint8_t> bytes;
	sslm_model model = nullptr;
	std::unique_ptr<AlignedBuffer> pool_buf;
	sslm_kv_pool pool = nullptr;
	int32_t vocab_size = 0;
	uint32_t num_hidden_layers = 0;

	~RealModelFixture() {
		if (pool) sslm_kv_pool_destroy(pool);
		if (model) sslm_model_unmap(model);
	}
};

static bool LoadRealModel(RealModelFixture* out, std::string* err) {
	if (!ReadFileBytes(g_model_path, &out->bytes)) {
		if (err) *err = "could not read " + g_model_path;
		return false;
	}
	superslm::SslmModelView view;
	if (superslm::SslmModel::Load(out->bytes.data(), out->bytes.size(), view, err) !=
	    superslm::SslmModelStatus::Ok) {
		return false;
	}
	out->vocab_size = view.config.vocab_size;
	out->num_hidden_layers = view.config.num_hidden_layers;
	if (sslm_model_map(out->bytes.data(), out->bytes.size(), &out->model) != SSLM_OK) {
		if (err) *err = "sslm_model_map failed";
		return false;
	}
	return true;
}

static bool PrefillPrompt(const RealModelFixture& fx, sslm_seq seq) {
	const int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	return sslm_prefill(fx.model, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK &&
	       consumed == 4;
}

static bool EnterMidToken(sslm_model model, sslm_seq seq) {
	sslm_decode_params params{};
	params.struct_size = sizeof(params);
	params.layer_budget = 1;
	sslm_seq batch[1] = {seq};
	int32_t out_token = 0;
	if (sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) != SSLM_OK) return false;
	out_token = 0;
	if (sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) != SSLM_OK) return false;
	return out_token < 0;
}

static std::vector<uint8_t> SaveSequence(sslm_seq seq) {
	size_t required = 0;
	CHECK(sslm_seq_save(seq, nullptr, &required) == SSLM_BUFFER_TOO_SMALL);
	std::vector<uint8_t> blob(required);
	size_t written = blob.size();
	CHECK(sslm_seq_save(seq, blob.data(), &written) == SSLM_OK);
	blob.resize(written);
	return blob;
}

static std::vector<int32_t> Greedy(const RealModelFixture& fx, sslm_seq seq, int steps, int32_t budget) {
	sslm_decode_params p{};
	p.struct_size = sizeof(p);
	p.layer_budget = budget;
	sslm_seq batch[1] = {seq};
	std::vector<int32_t> t;
	for (int i = 0; i < steps; ++i) {
		int32_t tok = -7;
		const sslm_status st = sslm_decode_step(fx.model, batch, 1, &p, nullptr, &tok);
		t.push_back(st == SSLM_OK ? tok : -1000 - static_cast<int>(st));
	}
	return t;
}

static sslm_seq Restore(RealModelFixture& fx, sslm_seq src) {
	std::vector<uint8_t> b = SaveSequence(src);
	sslm_seq r = nullptr;
	const sslm_status st = sslm_seq_restore(fx.model, &fx.pool, b.data(), b.size(), &r);
	if (st != SSLM_OK) {
		std::printf("  restore failed st=%d\n", static_cast<int>(st));
		return nullptr;
	}
	return r;
}

static std::string Tok(const std::vector<int32_t>& v) {
	std::string s;
	for (int32_t x : v) {
		s += std::to_string(x);
		s += ",";
	}
	return s;
}

struct Obs {
	sslm_status st;
	bool have;
	std::vector<uint8_t> blob;
	int64_t forced;
	int32_t acc;
	std::vector<int32_t> next;
};

static Obs Observe(RealModelFixture& fx, sslm_seq s, sslm_status st, int32_t L) {
	Obs o{};
	o.st = st;
	o.have = (st == SSLM_OK);
	if (!o.have) return o;
	o.blob = SaveSequence(s);
	sslm_stats_out so{};
	CHECK(sslm_stats(fx.model, s, &so) == SSLM_OK);
	o.forced = so.forced_token_count;
	o.acc = so.schema_accepting;
	o.next = Greedy(fx, s, 4, L);
	return o;
}

int main(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		const char* kFlag = "--model=";
		if (a.compare(0, std::strlen(kFlag), kFlag) == 0) g_model_path = a.c_str() + std::strlen(kFlag);
	}
	std::string schema_name = "g5_minimal_one_field";
	int only_pk = -1;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--schema=", 0) == 0) schema_name = a.substr(9);
		if (a.rfind("--pk=", 0) == 0) only_pk = std::atoi(a.c_str() + 5);
	}
	if (g_model_path.empty()) {
		std::printf("SKIP cell_adopt_prefix_census -- needs --model=PATH (a fixture carrying "
		            "the '%s' schema)\n",
		            schema_name.c_str());
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	RealModelFixture fx;
	std::string err;
	if (!LoadRealModel(&fx, &err)) {
		std::printf("FAIL cell_adopt_prefix_census -- load failed: %s\n", err.c_str());
		return 1;
	}
	{
		const uint32_t bc = 6;
		const size_t bb = sslm_kv_block_size(fx.model), ov = sslm_kv_pool_overhead_size(fx.model, bc);
		fx.pool_buf = std::make_unique<AlignedBuffer>(bc * bb + ov);
		if (sslm_kv_pool_create(fx.model, fx.pool_buf->data(), fx.pool_buf->size(), bc, &fx.pool) != SSLM_OK) {
			std::printf("FAIL cell_adopt_prefix_census -- sslm_kv_pool_create failed\n");
			return 1;
		}
	}
	const int32_t L = static_cast<int32_t>(fx.num_hidden_layers);
	sslm_schema schema = nullptr;
	if (sslm_schema_lookup(fx.model, schema_name.c_str(), &schema) != SSLM_OK) {
		std::printf("FAIL cell_adopt_prefix_census -- schema '%s' not found on --model\n", schema_name.c_str());
		return 1;
	}

	// Derive one schema-admitted token from the start state, on the real model and schema.
	int32_t admitted = -1;
	{
		sslm_seq d = nullptr;
		CHECK(sslm_seq_create(fx.model, &fx.pool, &d) == SSLM_OK);
		CHECK(sslm_seq_set_schema(d, schema) == SSLM_OK);
		CHECK(PrefillPrompt(fx, d));
		admitted = Greedy(fx, d, 1, L)[0];
		CHECK(sslm_seq_release(d) == SSLM_OK);
	}
	std::printf("layers=%d vocab=%d schema=%s admitted_from_start=%d\n", L, fx.vocab_size,
	            schema_name.c_str(), admitted);
	if (admitted < 0) {
		std::printf("FAIL cell_adopt_prefix_census -- no admitted token found from the start state\n");
		return 1;
	}

	enum PK { P_PROMPT, P_BOUND_START, P_PROGRESS };
	const char* pnames[] = {"P_PROMPT", "P_BOUND_START", "P_PROGRESS"};
	enum SK {
		FRESH,
		POST_PREFILL,
		POST_GREEDY,
		MIDTOKEN,
		B_FRESH,
		B_POST_PREFILL,
		B_ADVANCED,
		B_FORCED,
		B_DEADEND,
		B_FORCED_DEAD
	};
	const char* snames[] = {"FRESH",        "POST_PREFILL", "POST_GREEDY",  "MIDTOKEN",    "B_FRESH",
	                        "B_POST_PREFILL", "B_ADVANCED",   "B_FORCED",     "B_DEADEND",   "B_FORCED_DEAD"};
	int rows = 0, match = 0, mismatch = 0;
	int mismatch_walk = 0, mismatch_forced = 0;
	for (int pk = P_PROMPT; pk <= P_PROGRESS; ++pk) {
		if (only_pk >= 0 && pk != only_pk) continue;
		sslm_prefix prefix = nullptr;
		CHECK(sslm_prefix_begin(fx.model, &fx.pool, &prefix) == SSLM_OK);
		if (pk != P_PROMPT) CHECK(sslm_prefix_set_schema(prefix, schema) == SSLM_OK);
		const int32_t pp[4] = {0, 1, 2, 3};
		int32_t pc = 0;
		CHECK(sslm_prefix_prefill(fx.model, prefix, pp, 4, 8, SSLM_SPAN_PROMPT, nullptr, &pc) == SSLM_OK);
		if (pk == P_PROGRESS) {
			int32_t c2 = 0;
			CHECK(sslm_prefix_prefill(fx.model, prefix, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
			                           &c2) == SSLM_OK &&
			      c2 == 1);
		}
		CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);
		for (int sk = FRESH; sk <= B_FORCED_DEAD; ++sk) {
			const bool bound = sk >= B_FRESH;
			for (int restored = 0; restored <= 1; ++restored) {
				sslm_seq s = nullptr;
				CHECK(sslm_seq_create(fx.model, &fx.pool, &s) == SSLM_OK);
				if (bound) CHECK(sslm_seq_set_schema(s, schema) == SSLM_OK);
				switch (sk) {
					case FRESH:
					case B_FRESH:
						break;
					case POST_PREFILL:
					case B_POST_PREFILL:
						CHECK(PrefillPrompt(fx, s));
						break;
					case POST_GREEDY:
						CHECK(PrefillPrompt(fx, s));
						Greedy(fx, s, 2, L);
						break;
					case MIDTOKEN:
						CHECK(PrefillPrompt(fx, s));
						CHECK(EnterMidToken(fx.model, s));
						break;
					case B_ADVANCED:
						CHECK(PrefillPrompt(fx, s));
						Greedy(fx, s, 1, L);
						break;
					case B_FORCED: {
						CHECK(PrefillPrompt(fx, s));
						int32_t c = 0;
						CHECK(sslm_prefill(fx.model, s, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c) ==
						          SSLM_OK &&
						      c == 1);
						break;
					}
					case B_DEADEND:
						CHECK(PrefillPrompt(fx, s));
						Greedy(fx, s, 3, L);
						break;
					case B_FORCED_DEAD: {
						CHECK(PrefillPrompt(fx, s));
						int32_t c = 0;
						CHECK(sslm_prefill(fx.model, s, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c) ==
						          SSLM_OK &&
						      c == 1);
						Greedy(fx, s, 2, L);
						break;
					}
				}
				int64_t forced_before = -1;
				{
					sslm_stats_out so{};
					if (sslm_stats(fx.model, s, &so) == SSLM_OK) forced_before = so.forced_token_count;
				}
				sslm_seq target = s;
				if (restored) {
					target = Restore(fx, s);
					CHECK(sslm_seq_release(s) == SSLM_OK);
					if (!target) continue;
				}
				const sslm_status st = sslm_seq_adopt_prefix(target, prefix);
				const Obs got = Observe(fx, target, st, L);
				sslm_seq ref = nullptr;
				CHECK(sslm_seq_create(fx.model, &fx.pool, &ref) == SSLM_OK);
				if (bound) CHECK(sslm_seq_set_schema(ref, schema) == SSLM_OK);
				const sslm_status rst = sslm_seq_adopt_prefix(ref, prefix);
				const Obs want = Observe(fx, ref, rst, L);
				std::string why;
				bool walk_bad = false, forced_bad = false;
				if (got.st != want.st) why += " status";
				if (got.have && want.have) {
					if (got.blob != want.blob) why += " blob";
					if (got.next != want.next) {
						why += " next4";
						walk_bad = true;
					}
					if (got.acc != want.acc) {
						why += " schema_accepting";
						walk_bad = true;
					}
					if (got.forced != want.forced) {
						why += " forced_token_count(" + std::to_string(got.forced) + "vs" +
						       std::to_string(want.forced) + ")";
						forced_bad = true;
					}
				}
				++rows;
				if (why.empty()) {
					++match;
				} else {
					++mismatch;
					if (walk_bad) ++mismatch_walk;
					if (forced_bad) ++mismatch_forced;
				}
				std::printf(
				    "%-13s %-14s %-8s forced_before=%lld adopt=%d ref=%d next4[%s] ref[%s] "
				    "forced=%lld/%lld acc=%d/%d  %s\n",
				    pnames[pk], snames[sk], restored ? "restored" : "live",
				    static_cast<long long>(forced_before), static_cast<int>(got.st), static_cast<int>(want.st),
				    Tok(got.next).c_str(), Tok(want.next).c_str(), static_cast<long long>(got.forced),
				    static_cast<long long>(want.forced), got.acc, want.acc,
				    why.empty() ? "MATCH" : ("MISMATCH:" + why).c_str());
				CHECK(sslm_seq_release(ref) == SSLM_OK);
				CHECK(sslm_seq_release(target) == SSLM_OK);
			}
		}
		CHECK(sslm_prefix_release(prefix) == SSLM_OK);
	}
	std::printf(
	    "SUMMARY rows=%d match=%d mismatch=%d mismatch_walk=%d mismatch_forced=%d harness_checks=%d "
	    "harness_failures=%d\n",
	    rows, match, mismatch, mismatch_walk, mismatch_forced, GChecks, GFailures);
	// This cell's own acceptance bar (Sec3.10.7's promise + TE-364's own gap closed): 60 of 60
	// MATCH, zero harness failures. Pristine and the withdrawn-guard/mutant runs are expected
	// to print a nonzero mismatch count -- the mutant runner (run_mutants.bat) checks THOSE
	// counts against the specific rows named above, not against exit code 0.
	const bool acceptance = (rows == 60) && (mismatch == 0) && (GFailures == 0);
	std::printf("checks=%d failures=%d skips=0 rows=%d mismatch=%d acceptance=%d\n", GChecks, GFailures, rows,
	            mismatch, acceptance ? 1 : 0);
	return (GFailures == 0 && rows > 0) ? 0 : 1;
}
