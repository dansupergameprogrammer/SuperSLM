// Paged-KV plan (rev 16.2) §8: the debunker's commissioning constructions for three of the plan's
// test-side instruments (the fourth, the timing harness, is exercised through its own cells, 7.4 and
// 7.9, by run_commissioning.py). README.md beside this file lists every construction and the
// verdict each must draw.
//
// These cells are NOT part of the red suite and are never globbed into it: run_commissioning.py
// builds them only in a scratch copy of the tree (build_scratch appends one target to the copy's
// paged_kv.cmake), linked with the unmodified runner pkv_main.cpp. Every cell here calls the
// instrument under commission exactly as a suite cell would, and contributes no verdict of its
// own beyond the procedure §8 states for that instrument:
//   - byte-equality oracle: RunScenario + ExpectMatchesReference (v1.11.0), and Pin + RefLookup;
//   - legacy-create admission count: CountLegacyCreates at P and P - 1, expecting n and n - 1;
//   - two-sided fill probe: ProbeExactlyFree; one-state form: ProbeExactlyFreeOneState.
// Whether a construction is a must-accept or a must-reject is decided by the state it builds (an
// exact state, a claim off by one page, or a mutant library), never by anything in these cells.
//
// Authored blind: from §3.1, §3.4 and §8 of the plan and from pkv_common.h's code, without reading
// the builder's self-test or any cell's mutant notes.

#include "pkv_common.h"

#include <memory>
#include <string>
#include <vector>

namespace {

using namespace pkv;

// ---- 1. the byte-equality oracle ---------------------------------------------------------------

const char* const kRefTag = "v1.11.0";

// Every scenario of pkv_scenarios.h on one fixture, graded record for record against the v1.11.0
// reference, tokens, rows and blob (a legacy holder writes 1.9.0's SSB5, so the blob compares too).
void OracleFixture(const char* stem) {
	const Fixture& fx = GetFixture(stem);
	if (!fx.ok) return;
	size_t n = 0;
	const Scenario* sc = Scenarios(&n);
	for (size_t i = 0; i < n; ++i) {
		const std::vector<Record> got = RunScenario(fx, sc[i].name);
		ExpectMatchesReference(kRefTag, fx, sc[i].name, got, Compare::kTokensRowsAndBlob);
	}
}
void OracleDef() { OracleFixture("pkv_def"); }
void OracleQk() { OracleFixture("pkv_qk"); }
void OracleOdd() { OracleFixture("pkv_odd"); }
void Oracle32k() { OracleFixture("pkv_32k"); }

// One scenario only, for the tampered-reference constructions (each names the record it tampers).
void OracleDefLifecycle() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	ExpectMatchesReference(kRefTag, fx, "lifecycle", RunScenario(fx, "lifecycle"), Compare::kTokensRowsAndBlob);
}
void OracleQkPersist() {
	const Fixture& fx = GetFixture("pkv_qk");
	if (!fx.ok) return;
	ExpectMatchesReference(kRefTag, fx, "persist", RunScenario(fx, "persist"), Compare::kTokensRowsAndBlob);
}

// The pinned-blob half (9.2's shape): restore a v1.11.0 pin on the build under test, save it, and
// decode 8. The reference's own restore-then-save of the same blob is the record "<scenario>
// restored" (persist) and the save itself (saturating's "saturated"); the continuation is the
// "...continuation8" record. Graded field by field against RefLookup, with the same three
// comparisons ExpectMatchesReference makes (tokens, context length, rows, blob).
void PinReplay(const char* scenario, const char* pin_stage, const char* blob_stage, const char* cont_stage) {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const RefFile& ref = Reference(kRefTag, fx);
	const std::vector<uint8_t> blob = Pin(kRefTag, scenario, pin_stage);
	const RefRecord* want_blob = RefLookup(ref, scenario, blob_stage);
	const RefRecord* want_cont = RefLookup(ref, scenario, cont_stage);
	if (blob.empty() || !want_blob || !want_cont) return;
	Driver d;
	d.model = fx.model;
	d.geo = fx.geo;
	Driver::Pool pool;
	if (!d.MakePool(1, &pool)) {
		PKV_CHECK_MSG(false, "pin replay %s: %s", scenario, d.fail.c_str());
		return;
	}
	sslm_seq s = nullptr;
	const sslm_status st = sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &s);
	PKV_CHECK_MSG(st == SSLM_OK, "pin replay %s: restore -> %d", scenario, static_cast<int>(st));
	if (st != SSLM_OK) return;
	const bool ok = d.Mark(blob_stage, s) && d.Decode(s, 8) && d.Mark(cont_stage, nullptr);
	PKV_CHECK_MSG(ok && d.records.size() == 2, "pin replay %s: %s", scenario, d.fail.c_str());
	sslm_seq_release(s);
	if (!ok || d.records.size() != 2) return;
	const Record& b = d.records[0];
	const Record& c = d.records[1];
	PKV_CHECK_MSG(b.context_length == want_blob->context_length, "%s/%s: L %lld vs %lld", scenario, blob_stage,
	              static_cast<long long>(b.context_length), static_cast<long long>(want_blob->context_length));
	PKV_CHECK_MSG(b.rows_sha == want_blob->rows_sha, "%s/%s on pkv_def: K/V rows differ", scenario, blob_stage);
	PKV_CHECK_MSG(b.blob_sha == want_blob->blob_sha, "%s/%s on pkv_def: blob differs", scenario, blob_stage);
	PKV_CHECK_MSG(c.tokens == want_cont->tokens, "%s/%s on pkv_def: tokens differ", scenario, cont_stage);
}
void PinPersist() { PinReplay("persist", "saved", "restored", "restored+continuation8"); }
void PinSaturating() { PinReplay("saturating", "saturated", "saturated", "continuation8"); }

PKV_CELL("CM.oracle:pkv_def", "CM", OracleDef);
PKV_CELL("CM.oracle:pkv_qk", "CM", OracleQk);
PKV_CELL("CM.oracle:pkv_odd", "CM", OracleOdd);
PKV_CELL("CM.oracle:pkv_32k", "CM", Oracle32k);
PKV_CELL("CM.oracle.one:pkv_def/lifecycle", "CM", OracleDefLifecycle);
PKV_CELL("CM.oracle.one:pkv_qk/persist", "CM", OracleQkPersist);
PKV_CELL("CM.oracle.pin:persist", "CM", PinPersist);
PKV_CELL("CM.oracle.pin:saturating", "CM", PinSaturating);

// ---- the state every page-count construction probes ----------------------------------------------
//
// A page pool of `free_pages + kHeld` pages on pkv_def (B = 16, ceil(cap/B) = 256). A budget
// sequence of budget 96 holds R(96) = ceil(96/16) + 1 = 7 pages for the state's whole life (§3.4,
// written here). Then one 2-page create (budget B) is made and released: on the correct build the
// pool is back to exactly `free_pages` free, and on a mutant library that release is where a
// one-page leak (deficit) or a one-page double free (excess) happens -- the real release path,
// not a test-side stand-in.

constexpr int64_t kHeld = 7;
constexpr int32_t kHeldBudget = 96;

struct PoolState : ProbeState {
	std::unique_ptr<PagePool> pool;  // declared first: destroyed after the handles
	Handles h;
	bool ok = false;
	sslm_kv_pool* Pool() override { return &pool->pool; }
};

std::unique_ptr<PoolState> BuildState(const Fixture& fx, int64_t free_pages) {
	auto s = std::make_unique<PoolState>();
	s->pool = std::make_unique<PagePool>(fx.model, static_cast<uint32_t>(free_pages + kHeld));
	PKV_CHECK_MSG(s->pool->status == SSLM_OK, "state: page pool of %lld -> %d", static_cast<long long>(free_pages + kHeld),
	              static_cast<int>(s->pool->status));
	if (s->pool->status != SSLM_OK) return s;
	PKV_CHECK_EQ(fx.R(kHeldBudget), kHeld);
	sslm_seq held = nullptr;
	sslm_status st = sslm_seq_create_budgeted(fx.model, &s->pool->pool, kHeldBudget, &held);
	PKV_CHECK_MSG(st == SSLM_OK, "state: held create -> %d", static_cast<int>(st));
	if (st != SSLM_OK) return s;
	s->h.seqs.push_back(held);
	sslm_seq cycle = nullptr;
	st = sslm_seq_create_budgeted(fx.model, &s->pool->pool, static_cast<int32_t>(fx.B()), &cycle);
	PKV_CHECK_MSG(st == SSLM_OK, "state: cycle create -> %d", static_cast<int>(st));
	if (st != SSLM_OK) return s;
	sslm_seq_release(cycle);
	s->ok = true;
	return s;
}

// ---- 2. the legacy-create admission count -----------------------------------------------------------
//
// §8: size the pool P so the free pages are an exact multiple of ceil(cap/B), expect n at P and
// n - 1 at P - 1. Here the state's free pages are n * 256 at P and n * 256 - 1 at P - 1.

template <int N>
void CountLegacy() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.CapPages(), 256);
	const int64_t at_p = int64_t{N} * fx.CapPages();
	for (int64_t free_pages : {at_p, at_p - 1}) {
		std::unique_ptr<PoolState> s = BuildState(fx, free_pages);
		if (!s->ok) return;
		// The count alone, as §8 states the procedure (its refusal out-parameter is not consulted).
		const int count = CountLegacyCreates(fx.model, s->Pool());
		const int want = free_pages == at_p ? N : N - 1;
		PKV_CHECK_MSG(count == want, "legacy-create count at %s (%lld free pages expected): %d, expected %d",
		              free_pages == at_p ? "P" : "P - 1", static_cast<long long>(free_pages), count, want);
	}
}

PKV_CELL("CM.count:n=1", "CM", CountLegacy<1>);
PKV_CELL("CM.count:n=2", "CM", CountLegacy<2>);
PKV_CELL("CM.count:n=3", "CM", CountLegacy<3>);

// ---- 3. the fill probe, two-sided and one-state ---------------------------------------------------
//
// `actual` pages are free in the state; the probe is asked whether exactly `claim` are. claim ==
// actual is an exact state (must-accept); claim == actual + 1 is a one-page deficit against the
// claim and claim == actual - 1 a one-page excess (must-reject).

template <int Actual, int Claim>
void Probe2() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	ProbeExactlyFree(fx, [&]() -> std::unique_ptr<ProbeState> { return BuildState(fx, Actual); }, Claim);
}

template <int Actual, int Claim>
void Probe1() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::unique_ptr<PoolState> s = BuildState(fx, Actual);
	if (!s->ok) return;
	ProbeExactlyFreeOneState(fx, s->Pool(), Claim);
}

// PKV_CELL names its registrar by __LINE__, so a macro that registers several cells on one line
// uses __COUNTER__ instead.
#define CM_CELL(id, fn) static ::pkv::Registrar PKV_CAT(cm_registrar_, __COUNTER__)(id, "CM", fn)
#define CM_PROBE2(k)                                                        \
	CM_CELL("CM.probe2:exact:k=" #k, (Probe2<k, k>));                \
	CM_CELL("CM.probe2:deficit:k=" #k, (Probe2<k, k + 1>));          \
	CM_CELL("CM.probe2:excess:k=" #k, (Probe2<k, k - 1>))
#define CM_PROBE2_EXACT_DEFICIT(k)                                          \
	CM_CELL("CM.probe2:exact:k=" #k, (Probe2<k, k>));                \
	CM_CELL("CM.probe2:deficit:k=" #k, (Probe2<k, k + 1>))
#define CM_PROBE1(k)                                                        \
	CM_CELL("CM.probe1:exact:k=" #k, (Probe1<k, k>));                \
	CM_CELL("CM.probe1:deficit:k=" #k, (Probe1<k, k + 1>));          \
	CM_CELL("CM.probe1:excess:k=" #k, (Probe1<k, k - 1>))

// k = 2 has no excess form (a claim of 1 cannot be probed: every create is at least 2 pages).
CM_PROBE2_EXACT_DEFICIT(2);
CM_PROBE2(3);
CM_PROBE2(16);
CM_PROBE2(17);
CM_PROBE2(255);
CM_PROBE2(256);
CM_PROBE2(257);
CM_PROBE2(258);
CM_PROBE2(511);
CM_PROBE2(512);
CM_PROBE2(513);
CM_PROBE2(600);

// The one-state form is defined for k >= 3; its excess form claims k - 1 >= 3.
PKV_CELL("CM.probe1:exact:k=3", "CM", (Probe1<3, 3>));
PKV_CELL("CM.probe1:deficit:k=3", "CM", (Probe1<3, 4>));
CM_PROBE1(4);
CM_PROBE1(16);
CM_PROBE1(17);
CM_PROBE1(256);
CM_PROBE1(257);
CM_PROBE1(258);
CM_PROBE1(513);
CM_PROBE1(600);

}  // namespace
