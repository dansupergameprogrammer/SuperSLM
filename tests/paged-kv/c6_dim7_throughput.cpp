// Paged-KV plan (rev 16.1) step C1: cell 7.9, owned by C6 (§7, §8's timing harness).
//
// 7.9 CPU decode throughput: tokens per second, paged at B = 16 against the one-page view, on the 0.5B
// cohort at width 1,712 and on the 1.5B artifact at width 16,384. Bar (planner default): no more than 5%
// slower, stated with resolving power (§10 R9); an effect below the resolving power is no result.
//
// This file calls legacy verbs only, so the same harness builds against the v1.11.0 library (whose flat
// workspace is the one-page view) and against the paged build (a legacy holder in a page pool, B = 16):
// §8's "one harness, both binaries, same machine". The box run is two runs of this file:
//   1. built against v1.11.0, with SUPERSLM_PAGED_KV_79_OUT=<file>: writes one line per target,
//      "<target> <median tok/s> <relative spread>";
//   2. built as superslm_pkv_c6, with SUPERSLM_PAGED_KV_79_BASELINE=<that file>: measures the same
//      targets and grades slowdown = 1 - paged / one-page against the bar.
// The verdict is asserted only with SUPERSLM_PAGED_KV_TIMING_COMMISSIONED=1 (the harness is commissioned
// at C6, §8; before that its readings are quarantined and printed only).
//
// In the cloud ("7.9/C6:fixtures") the one-page view is in the same binary: pkv_odd is pkv_def's weights
// at cap 4100, which 16 does not divide, so B = cap and the holder is one page (§3.1). pkv_def at the
// cohort's width runs against it; pkv_32k at width 16,384 has no one-page twin and is measured only.
// The resolving power is the larger relative spread ((max - min) / median) of the two sides' runs.

#include "pkv_budget_c_helpers.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace pkv;
using namespace pkv::bc;

constexpr int kRuns = 5;
constexpr double kBar = 0.05;

struct Rate {
	bool ok = false;
	double median = 0;  // tokens per second
	double spread = 0;  // (max - min) / median over the runs
};

// Tokens per second decoding from width `width - n` to `width`, by a legacy holder: the prompt is
// prefilled once and saved; each run restores it into a fresh pool, emits the ready token (no row), and
// times n greedy tokens, one whole token per call so per-call overhead matches across binaries.
Rate Measure(const Fixture& fx, int64_t width, int64_t n) {
	Rate r;
	if (!fx.ok) return r;
	std::vector<uint8_t> blob;
	{
		LegacyPool pool(fx.model, 1);
		sslm_seq s = nullptr;
		if (pool.status != SSLM_OK || sslm_seq_create(fx.model, &pool.pool, &s) != SSLM_OK) {
			PKV_CHECK_MSG(false, "7.9 %s: pool or create failed", fx.stem.c_str());
			return r;
		}
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(79, static_cast<int32_t>(width - n), kVocab), kChunk), SSLM_OK);
		blob = Save(s);
		sslm_seq_release(s);
	}
	std::vector<double> rates;
	for (int run = 0; run < kRuns; ++run) {
		LegacyPool pool(fx.model, 1);
		sslm_seq s = nullptr;
		if (pool.status != SSLM_OK || sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &s) != SSLM_OK) {
			PKV_CHECK_MSG(false, "7.9 %s: pool or restore failed", fx.stem.c_str());
			return r;
		}
		Token(fx.model, s);  // the ready token writes no row
		sslm_decode_params p{};
		p.layer_budget = static_cast<int32_t>(fx.geo.layers);
		sslm_seq b[1] = {s};
		const auto t0 = std::chrono::steady_clock::now();
		int64_t made = 0;
		for (int64_t i = 0; i < n; ++i) {
			int32_t tok = -1;
			if (sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok) != SSLM_OK || tok < 0) break;
			++made;
		}
		const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		PKV_CHECK_MSG(made == n, "7.9 %s: %lld of %lld tokens decoded", fx.stem.c_str(), static_cast<long long>(made), static_cast<long long>(n));
		sslm_seq_release(s);
		rates.push_back(static_cast<double>(made) / sec);
	}
	std::sort(rates.begin(), rates.end());
	r.median = rates[rates.size() / 2];
	r.spread = (rates.back() - rates.front()) / r.median;
	r.ok = true;
	return r;
}

bool Commissioned() {
	const char* c = std::getenv("SUPERSLM_PAGED_KV_TIMING_COMMISSIONED");
	return c && std::string(c) == "1";
}

// Grades paged against one-page: slowdown <= 5%, with a resolving power fine enough to see 5%.
void Grade(const char* target, const Rate& paged, const Rate& one_page) {
	if (!paged.ok || !one_page.ok) return;
	const double slowdown = 1.0 - paged.median / one_page.median;
	const double resolving = std::max(paged.spread, one_page.spread);
	std::printf("7.9 %s: paged %.1f tok/s, one-page %.1f tok/s, slowdown %.2f%%, resolving power %.2f%%%s\n", target, paged.median,
	            one_page.median, slowdown * 100, resolving * 100, Commissioned() ? "" : " (quarantined: harness not commissioned)");
	if (!Commissioned()) return;
	// kills: no result reported as a pass (a spread too wide to resolve the bar)
	PKV_CHECK_MSG(resolving < kBar, "7.9 %s: resolving power %.2f%% cannot resolve the 5%% bar", target, resolving * 100);
	// kills: a per-page attention loop (or address path) that costs more than the bar
	PKV_CHECK_MSG(slowdown <= kBar, "7.9 %s: paged decode is %.2f%% slower than the one-page view", target, slowdown * 100);
}

void Cell79Fixtures() {
	const Fixture& paged = GetFixture("pkv_def");
	const Fixture& one = GetFixture("pkv_odd");
	if (paged.ok && one.ok) {
		PKV_CHECK_EQ(paged.B(), 16);
		PKV_CHECK_EQ(one.B(), one.geo.context_cap);  // the one-page view (§3.1's fallback)
		Grade("cohort width 1,712 (pkv_def vs pkv_odd)", Measure(paged, 1712, 512), Measure(one, 1712, 512));
	}
	const Fixture& big = GetFixture("pkv_32k");
	if (big.ok) {
		const Rate r = Measure(big, 16384, 64);
		std::printf("7.9 width 16,384 (pkv_32k, paged only): %.1f tok/s, spread %.2f%%\n", r.median, r.spread * 100);
	}
}

void Cell79Box() {
	struct Target {
		const char* name;
		const char* file;
		Geometry geo;
		int64_t width, n;
	};
	const Target targets[] = {{"0.5B_cohort_1712", "qwen2.5-0.5b-instruct-cap4096-aex.sslm", Geometry{24, 2, 64, 4096}, 1712, 512},
	                          {"1.5B_16384", "qwen2.5-1.5b-instruct.sslm", Geometry{28, 2, 128, 32768}, 16384, 64}};
	std::map<std::string, Rate> baseline;
	if (const char* path = std::getenv("SUPERSLM_PAGED_KV_79_BASELINE")) {
		std::ifstream in(path);
		PKV_CHECK_MSG(static_cast<bool>(in), "7.9: baseline file %s unreadable", path);
		std::string line;
		while (std::getline(in, line)) {
			std::istringstream ls(line);
			std::string name;
			Rate r;
			if (ls >> name >> r.median >> r.spread) {
				r.ok = true;
				baseline[name] = r;
			}
		}
	}
	std::ofstream out;
	if (const char* path = std::getenv("SUPERSLM_PAGED_KV_79_OUT")) out.open(path);
	for (const Target& t : targets) {
		const Rate r = Measure(GetArtifact(t.file, t.geo), t.width, t.n);
		if (!r.ok) continue;
		if (out) out << t.name << " " << r.median << " " << r.spread << "\n";
		auto it = baseline.find(t.name);
		if (it != baseline.end()) Grade(t.name, r, it->second);
		else std::printf("7.9 %s: %.1f tok/s, spread %.2f%% (no baseline given)\n", t.name, r.median, r.spread * 100);
	}
	// The paged build's verdict needs the v1.11.0 run's figures.
	PKV_CHECK_MSG(std::getenv("SUPERSLM_PAGED_KV_79_OUT") || !baseline.empty(),
	              "7.9: neither SUPERSLM_PAGED_KV_79_OUT (the v1.11.0 run) nor SUPERSLM_PAGED_KV_79_BASELINE (the paged run) is set");
}

PKV_CELL("7.9/C6", "C6", Cell79Box);
PKV_CELL("7.9/C6:fixtures", "C6", Cell79Fixtures);

}  // namespace
