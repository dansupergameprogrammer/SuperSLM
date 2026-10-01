// Paged-KV plan (rev 16.1) step C1: cell 7.4, owned by C6 (§7, §8's timing harness).
//
// 7.4 "Reset and adopt do not scale with the cap": the same live length timed at caps 4096 and 32768.
// Dimension 7 checks that the measurement exists; the value is calibration's (§8): the timing harness is
// commissioned at C6 on the box, and until then its readings are quarantined (recorded, never acted
// on). So the cell measures and prints, and asserts the ratio only when the box supplies the commissioned
// bound in SUPERSLM_PAGED_KV_74_MAX_RATIO (the largest cap-32768 / cap-4096 median ratio calibration
// accepts as "does not scale"; a cap-proportional verb shows about 8).
//
// The construction: a 1,000-token prefix (mid-page, so share adopt copies a tail page), and a holder
// that alternates adopt and reset, so each reset unmaps a live length of 1,000 and each adopt maps one.
// Both holder modes: budget (share adopt, §3.5) and whole_reserve (copy adopt of ceil(1000/B) pages; its
// reset is O(mapped) once the memset is gone, §3.3). Medians of 201 timed calls each.
//
// Registered twice: "7.4/C6" on the box's real artifacts (0.5B at cap 4096 and 1.5B at cap 32768 from
// SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR; per-token bytes differ, so the printed figures are also given
// per byte of the live rows), and "7.4/C6:fixtures" on pkv_def and pkv_32k, which differ only in cap.

#include "pkv_budget_c_helpers.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace pkv;
using namespace pkv::bc;

constexpr int kReps = 201;
constexpr int32_t kLive = 1000;

struct Medians {
	double reset_ns = 0, adopt_ns = 0;
	bool ok = false;
};

double Median(std::vector<double> v) {
	if (v.empty()) return 0;
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

Medians Measure(const Fixture& fx, bool budget_holder) {
	Medians m;
	if (!fx.ok) return m;
	const int64_t own = budget_holder ? fx.R(512) : fx.CapPages();
	Rig rig(fx, static_cast<uint32_t>(fx.R(kLive) + own + 4));
	sslm_prefix px = rig.Prefix(BudgetPrefix(fx, rig.Pool(), PrefixTokens(kLive)));
	sslm_seq s = nullptr;
	const sslm_status st = budget_holder ? sslm_seq_create_budgeted(fx.model, rig.Pool(), 512, &s) : sslm_seq_create(fx.model, rig.Pool(), &s);
	PKV_CHECK_EQ(st, SSLM_OK);
	if (!px || !rig.Seq(s)) return m;
	std::vector<double> reset, adopt;
	using clock = std::chrono::steady_clock;
	for (int i = 0; i < kReps; ++i) {
		const auto t0 = clock::now();
		const sslm_status a = sslm_seq_adopt_prefix(s, px);
		const auto t1 = clock::now();
		const sslm_status r = sslm_seq_reset(s);
		const auto t2 = clock::now();
		PKV_CHECK_MSG(a == SSLM_OK && r == SSLM_OK, "7.4 %s: adopt %d, reset %d", fx.stem.c_str(), static_cast<int>(a), static_cast<int>(r));
		adopt.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
		reset.push_back(std::chrono::duration<double, std::nano>(t2 - t1).count());
	}
	m.adopt_ns = Median(adopt);
	m.reset_ns = Median(reset);
	m.ok = true;
	return m;
}

void Run74(const Fixture& small, const Fixture& large) {
	if (!small.ok || !large.ok) return;
	PKV_CHECK_EQ(small.geo.context_cap, 4096);
	PKV_CHECK_EQ(large.geo.context_cap, 32768);
	const char* bound = std::getenv("SUPERSLM_PAGED_KV_74_MAX_RATIO");
	for (int budget_holder = 1; budget_holder >= 0; --budget_holder) {
		const Medians a = Measure(small, budget_holder), b = Measure(large, budget_holder);
		if (!a.ok || !b.ok) continue;
		const double live_a = static_cast<double>(kLive) * small.BytesPerToken(), live_b = static_cast<double>(kLive) * large.BytesPerToken();
		const char* mode = budget_holder ? "budget" : "whole_reserve";
		std::printf("7.4 %s holder, live length %d: reset %.0f ns (cap 4096, %s) / %.0f ns (cap 32768, %s), ratio %.3f; "
		            "adopt %.0f / %.0f ns, ratio %.3f; per live byte: reset %.4f / %.4f, adopt %.4f / %.4f ns\n",
		            mode, kLive, a.reset_ns, small.stem.c_str(), b.reset_ns, large.stem.c_str(), b.reset_ns / a.reset_ns, a.adopt_ns,
		            b.adopt_ns, b.adopt_ns / a.adopt_ns, a.reset_ns / live_a, b.reset_ns / live_b, a.adopt_ns / live_a, b.adopt_ns / live_b);
		if (bound) {
			const double max_ratio = std::atof(bound);
			// kills: a reset or adopt whose cost follows the cap (the memset, a cap-sized table fill,
			// a whole-block copy): about 8x from 4096 to 32768
			PKV_CHECK_MSG(b.reset_ns / a.reset_ns <= max_ratio, "7.4 %s reset: ratio %.3f above the commissioned %.3f", mode,
			              b.reset_ns / a.reset_ns, max_ratio);
			PKV_CHECK_MSG(b.adopt_ns / a.adopt_ns <= max_ratio, "7.4 %s adopt: ratio %.3f above the commissioned %.3f", mode,
			              b.adopt_ns / a.adopt_ns, max_ratio);
		} else {
			std::printf("7.4: quarantined reading (SUPERSLM_PAGED_KV_74_MAX_RATIO unset: the timing harness is not commissioned)\n");
		}
	}
}

void Cell74Box() {
	Run74(GetArtifact("qwen2.5-0.5b-instruct-cap4096-aex.sslm", Geometry{24, 2, 64, 4096}),
	      GetArtifact("qwen2.5-1.5b-instruct.sslm", Geometry{28, 2, 128, 32768}));
}
void Cell74Fixtures() { Run74(GetFixture("pkv_def"), GetFixture("pkv_32k")); }

PKV_CELL("7.4/C6", "C6", Cell74Box);
PKV_CELL("7.4/C6:fixtures", "C6", Cell74Fixtures);

}  // namespace
