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
// reset is O(mapped) once the memset is gone, §3.3).
//
// Each reading is the median of kBatches batch medians of kCalls timed calls (205 calls), the batches of
// the cap-4096 and cap-32768 holders interleaved so a drift reaches both alike. Its relative spread is
// (max - min) / median over the batch medians, and the resolving power of a ratio is the larger of the
// two sides' spreads (7.9's and c6_lifecycle_timing.cpp's convention). Each ratio is printed with it and
// graded three ways (pkv_common.h's timing verdicts) against the bound, by its relative distance
// d = ratio / bound - 1: NO RESULT when |d| is within the resolving power, PASS when d <= -resolving,
// FAIL when d > resolving. A NO RESULT asserts nothing.
//
// Registered twice: "7.4/C6" on the box's real artifacts (0.5B at cap 4096 and 1.5B at cap 32768 from
// SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR; per-token bytes differ, so the printed figures are also given
// per byte of the live rows), and "7.4/C6:fixtures" on pkv_def and pkv_32k, which differ only in cap.

#include "pkv_budget_c_helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace pkv;
using namespace pkv::bc;

constexpr int kBatches = 5;
constexpr int kCalls = 41;
constexpr int32_t kLive = 1000;

double Median(std::vector<double> v) {
	if (v.empty()) return 0;
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

struct Reading {
	double median = 0;  // ns per call: the median of the batch medians
	double spread = 0;  // (max - min) / median over the batch medians
};

Reading Of(std::vector<double> batch_medians) {
	Reading r;
	r.median = Median(batch_medians);
	std::sort(batch_medians.begin(), batch_medians.end());
	r.spread = r.median > 0 && !batch_medians.empty() ? (batch_medians.back() - batch_medians.front()) / r.median : 0;
	return r;
}

// One side: a rig with the prefix and the alternating holder, timed one batch at a time.
class Side {
public:
	Side(const Fixture& fx, bool budget_holder) : fx_(fx) {
		if (!fx.ok) return;
		const int64_t own = budget_holder ? fx.R(512) : fx.CapPages();
		rig_ = std::make_unique<Rig>(fx, static_cast<uint32_t>(fx.R(kLive) + own + 4));
		px_ = rig_->Prefix(BudgetPrefix(fx, rig_->Pool(), PrefixTokens(kLive)));
		sslm_seq s = nullptr;
		const sslm_status st =
		    budget_holder ? sslm_seq_create_budgeted(fx.model, rig_->Pool(), 512, &s) : sslm_seq_create(fx.model, rig_->Pool(), &s);
		PKV_CHECK_EQ(st, SSLM_OK);
		s_ = rig_->Seq(s);
		ok_ = px_ && s_;
	}
	bool ok() const { return ok_; }
	void Batch() {
		if (!ok_) return;
		std::vector<double> reset, adopt;
		using clock = std::chrono::steady_clock;
		for (int i = 0; i < kCalls; ++i) {
			const auto t0 = clock::now();
			const sslm_status a = sslm_seq_adopt_prefix(s_, px_);
			const auto t1 = clock::now();
			const sslm_status r = sslm_seq_reset(s_);
			const auto t2 = clock::now();
			PKV_CHECK_MSG(a == SSLM_OK && r == SSLM_OK, "7.4 %s: adopt %d, reset %d", fx_.stem.c_str(), static_cast<int>(a), static_cast<int>(r));
			if (a != SSLM_OK || r != SSLM_OK) {
				ok_ = false;
				return;
			}
			adopt.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
			reset.push_back(std::chrono::duration<double, std::nano>(t2 - t1).count());
		}
		adopt_.push_back(Median(adopt));
		reset_.push_back(Median(reset));
	}
	Reading Reset() const { return Of(reset_); }
	Reading Adopt() const { return Of(adopt_); }

private:
	const Fixture& fx_;
	std::unique_ptr<Rig> rig_;
	sslm_prefix px_ = nullptr;
	sslm_seq s_ = nullptr;
	bool ok_ = false;
	std::vector<double> reset_, adopt_;  // batch medians
};

// Prints one verb's ratio with its resolving power and, given a bound, its verdict; asserts a PASS or
// FAIL only (a bound is given only once the harness is commissioned, above).
void GradeRatio(const char* mode, const char* verb, const Reading& a, const Reading& b, const char* bound) {
	const double ratio = b.median / a.median;
	const double resolving = std::max(a.spread, b.spread);
	std::printf("7.4 %s %s: ratio %.3f (cap 32768 / cap 4096), resolving power %.2f%%\n", mode, verb, ratio, resolving * 100);
	if (!bound) return;
	const double max_ratio = std::atof(bound);
	// Relative units: the ratio over the bound, against 1, so the (relative) resolving power applies as is.
	// No "effect below the resolving power" rule here: 7.4's claim is the ratio's place against the bound,
	// and a ratio near 1 is that claim's pass, not an absent effect.
	const double d = ratio / max_ratio - 1.0;
	const TimingVerdict v = GradeAgainstBound(ratio / max_ratio, 1.0, resolving);
	const std::string why =
	    Fmt("ratio %.3f is %.2f%% %s the bound %.3f, %s the resolving power %.2f%%", ratio, std::fabs(d) * 100, d > 0 ? "above" : "below",
	        max_ratio, v == TimingVerdict::kNoResult ? "within" : "beyond", resolving * 100);
	PrintTimingVerdict("7.4", std::string(mode) + " " + verb, v, why, true);
	if (v == TimingVerdict::kNoResult) return;
	// kills: a reset or adopt whose cost follows the cap (the memset, a cap-sized table fill, a whole-block
	// copy): about 8x from 4096 to 32768
	PKV_CHECK_MSG(v == TimingVerdict::kPass, "7.4 %s %s: ratio %.3f above the commissioned %.3f by more than the resolving power %.2f%%",
	              mode, verb, ratio, max_ratio, resolving * 100);
}

void Run74(const Fixture& small, const Fixture& large) {
	if (!small.ok || !large.ok) return;
	PKV_CHECK_EQ(small.geo.context_cap, 4096);
	PKV_CHECK_EQ(large.geo.context_cap, 32768);
	const char* bound = std::getenv("SUPERSLM_PAGED_KV_74_MAX_RATIO");
	for (int budget_holder = 1; budget_holder >= 0; --budget_holder) {
		Side sa(small, budget_holder), sb(large, budget_holder);
		if (!sa.ok() || !sb.ok()) continue;
		for (int batch = 0; batch < kBatches; ++batch) {
			sa.Batch();
			sb.Batch();
		}
		if (!sa.ok() || !sb.ok()) continue;
		const Reading ar = sa.Reset(), br = sb.Reset(), aa = sa.Adopt(), ba = sb.Adopt();
		const double live_a = static_cast<double>(kLive) * small.BytesPerToken(), live_b = static_cast<double>(kLive) * large.BytesPerToken();
		const char* mode = budget_holder ? "budget" : "whole_reserve";
		std::printf("7.4 %s holder, live length %d: reset %.0f ns (cap 4096, %s, spread %.2f%%) / %.0f ns (cap 32768, %s, spread %.2f%%); "
		            "adopt %.0f (spread %.2f%%) / %.0f ns (spread %.2f%%); per live byte: reset %.4f / %.4f, adopt %.4f / %.4f ns\n",
		            mode, kLive, ar.median, small.stem.c_str(), ar.spread * 100, br.median, large.stem.c_str(), br.spread * 100, aa.median,
		            aa.spread * 100, ba.median, ba.spread * 100, ar.median / live_a, br.median / live_b, aa.median / live_a,
		            ba.median / live_b);
		// The FAIL message's "<mode> reset: ratio" is what run_commissioning.py's timing74 reject reads.
		GradeRatio(mode, "reset", ar, br, bound);
		GradeRatio(mode, "adopt", aa, ba, bound);
		if (!bound)
			std::printf("7.4: quarantined reading (SUPERSLM_PAGED_KV_74_MAX_RATIO unset: the timing harness is not commissioned)\n");
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
