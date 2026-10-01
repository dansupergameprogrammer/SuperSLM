// Paged-KV plan (rev 16.2) step C6: the lifecycle timings (§8's timing harness; §9's C6 row, "run ...
// the lifecycle timings on both binaries. Record the executed worst-setting figures (§5 a)").
//
// The cost of create, reset, adopt, save, restore and release, timed on the v1.11.0 binary (the plan's
// "v1.9.0", re-anchored) and on the paged binary, on the same machine. The plan states no pass/fail bar
// for a lifecycle verb: §5(a) and (b) state the expected orders (create O(reserve pages), reset
// O(mapped pages), release O(live length), legacy adopt O(prefix length)) and owe "executed figures at
// the worst reachable setting"; §8's claim for the instrument is "the lifecycle verb's cost ... changed
// by at least the reported amount"; §10 R9 states a bar for decode throughput (7.9) only. So this cell
// records figures and asserts no verdict on them: per verb it prints the baseline, the paged figure, the
// change and the resolving power, and a verdict line in the timing cells' shared vocabulary (pkv_common.h):
// NO RESULT when the change is below the resolving power (§8, R9), RESOLVED otherwise (there is no bar,
// so never PASS or FAIL).
// Until the timing harness is commissioned (SUPERSLM_PAGED_KV_TIMING_COMMISSIONED=1) the readings are
// also marked quarantined (§8: recorded, never acted on or headlined).
//
// This file calls legacy verbs only, so it builds against the v1.11.0 library and against the paged
// build (a legacy holder in a page pool), like c6_dim7_throughput.cpp. The box run is two runs of it:
//   1. built against v1.11.0, with SUPERSLM_PAGED_KV_LIFE_OUT=<file>: writes one line per reading,
//      "<target>:<verb> <median ns> <relative spread>";
//   2. built as superslm_pkv_c6, with SUPERSLM_PAGED_KV_LIFE_BASELINE=<that file>: measures the same
//      readings and prints each against its baseline.
// The cloud twin ("lifecycle/C6:fixtures", on pkv_def and pkv_32k) reads the same two variables, so the
// two-binary flow runs on the fixtures too; without them it prints the figures alone.
//
// Readings, per target (one legacy holder at a time in a legacy pool):
//   create                  sslm_seq_create into an empty one-block pool
//   release@0               releasing that fresh holder
//   restore@L, save@L, release@L, reset@L
//                           for live lengths L = 16, 1,000 and the cap (the worst reachable setting of
//                           §5(a): a full-length release is the paged build's one O(cap) lifecycle cost;
//                           at cap 32768 the cap state is the 1,000-token state's SSB5 blob with its
//                           context_length set to the cap -- SSB5 carries the whole block, so the restored
//                           holder has every position live -- because a prefill to 32,768 costs hours at
//                           1.5B; SUPERSLM_PAGED_KV_LIFE_PREFILL_CAP=1 prefills it instead; every state is
//                           checked to restore and save back at its stated length before any timing):
//                           restore of a saved L-token state into a fresh holder, sslm_seq_save of it
//                           into a pre-sized buffer, then its release; and, from a second restore,
//                           sslm_seq_reset of it
//   adopt@P                 copy adopt (legacy holder, legacy prefix) of a frozen prefix of P = 16, 1,000
//                           and 2,048 tokens, alternated with a reset that is not timed
// Each reading is the median of kBatches batch medians of kCalls calls; its relative spread is
// (max - min) / median over the batch medians (7.9's convention, one level up), the batches being
// interleaved across all readings so a drift reaches every reading alike. The resolving power of a
// change is the larger of the two sides' spreads.

#include "pkv_budget_c_helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace pkv;
using namespace pkv::bc;
using clock_type = std::chrono::steady_clock;

constexpr int kBatches = 5;
constexpr int kCalls = 21;

struct Reading {
	bool ok = false;
	double median = 0;  // ns per call
	double spread = 0;  // (max - min) / median over the batch medians
};

double MedianOf(std::vector<double> v) {
	if (v.empty()) return 0;
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

double Ns(clock_type::time_point a, clock_type::time_point b) { return std::chrono::duration<double, std::nano>(b - a).count(); }

// One target's readings, in print order. Each sample appends to the current batch of its reading.
class Lifecycle {
public:
	Lifecycle(const Fixture& fx, std::string target, bool synth_cap) : fx_(fx), target_(std::move(target)), synth_cap_(synth_cap) {}

	// Returns readings keyed "<target>:<verb>", in the order measured.
	std::vector<std::pair<std::string, Reading>> Run() {
		std::vector<std::pair<std::string, Reading>> out;
		if (!fx_.ok || !Prepare()) return out;
		for (int b = 0; b < kBatches; ++b) {
			for (int i = 0; i < kCalls && ok_; ++i) CreateRelease();
			for (size_t l = 0; l < blobs_.size(); ++l)
				for (int i = 0; i < kCalls && ok_; ++i) RestoreSaveRelease(l), RestoreReset(l);
			for (size_t p = 0; p < prefixes_.size(); ++p)
				for (int i = 0; i < kCalls && ok_; ++i) Adopt(p);
			for (auto& kv : samples_) {
				batches_[kv.first].push_back(MedianOf(kv.second));
				kv.second.clear();
			}
		}
		Teardown();
		if (!ok_) return out;
		for (const std::string& verb : order_) {
			std::vector<double> m = batches_[verb];
			Reading r;
			r.median = MedianOf(m);
			std::sort(m.begin(), m.end());
			r.spread = r.median > 0 ? (m.back() - m.front()) / r.median : 0;
			r.ok = r.median > 0;
			out.emplace_back(target_ + ":" + verb, r);
		}
		return out;
	}

private:
	void Sample(const std::string& verb, double ns) {
		if (samples_.find(verb) == samples_.end()) order_.push_back(verb);
		samples_[verb].push_back(ns);
	}
	bool Fail(const char* what, sslm_status st) {
		PKV_CHECK_MSG(false, "lifecycle %s: %s returned %d", target_.c_str(), what, static_cast<int>(st));
		ok_ = false;
		return false;
	}

	// The live lengths and prefix lengths, the saved states and the frozen prefixes (untimed).
	bool Prepare() {
		const int64_t cap = fx_.geo.context_cap;
		lengths_ = {16, 1000, cap};
		const std::vector<int64_t> plen = {16, 1000, 2048};
		const std::vector<int32_t> stream = Stream(97, static_cast<int32_t>(cap), kVocab);
		{
			LegacyPool pool(fx_.model, 1);
			sslm_seq s = nullptr;
			if (pool.status != SSLM_OK) return Fail("sslm_kv_pool_create", pool.status);
			sslm_status st = sslm_seq_create(fx_.model, &pool.pool, &s);
			if (st != SSLM_OK) return Fail("sslm_seq_create", st);
			int64_t at = 0;
			for (int64_t L : lengths_) {
				if (L == cap && synth_cap_) {
					// The cap state from the 1,000-token state: SSB5 carries the whole block, so setting
					// its context_length (offset 60) to the cap restores a holder with every position live.
					std::vector<uint8_t> b = blobs_.back();
					PutLe64(b, 60, static_cast<uint64_t>(cap));
					blobs_.push_back(std::move(b));
					continue;
				}
				const std::vector<int32_t> part(stream.begin() + at, stream.begin() + L);
				st = PrefillAll(fx_.model, s, part, kChunk);
				if (st != SSLM_OK) {
					sslm_seq_release(s);
					return Fail("prefill", st);
				}
				at = L;
				blobs_.push_back(Save(s));
				if (blobs_.back().empty()) {
					sslm_seq_release(s);
					ok_ = false;
					return false;
				}
			}
			sslm_seq_release(s);
		}
		// Every state restores to its stated length and saves back at the same size: the length each
		// reading is labelled with is the length the holder had.
		for (size_t l = 0; l < blobs_.size(); ++l) {
			LegacyPool pool(fx_.model, 1);
			sslm_seq s = nullptr;
			const sslm_status rs = sslm_seq_restore(fx_.model, &pool.pool, blobs_[l].data(), blobs_[l].size(), &s);
			if (rs != SSLM_OK) return Fail("sslm_seq_restore (check)", rs);
			const std::vector<uint8_t> back = Save(s);
			sslm_seq_release(s);
			const bool same = IsMagic(back, "SSB5") && back.size() == blobs_[l].size() && Le64(back, 60) == static_cast<uint64_t>(lengths_[l]);
			PKV_CHECK_MSG(same, "lifecycle %s: the state of length %lld does not round-trip (%s, %zu bytes, length %llu)", target_.c_str(),
			              static_cast<long long>(lengths_[l]), back.size() >= 4 ? std::string(back.begin(), back.begin() + 4).c_str() : "?",
			              back.size(), back.size() >= 68 ? static_cast<unsigned long long>(Le64(back, 60)) : 0ull);
			if (!same) {
				ok_ = false;
				return false;
			}
		}
		// adopt: one pool holding the adopting holder and every prefix (one block each).
		adopt_pool_ = std::make_unique<LegacyPool>(fx_.model, static_cast<uint32_t>(plen.size() + 1));
		if (adopt_pool_->status != SSLM_OK) return Fail("sslm_kv_pool_create", adopt_pool_->status);
		for (int64_t P : plen) {
			sslm_prefix px = LegacyPrefix(fx_, &adopt_pool_->pool, PrefixTokens(static_cast<int32_t>(P)));
			if (!px) {
				ok_ = false;
				return false;
			}
			prefixes_.push_back(px);
			prefix_lengths_.push_back(P);
		}
		const sslm_status st = sslm_seq_create(fx_.model, &adopt_pool_->pool, &adopter_);
		if (st != SSLM_OK) return Fail("sslm_seq_create (adopter)", st);
		save_buf_.assign(std::max_element(blobs_.begin(), blobs_.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); })->size(), 0);
		return true;
	}

	void Teardown() {
		if (adopter_) sslm_seq_release(adopter_);
		adopter_ = nullptr;
		for (sslm_prefix p : prefixes_) sslm_prefix_release(p);
		prefixes_.clear();
		adopt_pool_.reset();
	}

	void CreateRelease() {
		LegacyPool pool(fx_.model, 1);
		if (pool.status != SSLM_OK) return void(Fail("sslm_kv_pool_create", pool.status));
		sslm_seq s = nullptr;
		const auto t0 = clock_type::now();
		const sslm_status c = sslm_seq_create(fx_.model, &pool.pool, &s);
		const auto t1 = clock_type::now();
		if (c != SSLM_OK) return void(Fail("sslm_seq_create", c));
		const sslm_status r = sslm_seq_release(s);
		const auto t2 = clock_type::now();
		if (r != SSLM_OK) return void(Fail("sslm_seq_release", r));
		Sample("create", Ns(t0, t1));
		Sample("release@0", Ns(t1, t2));
	}

	void RestoreSaveRelease(size_t l) {
		LegacyPool pool(fx_.model, 1);
		if (pool.status != SSLM_OK) return void(Fail("sslm_kv_pool_create", pool.status));
		const std::vector<uint8_t>& blob = blobs_[l];
		const std::string at = "@" + std::to_string(lengths_[l]);
		sslm_seq s = nullptr;
		const auto t0 = clock_type::now();
		const sslm_status rs = sslm_seq_restore(fx_.model, &pool.pool, blob.data(), blob.size(), &s);
		const auto t1 = clock_type::now();
		if (rs != SSLM_OK) return void(Fail("sslm_seq_restore", rs));
		size_t wrote = save_buf_.size();
		const auto t2 = clock_type::now();
		const sslm_status sv = sslm_seq_save(s, save_buf_.data(), &wrote);
		const auto t3 = clock_type::now();
		if (sv != SSLM_OK || wrote != blob.size()) {
			sslm_seq_release(s);
			return void(Fail("sslm_seq_save", sv));
		}
		const auto t4 = clock_type::now();
		const sslm_status rl = sslm_seq_release(s);
		const auto t5 = clock_type::now();
		if (rl != SSLM_OK) return void(Fail("sslm_seq_release", rl));
		Sample("restore" + at, Ns(t0, t1));
		Sample("save" + at, Ns(t2, t3));
		Sample("release" + at, Ns(t4, t5));
	}

	void RestoreReset(size_t l) {
		LegacyPool pool(fx_.model, 1);
		if (pool.status != SSLM_OK) return void(Fail("sslm_kv_pool_create", pool.status));
		const std::vector<uint8_t>& blob = blobs_[l];
		sslm_seq s = nullptr;
		const sslm_status rs = sslm_seq_restore(fx_.model, &pool.pool, blob.data(), blob.size(), &s);
		if (rs != SSLM_OK) return void(Fail("sslm_seq_restore", rs));
		const auto t0 = clock_type::now();
		const sslm_status r = sslm_seq_reset(s);
		const auto t1 = clock_type::now();
		sslm_seq_release(s);
		if (r != SSLM_OK) return void(Fail("sslm_seq_reset", r));
		Sample("reset@" + std::to_string(lengths_[l]), Ns(t0, t1));
	}

	void Adopt(size_t p) {
		const auto t0 = clock_type::now();
		const sslm_status a = sslm_seq_adopt_prefix(adopter_, prefixes_[p]);
		const auto t1 = clock_type::now();
		if (a != SSLM_OK) return void(Fail("sslm_seq_adopt_prefix", a));
		const sslm_status r = sslm_seq_reset(adopter_);
		if (r != SSLM_OK) return void(Fail("sslm_seq_reset (after adopt)", r));
		Sample("adopt@" + std::to_string(prefix_lengths_[p]), Ns(t0, t1));
	}

	const Fixture& fx_;
	std::string target_;
	bool synth_cap_;
	bool ok_ = true;
	std::vector<int64_t> lengths_, prefix_lengths_;
	std::vector<std::vector<uint8_t>> blobs_;
	std::vector<uint8_t> save_buf_;
	std::unique_ptr<LegacyPool> adopt_pool_;
	std::vector<sslm_prefix> prefixes_;
	sslm_seq adopter_ = nullptr;
	std::vector<std::string> order_;
	std::map<std::string, std::vector<double>> samples_, batches_;
};

std::string Us(double ns) {
	char b[32];
	std::snprintf(b, sizeof b, "%.2f us", ns / 1000.0);
	return b;
}

// Prints one reading against its baseline, then its verdict in the timing cells' shared vocabulary
// (pkv_common.h): NO RESULT when |change| is below the resolving power, RESOLVED otherwise. No bar exists
// for a lifecycle verb, so there is no PASS or FAIL and nothing is asserted.
void Report(const std::string& key, const Reading& paged, const Reading* base) {
	if (!base) {
		std::printf("lifecycle %s: %s, spread %.2f%% (no baseline given)%s\n", key.c_str(), Us(paged.median).c_str(), paged.spread * 100,
		            TimingCommissioned() ? "" : " [quarantined: timing harness not commissioned]");
		return;
	}
	const double change = paged.median / base->median - 1.0;  // positive: the paged build is slower
	const double resolving = std::max(paged.spread, base->spread);
	const bool resolved = std::fabs(change) >= resolving;
	std::printf("lifecycle %s: baseline %s, paged %s, change %+.2f%%, resolving power %.2f%%\n", key.c_str(), Us(base->median).c_str(),
	            Us(paged.median).c_str(), change * 100, resolving * 100);
	PrintTimingVerdict("lifecycle", key, resolved ? TimingVerdict::kResolved : TimingVerdict::kNoResult,
	                   resolved ? Fmt("change %+.2f%% at resolving power %.2f%%; no bar for a lifecycle verb", change * 100, resolving * 100)
	                            : Fmt("|change| %.2f%% is below the resolving power %.2f%%", std::fabs(change) * 100, resolving * 100),
	                   TimingCommissioned());
}

struct Target {
	const Fixture* fx;
	std::string name;
	bool synth_cap;  // the cap state from the 1,000-token state (above) instead of a prefill to the cap
};

void RunTargets(const std::vector<Target>& targets, bool box) {
	std::map<std::string, Reading> baseline;
	const char* in_path = std::getenv("SUPERSLM_PAGED_KV_LIFE_BASELINE");
	if (in_path) {
		std::ifstream in(in_path);
		PKV_CHECK_MSG(static_cast<bool>(in), "lifecycle: baseline file %s unreadable", in_path);
		std::string line;
		while (GetTextLine(in, line)) {  // LF or CRLF (a baseline written on the box, read anywhere)
			std::istringstream ls(line);
			std::string key;
			Reading r;
			if (ls >> key >> r.median >> r.spread) {
				r.ok = true;
				baseline[key] = r;
			}
		}
		PKV_CHECK_MSG(!baseline.empty(), "lifecycle: baseline file %s holds no readings", in_path);
	}
	const char* out_path = std::getenv("SUPERSLM_PAGED_KV_LIFE_OUT");
	std::ofstream out;
	if (out_path) out.open(out_path);
	for (const auto& t : targets) {
		for (const auto& kr : Lifecycle(*t.fx, t.name, t.synth_cap).Run()) {
			if (out) out << kr.first << " " << kr.second.median << " " << kr.second.spread << "\n";
			auto it = baseline.find(kr.first);
			if (in_path) PKV_CHECK_MSG(it != baseline.end(), "lifecycle %s: no baseline reading (was the v1.11.0 run complete?)", kr.first.c_str());
			Report(kr.first, kr.second, it != baseline.end() ? &it->second : nullptr);
		}
	}
	if (out_path) PKV_CHECK_MSG(static_cast<bool>(out), "lifecycle: could not write %s", out_path);
	std::printf("lifecycle: no pass/fail bar -- the plan states none for a lifecycle verb (§5(a)-(b) give orders and owe the "
	            "executed figures; §8 claims only \"changed by at least the reported amount\"; §10 R9's 5%% bar is 7.9's)\n");
	// The box's paged run needs the v1.11.0 run's figures; the box's v1.11.0 run writes them.
	if (box)
		PKV_CHECK_MSG(out_path || in_path,
		              "lifecycle: neither SUPERSLM_PAGED_KV_LIFE_OUT (the v1.11.0 run) nor SUPERSLM_PAGED_KV_LIFE_BASELINE (the paged run) is set");
}

// SUPERSLM_PAGED_KV_LIFE_PREFILL_CAP=1 prefills the cap-32768 states to the cap instead (hours at 1.5B).
bool PrefillCap() {
	const char* c = std::getenv("SUPERSLM_PAGED_KV_LIFE_PREFILL_CAP");
	return c && std::string(c) == "1";
}

void CellLifecycleBox() {
	RunTargets({{&GetArtifact("qwen2.5-0.5b-instruct-cap4096-aex.sslm", Geometry{24, 2, 64, 4096}), "0.5B_cap4096", false},
	            {&GetArtifact("qwen2.5-1.5b-instruct.sslm", Geometry{28, 2, 128, 32768}), "1.5B_cap32768", !PrefillCap()}},
	           true);
}

// pkv_def runs the prefilled cap state and pkv_32k the derived one, so the cloud covers both paths.
void CellLifecycleFixtures() {
	RunTargets({{&GetFixture("pkv_def"), "pkv_def", false}, {&GetFixture("pkv_32k"), "pkv_32k", !PrefillCap()}}, false);
}

PKV_CELL("lifecycle/C6", "C6", CellLifecycleBox);
PKV_CELL("lifecycle/C6:fixtures", "C6", CellLifecycleFixtures);

}  // namespace
