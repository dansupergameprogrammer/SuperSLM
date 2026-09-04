// T-2296 — dimension 6, deterministic integer-only container sizing.
//
// This test deliberately avoids constructing enormous std::unordered_* tables. The old
// pre-remedy crash witness depended on MSVC STL internals, spawned Windows Error Reporting,
// consumed hundreds of MiB, and could outlive the suite timeout. The shipped invariant is
// narrower and stronger: production sizing is integer-only at the historical 2^24+1 boundary,
// and the production containers operate normally with realistic populations while floating-
// point exception masks are cleared.

#include "fixture_common.h"
#include "detail/int_hash.h"

namespace {

constexpr uint64_t kHistoricalBoundary = 16777217ull;  // 2^24 + 1
constexpr uint64_t kMergePopulation = 151936ull;       // real vocabulary-sized map site
constexpr uint64_t kSeenNamesPopulation = 400ull;      // real model-loader set site

int RunChild(const std::string& mode) {
	ClearAllMxcsrExceptionMasks();
	if (mode == "production_map") {
		superslm::detail::FixedIntMap<uint64_t, uint64_t, superslm::detail::MixKey64> map;
		map.Init(kMergePopulation);
		for (uint64_t i = 0; i < kMergePopulation; ++i) map.Insert(i, i ^ 0x5a5a5a5aull);
		const uint64_t* value = map.Find(kMergePopulation - 1);
		return value && *value == ((kMergePopulation - 1) ^ 0x5a5a5a5aull) ? 0 : 3;
	}
	if (mode == "production_set") {
		superslm::detail::FixedIntSet<uint64_t, superslm::detail::MixKey64> set;
		set.Init(kSeenNamesPopulation);
		for (uint64_t i = 0; i < kSeenNamesPopulation; ++i) {
			if (!set.InsertUnique(i)) return 4;
		}
		return set.InsertUnique(kSeenNamesPopulation - 1) ? 5 : 0;
	}
	std::fprintf(stderr, "unknown child mode: %s\n", mode.c_str());
	return 2;
}

}  // namespace

int main(int argc, char** argv) {
	std::string mode;
	if (ParseChildMode(argc, argv, &mode)) return RunChild(mode);
	SetSelfPathFromModule();

	std::printf("=== dim6_determinism_red: bounded production witness ===\n");

	// Pure sizing witness for the formerly problematic value: no allocation and no FP.
	CHECK(superslm::detail::BucketCountFor(0) == 1);
	CHECK(superslm::detail::BucketCountFor(kHistoricalBoundary) == (uint64_t{1} << 26));
	CHECK(superslm::detail::BucketCountFor(kMergePopulation) == (uint64_t{1} << 19));

	ChildResult map = RunChildMode("production_map");
	CHECK_MSG(map == ChildResult::kExitedZero,
	          "FixedIntMap at the real %llu-entry merge population must complete under cleared "
	          "FP masks; got %s",
	          static_cast<unsigned long long>(kMergePopulation), ChildResultName(map));

	ChildResult set = RunChildMode("production_set");
	CHECK_MSG(set == ChildResult::kExitedZero,
	          "FixedIntSet at the real %llu-entry seen-names population must complete under "
	          "cleared FP masks; got %s",
	          static_cast<unsigned long long>(kSeenNamesPopulation), ChildResultName(set));

	std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}
