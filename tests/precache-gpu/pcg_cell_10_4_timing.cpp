// Precached-prefixes plan §7 10.4 [F3]: the GPU speed bar's instrument. Restore-from-memory, digest included, must be at
// least 100x faster than building the same prefix by batched GPU prefill, at 1.5B on both GPUs and at 0.5B on the RTX
// 2080 SUPER, at 128 and 1,000 tokens (§7 10.4's derivation: the binding case is 1.5B on the 7900 XTX at 128 tokens,
// about 846x on assumed rates).
//
// The harness is 10.3's (§8 "GPU speed bar: the same harness and the same commissioning as 10.3"): per length, one
// untimed warm-up pair, then 8 pairs, each a prefill-built prefix (begin, one batched prompt prefill, freeze; the
// prefill's post-chunk readback makes it complete when the call returns) and a restore of its blob into the same pool
// (validation and digest, the draw, the host-mirror scatter and the synchronous upload). The ratio is reported as a 95%
// interval of the geometric mean over the pairs (Student t, 7 degrees of freedom), and the bar is met when the interval's
// lower end is at least 100. A pair whose prefill or restore falls under 100 timer ticks is refused as unresolved.
//
// COMMISSIONING (§8): the bar is a decision-bearing number, so its readings count only once a seat independent of the
// builder (the debunker) has commissioned this instrument on each GPU: a must-accept (this build: the bar must read met)
// and a must-reject (the same cell built against the engine with tests/precache-gpu/apply_gpu_mutant.py's
// restore-prefills mutant, a restore that runs a prefill of the blob's length internally: the bar must read missed).
// This program never decides: every line it prints is a reading, labelled with the --role it was run as (accept,
// reject or grade), the program's own SHA-256, the adapter and the model. tools/precache_gpu/grade_f3.py turns the
// accept and reject readings into the commissioning summary the debunker signs as a PC-GPU-TIMING-1 receipt, and only
// then grades the --role=grade readings against the bar.
//
// Run:  pcg_cell_10_4_timing.exe --model=FILE --tag=0p5|1p5 --ids=FILE --role=accept|reject|grade [--lengths=128,1000]
//       (--ids: world.ids for the 0.5B, world.1p5.ids for the 1.5B; the first L ids are the prefix)
// Configuration: plain only. Like the G6 decode timing harness, it refuses the test-seam configurations, so a reading
// is never taken on an engine compiled with fault-injection hooks.
#include "pcg_f3_common.h"

#include <cmath>
#include <cstdio>
#include <numeric>

#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION) || defined(SUPERSLM_GPU_TEST_SEAMS) || \
    defined(SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION) || defined(SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING)
#error "pcg_cell_10_4_timing times the shipping engine: build it in the plain configuration"
#endif

using namespace pcg;
using namespace pcg::f3;

namespace {

constexpr int kPairs = 8;
constexpr double kStudent95 = 2.365;  // two-sided t_0.975 with 7 degrees of freedom
constexpr double kBar = 100.0;

struct Pair {
	double prefill_ms = 0, restore_ms = 0;
	size_t bytes = 0;
	bool ok = false;
};

Pair OnePair(const Env& e, Rig& rig, const std::vector<int32_t>& tokens) {
	Pair p;
	const auto t0 = std::chrono::steady_clock::now();
	SslmGpuPrefixHandle* built = BatchedPrefix(rig, tokens);
	const auto t1 = std::chrono::steady_clock::now();
	if (!built) return p;
	const std::vector<uint8_t> b = SavePrefix(e, built);
	Release(rig, built);
	if (b.empty()) return p;
	const auto r0 = std::chrono::steady_clock::now();
	const RestoredPrefix r = RestorePrefix(rig, b);
	const auto r1 = std::chrono::steady_clock::now();
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "10.4 timing: restore %s", Name(r.status));
	if (!r.p) return p;
	Release(rig, r.p);
	p.prefill_ms = Ms(t0, t1);
	p.restore_ms = Ms(r0, r1);
	p.bytes = b.size();
	p.ok = p.prefill_ms > 0 && p.restore_ms > 0;
	return p;
}

void Cell(const Env& e, int argc, char** argv) {
	const std::string tag = Arg(argc, argv, "--tag"), role = Arg(argc, argv, "--role");
	std::string lengths = Arg(argc, argv, "--lengths");
	if (lengths.empty()) lengths = "128,1000";
	const bool args = !tag.empty() && (role == "accept" || role == "reject" || role == "grade");
	CHECK_MSG(args, "pass --tag=0p5|1p5 and --role=accept|reject|grade");
	if (!args) return;
	const std::vector<int32_t> ids = ReadIds(Arg(argc, argv, "--ids"));
	const std::vector<int32_t> Ls = Split(lengths);
	int64_t largest = 0;
	for (int32_t L : Ls) largest = std::max<int64_t>(largest, L);
	CHECK_MSG(!Ls.empty() && static_cast<int64_t>(ids.size()) >= largest, "10.4 timing: %zu ids for a largest length of %lld",
	          ids.size(), static_cast<long long>(largest));
	if (Ls.empty() || static_cast<int64_t>(ids.size()) < largest) return;
	// The built prefix, its restored copy, and (under the must-reject mutant) the restore's own internal prefill.
	const uint32_t pages = static_cast<uint32_t>(3 * e.geo.PrefixPages(largest) + 4);
	Rig rig(e, pages);
	if (!rig.pool || !RequireGpuGeometry(e, rig.pool, ("10.4 timing " + tag).c_str())) return;
	const std::string runner = RunnerSha(), adapter = AdapterTag(), model_sha = Sha(e.fx.bytes);
	CHECK_MSG(runner.size() == 64, "10.4 timing: the running program could not be hashed");
	const double tick_ms = 1000.0 * double(std::chrono::steady_clock::period::num) / double(std::chrono::steady_clock::period::den);
	std::printf("10.4 run role=%s runner=%s adapter=%s model=%s model_sha256=%s pairs=%d bar=%.0f tick_ms=%.6f\n",
	            role.c_str(), runner.empty() ? "unknown" : runner.c_str(), adapter.c_str(), tag.c_str(), model_sha.c_str(),
	            kPairs, kBar, tick_ms);
	std::fflush(stdout);
	for (int32_t L : Ls) {
		const std::vector<int32_t> tokens(ids.begin(), ids.begin() + L);
		(void)OnePair(e, rig, tokens);  // the untimed warm-up pair
		std::vector<double> logs;
		double pre = 0, res = 0;
		size_t bytes = 0;
		bool resolved = true;
		for (int j = 0; j < kPairs; ++j) {
			const Pair p = OnePair(e, rig, tokens);
			CHECK_MSG(p.ok, "10.4 timing %s L=%d: pair %d did not complete", tag.c_str(), L, j);
			if (!p.ok) return;
			resolved = resolved && p.prefill_ms > 100 * tick_ms && p.restore_ms > 100 * tick_ms;
			pre += p.prefill_ms;
			res += p.restore_ms;
			bytes = p.bytes;
			logs.push_back(std::log(p.prefill_ms / p.restore_ms));
		}
		const double mean = std::accumulate(logs.begin(), logs.end(), 0.0) / kPairs;
		double sq = 0;
		for (double v : logs) sq += (v - mean) * (v - mean);
		const double half = kStudent95 * std::sqrt(sq / (kPairs - 1) / kPairs);
		const double lo = std::exp(mean - half), hi = std::exp(mean + half);
		CHECK_MSG(resolved, "10.4 timing %s L=%d: a pair fell under the timer's resolving floor (100 ticks)", tag.c_str(), L);
		std::printf("10.4 reading role=%s runner=%s adapter=%s model=%s L=%d pairs=%d bytes_including_digest=%zu "
		            "prefill_ms=%.3f restore_ms=%.3f ratio95=[%.1f,%.1f] resolved=%d bar=%s\n",
		            role.c_str(), runner.empty() ? "unknown" : runner.c_str(), adapter.c_str(), tag.c_str(), L, kPairs, bytes,
		            pre / kPairs, res / kPairs, lo, hi, resolved ? 1 : 0, resolved && lo >= kBar ? "met" : "missed");
		std::fflush(stdout);
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		const std::string model = Arg(argc, argv, "--model");
		CHECK_MSG(!model.empty(), "pass --model=FILE");
		Env e;
		if (!model.empty() && e.Open(model, false)) Cell(e, argc, argv);
	}
	return FinishSuite("pcg_cell_10_4_timing");
}
