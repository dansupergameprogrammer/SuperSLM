// t2145_bench.cpp -- T-2145 (Laplace) disposable experiment harness.
//
// NOT product code, NOT a suite test. Built only by t2145_build.bat, never by
// build.bat or CMakeLists.txt. Its job is to answer one question by execution:
// is the SuperSLM int8 CPU decode kernel compute-bound or memory-bandwidth-
// bound at its current SIMD width?
//
// The isolation, stated so the A/B is attributable to one variable:
//
//   `roof`   measures the machine's achievable DRAM read bandwidth with the
//            widest loads available (AVX2), over a buffer far larger than L3.
//            This is a property of the machine, not of the kernel, so it uses
//            the widest instruction regardless of which kernel variant this
//            binary was built with.
//
//   `sweep`  runs the PRODUCTION kernel (GemmInt8AccumulateRow, whichever
//            variant this binary was compiled with) at a range of weight-
//            matrix sizes. Every parameter is held fixed except the size of
//            the weight matrix -- so the only thing that changes across the
//            sweep is which level of the memory hierarchy the weights come
//            from. A kernel whose rate is flat from L1 through DRAM is
//            compute-bound. A kernel whose rate collapses at the L3 boundary
//            and lands on the `roof` figure is memory-bandwidth-bound.
//
//   `compete` runs a pure memory-bandwidth competitor (no model, no kernel)
//            and reports the bandwidth IT achieves. Run standalone it gives
//            the competitor's uncontended rate; run concurrently with a decode
//            it gives the cost the model imposes on a co-resident workload.
//
// Timing is steady_clock. Every mode reports every repetition, never a best
// run -- spread is a result, not noise to be filtered.

#include "superslm/matmul.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point a, Clock::time_point b) {
	return std::chrono::duration<double>(b - a).count();
}

// A global sink so no measured loop can be optimized away.
std::atomic<int64_t> g_sink{0};

void* AlignedAlloc(size_t bytes) {
#ifdef _WIN32
	return _aligned_malloc(bytes, 64);
#else
	return std::aligned_alloc(64, bytes);
#endif
}
void AlignedFree(void* p) {
#ifdef _WIN32
	_aligned_free(p);
#else
	std::free(p);
#endif
}

void PinToCpu(unsigned index) {
#ifdef _WIN32
	// Best-effort affinity so a thread-count sweep measures thread count rather
	// than the scheduler's migration policy. Reported, not required.
	const DWORD_PTR mask = static_cast<DWORD_PTR>(1ULL) << (index % 64);
	SetThreadAffinityMask(GetCurrentThread(), mask);
#else
	(void)index;
#endif
}

// --- the streaming read primitive used by `roof` and `compete` ----------------
// Reads `n` bytes with 256-bit loads and folds them into an accumulator that is
// returned, so the read cannot be elided. This is the machine's bandwidth, not
// the kernel's -- it deliberately uses the widest load this CPU has.
int64_t ReadSweep(const uint8_t* p, size_t n) {
	__m256i a0 = _mm256_setzero_si256();
	__m256i a1 = _mm256_setzero_si256();
	__m256i a2 = _mm256_setzero_si256();
	__m256i a3 = _mm256_setzero_si256();
	size_t i = 0;
	for (; i + 128 <= n; i += 128) {
		a0 = _mm256_add_epi64(a0, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i)));
		a1 = _mm256_add_epi64(a1, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 32)));
		a2 = _mm256_add_epi64(a2, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 64)));
		a3 = _mm256_add_epi64(a3, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 96)));
	}
	a0 = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
	alignas(32) int64_t lanes[4];
	_mm256_store_si256(reinterpret_cast<__m256i*>(lanes), a0);
	int64_t acc = lanes[0] + lanes[1] + lanes[2] + lanes[3];
	for (; i < n; ++i) acc += p[i];
	return acc;
}

// ---------------------------------------------------------------------------
// roof: achievable DRAM read bandwidth, `threads` threads, `reps` repetitions.
// ---------------------------------------------------------------------------
int ModeRoof(size_t total_bytes, unsigned threads, int reps) {
	uint8_t* buf = static_cast<uint8_t*>(AlignedAlloc(total_bytes));
	if (!buf) {
		std::fprintf(stderr, "roof: allocation of %zu bytes failed\n", total_bytes);
		return 1;
	}
	// Commit every page before timing; a first-touch page fault is not bandwidth.
	std::memset(buf, 1, total_bytes);

	std::printf("mode=roof buffer_bytes=%zu threads=%u reps=%d\n", total_bytes, threads, reps);
	for (int r = 0; r < reps; ++r) {
		std::atomic<bool> go{false};
		std::atomic<unsigned> ready{0};
		std::vector<std::thread> pool;
		const size_t slice = total_bytes / threads;
		for (unsigned t = 0; t < threads; ++t) {
			pool.emplace_back([&, t] {
				PinToCpu(t);
				ready.fetch_add(1);
				while (!go.load(std::memory_order_acquire)) {}
				const int64_t v = ReadSweep(buf + t * slice, slice);
				g_sink.fetch_add(v, std::memory_order_relaxed);
			});
		}
		while (ready.load() < threads) {}
		const auto t0 = Clock::now();
		go.store(true, std::memory_order_release);
		for (auto& th : pool) th.join();
		const auto t1 = Clock::now();
		const double s = Seconds(t0, t1);
		const double bytes = static_cast<double>(slice) * threads;
		std::printf("roof rep=%d seconds=%.6f GB_per_s=%.3f\n", r, s, bytes / s / 1e9);
		std::fflush(stdout);
	}
	AlignedFree(buf);
	return 0;
}

// ---------------------------------------------------------------------------
// sweep: the production kernel at a range of weight-matrix sizes.
//
// in_channels is FIXED at the real 1.5B hidden size. Only out_channels moves,
// so the only variable across rows is how large the weight matrix is -- i.e.
// which level of the hierarchy supplies it. Everything else (the activation
// vector, the kernel, the compiler flags, the thread count) is held constant.
// ---------------------------------------------------------------------------
void RunKernelCell(size_t in_channels, size_t out_channels, unsigned threads,
                    double target_seconds, const char* label) {
	const size_t weight_bytes = in_channels * out_channels;

	// Per-thread private weight matrices would multiply the footprint; instead
	// every thread reads the SAME matrix, which is what a multi-threaded decode
	// over one model would do (weights are shared, read-only).
	int8_t* weights = static_cast<int8_t*>(AlignedAlloc(weight_bytes));
	int8_t* acts = static_cast<int8_t*>(AlignedAlloc(in_channels));
	if (!weights || !acts) {
		std::fprintf(stderr, "sweep: allocation failed for %zu bytes\n", weight_bytes);
		return;
	}
	for (size_t i = 0; i < weight_bytes; ++i) weights[i] = static_cast<int8_t>((i * 31u) & 0x7f) - 64;
	for (size_t i = 0; i < in_channels; ++i) acts[i] = static_cast<int8_t>((i * 17u) & 0x7f) - 64;

	// Calibrate the repetition count so every cell runs for roughly the same
	// wall time regardless of size -- a large cell must not run 1 iteration
	// while a small one runs 10^6, or the timer resolution differs per row.
	size_t iters = 1;
	{
		std::vector<int64_t> out(out_channels);
		const auto c0 = Clock::now();
		superslm::GemmInt8AccumulateRow(acts, weights, in_channels, out_channels, out.data());
		const auto c1 = Clock::now();
		g_sink.fetch_add(out[0], std::memory_order_relaxed);
		const double one = Seconds(c0, c1);
		if (one > 0) iters = static_cast<size_t>(target_seconds / one);
		if (iters < 1) iters = 1;
	}

	// Threads PARTITION the output channels, so each thread sweeps a DISJOINT
	// slice of the weight matrix. This is how a threaded decode over one model
	// would actually split the work, and it matters for the measurement: if
	// every thread re-read the whole matrix instead, they would supply each
	// other's cache lines and the reported bandwidth would be inflated by
	// sharing that a real partitioned decode never gets.
	std::atomic<bool> go{false};
	std::atomic<unsigned> ready{0};
	std::vector<std::thread> pool;
	const size_t rows_per_thread = out_channels / threads;
	for (unsigned t = 0; t < threads; ++t) {
		const size_t row0 = static_cast<size_t>(t) * rows_per_thread;
		const size_t nrows = (t + 1 == threads) ? (out_channels - row0) : rows_per_thread;
		pool.emplace_back([&, t, row0, nrows] {
			PinToCpu(t);
			std::vector<int64_t> out(nrows ? nrows : 1);
			ready.fetch_add(1);
			while (!go.load(std::memory_order_acquire)) {}
			if (nrows) {
				for (size_t k = 0; k < iters; ++k) {
					superslm::GemmInt8AccumulateRow(acts, weights + row0 * in_channels, in_channels,
					                                 nrows, out.data());
				}
			}
			g_sink.fetch_add(out[0], std::memory_order_relaxed);
		});
	}
	while (ready.load() < threads) {}
	const auto t0 = Clock::now();
	go.store(true, std::memory_order_release);
	for (auto& th : pool) th.join();
	const auto t1 = Clock::now();

	const double s = Seconds(t0, t1);
	// The matrix is partitioned, not replicated, so one iteration sweeps
	// weight_bytes in total across all threads -- NOT weight_bytes per thread.
	const double bytes = static_cast<double>(weight_bytes) * iters;
	const double macs = bytes;  // one int8 MAC per weight byte
	std::printf("sweep label=%-6s in_ch=%zu out_ch=%-8zu weight_bytes=%-12zu threads=%u iters=%-8zu "
	            "seconds=%.6f GB_per_s=%.3f GMAC_per_s=%.3f\n",
	            label, in_channels, out_channels, weight_bytes, threads, iters, s,
	            bytes / s / 1e9, macs / s / 1e9);
	std::fflush(stdout);

	AlignedFree(weights);
	AlignedFree(acts);
}

int ModeSweep(unsigned threads, double target_seconds, int reps) {
	const size_t in_ch = 1536;  // the real 1.5B hidden size
	// Sizes chosen to straddle this machine's hierarchy: 32 KB L1d, 512 KB L2,
	// 16 MB L3 per CCX (64 MB total, but one thread only sees its own CCX's 16 MB).
	struct Cell { const char* label; size_t bytes; };
	const Cell cells[] = {
	    {"L1",   24u * 1024},          // 24 KB  -- inside 32 KB L1d
	    {"L2",   384u * 1024},         // 384 KB -- inside 512 KB L2
	    {"L3",   8u * 1024 * 1024},    // 8 MB   -- inside one CCX's 16 MB L3
	    {"L3hi", 12u * 1024 * 1024},   // 12 MB  -- still inside L3
	    {"DRAM", 256u * 1024 * 1024},  // 256 MB -- far outside L3
	    {"DRAMh", 1024u * 1024 * 1024},// 1 GB   -- the real 1.5B weight scale
	};
	std::printf("mode=sweep threads=%u target_seconds=%.2f reps=%d\n", threads, target_seconds, reps);
	for (int r = 0; r < reps; ++r) {
		for (const Cell& c : cells) {
			const size_t out_ch = c.bytes / in_ch;
			RunKernelCell(in_ch, out_ch, threads, target_seconds, c.label);
		}
	}
	return 0;
}

// ---------------------------------------------------------------------------
// compete: a pure memory-bandwidth competitor. Reports the bandwidth it itself
// achieves, every second, until `seconds` elapse. Run alone for its baseline;
// run alongside a decode to measure what the model costs a co-resident tenant.
// ---------------------------------------------------------------------------
int ModeCompete(size_t total_bytes, unsigned threads, double seconds) {
	uint8_t* buf = static_cast<uint8_t*>(AlignedAlloc(total_bytes));
	if (!buf) {
		std::fprintf(stderr, "compete: allocation of %zu bytes failed\n", total_bytes);
		return 1;
	}
	std::memset(buf, 1, total_bytes);
	std::printf("mode=compete buffer_bytes=%zu threads=%u seconds=%.1f\n", total_bytes, threads, seconds);
	std::fflush(stdout);

	std::atomic<bool> stop{false};
	std::atomic<uint64_t> bytes_done{0};
	std::vector<std::thread> pool;
	const size_t slice = total_bytes / threads;
	for (unsigned t = 0; t < threads; ++t) {
		pool.emplace_back([&, t] {
			PinToCpu(t);
			while (!stop.load(std::memory_order_relaxed)) {
				const int64_t v = ReadSweep(buf + t * slice, slice);
				g_sink.fetch_add(v, std::memory_order_relaxed);
				bytes_done.fetch_add(slice, std::memory_order_relaxed);
			}
		});
	}

	const auto t0 = Clock::now();
	uint64_t last = 0;
	auto last_t = t0;
	while (Seconds(t0, Clock::now()) < seconds) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		const auto now = Clock::now();
		const uint64_t cur = bytes_done.load();
		const double dt = Seconds(last_t, now);
		std::printf("compete t=%.2f GB_per_s=%.3f\n", Seconds(t0, now),
		            static_cast<double>(cur - last) / dt / 1e9);
		std::fflush(stdout);
		last = cur;
		last_t = now;
	}
	stop.store(true);
	for (auto& th : pool) th.join();
	AlignedFree(buf);
	return 0;
}

// ---------------------------------------------------------------------------
// identity: bit-identity of the built kernel against the in-tree scalar
// reference (DotRowScalarRef, matmul.h) over a population that deliberately
// includes the shapes a widened kernel is most likely to get wrong: lengths
// straddling every vector width, the extreme operand values, and lengths long
// enough to exercise the accumulator-flush window.
//
// This is an EXECUTED comparison against a reference that shares no parameter
// with the kernel under test: DotRowScalarRef is the normative construction and
// takes nothing from the SIMD path.
// ---------------------------------------------------------------------------
uint64_t SplitMix(uint64_t& s) {
	s += 0x9e3779b97f4a7c15ULL;
	uint64_t z = s;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

int ModeIdentity() {
	std::vector<size_t> lengths;
	// Every length from 1..300 catches all tail remainders for widths 8/16/32/64.
	for (size_t n = 1; n <= 300; ++n) lengths.push_back(n);
	// Straddle the flush window (kFlushBlocks * lanes) and the real model shapes.
	for (size_t n : {size_t(896), size_t(1535), size_t(1536), size_t(1537), size_t(4864),
	                  size_t(8960), size_t(131071), size_t(131072), size_t(131073),
	                  size_t(262144), size_t(262145), size_t(1000003), size_t(1200000),
	                  size_t(2500000)}) {
		lengths.push_back(n);
	}

	size_t checked = 0, mismatches = 0;
	uint64_t s = 0x5157345ULL;

	for (size_t n : lengths) {
		std::vector<int8_t> a(n), w(n);
		// Three populations per length: random, all-extreme-negative (the -128 *
		// -128 corner a saturating construction gets wrong), and alternating
		// extremes (the corner that maximizes partial-sum magnitude).
		for (int pop = 0; pop < 3; ++pop) {
			for (size_t i = 0; i < n; ++i) {
				if (pop == 0) {
					a[i] = static_cast<int8_t>(SplitMix(s) & 0xff);
					w[i] = static_cast<int8_t>(SplitMix(s) & 0xff);
				} else if (pop == 1) {
					a[i] = -128;
					w[i] = -128;
				} else {
					a[i] = (i & 1) ? int8_t(-128) : int8_t(127);
					w[i] = (i & 1) ? int8_t(127) : int8_t(-128);
				}
			}
			int64_t got = 0;
			superslm::GemmInt8AccumulateRow(a.data(), w.data(), n, 1, &got);
			const int64_t want = superslm::DotRowScalarRef(a.data(), w.data(), n);
			++checked;
			if (got != want) {
				++mismatches;
				if (mismatches <= 10) {
					std::printf("identity MISMATCH n=%zu pop=%d got=%lld want=%lld\n", n, pop,
					            static_cast<long long>(got), static_cast<long long>(want));
				}
			}
		}
	}
	std::printf("identity checked=%zu mismatches=%zu\n", checked, mismatches);
	return mismatches == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr,
		             "usage:\n"
		             "  t2145_bench roof [buffer_MB] [threads] [reps]\n"
		             "  t2145_bench sweep [threads] [target_seconds] [reps]\n"
		             "  t2145_bench compete [buffer_MB] [threads] [seconds]\n"
		             "  t2145_bench identity\n");
		return 2;
	}
	const std::string mode = argv[1];
	auto arg = [&](int i, long dflt) -> long {
		return (argc > i) ? std::strtol(argv[i], nullptr, 10) : dflt;
	};
	auto argd = [&](int i, double dflt) -> double {
		return (argc > i) ? std::strtod(argv[i], nullptr) : dflt;
	};

	int rc = 0;
	if (mode == "roof") {
		rc = ModeRoof(static_cast<size_t>(arg(2, 2048)) * 1024u * 1024u,
		              static_cast<unsigned>(arg(3, 1)), static_cast<int>(arg(4, 3)));
	} else if (mode == "sweep") {
		rc = ModeSweep(static_cast<unsigned>(arg(2, 1)), argd(3, 0.5), static_cast<int>(arg(4, 3)));
	} else if (mode == "compete") {
		rc = ModeCompete(static_cast<size_t>(arg(2, 2048)) * 1024u * 1024u,
		                 static_cast<unsigned>(arg(3, 4)), argd(4, 30.0));
	} else if (mode == "identity") {
		rc = ModeIdentity();
	} else {
		std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
		return 2;
	}
	std::printf("sink=%lld\n", static_cast<long long>(g_sink.load()));
	return rc;
}
