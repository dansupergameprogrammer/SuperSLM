// sslm_gemm_bench.cpp -- tiled-matmul plan slice 1, cell 10.1: the per-shape GEMM bench.
//
// Times superslm::GemmInt8Accumulate on the prefill projection shapes of the Qwen2.5-0.5B, Qwen3-0.6B and
// Qwen2.5-1.5B geometries, with no artifact: activations and weights are filled from a fixed LCG. The tool
// is built several times from this one source (CMakeLists.txt): against the production library (auto
// dispatch), against the forced-AVX2 and forced-AVX-512 libraries, and against a "D-infinity" build of the
// forced libraries (SUPERSLM_TEST_TILED_MIN_TOKENS=SIZE_MAX: the shipped one-cell-at-a-time loop at every
// M). A ratio between two builds is therefore a same-source, same-host comparison of the dispatch.
//
// Every timed call's output is checked against the first call's (and, with --verify, against the scalar
// reference), so a build that got faster by computing something else fails loudly instead of reporting.
//
// Usage:
//   sslm_gemm_bench [--shape=NAME|all] [--m=8,32,128,512] [--reps=R] [--verify]
//     prints one line per (shape, M): the best-of-R and median milliseconds and GMAC/s.
//   sslm_gemm_bench --shape=NAME --m=M --reps=R --one
//     prints one line "time_ms=<best of R>", for an external driver that interleaves two builds
//     (tools/sslm_gemm_bench_rule.py applies the plan's decision rule to those pairs).
// NAME is <model>.<projection>, e.g. 1.5B.gate_up; --shape=list prints every name.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/matmul.h"

namespace {

struct Shape {
	std::string name;
	size_t in_channels;   // K
	size_t out_channels;  // N
};

std::vector<Shape> AllShapes() {
	struct Geometry {
		const char* model;
		size_t hidden, q_heads, kv_heads, head_dim, intermediate;
	};
	static const Geometry kGeometries[] = {
	    {"0.5B", 896, 14, 2, 64, 4864},
	    {"0.6B", 1024, 16, 8, 128, 3072},
	    {"1.5B", 1536, 12, 2, 128, 8960},
	};
	std::vector<Shape> shapes;
	for (const Geometry& g : kGeometries) {
		const std::string m = g.model;
		shapes.push_back({m + ".q", g.hidden, g.q_heads * g.head_dim});
		shapes.push_back({m + ".kv", g.hidden, g.kv_heads * g.head_dim});  // k and v each
		shapes.push_back({m + ".o", g.q_heads * g.head_dim, g.hidden});
		shapes.push_back({m + ".gate_up", g.hidden, g.intermediate});  // gate and up each
		shapes.push_back({m + ".down", g.intermediate, g.hidden});
	}
	return shapes;
}

struct Lcg {
	uint64_t s;
	explicit Lcg(uint64_t seed) : s(seed) {}
	int8_t Next() {
		s = s * 6364136223846793005ULL + 1442695040888963407ULL;
		return static_cast<int8_t>(static_cast<int>((s >> 33) % 255) - 127);  // [-127, 127]
	}
};

double TimeOnce(const std::vector<int8_t>& a, const std::vector<int8_t>& w, size_t m, const Shape& s,
                std::vector<int64_t>* out) {
	const auto t0 = std::chrono::steady_clock::now();
	superslm::GemmInt8Accumulate(a.data(), w.data(), m, s.in_channels, s.out_channels, out->data());
	const auto t1 = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::vector<size_t> ParseList(const std::string& text) {
	std::vector<size_t> out;
	size_t pos = 0;
	while (pos < text.size()) {
		const size_t comma = text.find(',', pos);
		out.push_back(static_cast<size_t>(std::strtoull(text.substr(pos, comma - pos).c_str(), nullptr, 10)));
		if (comma == std::string::npos) break;
		pos = comma + 1;
	}
	return out;
}

#if defined(SUPERSLM_TEST_TILED_MIN_TOKENS)
#define SUPERSLM_GEMM_BENCH_DINF ", D-infinity (tiled path off)"
#else
#define SUPERSLM_GEMM_BENCH_DINF ""
#endif

const char* BuildLabel() {
#if defined(SUPERSLM_FORCE_AVX2_MATMUL)
	return "forced AVX2" SUPERSLM_GEMM_BENCH_DINF;
#elif defined(SUPERSLM_FORCE_AVX512_MATMUL)
	return "forced AVX-512" SUPERSLM_GEMM_BENCH_DINF;
#else
	return "auto dispatch" SUPERSLM_GEMM_BENCH_DINF;
#endif
}

}  // namespace

int main(int argc, char** argv) {
	std::string shape_arg = "all";
	std::vector<size_t> ms = {8, 32, 128, 512};
	int reps = 5;
	bool one = false;
	bool verify = false;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--shape=", 0) == 0) shape_arg = a.substr(8);
		else if (a.rfind("--m=", 0) == 0) ms = ParseList(a.substr(4));
		else if (a.rfind("--reps=", 0) == 0) reps = std::max(1, std::atoi(a.c_str() + 7));
		else if (a == "--one") one = true;
		else if (a == "--verify") verify = true;
		else {
			std::fprintf(stderr, "unknown argument %s\n", a.c_str());
			return 2;
		}
	}
	const std::vector<Shape> all = AllShapes();
	if (shape_arg == "list") {
		for (const Shape& s : all) std::printf("%s K=%zu N=%zu\n", s.name.c_str(), s.in_channels, s.out_channels);
		return 0;
	}
	std::vector<Shape> shapes;
	for (const Shape& s : all)
		if (shape_arg == "all" || s.name == shape_arg) shapes.push_back(s);
	if (shapes.empty() || ms.empty() || (one && (shapes.size() != 1 || ms.size() != 1))) {
		std::fprintf(stderr, "bad --shape/--m (--one needs exactly one of each; --shape=list lists names)\n");
		return 2;
	}
	if (!one) std::printf("# sslm_gemm_bench: %s build; best and median of %d timed calls after one warm-up\n",
	                      BuildLabel(), reps);

	for (const Shape& s : shapes) {
		Lcg rng(0x5EED0000ULL + s.in_channels * 31 + s.out_channels);
		std::vector<int8_t> w(s.in_channels * s.out_channels);
		for (int8_t& v : w) v = rng.Next();
		for (size_t m : ms) {
			std::vector<int8_t> a(m * s.in_channels);
			for (int8_t& v : a) v = rng.Next();
			std::vector<int64_t> first(m * s.out_channels), out(m * s.out_channels);
			TimeOnce(a, w, m, s, &first);  // warm-up; also the output every timed call must reproduce
			if (verify) {
				for (size_t t = 0; t < m; ++t)
					for (size_t j = 0; j < s.out_channels; j += 7)
						if (first[t * s.out_channels + j] !=
						    superslm::DotRowScalarRef(a.data() + t * s.in_channels, w.data() + j * s.in_channels,
						                              s.in_channels)) {
							std::fprintf(stderr, "VERIFY FAILED: %s M=%zu t=%zu j=%zu\n", s.name.c_str(), m, t, j);
							return 3;
						}
			}
			std::vector<double> times;
			for (int r = 0; r < reps; ++r) {
				times.push_back(TimeOnce(a, w, m, s, &out));
				if (out != first) {
					std::fprintf(stderr, "OUTPUT CHANGED between calls: %s M=%zu\n", s.name.c_str(), m);
					return 3;
				}
			}
			std::sort(times.begin(), times.end());
			const double best = times.front();
			const double median = times[times.size() / 2];
			const double macs = static_cast<double>(m) * s.in_channels * s.out_channels;
			if (one) {
				std::printf("time_ms=%.4f\n", best);
			} else {
				std::printf("%-14s K=%-5zu N=%-5zu M=%-4zu best %9.3f ms  median %9.3f ms  %6.1f GMAC/s\n",
				            s.name.c_str(), s.in_channels, s.out_channels, m, best, median, macs / best / 1e6);
			}
		}
	}
	return 0;
}
