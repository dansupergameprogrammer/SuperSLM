// S5 forward probe (bench.md, reading 2). Not built by CMake. fx.h is tests/support/qk_attention_fixture.h with:
//   sed -e 's/kHidden = 256/kHidden = 1024/; s/kHeads = 4;/kHeads = 16;/; s/kKvHeads = 2;/kKvHeads = 8;/;
//           s/kHeadDim = 64;/kHeadDim = 128;/; s/kInter = 256;/kInter = 3072;/; s/kCap = 32;/kCap = 1024;/;
//           s/kPositions = 24;/kPositions = 1024;/; s/superslm_qk_fixture/q31fwd/g; s#"\.\./sslm_#"sslm_#'
//       -e 's/hidden_gain.assign(kHidden, 8192);/hidden_gain.assign(kHidden, HG);/; s/rng.InRange(4096, 8192));/rng.InRange(QKLO, QKHI));/g'
//   (plus the guard rename and the other constants made overridable, all left at the fixture's values).
// Built with: g++ -O3 -DNDEBUG -std=gnu++20 -ffp-contract=off -DHG=4096 -DQKLO=2048 -DQKHI=4096 -I<tree>/include -Itests
// and linked against the S4 head's or S5's libsuperslm.a / libsuperslm_avx2_forced.a. Usage: probe <T> <repeat>.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifndef KVE
#define KVE 12
#endif
#ifndef KCE
#define KCE -36
#endif
#ifndef SME
#define SME -72
#endif
#ifndef GSE
#define GSE -52
#endif
#ifndef KWE
#define KWE -60
#endif
#ifndef ACTE
#define ACTE INT64_C(-96)
#endif
#ifndef HG
#define HG 8192
#endif
#ifndef QKLO
#define QKLO 4096
#define QKHI 8192
#endif
#include "fx.h"
using namespace q31fwd;
int main(int argc, char** argv) {
	const size_t T = argc > 1 ? std::atoi(argv[1]) : 128;
	const int R = argc > 2 ? std::atoi(argv[2]) : 1;
	static QkAttentionFixture f;
	if (!f.loaded) { std::printf("load failed: %s\n", f.load_error.c_str()); return 2; }
	double best = 1e300; int st = 0; uint64_t h = 1469598103934665603ULL;
	for (int r = 0; r < R; ++r) {
		std::vector<uint8_t> workspace(kWorkspaceBytes, 0);
		std::vector<int8_t> chunk(f.hidden_in.begin(), f.hidden_in.begin() + T * kHidden);
		std::vector<superslm::CarriedScale> scales(f.scale_in.begin(), f.scale_in.begin() + T);
		uint64_t sat = 0;
		const auto t0 = std::chrono::steady_clock::now();
		st = static_cast<int>(superslm::RunLayerLoopChunkBatched(chunk.data(), scales.data(), T, &f.layer, 1, kHidden,
		    kHeadDim, kKvHeads, kInter, kCap, 0, f.view.rope_tables, workspace.data(), workspace.size(), false, &sat, {},
		    nullptr, kHeads * kHeadDim));
		const auto t1 = std::chrono::steady_clock::now();
		best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
		if (r == 0) { for (auto c : chunk) h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ULL;
			for (auto s : scales) h = (h ^ static_cast<uint64_t>(s.m) ^ (static_cast<uint64_t>(s.e) << 40)) * 1099511628211ULL;
			int clamp = 0; for (auto c : chunk) clamp += (c == 127 || c == -127 || c == -128);
			std::printf("T=%zu status %d sat %llu out-clamped %d/%zu hash %016llx\n", T, st, (unsigned long long)sat, clamp, chunk.size(), (unsigned long long)h); }
	}
	std::printf("T=%zu best-of-%d %.3f ms total, %.4f ms/token\n", T, R, best, best / T);
	return st;
}
