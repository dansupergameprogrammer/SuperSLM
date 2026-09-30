// Attention and per-row sites plan (rev 3.1), §3.3 evidence 3: the golden-pin generator.
//
// Built against the v1.9.0 TAG's library -- the normative per-element code, independent of every
// kernel and table under test -- it runs each slice's fixed input set and writes
// tests/attn_rowsite_golden_pin.h, one SHA-256 per slice, each serialized as little-endian int64
// exactly as the digest serializes the same stream:
//   - S1: tests/support/rowsite_cases.h through the three per-row sites (every call's status, output
//     scale and output row; the digest's `c_rowsites` section);
//   - S2: tests/support/attention_cases.h through GemmProbQ15Accumulate (every call's width, head_dim
//     and output row; the digest's `c32_attention` section);
//   - S3: tests/support/rowsite_cases.h's requant rows through RequantChainChecked (every call's
//     status, output scale and codes; appended to the digest's `c_rowsites` section);
//   - S4: tests/support/attention_cases.h's softmax rows through SoftmaxRowQ15 (every call's width,
//     bool and output row; appended to the digest's `c32_attention` section);
//   - S5: tests/support/attention_cases.h's Q31 rows through the v1.9.0 per-key QkQ31Score (every call's
//     width, head_dim and scores; the build under test runs them through QkQ31ScoreRow, and the digest
//     appends them to `c32_attention`), plus a second hash: tests/support/qk_attention_fixture.h's widened
//     QK-norm fixture through the decode loop (24 positions' output codes and scales, then the K/V bytes),
//     which the suite's cell 11.1(c) requires of the decode loop, the chunk loop and the decode loop with a
//     capture sink alike.
// Every tier of every build must reproduce every hash (tests/test_attn_rowsites.cpp, cell 6.3), so
// the reference takes no input from the code it grades. One hash per slice, so no later slice
// regenerates an earlier one's.
//
// Recipe (the one used for the committed pin; see docs/attention-rowsites/s1/golden.txt, s2/golden.txt,
// s3/golden.txt, s4/golden.txt and s5/golden.txt):
//   git worktree add /tmp/v190 v1.9.0
//   cmake -S /tmp/v190 -B /tmp/v190/build -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/v190/build --target superslm
//   c++ -std=c++20 -O2 -I/tmp/v190/include tools/gen_attn_rowsite_golden.cpp /tmp/v190/build/libsuperslm.a
//       -o gen_attn_rowsite_golden
//   ./gen_attn_rowsite_golden tests/attn_rowsite_golden_pin.h
// It also builds against the current tree (the CMake target of the same name), where it must
// print the same hash; that is a consistency check, not the pin's provenance.

#include <cstdint>
#include <cstdio>
#include <string>

#include "superslm/forward_sites.h"
#include "superslm/sha256.h"
#include "../tests/support/attention_cases.h"
#include "../tests/support/qk_attention_fixture.h"
#include "../tests/support/rowsite_cases.h"

namespace {

struct Hashed {
	std::string hex;
	unsigned long long values = 0;
};

template <class Run>
Hashed HashStream(Run run) {
	superslm::Sha256 h;
	Hashed r;
	auto emit = [&](int64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++r.values;
	};
	run(emit);
	uint8_t digest[32];
	h.Final(digest);
	r.hex = superslm::ToHex(digest);
	return r;
}

}  // namespace

int main(int argc, char** argv) {
	const Hashed s1 = HashStream([](auto& emit) { superslm_rowsite_cases::RunRowTableCases(emit); });
	std::printf("S1 row-table golden: %s over %llu values\n", s1.hex.c_str(), s1.values);
	const Hashed s2 = HashStream([](auto& emit) { superslm_attention_cases::RunProbVCases(emit); });
	std::printf("S2 prob-V golden: %s over %llu values\n", s2.hex.c_str(), s2.values);
	const Hashed s3 = HashStream([](auto& emit) { superslm_rowsite_cases::RunRequantRowCases(emit); });
	std::printf("S3 requant-row golden: %s over %llu values\n", s3.hex.c_str(), s3.values);
	const Hashed s4 = HashStream([](auto& emit) { superslm_attention_cases::RunSoftmaxCases(emit); });
	std::printf("S4 softmax golden: %s over %llu values\n", s4.hex.c_str(), s4.values);
	// v1.9.0 has no row entry: the per-key loop the layer loops ran.
	auto per_key_row = [](const int8_t* q, const int8_t* keys, const int64_t* ratio, size_t hd, size_t width,
	                      int64_t* out) {
		for (size_t j = 0; j < width; ++j) out[j] = superslm::QkQ31Score(q, keys + j * hd, ratio, hd);
	};
	const Hashed s5 = HashStream([&](auto& emit) { superslm_attention_cases::RunQ31Cases(emit, per_key_row); });
	std::printf("S5 Q31-row golden: %s over %llu values\n", s5.hex.c_str(), s5.values);
	int fixture_status = -1;
	const Hashed s5f = HashStream([&](auto& emit) {
		superslm_qk_fixture::QkAttentionFixture f;
		if (!f.loaded) return;
		fixture_status = static_cast<int>(superslm_qk_fixture::RunFixtureDecode(f, emit));
	});
	std::printf("S5 QK-norm fixture golden: %s over %llu values (status %d)\n", s5f.hex.c_str(), s5f.values,
	            fixture_status);
	if (fixture_status != static_cast<int>(superslm::SslmForwardStatus::Ok))
		return std::fprintf(stderr, "the QK-norm fixture did not run Ok\n"), 1;
	if (argc < 2) return 0;
	FILE* f = std::fopen(argv[1], "wb");
	if (!f) return std::fprintf(stderr, "cannot write %s\n", argv[1]), 1;
	std::fprintf(f,
	             "// GENERATED FILE. Do not hand-edit.\n"
	             "//\n"
	             "// Produced by tools/gen_attn_rowsite_golden.cpp built against the v1.9.0 tag's library (the\n"
	             "// normative per-element code), over tests/support/rowsite_cases.h's and\n"
	             "// tests/support/attention_cases.h's fixed input sets, one hash per slice. Attention and per-row\n"
	             "// sites plan, §3.3 evidence 3, coverage cell 6.3. Re-running the generator against v1.9.0 must\n"
	             "// reproduce this file byte-for-byte.\n"
	             "#ifndef SUPERSLM_TESTS_ATTN_ROWSITE_GOLDEN_PIN_H\n"
	             "#define SUPERSLM_TESTS_ATTN_ROWSITE_GOLDEN_PIN_H\n"
	             "\n"
	             "#include <cstdint>\n"
	             "\n"
	             "namespace superslm_test {\n"
	             "\n"
	             "// Slice S1: RmsNormSite, MlpActSite and ResidualReconcileSite over RunRowTableCases.\n"
	             "inline constexpr const char* kAttnRowsiteS1GoldenHash =\n"
	             "    \"%s\";\n"
	             "inline constexpr uint64_t kAttnRowsiteS1GoldenValues = %lluULL;\n"
	             "\n"
	             "// Slice S2: GemmProbQ15Accumulate over RunProbVCases.\n"
	             "inline constexpr const char* kAttnRowsiteS2GoldenHash =\n"
	             "    \"%s\";\n"
	             "inline constexpr uint64_t kAttnRowsiteS2GoldenValues = %lluULL;\n"
	             "\n"
	             "// Slice S3: RequantChainChecked's element loop over RunRequantRowCases.\n"
	             "inline constexpr const char* kAttnRowsiteS3GoldenHash =\n"
	             "    \"%s\";\n"
	             "inline constexpr uint64_t kAttnRowsiteS3GoldenValues = %lluULL;\n"
	             "\n"
	             "// Slice S4: SoftmaxRowQ15 over RunSoftmaxCases.\n"
	             "inline constexpr const char* kAttnRowsiteS4GoldenHash =\n"
	             "    \"%s\";\n"
	             "inline constexpr uint64_t kAttnRowsiteS4GoldenValues = %lluULL;\n"
	             "\n"
	             "// Slice S5: QkQ31Score per key over RunQ31Cases.\n"
	             "inline constexpr const char* kAttnRowsiteS5GoldenHash =\n"
	             "    \"%s\";\n"
	             "inline constexpr uint64_t kAttnRowsiteS5GoldenValues = %lluULL;\n"
	             "\n"
	             "// Slice S5, cell 11.1(c): the QK-norm fixture's forward through the decode loop.\n"
	             "inline constexpr const char* kAttnRowsiteS5FixtureGoldenHash =\n"
	             "    \"%s\";\n"
	             "inline constexpr uint64_t kAttnRowsiteS5FixtureGoldenValues = %lluULL;\n"
	             "\n"
	             "}  // namespace superslm_test\n"
	             "\n"
	             "#endif  // SUPERSLM_TESTS_ATTN_ROWSITE_GOLDEN_PIN_H\n",
	             s1.hex.c_str(), s1.values, s2.hex.c_str(), s2.values, s3.hex.c_str(), s3.values, s4.hex.c_str(), s4.values,
	             s5.hex.c_str(), s5.values, s5f.hex.c_str(), s5f.values);
	std::fclose(f);
	return 0;
}
