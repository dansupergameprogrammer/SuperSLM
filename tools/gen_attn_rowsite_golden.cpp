// Attention and per-row sites plan (rev 3.1), §3.3 evidence 3: the golden-pin generator.
//
// Built against the v1.9.0 TAG's library -- the normative per-element code, independent of every
// kernel and table under test -- it runs the fixed input set of tests/support/rowsite_cases.h
// through the three per-row sites and writes tests/attn_rowsite_golden_pin.h: a SHA-256 over every
// call's status, output scale and output row, serialized as little-endian int64 exactly as the
// digest's `c_rowsites` section serializes them. Every tier of every build must reproduce the hash
// (tests/test_attn_rowsites.cpp, cell 6.3), so the reference takes no input from the code it grades.
//
// Recipe (the one used for the committed pin; see docs/attention-rowsites/s1/golden.txt):
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

#include "superslm/sha256.h"
#include "../tests/support/rowsite_cases.h"

int main(int argc, char** argv) {
	superslm::Sha256 h;
	unsigned long long values = 0;
	auto emit = [&](int64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++values;
	};
	superslm_rowsite_cases::RunRowTableCases(emit);
	uint8_t digest[32];
	h.Final(digest);
	const std::string hex = superslm::ToHex(digest);
	std::printf("S1 row-table golden: %s over %llu values\n", hex.c_str(), values);
	if (argc < 2) return 0;
	FILE* f = std::fopen(argv[1], "wb");
	if (!f) return std::fprintf(stderr, "cannot write %s\n", argv[1]), 1;
	std::fprintf(f,
	             "// GENERATED FILE. Do not hand-edit.\n"
	             "//\n"
	             "// Produced by tools/gen_attn_rowsite_golden.cpp built against the v1.9.0 tag's library (the\n"
	             "// normative per-element code), over tests/support/rowsite_cases.h's fixed input set. Attention\n"
	             "// and per-row sites plan, §3.3 evidence 3, coverage cell 6.3. Re-running the generator against\n"
	             "// v1.9.0 must reproduce this file byte-for-byte.\n"
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
	             "}  // namespace superslm_test\n"
	             "\n"
	             "#endif  // SUPERSLM_TESTS_ATTN_ROWSITE_GOLDEN_PIN_H\n",
	             hex.c_str(), values);
	std::fclose(f);
	return 0;
}
