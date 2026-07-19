// tok_verify — run the runtime TokenizerView over a binary golden pack and report
// any divergence from the upstream ids. Build-time cross-check (not shipped): the
// committed test gate covers a curated corpus; this drives thousands of lines.
//   cl /std:c++20 /EHsc /Iinclude src\artifact.cpp src\sha256.cpp src\tokenizer.cpp tools\tok_verify.cpp
#include "superslm/artifact.h"
#include "superslm/tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace superslm;

static uint32_t Rd32(const uint8_t* p) {
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

int main(int argc, char** argv) {
	if (argc < 3) { std::printf("usage: tok_verify <artifact.sslm> <golden.gld>\n"); return 2; }
	SslmArtifact art;
	SslmError e;
	if (SslmArtifact::OpenFromFile(argv[1], art, &e) != SslmStatus::Ok) {
		std::printf("artifact load failed: %s\n", e.message.c_str());
		return 1;
	}
	TokenizerView tok;
	std::string terr;
	if (!TokenizerView::Open(art, tok, &terr)) { std::printf("tokenizer open failed: %s\n", terr.c_str()); return 1; }

	std::ifstream f(argv[2], std::ios::binary);
	std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (d.size() < 40 || std::string(d.begin(), d.begin() + 4) != "GLD1") { std::printf("bad golden\n"); return 1; }
	uint32_t count = Rd32(d.data() + 4);
	size_t pos = 40;
	int mism = 0;
	for (uint32_t r = 0; r < count; ++r) {
		uint32_t tlen = Rd32(d.data() + pos); pos += 4;
		std::string text(reinterpret_cast<const char*>(d.data() + pos), tlen); pos += tlen;
		uint32_t ic = Rd32(d.data() + pos); pos += 4;
		std::vector<int32_t> want(ic);
		for (uint32_t i = 0; i < ic; ++i) { want[i] = int32_t(Rd32(d.data() + pos)); pos += 4; }
		std::vector<int32_t> got = tok.Encode(text);
		if (got != want) {
			if (++mism <= 10) std::printf("MISMATCH record %u: %s\n", r, text.c_str());
		}
	}
	std::printf("%u records, %d mismatches\n", count, mism);
	return mism ? 1 : 0;
}
