// The portable-forced SHA-256 build: src/sha256.cpp compiled with
// SUPERSLM_FORCE_PORTABLE_SHA256 (CMakeLists.txt, target sha256_portable_forced_tests), so the
// streaming object and the one-shot hash take the portable path whatever the CPU reports.
// superslm_tests covers the dispatched path (the x86 SHA extensions where the CPU has them);
// this binary covers the other arm of the same dispatch through the public API, against the
// FIPS 180-4 answers and against the hardware path called directly where the CPU has it.
#include "superslm/sha256.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef SUPERSLM_FORCE_PORTABLE_SHA256
#error "sha256_portable_forced_tests must be built with SUPERSLM_FORCE_PORTABLE_SHA256"
#endif

using namespace superslm;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond)                                                     \
	do {                                                                \
		++GChecks;                                                      \
		if (!(cond)) {                                                  \
			++GFailures;                                                \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		}                                                               \
	} while (0)

static std::vector<uint8_t> Bytes(size_t n, uint32_t seed) {
	std::vector<uint8_t> v(n);
	uint32_t x = seed;
	for (auto& b : v) {
		x = x * 1664525u + 1013904223u;
		b = static_cast<uint8_t>(x >> 24);
	}
	return v;
}

static std::string Hex(const uint8_t* data, size_t len) {
	uint8_t d[32];
	Sha256Hash(data, len, d);
	return ToHex(d);
}

int main() {
	// The force macro pins the dispatch; the CPU probe itself still reports the hardware.
	CHECK(ActiveSha256Impl() == kSha256ImplPortable);

	const std::string abc = "abc";
	CHECK(Hex(reinterpret_cast<const uint8_t*>(""), 0) ==
	      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
	CHECK(Hex(reinterpret_cast<const uint8_t*>(abc.data()), abc.size()) ==
	      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
	const std::string m896 =
	    "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqr"
	    "lmnopqrsmnopqrstnopqrstu";
	CHECK(Hex(reinterpret_cast<const uint8_t*>(m896.data()), m896.size()) ==
	      "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
	const std::string million(1000000, 'a');
	CHECK(Hex(reinterpret_cast<const uint8_t*>(million.data()), million.size()) ==
	      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
	const std::vector<uint8_t> big = Bytes((size_t(3) << 20) + 17, 0x5A17u);
	CHECK(Hex(big.data(), big.size()) ==
	      "29fe1f49486287b54b24dd1461d5de93cf5cc5b8e0842fd8e2dcf64b0eb609dd");

	// Streaming (portable, forced) at every two-way split for lengths 0..256 and
	// byte-at-a-time for 0..1024, against the portable one-shot and, where the CPU has
	// the SHA extensions, the hardware one-shot.
	const bool hw = DetectSha256ImplForCpu() == kSha256ImplShaNi;
	const std::vector<uint8_t> bytes = Bytes(1024, 0x51u);
	int mismatches = 0;
	for (size_t len = 0; len <= 1024; ++len) {
		uint8_t ref[32];
		Sha256HashPortableRef(bytes.data(), len, ref);
#if SUPERSLM_SHA256_HAVE_SHANI_X64
		if (hw) {
			uint8_t h2[32];
			Sha256HashShaNiRef(bytes.data(), len, h2);
			if (std::memcmp(h2, ref, 32) != 0) ++mismatches;
		}
#endif
		const size_t max_split = len <= 256 ? len : 0;
		for (size_t k = 0; k <= max_split; ++k) {
			Sha256 h;
			h.Update(bytes.data(), k);
			h.Update(bytes.data() + k, len - k);
			uint8_t d[32];
			h.Final(d);
			if (std::memcmp(d, ref, 32) != 0) ++mismatches;
		}
		Sha256 h;
		for (size_t i = 0; i < len; ++i) h.Update(bytes.data() + i, 1);
		uint8_t d[32];
		h.Final(d);
		if (std::memcmp(d, ref, 32) != 0) ++mismatches;
	}
	CHECK(mismatches == 0);
	std::printf("sha256 portable-forced: hardware cross-check %s\n",
	            hw ? "executed" : "not executed (SHA extensions unavailable here)");
	std::printf("sha256 portable-forced tests: %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
