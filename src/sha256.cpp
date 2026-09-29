#include "superslm/sha256.h"

#include <algorithm>

#include "bad_alloc_wrap.h"

// Block compression has two implementations with identical output: the portable
// FIPS 180-4 code (the reference, and the only path on a non-x64 target) and the x86
// SHA extensions (sha256rnds2 / sha256msg1 / sha256msg2, plus SSSE3/SSE4.1 shuffles),
// selected once per process by a CPUID probe. The dispatch mirrors matmul.cpp's
// (T-2149 design §6): a per-function target attribute, never a TU-wide ISA flag; a
// pure resolver over CPUID fields that a test drives with fabricated values; and a
// force macro, SUPERSLM_FORCE_PORTABLE_SHA256, that pins the portable path so a build
// can test it deliberately (see include/superslm/sha256.h).
#if SUPERSLM_SHA256_HAVE_SHANI_X64
#include <immintrin.h>    // SHA-extension, SSSE3 and SSE4.1 intrinsics
#if defined(_MSC_VER)
#include <intrin.h>       // __cpuidex
#else
#include <cpuid.h>        // __cpuid_count
#endif
#endif  // SUPERSLM_SHA256_HAVE_SHANI_X64

// Same reasoning as matmul.cpp's SUPERSLM_AVX2_TARGET (design §6.4): GCC and Clang
// accept these intrinsics only in a function compiled with the ISA enabled, and a
// TU-wide -msha/-msse4.1 would let the compiler use them anywhere in this file. MSVC
// does not gate intrinsics by /arch, so the macro is empty there. clang-cl defines
// _MSC_VER but generates code with LLVM, which gates intrinsics like Clang on Linux,
// and it does not define __GNUC__ -- so it takes the attributed path through
// __clang__.
#if defined(__clang__) || (defined(__GNUC__) && !defined(_MSC_VER))
#define SUPERSLM_SHANI_TARGET __attribute__((target("sha,sse4.1,ssse3")))
#else
#define SUPERSLM_SHANI_TARGET
#endif

namespace superslm {
namespace {

inline uint32_t Ror(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

constexpr uint32_t kInit[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

constexpr uint32_t kK[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
	0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
	0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
	0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
	0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
	0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

// Compresses one 64-byte block at `p` into `h`. One block per call, with the block
// loop in the shared Absorb/Finish below: the loop is then exercised on every host
// whichever implementation runs, so the SHA-extension function has no branch of its
// own that a host without the extensions leaves unmeasured. Measured on the
// development host, the per-block indirect call costs nothing visible next to a
// multi-block loop inside the function (within run-to-run noise).
using CompressFn = void (*)(uint32_t h[8], const uint8_t* p);

// --- The portable compression (FIPS 180-4 §6.2.2, the reference) ------------------

void CompressPortable(uint32_t h_[8], const uint8_t* p) {
	uint32_t w[64];
	for (int i = 0; i < 16; ++i) {
		w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
		       (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
	}
	for (int i = 16; i < 64; ++i) {
		uint32_t s0 = Ror(w[i - 15], 7) ^ Ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
		uint32_t s1 = Ror(w[i - 2], 17) ^ Ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
	uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
	for (int i = 0; i < 64; ++i) {
		uint32_t S1 = Ror(e, 6) ^ Ror(e, 11) ^ Ror(e, 25);
		uint32_t ch = (e & f) ^ (~e & g);
		uint32_t t1 = h + S1 + ch + kK[i] + w[i];
		uint32_t S0 = Ror(a, 2) ^ Ror(a, 13) ^ Ror(a, 22);
		uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
		uint32_t t2 = S0 + maj;
		h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
	}
	h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
	h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

#if SUPERSLM_SHA256_HAVE_SHANI_X64

// --- The SHA-extension compression -------------------------------------------------
//
// The instructions keep the working variables as two vectors, ABEF and CDGH
// (lane 3 first), so the state is permuted into that layout on entry and back on
// exit. Each round group g (0..15) runs four rounds: sha256rnds2 does two rounds with
// the low two lanes of W+K, and the high two lanes follow after a shuffle. W for
// groups 4..15 is scheduled from the previous four groups:
//   W[g] = sha256msg2(sha256msg1(W[g-4], W[g-3]) + alignr(W[g-1], W[g-2], 4), W[g-1])
// which is FIPS 180-4's W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16], four
// words at a time (msg1 adds s0(W[t-15]), the alignr supplies W[t-7], msg2 adds
// s1(W[t-2]) in order within the group). The rounds are written out, not looped,
// so the four message vectors stay in registers.
#define SUPERSLM_SHANI_ROUNDS4(g, wg)                                                   \
	do {                                                                            \
		const __m128i msg_ = _mm_add_epi32(                                         \
		    (wg), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kK[4 * (g)]))); \
		cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg_);                             \
		abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32(msg_, 0x0E));    \
	} while (0)
#define SUPERSLM_SHANI_SCHEDULE(w0, w1, w2, w3)                                          \
	(w0) = _mm_sha256msg2_epu32(                                                        \
	    _mm_add_epi32(_mm_sha256msg1_epu32((w0), (w1)), _mm_alignr_epi8((w3), (w2), 4)), \
	    (w3))

SUPERSLM_SHANI_TARGET void CompressShaNi(uint32_t h[8], const uint8_t* p) {
	// Big-endian word loads: reverse the bytes of each 32-bit lane.
	const __m128i kBswap32 = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);

	__m128i dcba = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&h[0]));
	__m128i hgfe = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&h[4]));
	const __m128i cdab = _mm_shuffle_epi32(dcba, 0xB1);
	const __m128i efgh = _mm_shuffle_epi32(hgfe, 0x1B);
	__m128i abef = _mm_alignr_epi8(cdab, efgh, 8);
	__m128i cdgh = _mm_blend_epi16(efgh, cdab, 0xF0);

	const __m128i abef_in = abef;
	const __m128i cdgh_in = cdgh;
	__m128i w0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)), kBswap32);
	__m128i w1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16)), kBswap32);
	__m128i w2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32)), kBswap32);
	__m128i w3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 48)), kBswap32);

	SUPERSLM_SHANI_ROUNDS4(0, w0);
	SUPERSLM_SHANI_ROUNDS4(1, w1);
	SUPERSLM_SHANI_ROUNDS4(2, w2);
	SUPERSLM_SHANI_ROUNDS4(3, w3);
	SUPERSLM_SHANI_SCHEDULE(w0, w1, w2, w3); SUPERSLM_SHANI_ROUNDS4(4, w0);
	SUPERSLM_SHANI_SCHEDULE(w1, w2, w3, w0); SUPERSLM_SHANI_ROUNDS4(5, w1);
	SUPERSLM_SHANI_SCHEDULE(w2, w3, w0, w1); SUPERSLM_SHANI_ROUNDS4(6, w2);
	SUPERSLM_SHANI_SCHEDULE(w3, w0, w1, w2); SUPERSLM_SHANI_ROUNDS4(7, w3);
	SUPERSLM_SHANI_SCHEDULE(w0, w1, w2, w3); SUPERSLM_SHANI_ROUNDS4(8, w0);
	SUPERSLM_SHANI_SCHEDULE(w1, w2, w3, w0); SUPERSLM_SHANI_ROUNDS4(9, w1);
	SUPERSLM_SHANI_SCHEDULE(w2, w3, w0, w1); SUPERSLM_SHANI_ROUNDS4(10, w2);
	SUPERSLM_SHANI_SCHEDULE(w3, w0, w1, w2); SUPERSLM_SHANI_ROUNDS4(11, w3);
	SUPERSLM_SHANI_SCHEDULE(w0, w1, w2, w3); SUPERSLM_SHANI_ROUNDS4(12, w0);
	SUPERSLM_SHANI_SCHEDULE(w1, w2, w3, w0); SUPERSLM_SHANI_ROUNDS4(13, w1);
	SUPERSLM_SHANI_SCHEDULE(w2, w3, w0, w1); SUPERSLM_SHANI_ROUNDS4(14, w2);
	SUPERSLM_SHANI_SCHEDULE(w3, w0, w1, w2); SUPERSLM_SHANI_ROUNDS4(15, w3);

	abef = _mm_add_epi32(abef, abef_in);
	cdgh = _mm_add_epi32(cdgh, cdgh_in);

	const __m128i feba = _mm_shuffle_epi32(abef, 0x1B);
	const __m128i dchg = _mm_shuffle_epi32(cdgh, 0xB1);
	dcba = _mm_blend_epi16(feba, dchg, 0xF0);
	hgfe = _mm_alignr_epi8(dchg, feba, 8);
	_mm_storeu_si128(reinterpret_cast<__m128i*>(&h[0]), dcba);
	_mm_storeu_si128(reinterpret_cast<__m128i*>(&h[4]), hgfe);
}

#undef SUPERSLM_SHANI_ROUNDS4
#undef SUPERSLM_SHANI_SCHEDULE

// --- Run-time CPUID probe ------------------------------------------------------------

#if defined(_MSC_VER)
inline void QueryCpuId(int leaf, int subleaf, int regs[4]) { __cpuidex(regs, leaf, subleaf); }
#else
inline void QueryCpuId(int leaf, int subleaf, int regs[4]) {
	unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
	__cpuid_count(static_cast<unsigned int>(leaf), static_cast<unsigned int>(subleaf), eax, ebx,
	              ecx, edx);
	regs[0] = static_cast<int>(eax);
	regs[1] = static_cast<int>(ebx);
	regs[2] = static_cast<int>(ecx);
	regs[3] = static_cast<int>(edx);
}
#endif

#endif  // SUPERSLM_SHA256_HAVE_SHANI_X64

// Indexed by kSha256ImplPortable / kSha256ImplShaNi.
constexpr CompressFn kCompressByImpl[] = {
	&CompressPortable,
#if SUPERSLM_SHA256_HAVE_SHANI_X64
	&CompressShaNi,
#endif
};

inline int ResolveImplFromFields(int max_basic_leaf, int leaf1_ecx, int leaf7_ebx) {
	const bool ssse3 = (leaf1_ecx & (1 << 9)) != 0;    // leaf 1, ECX bit 9
	const bool sse41 = (leaf1_ecx & (1 << 19)) != 0;   // leaf 1, ECX bit 19
	// Leaf 7 is architecturally undefined below max basic leaf 7 (some CPUs echo the
	// highest leaf's registers), so its bits count only when the CPU reports it.
	const bool sha = max_basic_leaf >= 7 && (leaf7_ebx & (1 << 29)) != 0;  // leaf 7/0, EBX bit 29
	return (sha && ssse3 && sse41) ? kSha256ImplShaNi : kSha256ImplPortable;
}

inline int DetectImpl() {
#if SUPERSLM_SHA256_HAVE_SHANI_X64
	int regs0[4] = {0, 0, 0, 0};
	int regs1[4] = {0, 0, 0, 0};
	int regs7[4] = {0, 0, 0, 0};
	QueryCpuId(0, 0, regs0);
	QueryCpuId(1, 0, regs1);
	// Queried unconditionally: CPUID never faults on an unsupported leaf, and the
	// resolver ignores leaf 7's bits when leaf 0 says the CPU does not have it.
	QueryCpuId(7, 0, regs7);
	return ResolveImplFromFields(regs0[0], regs1[2], regs7[1]);
#else
	return kSha256ImplPortable;
#endif
}

// Resolved once per process (C++11 magic static: thread-safe, and every later read
// sees the first write).
inline int ActiveImpl() {
#if defined(SUPERSLM_FORCE_PORTABLE_SHA256)
	return kSha256ImplPortable;
#else
	static const int impl = DetectImpl();
	return impl;
#endif
}

inline CompressFn ActiveCompress() {
	static const CompressFn fn = kCompressByImpl[ActiveImpl()];
	return fn;
}

// Feeds `len` bytes into the running state: tops up and flushes a partial block in
// `buf`, compresses every whole block straight from `data`, and keeps the tail.
void Absorb(CompressFn compress, uint32_t h[8], uint8_t buf[64], size_t& buf_len,
            const uint8_t* data, size_t len) {
	if (buf_len > 0) {
		const size_t take = std::min(64 - buf_len, len);
		std::copy_n(data, take, buf + buf_len);
		buf_len += take;
		data += take;
		len -= take;
		if (buf_len < 64) return;
		compress(h, buf);
		buf_len = 0;
	}
	for (; len >= 64; data += 64, len -= 64) compress(h, data);
	std::copy_n(data, len, buf);
	buf_len = len;
}

// Pads (0x80, zeros to 56 mod 64, the 64-bit big-endian message length in bits),
// compresses the last one or two blocks, and writes the big-endian digest.
void Finish(CompressFn compress, uint32_t h[8], const uint8_t buf[64], size_t buf_len,
            uint64_t total_bits, uint8_t out[32]) {
	// buf_len is always below 64 (Absorb flushes a full block); the mask states that
	// to the compiler, which otherwise cannot bound the index into `tail`.
	const size_t used = buf_len & 63;
	uint8_t tail[128] = {};
	std::copy_n(buf, used, tail);
	tail[used] = 0x80;
	const size_t nblocks = used < 56 ? 1 : 2;
	uint8_t* len_be = tail + nblocks * 64 - 8;
	for (int i = 0; i < 8; ++i) len_be[i] = uint8_t(total_bits >> (56 - i * 8));
	compress(h, tail);
	if (nblocks == 2) compress(h, tail + 64);
	for (int i = 0; i < 8; ++i) {
		out[i * 4] = uint8_t(h[i] >> 24);
		out[i * 4 + 1] = uint8_t(h[i] >> 16);
		out[i * 4 + 2] = uint8_t(h[i] >> 8);
		out[i * 4 + 3] = uint8_t(h[i]);
	}
}

void HashOneShotWith(CompressFn compress, const uint8_t* data, size_t len, uint8_t out[32]) {
	uint32_t h[8];
	std::copy_n(kInit, 8, h);
	uint8_t buf[64];
	size_t buf_len = 0;
	Absorb(compress, h, buf, buf_len, data, len);
	Finish(compress, h, buf, buf_len, uint64_t(len) * 8, out);
}

} // namespace

void Sha256::Reset() {
	std::copy_n(kInit, 8, h_);
	total_bits_ = 0;
	buf_len_ = 0;
}

// S-HARDEN-7 (design Sec3.1): Update's *Impl body needs private access to
// Sha256 (total_bits_, buf_, buf_len_, h_), which a free function cannot
// have. Sha256Access is the sole friend (sha256.h's
// `friend struct Sha256Access;`) -- declared and defined only here, never
// in the header. See artifact.cpp's identical SslmArtifactAccess comment
// for the full reasoning.
struct Sha256Access {
	static void UpdateImpl(Sha256& self, const uint8_t* data, size_t len);
	static void FinalImpl(Sha256& self, uint8_t out[32]);
};

void Sha256::Update(const uint8_t* data, size_t len) {
	internal::WrapBadAllocContract([&] { Sha256Access::UpdateImpl(*this, data, len); });
}

void Sha256Access::UpdateImpl(Sha256& self, const uint8_t* data, size_t len) {
	internal::MaybeThrowInjectedBadAllocFault();
	self.total_bits_ += uint64_t(len) * 8;
	Absorb(ActiveCompress(), self.h_, self.buf_, self.buf_len_, data, len);
}

void Sha256::Final(uint8_t out[32]) {
	// Not wrapped: Final takes no caller-supplied bytes and allocates nothing (the
	// padding is built in a fixed stack block), so there is nothing for the
	// S-HARDEN-7 wrap to narrow.
	Sha256Access::FinalImpl(*this, out);
}

void Sha256Access::FinalImpl(Sha256& self, uint8_t out[32]) {
	Finish(ActiveCompress(), self.h_, self.buf_, self.buf_len_, self.total_bits_, out);
	self.buf_len_ = 0;
}

int ResolveSha256Impl(int max_basic_leaf, int leaf1_ecx, int leaf7_ebx) noexcept {
	// Test-reachable wrapper around the anonymous-namespace pure resolver (see
	// sha256.h), mirroring matmul.cpp's ResolveDotRowTier.
	return ResolveImplFromFields(max_basic_leaf, leaf1_ecx, leaf7_ebx);
}

int DetectSha256ImplForCpu() noexcept {
	static const int impl = DetectImpl();
	return impl;
}

int ActiveSha256Impl() noexcept { return ActiveImpl(); }

void Sha256HashPortableRef(const uint8_t* data, size_t len, uint8_t out[32]) noexcept {
	HashOneShotWith(&CompressPortable, data, len, out);
}

#if SUPERSLM_SHA256_HAVE_SHANI_X64
void Sha256HashShaNiRef(const uint8_t* data, size_t len, uint8_t out[32]) noexcept {
	HashOneShotWith(&CompressShaNi, data, len, out);
}
#endif

namespace {

// S-HARDEN-7: today's bodies, renamed; Sha256Hash/ToHex below wrap these
// with the shared catch-and-rethrow helper. Sha256HashImpl calls the
// public, already-wrapped Sha256::Update rather than reaching into
// Sha256's private UpdateImpl (which a free function has no access to) --
// Sha256HashImpl already runs inside Sha256Hash's own wrap and consults the
// injection seam at its own entry, so the one nested try/catch this costs is
// inert in practice, never exercised (the seam is single-shot and already
// fired, or was never armed).
void Sha256HashImpl(const uint8_t* data, size_t len, uint8_t out[32]) {
	internal::MaybeThrowInjectedBadAllocFault();
	Sha256 h;
	h.Update(data, len);
	h.Final(out);
}

std::string ToHexImpl(const uint8_t digest[32]) {
	internal::MaybeThrowInjectedBadAllocFault();
	static const char* kHex = "0123456789abcdef";
	std::string s;
	s.resize(64);
	for (int i = 0; i < 32; ++i) {
		s[i * 2] = kHex[digest[i] >> 4];
		s[i * 2 + 1] = kHex[digest[i] & 0xF];
	}
	return s;
}

}  // namespace

void Sha256Hash(const uint8_t* data, size_t len, uint8_t out[32]) {
	internal::WrapBadAllocContract([&] { Sha256HashImpl(data, len, out); });
}

std::string ToHex(const uint8_t digest[32]) {
	return internal::WrapBadAllocContract([&] { return ToHexImpl(digest); });
}

} // namespace superslm
