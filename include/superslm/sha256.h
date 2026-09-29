// Minimal SHA-256 (FIPS 180-4), standard library only. Used for the artifact
// integrity hash / fingerprint (docs/sslm_format.md). Layer 1 carries no
// third-party dependency, so the hash implementation is in-tree.
#ifndef SUPERSLM_SHA256_H
#define SUPERSLM_SHA256_H
#include "superslm/api.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace superslm {

// Streaming SHA-256. Feed bytes with Update; call Final once to get the 32-byte
// digest. Deterministic and host-independent.
class Sha256 {
public:
	Sha256() { Reset(); }
	SUPERSLM_API void Reset();
	// Throws only std::bad_alloc (S-HARDEN-7, F5).
	SUPERSLM_API void Update(const uint8_t* data, size_t len);
	// Writes 32 bytes to out[0..31]. The object is single-use after Final.
	SUPERSLM_API void Final(uint8_t out[32]);

private:
	// S-HARDEN-7 (design Sec3.1): grants src/sha256.cpp's Sha256Access
	// (defined only there) access to the private members below, so Update's
	// *Impl body can live entirely in the .cpp rather than as a private
	// member declaration here. See artifact.h's identical
	// SslmArtifactAccess comment for the full reasoning.
	friend struct Sha256Access;

	uint32_t h_[8];
	uint64_t total_bits_;
	uint8_t buf_[64];
	size_t buf_len_;
};

// One-shot: SHA-256 of a buffer into out[0..31]. Throws only std::bad_alloc
// (S-HARDEN-7, F5).
SUPERSLM_API void Sha256Hash(const uint8_t* data, size_t len, uint8_t out[32]);

// Lowercase hex of a 32-byte digest. Throws only std::bad_alloc (S-HARDEN-7,
// F5).
std::string ToHex(const uint8_t digest[32]);

// --- Block compression dispatch (verification seams) --------------------------
//
// src/sha256.cpp compresses blocks with the x86 SHA extensions (sha256rnds2,
// sha256msg1, sha256msg2) when the CPU reports them at run time, and with the
// portable FIPS 180-4 code otherwise. Both produce the same digest for the same
// bytes; the choice changes only speed. The portable path is the only one on a
// non-x64 target, and SUPERSLM_FORCE_PORTABLE_SHA256 (defined when compiling
// src/sha256.cpp) pins it on x64 too, the way SUPERSLM_FORCE_SCALAR_MATMUL pins
// matmul.cpp's scalar reference.
//
// The declarations below exist for verification, mirroring matmul.h's
// DotRowScalarRef/ResolveDotRowTier pattern: a test can drive each path directly
// and compare them, and can drive the CPUID decision with fabricated register
// values. They are noexcept: none of them allocates.

// Compile-time capability: the SHA-extension path exists in this build (the same
// target condition as matmul.h's SUPERSLM_MATMUL_HAVE_SIMD_X64).
#if defined(_M_X64) || defined(__x86_64__)
#define SUPERSLM_SHA256_HAVE_SHANI_X64 1
#else
#define SUPERSLM_SHA256_HAVE_SHANI_X64 0
#endif

inline constexpr int kSha256ImplPortable = 0;
inline constexpr int kSha256ImplShaNi = 1;

// Pure decision over CPUID fields: leaf 0 EAX (highest basic leaf), leaf 1 ECX
// (SSSE3 bit 9, SSE4.1 bit 19), leaf 7 sub-leaf 0 EBX (SHA bit 29; ignored when the
// highest basic leaf is below 7). Returns kSha256ImplShaNi only when all three
// feature bits are set, else kSha256ImplPortable.
int ResolveSha256Impl(int max_basic_leaf, int leaf1_ecx, int leaf7_ebx) noexcept;

// What this CPU supports: the resolver above applied to the real CPUID fields
// (always kSha256ImplPortable on a non-x64 build). Ignores the force macro.
int DetectSha256ImplForCpu() noexcept;

// What Sha256 and Sha256Hash actually use in this build on this CPU: the detected
// implementation, or kSha256ImplPortable under SUPERSLM_FORCE_PORTABLE_SHA256.
int ActiveSha256Impl() noexcept;

// One-shot SHA-256 through the portable compression, whatever the dispatch selects.
void Sha256HashPortableRef(const uint8_t* data, size_t len, uint8_t out[32]) noexcept;

#if SUPERSLM_SHA256_HAVE_SHANI_X64
// One-shot SHA-256 through the SHA-extension compression, whatever the dispatch
// selects. Precondition: DetectSha256ImplForCpu() == kSha256ImplShaNi (on any other
// CPU the instructions fault).
void Sha256HashShaNiRef(const uint8_t* data, size_t len, uint8_t out[32]) noexcept;
#endif

} // namespace superslm

#endif // SUPERSLM_SHA256_H
