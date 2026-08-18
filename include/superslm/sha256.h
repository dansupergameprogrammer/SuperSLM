// Minimal SHA-256 (FIPS 180-4), standard library only. Used for the artifact
// integrity hash / fingerprint (docs/sslm_format.md). Layer 1 carries no
// third-party dependency, so the hash implementation is in-tree.
#ifndef SUPERSLM_SHA256_H
#define SUPERSLM_SHA256_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace superslm {

// Streaming SHA-256. Feed bytes with Update; call Final once to get the 32-byte
// digest. Deterministic and host-independent.
class Sha256 {
public:
	Sha256() { Reset(); }
	void Reset();
	// Throws only std::bad_alloc (S-HARDEN-7, F5).
	void Update(const uint8_t* data, size_t len);
	// Writes 32 bytes to out[0..31]. The object is single-use after Final.
	void Final(uint8_t out[32]);

private:
	// S-HARDEN-7 (design Sec3.1): grants src/sha256.cpp's Sha256Access
	// (defined only there) access to the private members below, so Update's
	// *Impl body can live entirely in the .cpp rather than as a private
	// member declaration here. See artifact.h's identical
	// SslmArtifactAccess comment for the full reasoning.
	friend struct Sha256Access;

	void Block(const uint8_t* p);
	uint32_t h_[8];
	uint64_t total_bits_;
	uint8_t buf_[64];
	size_t buf_len_;
};

// One-shot: SHA-256 of a buffer into out[0..31]. Throws only std::bad_alloc
// (S-HARDEN-7, F5).
void Sha256Hash(const uint8_t* data, size_t len, uint8_t out[32]);

// Lowercase hex of a 32-byte digest. Throws only std::bad_alloc (S-HARDEN-7,
// F5).
std::string ToHex(const uint8_t digest[32]);

} // namespace superslm

#endif // SUPERSLM_SHA256_H
