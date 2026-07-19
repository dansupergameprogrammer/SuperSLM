// Minimal SHA-256 (FIPS 180-4), standard library only. Used for the artifact
// integrity hash / fingerprint (docs/sslm_format.md). Layer 1 carries no
// third-party dependency (DecisionLog D-SLM13), so the hash is in-tree.
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
	void Update(const uint8_t* data, size_t len);
	// Writes 32 bytes to out[0..31]. The object is single-use after Final.
	void Final(uint8_t out[32]);

private:
	void Block(const uint8_t* p);
	uint32_t h_[8];
	uint64_t total_bits_;
	uint8_t buf_[64];
	size_t buf_len_;
};

// One-shot: SHA-256 of a buffer into out[0..31].
void Sha256Hash(const uint8_t* data, size_t len, uint8_t out[32]);

// Lowercase hex of a 32-byte digest.
std::string ToHex(const uint8_t digest[32]);

} // namespace superslm

#endif // SUPERSLM_SHA256_H
