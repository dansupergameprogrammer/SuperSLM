#include "superslm/sha256.h"

#include "bad_alloc_wrap.h"

namespace superslm {
namespace {

inline uint32_t Ror(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

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

} // namespace

void Sha256::Reset() {
	h_[0] = 0x6a09e667; h_[1] = 0xbb67ae85; h_[2] = 0x3c6ef372; h_[3] = 0xa54ff53a;
	h_[4] = 0x510e527f; h_[5] = 0x9b05688c; h_[6] = 0x1f83d9ab; h_[7] = 0x5be0cd19;
	total_bits_ = 0;
	buf_len_ = 0;
}

void Sha256::Block(const uint8_t* p) {
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

// S-HARDEN-7 (design Sec3.1): Update's *Impl body needs private access to
// Sha256 (total_bits_, buf_, buf_len_, Block), which a free function cannot
// have. Sha256Access is the sole friend (sha256.h's
// `friend struct Sha256Access;`) -- declared and defined only here, never
// in the header. See artifact.cpp's identical SslmArtifactAccess comment
// for the full reasoning.
struct Sha256Access {
	static void UpdateImpl(Sha256& self, const uint8_t* data, size_t len);
};

void Sha256::Update(const uint8_t* data, size_t len) {
	internal::WrapBadAllocContract([&] { Sha256Access::UpdateImpl(*this, data, len); });
}

void Sha256Access::UpdateImpl(Sha256& self, const uint8_t* data, size_t len) {
	internal::MaybeThrowInjectedBadAllocFault();
	self.total_bits_ += uint64_t(len) * 8;
	while (len > 0) {
		size_t take = 64 - self.buf_len_;
		if (take > len) take = len;
		for (size_t i = 0; i < take; ++i) self.buf_[self.buf_len_ + i] = data[i];
		self.buf_len_ += take;
		data += take;
		len -= take;
		if (self.buf_len_ == 64) {
			self.Block(self.buf_);
			self.buf_len_ = 0;
		}
	}
}

void Sha256::Final(uint8_t out[32]) {
	// Append 0x80, pad with zeros to 56 mod 64, then the 64-bit big-endian length.
	// Calls Sha256Access::UpdateImpl directly, not the public wrapped Update
	// -- these bytes are fixed, already-trusted padding, never caller-supplied
	// artifact bytes, so routing them through the wrap's try/catch on every one of up
	// to 63 padding bytes per digest would add overhead for no benefit
	// (S-HARDEN-7 design Sec3.1).
	uint8_t pad = 0x80;
	uint64_t bits = total_bits_;
	Sha256Access::UpdateImpl(*this, &pad, 1);
	uint8_t zero = 0;
	while (buf_len_ != 56) Sha256Access::UpdateImpl(*this, &zero, 1);
	uint8_t lenbe[8];
	for (int i = 0; i < 8; ++i) lenbe[i] = uint8_t(bits >> (56 - i * 8));
	// Update() would re-add these 8 bytes to total_bits_; feed them through Block
	// directly instead so the length encodes the message, not the padding.
	for (int i = 0; i < 8; ++i) buf_[buf_len_ + i] = lenbe[i];
	buf_len_ = 64;
	Block(buf_);
	buf_len_ = 0;
	for (int i = 0; i < 8; ++i) {
		out[i * 4] = uint8_t(h_[i] >> 24);
		out[i * 4 + 1] = uint8_t(h_[i] >> 16);
		out[i * 4 + 2] = uint8_t(h_[i] >> 8);
		out[i * 4 + 3] = uint8_t(h_[i]);
	}
}

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
