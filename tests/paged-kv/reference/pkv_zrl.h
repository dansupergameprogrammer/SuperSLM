// Paged-KV plan §8 (step R0): the zero-run encoding of a pinned blob.
//
// A legacy blob carries the whole K/V block, and on the pkv fixtures most of it is the zeroed
// positions past the context length, so the pinned blobs are committed zero-run encoded:
// "PZR1", the decoded size as LE64, then pairs of (literal length LE32, literal bytes, zero-run
// length LE32) until the decoded size is reached. Lossless; the cells decode before use.

#ifndef SUPERSLM_PKV_ZRL_H
#define SUPERSLM_PKV_ZRL_H

#include <cstdint>
#include <cstring>
#include <vector>

namespace pkv {

inline void ZrlPut32(std::vector<uint8_t>& o, uint32_t v) {
	for (int i = 0; i < 4; ++i) o.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

inline std::vector<uint8_t> ZrlEncode(const std::vector<uint8_t>& in) {
	std::vector<uint8_t> o = {'P', 'Z', 'R', '1'};
	for (int i = 0; i < 8; ++i) o.push_back(static_cast<uint8_t>(static_cast<uint64_t>(in.size()) >> (8 * i)));
	size_t at = 0;
	while (at < in.size()) {
		// A literal runs until a zero run of at least 16 bytes (or the end).
		size_t lit_end = at;
		while (lit_end < in.size()) {
			if (in[lit_end] == 0) {
				size_t z = lit_end;
				while (z < in.size() && in[z] == 0 && z - lit_end < 16) ++z;
				if (z - lit_end >= 16 || z == in.size()) break;
				lit_end = z;
			} else {
				++lit_end;
			}
		}
		size_t zero_end = lit_end;
		while (zero_end < in.size() && in[zero_end] == 0 && zero_end - lit_end < 0xFFFFFFFFu) ++zero_end;
		ZrlPut32(o, static_cast<uint32_t>(lit_end - at));
		o.insert(o.end(), in.begin() + static_cast<std::ptrdiff_t>(at), in.begin() + static_cast<std::ptrdiff_t>(lit_end));
		ZrlPut32(o, static_cast<uint32_t>(zero_end - lit_end));
		at = zero_end;
	}
	return o;
}

// Returns false on a malformed encoding.
inline bool ZrlDecode(const std::vector<uint8_t>& in, std::vector<uint8_t>* out) {
	if (in.size() < 12 || std::memcmp(in.data(), "PZR1", 4) != 0) return false;
	uint64_t size = 0;
	for (int i = 0; i < 8; ++i) size |= static_cast<uint64_t>(in[4 + i]) << (8 * i);
	out->clear();
	out->reserve(static_cast<size_t>(size));
	size_t at = 12;
	auto get32 = [&](uint32_t* v) {
		if (at + 4 > in.size()) return false;
		*v = static_cast<uint32_t>(in[at]) | static_cast<uint32_t>(in[at + 1]) << 8 |
		     static_cast<uint32_t>(in[at + 2]) << 16 | static_cast<uint32_t>(in[at + 3]) << 24;
		at += 4;
		return true;
	};
	while (out->size() < size) {
		uint32_t lit = 0, zeros = 0;
		if (!get32(&lit) || at + lit > in.size()) return false;
		out->insert(out->end(), in.begin() + static_cast<std::ptrdiff_t>(at), in.begin() + static_cast<std::ptrdiff_t>(at + lit));
		at += lit;
		if (!get32(&zeros)) return false;
		out->insert(out->end(), zeros, 0);
	}
	return out->size() == size && at == in.size();
}

}  // namespace pkv

#endif  // SUPERSLM_PKV_ZRL_H
