// Curie's S2.4 red suite — loader for the real-activation SiLU-LUT vectors
// committed at `tests/fixtures/silu_lut_real_vectors.bin`.
//
// The fixture is the EXACT downstream-agreement subset the C10 solve measured
// (`Claude/Laplace/harness/silu_lut/experiment.py`'s "Q2" selection — the first
// 8 tokens of every layer, in original row order, 28 layers x 8 = 224 rows x
// 8960 elements), extracted from that same harness's `real_rows.npz` by a
// one-off script (not committed; the harness script + this file's format
// together are sufficient to regenerate it). Reusing the identical subset the
// Laplace result packet (`result_packet.json`, `downstream` key) measured lets
// the int8-agreement cell assert an EXACT reproduction of that measurement —
// 319/2007040 codes differ (0.0159%), max |code delta| = 1 — rather than a
// fuzzy band, since the same subset run through the same C19-C22 requant
// primitives (already shipped, S2.1) on the same construction (SIL1 N=1024
// X=16 Q12 index) is deterministic.
//
// Binary format ('SLV1', not a hostile-input trust boundary — this is
// committed test scaffolding read by trusted test code, not artifact-loader
// input, so the parser below fails closed on a mismatch but is not held to the
// T-129 bar):
//   'SLV1', u32 version(=1), u32 row_count, u32 width, u32 reserved(=0),
//   int64[row_count]  m        (per-row canonical scale mantissa, LE),
//   int32[row_count]  e        (per-row canonical scale exponent, LE),
//   int8[row_count*width]  g_codes  (row-major, the SwiGLU gate codes),
//   int8[row_count*width]  u_codes  (row-major, the SwiGLU up-projection codes).
#ifndef SUPERSLM_TESTS_SSLM_SILU_LUT_REAL_VECTORS_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_SILU_LUT_REAL_VECTORS_FIXTURES_H

#include "sslm_tokenizer_fixtures.h"  // ResolveFixturePath, ReadU32LE

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace superslm_test {

struct SiluLutRealVectors {
	bool ok = false;
	std::string error;
	uint32_t row_count = 0;
	uint32_t width = 0;
	std::vector<int64_t> m;         // [row_count]
	std::vector<int32_t> e;         // [row_count]
	std::vector<int8_t> g_codes;    // [row_count * width], row-major
	std::vector<int8_t> u_codes;    // [row_count * width], row-major
};

inline int64_t ReadI64LE(const std::vector<uint8_t>& b, size_t off) {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[off + static_cast<size_t>(i)]) << (8 * i);
	return static_cast<int64_t>(v);
}

inline SiluLutRealVectors LoadSiluLutRealVectors(const std::string& path) {
	SiluLutRealVectors out;
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		out.error = "cannot open silu-lut real-vectors fixture: " + path;
		return out;
	}
	std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	auto need = [&](size_t off, size_t len) { return off + len <= data.size(); };

	if (!need(0, 20)) {
		out.error = "silu-lut real-vectors fixture truncated before header: " + path;
		return out;
	}
	if (std::memcmp(data.data(), "SLV1", 4) != 0) {
		out.error = "silu-lut real-vectors fixture bad magic (want SLV1): " + path;
		return out;
	}
	const uint32_t version = ReadU32LE(data, 4);
	if (version != 1) {
		out.error = "silu-lut real-vectors fixture unsupported version: " + path;
		return out;
	}
	out.row_count = ReadU32LE(data, 8);
	out.width = ReadU32LE(data, 12);
	// data[16..20) is reserved — not asserted here.

	size_t pos = 20;
	if (!need(pos, static_cast<size_t>(out.row_count) * 8)) {
		out.error = "silu-lut real-vectors fixture truncated in m[]: " + path;
		return out;
	}
	out.m.resize(out.row_count);
	for (uint32_t r = 0; r < out.row_count; ++r) {
		out.m[r] = ReadI64LE(data, pos);
		pos += 8;
	}

	if (!need(pos, static_cast<size_t>(out.row_count) * 4)) {
		out.error = "silu-lut real-vectors fixture truncated in e[]: " + path;
		return out;
	}
	out.e.resize(out.row_count);
	for (uint32_t r = 0; r < out.row_count; ++r) {
		out.e[r] = static_cast<int32_t>(ReadU32LE(data, pos));
		pos += 4;
	}

	const size_t elems = static_cast<size_t>(out.row_count) * static_cast<size_t>(out.width);
	if (!need(pos, elems)) {
		out.error = "silu-lut real-vectors fixture truncated in g_codes: " + path;
		return out;
	}
	out.g_codes.resize(elems);
	for (size_t i = 0; i < elems; ++i) out.g_codes[i] = static_cast<int8_t>(data[pos + i]);
	pos += elems;

	if (!need(pos, elems)) {
		out.error = "silu-lut real-vectors fixture truncated in u_codes: " + path;
		return out;
	}
	out.u_codes.resize(elems);
	for (size_t i = 0; i < elems; ++i) out.u_codes[i] = static_cast<int8_t>(data[pos + i]);

	out.ok = true;
	return out;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_SILU_LUT_REAL_VECTORS_FIXTURES_H
