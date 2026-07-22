// Curie's S2.4 red suite — spec-faithful SIL1 sigmoid-LUT blob builder
// (SuperSLM_S2.4_SiLU_LUT_Design §4, §8; docs/sslm_format.md "Sigmoid-LUT blob
// — SIL1"). `ParseSigmoidLut` parses the `SigmoidLut` section's self-contained
// fixed 4116-byte `SIL1` struct (16-byte header + 1025 little-endian int32 Q15
// nodes) AFTER `SslmArtifact::OpenFromMemory` has verified whole-file structure
// and integrity — like CFG1 (sslm_cfg1_hostile_fixtures.h) before it, a
// crafted integrity-valid artifact can still carry a malformed SIL1 section, so
// this sub-parse is its own hostile-input trust boundary AND the §11
// reject-over-degrade gate: a wrong size/magic/version/count/reserved must be
// rejected, never repaired.
//
// `BuildSil1` is written directly against the spec's field table, independent
// of `src/model.cpp` (the code under test — currently a red-first stub that
// leaves `out` at `SslmSigmoidLut{}` defaults and reports Ok unconditionally).
// A rejection cell takes `MakeMinimalValidSil1()` and mutates exactly one field
// via `Sil1*Off`/`PutU32` (from sslm_fixtures.h), so the byte buffer differs
// from a valid SIL1 blob in exactly the one way the cell names.
//
// `BuildReferenceSigmoidLutQ15` is an INDEPENDENT from-scratch computation of
// the offline table (SuperSLM_S2.4_SiLU_LUT_Design §4: entry i holds
// round_half_even(sigmoid(-X + i*(2X/N)) * 2^15)), using <cmath> double
// precision and the default (round-to-nearest-even) floating-point environment
// — NOT a read of `tools/sslm_model_writer.py::build_sigmoid_lut`'s output and
// NOT a call into any code under test. This is the reference/exact-value oracle
// every SIL1 and SiluSigmoidQ15 cell below checks against.
//
// SIL1 is a single FIXED-layout struct (like CFG1) — every field lives at a
// constant byte offset, no size-dependent arithmetic — so BadSigmoidLutSize
// (byte_size != kSigmoidLutBytes exactly) is the entire bounds surface; there
// is no "*OutOfBounds" truncation-region class here.
#ifndef SUPERSLM_TESTS_SSLM_SIL1_HOSTILE_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_SIL1_HOSTILE_FIXTURES_H

#include "sslm_fixtures.h"
#include "sslm_model_hostile_fixtures.h"  // WriteU32LE
#include "silu_lut_golden_table.h"  // kSiluLutGoldenTable — the pinned production-identical table
#include "superslm/artifact.h"
#include "superslm/model.h"
#include "superslm/silu_lut.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace superslm_test {

// --- Fixed-formula byte offsets from docs/sslm_format.md's SIL1 field table —
//     not derived from the parser under test. ---

inline constexpr size_t kSil1MagicOff = 0;
inline constexpr size_t kSil1VersionOff = 4;
inline constexpr size_t kSil1EntryCountOff = 8;
inline constexpr size_t kSil1ReservedOff = 12;
inline constexpr size_t kSil1NodesOff = 16;

// The independent reference table: N+1 entries, round_half_even(sigmoid(x)*2^15)
// in double precision (SuperSLM_S2.4_SiLU_LUT_Design §4). Uses the default
// (round-to-nearest, ties-to-even) floating-point environment, which is what
// `std::nearbyint`/`std::rint` apply and what NumPy's `np.rint` implements —
// ties are not exercised by this construction in practice (sigmoid(x)*2^15 is
// essentially never exactly a half-integer over a continuous domain), so the
// tie rule only needs to be a defensible, standard one, not independently
// reproduced bit-for-bit against a second implementation.
inline std::vector<int32_t> BuildReferenceSigmoidLutQ15() {
	std::vector<int32_t> table(static_cast<size_t>(superslm::kSiluLutN) + 1);
	const double X = static_cast<double>(superslm::kSiluLutX);
	const double N = static_cast<double>(superslm::kSiluLutN);
	for (int i = 0; i <= superslm::kSiluLutN; ++i) {
		const double x = -X + static_cast<double>(i) * (2.0 * X / N);
		const double sig = 1.0 / (1.0 + std::exp(-x));
		const double q15 = std::nearbyint(sig * 32768.0);  // round-half-to-even
		table[static_cast<size_t>(i)] = static_cast<int32_t>(q15);
	}
	return table;
}

// Builds a spec-faithful 4116-byte SIL1 blob from an explicit table (defaults
// to the reference table above): magic, version, entry_count, reserved, then
// kSigmoidLutEntries little-endian int32 nodes, in spec field order.
inline std::vector<uint8_t> BuildSil1(const std::vector<int32_t>& table = BuildReferenceSigmoidLutQ15(),
                                       uint32_t version = superslm::kManifestVersion,
                                       uint32_t entry_count = superslm::kSigmoidLutEntries,
                                       uint32_t reserved = 0) {
	std::vector<uint8_t> b;
	b.insert(b.end(), superslm::kSigmoidLutMagic, superslm::kSigmoidLutMagic + 4);
	WriteU32LE(b, version);
	WriteU32LE(b, entry_count);
	WriteU32LE(b, reserved);
	for (int32_t v : table) WriteU32LE(b, static_cast<uint32_t>(v));
	return b;
}

// The minimal valid SIL1 blob every hostile cell mutates from, and the feature
// oracle's own fixture: the spec-faithful reference table, correct header.
//
// S-HARDEN-1 (F20): defaults to the PINNED production table
// (silu_lut_golden_table.h's kSiluLutGoldenTable — generated by
// tools/gen_silu_lut_golden.py from the SAME run that produces
// include/superslm/silu_lut_canonical.h, so the two are byte-identical by
// construction) rather than `BuildReferenceSigmoidLutQ15()`'s from-scratch
// std::exp computation. ParseSigmoidLut now rejects any SIL1 payload that is
// not byte-for-byte the compiled-in canonical table (the universal-construction
// content-pinning check), and std::exp is not guaranteed correctly-rounded
// identically across libm implementations (the same host-dependency class F8
// names) — so a fixture meant to parse Ok must carry the literal pinned bytes,
// never a recomputation, even one expected to agree in practice.
// BuildReferenceSigmoidLutQ15 remains the independent oracle the structural
// tests below check parsed VALUES against (a from-scratch derivation, not a
// read of the code under test) — the pinned table and the from-scratch
// computation are expected to agree bit-for-bit (both round-half-to-even over
// the same formula), so using the pinned bytes as content does not weaken that
// check; it only removes host-libm variance from whether the fixture ITSELF
// parses Ok.
inline std::vector<uint8_t> MakeMinimalValidSil1() {
	return BuildSil1(std::vector<int32_t>(std::begin(superslm_test::kSiluLutGoldenTable),
	                                       std::end(superslm_test::kSiluLutGoldenTable)));
}

// An outer artifact-level SigmoidLut section carrying the pinned canonical
// table — what every fixture-level "Accepts"/"valid artifact" test now needs
// to satisfy the version-indexed required-section schema (S-HARDEN-1, F1:
// SigmoidLut required from v2).
inline FixtureSection MakeSigmoidLutSection() {
	return MakeSection(superslm::SslmSectionType::SigmoidLut, superslm::SslmDtype::Int32, MakeMinimalValidSil1());
}

// A validated SslmSectionView directly over `bytes`, of section type
// SigmoidLut — ParseSigmoidLut's actual input. No outer SslmArtifact is built:
// the SIL1 sub-parse never reads back through the artifact once it holds a
// validated section (the S0 suite already covers the outer loader's own
// hostile-input sweep). dtype is always Int32 (docs/sslm_format.md: "SigmoidLut"
// section carries dtype Int32).
inline superslm::SslmSectionView MakeSigmoidLutSectionView(const std::vector<uint8_t>& bytes) {
	superslm::SslmSectionView v;
	v.type = superslm::SslmSectionType::SigmoidLut;
	v.dtype = superslm::SslmDtype::Int32;
	v.data = bytes.data();
	v.byte_size = bytes.size();
	v.elem_count = superslm::kSigmoidLutEntries;
	v.alignment = 64;
	return v;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_SIL1_HOSTILE_FIXTURES_H
