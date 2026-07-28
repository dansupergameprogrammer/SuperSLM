// Curie's S2.0b red suite — spec-faithful CFG1 config-blob builder
// (SuperSLM_Plan.md S2.0b; docs/sslm_format.md "Config blob — CFG1"). `ParseConfig`
// parses the `Config` section's self-contained fixed 84-byte `CFG1` struct AFTER
// `SslmArtifact::OpenFromMemory` has verified whole-file structure and integrity —
// like WGT1/BIA1/ROP1 (sslm_model_hostile_fixtures.h) and KVC1
// (sslm_kvc1_hostile_fixtures.h) before it, a crafted integrity-valid artifact can
// still carry a malformed Config section, so this sub-parse is its own
// hostile-input trust boundary AND the §11 reject-over-degrade gate: a zero
// dimension or a defaulted field must be rejected, not repaired ("a model that
// loads, runs, and is not the source model").
//
// `BuildCfg1` is written directly against the spec's field table, independent of
// `src/model.cpp` (the code under test) — derived from the spec, never from
// reading the parser's own implementation. A rejection cell takes
// `MakeMinimalValidCfg1()` and mutates exactly one field via
// `Cfg1*Off`/`PutU32`/`PutU64` (from sslm_fixtures.h), so the byte buffer differs
// from a valid CFG1 blob in exactly the one way the cell names.
//
// CFG1 is a single FIXED-layout struct — unlike WGT1/BIA1/ROP1 and KVC1 (which
// carry variable-length descriptor/name/value regions), every field lives at a
// constant byte offset with no size-dependent arithmetic, so there is no
// "*OutOfBounds" truncation-region class here: the whole-struct size check
// (`BadConfigSize`, byte_size != kConfigBytes exactly) is the entire bounds
// surface.
#ifndef SUPERSLM_TESTS_SSLM_CFG1_HOSTILE_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_CFG1_HOSTILE_FIXTURES_H

#include "sslm_fixtures.h"
#include "sslm_model_hostile_fixtures.h"  // WriteU32LE/WriteU64LE/WriteBytes
#include "superslm/artifact.h"
#include "superslm/model.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace superslm_test {

// --- Fixed-formula byte offsets from docs/sslm_format.md's CFG1 field table —
//     not derived from the parser under test. Every field is a constant offset;
//     there is no per-entry stride formula (CFG1 has exactly one "entry"). ---

inline constexpr size_t kCfg1MagicOff = 0;
inline constexpr size_t kCfg1VersionOff = 4;
inline constexpr size_t kCfg1HiddenSizeOff = 8;
inline constexpr size_t kCfg1NumHiddenLayersOff = 12;
inline constexpr size_t kCfg1NumAttentionHeadsOff = 16;
inline constexpr size_t kCfg1NumKeyValueHeadsOff = 20;
inline constexpr size_t kCfg1HeadDimOff = 24;
inline constexpr size_t kCfg1IntermediateSizeOff = 28;
inline constexpr size_t kCfg1VocabSizeOff = 32;
inline constexpr size_t kCfg1ContextCapOff = 36;
inline constexpr size_t kCfg1TieWordEmbeddingsOff = 40;
inline constexpr size_t kCfg1KvPrecisionOff = 44;
inline constexpr size_t kCfg1KvBlockSizeOff = 48;
inline constexpr size_t kCfg1UnicodeMajorOff = 52;
inline constexpr size_t kCfg1UnicodeMinorOff = 56;
inline constexpr size_t kCfg1UnicodePatchOff = 60;
inline constexpr size_t kCfg1ReservedOff = 64;
inline constexpr size_t kCfg1RopeThetaOff = 68;
inline constexpr size_t kCfg1RmsNormEpsOff = 76;

// Append-style little-endian f64 writer (bit-pattern memcpy, then written byte
// by byte like WriteU64LE — never a struct cast over untrusted bytes, matching
// docs/sslm_format.md choice 1). x86/ARM hosts are both little-endian and
// IEEE-754, so memcpy'ing the bit pattern and emitting it byte-by-byte via
// WriteU64LE produces the same bytes ParseConfig's little-endian f64 reader
// must reconstruct.
inline void WriteF64LE(std::vector<uint8_t>& b, double v) {
	uint64_t bits;
	std::memcpy(&bits, &v, sizeof(bits));
	WriteU64LE(b, bits);
}

// Reads a little-endian f64 back out of a byte buffer at a fixed offset, for
// oracle assertions that want to state the expected double directly rather than
// via GetU64+memcpy at the call site.
inline double GetF64LE(const std::vector<uint8_t>& b, size_t off) {
	uint64_t bits = GetU64(b, off);
	double v;
	std::memcpy(&v, &bits, sizeof(v));
	return v;
}

// Every CFG1 field, spec-faithful defaults chosen to be pairwise distinct (no
// two fields share a baked-in value) so the feature oracle's field-by-field
// assertions catch an offset swap, not just a zero/nonzero check. `rope_theta`
// and `rms_norm_eps` are distinctive non-round doubles (docs/sslm_format.md's
// commission note) to pin the little-endian f64 read.
struct Cfg1Spec {
	uint32_t hidden_size = 4096;
	uint32_t num_hidden_layers = 32;
	uint32_t num_attention_heads = 24;
	uint32_t num_key_value_heads = 8;
	uint32_t head_dim = 128;
	uint32_t intermediate_size = 11008;
	uint32_t vocab_size = 32001;
	uint32_t context_cap = 8192;
	uint32_t tie_word_embeddings = 1;  // true — pin the true case, per commission
	uint32_t kv_precision = 1;         // Int16 — pin the enum mapping, not just default 0
	uint32_t kv_block_size = 256;
	uint32_t unicode_major = 15;
	uint32_t unicode_minor = 1;
	uint32_t unicode_patch = 2;
	uint32_t reserved = 0;
	double rope_theta = 1000000.0;
	double rms_norm_eps = 1e-6;
};

// Builds a spec-faithful 84-byte CFG1 blob (docs/sslm_format.md "Config blob —
// CFG1"): magic, version, the eight dimension fields, the bool/enum fields, the
// three unicode fields, reserved, then the two f64s — written field by field in
// the spec's exact offset order.
inline std::vector<uint8_t> BuildCfg1(const Cfg1Spec& s = Cfg1Spec{}) {
	std::vector<uint8_t> b;
	b.insert(b.end(), superslm::kConfigMagic, superslm::kConfigMagic + 4);
	WriteU32LE(b, superslm::kManifestVersion);  // CFG1 version field, == 1, shared with WGT1/BIA1/ROP1/KVC1
	WriteU32LE(b, s.hidden_size);
	WriteU32LE(b, s.num_hidden_layers);
	WriteU32LE(b, s.num_attention_heads);
	WriteU32LE(b, s.num_key_value_heads);
	WriteU32LE(b, s.head_dim);
	WriteU32LE(b, s.intermediate_size);
	WriteU32LE(b, s.vocab_size);
	WriteU32LE(b, s.context_cap);
	WriteU32LE(b, s.tie_word_embeddings);
	WriteU32LE(b, s.kv_precision);
	WriteU32LE(b, s.kv_block_size);
	WriteU32LE(b, s.unicode_major);
	WriteU32LE(b, s.unicode_minor);
	WriteU32LE(b, s.unicode_patch);
	WriteU32LE(b, s.reserved);
	WriteF64LE(b, s.rope_theta);
	WriteF64LE(b, s.rms_norm_eps);
	return b;
}

// The minimal valid CFG1 blob every hostile cell mutates from, and the feature
// oracle's own fixture.
inline std::vector<uint8_t> MakeMinimalValidCfg1() { return BuildCfg1(Cfg1Spec{}); }

// An outer artifact-level Config section carrying a spec-faithful CFG1 blob —
// what a fixture routed through SslmModel::Load needs (S-HARDEN-1, D-SLM141:
// Load sub-parses every present Config section via ParseConfig, so the
// simpler `{'{','}'}` placeholder MakeConfigSection() in sslm_fixtures.h,
// which only ever satisfied the outer loader's structural check, is not
// enough once a fixture is driven through Load rather than OpenFromMemory
// alone).
inline FixtureSection MakeValidConfigSection() {
	return MakeSection(superslm::SslmSectionType::Config, superslm::SslmDtype::Raw, MakeMinimalValidCfg1());
}

// A validated SslmSectionView directly over `bytes`, of section type Config —
// ParseConfig's actual input. No outer SslmArtifact is built: the CFG1 sub-parse
// never reads back through the artifact once it holds a validated section (the
// S0 suite already covers the outer loader's own hostile-input sweep). dtype is
// always Raw (docs/sslm_format.md: "Config" section carries dtype Raw).
inline superslm::SslmSectionView MakeConfigSectionView(const std::vector<uint8_t>& bytes) {
	superslm::SslmSectionView v;
	v.type = superslm::SslmSectionType::Config;
	v.dtype = superslm::SslmDtype::Raw;
	v.data = bytes.data();
	v.byte_size = bytes.size();
	v.elem_count = bytes.size();
	v.alignment = 64;
	return v;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_CFG1_HOSTILE_FIXTURES_H
