// The 'SPX1' precached-prefix blob: the one writer and the one reader both backends use.
//
// Layout (little-endian), 184-byte header, then the residual (hidden_size bytes), then rows [0, L) of every
// run in canonical order [layer][K|V][kv_head][pos < L][d]:
//
//   0 magic 'SPX1'         4 backend (0 CPU, 1 GPU)   8/12/16 engine major/minor/patch
//   20 adapter vendor      24 adapter device           28 digest (32 bytes, SHA-256 of the blob with it zeroed)
//   60 model hash (32)     92 kv_precision             96 schema name hash (0 = unbound)
//   104 DFA walk state     108 current token           112 length L          120 origin
//   128 hidden scale m     136 hidden scale e          144..176 the five saturation counts
//
// The backends own everything around the blob: which handles may be saved, where the rows come from and go
// to, and how a refusal below maps to their own status codes. This unit owns the bytes and the order in which
// a reader checks them.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "superslm/schema_masks.h"

namespace superslm::spx1 {

inline constexpr size_t kHeaderBytes = 184;
inline constexpr uint32_t kBackendCpu = 0;
inline constexpr uint32_t kBackendGpu = 1;
// The walk-state value of a blob bound to no schema.
inline constexpr uint32_t kWalkUnbound = 0xFFFFFFFFu;

// Every header field but the magic, the engine version and the digest, which the writer supplies.
struct Header {
	uint32_t backend = 0;
	uint32_t adapter_vendor = 0;
	uint32_t adapter_device = 0;
	std::array<uint8_t, 32> model_hash{};
	uint32_t kv_precision = 0;
	uint64_t schema_hash = 0;
	uint32_t walk = kWalkUnbound;
	int32_t current_token = -1;
	int64_t length = 0;
	int64_t origin = 0;
	int64_t scale_m = 0;
	int64_t scale_e = 0;
	uint64_t kv_saturation_count = 0;
	uint64_t kv_landing_saturation_count = 0;
	uint64_t k_channel_landing_saturation_count = 0;
	uint64_t rope_q_saturation_count = 0;
	uint64_t rope_k_saturation_count = 0;
};

// The blob size for `length` positions: 184 + hidden_size + length * runs * head_dim, by checked arithmetic.
// False on overflow.
bool BlobSize(size_t runs, size_t head_dim, size_t hidden_size, uint64_t length, size_t* out) noexcept;

// Where the residual and the rows start.
inline uint8_t* Residual(uint8_t* blob) noexcept { return blob + kHeaderBytes; }
inline uint8_t* Rows(uint8_t* blob, size_t hidden_size) noexcept { return blob + kHeaderBytes + hidden_size; }

// Writes the 184-byte header: magic, h, this library's engine version, and a zeroed digest field.
void WriteHeader(uint8_t* blob, const Header& h) noexcept;

// Writes the digest over the whole n-byte blob. Call it last, after the residual and the rows. Seal and Read
// throw only what superslm::Sha256 throws.
void Seal(uint8_t* blob, size_t n);

// What the reader expects of a blob: the reading backend, model and pool.
struct Expect {
	uint32_t backend = 0;
	// The producer adapter this blob must come from. Null means 0, 0 (the CPU). The callback runs after the
	// backend check and before the producer check; false refuses with kAdapterUnavailable.
	bool (*adapter_ids)(void* user, uint32_t* vendor, uint32_t* device) = nullptr;
	void* adapter_user = nullptr;
	const uint8_t* model_hash = nullptr;  // 32 bytes
	uint32_t kv_precision = 0;
	const SchemaMasksTable* schemas = nullptr;
	int64_t context_cap = 0;
	uint64_t vocab_size = 0;
	size_t runs = 0;
	size_t head_dim = 0;
	size_t hidden_size = 0;
};

// The reader's verdicts, in the order it checks them (the first failing check decides).
enum class Verdict {
	kOk,
	kSizeOrMagic,         // n < 184, or the magic is not 'SPX1'
	kDigest,              // the digest does not match
	kBackend,             // backend != Expect::backend
	kAdapterUnavailable,  // Expect::adapter_ids returned false
	kProducer,            // engine major, minor or patch, or adapter vendor or device, differ
	kModel,               // model hash
	kKvPrecision,         // kv_precision
	kSchema,              // the schema hash names no schema, the walk is past its states, or unbound with a walk
	kDomain,              // L outside [0, cap], n != the size for L, origin outside [0, L], the token, m outside int32
};

// The fields a validated blob carries.
struct Fields {
	Header header;
	int32_t schema_index = -1;  // the schema table index the hash resolved to, or -1 when unbound
	const uint8_t* residual = nullptr;
	const uint8_t* rows = nullptr;
};

// Validates the n-byte blob in the order above and, on kOk, fills *out. Reads nothing else and changes nothing.
Verdict Read(const uint8_t* blob, size_t n, const Expect& expect, Fields* out);

}  // namespace superslm::spx1
