// The 'SPX1' blob's one writer and one reader (src/spx1_format.h).

#include "spx1_format.h"

#include <climits>
#include <cstring>

#include "superslm/sha256.h"
#include "superslm/sslm_version.h"

namespace superslm::spx1 {

namespace {

constexpr uint8_t kMagic[4] = {'S', 'P', 'X', '1'};
constexpr size_t kDigestOffset = 28;
constexpr size_t kDigestBytes = 32;

void Put32(uint8_t* p, size_t off, uint32_t v) noexcept {
	for (int i = 0; i < 4; ++i) p[off + i] = static_cast<uint8_t>(v >> (8 * i));
}
void Put64(uint8_t* p, size_t off, uint64_t v) noexcept {
	for (int i = 0; i < 8; ++i) p[off + i] = static_cast<uint8_t>(v >> (8 * i));
}
uint32_t Get32(const uint8_t* p, size_t off) noexcept {
	uint32_t v = 0;
	for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[off + i]) << (8 * i);
	return v;
}
uint64_t Get64(const uint8_t* p, size_t off) noexcept {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[off + i]) << (8 * i);
	return v;
}

bool MulChecked(size_t a, size_t b, size_t* out) noexcept {
	if (a != 0 && b > SIZE_MAX / a) return false;
	*out = a * b;
	return true;
}
bool AddChecked(size_t a, size_t b, size_t* out) noexcept {
	if (b > SIZE_MAX - a) return false;
	*out = a + b;
	return true;
}

}  // namespace

bool BlobSize(size_t runs, size_t head_dim, size_t hidden_size, uint64_t length, size_t* out) noexcept {
	if (static_cast<uint64_t>(static_cast<size_t>(length)) != length) return false;
	size_t row_bytes = 0, required = kHeaderBytes;
	if (!MulChecked(runs, head_dim, &row_bytes) || !MulChecked(row_bytes, static_cast<size_t>(length), &row_bytes) ||
	    !AddChecked(required, hidden_size, &required) || !AddChecked(required, row_bytes, &required)) {
		return false;
	}
	*out = required;
	return true;
}

void WriteHeader(uint8_t* p, const Header& h) noexcept {
	std::memcpy(p, kMagic, 4);
	Put32(p, 4, h.backend);
	Put32(p, 8, SUPERSLM_VERSION_MAJOR);
	Put32(p, 12, SUPERSLM_VERSION_MINOR);
	Put32(p, 16, SUPERSLM_VERSION_PATCH);
	Put32(p, 20, h.adapter_vendor);
	Put32(p, 24, h.adapter_device);
	std::memset(p + kDigestOffset, 0, kDigestBytes);
	std::memcpy(p + 60, h.model_hash.data(), 32);
	Put32(p, 92, h.kv_precision);
	Put64(p, 96, h.schema_hash);
	Put32(p, 104, h.walk);
	Put32(p, 108, static_cast<uint32_t>(h.current_token));
	Put64(p, 112, static_cast<uint64_t>(h.length));
	Put64(p, 120, static_cast<uint64_t>(h.origin));
	Put64(p, 128, static_cast<uint64_t>(h.scale_m));
	Put64(p, 136, static_cast<uint64_t>(h.scale_e));
	Put64(p, 144, h.kv_saturation_count);
	Put64(p, 152, h.kv_landing_saturation_count);
	Put64(p, 160, h.k_channel_landing_saturation_count);
	Put64(p, 168, h.rope_q_saturation_count);
	Put64(p, 176, h.rope_k_saturation_count);
}

void Seal(uint8_t* p, size_t n) { Sha256Hash(p, n, p + kDigestOffset); }

namespace {

constexpr uint8_t kDeltaMagic[4] = {'S', 'P', 'D', '1'};

// The identity half of the reader, shared by SPX1 and SPD1: size and magic, the digest over the blob with its
// digest field zeroed (before any other field is trusted), backend, producer, model, precision and schema.
struct Identity {
	uint32_t vendor = 0, device = 0;
	uint64_t schema_hash = 0;
	uint32_t walk = kWalkUnbound;
	int32_t schema_index = -1;
};

bool DigestMatches(const uint8_t* p, size_t n) {
	uint8_t digest[kDigestBytes];
	static constexpr uint8_t zeros[kDigestBytes] = {};
	Sha256 hasher;
	hasher.Update(p, kDigestOffset);
	hasher.Update(zeros, kDigestBytes);
	hasher.Update(p + kDigestOffset + kDigestBytes, n - (kDigestOffset + kDigestBytes));
	hasher.Final(digest);
	return std::memcmp(digest, p + kDigestOffset, kDigestBytes) == 0;
}

Verdict CheckIdentity(const uint8_t* p, size_t n, size_t header_bytes, const uint8_t* magic, const Expect& x,
                      Identity* id) {
	if (n < header_bytes || std::memcmp(p, magic, 4) != 0) return Verdict::kSizeOrMagic;
	if (!DigestMatches(p, n)) return Verdict::kDigest;
	if (Get32(p, 4) != x.backend) return Verdict::kBackend;
	// The producer: engine version, adapter vendor and device.
	if (x.adapter_ids != nullptr && !x.adapter_ids(x.adapter_user, &id->vendor, &id->device)) {
		return Verdict::kAdapterUnavailable;
	}
	if (Get32(p, 8) != SUPERSLM_VERSION_MAJOR || Get32(p, 12) != SUPERSLM_VERSION_MINOR ||
	    Get32(p, 16) != SUPERSLM_VERSION_PATCH || Get32(p, 20) != id->vendor || Get32(p, 24) != id->device) {
		return Verdict::kProducer;
	}
	if (std::memcmp(p + 60, x.model_hash, 32) != 0) return Verdict::kModel;
	if (Get32(p, 92) != x.kv_precision) return Verdict::kKvPrecision;
	// The schema, by name hash (unique for every mapped model); the walk within its states, or the unbound
	// value when no schema is named.
	id->schema_hash = Get64(p, 96);
	id->walk = Get32(p, 104);
	if (id->schema_hash != 0) {
		size_t idx = 0;
		const SchemaEntry* entry = x.schemas != nullptr ? x.schemas->ByNameHash(id->schema_hash, &idx) : nullptr;
		if (entry == nullptr || id->walk >= entry->state_count) return Verdict::kSchema;
		id->schema_index = static_cast<int32_t>(idx);
	} else if (id->walk != kWalkUnbound) {
		return Verdict::kSchema;
	}
	return Verdict::kOk;
}

// The token and m domains, the last of both readers' checks.
bool TokenAndScaleInDomain(const uint8_t* p, int64_t length, uint64_t vocab_size) {
	const int32_t token = static_cast<int32_t>(Get32(p, 108));
	if ((length == 0 && token != -1) || (length > 0 && (token < 0 || static_cast<uint64_t>(token) >= vocab_size))) {
		return false;
	}
	const int64_t m = static_cast<int64_t>(Get64(p, 128));
	return m >= INT32_MIN && m <= INT32_MAX;
}

void FillHeader(const uint8_t* p, const Expect& x, const Identity& id, Header* h) {
	h->backend = x.backend;
	h->adapter_vendor = id.vendor;
	h->adapter_device = id.device;
	std::memcpy(h->model_hash.data(), p + 60, 32);
	h->kv_precision = x.kv_precision;
	h->schema_hash = id.schema_hash;
	h->walk = id.walk;
	h->current_token = static_cast<int32_t>(Get32(p, 108));
	h->length = static_cast<int64_t>(Get64(p, 112));
	h->origin = static_cast<int64_t>(Get64(p, 120));
	h->scale_m = static_cast<int64_t>(Get64(p, 128));
	h->scale_e = static_cast<int64_t>(Get64(p, 136));
	h->kv_saturation_count = Get64(p, 144);
	h->kv_landing_saturation_count = Get64(p, 152);
	h->k_channel_landing_saturation_count = Get64(p, 160);
	h->rope_q_saturation_count = Get64(p, 168);
	h->rope_k_saturation_count = Get64(p, 176);
}

}  // namespace

Verdict Read(const uint8_t* p, size_t n, const Expect& x, Fields* out) {
	Identity id;
	const Verdict identity = CheckIdentity(p, n, kHeaderBytes, kMagic, x, &id);
	if (identity != Verdict::kOk) return identity;
	// The domains, in order: L within the cap before any arithmetic on it, the exact size, the origin, the
	// token, and the hidden scale's m within int32.
	const int64_t length = static_cast<int64_t>(Get64(p, 112));
	size_t required = 0;
	if (length < 0 || length > x.context_cap ||
	    !BlobSize(x.runs, x.head_dim, x.hidden_size, static_cast<uint64_t>(length), &required) || n != required) {
		return Verdict::kDomain;
	}
	const int64_t origin = static_cast<int64_t>(Get64(p, 120));
	if (origin < 0 || origin > length) return Verdict::kDomain;
	if (!TokenAndScaleInDomain(p, length, x.vocab_size)) return Verdict::kDomain;

	FillHeader(p, x, id, &out->header);
	out->schema_index = id.schema_index;
	out->residual = p + kHeaderBytes;
	out->rows = p + kHeaderBytes + x.hidden_size;
	return Verdict::kOk;
}

bool DeltaSize(size_t runs, size_t head_dim, size_t hidden_size, uint64_t length, uint64_t origin,
               size_t* out) noexcept {
	if (origin < 1 || origin > length) return false;
	const uint64_t own = length - origin;
	if (static_cast<uint64_t>(static_cast<size_t>(own)) != own) return false;
	size_t row_bytes = 0, required = kDeltaHeaderBytes;
	if (!MulChecked(runs, head_dim, &row_bytes) || !MulChecked(row_bytes, static_cast<size_t>(own), &row_bytes) ||
	    !AddChecked(required, hidden_size, &required) || !AddChecked(required, row_bytes, &required)) {
		return false;
	}
	*out = required;
	return true;
}

void WriteDeltaHeader(uint8_t* p, const Header& h, const uint8_t* parent_rows_digest, uint32_t runs,
                      uint32_t head_dim, uint32_t hidden_size) noexcept {
	WriteHeader(p, h);
	std::memcpy(p, kDeltaMagic, 4);
	std::memcpy(p + kHeaderBytes, parent_rows_digest, 32);
	Put32(p, 216, runs);
	Put32(p, 220, head_dim);
	Put32(p, 224, hidden_size);
	Put32(p, 228, 0);
}

bool IsDelta(const uint8_t* p, size_t n) noexcept {
	return p != nullptr && n >= 4 && std::memcmp(p, kDeltaMagic, 4) == 0;
}

Verdict ReadDelta(const uint8_t* p, size_t n, const Expect& x, DeltaFields* out) {
	Identity id;
	const Verdict identity = CheckIdentity(p, n, kDeltaHeaderBytes, kDeltaMagic, x, &id);
	if (identity != Verdict::kOk) return identity;
	// The domains, in order: L within the cap, the origin within [1, L], the geometry and the reserved word, the
	// exact size, the token, and m.
	const int64_t length = static_cast<int64_t>(Get64(p, 112));
	if (length < 0 || length > x.context_cap) return Verdict::kDomain;
	const int64_t origin = static_cast<int64_t>(Get64(p, 120));
	if (origin < 1 || origin > length) return Verdict::kDomain;
	if (Get32(p, 216) != x.runs || Get32(p, 220) != x.head_dim || Get32(p, 224) != x.hidden_size ||
	    Get32(p, 228) != 0) {
		return Verdict::kDomain;
	}
	size_t required = 0;
	if (!DeltaSize(x.runs, x.head_dim, x.hidden_size, static_cast<uint64_t>(length), static_cast<uint64_t>(origin),
	               &required) || n != required) {
		return Verdict::kDomain;
	}
	if (!TokenAndScaleInDomain(p, length, x.vocab_size)) return Verdict::kDomain;

	FillHeader(p, x, id, &out->header);
	out->schema_index = id.schema_index;
	std::memcpy(out->parent_rows_digest.data(), p + kHeaderBytes, 32);
	out->residual = p + kDeltaHeaderBytes;
	out->rows = p + kDeltaHeaderBytes + x.hidden_size;
	return Verdict::kOk;
}

AssembleVerdict Assemble(const uint8_t* parent, size_t parent_n, const uint8_t* delta, size_t delta_n, uint8_t* out,
                         size_t* out_n) {
	if (parent == nullptr || delta == nullptr || out_n == nullptr) return AssembleVerdict::kMalformed;
	// The delta's envelope, with no model: size and magic, digest, geometry, reserved, origin within [1, L], size.
	if (delta_n < kDeltaHeaderBytes || std::memcmp(delta, kDeltaMagic, 4) != 0 || !DigestMatches(delta, delta_n)) {
		return AssembleVerdict::kMalformed;
	}
	const size_t runs = Get32(delta, 216), head_dim = Get32(delta, 220), hidden = Get32(delta, 224);
	const uint64_t length = Get64(delta, 112), origin = Get64(delta, 120);
	size_t delta_required = 0;
	if (Get32(delta, 228) != 0 || length > static_cast<uint64_t>(INT64_MAX) ||
	    !DeltaSize(runs, head_dim, hidden, length, origin, &delta_required) || delta_n != delta_required) {
		return AssembleVerdict::kMalformed;
	}
	// The parent's envelope: an SPX1 (never a delta) with a valid digest, and a size that is its length's under the
	// delta's geometry.
	if (parent_n < kHeaderBytes || std::memcmp(parent, kMagic, 4) != 0 || !DigestMatches(parent, parent_n)) {
		return AssembleVerdict::kMalformed;
	}
	const uint64_t parent_length = Get64(parent, 112);
	size_t parent_required = 0;
	if (parent_length > static_cast<uint64_t>(INT64_MAX) ||
	    !BlobSize(runs, head_dim, hidden, parent_length, &parent_required) || parent_n != parent_required) {
		return AssembleVerdict::kMalformed;
	}
	// The pair: the parent's length is the delta's origin; backend, producer, model and precision are equal; the
	// parent's rows hash to the delta's parent rows digest.
	if (parent_length != origin) return AssembleVerdict::kParentMismatch;
	if (std::memcmp(parent + 4, delta + 4, 24) != 0 || std::memcmp(parent + 60, delta + 60, 36) != 0) {
		return AssembleVerdict::kParentMismatch;
	}
	const uint8_t* parent_rows = parent + kHeaderBytes + hidden;
	uint8_t rows_digest[kDigestBytes];
	Sha256Hash(parent_rows, parent_n - kHeaderBytes - hidden, rows_digest);
	if (std::memcmp(rows_digest, delta + kHeaderBytes, kDigestBytes) != 0) return AssembleVerdict::kParentMismatch;

	size_t required = 0;
	if (!BlobSize(runs, head_dim, hidden, length, &required)) return AssembleVerdict::kMalformed;
	if (out == nullptr || *out_n < required) {
		*out_n = required;
		return AssembleVerdict::kBufferTooSmall;
	}
	// The output is written while both inputs are read, so it must overlap neither.
	auto overlaps = [&](const uint8_t* in, size_t in_n) {
		const auto o = reinterpret_cast<uintptr_t>(out), i = reinterpret_cast<uintptr_t>(in);
		return o < i + in_n && i < o + required;
	};
	if (overlaps(parent, parent_n) || overlaps(delta, delta_n)) return AssembleVerdict::kMalformed;
	std::memcpy(out, delta, kHeaderBytes);
	std::memcpy(out, kMagic, 4);
	std::memset(out + kDigestOffset, 0, kDigestBytes);
	std::memcpy(out + kHeaderBytes, delta + kDeltaHeaderBytes, hidden);
	const size_t run_parent = static_cast<size_t>(origin) * head_dim;
	const size_t run_own = static_cast<size_t>(length - origin) * head_dim;
	const uint8_t* own_rows = delta + kDeltaHeaderBytes + hidden;
	uint8_t* dst = out + kHeaderBytes + hidden;
	for (size_t r = 0; r < runs; ++r) {
		std::memcpy(dst, parent_rows + r * run_parent, run_parent);
		dst += run_parent;
		std::memcpy(dst, own_rows + r * run_own, run_own);
		dst += run_own;
	}
	Seal(out, required);
	*out_n = required;
	return AssembleVerdict::kOk;
}

}  // namespace superslm::spx1
