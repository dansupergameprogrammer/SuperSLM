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

Verdict Read(const uint8_t* p, size_t n, const Expect& x, Fields* out) {
	// The envelope: size and magic, then the digest, over the blob with its digest field zeroed, before any
	// other field is trusted.
	if (n < kHeaderBytes || std::memcmp(p, kMagic, 4) != 0) return Verdict::kSizeOrMagic;
	{
		uint8_t digest[kDigestBytes];
		static constexpr uint8_t zeros[kDigestBytes] = {};
		Sha256 hasher;
		hasher.Update(p, kDigestOffset);
		hasher.Update(zeros, kDigestBytes);
		hasher.Update(p + kDigestOffset + kDigestBytes, n - (kDigestOffset + kDigestBytes));
		hasher.Final(digest);
		if (std::memcmp(digest, p + kDigestOffset, kDigestBytes) != 0) return Verdict::kDigest;
	}
	if (Get32(p, 4) != x.backend) return Verdict::kBackend;
	// The producer: engine version, adapter vendor and device.
	uint32_t vendor = 0, device = 0;
	if (x.adapter_ids != nullptr && !x.adapter_ids(x.adapter_user, &vendor, &device)) {
		return Verdict::kAdapterUnavailable;
	}
	if (Get32(p, 8) != SUPERSLM_VERSION_MAJOR || Get32(p, 12) != SUPERSLM_VERSION_MINOR ||
	    Get32(p, 16) != SUPERSLM_VERSION_PATCH || Get32(p, 20) != vendor || Get32(p, 24) != device) {
		return Verdict::kProducer;
	}
	if (std::memcmp(p + 60, x.model_hash, 32) != 0) return Verdict::kModel;
	if (Get32(p, 92) != x.kv_precision) return Verdict::kKvPrecision;
	// The schema, by name hash (unique for every mapped model); the walk within its states, or the unbound
	// value when no schema is named.
	const uint64_t schema_hash = Get64(p, 96);
	const uint32_t walk = Get32(p, 104);
	int32_t schema_index = -1;
	if (schema_hash != 0) {
		size_t idx = 0;
		const SchemaEntry* entry = x.schemas != nullptr ? x.schemas->ByNameHash(schema_hash, &idx) : nullptr;
		if (entry == nullptr || walk >= entry->state_count) return Verdict::kSchema;
		schema_index = static_cast<int32_t>(idx);
	} else if (walk != kWalkUnbound) {
		return Verdict::kSchema;
	}
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
	const int32_t token = static_cast<int32_t>(Get32(p, 108));
	if ((length == 0 && token != -1) ||
	    (length > 0 && (token < 0 || static_cast<uint64_t>(token) >= x.vocab_size))) {
		return Verdict::kDomain;
	}
	const int64_t m = static_cast<int64_t>(Get64(p, 128));
	if (m < INT32_MIN || m > INT32_MAX) return Verdict::kDomain;

	Header& h = out->header;
	h.backend = x.backend;
	h.adapter_vendor = vendor;
	h.adapter_device = device;
	std::memcpy(h.model_hash.data(), p + 60, 32);
	h.kv_precision = x.kv_precision;
	h.schema_hash = schema_hash;
	h.walk = walk;
	h.current_token = token;
	h.length = length;
	h.origin = origin;
	h.scale_m = m;
	h.scale_e = static_cast<int64_t>(Get64(p, 136));
	h.kv_saturation_count = Get64(p, 144);
	h.kv_landing_saturation_count = Get64(p, 152);
	h.k_channel_landing_saturation_count = Get64(p, 160);
	h.rope_q_saturation_count = Get64(p, 168);
	h.rope_k_saturation_count = Get64(p, 176);
	out->schema_index = schema_index;
	out->residual = p + kHeaderBytes;
	out->rows = p + kHeaderBytes + x.hidden_size;
	return Verdict::kOk;
}

}  // namespace superslm::spx1
