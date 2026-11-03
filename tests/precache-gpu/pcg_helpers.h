// Precached-prefixes plan, step F1: the GPU red suite's helpers, for every [F2] cell of plan §7 (1.8, 2.7,
// 2.13's GPU leg, 3.4, 5.3, 5.7's GPU pairs, 5.10's GPU twin, 5.11's GPU half, 8.7, 11.5, 11.6's GPU leg).
//
// Conventions reused, not invented: the CHECK/CHECK_MSG harness, FinishSuite and StatusName of
// tests/t2791-gpu-prefill-read-red-suite/fixture_common.h, and the paged-KV GPU suite's environment, rig,
// token driver and fill probe (tests/paged-kv-gpu/g1e_helpers.h), so the precache cells run on the same
// pkv_def fixture (cap 4096, kv_block_size 16, so the GPU pool's B is 16) and probe pages the same way.
// One cell group per executable, each its own main().
//
// The 'SPX1' layout below is written here from plan §3.2's table, independently of the engine's writer
// (src/sslm_abi.cpp, src/gpu/gpu_1p0.cpp): every cell reads blob fields only through Spx1Parse, and edits
// them only through Spx1Edit*, which recompute the digest (§3.2: SHA-256 of the whole blob with the digest
// field zeroed).
//
// HOW THESE CELLS ARE RED BEFORE F2. The GPU verbs sslm_gpu_prefix_save and sslm_gpu_prefix_restore are
// declared in include/superslm/gpu_1p0.h and defined at F1 as stubs that return SSLM_DEVICE_LOST and write
// nothing, so every cell links and is red at run: the first save or restore a cell grades reads
// SSLM_DEVICE_LOST where it expects SSLM_OK or a named refusal. The new status
// SSLM_RESTORE_PRODUCER_MISMATCH is compared through the test-side constant kProducerMismatch (ordinal 30,
// plan §4 item 3 step 6), which F2's enumerator must equal (the static_assert below).
#ifndef SSLM_PCG_HELPERS_H
#define SSLM_PCG_HELPERS_H

#include "g1e_helpers.h"  // tests/paged-kv-gpu
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/schema_masks.h"
#include "superslm/sha256.h"
#include "superslm/sslm_version.h"

#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pcg {

using namespace g1e;

// Plan §4 item 3 step 6: SSLM_RESTORE_PRODUCER_MISMATCH, appended to SslmGpuStatus after paged KV's six.
constexpr SslmGpuStatus kProducerMismatch = static_cast<SslmGpuStatus>(30);
static_assert(kProducerMismatch == SslmGpuStatus::SSLM_RESTORE_PRODUCER_MISMATCH, "plan §4 item 3: ordinal 30");
// Schema skip-ahead on the GPU (1.17.0) appends two statuses after it, at 31 and 32; 30 stays.
static_assert(static_cast<uint32_t>(SslmGpuStatus::SSLM_GPU_FORCED_TABLE_ABSENT) == 31, "appended at 31");
static_assert(static_cast<uint32_t>(SslmGpuStatus::SSLM_GPU_CONTEXT_CAP_EXCEEDED) == 32, "appended at 32");

inline const char* Name(GS s) { return s == kProducerMismatch ? "SSLM_RESTORE_PRODUCER_MISMATCH" : StatusName(s); }

// ---- 'SPX1', plan §3.2, read and edited test-side ----------------------------------------------------

constexpr size_t kHeader = 184;
constexpr size_t kOffBackend = 4, kOffMajor = 8, kOffMinor = 12, kOffPatch = 16, kOffVendor = 20, kOffDevice = 24;
constexpr size_t kOffDigest = 28, kOffModelHash = 60, kOffKvPrecision = 92, kOffSchemaHash = 96, kOffWalk = 104;
constexpr size_t kOffToken = 108, kOffLength = 112, kOffOrigin = 120, kOffScaleM = 128, kOffScaleE = 136;
constexpr size_t kOffSat = 144, kOffKvLanding = 152, kOffKChannel = 160, kOffRopeQ = 168, kOffRopeK = 176;

struct Spx1 {
	bool magic = false;
	uint32_t backend = 0, major = 0, minor = 0, patch = 0, vendor = 0, device = 0;
	uint8_t digest[32] = {};
	uint8_t model_hash[32] = {};
	uint32_t kv_precision = 0;
	uint64_t schema_hash = 0;
	uint32_t walk = 0;
	int32_t token = 0;
	int64_t length = 0, origin = 0, m = 0, e = 0;
	uint64_t counts[5] = {};
	bool size_exact = false;   // size == 184 + hidden + L * bytes_per_token
	bool digest_ok = false;    // the stored digest is SHA-256 of the blob with the field zeroed
};

inline void Digest(const std::vector<uint8_t>& b, uint8_t out[32]) {
	std::vector<uint8_t> z(b);
	if (z.size() >= kOffDigest + 32) std::memset(z.data() + kOffDigest, 0, 32);
	superslm::Sha256Hash(z.data(), z.size(), out);
}
inline void Redigest(std::vector<uint8_t>& b) {
	if (b.size() < kHeader) return;
	uint8_t d[32];
	Digest(b, d);
	std::memcpy(b.data() + kOffDigest, d, 32);
}

inline Spx1 Spx1Parse(const std::vector<uint8_t>& b, const Geo& g) {
	Spx1 v;
	if (b.size() < kHeader) return v;
	v.magic = std::memcmp(b.data(), "SPX1", 4) == 0;
	v.backend = Le32(b, kOffBackend);
	v.major = Le32(b, kOffMajor);
	v.minor = Le32(b, kOffMinor);
	v.patch = Le32(b, kOffPatch);
	v.vendor = Le32(b, kOffVendor);
	v.device = Le32(b, kOffDevice);
	std::memcpy(v.digest, b.data() + kOffDigest, 32);
	std::memcpy(v.model_hash, b.data() + kOffModelHash, 32);
	v.kv_precision = Le32(b, kOffKvPrecision);
	v.schema_hash = Le64(b, kOffSchemaHash);
	v.walk = Le32(b, kOffWalk);
	v.token = static_cast<int32_t>(Le32(b, kOffToken));
	v.length = static_cast<int64_t>(Le64(b, kOffLength));
	v.origin = static_cast<int64_t>(Le64(b, kOffOrigin));
	v.m = static_cast<int64_t>(Le64(b, kOffScaleM));
	v.e = static_cast<int64_t>(Le64(b, kOffScaleE));
	for (int i = 0; i < 5; ++i) v.counts[i] = Le64(b, kOffSat + 8 * static_cast<size_t>(i));
	v.size_exact = v.length >= 0 && v.length <= g.cap &&
	               b.size() == kHeader + g.hidden + static_cast<size_t>(v.length) * g.BytesPerToken();
	uint8_t d[32];
	Digest(b, d);
	v.digest_ok = std::memcmp(d, v.digest, 32) == 0;
	return v;
}
// The canonical row (layer, half, head, pos) of an SPX1 blob: [layer][K|V][head][pos < L][d] after the residual.
inline size_t Spx1RowOff(const Geo& g, int64_t L, uint32_t layer, uint32_t half, uint32_t head, int64_t pos) {
	return kHeader + g.hidden + ((size_t{layer} * 2 + half) * g.Hkv + head) * static_cast<size_t>(L) * g.D +
	       static_cast<size_t>(pos) * g.D;
}
inline std::vector<uint8_t> Edit32(std::vector<uint8_t> b, size_t off, uint32_t v) {
	Put32(b, off, v);
	Redigest(b);
	return b;
}
inline std::vector<uint8_t> Edit64(std::vector<uint8_t> b, size_t off, uint64_t v) {
	Put64(b, off, v);
	Redigest(b);
	return b;
}

// The model's integrity hash, independently of the GPU (the view the fixture loaded).
inline std::array<uint8_t, 32> ModelHash(const Env& e) { return e.fx.view.RawIntegrityHash(); }

// Fnv1a64 of a schema name, written here (S6): the hash the blob must carry for a bound prefix.
inline uint64_t Fnv1a64(const char* s) {
	uint64_t h = 14695981039346656037ull;
	for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p; ++p) h = (h ^ *p) * 1099511628211ull;
	return h;
}

// The fixture's schema `index`, read with the shared SCM1 parser; its name hash is recomputed here.
struct SchemaInfo {
	bool ok = false;
	std::string name;
	uint64_t hash = 0;
	uint32_t state_count = 0;
};
inline SchemaInfo Schema(const Env& e, size_t index) {
	SchemaInfo s;
	const superslm::SslmSectionView* sec = e.fx.view.Section(superslm::SslmSectionType::SchemaMasks);
	superslm::SchemaMasksTable table;
	std::string err;
	if (!sec || !superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size, e.fx.view.config.vocab_size, table, &err))
		return s;
	const superslm::SchemaEntry* entry = table.ByIndex(index);
	if (!entry) return s;
	s.name = std::string(entry->name);
	s.hash = Fnv1a64(s.name.c_str());
	s.state_count = entry->state_count;
	s.ok = true;
	return s;
}

// The adapter the engine's submission device uses, read here through DXGI the way the harness selects
// it (SSLM_GPU_ADAPTER_INDEX when set, else the first non-software adapter by high-performance
// preference): the vendor and device ids a GPU save must carry (plan §3.2).
inline bool AdapterIds(uint32_t* vendor, uint32_t* device) {
	Microsoft::WRL::ComPtr<IDXGIFactory6> f;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f)))) return false;
	const char* env = std::getenv("SSLM_GPU_ADAPTER_INDEX");
	Microsoft::WRL::ComPtr<IDXGIAdapter1> a;
	if (env && *env) {
		if (FAILED(f->EnumAdapters1(static_cast<UINT>(std::atoi(env)), &a))) return false;
	} else {
		for (UINT i = 0;; ++i) {
			Microsoft::WRL::ComPtr<IDXGIAdapter1> c;
			if (f->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&c)) ==
			    DXGI_ERROR_NOT_FOUND)
				break;
			DXGI_ADAPTER_DESC1 d{};
			c->GetDesc1(&d);
			if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
			a = c;
			break;
		}
	}
	if (!a) return false;
	DXGI_ADAPTER_DESC1 d{};
	if (FAILED(a->GetDesc1(&d))) return false;
	*vendor = d.VendorId;
	*device = d.DeviceId;
	return true;
}

// ---- the verbs, with out-handles preset to the sentinel --------------------------------------------

inline std::vector<uint8_t> SavePrefix(const Env& e, const SslmGpuPrefixHandle* p, GS* status = nullptr) {
	size_t need = 0;
	const GS q = sslm_gpu_prefix_save(e.ctx, p, nullptr, &need);
	std::vector<uint8_t> blob;
	GS st = q;
	if (q == GS::SSLM_OUTPUT_BUFFER_TOO_SMALL && need >= kHeader) {
		blob.resize(need);
		size_t n = need;
		st = sslm_gpu_prefix_save(e.ctx, p, blob.data(), &n);
		if (st == GS::SSLM_OK) blob.resize(n);
	}
	if (status) {
		*status = st;
	} else {
		CHECK_MSG(st == GS::SSLM_OK, "sslm_gpu_prefix_save: size query %s (size %zu), save %s", Name(q), need, Name(st));
	}
	if (st != GS::SSLM_OK) blob.clear();
	return blob;
}

struct RestoredPrefix {
	GS status = GS::SSLM_DEVICE_LOST;
	SslmGpuPrefixHandle* p = nullptr;        // tracked by the rig when real
	SslmGpuPrefixHandle* raw_out = nullptr;  // what the verb left in *out
	uint32_t shared = 0xFFFFFFFFu;           // *out_shared_pages (a diagnostic: verdicts count by probe)
};
inline RestoredPrefix RestorePrefix(Rig& rig, const std::vector<uint8_t>& blob, SslmGpuPrefixHandle* hint = nullptr,
                                    SslmGpuContext* ctx = nullptr, SslmGpuModelHandle* model = nullptr,
                                    SslmGpuKvPoolHandle* pool = nullptr) {
	RestoredPrefix r;
	SslmGpuPrefixHandle* out = OutSentinel<SslmGpuPrefixHandle>();
	r.status = sslm_gpu_prefix_restore(ctx ? ctx : rig.ctx, model ? model : rig.model, pool ? pool : rig.pool,
	                                   blob.empty() ? nullptr : blob.data(), blob.size(), hint, &out, &r.shared);
	r.raw_out = out;
	r.p = rig.Prefix(out);
	return r;
}
// A refusal's shape: the named status and a null out-handle.
inline bool Refused(const RestoredPrefix& r, GS want, const std::string& what) {
	const bool ok = r.status == want && r.raw_out == nullptr;
	CHECK_MSG(ok, "%s: restore returned %s (want %s), out-handle %s", what.c_str(), Name(r.status), Name(want),
	          r.raw_out == nullptr ? "null" : (r.raw_out == OutSentinel<SslmGpuPrefixHandle>() ? "untouched" : "set"));
	return ok;
}

// A frozen budget prefix of `tokens` in the rig's pool (g1e's BudgetPrefix), with the S8 schema bound
// first when `schema` >= 0.
inline SslmGpuPrefixHandle* Prefix(Rig& rig, const std::vector<int32_t>& tokens, int32_t schema = -1, int32_t kind = 0) {
	SslmGpuPrefixHandle* p = nullptr;
	const int32_t budget = tokens.empty() ? 1 : static_cast<int32_t>(tokens.size());
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, budget, &p);
	if (st == GS::SSLM_OK && schema >= 0) st = sslm_gpu_prefix_set_schema(rig.ctx, p, schema);
	if (st == GS::SSLM_OK && !tokens.empty()) st = PrefixPrefillAll(rig.ctx, p, tokens, kind, rig.e.full_budget);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, p);
	CHECK_MSG(st == GS::SSLM_OK, "GPU prefix of %zu tokens (schema %d, kind %d): %s", tokens.size(), schema, kind,
	          StatusName(st));
	if (st != GS::SSLM_OK) {
		if (Real(p)) sslm_gpu_prefix_release(rig.ctx, p);
		return nullptr;
	}
	return rig.Prefix(p);
}

// `n` greedy tokens of an adopter of `p` (a fresh budgeted sequence of the rig), released after.
inline std::vector<int32_t> AdopterTokens(Rig& rig, SslmGpuPrefixHandle* p, int64_t prefix_len, int64_t n,
                                          int32_t last_token, int32_t budget = 256) {
	GSeq s = rig.Budgeted(budget);
	if (!s.h) return {};
	const GS st = Adopt(rig, s, p, prefix_len);
	CHECK_MSG(st == GS::SSLM_OK, "adopt: %s", StatusName(st));
	s.last = last_token;
	std::vector<int32_t> out = st == GS::SSLM_OK ? DecodeN(rig.e, s, n) : std::vector<int32_t>{};
	rig.ReleaseSeq(s.h);
	return out;
}

// The pages a prefix's table maps (test-only seam): its i-th entry for every i < mapped.
inline std::vector<uint32_t> TablePages(const SslmGpuPrefixHandle* p) {
	std::vector<uint32_t> pages;
	uint32_t mapped = 0, page = 0;
	if (sslm_gpu_test_only_prefix_table_entry(p, 0, &page, &mapped) != GS::SSLM_OK && mapped == 0) return pages;
	for (uint32_t i = 0; i < mapped; ++i) {
		if (sslm_gpu_test_only_prefix_table_entry(p, i, &page, nullptr) != GS::SSLM_OK) break;
		pages.push_back(page);
	}
	return pages;
}

// A sequence of the env's model held Submitted (pkvg's construction: a pooled decode left un-drained).
struct SubmittedSeq {
	SslmGpuSequenceHandle* h = nullptr;
	bool ok = false;
};
inline SubmittedSeq MakeSubmitted(Rig& rig) {
	SubmittedSeq s;
	GSeq g = rig.Budgeted(64);
	if (!g.h) return s;
	const std::vector<int32_t> t = Stream(91, 4, kVocab);
	CHECK(Prefill(rig.e, g, t) == GS::SSLM_OK);
	GS st = sslm_gpu_seq_embed_token(rig.ctx, g.h, t.back());
	if (st == GS::SSLM_OK) st = sslm_decode_step_gpu(rig.ctx, g.h, nullptr, rig.e.fx.one_layer_budget);
	CHECK_MSG(st == GS::SSLM_OK, "making a sequence Submitted: %s", StatusName(st));
	s.h = g.h;
	s.ok = st == GS::SSLM_OK;
	return s;
}

}  // namespace pcg

#endif  // SSLM_PCG_HELPERS_H
