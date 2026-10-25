// Precached-prefixes plan §7, step F1 (green at F2): the GPU 'SPX1' format and validation cells.
//   F2 capture  a real GPU save, independently checked (plan §9 F2): magic, backend 1, engine version,
//               the adapter's vendor and device, model hash, kv_precision, schema hash and walk, token,
//               length, origin, exact size, digest, and every row against the pool's host mirror.
//   2.7 [F2]    GPU twins of 2.1-2.5 with §4 item 3's statuses, plus a cross-context model and pool.
//               (2.6's GPU twin is in pcg_cell_pool.cpp.) Nothing drawn: the pool's free count is read
//               by the fill probe after every refusal family.
//   5.7 [F2]    the GPU pairs of §7 5.7, and the CPU-side pair (backend + producer) on a real GPU save.
//   8.7 [F2]    a CPU 'SPX1' given to the GPU restore -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH; a GPU one given
//               to the CPU restore -> SSLM_INVALID_ARGUMENT.
//   save        2.4's GPU twin: null arguments, another context's prefix, unfrozen, short buffer (size
//               written, nothing else), exact size with a canary after it.
//   smoke       save -> restore -> save byte-identical (6.4's shape on GPU), and the GPU blob against the
//               CPU twin's blob of the same tokens outside the producer fields (4.6's shape [F3]; on the
//               software GPU a smoke reading, recorded either way).
//
// RED AT F1: the save stub returns SSLM_DEVICE_LOST, so the capture's first check fails, and every
// restore case reads SSLM_DEVICE_LOST where it expects its named status.
//
// Run: pcg_cell_format.exe --fixture-dir=DIR (tools/gen_paged_kv_fixture.py's output; pkv_def).
#include "pcg_helpers.h"
#include "g1e_cpu_twin.h"

#include <climits>
#include <cstdio>

using namespace pcg;

namespace {

constexpr int32_t kLen = 40;      // a root prefix spanning three pages at B = 16, the last partial
constexpr uint32_t kPool = 64;    // pages: enough for every leg's prefixes, restores and probes
constexpr uint32_t kRefusalPool = 16;  // the refusal legs' own pool, read whole by the probe afterwards

// The fill-probe check every refusal family ends with: the rig's pool still has `free` pages.
void FreeIs(Rig& rig, int64_t free, const char* what) {
	struct Same : ProbeState {
		Rig& r;
		explicit Same(Rig& rr) : r(rr) {}
		SslmGpuKvPoolHandle* Pool() override { return r.pool; }
		SslmGpuContext* Ctx() override { return r.ctx; }
		SslmGpuModelHandle* Model() override { return r.model; }
		void DrainAll() override { r.DrainAll(); }
	} s(rig);
	ProbeExactlyFreeOneState(rig.e, s, free, what);
}

// ---- the capture ------------------------------------------------------------------------------------

void Capture(const Env& e, Rig& rig, SslmGpuPrefixHandle* p, const std::vector<int32_t>& tokens, int32_t schema,
             const char* what) {
	const std::vector<uint8_t> b = SavePrefix(e, p);
	if (b.empty()) return;
	const Spx1 v = Spx1Parse(b, e.geo);
	uint32_t vendor = 0, device = 0;
	const bool ids = AdapterIds(&vendor, &device);
	CHECK_MSG(ids, "%s: the adapter's ids could not be read through DXGI", what);
	const auto mh = ModelHash(e);
	const SchemaInfo si = Schema(e, 0);
	const uint64_t want_schema = schema >= 0 ? si.hash : 0;
	CHECK_MSG(v.magic, "%s: magic is not SPX1", what);
	CHECK_MSG(v.backend == 1, "%s: backend %u, want 1", what, v.backend);
	CHECK_MSG(v.major == SUPERSLM_VERSION_MAJOR && v.minor == SUPERSLM_VERSION_MINOR && v.patch == SUPERSLM_VERSION_PATCH,
	          "%s: engine version %u.%u.%u, want %d.%d.%d", what, v.major, v.minor, v.patch, SUPERSLM_VERSION_MAJOR,
	          SUPERSLM_VERSION_MINOR, SUPERSLM_VERSION_PATCH);
	CHECK_MSG(ids && v.vendor == vendor && v.device == device, "%s: adapter %#x:%#x, DXGI reads %#x:%#x", what, v.vendor,
	          v.device, vendor, device);
	CHECK_MSG(v.vendor != 0, "%s: the vendor id is 0, the CPU's value", what);
	CHECK_MSG(std::memcmp(v.model_hash, mh.data(), 32) == 0, "%s: model hash differs from the fixture's", what);
	CHECK_MSG(v.kv_precision == static_cast<uint32_t>(e.fx.view.config.kv_precision), "%s: kv_precision %u", what,
	          v.kv_precision);
	CHECK_MSG(v.schema_hash == want_schema, "%s: schema hash %#llx, want %#llx", what,
	          static_cast<unsigned long long>(v.schema_hash), static_cast<unsigned long long>(want_schema));
	CHECK_MSG(schema >= 0 ? v.walk < si.state_count : v.walk == 0xFFFFFFFFu, "%s: walk state %u", what, v.walk);
	CHECK_MSG(v.length == static_cast<int64_t>(tokens.size()) && v.origin == 0, "%s: length %lld origin %lld", what,
	          static_cast<long long>(v.length), static_cast<long long>(v.origin));
	CHECK_MSG(tokens.empty() ? v.token == -1 : v.token == tokens.back(), "%s: current_token %d, want %d", what, v.token,
	          tokens.empty() ? -1 : tokens.back());
	CHECK_MSG(v.m >= INT32_MIN && v.m <= INT32_MAX, "%s: scale m %lld out of int32", what, static_cast<long long>(v.m));
	CHECK_MSG(v.size_exact, "%s: size %zu is not 184 + hidden + L * bytes_per_token", what, b.size());
	CHECK_MSG(v.digest_ok, "%s: digest is not SHA-256 of the blob with the field zeroed", what);
	// Every row equals the pool's host mirror, read through the prefix's own table.
	const std::vector<uint32_t> pages = TablePages(p);
	CHECK_MSG(static_cast<int64_t>(pages.size()) == e.geo.PrefixPages(v.length), "%s: the prefix maps %zu pages, want %lld",
	          what, pages.size(), static_cast<long long>(e.geo.PrefixPages(v.length)));
	if (static_cast<int64_t>(pages.size()) != e.geo.PrefixPages(v.length) || !v.size_exact) return;
	size_t bad = 0;
	for (int64_t pos = 0; pos < v.length; ++pos) {
		const std::vector<uint8_t> page = PeekPage(e, rig.pool, pages[static_cast<size_t>(pos / e.geo.B)]);
		for (uint32_t l = 0; l < e.geo.L; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < e.geo.Hkv; ++h)
					bad += std::memcmp(page.data() + e.geo.InPage(l, half, h, pos),
					                   b.data() + Spx1RowOff(e.geo, v.length, l, half, h, pos), e.geo.D) != 0;
	}
	CHECK_MSG(bad == 0, "%s: %zu rows differ from the host mirror", what, bad);
}

// ---- 2.4's GPU twin -----------------------------------------------------------------------------------

void SaveRefusals(const Env& e, Rig& rig, SslmGpuPrefixHandle* frozen) {
	size_t n = 0;
	CHECK_MSG(sslm_gpu_prefix_save(e.ctx, frozen, nullptr, nullptr) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	          "save: a null size pointer is not SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
	n = 7;
	CHECK_MSG(sslm_gpu_prefix_save(e.ctx, nullptr, nullptr, &n) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && n == 7,
	          "save: a null prefix is not SSLM_SEQUENCE_KV_BUFFER_MISMATCH with nothing written");
	CHECK_MSG(sslm_gpu_prefix_save(nullptr, frozen, nullptr, &n) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && n == 7,
	          "save: a null context is not SSLM_SEQUENCE_KV_BUFFER_MISMATCH with nothing written");
	SslmGpuContext* other = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &other) == GS::SSLM_OK && other) {
		CHECK_MSG(sslm_gpu_prefix_save(other, frozen, nullptr, &n) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && n == 7,
		          "save: a prefix of another context is not SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
		sslm_gpu_context_destroy(other);
	}
	// Unfrozen: SSLM_GPU_PREFIX_REJECTED, nothing written.
	SslmGpuPrefixHandle* open = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, 8, &open);
	if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, open, PrefixTokens(5, 3), 0, e.full_budget);
	if (st == GS::SSLM_OK) {
		rig.Prefix(open);
		std::vector<uint8_t> buf(4096, 0xAB);
		size_t m = buf.size();
		const GS s2 = sslm_gpu_prefix_save(e.ctx, open, buf.data(), &m);
		CHECK_MSG(s2 == GS::SSLM_GPU_PREFIX_REJECTED && m == buf.size() && buf[0] == 0xAB,
		          "save: an unfrozen prefix returned %s (size %zu), want SSLM_GPU_PREFIX_REJECTED with nothing written",
		          Name(s2), m);
	}
	// Short buffer: the exact size written, nothing else; then exactly that many bytes, canary intact.
	size_t need = 0;
	const GS q = sslm_gpu_prefix_save(e.ctx, frozen, nullptr, &need);
	const size_t want = kHeader + e.geo.hidden + static_cast<size_t>(kLen) * e.geo.BytesPerToken();
	CHECK_MSG(q == GS::SSLM_OUTPUT_BUFFER_TOO_SMALL && need == want, "save: size query %s, size %zu (want %zu)", Name(q),
	          need, want);
	std::vector<uint8_t> buf(want + 64, 0xAB);
	size_t m = want - 1;
	const GS s3 = sslm_gpu_prefix_save(e.ctx, frozen, buf.data(), &m);
	CHECK_MSG(s3 == GS::SSLM_OUTPUT_BUFFER_TOO_SMALL && m == want && buf[0] == 0xAB,
	          "save: a buffer one byte short returned %s, size %zu, first byte %#x", Name(s3), m, buf[0]);
	m = buf.size();
	const GS s4 = sslm_gpu_prefix_save(e.ctx, frozen, buf.data(), &m);
	bool canary = true;
	for (size_t i = want; i < buf.size(); ++i) canary = canary && buf[i] == 0xAB;
	CHECK_MSG(s4 == GS::SSLM_OK && m == want && canary, "save: %s, wrote %zu (want %zu), canary %s", Name(s4), m, want,
	          canary ? "intact" : "overwritten");
}

// ---- 2.7: the restore's refusals ----------------------------------------------------------------------

void Truncations(Rig& rig, const std::vector<uint8_t>& blob, const char* what, bool every_offset) {
	size_t cases = 0, bad = 0;
	auto one = [&](size_t len) {
		std::vector<uint8_t> t(blob.begin(), blob.begin() + static_cast<std::ptrdiff_t>(std::min(len, blob.size())));
		if (len > blob.size()) t.push_back(0);
		const RestoredPrefix r = RestorePrefix(rig, t);
		++cases;
		if (!(r.status == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && r.raw_out == nullptr)) {
			if (bad++ < 4) CHECK_MSG(false, "%s: truncated to %zu bytes: %s", what, len, Name(r.status));
		}
	};
	for (size_t len = 0; len < blob.size(); ++len)
		if (every_offset || len <= kHeader || (len - kHeader) % 4096 == 0) one(len);
	one(blob.size() + 1);
	CHECK_MSG(bad == 0, "%s: %zu of %zu truncations or extensions not refused SSLM_SEQUENCE_KV_BUFFER_MISMATCH", what, bad,
	          cases);
	std::printf("%s: %zu truncations/extensions, %zu not refused\n", what, cases, bad);
}

void Flips(Rig& rig, const std::vector<uint8_t>& blob) {
	size_t bad = 0;
	for (size_t i = 0; i < blob.size(); ++i) {
		std::vector<uint8_t> f(blob);
		f[i] ^= 0x01;
		const RestoredPrefix r = RestorePrefix(rig, f);
		if (!(r.status == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && r.raw_out == nullptr) && bad++ < 4)
			CHECK_MSG(false, "2.7/2.2: byte %zu flipped: %s", i, Name(r.status));
	}
	CHECK_MSG(bad == 0, "2.7/2.2: %zu of %zu single-byte flips not refused", bad, blob.size());
	std::printf("2.7/2.2: %zu flips, %zu not refused\n", blob.size(), bad);
}

struct Case {
	std::string name;
	std::vector<uint8_t> blob;
	GS want;
};

std::vector<Case> HostileFields(const Env& e, const std::vector<uint8_t>& b, const std::vector<uint8_t>& bound) {
	const Spx1 v = Spx1Parse(b, e.geo);
	const SchemaInfo si = Schema(e, 0);
	std::vector<Case> c;
	auto add = [&](const std::string& n, std::vector<uint8_t> x, GS w) { c.push_back(Case{n, std::move(x), w}); };
	const GS KV = GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	add("backend 0", Edit32(b, kOffBackend, 0), KV);
	add("backend 2", Edit32(b, kOffBackend, 2), KV);
	add("engine major + 1", Edit32(b, kOffMajor, v.major + 1), kProducerMismatch);
	add("engine minor + 1", Edit32(b, kOffMinor, v.minor + 1), kProducerMismatch);
	add("engine patch + 1", Edit32(b, kOffPatch, v.patch + 1), kProducerMismatch);
	add("adapter vendor alone", Edit32(b, kOffVendor, v.vendor ^ 0x1u), kProducerMismatch);
	add("adapter device alone", Edit32(b, kOffDevice, v.device ^ 0x1u), kProducerMismatch);
	add("adapter vendor and device 0 (the CPU's)", Edit32(Edit32(b, kOffVendor, 0), kOffDevice, 0), kProducerMismatch);
	{
		std::vector<uint8_t> x(b);
		x[kOffModelHash] ^= 0x80;
		Redigest(x);
		add("other model hash", x, GS::SSLM_RESTORE_MODEL_MISMATCH);
	}
	add("other kv_precision", Edit32(b, kOffKvPrecision, v.kv_precision ^ 1u), KV);
	add("unknown schema hash", Edit64(b, kOffSchemaHash, 0x0123456789ABCDEFull), KV);
	add("unbound with a non-sentinel walk", Edit32(b, kOffWalk, 0), KV);
	if (!bound.empty() && si.ok) add("walk = state_count", Edit32(bound, kOffWalk, si.state_count), KV);
	add("origin > L", Edit64(b, kOffOrigin, static_cast<uint64_t>(v.length + 1)), KV);
	add("origin = -1", Edit64(b, kOffOrigin, static_cast<uint64_t>(-1)), KV);
	add("L = -1", Edit64(b, kOffLength, static_cast<uint64_t>(-1)), KV);
	add("L = cap + 1", Edit64(b, kOffLength, static_cast<uint64_t>(e.geo.cap + 1)), KV);
	{
		// An L whose L * bytes_per_token wraps the size formula onto n: refused by the cap clause.
		const uint64_t bpt = e.geo.BytesPerToken();
		const uint64_t wrap = (UINT64_MAX / bpt) + 1 + static_cast<uint64_t>(v.length);
		add("L wrapping the size formula", Edit64(b, kOffLength, wrap), KV);
	}
	for (int32_t t : {-1, -2, INT32_MIN})
		add("current_token " + std::to_string(t) + " with L > 0", Edit32(b, kOffToken, static_cast<uint32_t>(t)), KV);
	add("current_token = vocab", Edit32(b, kOffToken, static_cast<uint32_t>(e.fx.vocab)), KV);
	add("m = INT32_MAX + 1", Edit64(b, kOffScaleM, static_cast<uint64_t>(int64_t{INT32_MAX} + 1)), KV);
	add("m = INT32_MIN - 1", Edit64(b, kOffScaleM, static_cast<uint64_t>(int64_t{INT32_MIN} - 1)), KV);
	{
		std::vector<uint8_t> x(b);
		x.push_back(0);
		Redigest(x);
		add("size != formula (one byte appended, digest recomputed)", x, KV);
	}
	return c;
}

void RestoreRefusals(const Env& e, Rig& rig, const std::vector<uint8_t>& b1, const std::vector<uint8_t>& b40,
                     const std::vector<uint8_t>& bound) {
	Truncations(rig, b1, "2.7/2.1 L = 1, every offset", true);
	Truncations(rig, b40, "2.7/2.1 L = 40, header offsets and every 4096th byte", false);
	Flips(rig, b1);
	// Foreign blobs by magic: SLM6 and SLM5 (GPU sequence saves), SSB6 and SSB5 (CPU sequence saves).
	{
		GSeq s = rig.Budgeted(32);
		if (s.h) {
			CHECK(Prefill(e, s, PrefixTokens(5, 5)) == GS::SSLM_OK);
			const std::vector<uint8_t> slm6 = Save(e, s.h);
			Refused(RestorePrefix(rig, slm6), GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.7/2.2 an SLM6 blob");
			rig.ReleaseSeq(s.h);
		}
		SslmGpuSequenceHandle* d = nullptr;
		if (sslm_gpu_seq_create(e.ctx, e.model(), 64, &d) == GS::SSLM_OK && d) {
			GSeq g;
			g.h = d;
			CHECK(Prefill(e, g, PrefixTokens(5, 5)) == GS::SSLM_OK);
			const std::vector<uint8_t> slm5 = Save(e, d);
			Refused(RestorePrefix(rig, slm5), GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.7/2.2 an SLM5 blob");
			sslm_gpu_seq_release(e.ctx, d);
		}
		for (const char* m : {"SSB6", "SSB5"}) {
			std::vector<uint8_t> x(b40);
			std::memcpy(x.data(), m, 4);
			Refused(RestorePrefix(rig, x), GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, std::string("2.7/2.2 magic ") + m);
		}
		// X9's GPU half: an SPX1 given to the GPU sequence restores.
		SslmGpuSequenceHandle* o = OutSentinel<SslmGpuSequenceHandle>();
		const GS r1 = sslm_gpu_seq_restore(e.ctx, e.model(), b40.data(), b40.size(), &o);
		CHECK_MSG(r1 != GS::SSLM_OK && o == nullptr, "2.7/2.2: sslm_gpu_seq_restore accepted an SPX1 blob (%s)", Name(r1));
		o = OutSentinel<SslmGpuSequenceHandle>();
		const GS r2 = sslm_gpu_seq_restore_into_pool(e.ctx, e.model(), rig.pool, b40.data(), b40.size(), nullptr, &o, nullptr);
		CHECK_MSG(r2 == GS::SSLM_GPU_HOLDER_MODE_MISMATCH && o == nullptr,
		          "2.7/2.2: sslm_gpu_seq_restore_into_pool given an SPX1 returned %s, want SSLM_GPU_HOLDER_MODE_MISMATCH",
		          Name(r2));
	}
	for (const Case& c : HostileFields(e, b40, bound))
		Refused(RestorePrefix(rig, c.blob), c.want, std::string("2.7/2.3 ") + c.name);
	// The L = 1 blob's own domain case: its size against the formula.
	{
		std::vector<uint8_t> x(b1);
		x.pop_back();
		Redigest(x);
		Refused(RestorePrefix(rig, x), GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "2.7/2.3 L = 1, one byte short, digest recomputed");
	}
}

void ArgumentRefusals(const Env& e, Rig& rig, const std::vector<uint8_t>& b) {
	const GS KV = GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	uint32_t shared = 77;
	CHECK_MSG(sslm_gpu_prefix_restore(e.ctx, e.model(), rig.pool, b.data(), b.size(), nullptr, nullptr, &shared) == KV,
	          "2.7/2.5: a null out-handle is not SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
	{
		const RestoredPrefix ok = RestorePrefix(rig, b);
		CHECK_MSG(ok.status == GS::SSLM_OK && ok.p, "2.7 control: the unedited blob restores: %s", Name(ok.status));
		if (ok.p) {
			rig.prefixes.pop_back();
			sslm_gpu_prefix_release(rig.ctx, ok.p);
		}
	}
	{
		SslmGpuPrefixHandle* out = OutSentinel<SslmGpuPrefixHandle>();
		const GS st = sslm_gpu_prefix_restore(nullptr, e.model(), rig.pool, b.data(), b.size(), nullptr, &out, nullptr);
		CHECK_MSG(st == KV && out == nullptr, "2.7/2.5 null context: %s", Name(st));
		out = OutSentinel<SslmGpuPrefixHandle>();
		const GS s2 = sslm_gpu_prefix_restore(e.ctx, nullptr, rig.pool, b.data(), b.size(), nullptr, &out, nullptr);
		CHECK_MSG(s2 == KV && out == nullptr, "2.7/2.5 null model: %s", Name(s2));
		out = OutSentinel<SslmGpuPrefixHandle>();
		const GS s3 = sslm_gpu_prefix_restore(e.ctx, e.model(), nullptr, b.data(), b.size(), nullptr, &out, nullptr);
		CHECK_MSG(s3 == KV && out == nullptr, "2.7/2.5 null pool: %s", Name(s3));
		out = OutSentinel<SslmGpuPrefixHandle>();
		const GS s4 = sslm_gpu_prefix_restore(e.ctx, e.model(), rig.pool, nullptr, b.size(), nullptr, &out, nullptr);
		CHECK_MSG(s4 == KV && out == nullptr, "2.7/2.5 null blob: %s", Name(s4));
	}
	// A pool of another model (a second mapping of the fixture on the same context).
	GpuModelFixture other;
	if (other.Open(e.fx.path, e.ctx)) {
		Rig orig(e, 16, e.ctx, other.model);
		Refused(RestorePrefix(rig, b, nullptr, nullptr, nullptr, orig.pool), KV, "2.7/2.5 a pool of another model");
		// The other model's own pool, with the other model: the blob's model hash matches the artifact, so it
		// restores (the model check is by content hash, as the CPU's is).
	}
	// Cross-context model and pool (§4 item 3 step 1).
	SslmGpuContext* ctx2 = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx2) == GS::SSLM_OK && ctx2) {
		{
			GpuModelFixture m2;
			const bool opened = m2.Open(e.fx.path, ctx2);
			CHECK_MSG(opened, "2.7: the model did not map on a second context");
			if (opened) {
				Rig r2(e, 16, ctx2, m2.model);
				Refused(RestorePrefix(rig, b, nullptr, nullptr, m2.model, nullptr), KV, "2.7 a model of another context");
				Refused(RestorePrefix(rig, b, nullptr, nullptr, nullptr, r2.pool), KV, "2.7 a pool of another context");
				Refused(RestorePrefix(rig, b, nullptr, ctx2, nullptr, nullptr), KV, "2.7 a context the model and pool are not of");
			}
			m2.Close();
		}
		sslm_gpu_context_destroy(ctx2);
	}
}

// ---- 5.7's GPU pairs ----------------------------------------------------------------------------------

void Pairs(const Env& e, Rig& rig, const std::vector<uint8_t>& b, const std::vector<uint8_t>& b1) {
	const GS KV = GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const Spx1 v = Spx1Parse(b, e.geo);
	std::vector<uint8_t> magic(b), digest(b), backend = Edit32(b, kOffBackend, 0);
	std::memcpy(magic.data(), "SPX2", 4);
	digest[kHeader + 3] ^= 0x10;  // a residual byte, the digest left stale
	const std::vector<uint8_t> producer = Edit32(b, kOffMinor, v.minor + 1);
	// Each of steps 3, 4 and 5 + a producer mismatch -> the earlier step's status.
	std::vector<uint8_t> magic_p(producer), digest_p(producer), backend_p = Edit32(producer, kOffBackend, 0);
	std::memcpy(magic_p.data(), "SPX2", 4);
	digest_p[kHeader + 3] ^= 0x10;
	Refused(RestorePrefix(rig, magic_p), KV, "5.7 GPU: step 3 (magic) + producer");
	Refused(RestorePrefix(rig, digest_p), KV, "5.7 GPU: step 4 (digest) + producer");
	Refused(RestorePrefix(rig, backend_p), KV, "5.7 GPU: step 5 (backend) + producer");
	// Each of steps 8, 9 and 10 + a model mismatch -> SSLM_RESTORE_MODEL_MISMATCH (step 7 precedes them).
	auto model_edit = [](std::vector<uint8_t> x) {
		x[kOffModelHash + 5] ^= 0x01;
		Redigest(x);
		return x;
	};
	Refused(RestorePrefix(rig, model_edit(Edit32(b, kOffKvPrecision, v.kv_precision ^ 1u))), GS::SSLM_RESTORE_MODEL_MISMATCH,
	        "5.7 GPU: step 8 (kv_precision) + model");
	Refused(RestorePrefix(rig, model_edit(Edit64(b, kOffSchemaHash, 0x0123456789ABCDEFull))), GS::SSLM_RESTORE_MODEL_MISMATCH,
	        "5.7 GPU: step 9 (schema) + model");
	Refused(RestorePrefix(rig, model_edit(Edit64(b, kOffOrigin, static_cast<uint64_t>(v.length + 1)))),
	        GS::SSLM_RESTORE_MODEL_MISMATCH, "5.7 GPU: step 10 (domain) + model");
	// Step 6 + 7: the producer decides.
	Refused(RestorePrefix(rig, model_edit(producer)), kProducerMismatch, "5.7 GPU: step 6 (producer) + model");
	// Step 10 + an exhausted pool: the domain decides (validation precedes the draw).
	{
		Rig full(e, 4);
		std::vector<SslmGpuSequenceHandle*> held;
		AdmitPages(e.geo, full, 4, &held);
		Refused(RestorePrefix(full, Edit64(b1, kOffOrigin, 2)), KV, "5.7 GPU: step 10 (domain) + exhausted pool");
		Refused(RestorePrefix(full, b1), GS::SSLM_GPU_KV_POOL_EXHAUSTED, "5.7 GPU control: the valid blob, exhausted pool");
		for (auto it = held.rbegin(); it != held.rend(); ++it) sslm_gpu_seq_release(e.ctx, *it);
	}
	// Steps 3, 4 and 5 while a sequence is Submitted -> SSLM_BUSY (step 2 precedes them); a cross-context pool
	// while Submitted -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH (step 1 precedes the guard).
	{
		// The cross-context pool is made before anything is Submitted: a model map is itself refused Busy.
		SslmGpuContext* ctx2 = nullptr;
		GpuModelFixture m2;
		SslmGpuKvPoolHandle* p2 = nullptr;
		const bool have2 = sslm_gpu_context_create(GpuContextConfig{}, &ctx2) == GS::SSLM_OK && ctx2 &&
		                   m2.Open(e.fx.path, ctx2) && sslm_gpu_kv_pool_create(ctx2, m2.model, 8, &p2) == GS::SSLM_OK;
		CHECK_MSG(have2, "5.7 GPU: the second context, its model and its pool did not come up");
		SubmittedSeq sub = MakeSubmitted(rig);
		if (sub.ok) {
			Refused(RestorePrefix(rig, magic), GS::SSLM_BUSY, "5.7 GPU: step 3 (magic) + Submitted");
			Refused(RestorePrefix(rig, digest), GS::SSLM_BUSY, "5.7 GPU: step 4 (digest) + Submitted");
			Refused(RestorePrefix(rig, backend), GS::SSLM_BUSY, "5.7 GPU: step 5 (backend) + Submitted");
			if (have2) Refused(RestorePrefix(rig, b, nullptr, nullptr, nullptr, p2), KV, "5.7 GPU: a cross-context pool + Submitted");
			CHECK(Drain(e, sub.h) == GS::SSLM_OK);
			rig.ReleaseSeq(sub.h);
		}
		if (p2) sslm_gpu_kv_pool_destroy(ctx2, p2);
		m2.Close();
		if (ctx2) sslm_gpu_context_destroy(ctx2);
	}
}

// ---- the CPU twin: 8.7, the CPU-side 5.7 pair on a real GPU save, the byte-identity smoke ---------------

void CpuLegs(const Env& e, Rig& rig, const std::vector<uint8_t>& gpu_blob, const std::vector<int32_t>& tokens) {
	cpu::Model m(e.fx.bytes);
	if (!m.ok) return;
	cpu::Pool pool(m.model, 32);
	if (!pool.pool) return;
	// The CPU twin prefix of the same tokens.
	sslm_prefix cp = nullptr;
	sslm_status cs = sslm_prefix_begin_budgeted(m.model, &pool.pool, static_cast<int32_t>(tokens.size()), &cp);
	if (cs == SSLM_OK) {
		pool.Prefix(cp);
		cs = cpu::PrefixPrefillAll(m.model, cp, tokens);
	}
	if (cs == SSLM_OK) cs = sslm_prefix_freeze(cp);
	CHECK_MSG(cs == SSLM_OK, "CPU twin prefix: %d", static_cast<int>(cs));
	if (cs != SSLM_OK) return;
	size_t n = 0;
	(void)sslm_prefix_save(cp, nullptr, &n);
	std::vector<uint8_t> cpu_blob(n);
	CHECK(sslm_prefix_save(cp, cpu_blob.data(), &n) == SSLM_OK);
	// 8.7: a CPU blob given to the GPU restore.
	Refused(RestorePrefix(rig, cpu_blob), GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "8.7 a CPU SPX1 given to the GPU restore");
	if (gpu_blob.empty()) return;
	// 8.7 and the CPU-side 5.7 pair on the real GPU save (backend 1, the adapter's ids, this version): step 4
	// refuses it, SSLM_INVALID_ARGUMENT; with its version edited too (steps 4 + 5), still step 4's status.
	auto cpu_restore = [&](const std::vector<uint8_t>& x) {
		sslm_prefix out = reinterpret_cast<sslm_prefix>(uintptr_t{1});
		const sslm_status st = sslm_prefix_restore(m.model, &pool.pool, x.data(), x.size(), nullptr, &out, nullptr);
		if (st == SSLM_OK && out) sslm_prefix_release(out);
		return std::pair<sslm_status, bool>(st, out == nullptr);
	};
	const Spx1 gv = Spx1Parse(gpu_blob, e.geo);
	CHECK_MSG(gv.backend == 1 && gv.vendor != 0 && gv.digest_ok && gv.size_exact,
	          "5.7 CPU pair: the GPU save is not a checked GPU-shaped blob (backend %u, vendor %#x)", gv.backend, gv.vendor);
	const auto r1 = cpu_restore(gpu_blob);
	CHECK_MSG(r1.first == SSLM_INVALID_ARGUMENT && r1.second, "8.7 / 5.7 CPU: a GPU blob given to the CPU restore: %d",
	          static_cast<int>(r1.first));
	const auto r2 = cpu_restore(Edit32(gpu_blob, kOffPatch, gv.patch + 1));
	CHECK_MSG(r2.first == SSLM_INVALID_ARGUMENT && r2.second, "5.7 CPU: step 4 + step 5 on a real GPU save: %d",
	          static_cast<int>(r2.first));
	// The control: the GPU blob relabelled CPU (backend 0, ids 0, digest recomputed) restores on the CPU
	// iff its bytes are a valid CPU prefix; recorded, not graded (4.6 [F3] decides CPU/GPU identity).
	const auto r3 = cpu_restore(Edit32(Edit32(Edit32(gpu_blob, kOffBackend, 0), kOffVendor, 0), kOffDevice, 0));
	std::printf("smoke: the GPU blob relabelled backend 0 restores on the CPU: %d\n", static_cast<int>(r3.first));
	// The byte-identity smoke (4.6's shape): equal outside [4, 60) (backend, producer, digest).
	const bool same_size = cpu_blob.size() == gpu_blob.size();
	size_t diff = 0, first = SIZE_MAX;
	if (same_size)
		for (size_t i = 0; i < cpu_blob.size(); ++i)
			if ((i < 4 || i >= 60) && cpu_blob[i] != gpu_blob[i]) {
				++diff;
				if (first == SIZE_MAX) first = i;
			}
	std::printf("smoke 4.6-shape: CPU blob %zu bytes, GPU blob %zu bytes, %zu bytes differ outside [4, 60) (first at %zu)\n",
	            cpu_blob.size(), gpu_blob.size(), diff, first == SIZE_MAX ? size_t{0} : first);
	CHECK_MSG(same_size, "smoke: the CPU and GPU blobs of the same prefix differ in size");
}

void Cells(const Env& e) {
	Rig rig(e, kPool);
	if (!rig.pool) return;
	const std::vector<int32_t> t40 = PrefixTokens(kLen, 1), t1 = PrefixTokens(1, 2);
	SslmGpuPrefixHandle* p40 = Prefix(rig, t40);
	SslmGpuPrefixHandle* p1 = Prefix(rig, t1);
	SslmGpuPrefixHandle* pb = Prefix(rig, PrefixTokens(20, 4), 0);
	SslmGpuPrefixHandle* p0 = Prefix(rig, {});
	if (!p40 || !p1 || !pb || !p0) return;
	Capture(e, rig, p40, t40, -1, "F2 capture, root L = 40");
	Capture(e, rig, p1, t1, -1, "F2 capture, root L = 1");
	Capture(e, rig, pb, PrefixTokens(20, 4), 0, "F2 capture, schema-bound L = 20");
	Capture(e, rig, p0, {}, -1, "F2 capture, empty prefix");
	SaveRefusals(e, rig, p40);
	const std::vector<uint8_t> b40 = SavePrefix(e, p40), b1 = SavePrefix(e, p1), bb = SavePrefix(e, pb);
	if (b40.empty() || b1.empty()) return;
	// The refusals run against a pool of their own, every page free; nothing drawn by any of them: the
	// fill probe reads every page free after them all.
	{
		Rig rr(e, kRefusalPool);
		if (rr.pool) {
			ArgumentRefusals(e, rr, b40);
			RestoreRefusals(e, rr, b1, b40, bb);
			Pairs(e, rr, b40, b1);
			FreeIs(rr, kRefusalPool, "2.7: after every refusal, the refusal pool");
		}
	}
	// Round trip (6.4's shape on the GPU): save -> restore -> save, byte-identical, for root, empty, bound.
	for (auto* pr : {p40, p1, pb, p0}) {
		const std::vector<uint8_t> a = SavePrefix(e, pr);
		const RestoredPrefix r = RestorePrefix(rig, a);
		CHECK_MSG(r.status == GS::SSLM_OK && r.p, "round trip: restore %s", Name(r.status));
		if (!r.p) continue;
		const std::vector<uint8_t> c = SavePrefix(e, r.p);
		CHECK_MSG(a == c, "round trip: the re-saved blob differs (%zu vs %zu bytes)", a.size(), c.size());
	}
	CpuLegs(e, rig, b40, t40);
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) Cells(e);
	}
	return FinishSuite("pcg_cell_format");
}
