// Precached-prefixes plan, cell 9.5 [E5 pin, F4 run] (rev 1.18.14, section 15.34 (2)): the 1.14.0 SPX1 pin generator.
//
// Section 8: "At E5, one 1.14.0 SPX1 blob per geometry is pinned for 9.5". F4 runs 9.5 on 1.16.0 against these blobs:
// each must be refused there with SSLM_RESTORE_PRODUCER_MISMATCH, by its producer. Section 15.34 names the geometries
// (pkv_def, pkv_odd, pkv_32k, pkv_qk: the E2/E3 cells' parse paths) and fixes three things this tool does per fixture:
//   - B is taken from the rule, not written down: sslm_kv_page_positions(model) must equal PageRuleB(D, cap, elem),
//     computed here from the fixture's geometry, and the B read is recorded. (The expected readings are 128, 256, 256
//     and 64; nothing here compares against those numbers.)
//   - The pinned prefix's length L is not a multiple of B, so each blob carries a partial last page.
//   - The manifest records each blob's SHA-256, the fixture's SHA-256, B, L and the build commit.
//
// Built against the 1.14.0 release tree (tools/build_precache_9p5_pins.sh builds it, and checks each blob's producer
// independently with tests/precache/check_version_values.py --expect 1.14.0 --blob). Any failed call or assertion fails
// the run (exit 1) and writes no manifest line for that fixture.
//
// Usage: pin_9p5_spx1 COMMIT FIXTURE_DIR OUT_DIR
//   writes OUT_DIR/1.14.0_<stem>.spx1 per fixture and OUT_DIR/manifest.txt

#include "pc_seq_saves.h"
#include "superslm/sha256.h"
#include "superslm/sslm_version.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct AlignedFree {
	void operator()(void* p) const { ::operator delete(p, std::align_val_t(64)); }
};

std::string Sha(const uint8_t* d, size_t n) {
	static const char* k = "0123456789abcdef";
	uint8_t out[32];
	superslm::Sha256Hash(d, n, out);
	std::string s;
	for (uint8_t b : out) {
		s.push_back(k[b >> 4]);
		s.push_back(k[b & 15]);
	}
	return s;
}

// The page-size rule (paged-KV section 3.1), written here independently of the engine: S = 4 KiB when D <= 64 and
// cap <= 4,096, else 8 KiB; B is the smallest power of two >= 16 with B * D * elem >= S, and the cap when that exceeds it.
int64_t PageRuleB(uint64_t D, uint64_t cap, uint64_t elem) {
	const uint64_t S = (D <= 64 && cap <= 4096) ? 4096 : 8192;
	uint64_t B = 16;
	while (B * D * elem < S) B *= 2;
	return static_cast<int64_t>(B > cap ? cap : B);
}

// The four geometries of section 15.34 (2), as tools/gen_paged_kv_fixture.py generates them and
// tests/paged-kv/pkv_common.h's GeometryForStem records them: head_dim and context_cap; int8 K/V (elem 1).
struct Geometry {
	const char* stem;
	uint64_t head_dim;
	uint64_t context_cap;
	uint64_t elem;
};
constexpr Geometry kGeometries[] = {
    {"pkv_def", 48, 4096, 1},
    {"pkv_odd", 48, 4100, 1},
    {"pkv_32k", 48, 32768, 1},
    {"pkv_qk", 128, 4096, 1},
};

// L = B + kTail: past one full page and short of the second, so the last page is partial whatever B the rule gives
// (kTail < 16 <= B).
constexpr int64_t kTail = 13;

// The writer's version, from sslm_version.h. The pin is 1.14.0's by definition, so another version is refused.
std::string Version() {
	return std::to_string(SUPERSLM_VERSION_MAJOR) + "." + std::to_string(SUPERSLM_VERSION_MINOR) + "." +
	       std::to_string(SUPERSLM_VERSION_PATCH);
}
constexpr const char* kPinVersion = "1.14.0";

bool ReadFile(const std::string& path, std::unique_ptr<void, AlignedFree>* buf, size_t* n) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	const std::streamoff size = f ? static_cast<std::streamoff>(f.tellg()) : 0;
	if (size <= 0) return false;
	*n = static_cast<size_t>(size);
	buf->reset(::operator new(*n, std::align_val_t(64)));
	f.seekg(0);
	return static_cast<bool>(f.read(static_cast<char*>(buf->get()), size));
}

// One fixture: map, assert the rule's B, prefill a budget prefix of L tokens, freeze, save, write the blob. Returns the
// manifest line, or an empty string with `fail` set.
std::string PinOne(const Geometry& g, const std::string& fixture_dir, const std::string& out_dir, std::string* fail) {
	const std::string path = fixture_dir + "/" + g.stem + ".sslm";
	std::unique_ptr<void, AlignedFree> fx;
	size_t fx_n = 0;
	if (!ReadFile(path, &fx, &fx_n)) {
		*fail = "cannot read " + path;
		return {};
	}
	const std::string fx_sha = Sha(static_cast<const uint8_t*>(fx.get()), fx_n);
	pcsave::Run r;
	if (!r.Ok(sslm_model_map(fx.get(), fx_n, &r.model), "model_map")) {
		*fail = r.fail;
		return {};
	}
	std::string line;
	sslm_prefix p = nullptr;
	std::unique_ptr<void, AlignedFree> mem;
	std::vector<uint8_t> blob;
	const int64_t B = sslm_kv_page_positions(r.model);
	const int64_t rule = PageRuleB(g.head_dim, g.context_cap, g.elem);
	const int64_t L = B + kTail;
	if (B != rule) {
		r.Ok(SSLM_INVALID_ARGUMENT, ("sslm_kv_page_positions " + std::to_string(B) + " != PageRuleB " +
		                             std::to_string(rule)).c_str());
	} else if (L % B == 0 || L > static_cast<int64_t>(g.context_cap)) {
		r.Ok(SSLM_INVALID_ARGUMENT, ("L " + std::to_string(L) + " has no partial last page at B " +
		                             std::to_string(B)).c_str());
	} else {
		const uint32_t pages = 8;  // R(L) = ceil(L / B) + 1 = 3 at most, with room to spare
		const size_t pool_bytes = sslm_kv_page_size(r.model) * pages + sslm_kv_page_pool_overhead_size(r.model, pages);
		const size_t rounded = (pool_bytes + 63) / 64 * 64;
		mem.reset(::operator new(rounded, std::align_val_t(64)));
		std::memset(mem.get(), 0, rounded);
		size_t n = 0;
		if (r.Ok(sslm_kv_page_pool_create(r.model, mem.get(), rounded, pages, &r.pool), "page_pool_create") &&
		    r.Ok(sslm_prefix_begin_budgeted(r.model, &r.pool, static_cast<int32_t>(L), &p), "prefix_begin_budgeted") &&
		    pcsave::PrefillPrefix(r, p, pcsave::Prompt(95, static_cast<int32_t>(L))) &&
		    r.Ok(sslm_prefix_freeze(p), "prefix_freeze")) {
			const sslm_status q = sslm_prefix_save(p, nullptr, &n);
			if (q != SSLM_BUFFER_TOO_SMALL) r.Ok(q == SSLM_OK ? SSLM_INVALID_ARGUMENT : q, "prefix_save size query");
			else {
				blob.resize(n);
				if (r.Ok(sslm_prefix_save(p, blob.data(), &n), "prefix_save") && n != blob.size())
					r.Ok(SSLM_INVALID_ARGUMENT, "prefix_save wrote another size than it reported");
			}
		}
	}
	if (r.fail.empty() && (blob.size() < 4 || std::memcmp(blob.data(), "SPX1", 4) != 0))
		r.Ok(SSLM_INVALID_ARGUMENT, "the saved blob is not SPX1");
	if (r.fail.empty()) {
		const std::string name = Version() + "_" + g.stem + ".spx1";
		std::ofstream out(out_dir + "/" + name, std::ios::binary);
		out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
		if (!out) r.Ok(SSLM_INVALID_ARGUMENT, ("cannot write " + name).c_str());
		std::ostringstream m;
		m << g.stem << " fixture_sha256=" << fx_sha << " D=" << g.head_dim << " cap=" << g.context_cap
		  << " B=" << B << " L=" << L << " last_page=" << (L % B) << "/" << B << " blob=" << name
		  << " size=" << blob.size() << " blob_sha256=" << Sha(blob.data(), blob.size());
		line = m.str();
	}
	if (p) sslm_prefix_release(p);
	if (r.pool) sslm_kv_pool_destroy(r.pool);
	sslm_model_unmap(r.model);
	*fail = r.fail;
	return r.fail.empty() ? line : std::string();
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 4) {
		std::cerr << "usage: pin_9p5_spx1 COMMIT FIXTURE_DIR OUT_DIR\n";
		return 2;
	}
	const std::string commit = argv[1], fixture_dir = argv[2], out_dir = argv[3];
	if (Version() != kPinVersion) {
		std::cerr << "built at " << Version() << ", not " << kPinVersion << ": 9.5's pin is the release build's (after R1)\n";
		return 1;
	}
	std::ostringstream manifest;
	manifest << "# plan 9.5 pins (rev 1.18.14, section 15.34): writer=" << Version() << " commit=" << commit
	         << " rule=PageRuleB(D,cap,1) L=B+" << kTail << "\n";
	int failures = 0;
	for (const Geometry& g : kGeometries) {
		std::string fail;
		const std::string line = PinOne(g, fixture_dir, out_dir, &fail);
		if (line.empty()) {
			std::cerr << g.stem << ": " << fail << "\n";
			++failures;
			continue;
		}
		std::cout << line << "\n";
		manifest << line << "\n";
	}
	if (failures) return 1;
	std::ofstream m(out_dir + "/manifest.txt", std::ios::binary);
	m << manifest.str();
	return m ? 0 : 1;
}
