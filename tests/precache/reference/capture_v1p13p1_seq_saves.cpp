// Precached-prefixes plan, cell 9.9 [E5] (rev 1.18.1, audit G14 (c); E5 row: "Capture and pin 9.9's ...
// sequence saves on a v1.13.1 build (d697927)"): the capture generator.
//
// Built against the frozen v1.13.1 tree, never against the tree under work
// (tools/build_precache_v1p13p1_captures.sh builds it, as tools/build_paged_kv_reference.sh builds
// pkv_reference.cpp from its tags). It maps pkv_def at the production page rule (no test-only pin), runs
// each case of pc_seq_saves.h in a fresh page pool, saves it, then decodes the saved state's 64-token
// continuation on that same 1.13.1 build, and writes:
//   - OUT_REF: `# writer=<tag> commit=<sha> fixture=pkv_def sha256=<file digest> B=<page positions>`, then one
//     line per case: `<case> magic=<4cc> size=<n> blob=<sha256> tokens=<64 step outcomes>` (pc_seq_saves.h's Continue);
//   - into PIN_DIR, each blob zero-run encoded (tests/paged-kv/reference/pkv_zrl.h) as
//     `<tag>_pkv_def_<case>.zrl`.
// Any failing call fails the run (exit 1) and writes no record for that case.
//
// Usage: capture_v1p13p1_seq_saves TAG COMMIT pkv_def.sslm OUT_REF PIN_DIR

#include "pc_seq_saves.h"
#include "../../paged-kv/reference/pkv_zrl.h"
#include "superslm/sha256.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <string>

namespace {

struct AlignedFree {
	void operator()(void* p) const { ::operator delete(p, std::align_val_t(64)); }
};

std::string Hex(const uint8_t* d, size_t n) {
	static const char* k = "0123456789abcdef";
	std::string s;
	for (size_t i = 0; i < n; ++i) {
		s.push_back(k[d[i] >> 4]);
		s.push_back(k[d[i] & 15]);
	}
	return s;
}

std::string Sha(const uint8_t* d, size_t n) {
	uint8_t out[32];
	superslm::Sha256Hash(d, n, out);
	return Hex(out, 32);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 6) {
		std::cerr << "usage: capture_v1p13p1_seq_saves TAG COMMIT pkv_def.sslm OUT_REF PIN_DIR\n";
		return 2;
	}
	const std::string tag = argv[1], commit = argv[2], fixture = argv[3], out_ref = argv[4], pin_dir = argv[5];
	std::ifstream f(fixture, std::ios::binary | std::ios::ate);
	const std::streamoff size = f ? static_cast<std::streamoff>(f.tellg()) : 0;
	if (size <= 0) {
		std::cerr << "cannot read " << fixture << "\n";
		return 1;
	}
	const size_t bytes = static_cast<size_t>(size);
	std::unique_ptr<void, AlignedFree> buf(::operator new(bytes, std::align_val_t(64)));
	f.seekg(0);
	if (!f.read(static_cast<char*>(buf.get()), size)) {
		std::cerr << "cannot read " << fixture << "\n";
		return 1;
	}
	sslm_model model = nullptr;
	if (sslm_model_map(buf.get(), bytes, &model) != SSLM_OK) {
		std::cerr << "map failed\n";
		return 1;
	}
	std::ofstream out(out_ref, std::ios::binary);
	out << "# writer=" << tag << " commit=" << commit << " fixture=pkv_def sha256="
	    << Sha(static_cast<const uint8_t*>(buf.get()), bytes) << " B=" << sslm_kv_page_positions(model) << "\n";
	int failures = 0;
	for (size_t i = 0; i < pcsave::kCaseCount; ++i) {
		const pcsave::Case& c = pcsave::kCases[i];
		const uint32_t pages = pcsave::kPoolPages;
		const size_t pool_bytes = sslm_kv_page_size(model) * pages + sslm_kv_page_pool_overhead_size(model, pages);
		const size_t rounded = (pool_bytes + 63) / 64 * 64;
		std::unique_ptr<void, AlignedFree> mem(::operator new(rounded, std::align_val_t(64)));
		std::memset(mem.get(), 0, rounded);
		pcsave::Run r;
		r.model = model;
		sslm_seq seq = nullptr;
		sslm_prefix parent = nullptr, child = nullptr;
		std::vector<uint8_t> blob;
		std::vector<int32_t> tokens;
		if (r.Ok(sslm_kv_page_pool_create(model, mem.get(), rounded, pages, &r.pool), "page_pool_create") &&
		    pcsave::Build(r, c, &seq, &parent, &child) && pcsave::SaveSeq(r, seq, &blob)) {
			if (blob.size() < 4 || std::memcmp(blob.data(), c.magic, 4) != 0)
				r.Ok(SSLM_INVALID_ARGUMENT, "the saved magic is not the case's");
			else
				tokens = pcsave::Continue(r, seq, pcsave::kContinue);
		}
		if (seq) sslm_seq_release(seq);
		if (child) sslm_prefix_release(child);
		if (parent) sslm_prefix_release(parent);
		if (r.pool) sslm_kv_pool_destroy(r.pool);
		if (!r.fail.empty()) {
			std::cerr << c.name << ": " << r.fail << "\n";
			out << c.name << " FAILED " << r.fail << "\n";
			++failures;
			continue;
		}
		out << c.name << " magic=" << std::string(reinterpret_cast<const char*>(blob.data()), 4)
		    << " size=" << blob.size() << " blob=" << Sha(blob.data(), blob.size())
		    << " tokens=";
		for (size_t t = 0; t < tokens.size(); ++t) out << (t ? "," : "") << tokens[t];
		out << "\n";
		const std::vector<uint8_t> enc = pkv::ZrlEncode(blob);
		std::ofstream pin(pin_dir + "/" + tag + "_pkv_def_" + c.name + ".zrl", std::ios::binary);
		pin.write(reinterpret_cast<const char*>(enc.data()), static_cast<std::streamsize>(enc.size()));
		if (!pin) {
			std::cerr << c.name << ": cannot write its pin\n";
			++failures;
		}
	}
	sslm_model_unmap(model);
	return failures ? 1 : 0;
}
