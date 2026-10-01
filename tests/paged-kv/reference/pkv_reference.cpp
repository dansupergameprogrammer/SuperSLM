// Paged-KV plan (rev 16.1) §8 and step R0: the reference generator.
//
// Built against a frozen tag (tools/build_paged_kv_reference.sh builds it from v1.11.0, the
// reference, and from v1.8.1, the last SSB4 writer), never against the tree under work. It runs
// every scenario of pkv_scenarios.h on one fixture and writes:
//   - OUT_REF: one header line (`# writer=<tag> fixture=<name> sha256=<file digest>`), then one
//     record per stage (pkv_scenarios.h's FormatRecord);
//   - into PIN_DIR, each pinned blob, zero-run encoded (pkv_zrl.h) as
//     `<tag>_<fixture stem>_<scenario>_<stage>.zrl`.
//
// Usage: pkv_reference [--only=SCENARIO[,SCENARIO...]] TAG FIXTURE.sslm OUT_REF [PIN_DIR]
//
// FIXTURE is one of the four pkv fixtures or one of the two real artifacts the C6 cells read
// (tests/paged-kv/c6_dim10_cohort.cpp; their geometry below is the cells' own). --only runs just the
// named scenarios: the box's real-artifact reference (tools/build_paged_kv_reference.ps1) runs
// `--only=cohort`, the one scenario the C6 cells read, since the others cost minutes at a real
// model's size and no real-artifact cell asks for them. Without --only every scenario runs, which is
// what tools/build_paged_kv_reference.sh does for the fixtures.

#include "pkv_scenarios.h"
#include "pkv_zrl.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <new>

namespace {

struct AlignedFree {
	void operator()(char* p) const { ::operator delete(p, std::align_val_t(64)); }
};

// Every pkv fixture has 2 layers and 2 KV heads (tools/gen_paged_kv_fixture.py's VARIANTS). The real
// artifacts' rows are the C6 cells' Geometry constants: Qwen2.5 0.5B (24 layers, 2 KV heads, head_dim
// 64) at cap 4096, and Qwen2.5 1.5B (28, 2, 128) at cap 32768.
bool GeometryFor(const std::string& stem, pkv::Geometry* g) {
	if (stem == "pkv_def") *g = {2, 2, 48, 4096};
	else if (stem == "pkv_qk") *g = {2, 2, 128, 4096};
	else if (stem == "pkv_odd") *g = {2, 2, 48, 4100};
	else if (stem == "pkv_32k") *g = {2, 2, 48, 32768};
	else if (stem == "qwen2.5-0.5b-instruct-cap4096-aex") *g = {24, 2, 64, 4096};
	else if (stem == "qwen2.5-1.5b-instruct") *g = {28, 2, 128, 32768};
	else return false;
	return true;
}

bool Selected(const std::string& only, const char* name) {
	if (only.empty()) return true;
	const std::string n = name;
	size_t p = 0;
	while (p <= only.size()) {
		const size_t c = only.find(',', p);
		if (only.substr(p, c == std::string::npos ? std::string::npos : c - p) == n) return true;
		if (c == std::string::npos) break;
		p = c + 1;
	}
	return false;
}

std::string Stem(const std::string& path) {
	const size_t slash = path.find_last_of("/\\");
	std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
	const size_t dot = base.rfind('.');
	return dot == std::string::npos ? base : base.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
	std::string only;
	if (argc > 1 && std::string(argv[1]).rfind("--only=", 0) == 0) {
		only = std::string(argv[1]).substr(7);
		--argc;
		++argv;
	}
	if (argc < 4 || argc > 5) {
		std::cerr << "usage: pkv_reference [--only=SCENARIO[,SCENARIO...]] TAG FIXTURE.sslm OUT_REF [PIN_DIR]\n";
		return 2;
	}
	const std::string tag = argv[1], fixture = argv[2], out_ref = argv[3];
	const std::string pin_dir = argc == 5 ? argv[4] : "";
	// Read whole into 64-byte-aligned memory (a real artifact is gigabytes: one sized read).
	std::ifstream f(fixture, std::ios::binary | std::ios::ate);
	const std::streamoff size = f ? static_cast<std::streamoff>(f.tellg()) : 0;
	if (size <= 0) {
		std::cerr << "cannot read " << fixture << "\n";
		return 1;
	}
	const std::unique_ptr<char, AlignedFree> buf(static_cast<char*>(::operator new(static_cast<size_t>(size), std::align_val_t(64))));
	f.seekg(0);
	if (!f.read(buf.get(), size)) {
		std::cerr << "cannot read " << fixture << "\n";
		return 1;
	}
	const char* const data = buf.get();
	const size_t bytes = static_cast<size_t>(size);
	pkv::Driver d;
	const std::string stem = Stem(fixture);
	if (!GeometryFor(stem, &d.geo)) {
		std::cerr << "unknown fixture " << stem << "\n";
		return 1;
	}
	if (sslm_model_map(data, bytes, &d.model) != SSLM_OK) {
		std::cerr << "map failed\n";
		return 1;
	}
	// The row extractor addresses the block by the geometry above; a model whose block is not
	// cap * layers * 2 * kv_heads * head_dim bytes is not the model that geometry describes.
	const size_t want_block = static_cast<size_t>(d.geo.context_cap) * d.geo.layers * 2u * d.geo.kv_heads * d.geo.head_dim;
	if (sslm_kv_block_size(d.model) != want_block) {
		std::cerr << stem << ": kv block " << sslm_kv_block_size(d.model) << " bytes, the stated geometry gives " << want_block
		          << "\n";
		sslm_model_unmap(d.model);
		return 1;
	}
	size_t n = 0;
	const pkv::Scenario* sc = pkv::Scenarios(&n);
	size_t known = 0;
	for (size_t i = 0; i < n; ++i) known += Selected(only, sc[i].name) ? 1 : 0;
	if (known == 0) {
		std::cerr << "--only=" << only << " names no scenario\n";
		sslm_model_unmap(d.model);
		return 2;
	}
	std::ofstream out(out_ref, std::ios::binary);
	out << "# writer=" << tag << " fixture=" << stem << " sha256="
	    << pkv::Sha(reinterpret_cast<const uint8_t*>(data), bytes) << "\n";
	int failures = 0;
	for (size_t i = 0; i < n; ++i) {
		if (!Selected(only, sc[i].name)) continue;
		d.fail.clear();
		d.records.clear();
		d.pending.clear();
		std::vector<pkv::Pinned> pins;
		sc[i].run(d, &pins);
		for (const pkv::Record& r : d.records) out << pkv::FormatRecord(sc[i].name, r) << "\n";
		if (!d.fail.empty()) {
			out << sc[i].name << " FAILED " << d.fail << "\n";
			std::cerr << stem << " " << sc[i].name << ": " << d.fail << "\n";
			++failures;
		}
		if (!pin_dir.empty())
			for (const pkv::Pinned& p : pins) {
				const std::string path =
				    pin_dir + "/" + tag + "_" + stem + "_" + sc[i].name + "_" + p.stage + ".zrl";
				const std::vector<uint8_t> enc = pkv::ZrlEncode(p.blob);
				std::ofstream pf(path, std::ios::binary);
				pf.write(reinterpret_cast<const char*>(enc.data()), static_cast<std::streamsize>(enc.size()));
			}
	}
	sslm_model_unmap(d.model);
	return failures ? 1 : 0;
}
