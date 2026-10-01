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
// Usage: pkv_reference TAG FIXTURE.sslm OUT_REF [PIN_DIR]

#include "pkv_scenarios.h"
#include "pkv_zrl.h"

#include <fstream>
#include <iostream>
#include <iterator>

namespace {

// Every pkv fixture has 2 layers and 2 KV heads (tools/gen_paged_kv_fixture.py's VARIANTS).
bool GeometryFor(const std::string& stem, pkv::Geometry* g) {
	if (stem == "pkv_def") *g = {2, 2, 48, 4096};
	else if (stem == "pkv_qk") *g = {2, 2, 128, 4096};
	else if (stem == "pkv_odd") *g = {2, 2, 48, 4100};
	else if (stem == "pkv_32k") *g = {2, 2, 48, 32768};
	else return false;
	return true;
}

std::string Stem(const std::string& path) {
	const size_t slash = path.find_last_of("/\\");
	std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
	const size_t dot = base.rfind('.');
	return dot == std::string::npos ? base : base.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 4 || argc > 5) {
		std::cerr << "usage: pkv_reference TAG FIXTURE.sslm OUT_REF [PIN_DIR]\n";
		return 2;
	}
	const std::string tag = argv[1], fixture = argv[2], out_ref = argv[3];
	const std::string pin_dir = argc == 5 ? argv[4] : "";
	std::ifstream f(fixture, std::ios::binary);
	std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
	if (bytes.empty()) {
		std::cerr << "cannot read " << fixture << "\n";
		return 1;
	}
	pkv::Driver d;
	const std::string stem = Stem(fixture);
	if (!GeometryFor(stem, &d.geo)) {
		std::cerr << "unknown fixture " << stem << "\n";
		return 1;
	}
	if (sslm_model_map(bytes.data(), bytes.size(), &d.model) != SSLM_OK) {
		std::cerr << "map failed\n";
		return 1;
	}
	std::ofstream out(out_ref, std::ios::binary);
	out << "# writer=" << tag << " fixture=" << stem << " sha256="
	    << pkv::Sha(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()) << "\n";
	int failures = 0;
	size_t n = 0;
	const pkv::Scenario* sc = pkv::Scenarios(&n);
	for (size_t i = 0; i < n; ++i) {
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
