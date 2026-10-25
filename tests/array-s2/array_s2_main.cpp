// array_s2_main.cpp -- S2 runtime harness for JSON-array schemas on a real artifact (CPU and GPU).
// Reads the fixtures tests/array-s2/fixtures.py writes and prints line-oriented results that
// fixtures.py post-processes. Usage:
//   array_s2 map     <base.sslm> <schema.sslm> <schema>
//   array_s2 walks   <schema.sslm> <schema> <fixtures> <cpu|gpu>
//   array_s2 gen     <schema.sslm> <schema> <fixtures> <cpu|gpu> <plain|canon|exact|full> [row ...]
//   array_s2 restore <schema.sslm> <schema> <fixtures> <gen file> <save plan> <cpu|gpu> [row ...]
// Exit status 0 when every check passed, 1 on a failed check, 2 on a usage or setup error.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#include "array_s2_shared.h"

namespace {

constexpr int32_t kMaxTokens = 400;
constexpr int32_t kMaxChunk = 512;

struct Record {
	std::string kind, name;
	std::vector<int32_t> ids;
};

std::vector<uint8_t> ReadFile(const char* path) {
	std::ifstream f(path, std::ios::binary);
	return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// `kind name count id ...` (fixtures) or `gen backend row end count id ...` (gen output).
std::vector<Record> ReadRecords(const char* path) {
	std::vector<Record> out;
	std::ifstream f(path);
	std::string line;
	while (std::getline(f, line)) {
		std::istringstream in(line);
		Record r;
		int32_t n = 0;
		if (!(in >> r.kind >> r.name >> n)) continue;
		r.ids.resize(static_cast<size_t>(n));
		for (auto& id : r.ids) in >> id;
		out.push_back(std::move(r));
	}
	return out;
}

std::map<std::string, std::vector<int32_t>> ReadGen(const char* path, const std::string& backend) {
	std::map<std::string, std::vector<int32_t>> out;
	std::ifstream f(path);
	std::string line;
	while (std::getline(f, line)) {
		std::istringstream in(line);
		std::string kind, b, row, end;
		int32_t n = 0;
		if (!(in >> kind >> b >> row >> end >> n) || kind != "gen" || b != backend) continue;
		std::vector<int32_t> ids(static_cast<size_t>(n));
		for (auto& id : ids) in >> id;
		out[row] = std::move(ids);
	}
	return out;
}

void PrintIds(const std::vector<int32_t>& ids) {
	for (int32_t id : ids) std::printf(" %d", id);
	std::printf("\n");
}

bool Wanted(const std::vector<std::string>& rows, const std::string& row) {
	if (rows.empty()) return true;
	for (const auto& r : rows) {
		if (r == row) return true;
	}
	return false;
}

int Map(const char* base_path, const char* schema_path, const std::string& schema) {
	int failures = 0;
	for (const char* which : {"base", "schema"}) {
		const bool is_schema = std::string(which) == "schema";
		const auto bytes = ReadFile(is_schema ? schema_path : base_path);
		MapResult cm;
		CpuSession* c = CpuOpen(bytes.data(), bytes.size(), is_schema ? schema : "", false, kMaxChunk, &cm);
		std::printf("map cpu %s ok=%d ms=%.1f %s\n", which, cm.ok, cm.map_ms, cm.error.c_str());
		failures += !cm.ok;
		CpuClose(c);
		MapResult gm;
		GpuSession* g = GpuOpen(bytes.data(), bytes.size(), is_schema ? schema : "", &gm);
		std::printf("map gpu %s ok=%d ms=%.1f vram_delta=%lld %s\n", which, gm.ok, gm.map_ms,
		            static_cast<long long>(gm.vram_delta_bytes), gm.error.c_str());
		failures += !gm.ok;
		GpuClose(g);
	}
	return failures ? 1 : 0;
}

int Walks(const std::vector<uint8_t>& bytes, const std::string& schema, const std::vector<Record>& recs,
          const std::string& backend) {
	MapResult m;
	CpuSession* c = nullptr;
	GpuSession* g = nullptr;
	if (backend == "cpu") c = CpuOpen(bytes.data(), bytes.size(), schema, false, kMaxChunk, &m);
	else g = GpuOpen(bytes.data(), bytes.size(), schema, &m);
	if (!m.ok) {
		std::printf("setup failed: %s\n", m.error.c_str());
		return 2;
	}
	int failures = 0;
	for (const auto& r : recs) {
		if (r.kind != "walk" && r.kind != "reject") continue;
		const auto t0 = std::chrono::steady_clock::now();
		const WalkResult w = c ? CpuWalk(c, r.ids) : GpuWalk(g, r.ids);
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		bool pass;
		if (r.kind == "walk") {
			pass = w.refused_at < 0 && w.admitted == static_cast<int32_t>(r.ids.size()) && w.accepting_end == 1;
		} else {
			const int32_t expect = std::stoi(r.name.substr(r.name.find('@') + 1));
			pass = w.refused_at == expect && w.admitted == expect && w.refusal == "unreachable";
		}
		failures += !pass;
		std::printf("%s %s %s %s tokens=%zu admitted=%d refused_at=%d refusal=%s accepting_end=%d ms=%.0f",
		            pass ? "PASS" : "FAIL", backend.c_str(), r.kind.c_str(), r.name.c_str(), r.ids.size(),
		            w.admitted, w.refused_at, w.refusal.empty() ? "-" : w.refusal.c_str(), w.accepting_end, ms);
		if (!w.states.empty()) {
			std::printf(" states");
			for (uint32_t st : w.states) std::printf(" %u", st);
		}
		std::printf("\n");
	}
	CpuClose(c);
	GpuClose(g);
	return failures ? 1 : 0;
}

int Gen(const std::vector<uint8_t>& bytes, const std::string& schema, const std::vector<Record>& recs,
        const std::string& backend, const std::string& variant, const std::vector<std::string>& rows) {
	GenMode mode = GenMode::Plain;
	if (variant == "canon") mode = GenMode::JumpCanonical;
	else if (variant == "exact") mode = GenMode::JumpExact;
	else if (variant != "plain" && variant != "full") return 2;
	if (backend == "gpu" && variant != "plain") return 2;
	MapResult m;
	CpuSession* c = nullptr;
	GpuSession* g = nullptr;
	if (backend == "cpu") c = CpuOpen(bytes.data(), bytes.size(), schema, variant == "full", kMaxChunk, &m);
	else g = GpuOpen(bytes.data(), bytes.size(), schema, &m);
	if (!m.ok) {
		std::printf("# setup failed: %s\n", m.error.c_str());
		return 2;
	}
	const std::string label = variant == "plain" ? backend : backend + "_" + variant;
	int failures = 0;
	for (const auto& r : recs) {
		if (r.kind != "prompt" || !Wanted(rows, r.name)) continue;
		const GenResult res = c ? CpuGen(c, r.ids, kMaxTokens, mode) : GpuGen(g, r.ids, kMaxTokens);
		failures += res.end != "dead_end" || res.accepting != 1;
		std::printf("# %s %s end=%s accepting=%d tokens=%zu forced=%lld jump_calls=%lld ms=%.0f ms_per_token=%.1f\n",
		            label.c_str(), r.name.c_str(), res.end.c_str(), res.accepting, res.ids.size(),
		            static_cast<long long>(res.forced), static_cast<long long>(res.jump_calls), res.ms,
		            res.ids.empty() ? 0.0 : res.ms / static_cast<double>(res.ids.size()));
		std::printf("gen %s %s %s %zu", label.c_str(), r.name.c_str(), res.end.c_str(), res.ids.size());
		PrintIds(res.ids);
		std::fflush(stdout);
	}
	CpuClose(c);
	GpuClose(g);
	return failures ? 1 : 0;
}

int Restore(const std::vector<uint8_t>& bytes, const std::string& schema, const std::vector<Record>& recs,
            const char* gen_path, const char* plan_path, const std::string& backend,
            const std::vector<std::string>& rows) {
	const auto refs = ReadGen(gen_path, backend);
	// Save plan lines: `save row index kind`.
	std::map<std::string, std::vector<std::pair<int32_t, std::string>>> plan;
	{
		std::ifstream f(plan_path);
		std::string kind, row, what;
		int32_t idx = 0;
		while (f >> kind >> row >> idx >> what) plan[row].push_back({idx, what});
	}
	MapResult m;
	CpuSession* c = nullptr;
	GpuSession* g = nullptr;
	if (backend == "cpu") c = CpuOpen(bytes.data(), bytes.size(), schema, false, kMaxChunk, &m);
	else g = GpuOpen(bytes.data(), bytes.size(), schema, &m);
	if (!m.ok) {
		std::printf("# setup failed: %s\n", m.error.c_str());
		return 2;
	}
	int failures = 0;
	for (const auto& r : recs) {
		if (r.kind != "prompt" || !Wanted(rows, r.name) || !plan.count(r.name) || !refs.count(r.name)) continue;
		const auto& ref = refs.at(r.name);
		std::vector<int32_t> after;
		for (const auto& p : plan[r.name]) after.push_back(p.first);
		std::string error;
		const auto results = c ? CpuRestores(c, r.ids, ref, after, kMaxTokens, &error)
		                       : GpuRestores(g, r.ids, ref, after, kMaxTokens, &error);
		if (!error.empty()) {
			++failures;
			std::printf("FAIL %s %s setup: %s\n", backend.c_str(), r.name.c_str(), error.c_str());
			continue;
		}
		for (size_t k = 0; k < results.size(); ++k) {
			const RestoreResult& rr = results[k];
			const std::vector<int32_t> want(ref.begin() + rr.after + 1, ref.end());
			int32_t diff = -1;
			for (size_t j = 0; j < std::max(want.size(), rr.continuation.size()); ++j) {
				if (j >= want.size() || j >= rr.continuation.size() || want[j] != rr.continuation[j]) {
					diff = static_cast<int32_t>(j);
					break;
				}
			}
			const bool pass = diff < 0 && rr.end == "dead_end";
			failures += !pass;
			std::printf("%s %s %s after=%d kind=%s blob=%zu continuation=%zu end=%s", pass ? "PASS" : "FAIL",
			            backend.c_str(), r.name.c_str(), rr.after, plan[r.name][k].second.c_str(), rr.blob_bytes,
			            rr.continuation.size(), rr.end.c_str());
			if (diff >= 0) {
				std::printf(" first_diff=%d want=%d got=%d", rr.after + 1 + diff,
				            diff < static_cast<int32_t>(want.size()) ? want[static_cast<size_t>(diff)] : -1,
				            diff < static_cast<int32_t>(rr.continuation.size()) ? rr.continuation[static_cast<size_t>(diff)] : -1);
			}
			std::printf("\n");
			std::fflush(stdout);
		}
	}
	CpuClose(c);
	GpuClose(g);
	return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) return 2;
	const std::string mode = argv[1];
	if (mode == "map" && argc == 5) return Map(argv[2], argv[3], argv[4]);
	if (argc < 6) return 2;
	const auto bytes = ReadFile(argv[2]);
	const std::string schema = argv[3];
	const auto recs = ReadRecords(argv[4]);
	if (mode == "walks") return Walks(bytes, schema, recs, argv[5]);
	if (mode == "gen" && argc >= 7) {
		std::vector<std::string> rows(argv + 7, argv + argc);
		return Gen(bytes, schema, recs, argv[5], argv[6], rows);
	}
	if (mode == "restore" && argc >= 8) {
		std::vector<std::string> rows(argv + 8, argv + argc);
		return Restore(bytes, schema, recs, argv[5], argv[6], argv[7], rows);
	}
	return 2;
}
