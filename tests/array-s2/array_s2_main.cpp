// array_s2_main.cpp -- S2 runtime harness for JSON-array schemas on a real artifact (CPU and GPU).
// Reads the fixtures tests/array-s2/fixtures.py writes and prints line-oriented results that
// fixtures.py post-processes. Usage:
//   array_s2 map     <base.sslm> <schema.sslm> <schema>
//   array_s2 walks   <schema.sslm> <schema> <fixtures> <cpu|gpu>
//   array_s2 gen     <schema.sslm> <schema> <fixtures> <cpu|gpu> <plain|canon|exact|full> [row ...]
//   array_s2 restore <schema.sslm> <schema> <fixtures> <gen file> <save plan> <cpu|gpu> [row ...]
//   array_s2 tfrestore <schema.sslm> <schema> <fixtures> <cpu|gpu> <prompt row> [doc ...]
//   array_s2 gputime <schema.sslm> <schema> <fixtures> <gen file> [row ...]
// `--state-zero` anywhere after the mode runs restore or tfrestore against the restore-at-state-zero
// mutant (every save blob's walk state overwritten with 0 before it is restored); the run is then
// expected to fail, and the caller asserts that it does.
// Exit status 0 when every check passed, 1 on a failed check (including a mode that checked
// nothing, a planned row with no reference, and a result count that differs from the plan), 2 on
// a usage or setup error.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#include "array_s2_shared.h"

namespace {

constexpr int32_t kMaxTokens = 400;
constexpr int32_t kMaxChunk = 512;
// Greedy tokens decoded from each teacher-forced save point, unbroken and restored.
constexpr int32_t kTfContinuation = 64;

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
	int checked = 0;
	for (const auto& r : recs) {
		if (r.kind != "walk" && r.kind != "reject") continue;
		++checked;
		const auto t0 = std::chrono::steady_clock::now();
		const WalkResult w = c ? CpuWalk(c, r.ids) : GpuWalk(g, r.ids);
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		// The accepting flag after every admitted token: 1 after a walk's last token, 0 everywhere else.
		int32_t bad_accepting = -1;
		for (size_t i = 0; i < w.accepting.size(); ++i) {
			const bool last = r.kind == "walk" && i + 1 == r.ids.size();
			if (w.accepting[i] != (last ? 1 : 0)) {
				bad_accepting = static_cast<int32_t>(i);
				break;
			}
		}
		const bool trace_ok = bad_accepting < 0 && w.accepting.size() == static_cast<size_t>(w.admitted) &&
		                      (backend != "gpu" || w.states.size() == static_cast<size_t>(w.admitted));
		bool pass;
		if (r.kind == "walk") {
			pass = w.refused_at < 0 && w.admitted == static_cast<int32_t>(r.ids.size()) && w.accepting_end == 1;
		} else {
			const int32_t expect = std::stoi(r.name.substr(r.name.find('@') + 1));
			pass = w.refused_at == expect && w.admitted == expect && w.refusal == "unreachable";
		}
		pass = pass && trace_ok;
		failures += !pass;
		std::printf("%s %s %s %s tokens=%zu admitted=%d refused_at=%d refusal=%s accepting_end=%d bad_accepting_at=%d ms=%.0f",
		            pass ? "PASS" : "FAIL", backend.c_str(), r.kind.c_str(), r.name.c_str(), r.ids.size(),
		            w.admitted, w.refused_at, w.refusal.empty() ? "-" : w.refusal.c_str(), w.accepting_end,
		            bad_accepting, ms);
		if (!w.states.empty()) {
			std::printf(" states");
			for (uint32_t st : w.states) std::printf(" %u", st);
		}
		std::printf("\n");
	}
	if (checked == 0) {
		std::printf("FAIL %s no walk or reject record in the fixtures\n", backend.c_str());
		++failures;
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
	int checked = 0;
	for (const auto& r : recs) {
		if (r.kind != "prompt" || !Wanted(rows, r.name)) continue;
		++checked;
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
	if (checked == 0 || (!rows.empty() && checked != static_cast<int>(rows.size()))) {
		std::printf("# FAIL %s generated %d prompts, wanted %zu\n", label.c_str(), checked,
		            rows.empty() ? size_t{1} : rows.size());
		++failures;
	}
	CpuClose(c);
	GpuClose(g);
	return failures ? 1 : 0;
}

// The first index where two traces' vectors differ (a missing element counts), or -1.
template <class T>
int32_t FirstDiff(const std::vector<T>& a, const std::vector<T>& b) {
	for (size_t j = 0; j < std::max(a.size(), b.size()); ++j) {
		if (j >= a.size() || j >= b.size() || a[j] != b[j]) return static_cast<int32_t>(j);
	}
	return -1;
}

template <class T>
std::vector<T> Tail(const std::vector<T>& v, int32_t from) {
	return from >= static_cast<int32_t>(v.size()) ? std::vector<T>() : std::vector<T>(v.begin() + from, v.end());
}

// Collects the first failed check of one save point as `name at=<position>`.
struct Verdict {
	std::string failed;
	template <class T>
	void Same(const char* name, const std::vector<T>& want, const std::vector<T>& got, int32_t base) {
		if (!failed.empty()) return;
		const int32_t d = FirstDiff(want, got);
		if (d < 0) return;
		const size_t k = static_cast<size_t>(d);
		failed = std::string(name) + " at=" + std::to_string(base + d) + " want=" +
		         (k < want.size() ? std::to_string(want[k]) : "none") + " got=" +
		         (k < got.size() ? std::to_string(got[k]) : "none");
	}
	void Is(const char* name, bool ok, const std::string& detail = "") {
		if (failed.empty() && !ok) failed = std::string(name) + (detail.empty() ? "" : " " + detail);
	}
};

// The accepting flag after every token of a complete document: 0, then 1 after the last token.
bool AcceptsOnlyAtEnd(const std::vector<int32_t>& accepting) {
	if (accepting.empty() || accepting.back() != 1) return false;
	return std::all_of(accepting.begin(), accepting.end() - 1, [](int32_t a) { return a == 0; });
}

void PrintResult(bool pass, const std::string& backend, const std::string& label, const RestoreResult& rr,
                 const Verdict& v) {
	std::printf("%s %s %s after=%d blob=%zu continuation=%zu end=%s", pass ? "PASS" : "FAIL", backend.c_str(),
	            label.c_str(), rr.after, rr.blob_bytes, rr.continuation.ids.size(),
	            rr.continuation.end.empty() ? "-" : rr.continuation.end.c_str());
	if (!rr.forced.end.empty()) std::printf(" forced=%zu", rr.forced.ids.size());
	if (!pass) std::printf(" check=%s", v.failed.c_str());
	std::printf("\n");
	std::fflush(stdout);
}

struct Session {
	CpuSession* c = nullptr;
	GpuSession* g = nullptr;
	MapResult m;
	Session(const std::vector<uint8_t>& bytes, const std::string& schema, const std::string& backend) {
		if (backend == "cpu") c = CpuOpen(bytes.data(), bytes.size(), schema, false, kMaxChunk, &m);
		else g = GpuOpen(bytes.data(), bytes.size(), schema, &m);
	}
	~Session() {
		CpuClose(c);
		GpuClose(g);
	}
};

int Restore(const std::vector<uint8_t>& bytes, const std::string& schema, const std::vector<Record>& recs,
            const char* gen_path, const char* plan_path, const std::string& backend,
            const std::vector<std::string>& rows, bool state_zero) {
	const bool gpu = backend == "gpu";
	const auto refs = ReadGen(gen_path, backend);
	// Save plan lines: `save row index kind`.
	std::map<std::string, std::vector<std::pair<int32_t, std::string>>> plan;
	{
		std::ifstream f(plan_path);
		std::string kind, row, what;
		int32_t idx = 0;
		while (f >> kind >> row >> idx >> what) plan[row].push_back({idx, what});
	}
	Session ss(bytes, schema, backend);
	if (!ss.m.ok) {
		std::printf("# setup failed: %s\n", ss.m.error.c_str());
		return 2;
	}
	int failures = 0;
	int checked = 0;
	std::map<std::string, bool> seen;
	for (const auto& r : recs) {
		if (r.kind != "prompt" || !Wanted(rows, r.name)) continue;
		seen[r.name] = true;
		if (!plan.count(r.name) || !refs.count(r.name)) {
			++failures;
			std::printf("FAIL %s %s setup: %s\n", backend.c_str(), r.name.c_str(),
			            !plan.count(r.name) ? "no save plan row" : "no reference row in the gen file");
			continue;
		}
		const auto& ref = refs.at(r.name);
		std::vector<int32_t> after;
		for (const auto& p : plan[r.name]) after.push_back(p.first);
		RestoreOptions opt;
		opt.max_tokens = kMaxTokens;
		opt.state_zero = state_zero;
		Trace unbroken;
		std::string error;
		const auto results = ss.c ? CpuRestores(ss.c, r.ids, ref, after, opt, &unbroken, &error)
		                          : GpuRestores(ss.g, r.ids, ref, after, opt, &unbroken, &error);
		if (!error.empty()) {
			++failures;
			std::printf("FAIL %s %s setup: %s\n", backend.c_str(), r.name.c_str(), error.c_str());
			continue;
		}
		// The unbroken run's own trace: the reference tokens, accepting only after the last one.
		{
			Verdict v;
			v.Same("unbroken_tokens", ref, unbroken.ids, 0);
			v.Is("unbroken_accepting", AcceptsOnlyAtEnd(unbroken.accepting));
			v.Is("unbroken_states", !gpu || unbroken.states.size() == unbroken.ids.size());
			const bool pass = v.failed.empty();
			failures += !pass;
			std::printf("%s %s %s trace tokens=%zu%s%s\n", pass ? "PASS" : "FAIL", backend.c_str(), r.name.c_str(),
			            unbroken.ids.size(), pass ? "" : " check=", v.failed.c_str());
		}
		if (results.size() != after.size()) {
			++failures;
			std::printf("FAIL %s %s setup: %zu results for %zu planned saves\n", backend.c_str(), r.name.c_str(),
			            results.size(), after.size());
		}
		for (size_t k = 0; k < results.size() && k < after.size(); ++k) {
			const RestoreResult& rr = results[k];
			const int32_t a = rr.after;
			++checked;
			Verdict v;
			v.Is("error", rr.error.empty(), rr.error);
			v.Is("after", a == after[k] && a >= 0 && a < static_cast<int32_t>(unbroken.accepting.size()));
			if (v.failed.empty()) {
				const size_t ai = static_cast<size_t>(a);
				v.Same("tokens", Tail(ref, a + 1), rr.continuation.ids, a + 1);
				v.Is("end", rr.continuation.end == "dead_end", rr.continuation.end);
				v.Is("restored_accepting", rr.restored_accepting == unbroken.accepting[ai],
				     "want=" + std::to_string(unbroken.accepting[ai]) + " got=" + std::to_string(rr.restored_accepting));
				if (gpu) {
					v.Is("restored_state", rr.restored_state == unbroken.states[ai],
					     "want=" + std::to_string(unbroken.states[ai]) + " got=" + std::to_string(rr.restored_state));
				}
				v.Same("accepting", Tail(unbroken.accepting, a + 1), rr.continuation.accepting, a + 1);
				if (gpu) v.Same("states", Tail(unbroken.states, a + 1), rr.continuation.states, a + 1);
			}
			const bool pass = v.failed.empty();
			failures += !pass;
			PrintResult(pass, backend, r.name + " kind=" + plan[r.name][k].second, rr, v);
		}
	}
	for (const auto& kv : plan) {
		if (Wanted(rows, kv.first) && !seen.count(kv.first)) {
			++failures;
			std::printf("FAIL %s %s setup: save plan row has no prompt in the fixtures\n", backend.c_str(),
			            kv.first.c_str());
		}
	}
	if (checked == 0) {
		++failures;
		std::printf("FAIL %s no save point was checked\n", backend.c_str());
	}
	return failures ? 1 : 0;
}

int TfRestore(const std::vector<uint8_t>& bytes, const std::string& schema, const std::vector<Record>& recs,
              const std::string& backend, const std::string& prompt_row, const std::vector<std::string>& docs,
              bool state_zero) {
	const bool gpu = backend == "gpu";
	const Record* prompt = nullptr;
	std::map<std::string, std::vector<int32_t>> doc_ids;
	std::map<std::string, std::vector<std::pair<int32_t, std::string>>> plan;
	for (const auto& r : recs) {
		if (r.kind == "prompt" && r.name == prompt_row) prompt = &r;
		if (r.kind == "tfdoc") doc_ids[r.name] = r.ids;
		if (r.kind == "tfsave" && r.ids.size() == 1) {
			const size_t colon = r.name.find(':');
			plan[r.name.substr(0, colon)].push_back({r.ids[0], r.name.substr(colon + 1)});
		}
	}
	if (!prompt) {
		std::printf("FAIL %s no prompt record %s\n", backend.c_str(), prompt_row.c_str());
		return 1;
	}
	Session ss(bytes, schema, backend);
	if (!ss.m.ok) {
		std::printf("# setup failed: %s\n", ss.m.error.c_str());
		return 2;
	}
	int failures = 0;
	int checked = 0;
	for (const auto& kv : plan) {
		const std::string& name = kv.first;
		if (!Wanted(docs, name)) continue;
		if (!doc_ids.count(name)) {
			++failures;
			std::printf("FAIL %s tf %s setup: no tfdoc record\n", backend.c_str(), name.c_str());
			continue;
		}
		const auto& doc = doc_ids.at(name);
		std::vector<int32_t> after;
		for (const auto& p : kv.second) after.push_back(p.first);
		RestoreOptions opt;
		opt.max_tokens = kTfContinuation;
		opt.state_zero = state_zero;
		Trace walk;
		std::string error;
		const auto results = ss.c ? CpuTfRestores(ss.c, prompt->ids, doc, after, opt, &walk, &error)
		                          : GpuTfRestores(ss.g, prompt->ids, doc, after, opt, &walk, &error);
		if (!error.empty()) {
			++failures;
			std::printf("FAIL %s tf %s setup: %s\n", backend.c_str(), name.c_str(), error.c_str());
			continue;
		}
		{
			Verdict v;
			v.Is("walk_end", walk.end == "forced", walk.end);
			v.Same("walk_tokens", doc, walk.ids, 0);
			v.Is("walk_accepting", AcceptsOnlyAtEnd(walk.accepting));
			v.Is("walk_states", !gpu || walk.states.size() == walk.ids.size());
			const bool pass = v.failed.empty();
			failures += !pass;
			std::printf("%s %s tf %s walk tokens=%zu%s%s", pass ? "PASS" : "FAIL", backend.c_str(), name.c_str(),
			            walk.ids.size(), pass ? "" : " check=", v.failed.c_str());
			if (gpu) {
				std::printf(" states");
				for (uint32_t st : walk.states) std::printf(" %u", st);
			}
			std::printf("\n");
			if (!pass) continue;
		}
		if (results.size() != after.size()) {
			++failures;
			std::printf("FAIL %s tf %s setup: %zu results for %zu planned saves\n", backend.c_str(), name.c_str(),
			            results.size(), after.size());
		}
		for (size_t k = 0; k < results.size() && k < after.size(); ++k) {
			const RestoreResult& rr = results[k];
			const int32_t a = rr.after;
			++checked;
			Verdict v;
			v.Is("error", rr.error.empty(), rr.error);
			v.Is("after", a == after[k] && a >= 0 && a + 1 < static_cast<int32_t>(doc.size()));
			if (v.failed.empty()) {
				const size_t ai = static_cast<size_t>(a);
				// The bulk-prefilled prefix reached the state the one-token walk reached.
				v.Is("saved_accepting", rr.saved_accepting == walk.accepting[ai]);
				if (gpu) {
					v.Is("saved_state", rr.saved_state == walk.states[ai],
					     "want=" + std::to_string(walk.states[ai]) + " got=" + std::to_string(rr.saved_state));
				}
				v.Is("restored_accepting", rr.restored_accepting == walk.accepting[ai],
				     "want=" + std::to_string(walk.accepting[ai]) + " got=" + std::to_string(rr.restored_accepting));
				if (gpu) {
					v.Is("restored_state", rr.restored_state == walk.states[ai],
					     "want=" + std::to_string(walk.states[ai]) + " got=" + std::to_string(rr.restored_state));
				}
				// The restored decode continues exactly as the unbroken one.
				v.Is("unbroken_end", rr.unbroken.end == "dead_end" || rr.unbroken.end == "max_tokens", rr.unbroken.end);
				v.Same("tokens", rr.unbroken.ids, rr.continuation.ids, a + 1);
				v.Same("accepting", rr.unbroken.accepting, rr.continuation.accepting, a + 1);
				if (gpu) {
					v.Is("unbroken_states", rr.unbroken.states.size() == rr.unbroken.ids.size());
					v.Same("states", rr.unbroken.states, rr.continuation.states, a + 1);
				}
				v.Is("end", rr.continuation.end == rr.unbroken.end, rr.continuation.end + " vs " + rr.unbroken.end);
				// The rest of the document, teacher-forced after a second restore, walks as the unbroken walk did.
				v.Is("forced_end", rr.forced.end == "forced", rr.forced.end);
				v.Same("forced_tokens", Tail(doc, a + 1), rr.forced.ids, a + 1);
				v.Same("forced_accepting", Tail(walk.accepting, a + 1), rr.forced.accepting, a + 1);
				if (gpu) v.Same("forced_states", Tail(walk.states, a + 1), rr.forced.states, a + 1);
			}
			const bool pass = v.failed.empty();
			failures += !pass;
			PrintResult(pass, backend, "tf " + name + ":" + kv.second[k].second, rr, v);
		}
	}
	for (const auto& d : docs) {
		if (!plan.count(d)) {
			++failures;
			std::printf("FAIL %s tf %s setup: no tfsave record\n", backend.c_str(), d.c_str());
		}
	}
	if (checked == 0) {
		++failures;
		std::printf("FAIL %s no teacher-forced save point was checked\n", backend.c_str());
	}
	return failures ? 1 : 0;
}

}  // namespace

namespace {

// GPU decode timing with SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, at the slice size
// ARRAY_S2_GPU_LAYERS_PER_SLICE sets (the box script runs it at one slice per token). Times run
// from the first decode call to the dead end and exclude the prompt prefill. Every stream must
// equal the plain GPU run in the gen file (default residency).
int GpuTime(const std::vector<uint8_t>& bytes, const std::string& schema, const std::vector<Record>& recs,
            const char* gen_path, const std::vector<std::string>& rows) {
	const auto refs = ReadGen(gen_path, "gpu");
	const char* env = std::getenv("ARRAY_S2_GPU_LAYERS_PER_SLICE");
	const std::string layers = env ? env : "1";
	GpuOptions opt;
	opt.head_on_device = true;
	MapResult m;
	GpuSession* g = GpuOpen(bytes.data(), bytes.size(), schema, &m, opt);
	if (!m.ok) {
		std::printf("# setup failed (head on device): %s\n", m.error.c_str());
		return 2;
	}
	int failures = 0;
	size_t tokens = 0;
	double ms = 0.0;
	int checked = 0;
	for (const auto& r : recs) {
		if (r.kind != "prompt" || !Wanted(rows, r.name)) continue;
		++checked;
		const GenResult res = GpuGen(g, r.ids, kMaxTokens);
		const bool have_ref = refs.count(r.name) != 0;
		const bool same = have_ref && res.ids == refs.at(r.name);
		const bool pass = same && res.end == "dead_end" && res.accepting == 1;
		failures += !pass;
		tokens += res.ids.size();
		ms += res.ms;
		std::printf("%s gputime head_on_device layers_per_slice=%s %s tokens=%zu ms=%.0f ms_per_token=%.1f end=%s "
		            "same_as_plain_gpu=%s\n",
		            pass ? "PASS" : "FAIL", layers.c_str(), r.name.c_str(), res.ids.size(), res.ms,
		            res.ids.empty() ? 0.0 : res.ms / static_cast<double>(res.ids.size()), res.end.c_str(),
		            !have_ref ? "no_reference" : (same ? "yes" : "no"));
		std::fflush(stdout);
	}
	if (checked == 0) {
		++failures;
		std::printf("FAIL gputime no prompt was timed\n");
	}
	std::printf("total gputime head_on_device layers_per_slice=%s prompts=%d tokens=%zu ms=%.0f ms_per_token=%.1f "
	            "tok_s=%.2f\n", layers.c_str(), checked, tokens, ms, tokens ? ms / static_cast<double>(tokens) : 0.0,
	            ms > 0 ? 1000.0 * static_cast<double>(tokens) / ms : 0.0);
	GpuClose(g);
	return failures ? 1 : 0;
}

}  // namespace

bool ZeroBlobWalkState(std::vector<uint8_t>* blob, bool gpu) {
	const size_t off = gpu ? 124 : 48;
	if (blob->size() < off + 4) return false;
	const uint8_t* m = blob->data();
	const bool ok = gpu ? (m[0] == 'S' && m[1] == 'L' && m[2] == 'M' && (m[3] == '5' || m[3] == '6'))
	                    : (m[0] == 'S' && m[1] == 'S' && m[2] == 'B' && (m[3] == '5' || m[3] == '6'));
	if (!ok) return false;
	std::fill(blob->begin() + static_cast<std::ptrdiff_t>(off), blob->begin() + static_cast<std::ptrdiff_t>(off + 4),
	          uint8_t{0});
	return true;
}

int main(int argc, char** argv) {
	std::vector<std::string> args(argv + 1, argv + argc);
	const auto flag = std::find(args.begin(), args.end(), "--state-zero");
	const bool state_zero = flag != args.end();
	if (state_zero) args.erase(flag);
	const size_t n = args.size();
	if (n < 1) return 2;
	const std::string mode = args[0];
	if (mode == "map" && n == 4) return Map(args[1].c_str(), args[2].c_str(), args[3]);
	if (n < 5) return 2;
	const auto bytes = ReadFile(args[1].c_str());
	const std::string schema = args[2];
	const auto recs = ReadRecords(args[3].c_str());
	if (mode == "walks" && !state_zero) return Walks(bytes, schema, recs, args[4]);
	if (mode == "gen" && n >= 6 && !state_zero) {
		std::vector<std::string> rows(args.begin() + 6, args.end());
		return Gen(bytes, schema, recs, args[4], args[5], rows);
	}
	if (mode == "restore" && n >= 7) {
		std::vector<std::string> rows(args.begin() + 7, args.end());
		return Restore(bytes, schema, recs, args[4].c_str(), args[5].c_str(), args[6], rows, state_zero);
	}
	if (mode == "gputime" && n >= 5 && !state_zero) {
		std::vector<std::string> rows(args.begin() + 5, args.end());
		return GpuTime(bytes, schema, recs, args[4].c_str(), rows);
	}
	if (mode == "tfrestore" && n >= 6) {
		std::vector<std::string> docs(args.begin() + 6, args.end());
		return TfRestore(bytes, schema, recs, args[4], args[5], docs, state_zero);
	}
	return 2;
}
