// Paged-KV plan (rev 16.2) step G1: the GPU red suite's in-process runner. It replaces the groups' draft
// runners (g1a_main_draft.cpp, g1b_main_draft.cpp, g1c_main_draft.cpp), whose rules it keeps:
//   - with no selector every registered cell runs, except g1b's ':' legs (exact name only);
//   - a selector is an id prefix ("6.5b", "3.6d/G3"), or '=' and an exact id ("=11.4/G3:zerofill");
//   - a selector that matches no registered cell fails the run (pkv_main.cpp's rule);
//   - a g1c skip fails the run unless --allow-skip;
//   - exit 0 only when at least one cell ran, every selector matched and every check passed.
// Which cells exist depends on the executable: pkvg_g2 .. pkvg_g5 (default engine; every in-process cell
// whose bracket is at or before the step), pkvg_alloc_g3 (allocation-fault engine: 11.4/G3),
// pkvg_bench_g3 (bench engine: 3.6d bench prefill), pkvg_firsttouch_g3 (allocation-fault engine and
// te425's hooks: 3.6d first touch). See build_paged_kv_gpu.bat.
//
// Usage: pkvg_<exe> [common flags] [group flags] [--list] [SELECTOR ...]
// Common flags (mapped onto every group that needs them):
//   --pkv-fixtures=DIR   the fixture directory (g1a fixtures; g1b --tiny and g1c --model default to
//                        DIR/pkv_def.sslm). Defaults to env SUPERSLM_PAGED_KV_FIXTURE_DIR.
//   --pkv-reference=DIR  tests/paged-kv/reference (default: that path relative to the working directory)
//   --model05=PATH (alias --m05=)   --model15=PATH (alias --m15=)
// Group flags are listed by --help. Run once per certified GPU (SSLM_GPU_ADAPTER_INDEX=0, then 1) with
// SSLM_GPU_ENABLE_DEBUG_LAYER=1.
#include "pkvg_runner.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

bool Take(const std::string& a, const char* flag, std::string* out) {
	const size_t n = std::strlen(flag);
	if (a.compare(0, n, flag) != 0) return false;
	*out = a.substr(n);
	return true;
}

bool Matches(const pkvg::Group& g, size_t i, const std::string& sel) {
	const char* id = g.id(i);
	if (!sel.empty() && sel[0] == '=') return sel.compare(1, std::string::npos, id) == 0;
	return !g.exact_only(i) && std::strncmp(id, sel.c_str(), sel.size()) == 0;
}

}  // namespace

int main(int argc, char** argv) {
	std::vector<const pkvg::Group*> groups = pkvg::Groups();
	std::sort(groups.begin(), groups.end(),
	          [](const pkvg::Group* a, const pkvg::Group* b) { return std::strcmp(a->name, b->name) < 0; });

	pkvg::CommonOptions common;
	std::vector<std::string> sel;
	bool list = false;
	int bad_flags = 0;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (Take(a, "--pkv-fixtures=", &common.fixture_dir) || Take(a, "--pkv-reference=", &common.reference_dir) ||
		    Take(a, "--model05=", &common.model05) || Take(a, "--m05=", &common.model05) ||
		    Take(a, "--model15=", &common.model15) || Take(a, "--m15=", &common.model15))
			continue;
		if (a == "--list") {
			list = true;
			continue;
		}
		if (a == "--help") {
			std::printf("common: --pkv-fixtures=DIR --pkv-reference=DIR --model05=PATH --model15=PATH --list\n");
			for (const pkvg::Group* g : groups) std::printf("%s: %s\n", g->name, g->flags_help);
			return 0;
		}
		if (a.rfind("--", 0) == 0) {
			bool taken = false;
			for (const pkvg::Group* g : groups) taken = g->take_flag(a) || taken;
			if (!taken) {
				std::fprintf(stderr, "flag '%s' is taken by no group linked into this executable\n", a.c_str());
				++bad_flags;
			}
			continue;
		}
		sel.push_back(a);
	}
	if (common.fixture_dir.empty()) {
		if (const char* d = std::getenv("SUPERSLM_PAGED_KV_FIXTURE_DIR")) common.fixture_dir = d;
	}
	if (common.reference_dir.empty()) common.reference_dir = "tests/paged-kv/reference";
	for (const pkvg::Group* g : groups) g->apply_common(common);

	// Ids are unique across the executable's groups, or a selector could not name one cell.
	std::set<std::string> ids;
	int dup = 0;
	for (const pkvg::Group* g : groups)
		for (size_t i = 0; i < g->count(); ++i)
			if (!ids.insert(g->id(i)).second) {
				std::fprintf(stderr, "cell id '%s' is registered twice\n", g->id(i));
				++dup;
			}

	if (list) {
		for (const pkvg::Group* g : groups)
			for (size_t i = 0; i < g->count(); ++i)
				std::printf("[%s %s] %s%s\n", g->step(i), g->name, g->id(i), g->exact_only(i) ? "   (exact name only)" : "");
		return dup ? 1 : 0;
	}

	int unmatched = 0;
	for (const std::string& s : sel) {
		bool any = false;
		for (const pkvg::Group* g : groups)
			for (size_t i = 0; i < g->count() && !any; ++i) any = Matches(*g, i, s);
		if (!any) {
			std::fprintf(stderr, "argument '%s' matches no registered cell\n", s.c_str());
			++unmatched;
		}
	}

	int ran = 0, red = 0;
	for (const pkvg::Group* g : groups) {
		for (size_t i = 0; i < g->count(); ++i) {
			bool selected = sel.empty() && !g->exact_only(i);
			for (const std::string& s : sel) selected = selected || Matches(*g, i, s);
			if (!selected) continue;
			std::printf("[%s %s] %s ...\n", g->step(i), g->name, g->id(i));
			std::fflush(stdout);
			const pkvg::Tally before = g->tally();
			const int failed = g->run(i);
			const pkvg::Tally after = g->tally();
			if (pkvg::DrainFn drain = pkvg::CellBoundaryDrain()) {
				bool layer_on = false;
				const size_t messages = drain(&layer_on);
				if (layer_on) {
					std::printf("[%s %s] %s: %zu WARNING-or-worse debug-layer message(s) or tracker violation(s) during "
					            "the cell (diagnostic, not graded)\n", g->step(i), g->name, g->id(i), messages);
				}
			}
			++ran;
			if (failed) ++red;
			const int skipped = after.skips - before.skips;
			std::printf("[%s %s] %s %s (%d checks, %d failed%s)\n", g->step(i), g->name, g->id(i),
			            failed ? "RED" : (skipped ? "SKIPPED (allowed)" : "green"), after.checks - before.checks,
			            after.failures - before.failures, skipped ? ", skipped" : "");
			std::fflush(stdout);
		}
	}

	int checks = 0, failures = 0, skips = 0;
	for (const pkvg::Group* g : groups) {
		const pkvg::Tally t = g->tally();
		checks += t.checks;
		failures += t.failures;
		skips += t.skips;
		std::printf("%s: %d checks, %d failures, %d skips%s\n", g->name, t.checks, t.failures, t.skips, g->summary_note());
	}
	std::printf("%d cells, %d red; %d checks, %d failures, %d skips\n", ran, red, checks, failures, skips);
	if (unmatched) std::printf("%d argument(s) matched no registered cell (named on stderr)\n", unmatched);
	if (bad_flags) std::printf("%d flag(s) taken by no group (named on stderr)\n", bad_flags);
	return ran > 0 && red == 0 && unmatched == 0 && bad_flags == 0 && dup == 0 && failures == 0 ? 0 : 1;
}
