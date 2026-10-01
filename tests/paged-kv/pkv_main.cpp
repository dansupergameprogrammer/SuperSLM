// Paged-KV plan (rev 16.1): the red suite's runner. Each per-step executable links this file with
// its own cell translation units (tests/paged-kv/CMakeLists.txt).
//
// Usage: superslm_pkv_<step> [CELL_ID_PREFIX ...]
// With no arguments every registered cell runs; otherwise only cells whose id starts with one of
// the given prefixes ("6.1", "1.13/C4"). An argument that starts with '=' names one cell exactly
// ("=10.1/C6" runs the box cell and not its cloud twin "10.1/C6:pkv_def"; the box runner,
// tools/run_paged_kv_c6_box.ps1, selects cells this way). Exit status 0 only when every check of
// every cell passed, at least one cell ran, and every argument selected at least one cell: an
// argument that matches no registered id (a mistyped or renamed cell, "6.3/C5" against a cell
// registered as "6.3") is named on stderr and fails the run, so a selection never silently
// shrinks to the cells that happen to match.

#include "pkv_common.h"

#include <cstdio>
#include <cstring>

namespace {
bool Matches(const char* id, const char* arg) {
	return arg[0] == '=' ? std::strcmp(id, arg + 1) == 0 : std::strncmp(id, arg, std::strlen(arg)) == 0;
}
}  // namespace

int main(int argc, char** argv) {
	int ran = 0, failed_cells = 0;
	// Every argument must name at least one registered cell, checked before any cell runs.
	int unmatched = 0;
	for (int i = 1; i < argc; ++i) {
		bool any = false;
		for (const pkv::CellEntry& c : pkv::Registry()) any = any || Matches(c.id, argv[i]);
		if (!any) {
			std::fprintf(stderr, "argument '%s' matches no registered cell\n", argv[i]);
			++unmatched;
		}
	}
	for (const pkv::CellEntry& c : pkv::Registry()) {
		bool selected = argc < 2;
		for (int i = 1; i < argc && !selected; ++i) selected = Matches(c.id, argv[i]);
		if (!selected) continue;
		pkv::RunState& s = pkv::State();
		s.cell = c.id;
		s.cell_failures = 0;
		const int before = s.checks;
		std::printf("[%s] %s ...\n", c.step, c.id);
		std::fflush(stdout);
		c.fn();
		++ran;
		if (s.cell_failures) ++failed_cells;
		std::printf("[%s] %s %s (%d checks, %d failed)\n", c.step, c.id, s.cell_failures ? "RED" : "green",
		            s.checks - before, s.cell_failures);
	}
	std::printf("%d cells, %d red; %d checks, %d failures\n", ran, failed_cells, pkv::State().checks,
	            pkv::State().failures);
	if (unmatched) std::printf("%d argument(s) matched no registered cell (named on stderr)\n", unmatched);
	return ran > 0 && unmatched == 0 && pkv::State().failures == 0 ? 0 : 1;
}
