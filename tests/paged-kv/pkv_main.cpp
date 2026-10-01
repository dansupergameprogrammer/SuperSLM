// Paged-KV plan (rev 16.1): the red suite's runner. Each per-step executable links this file with
// its own cell translation units (tests/paged-kv/CMakeLists.txt).
//
// Usage: superslm_pkv_<step> [CELL_ID_PREFIX ...]
// With no arguments every registered cell runs; otherwise only cells whose id starts with one of
// the given prefixes ("6.1", "1.13/C4"). Exit status 0 only when every check of every cell passed
// and at least one cell ran.

#include "pkv_common.h"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
	int ran = 0, failed_cells = 0;
	for (const pkv::CellEntry& c : pkv::Registry()) {
		bool selected = argc < 2;
		for (int i = 1; i < argc && !selected; ++i) selected = std::strncmp(c.id, argv[i], std::strlen(argv[i])) == 0;
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
	return ran > 0 && pkv::State().failures == 0 ? 0 : 1;
}
