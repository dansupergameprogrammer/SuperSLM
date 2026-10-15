// GPU paged-KV plan (rev 16), step G3: the host-only cells' runner (superslm_g3_core_tests).
//
// Usage: superslm_g3_core_tests [CELL_ID_PREFIX | =CELL_ID | @param=value ...]
// With no cell argument every registered cell runs. Exit status 0 only when every check passed, at
// least one cell ran, and every cell argument selected at least one cell.
#include "g3_harness.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
	std::vector<std::string> prefixes;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.size() > 1 && a[0] == '@') {
			const size_t eq = a.find('=');
			if (eq == std::string::npos) {
				std::fprintf(stderr, "parameter '%s' has no value\n", a.c_str());
				return 2;
			}
			g3::Params()[a.substr(1, eq - 1)] = std::atol(a.c_str() + eq + 1);
		} else {
			prefixes.push_back(a);
		}
	}
	int unmatched = 0;
	const int ran = g3::RunCells(prefixes, &unmatched);
	const g3::RunState& s = g3::State();
	std::printf("%d cells; %d checks, %d failures\n", ran, s.checks, s.failures);
	return ran > 0 && unmatched == 0 && s.failures == 0 ? 0 : 1;
}
