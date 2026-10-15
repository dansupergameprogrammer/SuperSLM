// Paged-KV plan (rev 16.2) step G1: the GPU red suite's one in-process runner (pkvg_main.cpp) and the
// interface each cell group's registry is driven through.
//
// The suite's cells were written by five groups (g1a..g1e). g1a, g1b and g1c register cells in a
// per-group registry (G1A_CELL, G1B_CELL, G1C_CELL: one namespace each, one run state each, one option
// set each). Their cell code is unchanged; each group gets a small adapter TU, pkvg_group_<g>.cpp, which
// includes that group's helpers and presents its registry, its run state and its flags through `Group`
// below. pkvg_main.cpp drives every group linked into the executable, so one exe per step and
// configuration carries every in-process cell of that step (see build_paged_kv_gpu.bat).
// g1d and g1e cells are processes of their own (their own main(): per-cell engine configurations, fresh
// processes for terminal legs, files written for another tree's reader); pkvg_suite.cpp runs them.
#ifndef SSLM_PKVG_RUNNER_H
#define SSLM_PKVG_RUNNER_H

#include <cstddef>
#include <string>
#include <vector>

namespace pkvg {

// Flags shared across groups. Each adapter maps them onto its group's own option fields.
struct CommonOptions {
	std::string fixture_dir;    // --pkv-fixtures=DIR: tools/gen_paged_kv_fixture.py's output directory
	std::string reference_dir;  // --pkv-reference=DIR: tests/paged-kv/reference (R0's records)
	std::string model05;        // --model05= / --m05=: the real Qwen2.5-0.5B artifact
	std::string model15;        // --model15= / --m15=: the real Qwen2.5-1.5B artifact
};

struct Tally {
	int checks = 0;
	int failures = 0;
	int skips = 0;
};

struct Group {
	const char* name;  // "g1a"
	// One group-specific flag ("--gpu-ref=DIR"): true when this group took it.
	bool (*take_flag)(const std::string& arg);
	// Called once after every flag is parsed: fill the group's options from the common ones where the
	// group's own flag was not given.
	void (*apply_common)(const CommonOptions& common);
	size_t (*count)();
	const char* (*id)(size_t i);
	const char* (*step)(size_t i);
	// A cell that runs only when named exactly ("=id"): g1b's ':' legs leave terminal process state.
	bool (*exact_only)(size_t i);
	// Runs cell i; returns its failures (checks failed, plus a disallowed skip counted as one).
	int (*run)(size_t i);
	Tally (*tally)();
	// Extra summary text for the group (e.g. g1c's allowed-skip note), or "".
	const char* (*summary_note)();
	// Flags this group takes, for --help.
	const char* flags_help;
};

// The cell-boundary drain: when a translation unit that can read the device's info queue is linked in
// (g1c_debug_layer.cpp registers g1c::DebugLayerReportable here), pkvg_main.cpp calls it after every
// cell and prints how many WARNING-or-worse debug-layer messages (with the host state tracker's
// violations, in a tracking build) arrived during that cell. A diagnostic, never graded: it attributes
// GPU-based validation messages to the cell whose work raised them, where otherwise only the cells that
// grade the count (3.6, 3.6d's scope control) would show them, inside their own windows. Null when no
// such unit is linked (pkvg_g2).
using DrainFn = size_t (*)(bool* enabled);
inline DrainFn& CellBoundaryDrain() {
	static DrainFn f = nullptr;
	return f;
}

inline std::vector<const Group*>& Groups() {
	static std::vector<const Group*> g;
	return g;
}
struct GroupRegistrar {
	explicit GroupRegistrar(const Group* g) { Groups().push_back(g); }
};

}  // namespace pkvg

#endif  // SSLM_PKVG_RUNNER_H
