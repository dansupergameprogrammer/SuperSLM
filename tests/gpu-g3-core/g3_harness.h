// GPU paged-KV plan (rev 16), step G3: the host-only cells' harness (the D3D12-free core of G3).
//
// A cell registers with G3_CELL(id, fn) and checks with G3_CHECK*. g3_main.cpp runs the cells of
// the superslm_g3_core_tests executable (an argument selects cells by id prefix, as pkv_main.cpp
// does); superslm_tests runs the ledger's cells (3.6h) through RunGpuLedgerG3Cells
// (g3_superslm_tests_adapter.cpp). A cell's length parameter is read with G3Param("name", default)
// from an argument "@name=value", so the cell's own length is the default and a shorter run is
// asked for by name.
#ifndef SUPERSLM_TESTS_G3_HARNESS_H
#define SUPERSLM_TESTS_G3_HARNESS_H

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace g3 {

struct CellEntry {
	const char* id;
	void (*fn)();
};

inline std::vector<CellEntry>& Registry() {
	static std::vector<CellEntry> r;
	return r;
}

struct Registrar {
	Registrar(const char* id, void (*fn)()) { Registry().push_back({id, fn}); }
};

#define G3_CAT2(a, b) a##b
#define G3_CAT(a, b) G3_CAT2(a, b)
#define G3_CELL(id, fn) static ::g3::Registrar G3_CAT(g3_registrar_, __LINE__)(id, fn)

struct RunState {
	const char* cell = "(none)";
	int checks = 0;
	int failures = 0;
	int cell_failures = 0;
};

inline RunState& State() {
	static RunState s;
	return s;
}

inline std::map<std::string, long>& Params() {
	static std::map<std::string, long> p;
	return p;
}

inline long G3Param(const char* name, long dflt) {
	auto it = Params().find(name);
	return it == Params().end() ? dflt : it->second;
}

inline void Fail(const char* file, int line, const char* fmt, ...) {
	RunState& s = State();
	++s.failures;
	++s.cell_failures;
	std::fprintf(stderr, "FAIL [%s] %s:%d: ", s.cell, file, line);
	va_list ap;
	va_start(ap, fmt);
	std::vfprintf(stderr, fmt, ap);
	va_end(ap);
	std::fputc('\n', stderr);
}

#define G3_CHECK(cond)                                              \
	do {                                                            \
		++::g3::State().checks;                                     \
		if (!(cond)) ::g3::Fail(__FILE__, __LINE__, "%s", #cond);   \
	} while (0)
#define G3_CHECK_MSG(cond, ...)                                     \
	do {                                                            \
		++::g3::State().checks;                                     \
		if (!(cond)) ::g3::Fail(__FILE__, __LINE__, __VA_ARGS__);   \
	} while (0)
#define G3_CHECK_EQ(a, b)                                                                                   \
	do {                                                                                                    \
		++::g3::State().checks;                                                                             \
		const long long g3_a_ = static_cast<long long>(a), g3_b_ = static_cast<long long>(b);               \
		if (g3_a_ != g3_b_) ::g3::Fail(__FILE__, __LINE__, "%s == %s (%lld vs %lld)", #a, #b, g3_a_, g3_b_); \
	} while (0)

inline bool Matches(const char* id, const char* arg) {
	return arg[0] == '=' ? std::strcmp(id, arg + 1) == 0 : std::strncmp(id, arg, std::strlen(arg)) == 0;
}

// Runs every registered cell whose id starts with one of `prefixes` (all cells when empty).
// Returns the number of cells run; `unmatched` counts prefixes that selected nothing.
inline int RunCells(const std::vector<std::string>& prefixes, int* unmatched) {
	*unmatched = 0;
	for (const std::string& a : prefixes) {
		bool any = false;
		for (const CellEntry& c : Registry()) any = any || Matches(c.id, a.c_str());
		if (!any) {
			std::fprintf(stderr, "argument '%s' matches no registered cell\n", a.c_str());
			++*unmatched;
		}
	}
	int ran = 0;
	for (const CellEntry& c : Registry()) {
		bool selected = prefixes.empty();
		for (size_t i = 0; i < prefixes.size() && !selected; ++i) selected = Matches(c.id, prefixes[i].c_str());
		if (!selected) continue;
		RunState& s = State();
		s.cell = c.id;
		s.cell_failures = 0;
		const int before = s.checks;
		std::printf("[G3] %s ...\n", c.id);
		std::fflush(stdout);
		c.fn();
		++ran;
		std::printf("[G3] %s %s (%d checks, %d failed)\n", c.id, s.cell_failures ? "RED" : "green", s.checks - before,
		            s.cell_failures);
		std::fflush(stdout);
	}
	return ran;
}

}  // namespace g3

#endif  // SUPERSLM_TESTS_G3_HARNESS_H
