// T-2333 (Curie) -- Sec7 dimension 11's FIRST commissioning population: a
// reserve()-driven std::unordered_map/set growth mutant on the ORIGINAL two
// replacement sites (tokenizer.cpp's own std::unordered_map family, model.cpp's
// own std::unordered_set<std::string_view> family).
//
// PROVENANCE: this scope was DENIED in T-2326's own session (Claude/Loki/
// te32-probe/ was outside that ticket's granted read-only scope) and is
// GRANTED for T-2333 (coordinator's own brief, item 5). This file reproduces
// TE-32's own construction verbatim from
// Claude/Loki/te32-probe/te32_fp_observability_cell.cpp's own `DoReserve`
// function (lines ~101-117 of that file), kind 0 (`map_str_u32` --
// std::unordered_map<std::string, uint32_t>, mirroring tokenizer.cpp:162-169's
// own four std::unordered_map sites) and kind 1 (`set_string_view` --
// std::unordered_set<std::string_view>, mirroring model.cpp:274/533/684's own
// three std::unordered_set<std::string_view> sites) -- the ORIGINAL two site
// families this population's own text names, as distinct from population
// three's own EXTENDED site (sslm_abi.cpp's g_live_seqs, a different
// container entirely, realized in pop03_seqreg_growth_mutant.cpp).
//
// TE-32's own file drives these through reserve() at 23 sizes spanning the
// float-representability boundary (2^24) inside a runtime trap/sticky-flag
// harness (Part B/C) -- 69 of 69 legs trapped under cleared exception masks
// (Claude/Loki/te32-probe/output-te32.txt, "PART B CENSUS: 0 of 69 legs
// SURVIVED"). This design's own population one claim is the BYTE-LEVEL half
// of that same corroboration: the identical reserve()-driven rehash-sizing
// machinery, scanned, must REJECT under checks (A)/(B) -- not merely trap at
// runtime. The runtime harness itself (argv-mode dispatch, child-process
// trap legs) is not reproduced here; only the two container-growth bodies a
// byte-level scan can be pointed at.
//
// Build: MSVC cl.exe (/std:c++20 /O2 /EHsc), matching TE-32's own build.bat
// exactly (`cl /std:c++20 /O2 /W4 /fp:precise /EHsc`).

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace {
volatile size_t g_sink = 0;
}

// TE-32's own DoReserve, kind 0 -- mirrors tokenizer.cpp's own
// std::unordered_map<std::string, uint32_t> family.
extern "C" __declspec(dllexport) void Te32ReserveMapStrU32(uint64_t n) {
	static std::unordered_map<std::string, uint32_t> m;
	m.clear();
	m.reserve(static_cast<size_t>(n));
	g_sink += m.bucket_count();
}

// TE-32's own DoReserve, kind 1 -- mirrors model.cpp's own
// std::unordered_set<std::string_view> family.
extern "C" __declspec(dllexport) void Te32ReserveSetStringView(uint64_t n) {
	static std::unordered_set<std::string_view> s;
	s.clear();
	s.reserve(static_cast<size_t>(n));
	g_sink += s.bucket_count();
}
