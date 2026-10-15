// Paged-KV plan (rev 16.2) step G1: adapter presenting group g1c's registry (G1C_CELL) to pkvg_main.cpp.
// g1c's flags: --model=PATH (default: <--pkv-fixtures>/pkv_def.sslm), --adapter=PATH,
// --adapter-other-base=PATH, --schema=NAME, --pool-pages=N, --allow-skip. A skip fails its cell unless
// --allow-skip (g1c's rule: a run with an allowed skip is not an acceptance run).
#include "g1c_helpers.h"
#include "pkvg_runner.h"

namespace {
bool TakeFlag(const std::string& a) {
	g1c::Options& o = g1c::Opt();
	auto take = [&](const char* flag) -> const char* {
		const size_t n = std::strlen(flag);
		return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
	};
	if (const char* v = take("--model=")) o.model = v;
	else if (const char* v = take("--adapter-other-base=")) o.adapter_other_base = v;
	else if (const char* v = take("--adapter=")) o.adapter = v;
	else if (const char* v = take("--schema=")) o.schema = v;
	else if (const char* v = take("--pool-pages=")) o.pool_pages = static_cast<uint32_t>(std::atoi(v));
	else if (a == "--allow-skip") o.allow_skip = true;
	else return false;
	return true;
}
void ApplyCommon(const pkvg::CommonOptions& c) {
	g1c::Options& o = g1c::Opt();
	if (o.model.empty() && !c.fixture_dir.empty()) o.model = c.fixture_dir + "/pkv_def.sslm";
}
size_t Count() { return g1c::Registry().size(); }
const char* Id(size_t i) { return g1c::Registry()[i].id; }
const char* Step(size_t i) { return g1c::Registry()[i].step; }
bool ExactOnly(size_t) { return false; }
int Run(size_t i) {
	g1c::RunState& s = g1c::State();
	s.cell = g1c::Registry()[i].id;
	s.cell_failures = 0;
	const int skips_before = s.skips;
	g1c::Registry()[i].fn();
	const bool disallowed_skip = s.skips > skips_before && !g1c::Opt().allow_skip;
	return s.cell_failures + (disallowed_skip ? 1 : 0);
}
pkvg::Tally Tally() {
	pkvg::Tally t;
	t.checks = g1c::State().checks;
	t.failures = g1c::State().failures;
	t.skips = g1c::State().skips;
	return t;
}
const char* Note() {
	return g1c::State().skips && g1c::Opt().allow_skip ? " (skips allowed by --allow-skip: not an acceptance run)" : "";
}

const pkvg::Group kGroup = {"g1c", TakeFlag, ApplyCommon, Count, Id, Step, ExactOnly, Run, Tally, Note,
                            "--model=PATH --adapter=PATH --adapter-other-base=PATH --schema=NAME --pool-pages=N --allow-skip"};
const pkvg::GroupRegistrar kRegistrar(&kGroup);
}  // namespace
