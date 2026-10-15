// Paged-KV plan (rev 16.2) step G1: adapter presenting group g1b's registry (G1B_CELL) to pkvg_main.cpp.
// g1b's flags: --tiny=PATH (default: <--pkv-fixtures>/pkv_def.sslm), --shaders=DIR; --m05/--m15 are the
// common --model05/--model15. Cells whose id carries a ':' leg suffix ("11.4/G3:zerofill") leave terminal
// process state behind them and run only when named exactly, in a process of their own (g1b's rule).
#include "g1b_helpers.h"
#include "pkvg_runner.h"

namespace {
bool TakeFlag(const std::string& a) {
	g1b::Args& o = g1b::GArgs();
	if (a.rfind("--tiny=", 0) == 0) {
		o.tiny = a.substr(7);
		return true;
	}
	if (a.rfind("--shaders=", 0) == 0) {
		o.shaders = a.substr(10);
		return true;
	}
	return false;
}
void ApplyCommon(const pkvg::CommonOptions& c) {
	g1b::Args& o = g1b::GArgs();
	if (o.tiny.empty() && !c.fixture_dir.empty()) o.tiny = c.fixture_dir + "/pkv_def.sslm";
	if (o.m05.empty()) o.m05 = c.model05;
	if (o.m15.empty()) o.m15 = c.model15;
}
size_t Count() { return g1b::Registry().size(); }
const char* Id(size_t i) { return g1b::Registry()[i].id; }
const char* Step(size_t i) { return g1b::Registry()[i].step; }
bool ExactOnly(size_t i) { return std::strchr(g1b::Registry()[i].id, ':') != nullptr; }
int Run(size_t i) {
	g1b::RunState& s = g1b::State();
	s.cell = g1b::Registry()[i].id;
	s.cell_failures = 0;
	g1b::Registry()[i].fn();
	return s.cell_failures;
}
pkvg::Tally Tally() {
	pkvg::Tally t;
	t.checks = g1b::State().checks;
	t.failures = g1b::State().failures;
	return t;
}
const char* Note() { return ""; }

const pkvg::Group kGroup = {"g1b", TakeFlag, ApplyCommon, Count, Id, Step, ExactOnly, Run, Tally, Note,
                            "--tiny=PATH --shaders=DIR"};
const pkvg::GroupRegistrar kRegistrar(&kGroup);
}  // namespace
