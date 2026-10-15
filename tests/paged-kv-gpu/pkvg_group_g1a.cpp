// Paged-KV plan (rev 16.2) step G1: adapter presenting group g1a's registry (G1A_CELL) to pkvg_main.cpp.
// g1a's flags: --gpu-ref=DIR, --record-gpu-ref (6.5a's pre-G2 GPU records); the common flags fill
// g1a::Opt()'s fixture_dir, reference_dir, model05 and model15.
#include "g1a_helpers.h"
#include "pkvg_runner.h"

namespace {
bool TakeFlag(const std::string& a) {
	g1a::Options& o = g1a::Opt();
	if (a.rfind("--gpu-ref=", 0) == 0) {
		o.gpu_ref_dir = a.substr(10);
		return true;
	}
	if (a == "--record-gpu-ref") {
		o.record_gpu_ref = true;
		return true;
	}
	return false;
}
void ApplyCommon(const pkvg::CommonOptions& c) {
	g1a::Options& o = g1a::Opt();
	if (o.fixture_dir.empty()) o.fixture_dir = c.fixture_dir;
	if (o.reference_dir.empty()) o.reference_dir = c.reference_dir;
	if (o.model05.empty()) o.model05 = c.model05;
	if (o.model15.empty()) o.model15 = c.model15;
}
size_t Count() { return g1a::Registry().size(); }
const char* Id(size_t i) { return g1a::Registry()[i].id; }
const char* Step(size_t i) { return g1a::Registry()[i].step; }
bool ExactOnly(size_t) { return false; }
int Run(size_t i) {
	g1a::RunState& s = g1a::State();
	s.cell = g1a::Registry()[i].id;
	s.cell_failures = 0;
	g1a::Registry()[i].fn();
	return s.cell_failures;
}
pkvg::Tally Tally() {
	pkvg::Tally t;
	t.checks = g1a::State().checks;
	t.failures = g1a::State().failures;
	return t;
}
const char* Note() { return ""; }

const pkvg::Group kGroup = {"g1a", TakeFlag, ApplyCommon, Count, Id, Step, ExactOnly, Run, Tally, Note,
                            "--gpu-ref=DIR --record-gpu-ref"};
const pkvg::GroupRegistrar kRegistrar(&kGroup);
}  // namespace
