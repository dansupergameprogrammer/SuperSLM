// Engine 1.19 clean stop, cell G2 (sfdg_cleanstop.h has the contract): one row, sliced. Every token after the
// prompt's ready one is embedded, driven one layer per sslm_decode_step_gpu submission, and finished by the finish
// bridge; the last embed carries the closing token in, and its finish is the -2. The end is -2 at SSLM_OK with
// accepting 1, the walk kept, ready re-armed; retries give -2 with the blob unchanged and no K/V row added. The stream
// is the CPU's sliced stream (cpu::Token, layer budget 1; CPU cell CS.E3), one slice per layer per embedded token, and
// the end state equals the whole-token run's (G1) byte for byte. Box only. Run: sfdg_cell_cs_g2.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw, 16);
		for (const Case& k : kCases) {
			const std::vector<int32_t> prompt = c.Tokenize(k.prompt_text);
			const CpuRun want = RunCpu(c, k.schema, prompt, /*sliced=*/true);
			CHECK_MSG(want.dead && want.accepting == 1, "[CS.G2] %s: the CPU sliced reference ended dead %d accepting %d",
			          k.schema, want.dead, want.accepting);
			GSeq w = rig.New(k.schema);
			if (!w.h || rig.Prompt(w, prompt) != GS::SSLM_OK) continue;
			const Run whole = RunWhole(rig, w);
			GSeq s = rig.New(k.schema);
			if (!s.h || rig.Prompt(s, prompt) != GS::SSLM_OK) continue;
			const Run r = RunSliced(rig, s);
			CheckCleanStop("CS.G2", "sliced", k.schema, rig, s, r, prompt.size());
			CHECK_MSG(r.stream == want.stream, "[CS.G2] %s: stream %s, the CPU sliced stream %s", k.schema,
			          Str(r.stream).c_str(), Str(want.stream).c_str());
			CHECK_MSG(r.slices == static_cast<int>(r.stream.size() * rig.env.fx.layers), "[CS.G2] %s: %d slices for %zu "
			          "embedded tokens of %u layers, want one per layer", k.schema, r.slices, r.stream.size(), rig.env.fx.layers);
			CHECK_MSG(!whole.end_blob.empty() && r.end_blob == whole.end_blob, "[CS.G2] %s: the sliced end state differs "
			          "from the whole-token end state (contexts %lld and %lld)", k.schema,
			          static_cast<long long>(r.ctx_after), static_cast<long long>(whole.ctx_after));
		}
		rig.ReleaseSeqs();
	}
	return Finish("sfdg_cell_cs_g2");
}
