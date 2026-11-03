// Engine 1.19 clean stop, cell G9 (sfdg_cleanstop.h has the contract): CPU and GPU agree on the end. For POTION and
// PROMPT_RESULT, four runs to the end -- CPU whole-token (CS.E1's shape), CPU one layer per call (CS.E3's), GPU
// decode bridge (CS.G1's), GPU one layer per submission (CS.G2's) -- give the identical stream, each ends at -2 at
// SSLM_OK, every accepting read is 1, and the forced-token query after the end agrees (count and accepting). The
// streams equal cell 6.X5's pinned Off streams (kOffGoldens). Box only. Run: sfdg_cell_cs_g9.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw, 16);
		for (const Case& k : kCases) {
			const std::vector<int32_t> prompt = c.Tokenize(k.prompt_text);
			const CpuRun cw = RunCpu(c, k.schema, prompt, /*sliced=*/false);
			const CpuRun cl = RunCpu(c, k.schema, prompt, /*sliced=*/true);
			GSeq a = rig.New(k.schema), b = rig.New(k.schema);
			if (!a.h || !b.h || rig.Prompt(a, prompt) != GS::SSLM_OK || rig.Prompt(b, prompt) != GS::SSLM_OK) continue;
			const Run gw = RunWhole(rig, a);
			const Run gl = RunSliced(rig, b);
			CHECK_MSG(cw.dead && cl.dead && gw.dead && gl.dead && gw.st == GS::SSLM_OK && gl.st == GS::SSLM_OK,
			          "[CS.G9] %s: ends: CPU whole %d, CPU sliced %d, GPU whole %d (%s), GPU sliced %d (%s)", k.schema, cw.dead,
			          cl.dead, gw.dead, Name(gw.st), gl.dead, Name(gl.st));
			CHECK_MSG(cw.stream == cl.stream && cw.stream == gw.stream && cw.stream == gl.stream, "[CS.G9] %s: streams differ: "
			          "CPU whole %s, CPU sliced %s, GPU whole %s, GPU sliced %s", k.schema, Str(cw.stream).c_str(),
			          Str(cl.stream).c_str(), Str(gw.stream).c_str(), Str(gl.stream).c_str());
			CHECK_MSG(cw.accepting == 1 && cl.accepting == 1 && gw.accepting == 1 && gl.accepting == 1, "[CS.G9] %s: accepting "
			          "CPU whole %d, CPU sliced %d, GPU whole %d, GPU sliced %d (want 1 on every path)", k.schema, cw.accepting,
			          cl.accepting, gw.accepting, gl.accepting);
			const GQuery qa = QueryGpu(rig.ctx(), a.h);
			CHECK_MSG(qa.st == GS::SSLM_OK && qa.count == cw.query_count && qa.accepting == 1, "[CS.G9] %s: the forced query "
			          "after the end: GPU %s count %d accepting %d, CPU count %d", k.schema, Name(qa.st), qa.count, qa.accepting,
			          cw.query_count);
			for (const GoldenStream& g : kOffGoldens)
				if (std::string(g.schema) == k.schema)
					CHECK_MSG(gw.stream.size() == g.length && HexDigest(gw.stream) == g.digest, "[CS.G9] %s: the stream (%zu, %s) "
					          "is not the pinned Off stream (%zu, %s)", k.schema, gw.stream.size(), HexDigest(gw.stream).c_str(),
					          g.length, g.digest);
			rig.ReleaseSeqs();
		}
	}
	return Finish("sfdg_cell_cs_g9");
}
