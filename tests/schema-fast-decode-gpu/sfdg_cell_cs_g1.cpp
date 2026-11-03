// Engine 1.19 clean stop, cell G1 (sfdg_cleanstop.h has the contract): one row, the decode bridge, whole tokens, on
// F-SFD's POTION and PROMPT_RESULT run to their end, with the head on the host and on the device. The end is -2 at
// SSLM_OK with accepting 1 on both GPU queries, the walk kept, the ready flag re-armed, the context the prompt plus
// the stream; retries through the decode bridge and the finish bridge give -2 again with the blob unchanged. The
// stream is the CPU's whole-token stream. Box only. Run: sfdg_cell_cs_g1.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		const std::string dir = FixtureDir(argc, argv);
		CRig c(rig.raw, 16);
		Mapped host, dev;
		MapVariant(rig, dir + "/fsfd.sslm", 0, &host);
		MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev);
		for (const Case& k : kCases) {
			const std::vector<int32_t> prompt = c.Tokenize(k.prompt_text);
			const CpuRun want = RunCpu(c, k.schema, prompt, /*sliced=*/false);
			CHECK_MSG(want.dead && want.accepting == 1, "[CS.G1] %s: the CPU reference ended dead %d accepting %d", k.schema,
			          want.dead, want.accepting);
			for (Mapped* m : {&host, &dev}) {
				if (!m->m) continue;
				const char* arm = m == &host ? "host head" : "device head";
				GSeq s = rig.New(k.schema, m->m);
				if (!s.h || rig.Prompt(s, prompt) != GS::SSLM_OK) continue;
				const Run r = RunWhole(rig, s);
				CheckCleanStop("CS.G1", arm, k.schema, rig, s, r, prompt.size());
				CHECK_MSG(r.stream == want.stream, "[CS.G1] %s %s: stream %s, the CPU's %s", arm, k.schema,
				          Str(r.stream).c_str(), Str(want.stream).c_str());
			}
		}
		rig.ReleaseSeqs();
		Unmap(rig, &host);
		Unmap(rig, &dev);
	}
	return Finish("sfdg_cell_cs_g1");
}
