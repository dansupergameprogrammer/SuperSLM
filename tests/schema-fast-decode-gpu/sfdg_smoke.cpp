// Schema fast-decode plan rev 1.4.4 §10, GB0 (audit N-5; rev 1.4.3, re-check note 4): the F-SFD smoke and the
// debug-layer baseline. It maps F-SFD on the GPU, prefills the potion prompt and decodes one token, unbound and
// then bound, through existing entry points only -- none of G1's verbs -- so it builds and runs at G1-F's red-suite
// commit, and the message IDs it reports at exit (Finish) are the baseline every later GB step is compared with.
// Box only. The driver runs it at SSLM_GPU_ENABLE_DEBUG_LAYER=2. Run: sfdg_smoke.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

using namespace sfdg;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		CHECK_MSG(!prompt.empty(), "[smoke] the potion prompt tokenizes");
		for (const char* schema : {static_cast<const char*>(nullptr), kPotion}) {
			GSeq s = rig.New(schema);
			if (!s.h) continue;
			const GS ps = rig.Prompt(s, prompt);
			GS st = GS::SSLM_OK;
			const int32_t t = ps == GS::SSLM_OK ? Token(rig.env, s, &st) : -1;
			CHECK_MSG(ps == GS::SSLM_OK && st == GS::SSLM_OK && t >= 0, "[smoke %s] prefill %s, one token %d (%s)",
			          schema ? schema : "unbound", Name(ps), t, Name(st));
			std::printf("[smoke %s] token %d\n", schema ? schema : "unbound", t);
		}
		rig.ReleaseSeqs();
	}
	return Finish("sfdg_smoke");
}
