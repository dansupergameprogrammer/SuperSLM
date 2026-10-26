// Schema fast-decode plan rev 1.4.4 §11.2, cell G7.1 (slice G1-J): the GPU jump-forward stream equals the CPU stream
// on F-SFD, POTION and PROMPT_RESULT, with a pinned digest. The loop is the §5.6 host contract on both backends:
// the verb until the query reports no chain, then one greedy decode, until the dead end; chunk budget 64; the fixed
// prompts. The digest is cell 6.6's (the CPU stream pinned on main), so a GPU table built from the wrong schema
// index (XG3) or a verb that walks the pending token again (XG2, XG8) fails here.
// Arms: the head on the host and on the device (tied). Box only. Run: sfdg_cell_g7_1.exe [--fixture-dir=DIR]
// MR5 RE-POINT (multi-row GPU plan, ruling 29: the MRX review's M1). The pinned digests are 1.17's Canonical streams
// (24 and 42 tokens); 1.18 removes Canonical, and Exact's stream is the decode-only run's. As the CPU suite's cell 6.6
// does since MRX, the expectation is the CPU decode-only stream (greedy, same prompt, to the dead end), computed live
// without the jump verb. Red until MR5 (the GPU refuses Exact).
#include "sfdg_helpers.h"

using namespace sfdg;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		const std::string dir = FixtureDir(argc, argv);
		CRig c(rig.raw);
		Mapped host, dev;
		MapVariant(rig, dir + "/fsfd.sslm", 0, &host);
		MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev);
		for (const GoldenStream& g : kGoldenStreams) {
			const bool potion = std::string(g.schema) == kPotion;
			const std::vector<int32_t> prompt = c.Tokenize(potion ? kPromptPotionText : kPromptPrText);
			// The expectation: CPU decode-only to the dead end (no jump verb).
			std::vector<int32_t> want;
			bool want_dead = false;
			if (sslm_seq cs = c.New(g.schema)) {
				if (c.Prefill(cs, prompt, SSLM_SPAN_PROMPT) == SSLM_OK) {
					for (int i = 0; i < 400; ++i) {
						const int32_t t = c.Decode(cs);
						if (t == -2) { want_dead = true; break; }
						if (t < 0) break;
						want.push_back(t);
					}
				}
			}
			CHECK_MSG(want_dead, "[G7.1] %s: the CPU decode-only run did not reach the dead end (%zu tokens)", g.schema, want.size());
			const std::string want_digest = HexDigest(want);
			for (Mapped* m : {&host, &dev}) {
				if (!m->m) continue;
				const char* arm = m == &host ? "host head" : "device head";
				GSeq s = rig.New(g.schema, m->m);
				if (!s.h) continue;
				const GS ps = rig.Prompt(s, prompt);
				CHECK_MSG(ps == GS::SSLM_OK, "[G7.1 %s] %s prompt: %s", arm, g.schema, Name(ps));
				const GJfRun r = RunGpuJf(rig, s, 64);
				const std::string got = HexDigest(r.stream);
				CHECK_MSG(r.dead_end && r.end == GS::SSLM_OK, "[G7.1 %s] %s: the loop ended with %s (dead end %d) after %d "
				          "verb calls and %d decodes", arm, g.schema, Name(r.end), r.dead_end, r.verb_calls, r.decodes);
				CHECK_MSG(r.stream == want,
				          "[G7.1 %s] %s: stream %s (%zu tokens, digest %s); the CPU decode-only %zu, %s (1.17's Canonical pin was "
				          "%zu, %s)", arm, g.schema, Str(r.stream).c_str(), r.stream.size(), got.c_str(), want.size(),
				          want_digest.c_str(), g.length, g.digest);
			}
		}
		rig.ReleaseSeqs();
		Unmap(rig, &host);
		Unmap(rig, &dev);
	}
	return Finish("sfdg_cell_g7_1");
}
