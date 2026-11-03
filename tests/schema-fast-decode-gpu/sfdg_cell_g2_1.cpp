// Schema fast-decode plan rev 1.4.4 §11.2, cell G2.1 (slice G1-F; rev 1.4.1, review L6): a base view without a
// tokenizer maps on the GPU with no forced-next table. The query returns SSLM_GPU_FORCED_TABLE_ABSENT (the GPU
// enum has no artifact refusal), and schema-bound decode still works. G1-J's arm: the verb returns the same status,
// from the fresh shape and from the ready shape, with nothing changed and the count unwritten.
//
// The construction: F-SFD's own view with has_tokenizer cleared, which is what the GPU map reads to decide
// (an artifact without TOK1 loads with has_tokenizer false). Box only. Run: sfdg_cell_g2_1.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

using namespace sfdg;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, false)) {
		superslm::SslmModelView view{};
		std::string err;
		const bool loaded = superslm::SslmModel::Load(rig.raw.data(), rig.raw.size(), view, &err) ==
		                    superslm::SslmModelStatus::Ok;
		CHECK_MSG(loaded && view.has_tokenizer, "[G2.1] F-SFD's second view: %s", err.c_str());
		view.has_tokenizer = false;
		SslmGpuModelHandle* m = OutSentinel<SslmGpuModelHandle>();
		const GS ms = sslm_gpu_model_map(rig.ctx(), &view, GpuResidencyConfig{}, &m);
		CHECK_MSG(ms == GS::SSLM_OK && Real(m), "[G2.1] a view without a tokenizer must map: %s", Name(ms));
		if (ms == GS::SSLM_OK && Real(m)) {
			{
				GSeq s = rig.New(kPotion, m);
				const GQuery q = QueryGpu(rig.ctx(), s.h);
				CHECK_MSG(q.st == kForcedTableAbsent && q.count == -9 && q.accepting == -9,
				          "[G2.1] the query without a table: %s, count %d, accepting %d (want SSLM_GPU_FORCED_TABLE_ABSENT, "
				          "nothing written)", Name(q.st), q.count, q.accepting);
				// Decode works: the prompt, then greedy tokens, all legal under the schema.
				CRig c(rig.raw);
				const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
				// G1-J: the verb, fresh and then ready.
				for (int shape = 0; shape < 2; ++shape) {
					if (shape == 1) {
						const GS ps = rig.Prompt(s, prompt);
						CHECK_MSG(ps == GS::SSLM_OK, "[G2.1] prompt prefill on the table-less model: %s", Name(ps));
					}
					const SeqObs before = Observe(rig, s.h);
					int32_t out_ids[8], n = -9;
					const GS js = sslm_gpu_seq_jump_forward(rig.ctx(), s.h, -1, 8, nullptr, out_ids, 8, &n);
					CHECK_MSG(js == kForcedTableAbsent && Observe(rig, s.h) == before,
					          "[G2.1] the verb without a table (%s shape): %s, sequence %s (want SSLM_GPU_FORCED_TABLE_ABSENT, "
					          "nothing changed)", shape ? "ready" : "fresh", Name(js),
					          Observe(rig, s.h) == before ? "unchanged" : "changed");
					if (shape == 0) {
						const GS b = SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, SslmGpuSchemaLookupForG5Bridge(m, kPotion));
						CHECK_MSG(b == GS::SSLM_OK, "[G2.1] the refused fresh sequence is no longer bindable: %s", Name(b));
					}
				}
				GS last = GS::SSLM_OK;
				const std::vector<int32_t> out = DecodeN(rig.env, s, 8, &last);
				CHECK_MSG(last == GS::SSLM_OK && !out.empty(), "[G2.1] decode on the table-less model: %s, %zu tokens",
				          Name(last), out.size());
				// The same tokens as the table-carrying model: the table changes no decode.
				GSeq t = rig.New(kPotion);
				rig.Prompt(t, prompt);
				const std::vector<int32_t> want = DecodeN(rig.env, t, 8);
				CHECK_MSG(out == want, "[G2.1] decode differs with and without the table: %s vs %s", Str(out).c_str(),
				          Str(want).c_str());
				sslm_gpu_seq_release(rig.ctx(), s.h);
				rig.seqs.erase(std::remove(rig.seqs.begin(), rig.seqs.end(), s.h), rig.seqs.end());
			}
			// The table-carrying model answers the same query with OK.
			GSeq u = rig.New(kPotion);
			const GQuery q2 = QueryGpu(rig.ctx(), u.h);
			CHECK_MSG(q2.st == GS::SSLM_OK && q2.count >= 0, "[G2.1] F-SFD with its tokenizer: %s, count %d", Name(q2.st), q2.count);
			CHECK_MSG(sslm_gpu_model_unmap(rig.ctx(), m) == GS::SSLM_OK, "[G2.1] unmap");
		}
	}
	return Finish("sfdg_cell_g2_1");
}
