// Schema fast-decode plan rev 1.4.4 §11.2, cell G12.4 (slice G1-F; rev 1.4.1, audit G-7): query parity.
//   (a) sslm_gpu_seq_forced_tokens equals the CPU's sslm_seq_forced_tokens on every state of every F-SFD schema,
//       ids, count and accepting, each state reached by SCHEMA_CONTENT prefill of the reference path on both
//       backends (and the reference chain of fsfd_ref.txt).
//   (b) The GPU table's golden hashes equal cell 6.6's (read through the table seam).
//   (c) Refusals: a malformed handle (null context, sequence or outputs; a sequence of another context; a
//       negative cap; a positive cap with null ids) is SSLM_SEQUENCE_KV_BUFFER_MISMATCH, a Submitted sequence
//       SSLM_BUSY, each writing nothing.
//   (d) The query changes nothing: the SLM5 blob is unchanged, it submits nothing (S-GSUB), and a reset
//       sequence it queried is still bindable.
// Box only. Run: sfdg_cell_g12_4.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

using namespace sfdg;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv)) {
		CRig c(rig.raw);
		// (a) every state, both backends.
		for (const char* schema : kSchemas) {
			const RefSchema& r = rig.ref[schema];
			GSeq g = rig.New(schema);
			sslm_seq cs = c.New(schema);
			int bad = 0;
			for (uint32_t s = 0; s < r.st.size() && g.h && cs; ++s) {
				const RefState& rs = r.st[s];
				const GS rg = sslm_gpu_seq_reset(rig.ctx(), g.h);
				g = GSeq{g.h};
				const sslm_status rc = sslm_seq_reset(cs);
				GS pg = rg;
				if (rg == GS::SSLM_OK) pg = rig.Content(g, rs.path);
				const sslm_status pc = rc == SSLM_OK ? c.Prefill(cs, rs.path, SSLM_SPAN_SCHEMA_CONTENT) : rc;
				if (pg != GS::SSLM_OK || pc != SSLM_OK) {
					if (bad++ < 5)
						CHECK_MSG(false, "[G12.4] %s state %u: reaching it: GPU %s, CPU %d", schema, s, Name(pg), static_cast<int>(pc));
					continue;
				}
				const GQuery qg = QueryGpu(rig.ctx(), g.h);
				const CQuery qc = QueryCpu(c.M(), cs);
				const bool same = qg.st == GS::SSLM_OK && qc.st == SSLM_OK && qg.ids == qc.ids && qg.count == qc.count &&
				                  qg.accepting == qc.accepting;
				const bool ref_ok = qg.ids == rs.chain && qg.accepting == (rs.accepting ? 1 : 0);
				if ((!same || !ref_ok) && bad++ < 5)
					CHECK_MSG(false, "[G12.4] %s state %u: GPU %s %s count %d acc %d; CPU %d %s count %d acc %d; ref %s acc %d",
					          schema, s, Name(qg.st), Str(qg.ids).c_str(), qg.count, qg.accepting, static_cast<int>(qc.st),
					          Str(qc.ids).c_str(), qc.count, qc.accepting, Str(rs.chain).c_str(), rs.accepting ? 1 : 0);
			}
			CHECK_MSG(bad == 0, "[G12.4] %s: %d of %zu states differ", schema, bad, r.st.size());
			std::printf("[G12.4] %s: %zu states compared\n", schema, r.st.size());
		}
		// (b) the table's golden hashes.
		for (const GoldenTable& gt : kGoldenTables) {
			const int32_t si = rig.Index(gt.schema);
			const RefSchema& r = rig.ref[gt.schema];
			std::vector<int32_t> words;
			bool ok = si >= 0;
			for (uint32_t s = 0; s < r.st.size() && ok; ++s) {
				int32_t nf = -9;
				uint32_t fl = 0;
				ok = sslm_gpu_test_only_forced_table_entry(rig.model(), si, s, &nf, &fl) == GS::SSLM_OK;
				words.push_back(nf);
				words.push_back(static_cast<int32_t>(fl));
			}
			const std::string got = HexDigest(words);
			CHECK_MSG(ok && got == gt.digest && r.digest == gt.digest, "[G12.4] %s: the GPU table digest %s, cell 6.6 pins %s, "
			          "fsfd_ref.txt %s", gt.schema, got.c_str(), gt.digest, r.digest.c_str());
		}
		// (c) refusals, each writing nothing.
		{
			GSeq g = rig.New(kLongKey);
			int32_t ids[4] = {-7, -7, -7, -7}, n = -9, acc = -9;
			auto untouched = [&] { return n == -9 && acc == -9 && ids[0] == -7; };
			GS st = sslm_gpu_seq_forced_tokens(nullptr, g.h, ids, 4, &n, &acc);
			CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && untouched(), "[G12.4] null context: %s", Name(st));
			st = sslm_gpu_seq_forced_tokens(rig.ctx(), nullptr, ids, 4, &n, &acc);
			CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && untouched(), "[G12.4] null sequence: %s", Name(st));
			st = sslm_gpu_seq_forced_tokens(rig.ctx(), g.h, ids, 4, nullptr, &acc);
			CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && untouched(), "[G12.4] null count: %s", Name(st));
			st = sslm_gpu_seq_forced_tokens(rig.ctx(), g.h, ids, 4, &n, nullptr);
			CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && untouched(), "[G12.4] null accepting: %s", Name(st));
			st = sslm_gpu_seq_forced_tokens(rig.ctx(), g.h, ids, -1, &n, &acc);
			CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && untouched(), "[G12.4] negative cap: %s", Name(st));
			st = sslm_gpu_seq_forced_tokens(rig.ctx(), g.h, nullptr, 4, &n, &acc);
			CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && untouched(), "[G12.4] null ids at cap 4: %s", Name(st));
			SslmGpuContext* other = nullptr;
			if (sslm_gpu_context_create(GpuContextConfig{}, &other) == GS::SSLM_OK && other) {
				st = sslm_gpu_seq_forced_tokens(other, g.h, ids, 4, &n, &acc);
				CHECK_MSG(st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && untouched(), "[G12.4] another context: %s", Name(st));
				sslm_gpu_context_destroy(other);
			}
			// Cap 0 with null ids is a count query.
			st = sslm_gpu_seq_forced_tokens(rig.ctx(), g.h, nullptr, 0, &n, &acc);
			CHECK_MSG(st == GS::SSLM_OK && n == static_cast<int32_t>(rig.ref[kLongKey].st[rig.ref[kLongKey].start].chain.size()) &&
			              acc == 0, "[G12.4] cap 0, null ids: %s count %d acc %d", Name(st), n, acc);
			// A Submitted sequence: one layer submitted, not yet drained.
			n = acc = -9;
			ids[0] = -7;
			GS e = sslm_gpu_seq_embed_token(rig.ctx(), g.h, 0);
			if (e == GS::SSLM_OK) e = sslm_decode_step_gpu(rig.ctx(), g.h, nullptr, rig.env.fx.one_layer_budget);
			CHECK_MSG(e == GS::SSLM_OK, "[G12.4] setup: one submitted layer: %s", Name(e));
			if (e == GS::SSLM_OK) {
				st = sslm_gpu_seq_forced_tokens(rig.ctx(), g.h, ids, 4, &n, &acc);
				CHECK_MSG(st == GS::SSLM_BUSY && untouched(), "[G12.4] a Submitted sequence: %s", Name(st));
				Drain(rig.env, g.h);
			}
		}
		// (d) the query changes nothing.
		{
			GSeq g = rig.New(kPotion);
			const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
			rig.Prompt(g, prompt);
			const std::vector<uint8_t> before = rig.Blob(g.h);
			const uint64_t sub0 = Submissions(rig.ctx());
			const GQuery q = QueryGpu(rig.ctx(), g.h, 3);
			const uint64_t sub1 = Submissions(rig.ctx());
			CHECK_MSG(q.st == GS::SSLM_OK, "[G12.4] the query after a prompt: %s", Name(q.st));
			CHECK_MSG(rig.Blob(g.h) == before, "[G12.4] the query changed the SLM5 blob");
			CHECK_MSG(sub1 == sub0, "[G12.4] the query made %llu submissions", static_cast<unsigned long long>(sub1 - sub0));
			// Bindability: reset (eligible again), query, then rebind succeeds.
			CHECK_MSG(sslm_gpu_seq_reset(rig.ctx(), g.h) == GS::SSLM_OK, "[G12.4] reset");
			const GQuery qr = QueryGpu(rig.ctx(), g.h);
			CHECK_MSG(qr.st == GS::SSLM_OK, "[G12.4] the query after a reset: %s", Name(qr.st));
			CHECK_MSG(rig.Bindable(g.h, kLongKey), "[G12.4] the query cleared bind eligibility");
		}
	}
	return Finish("sfdg_cell_g12_4");
}
