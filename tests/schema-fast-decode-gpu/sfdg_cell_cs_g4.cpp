// Engine 1.19 clean stop, cell G4 (sfdg_cleanstop.h has the contract): a multi-row EXACT_DRAFT chunk whose draft
// runs past the closing token. Rows [o_j .. o_(e-1), o_(e-1), <|endoftext|>]: nothing is legal after the closing
// token, so admission stops there and the extra rows never enter the chunk. The close finishes the closing row and
// gets -2: rows_produced -2 at SSLM_OK, last_rows_committed is the true rows' count, accepting 1, the context the
// prompt plus e (nothing past the end committed), and the blob equals the Off run's after its -2. M (true rows) in
// {1, 3, 6}, so the chunk spec stays within SSLM_GPU_MAX_ROWS. Device head. Box only.
// Run: sfdg_cell_cs_g4.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		const std::string dir = FixtureDir(argc, argv);
		CRig c(rig.raw);
		Mapped dev;
		if (MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev)) {
			for (const Case& k : kCases) {
				const std::vector<int32_t> prompt = c.Tokenize(k.prompt_text);
				const Off off = RunOff(rig, dev.m, k.schema, prompt);
				CHECK_MSG(off.ok, "[CS.G4] %s: the Off run did not reach its end", k.schema);
				if (!off.ok) continue;
				const size_t e = off.toks.size();
				for (size_t M : {size_t(1), size_t(3), size_t(6)}) {
					if (M > e) continue;
					const size_t j = e - M;
					GSeq s = After(rig, dev.m, k.schema, prompt, j + 1);
					if (!s.h) {
						CHECK_MSG(false, "[CS.G4] %s M%zu: the history did not replay", k.schema, M);
						continue;
					}
					std::vector<int32_t> rows(off.toks.begin() + static_cast<std::ptrdiff_t>(j), off.toks.end());
					rows.push_back(off.toks.back());
					rows.push_back(kSpecial);
					const Exact x = RunExact(rig, s, rows, rig.env.full_budget);
					CHECK_MSG(x.embed == GS::SSLM_OK && x.admitted == static_cast<int32_t>(M), "[CS.G4] %s M%zu: embed_rows %s, "
					          "admitted %d of %zu (want the %zu rows through the closing token)", k.schema, M, Name(x.embed),
					          x.admitted, rows.size(), M);
					CHECK_MSG(x.close == GS::SSLM_OK && x.prod == GS::SSLM_OK && x.token == -2, "[CS.G4] %s M%zu: close %s, "
					          "rows_produced %s token %d (want OK, OK, -2)", k.schema, M, Name(x.close), Name(x.prod), x.token);
					CHECK_MSG(LastCommitted(s.h) == static_cast<int32_t>(M), "[CS.G4] %s M%zu: last_rows_committed %d", k.schema,
					          M, LastCommitted(s.h));
					CHECK_MSG(Accepting(rig.ctx(), s.h) == 1 && ReadyFlag(rig.ctx(), s.h) == 1, "[CS.G4] %s M%zu: accepting %d "
					          "ready %d, want 1 and 1", k.schema, M, Accepting(rig.ctx(), s.h), ReadyFlag(rig.ctx(), s.h));
					CHECK_MSG(CtxLen(s.h) == static_cast<int64_t>(prompt.size() + e), "[CS.G4] %s M%zu: context %lld, want %zu "
					          "(nothing past the end)", k.schema, M, static_cast<long long>(CtxLen(s.h)), prompt.size() + e);
					CHECK_MSG(rig.Blob(s.h) == off.end_blob, "[CS.G4] %s M%zu: the blob differs from the Off run's after its -2",
					          k.schema, M);
				}
				rig.ReleaseSeqs();
			}
			Unmap(rig, &dev);
		}
	}
	return Finish("sfdg_cell_cs_g4");
}
