// Engine 1.19 clean stop, cell G7 (sfdg_cleanstop.h has the contract): save and restore across the end. A dedicated
// sequence saves SLM5 (sslm_gpu_seq_restore); a pooled one (sslm_gpu_seq_create_budgeted) saves SLM6
// (sslm_gpu_seq_restore_into_pool). After the -2 the blob is saved; the restored sequence reads accepting 1, saves the
// same bytes, decodes -2 again at SSLM_OK with the bytes unchanged, and resets; the original resets too. Box only.
// Run: sfdg_cell_cs_g7.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

namespace {

void Arm(GRig& rig, const Case& k, const std::vector<int32_t>& prompt, bool pooled) {
	const char* arm = pooled ? "pooled (SLM6)" : "dedicated (SLM5)";
	SslmGpuKvPoolHandle* pool = nullptr;
	GSeq s;
	if (pooled) {
		const int64_t B = rig.env.geo.B > 0 ? rig.env.geo.B : 16;
		const int32_t budget = static_cast<int32_t>(std::min<int64_t>(rig.env.fx.model_cap, int64_t(prompt.size()) + 200));
		const uint32_t pages = static_cast<uint32_t>(3 * ((budget + B - 1) / B) + 8);
		const GS ps = sslm_gpu_kv_pool_create(rig.ctx(), rig.model(), pages, &pool);
		CHECK_MSG(ps == GS::SSLM_OK && pool, "[CS.G7] %s %s: pool of %u pages: %s", arm, k.schema, pages, Name(ps));
		if (!pool) return;
		const GS cs = sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), pool, budget, &s.h);
		const GS bs = cs == GS::SSLM_OK ? SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, rig.Index(k.schema)) : cs;
		CHECK_MSG(bs == GS::SSLM_OK, "[CS.G7] %s %s: budgeted create and bind: %s", arm, k.schema, Name(bs));
	} else {
		s = rig.New(k.schema);
	}
	SslmGpuSequenceHandle* q = nullptr;
	if (s.h && rig.Prompt(s, prompt) == GS::SSLM_OK) {
		const Run r = RunWhole(rig, s);
		CheckCleanStop("CS.G7", arm, k.schema, rig, s, r, prompt.size(), 1);
		if (r.dead) {
			const std::vector<uint8_t>& blob = r.end_blob;
			CHECK_MSG(IsMagic(blob, pooled ? "SLM6" : "SLM5"), "[CS.G7] %s %s: the end blob's magic is not %s", arm, k.schema,
			          pooled ? "SLM6" : "SLM5");
			const GS rs = pooled ? sslm_gpu_seq_restore_into_pool(rig.ctx(), rig.model(), pool, blob.data(), blob.size(),
			                                                      nullptr, &q, nullptr)
			                     : sslm_gpu_seq_restore(rig.ctx(), rig.model(), blob.data(), blob.size(), &q);
			CHECK_MSG(rs == GS::SSLM_OK && q, "[CS.G7] %s %s: restore: %s", arm, k.schema, Name(rs));
			if (q) {
				CHECK_MSG(Accepting(rig.ctx(), q) == 1 && ReadyFlag(rig.ctx(), q) == 1, "[CS.G7] %s %s: the restored sequence "
				          "reads accepting %d ready %d", arm, k.schema, Accepting(rig.ctx(), q), ReadyFlag(rig.ctx(), q));
				CHECK_MSG(rig.Blob(q) == blob, "[CS.G7] %s %s: the restored sequence saves other bytes", arm, k.schema);
				GSeq rq{q};
				rq.last = r.stream.empty() ? 0 : r.stream.back();
				rq.ready = rq.primed = true;
				GS st = GS::SSLM_OK;
				const int32_t t = Token(rig.env, rq, &st);
				CHECK_MSG(st == GS::SSLM_OK && t == -2 && Accepting(rig.ctx(), q) == 1 && rig.Blob(q) == blob,
				          "[CS.G7] %s %s: the restored decode gave %s token %d accepting %d, bytes %s", arm, k.schema, Name(st),
				          t, Accepting(rig.ctx(), q), rig.Blob(q) == blob ? "unchanged" : "changed");
				const GS rq_reset = sslm_gpu_seq_reset(rig.ctx(), q);
				CHECK_MSG(rq_reset == GS::SSLM_OK, "[CS.G7] %s %s: the restored sequence's reset: %s", arm, k.schema,
				          Name(rq_reset));
			}
			const GS s_reset = sslm_gpu_seq_reset(rig.ctx(), s.h);
			CHECK_MSG(s_reset == GS::SSLM_OK, "[CS.G7] %s %s: the original's reset after the end: %s", arm, k.schema,
			          Name(s_reset));
		}
	}
	if (q) sslm_gpu_seq_release(rig.ctx(), q);
	if (pooled) {
		if (s.h) sslm_gpu_seq_release(rig.ctx(), s.h);
		if (pool) sslm_gpu_kv_pool_destroy(rig.ctx(), pool);
	}
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		for (const Case& k : kCases) {
			const std::vector<int32_t> prompt = c.Tokenize(k.prompt_text);
			Arm(rig, k, prompt, /*pooled=*/false);
			Arm(rig, k, prompt, /*pooled=*/true);
			rig.ReleaseSeqs();
		}
	}
	return Finish("sfdg_cell_cs_g7");
}
