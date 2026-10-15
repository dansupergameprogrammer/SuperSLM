// Paged-KV plan (rev 16.2) step G1, group g1a: the G2 cells (the GPU legacy one-page view).
//
//   4.8 [G2 legacy part]  GPU matrix: both certified GPUs; the one-page legacy view (a dedicated
//                         sequence at the model's cap) on every geometry the suite has.
//   6.5a [G2]             GPU legacy one-page view: tokens and K/V byte-equal today's GPU and CPU,
//                         on both vendors.
//
// Both use only verbs that exist on the base tree (sslm_gpu_seq_create and the bridges), so they
// LINK today. They are G2's regression pins: G2 rewrites every K/V address in the shaders and the
// host offset builders through the page table, and these cells hold the dedicated sequence's output
// to what it is before that change.
//
// RED NOW: 6.5a's "today's GPU" operand is a record of this cell's own run on the pre-G2 tree, one
// file per certified GPU (gpu_pre_g2_<fixture>_<adapter>.ref under --gpu-ref=DIR). It does not exist
// yet; the box writes it with --record-gpu-ref on the base build (G1's box run), and until then the
// cell fails on the missing record. 4.8's legacy part and 6.5a's CPU half compare with the R0
// reference and the in-process CPU twin; they are expected green on the base build, and G2 must keep
// them green (a G2 that moves one K/V byte or token turns them red).
// GREEN AT G2: every record equal on both GPUs (run once with SSLM_GPU_ADAPTER_INDEX per GPU).

#include "g1a_helpers.h"

namespace {

using namespace g1a;

// 6.5a's record file for this fixture on this GPU.
std::string GpuRecordPath(const Fx& fx, const std::string& adapter) {
	const std::string dir = EnvOr("SUPERSLM_G1A_GPU_REF_DIR", Opt().gpu_ref_dir);
	G1A_CHECK_MSG(!dir.empty(), "no GPU record directory: pass --gpu-ref=DIR");
	return dir + "/gpu_pre_g2_" + fx.name + "_" + adapter + ".ref";
}

// ---- 6.5a [G2] ------------------------------------------------------------------------------------
// The legacy one-page view (a dedicated sequence at cap 4096, B = cap), on R0's long_prefill and
// widths scenarios (every width of 4.1, cap - 1 and cap included): tokens, context length and the
// K/V rows [0, L) read from the sequence's 'SLM5' blob by the flat formula equal
//   - the R0 reference (v1.11.0's CPU, tests/paged-kv/reference), and
//   - today's GPU: the same run recorded on the pre-G2 build on this GPU.
// kills: a page-table address that is not the identity at B = cap (a K/V byte moves); a shader that
// reads the table for a dedicated sequence and finds no entry (tokens change); a legacy view whose B
// is 16 instead of cap (rows land in the wrong half-stride).
void Cell65a() {
	const std::string adapter = AdapterTag();
	for (const char* stem : {"pkv_def", "pkv_qk"}) {
		const Fx& fx = PkvFixture(stem);
		if (!fx.ok) continue;
		GpuDedicatedBackend gpu(fx);
		SRun lp(gpu, true), wd(gpu, true);
		LongPrefill(lp);
		Widths(wd);
		G1A_CHECK_MSG(lp.fail.empty() && wd.fail.empty(), "6.5a %s: %s%s", stem, lp.fail.c_str(), wd.fail.c_str());
		const std::string what = std::string("6.5a ") + stem + " " + adapter;
		ExpectReference("v1.11.0", fx, "long_prefill", lp.records, Cmp::kTokensAndRows, what + " vs CPU reference");
		ExpectReference("v1.11.0", fx, "widths", wd.records, Cmp::kTokensAndRows, what + " vs CPU reference");
		const std::string path = GpuRecordPath(fx, adapter);
		if (Opt().record_gpu_ref) {
			const std::string header = "writer=pre-G2 GPU fixture=" + fx.name + " sha256=" + fx.sha + " " + adapter;
			G1A_CHECK_MSG(WriteRecords(path, header, "long_prefill", lp.records, false) &&
			                  WriteRecords(path, header, "widths", wd.records, true),
			              "6.5a: cannot write the GPU record %s", path.c_str());
			std::printf("  6.5a: recorded today's GPU at %s\n", path.c_str());
			continue;
		}
		const RefFile today = ReadRef(path);
		// Red until the box records the pre-G2 run (G1's box run, --record-gpu-ref on the base build).
		G1A_CHECK_MSG(today.ok, "6.5a: today's GPU record %s is absent (record it on the pre-G2 build)", path.c_str());
		if (!today.ok) continue;
		G1A_CHECK_MSG(today.fixture_sha == fx.sha, "6.5a: the GPU record was made from another fixture");
		ExpectRecords(today.by_key, "long_prefill", lp.records, Cmp::kTokensAndRows, what + " vs today's GPU");
		ExpectRecords(today.by_key, "widths", wd.records, Cmp::kTokensAndRows, what + " vs today's GPU");
	}
}

// ---- 4.8 [G2: the legacy one-page view] --------------------------------------------------------------
// The matrix's legacy column: on this GPU, a dedicated sequence on each geometry -- pkv_def (D 48),
// pkv_qk (D 128, the direct_qk branch), the real 0.5B (cap 4096) and 1.5B (cap 32768) -- prefills
// 100 tokens and decodes 16; its tokens, context length and K/V rows [0, L) equal a CPU no-budget
// holder's on the same artifact (read from its 'SSB5' save, the flat block, as 1.12.0's CPU cells read a
// no-budget holder: since 1.12.0 the CPU's B is §3.1's rule, not the kv_block_size the GPU geometry here is
// written at, so its pages are not readable at this file's B; g1a_helpers.h CpuRows).
// The real artifacts' CPU is pinned against R0 by 10.1. Run once per certified GPU.
void ShortRun(SRun& d) {
	void* s = nullptr;
	if (!d.Create(static_cast<int32_t>(d.b.fx.geo.context_cap), &s)) return;
	if (d.Prefill(s, pkv::Stream(81, 100, d.b.fx.vocab)) && d.Mark("prefill100", s) && d.Decode(s, 16))
		d.Mark("prefill100+decode16", s);
	d.Release(s);
}

void Cell48Legacy() {
	const std::string adapter = AdapterTag();
	const Fx* fxs[] = {&PkvFixture("pkv_def"), &PkvFixture("pkv_qk"), &Real05(), &Real15()};
	for (const Fx* f : fxs) {
		const Fx& fx = *f;
		if (!fx.ok) continue;
		GpuDedicatedBackend gpu(fx);
		CpuPagedBackend cpu(fx, static_cast<uint32_t>(fx.CapPages()), /*legacy_holders=*/true);
		SRun g(gpu, true), c(cpu, true);
		ShortRun(g);
		ShortRun(c);
		G1A_CHECK_MSG(g.fail.empty() && c.fail.empty(), "4.8 legacy %s: %s%s", fx.name.c_str(), g.fail.c_str(), c.fail.c_str());
		// kills: a legacy view on any geometry or vendor whose addresses are not the flat layout's
		ExpectSameRecords(g.records, c.records, Cmp::kTokensAndRows, "4.8 legacy view " + fx.name + " " + adapter);
	}
}

}  // namespace

G1A_CELL("6.5a/G2", "G2", Cell65a);
G1A_CELL("4.8/G2", "G2", Cell48Legacy);
