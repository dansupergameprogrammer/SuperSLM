// Paged-KV plan (rev 16.2) §7 cell 9.8 [G5]: "GPU old reader, new data: the v1.10.0 (rev 16; was
// v1.9.0) sslm_gpu_seq_restore rejects an SLM6 blob on magic. Run on the box with both binaries."
// The plan's v1.10.0 is v1.12.0 here (BRIEF-COMMON: every "1.10.0" means 1.12.0): the GPU-slice base,
// whose sslm_gpu_seq_restore accepts exactly the magics 'SLM4' and 'SLM5' (src/gpu/superslm_gpu.cpp,
// PeekGpuSeqBlobWorkspaceSize: any other magic is a malformed blob, SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
// before anything is allocated). Group g1e, step G1.
//
// Designed as the CPU 9.3 box leg was (tests/paged-kv/c4_dim9_persistence.cpp Cell93OldReaderNewData,
// SUPERSLM_PKV_93_OUT; c5_dim9_persistence.cpp Cell93Budget): two legs, two binaries.
//   Leg 1 (this cell, the paged build): a pooled budget holder that adopted a 1,000-token prefix and
//     took 100 tokens saves 'SLM6'; the cell checks the magic is neither of the two the old reader
//     accepts, and the header carries the budget and origin. A dedicated twin of the same token
//     history saves 'SLM5' (the writer rule), the old reader's positive control. With
//     SUPERSLM_PKV_GPU_98_OUT set, it writes there:
//       g1e_98_pooled.slm6        the SLM6 blob (the new data)
//       g1e_98_dedicated.slm5     the dedicated twin's SLM5 (the control)
//       g1e_98_expect.txt         "last=<token> next=<t1>,...,<t8>": the dedicated twin's last token
//                                 and its next 8 greedy tokens after the save point
//   Leg 2 (g1e_old_reader_9_8.cpp, built against the v1.12.0 tree, the dedicated restore of the tree
//     before G5): restores the SLM5 control (SSLM_OK, next 8 tokens as recorded: the reader, fixture
//     and token driver work), then the SLM6 (SSLM_SEQUENCE_KV_BUFFER_MISMATCH and a null out-handle:
//     rejected on magic, nothing restored). Run against the G5 binary instead, leg 2 fails (the new
//     reader returns SSLM_GPU_HOLDER_MODE_MISMATCH, A11), so the leg separates the binaries.
// The cloud has neither binary running; this leg's checks are the ones that need only the new build.
//
// Mutants killed: a pooled holder writing SLM5 (an old reader would then read the blob as a dedicated
// sequence sized to L', losing budget and origin: the strike's class on a fourth reader), a new magic
// that collides with 'SLM4'/'SLM5'.
//
// RED NOW: link-red (G3 pool/budgeted create, G4 prefix/adopt, G5 the SLM6 writer); after G4, red at
// run until G5 (the pooled save is not 'SLM6'). Leg 2 is red until leg 1 has written its files.
// GREEN: both legs pass on both GPUs.
//
// Run (box, leg 1):  set SUPERSLM_PKV_GPU_98_OUT=D:\_scratch\pkv-gpu-98 & g1e_cell_9_8.exe --fixture-dir=DIR
#include "g1e_helpers.h"

#include <cstdio>
#include <cstdlib>

using namespace g1e;

namespace {

constexpr int32_t kPrefixLen = 1000;
constexpr int32_t kBudget = 512;
constexpr int32_t kTurn = 100;

bool WriteFile(const std::string& path, const void* p, size_t n) {
	std::FILE* f = std::fopen(path.c_str(), "wb");
	const bool ok = f && std::fwrite(p, 1, n, f) == n;
	if (f) std::fclose(f);
	CHECK_MSG(ok, "9.8: cannot write %s", path.c_str());
	return ok;
}

void Cell98(const Env& e) {
	// The pooled holder.
	std::vector<uint8_t> slm6;
	{
		Rig rig(e, static_cast<uint32_t>(e.geo.PrefixPages(kPrefixLen) + e.geo.R(kBudget) + 4));
		SslmGpuPrefixHandle* px = rig.pool ? BudgetPrefix(rig, PrefixTokens(kPrefixLen)) : nullptr;
		GSeq s = px ? rig.Budgeted(kBudget) : GSeq{};
		if (!s.h) return;
		CHECK(Adopt(rig, s, px, kPrefixLen) == GS::SSLM_OK);
		CHECK(static_cast<int>(DecodeN(e, s, kTurn).size()) == kTurn);
		slm6 = Save(e, s.h);
		Slm6SelfCheck(slm6, e.geo, kBudget, kPrefixLen, ContextLength(s.h), "9.8 pooled save");
	}
	// kills: a magic the v1.12.0 reader accepts (it would read the blob as a dedicated sequence)
	for (const char* old : {"SLM4", "SLM5"})
		CHECK_MSG(!IsMagic(slm6, old), "9.8: the SLM6 blob carries the v1.12.0 reader's magic %s", old);

	// The dedicated twin: the same token history (the prefix's tokens prefilled, then the same turn).
	std::vector<uint8_t> slm5;
	std::vector<int32_t> next8;
	int32_t last = -1;
	{
		SslmGpuSequenceHandle* h = nullptr;
		const GS st = sslm_gpu_seq_create(e.ctx, e.model(), 2048, &h);
		CHECK_MSG(st == GS::SSLM_OK && h, "9.8: sslm_gpu_seq_create: %s", StatusName(st));
		if (!h) return;
		GSeq d;
		d.h = h;
		CHECK(Prefill(e, d, PrefixTokens(kPrefixLen)) == GS::SSLM_OK);
		CHECK(static_cast<int>(DecodeN(e, d, kTurn).size()) == kTurn);
		slm5 = Save(e, h);
		last = d.last;
		next8 = DecodeN(e, d, 8);
		sslm_gpu_seq_release(e.ctx, h);
	}
	CHECK_MSG(IsMagic(slm5, "SLM5"), "9.8: the dedicated twin's save is not 'SLM5' (writer rule)");
	CHECK_MSG(next8.size() == 8, "9.8: the dedicated twin produced %zu of 8 tokens", next8.size());

	const char* dir = std::getenv("SUPERSLM_PKV_GPU_98_OUT");
	if (!dir || !*dir) {
		std::printf("9.8: SUPERSLM_PKV_GPU_98_OUT not set: leg 2's inputs not written (box leg owed)\n");
		return;
	}
	const std::string d = dir;
	WriteFile(d + "/g1e_98_pooled.slm6", slm6.data(), slm6.size());
	WriteFile(d + "/g1e_98_dedicated.slm5", slm5.data(), slm5.size());
	std::string expect = "last=" + std::to_string(last) + " next=";
	for (size_t i = 0; i < next8.size(); ++i) expect += (i ? "," : "") + std::to_string(next8[i]);
	expect += "\n";
	WriteFile(d + "/g1e_98_expect.txt", expect.data(), expect.size());
	std::printf("9.8: leg 2 inputs written to %s (v1.12.0 must restore the SLM5 and refuse the SLM6 with "
	            "SSLM_SEQUENCE_KV_BUFFER_MISMATCH)\n",
	            d.c_str());
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) Cell98(e);
	}
	return FinishSuite("g1e_cell_9_8");
}
