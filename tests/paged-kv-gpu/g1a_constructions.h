// Paged-KV plan (rev 16.2) step G1, group g1a: constructions shared by a cell's parts at different
// steps (1.7's G3 and G5 parts run one construction), inline so each per-step executable emits only
// what its own cells call.

#ifndef SUPERSLM_TESTS_G1A_CONSTRUCTIONS_H
#define SUPERSLM_TESTS_G1A_CONSTRUCTIONS_H

#include "g1a_helpers.h"

namespace g1a {

// ---- 1.7: GPU neighbour invariance ---------------------------------------------------------------
// Two pooled GPU sequences A and B share one pool. A prefills across 3 pages (40 positions) and B
// prefills 20; A is reset, then released; B decodes 16 tokens. With `with_a` false the same pool
// size holds B alone (the run where A never existed). B's tokens, its K/V rows [0, L) read from the
// pool's host mirror through the GPU mirror peek (§3.1 formula, the table seam), and, when asked, its
// saved blob are returned.
struct Run17Out {
	std::vector<int32_t> tokens;
	std::vector<uint8_t> rows;
	std::vector<uint8_t> blob;  // B's sslm_gpu_seq_save, 'SLM6' from G5
	int64_t L = -1;
	bool ok = false;
};

inline Run17Out Run17(const Fx& fx, bool with_a, bool save_b) {
	Run17Out out;
	GpuPooledBackend gpu(fx, 64);
	if (!gpu.pool) return out;
	SRun d(gpu, true);
	void* a = nullptr;
	void* b = nullptr;
	const int32_t V = fx.vocab;
	bool ok = (!with_a || d.Create(64, &a)) && d.Create(64, &b);
	ok = ok && (!with_a || d.Prefill(a, pkv::Stream(171, 40, V)));  // A across 3 pages
	ok = ok && d.Prefill(b, pkv::Stream(172, 20, V));                // B across 2 pages
	ok = ok && (!with_a || d.Reset(a));
	if (ok && with_a) {
		d.Release(a);
		a = nullptr;
	}
	ok = ok && d.Decode(b, 16) && d.Mark("b16", b);
	G1A_CHECK_MSG(ok && d.fail.empty(), "1.7 (%s A): %s", with_a ? "with" : "without", d.fail.c_str());
	if (ok) {
		out.tokens = d.records.back().tokens;
		out.L = d.records.back().L;
		out.ok = gpu.Rows(b, out.L, &out.rows);
		G1A_CHECK_MSG(out.ok, "1.7: B's rows are unreadable through the GPU mirror peek");
		if (save_b) out.blob = GpuSave(static_cast<SslmGpuSequenceHandle*>(b));
	}
	if (a) d.Release(a);
	if (b) d.Release(b);
	return out;
}

}  // namespace g1a

#endif  // SUPERSLM_TESTS_G1A_CONSTRUCTIONS_H
