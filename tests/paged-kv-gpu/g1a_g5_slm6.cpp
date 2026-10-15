// Paged-KV plan (rev 16.2) step G1, group g1a: the G5 cell parts (the GPU 'SLM6' blob).
//
//   1.7 [G5 part]   B's 'SLM6' K/V section is added to the neighbour-invariance verdict.
//   1.8 [G5 part]   'SLM6' equals a fresh handle's at each lifecycle stage.
//
// Both compare the WHOLE saved blob, a superset of its K/V section (the canonical [0, L') gather of
// §3.7 from the host mirror): the two runs share budget, origin and every 'SLM5' field, so the whole
// blob is equal exactly when the K/V section is (notes: plan ambiguity, the reading taken).
//
// RED NOW (LINK): the constructions call the G3 pool verbs and seams and, for 1.8, the G4 prefix
// verbs, all declared and undefined. RED AT G3/G4: a pooled sequence's save is not 'SLM6' until G5's
// writer rule (§3.7 item 10), so the magic checks fail.
// GREEN AT G5: on both GPUs, every blob is 'SLM6' and equal to its operand, debug layer clean.

#include "g1a_constructions.h"

namespace {

using namespace g1a;

// ---- 1.7 [G5] -------------------------------------------------------------------------------------
// The 1.7 construction (g1a_constructions.h: A prefills 40 across 3 pages, B prefills 20, reset A,
// release A, B decodes 16) run with and without A; B's saved blobs are 'SLM6' and byte-equal, beside
// the G3 part's tokens and mirror rows. kills: a reset that fills the whole pool mirror (B's gathered
// rows are zeros in the blob); a gather that walks A's former pages; a save that records the pool's
// page ids instead of the canonical position-major gather.
void Cell17G5() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	const Run17Out with = Run17(fx, true, /*save_b=*/true);
	const Run17Out without = Run17(fx, false, /*save_b=*/true);
	if (!with.ok || !without.ok) return;
	G1A_CHECK(with.tokens == without.tokens);
	G1A_CHECK(with.rows == without.rows);
	G1A_CHECK_MSG(IsMagic(with.blob, "SLM6") && IsMagic(without.blob, "SLM6"),
	              "1.7: B's save is not 'SLM6' (sizes %zu, %zu): G5's writer rule", with.blob.size(), without.blob.size());
	G1A_CHECK_MSG(with.blob == without.blob, "1.7: B's 'SLM6' differs from the run where A never existed");
}

// ---- 1.8 [G5] -------------------------------------------------------------------------------------
// R0's lifecycle on a pooled GPU handle (used: reset between stages) and on fresh handles (every
// reset a release and a create); at every marked stage both holders' saves are 'SLM6' and byte-equal.
// kills: a reset that keeps the old origin or budget field; an adopt onto a used holder whose blob
// gathers a stale page; a save that differs between a used and a fresh handle at the same history.
void Cell18G5() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	GpuPrefixBackend g_used(fx, 64), g_fresh(fx, 64);
	SRun u(g_used, false), f(g_fresh, false);
	std::vector<std::pair<std::string, std::vector<uint8_t>>> ub, fb;
	u.on_mark = [&](const char* stage, void* s) { ub.emplace_back(stage, GpuSave(static_cast<SslmGpuSequenceHandle*>(s))); };
	f.on_mark = [&](const char* stage, void* s) { fb.emplace_back(stage, GpuSave(static_cast<SslmGpuSequenceHandle*>(s))); };
	Lifecycle(u, 128, 300, /*fresh=*/false);
	Lifecycle(f, 128, 300, /*fresh=*/true);
	G1A_CHECK_MSG(u.fail.empty() && f.fail.empty(), "1.8 G5: %s %s", u.fail.c_str(), f.fail.c_str());
	// The lifecycle marks seven stages with a holder (continue8 carries none).
	G1A_CHECK_EQ(ub.size(), 7);
	G1A_CHECK_EQ(fb.size(), ub.size());
	for (size_t i = 0; i < ub.size() && i < fb.size(); ++i) {
		G1A_CHECK(ub[i].first == fb[i].first);
		G1A_CHECK_MSG(IsMagic(ub[i].second, "SLM6") && IsMagic(fb[i].second, "SLM6"),
		              "1.8 %s: the save is not 'SLM6': G5's writer rule", ub[i].first.c_str());
		G1A_CHECK_MSG(ub[i].second == fb[i].second, "1.8 %s: 'SLM6' differs from a fresh handle's", ub[i].first.c_str());
	}
}

}  // namespace

G1A_CELL("1.7/G5", "G5", Cell17G5);
G1A_CELL("1.8/G5", "G5", Cell18G5);
