// Paged-KV plan (rev 16.2) §7 cell 3.6d [G3], the FIRST-TOUCH CONTROL, group g1c. Built ONLY in the
// allocation-fault build (SUPERSLM_GPU_ALLOC_FAULT_INJECTION with SUPERSLM_GPU_TEST_SEAMS), where te425's
// hooks live, and run ALONE in a fresh process (pkvg_firsttouch_g3.exe; see build_paged_kv_gpu.bat).
//
//   "first touch (rev 9). In a fresh process, with the device never constructed, the process's first
//    singleton-reaching call is a guarded refusal after the guard: the prompt prefill with count == 0. It
//    returns SSLM_OK. The operand is te425's first-in-process setup-fault hook, E-7 (CreateCommittedResource
//    of the READBACK buffer), in the allocation-fault build: armed before the call, it fires only if the call
//    constructs the device, and its fired count, g_arm.fired (te425 CellFirstTouchDirect), is test-owned. The
//    correct build leaves it at 0. The named mutant is the guard's ledger term read through
//    harness::GetDevice() (rev 8's wording), which constructs the device and fires the hook."
//
// CONSTRUCTION. te425::InstallHooks; the stack (context and the two model maps reach only the context's own
// device, as in te425's FirstTouch); pool 0 and a budgeted holder B on it, the holder the §7 run uses ("on a
// pooled holder B where it takes one"); then, with the hook armed exactly as CellFirstTouchDirect arms it,
// the prompt prefill on B with count == 0.
// OPERANDS: its status SSLM_OK, and g_arm.fired == 0 (test-owned; the GetDevice() mutant fires it).
// ANTI-VACUITY (the control's own check that the operand can see a construction): the hook re-armed, the
// prompt prefill on B with count == 1, which reaches the singleton (§4.4's device column: the prompt prefill
// is "singleton") and so constructs the device: g_arm.fired > 0 is required, or the cell is INVALID, not
// green -- an earlier step of this process already built the device, or the hook does not reach it. That
// leg's status is printed (E-7's own expectation is te425's, not graded here).
//
// RED NOW (63e7b0e): BY LINK -- sslm_gpu_kv_pool_create and sslm_gpu_seq_create_budgeted (G3).
// (On a dedicated holder today's build passes trivially, since no guard exists and count == 0 returns
// before the device; the pooled holder is what the §7 run names, and G3 is when it can exist.)
// GREEN AT G3 when both GPUs give SSLM_OK with fired == 0 and the anti-vacuity leg fires.
#include "g1c_helpers.h"
#include "te425_harness.h"

namespace {
using namespace g1c;

constexpr uint64_t kE7Width = 8192ull * 8ull;  // te425_cells.cpp kTsWidth: kMaxTimestampSlots * sizeof(UINT64)

void ArmE7() {
	te425::HookArmCall(te425::M_CCR, 1, E_OUTOFMEMORY, te425::HB_FAIL_NO_CALL, 1, D3D12_HEAP_TYPE_READBACK, kE7Width);
}

void FirstTouch() {
	std::string why;
	if (!te425::InstallHooks(&why)) {
		G1C_CHECK(false, "SETUP: te425::InstallHooks: %s", why.c_str());
		return;
	}
	Stack* s = GetStack();
	if (!s) return;
	World w(*s);
	w.Pool(16);
	SslmGpuSequenceHandle* b = w.Pooled(0, 32);
	if (!w.ok || !b) return;
	const std::vector<int32_t> toks = Toks(s->art, 4, 61);

	ArmE7();
	const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(s->ctx, b, toks.data(), 0, s->one_layer);
	const uint32_t fired = te425::g_arm.fired.load();
	te425::HookDisarm();
	G1C_CHECK(st == SSLM_OK, "first touch: prompt prefill count == 0 -> %s, want SSLM_OK", St(st));
	G1C_CHECK(fired == 0, "first touch: the E-7 hook fired %u time(s): the refused call constructed the device "
	                      "(the GetDevice() mutant)", fired);

	// Anti-vacuity: the operand must be able to see a construction in this process.
	ArmE7();
	const SslmGpuStatus st1 = SslmGpuSeqPrefillPromptForG5Bridge(s->ctx, b, toks.data(), 1, s->one_layer);
	const uint32_t fired1 = te425::g_arm.fired.load();
	te425::HookDisarm();
	std::printf("  anti-vacuity: prompt prefill count == 1 with E-7 armed -> %s, fired %u\n", St(st1), fired1);
	G1C_CHECK(fired1 > 0, "INVALID: the count == 1 prefill did not fire E-7, so fired == 0 above proves nothing "
	                      "(the device was built earlier in this process, or the hook does not reach it)");
	if (st1 == SSLM_OK) (void)Ready(s->ctx, b);
}
}  // namespace

G1C_CELL("3.6d/G3:first_touch", "G3", FirstTouch);
