"""The named mutants of the GPU prefix restore (precached-prefixes plan §7: 11.5's GPU leg, and the
upload and sharing-condition legs of 1.8 and 2.6).

usage: apply_gpu_mutant.py <path to src/gpu/gpu_1p0.cpp> <mutant>

Applies one mutant to the engine source in place, by exact single-occurrence replacement (it refuses when
the anchor text is missing or repeated). Build the cell that kills it with
tests/paged-kv-gpu/build_paged_kv_gpu_mingw.sh (or the .bat), run it, then restore the file with
`git checkout -- src/gpu/gpu_1p0.cpp`. Never commit a mutated tree.

  no-entry-guard   the restore without EntrySubmitGuard           -> pcg_cell_busy (3.4) goes red
  upload-skip      the restore writes the host mirror, no upload   -> pcg_cell_pool (1.8) goes red
  share-skip-f     the hint shared without the row comparison (f)  -> pcg_cell_pool (2.6/5.10) goes red
  freeze-set-removed the freeze-time setting removed (ac2fc2c7's normalisation in sslm_gpu_prefix_freezeImpl:
                   the snapshot and the bind permission left as the call history set them; plan §15.22,
                   rev 1.18.4)              -> pcg_cell_budget_snapshot (6.8) red on legs A, B and D;
                                               pcg_cell_restore_derived (8.16) red on its L = 0 twin
  bind-as-ready    restore's bind permission derived as for L > 0 (bind_eligible = false whatever L)
                                            -> pcg_cell_restore_derived (8.16) red on the L = 0 restore
"""
import sys

GUARD = """	// 2. EntrySubmitGuard (§4.4): the pool upload reaches the singleton; scope any sequence; after the
	// argument refusals, before the blob is read.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kReachesSingleton); g != SSLM_OK) {
		return g;
	}
	// 3-10. The blob, through the SPX1 reader,"""

UPLOAD = """	try {
		superslm_gpu::UploadPoolRegionsSync(dev, pool, regions);
	} catch (...) {
		const SslmGpuStatus fault = ClassifyHandleFault(&dev);
		superslm::kv_pages::HolderRelease(kv);
		return fault;
	}
	// The prefix as saved:"""

SHARE_F = """	if (share) {  // (f)
		const size_t span_bytes = static_cast<size_t>(B) * D;
		for (size_t r = 0; share && r < runs; ++r) {
			const uint8_t* run_src = rows + r * static_cast<size_t>(L) * D;"""

FREEZE_SET = """	SslmGpuSequenceHandle* seq = prefix->seq;
	const bool ready = seq->context_length > 0 && seq->ready_for_logits;  // restore: ready = L > 0 (a savable prefix)
	seq->prefill_hidden_codes = seq->hidden_codes;
	seq->prefill_hidden_valid = ready;
	seq->bind_eligible = seq->context_length == 0;                      // restore: bind_eligible = !(L > 0)
	return SSLM_OK;"""

RESTORE_BIND = """	h->prefill_hidden_valid = ready;
	h->bind_eligible = !ready;"""

M = {
    "no-entry-guard": [(GUARD, "\t// MUTANT no-entry-guard\n\t// 3-10. The blob, through the SPX1 reader,")],
    "upload-skip": [(UPLOAD, "\t(void)dev;  // MUTANT upload-skip\n\t// The prefix as saved:")],
    "share-skip-f": [(SHARE_F, SHARE_F.replace("if (share) {  // (f)", "if (false) {  // MUTANT share-skip-f"))],
    "freeze-set-removed": [(FREEZE_SET, "\t// MUTANT freeze-set-removed\n\treturn SSLM_OK;")],
    "bind-as-ready": [(RESTORE_BIND, "\th->prefill_hidden_valid = ready;\n\th->bind_eligible = false;  // MUTANT bind-as-ready")],
}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2] not in M:
        print(__doc__)
        return 2
    path, name = sys.argv[1], sys.argv[2]
    src = open(path, encoding="utf-8").read()
    for old, new in M[name]:
        n = src.count(old)
        if n != 1:
            print(f"apply_gpu_mutant: {name}: anchor found {n} times (want 1)")
            return 1
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print(f"apply_gpu_mutant: {name} applied to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
