// Paged-KV plan (rev 16.2) §7 cell 3.6b [G3 decode, batch decode, prefill; G4 adopt, begin_from,
// prefix prefill; G5 restore-into-pool and the re-save], group g1c.
//
//   "Sequence A on a pool is Submitted, and holder B on the same pool calls the verb. B gets
//    SSLM_BUSY. After sslm_gpu_ready(A) the cell reads B's state; then B proceeds, and its output is
//    byte-equal. The mutant with no guard (B gets SSLM_OK) fails for every verb."
//
// Each run is g1c_guard_runs.h's GUARDED RUN with A pooled on B's pool (AKind::PooledSamePool), one
// run per submitting verb. The operands per part, as §7 states them:
//   [G3] decode, batch decode: B rests at a page boundary (16). B's state unchanged: the whole GPU
//        mirror byte-equal to the twin's (the GPU mirror peek) and the GPU fill probe unchanged. The
//        guard-at-the-reset-only mutant is EQUIVALENT here (its only pre-reset writes are
//        bind_eligible = false, already false after B's embed, and a page map from B's own reserve),
//        so it is not named. The batch also asserts SSLM_BUSY in every out_statuses slot.
//   [G3] prefill: B fresh, so it can still bind; SslmGpuSeqSetSchemaForG5Bridge(B) after the drain
//        returns SSLM_OK. The reset-only mutant FAILS there: before its reset the prompt prefill
//        revokes bind_eligible, writes live_state and clears prefill_hidden_valid, so SetSchema
//        returns SSLM_SEQUENCE_REJECTED.
//   [G4] adopt, begin_from and prefix prefill, graded as the G3 part, with the prefix MID-PAGE at
//        1,000 so adopt's tail copy shows in the mirror. The reset-only mutant fails for adopt (the
//        tail copy: mirror), for begin_from (its draw: fill probe), and for prefix prefill (on a
//        fresh prefix, sslm_gpu_prefix_set_schema after the drain is SSLM_OK; under the mutant
//        SSLM_GPU_PREFIX_REJECTED, the internal sequence's bind permission revoked).
//   [G5] restore-into-pool, B being the holder the verb would make: the fill probe sees the reset-only
//        mutant's draw. And every verb above whose B is a sequence adds B's re-saved SLM6 blob, equal
//        to the one saved before the refused call (the re-save sees adopt's mutant at 1,008 too, and
//        not the decode verbs').
// Byte-equal output: the twin (the same construction, the call made after the drain) produces the
// statuses and tokens B's proceeding must reproduce.
//
// RED NOW (63e7b0e): BY LINK -- the pool verbs and sslm_gpu_seq_create_budgeted (G3), the prefix verbs
// and sslm_gpu_seq_adopt_prefix (G4), sslm_gpu_seq_restore_into_pool (G5) and the mirror peek (G3) have
// no definition. At run, before G3 a pooled B cannot exist; the no-guard mutant is today's state of
// every existing entry point for a pooled holder (no pool scope exists to refuse it).
// GREEN: each part at its step, on both GPUs.
#include "g1c_guard_runs.h"
// pkvg unification: the parts after PKVG_STEP (g1c_helpers.h) are compiled out.

namespace {
using namespace g1c;

void Run(GuardedSpec g, const char* id) {
	g.name = std::string(id) + " " + g.name;
	GuardedRun(g);
}
// G3 and G4 parts carry no re-save: SLM6 is G5's (§9.3; carrier row CA4-M2).
GuardedSpec NoResave(GuardedSpec g) {
	g.resave_holder = nullptr;
	return g;
}

void G3Decode() { Run(NoResave(SpecDecode(AKind::PooledSamePool)), "3.6b/G3"); }
void G3Batch() { Run(NoResave(SpecBatch(AKind::PooledSamePool)), "3.6b/G3"); }
void G3Prefill() { Run(NoResave(SpecPromptPrefill(AKind::PooledSamePool)), "3.6b/G3"); }
// "One run per submitting verb": the schema-content bridge is the second prefill verb.
void G3SchemaPrefill() { Run(NoResave(SpecSchemaPrefill(AKind::PooledSamePool)), "3.6b/G3"); }

#if PKVG_STEP >= 4
void G4Adopt() { Run(NoResave(SpecAdopt(AKind::PooledSamePool)), "3.6b/G4"); }
void G4BeginFrom() { Run(SpecBeginFrom(AKind::PooledSamePool), "3.6b/G4"); }
void G4PrefixPrefill() { Run(SpecPrefixPrefill(AKind::PooledSamePool), "3.6b/G4"); }

#endif  // PKVG_STEP >= 4
#if PKVG_STEP >= 5
void G5RestoreIntoPool() { Run(SpecRestoreIntoPool(AKind::PooledSamePool), "3.6b/G5"); }
// The re-save added to every verb whose B is a sequence. begin_from's and the prefix prefill's B is a
// prefix, which has no save verb (see the notes: reading taken).
void G5ResaveDecode() { Run(SpecDecode(AKind::PooledSamePool), "3.6b/G5 re-save"); }
void G5ResaveBatch() { Run(SpecBatch(AKind::PooledSamePool), "3.6b/G5 re-save"); }
void G5ResavePrefill() { Run(SpecPromptPrefill(AKind::PooledSamePool), "3.6b/G5 re-save"); }
void G5ResaveSchemaPrefill() { Run(SpecSchemaPrefill(AKind::PooledSamePool), "3.6b/G5 re-save"); }
void G5ResaveAdopt() { Run(SpecAdopt(AKind::PooledSamePool), "3.6b/G5 re-save"); }
// §7: "the re-save sees adopt's mutant at 1,008 too" -- the aligned prefix, where no G4 operand does.
void G5ResaveAdoptAligned() { Run(SpecAdopt(AKind::PooledSamePool, 1008), "3.6b/G5 re-save"); }
#endif  // PKVG_STEP >= 5
}  // namespace

G1C_CELL("3.6b/G3:decode", "G3", G3Decode);
G1C_CELL("3.6b/G3:batch", "G3", G3Batch);
G1C_CELL("3.6b/G3:prefill", "G3", G3Prefill);
G1C_CELL("3.6b/G3:schema_prefill", "G3", G3SchemaPrefill);
#if PKVG_STEP >= 4
G1C_CELL("3.6b/G4:adopt", "G4", G4Adopt);
G1C_CELL("3.6b/G4:begin_from", "G4", G4BeginFrom);
G1C_CELL("3.6b/G4:prefix_prefill", "G4", G4PrefixPrefill);
#endif
#if PKVG_STEP >= 5
G1C_CELL("3.6b/G5:restore_into_pool", "G5", G5RestoreIntoPool);
G1C_CELL("3.6b/G5:resave-decode", "G5", G5ResaveDecode);
G1C_CELL("3.6b/G5:resave-batch", "G5", G5ResaveBatch);
G1C_CELL("3.6b/G5:resave-prefill", "G5", G5ResavePrefill);
G1C_CELL("3.6b/G5:resave-schema_prefill", "G5", G5ResaveSchemaPrefill);
G1C_CELL("3.6b/G5:resave-adopt", "G5", G5ResaveAdopt);
G1C_CELL("3.6b/G5:resave-adopt-1008", "G5", G5ResaveAdoptAligned);
#endif
