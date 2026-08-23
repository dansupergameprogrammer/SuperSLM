#ifndef SSLM_T2246_SPECDEC_RED_CONTRACT_H
#define SSLM_T2246_SPECDEC_RED_CONTRACT_H

// T-2246 (test design) -- PROMOTED EXPECTED-SURFACE HEADER for the speculative-decoding
// mechanism (plan Claude/Plans/SuperSLM_SpecDecoding_SubPlan_2026-08-22.md r6, SS5 S-E /
// SS3 / SS6 Coverage Model). This file plays the same role tests/t2138-abi-red-suite/sslm_abi.h
// played for the Layer-1 CPU ABI: it is the suite's own declared copy of the surface under
// contract, authored BEFORE the implementation exists, so every cell compiles clean today and
// fails RED BY LINK (LNK2019) on exactly the symbols the mechanism has yet to land.
//
// Every declaration below is an EXPECTATION RECORDED BY THE TEST AUTHOR, derived from the plan's
// pinned requirements -- not a shipped fact:
//   - sslm_speculate_params / sslm_speculate_step_v3 / sslm_speculate_params_init:
//     versioned-additive `_v3` verbs on the `_v2` params-struct_size pattern
//     (include/superslm/sslm_abi.h:207-224 precedent), single-sequence by signature (plan SS4:
//     `sslm_seq* seq`, scalar; batched speculation ruled out).
//   - Emission walk semantics the calls pin: append-before-stop-test, stop-test-then-cap-test
//     per emitted position, truncation to remaining budget, zero-budget entry arm
//     (forward_sites.cpp:2592-2630; plan SS3.2).
//   - Retention read-back pair: the plan asserts `committed_tokens` sizes DIRECTLY at lifecycle
//     checkpoints (SS3.0 table, SS5 S-A retention cells, SS6 row 3) and rules the state
//     NOT serialized (SS4 persistence ruling), so some read-back surface is required for any
//     direct assertion; the plan names none (routed to the planner -- casebook finding CM-G1).
//   - superslm::SpecdecDraftPropose: the S-D drafter's direct pure-function unit seam (plan
//     SS3.1 "pure function of committed_tokens"; SS5 S-A drafter unit cells). Internal linkage
//     shape follows the house pattern of testing engine internals through include/superslm/*.h
//     declarations (superslm::RunGreedyDecodeLoop precedent). Naming routed to the planner --
//     casebook finding CM-G2.
//   - superslm_test::g_inject_specdec_fault: the mid-verify non-Ok injection seam the failure-
//     path cell requires (plan SS3.3 staging rule's own red cell; SS5 S-A "injected mid-verify
//     non-Ok"). Shape follows tests/support/bad_alloc_injection.h's own thread-local kind-flag
//     convention. Wiring owner routed to the planner -- casebook finding CM-G3.
//
// RECONCILIATION RULE: when S-E/S-D/S-B land the real names, the builder either implements
// these names or renames HERE, in this file, in the same commit that lands the production
// symbol -- never silently. A rename without this header moving breaks the whole suite's link
// loudly, which is the point of recording the expectation.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stop reasons reported through sslm_speculate_step_v3's out_stop_reason, mirroring
 * superslm::SslmDecodeStopReason's two shipped values (include/superslm/forward_sites.h:1070-1073,
 * MaxTokensReached = 0, StopTokenMatched = 1). Declared here because the C ABI has no shipped
 * stop-reason type yet; values pinned to the internal enum's so one mapping serves both sides. */
#define SSLM_SPECULATE_STOP_MAX_TOKENS 0
#define SSLM_SPECULATE_STOP_TOKEN_MATCHED 1

/* Versioned-additive params struct on the sslm_decode_params pattern (sslm_abi.h:207-224):
 * `struct_size` is the FIRST field, caller-set to sizeof(sslm_speculate_params), validated
 * before anything else (the trust-boundary matrix's malformed-struct_size rejection, plan SS6
 * Trust boundaries row). Zero-init is NOT a valid params state here (unlike sslm_detok_state):
 * a zero struct_size is exactly the malformed-input rejection the matrix fires on. */
typedef struct sslm_speculate_params {
    uint32_t struct_size;        /* caller sets sizeof(sslm_speculate_params) */
    int32_t max_draft_tokens;    /* K >= 1: max drafts proposed per call (plan SS3.1) */
    int32_t max_new_tokens;      /* remaining emission budget r for this call's walk (SS3.2);
                                  * r == 0 is the exhausted-budget entry arm: emits nothing,
                                  * reports SSLM_SPECULATE_STOP_MAX_TOKENS, no forward pass */
    const int32_t* stop_ids;     /* per-emitted-token stop set, composed per SS3.2's ordered
                                  * walk; may be null when stop_count == 0 */
    int32_t stop_count;
} sslm_speculate_params;

/* Fills a KNOWN-VALID params shape (max_draft_tokens defaulted, budget defaulted generous,
 * no stop set) -- the sslm_decode_params_init precedent (sslm_abi_functions.inc:30-31). */
sslm_status sslm_speculate_params_init(sslm_model model, sslm_speculate_params* out);

/* The `_v3` speculate step: proposes up to params->max_draft_tokens drafts from the sequence's
 * retained committed tokens (SS3.0/SS3.1), verifies them integer-exactly against the target
 * argmax in ONE chunk-batched pass (SS3.2), emits accepted drafts plus the bonus/divergence
 * target applying the shipped per-token stop/cap tests in order, commits per SS3.4, and rolls
 * back per SS3.3 on mismatch. Single sequence by signature (plan SS4 ruling).
 *
 * out_tokens receives up to out_tokens_capacity ids; out_logit_rows receives the emitted
 * tokens' target-argmax logit rows, row-major, up to out_rows_capacity elements -- the
 * caller-owned-rows convention the shipped loop itself establishes
 * (include/superslm/forward_sites.h:1102-1109); WITHOUT some emitted-row observation surface,
 * plan SS1's pinned logit-row digest (D-SLM3795 mapping: ComputeFinalLogitDigest covers those
 * rows) is unassertable by any caller. The plan never states this surface explicitly -- routed
 * to the planner as casebook finding CM-G4; this signature records the suite's expectation.
 * *out_tokens_produced counts emissions; *out_stop_reason receives one SSLM_SPECULATE_STOP_*
 * value. On any rejection the call leaves the sequence byte-identical to its pre-call state
 * over the canonicalized save-blob comparison (SS3.3 canonicalization) -- the containment
 * property dim07_failure_red.cpp asserts. */
sslm_status sslm_speculate_step_v3(sslm_model model, sslm_seq seq,
                                   const sslm_speculate_params* params, sslm_workspace ws,
                                   int32_t* out_tokens, int32_t out_tokens_capacity,
                                   int32_t* out_logit_rows, int32_t out_rows_capacity,
                                   int32_t* out_tokens_produced, int32_t* out_stop_reason);

/* Committed-token retention read-back (expected surface -- see header comment, CM-G1).
 * count: current retained id count (the quantity every SS3.0 lifecycle checkpoint asserts).
 * peek: copies up to *io_count retained ids starting at start_index into out_ids, setting
 * *io_count to the number written; reading past the retained count rejects. */
sslm_status sslm_seq_committed_token_count(sslm_seq seq, int64_t* out_count);
sslm_status sslm_seq_committed_tokens_peek(sslm_seq seq, int64_t start_index,
                                           int32_t* out_ids, int64_t* io_count);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <vector>

namespace superslm {

/* Expected S-D drafter unit seam (CM-G2): the pure function of committed_tokens plan SS3.1
 * defines -- proposes up to k_max ids by longest-suffix match over committed_tokens, earliest
 * occurrence winning equal-length ties (C16-style lowest-index discipline), extending the
 * matched suffix greedily by its continuation in history. Returns the number of proposals
 * written to *out_drafts (0 when nothing matches -- the empty-draft fallback arm's input
 * condition, SS6 Failure paths row). A fixture whose history repeats no token therefore drives
 * the never-matching-drafter negative control WITHOUT any injection seam. */
size_t SpecdecDraftPropose(const std::vector<int32_t>& committed_tokens, int32_t k_max,
                           std::vector<int32_t>* out_drafts);

}  // namespace superslm

namespace superslm_test {

/* Expected mid-verify non-Ok injection seam (CM-G3), bad_alloc_injection.h's shape: the test
 * arms the kind before the call under test; the verify primitive consults (and disarms) it at
 * the staged midpoint of its chunk walk. kNone is the unarmed default. */
enum class SpecDecFaultKind {
    kNone = 0,
    kNonOkMidVerify,  /* a genuine non-Ok raised between per-position stages, after at least
                       * one KV landing -- the granularity forward_sites.cpp:2085-2097 names */
};

inline thread_local SpecDecFaultKind g_inject_specdec_fault = SpecDecFaultKind::kNone;

}  // namespace superslm_test
#endif  // __cplusplus

/* EXPECTED-MISSING SYMBOL SET (machine-checked by build_link_red.bat): every unresolved
 * external a pre-build link of any cell in this suite produces MUST name one of:
 *   sslm_speculate_params_init
 *   sslm_speculate_step_v3
 *   sslm_seq_committed_token_count
 *   sslm_seq_committed_tokens_peek
 *   SpecdecDraftPropose            (C++ mangled; substring-matched)
 *   g_inject_specdec_fault         (C++ mangled; substring-matched)
 * A LNK2019 naming anything else is a defect (a stale suite source list or an unexpected
 * production-symbol gap), never this suite's expected pre-build state -- the N1 lesson from
 * tests/t2138-abi-red-suite/build_link_red.bat applied symbol-scoped instead of absorbed.
 *
 * PLANNER ROUTINGS recorded while authoring (casebook CM-G1..CM-G4):
 *   CM-G1  retention read-back surface unnamed in the plan (this header declares the pair).
 *   CM-G2  drafter direct-unit seam unnamed (SpecdecDraftPropose declared here).
 *   CM-G3  mid-verify non-Ok injection seam unowned (g_inject_specdec_fault declared here).
 *   CM-G4  emitted logit-row observability at the _v3 boundary unstated in the plan
 *          (out_logit_rows declared here, mirroring the shipped loop's caller-owned rows). */

#endif  // SSLM_T2246_SPECDEC_RED_CONTRACT_H
