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
//
// RECONCILED 2026-08-22 (S-B..S-E build): every expected symbol below now exists in
// production under these exact names -- sslm_speculate_params / sslm_speculate_step_v3 /
// sslm_speculate_params_init / SSLM_SPECULATE_STOP_* in include/superslm/sslm_abi.h and the
// sole declaration source include/superslm/sslm_abi_functions.inc; superslm::
// SpecdecDraftPropose in include/superslm/specdec_drafter.h; superslm_test::g_inject_
// specdec_fault in tests/support/specdec_injection.h (the bad_alloc_injection.h home, which
// is where a production-consulted test seam lives per that precedent). This header therefore
// INCLUDES those surfaces instead of restating them -- one declaration each, never two --
// and keeps its recording comments below.

#include <stdint.h>

#include "superslm/sslm_abi.h"

#ifdef __cplusplus
#include "superslm/specdec_drafter.h"
#include "../support/specdec_injection.h"
#endif

/* The expected C surface below (stop-reason macros, sslm_speculate_params,
 * sslm_speculate_params_init, sslm_speculate_step_v3, and the retention read-back pair) is
 * now SUPPLIED BY include/superslm/sslm_abi.h, included above -- see the reconciliation note
 * in this file's header comment. The recording comments in the replaced blocks are preserved
 * here so the expectation record survives alongside the production declarations:
 *
 *   - SSLM_SPECULATE_STOP_MAX_TOKENS = 0 / SSLM_SPECULATE_STOP_TOKEN_MATCHED = 1, mirroring
 *     superslm::SslmDecodeStopReason's two shipped values; the C ABI had no shipped
 *     stop-reason type, so the values are pinned to the internal enum's.
 *   - sslm_speculate_params: struct_size FIRST, caller-set to sizeof(...), validated before
 *     anything else; zero-init is NOT a valid params state -- a zero struct_size is exactly
 *     the malformed-input rejection the trust matrix fires on. Fields: max_draft_tokens
 *     (K >= 1), max_new_tokens (remaining budget r; r == 0 is the exhausted-budget entry
 *     arm), stop_ids/stop_count.
 *   - sslm_speculate_step_v3: proposes up to K drafts from retained committed tokens,
 *     verifies integer-exactly against target argmax in ONE chunk-batched pass, emits
 *     accepted drafts plus bonus/divergence under the shipped append-before-stop-test /
 *     stop-then-cap ordering, commits on Ok, restores on rejection. Single sequence by
 *     signature. out_tokens/out_logit_rows are caller-owned (the shipped loop's convention);
 *     row i receives the full int32 logit row composed immediately before the i-th emitted
 *     token was selected, so logit-digest equality versus pure greedy is checkable
 *     byte-for-byte. On any rejection the sequence is byte-identical to pre-call over the
 *     canonicalized save-blob comparison. */

#ifdef __cplusplus

/* The expected S-D drafter unit seam (CM-G2) is now SUPPLIED BY
 * include/superslm/specdec_drafter.h, included above: superslm::SpecdecDraftPropose, the
 * pure function of committed_tokens -- proposes up to k_max ids by longest-suffix match,
 * earliest occurrence winning equal-length ties, extending the matched suffix greedily by
 * its continuation in history; returns the proposal count, 0 when nothing matches (the
 * empty-draft fallback arm's input condition). A fixture whose history repeats no token
 * therefore drives the never-matching-drafter negative control WITHOUT any injection seam.
 *
 * The expected mid-verify non-Ok injection seam (CM-G3) is now SUPPLIED BY
 * tests/support/specdec_injection.h, included above:
 * superslm_test::SpecDecFaultKind (kNone default) plus the single-shot
 * thread_local g_inject_specdec_fault, consulted (and disarmed) by the verify primitive's
 * staged midpoint after at least one KV landing -- one INDEPENDENT slot per region, never a
 * reuse of g_inject_throw (the post-load pin's lesson, D-SLM3466); production consults it
 * only under SUPERSLM_ENABLE_T2246_SPECDEC_FAULT_INJECTION. */
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
