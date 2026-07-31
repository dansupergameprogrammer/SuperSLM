// SuperSLM numeric-record trace instrument (SuperSLM_S3a_WalkingSkeleton_Plan.md
// §11 S3.1a, §10.2's trace-row exclusion, §10.3's instrumentation axis;
// Claude/Curie/superslm-s3.1a-trace-hook-test-design-2026-07-28.md §4.1-§4.2).
//
// The record schema is pinned field-for-field against the reference's own chain
// records (Tools/superslm_spike/dynamic_engine.py:172-203, `_chain_record_vec`,
// `trace.append` at :197-202) and its K/V-landing sibling (:367-397, inside
// `_forward_dynamic_vec_layers`). The reference's conformance contract is
// bit-equality on logits AND trace, record-for-record (dynamic_engine.py:5-7);
// this header is what makes that contract checkable in C++.
//
// THIS IS A DIFFERENT INSTRUMENT from the master plan's profiling hook
// (SuperSLM_Plan.md:1436, `sslm_set_trace_hook(model, fn, user)` carrying span
// identity and `sslm_stats` counters at kernel-phase boundaries). That hook
// carries execution-shape telemetry; this one carries the numeric chain/landing
// records the reference's own trace produces. Neither this file nor any site it
// touches builds the master plan's instrument (§10.3). Being a different
// instrument does not exempt this one from §3's Layer-1-wide "no global state"
// law, which is not scoped to the profiling hook -- it is scoped to all of
// Layer 1 (Claude/Vitruvius/SuperSLM_S3.1a_TraceHookGlobal_Ruling-2026-07-28.md
// Sec3/Sec4, D-SLM353). That is why the hook's state below is owned per model
// handle (SslmModelView::trace_hook, model.h) and threaded as an explicit
// parameter, never held in a process-wide static.
//
// Read-only by construction: installing a hook adds a notification after a
// site's own arithmetic has already produced its outputs. No hook call
// participates in, or can change, any value a site returns or writes. This is
// what §10.3's instrumentation axis requires -- a forward with the hook
// installed produces digests identical to one without.
#ifndef SUPERSLM_TRACE_HOOK_H
#define SUPERSLM_TRACE_HOOK_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace superslm {

// One per-token-per-site chain record: MaxAbsReduceWide -> D' -> NormalizeScale
// / DynamicScaleReciprocal -> RequantTokenCodeWide per element -> carried-scale
// product (RequantChainChecked's own steps, checked_chain_funnel.h). Field-for-
// field against the reference's trace dict: site, token_index, x_int, Dprime,
// Dn, s, R, codes, m_out, e_out.
//
// `x_int` and `codes` are non-owning views into the caller's own row/output
// buffers -- valid only for the duration of the hook call that carries them; a
// hook that needs the data after returning copies it.
struct SslmChainTraceRecord {
	std::string_view site;
	size_t token_index = 0;
	std::span<const int64_t> x_int;  // the wide row, as read at the site
	int64_t d_prime = 0;
	int64_t dn = 0;
	int32_t s = 0;
	int64_t r = 0;
	std::span<const int8_t> codes;
	int64_t m_out = 0;
	int64_t e_out = 0;
};

// The K/V-landing sibling (dynamic_engine.py:367-397): one per (head,
// projection) per token -- K then V, the reference's own emission order --
// same shape with `head`, `m_in`, `e_in` in place of `Dprime`/`Dn`/`s`/`R`.
// Emitted by RunLayerLoop's K/V landing loop (forward_sites.cpp; T-1412),
// gated on an installed hook and on the layer's trace-only landing-target
// arrays being wired (LayerWeights' kv_landing_m_target_*/e_target_* doc).
// `m_out`/`e_out` carry KvLandingScales' static per-head target, matching
// the reference's `kv_landing_scales` unpack -- never a runtime-derived
// scale.
struct SslmKvLandingTraceRecord {
	std::string_view site;
	size_t token_index = 0;
	uint32_t head = 0;
	std::span<const int64_t> x_int;  // the head's folded segment
	int64_t m_in = 0;
	int64_t e_in = 0;
	std::span<const int8_t> codes;
	int64_t m_out = 0;
	int64_t e_out = 0;
};

// A hook receives exactly one of the two record pointers non-null per call --
// the chain record for a chain-funnel emission, the K/V-landing record for a
// landing-site emission -- and the other nullptr. Never both, never neither.
using SslmTraceHookFn = void (*)(const SslmChainTraceRecord*,
                                  const SslmKvLandingTraceRecord*, void* user);

// Per-model trace-hook state (D-SLM353, corrected storage): a function
// pointer + opaque user pointer, exactly the same pair the process-global it
// replaces used to hold, now owned by the model handle instead of a file
// static. Two model handles never share one instance -- each carries its own
// (SslmModelView::trace_hook, model.h) -- so installing a hook through one
// handle cannot install, uninstall, or redirect another handle's hook.
// Default-constructed to "no hook installed", the same state
// `SslmSetTraceHook(state, nullptr, nullptr)` produces.
struct SslmTraceHookState {
	SslmTraceHookFn fn = nullptr;
	void* user = nullptr;
};

// Installs (or, with fn == nullptr, uninstalls) the trace hook carried by
// `state` -- the caller's own model handle's trace-hook state, never a
// process-wide static. `user` is opaque and passed back unchanged on every
// call through this same `state`; it is discarded (reset to nullptr) when fn
// is nullptr, so a stale user pointer never survives past an uninstall.
void SslmSetTraceHook(SslmTraceHookState& state, SslmTraceHookFn fn, void* user);

// True when `state` currently carries an installed hook. Emission sites
// branch on this before doing any work to build a record, so with no hook
// installed a site performs no extra computation and touches no sink (§11
// S3.1a's third red cell: "a forward with no hook installed emits no records
// and touches no sink").
bool SslmTraceHookInstalled(const SslmTraceHookState& state) noexcept;

// Invokes the hook installed in `state`, if any, with a chain record (the
// K/V-landing pointer is passed as nullptr). No-op when `state` carries no
// hook. Emission sites call through this rather than touching `state.fn`
// directly, so the "installed" check and the call are always the same path.
void SslmEmitChainTrace(const SslmTraceHookState& state, const SslmChainTraceRecord& record);

// The K/V-landing sibling of SslmEmitChainTrace. Called by RunLayerLoop's
// K/V landing loop (forward_sites.cpp; T-1412) — one record per (head,
// projection) per token, K then V, under the gating the record struct's own
// comment above states.
void SslmEmitKvLandingTrace(const SslmTraceHookState& state,
                             const SslmKvLandingTraceRecord& record);

}  // namespace superslm

#endif  // SUPERSLM_TRACE_HOOK_H
