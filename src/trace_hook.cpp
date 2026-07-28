// SuperSLM numeric-record trace instrument -- per-model hook state.
//
// See include/superslm/trace_hook.h for the contract (§11 S3.1a; D-SLM353's
// corrected storage). Each function operates on the SslmTraceHookState the
// caller passes in -- the state owned by that caller's own model handle
// (SslmModelView::trace_hook, model.h) -- never a process-wide static.
// Neither function here reads or writes any site's own outputs; both fire
// strictly after a caller has already finished computing them (§10.3's
// instrumentation axis).
#include "superslm/trace_hook.h"

namespace superslm {

void SslmSetTraceHook(SslmTraceHookState& state, SslmTraceHookFn fn, void* user) {
	state.fn = fn;
	// A stale user pointer must not survive an uninstall (fn == nullptr): the
	// next SslmSetTraceHook(state, some_fn, ...) call is what supplies the next
	// valid user pointer, never a pointer left over from a previous
	// installation.
	state.user = (fn != nullptr) ? user : nullptr;
}

bool SslmTraceHookInstalled(const SslmTraceHookState& state) noexcept {
	return state.fn != nullptr;
}

void SslmEmitChainTrace(const SslmTraceHookState& state, const SslmChainTraceRecord& record) {
	if (state.fn == nullptr) return;
	state.fn(&record, nullptr, state.user);
}

void SslmEmitKvLandingTrace(const SslmTraceHookState& state,
                             const SslmKvLandingTraceRecord& record) {
	if (state.fn == nullptr) return;
	state.fn(nullptr, &record, state.user);
}

}  // namespace superslm
