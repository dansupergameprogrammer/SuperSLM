// SuperSLM numeric-record trace instrument -- process-wide hook state.
//
// See include/superslm/trace_hook.h for the contract (§11 S3.1a). This file
// holds the single installed hook (function pointer + opaque user pointer) and
// the two emission seams sites call through. Neither function here reads or
// writes any site's own outputs; both fire strictly after a caller has already
// finished computing them (§10.3's instrumentation axis).
#include "superslm/trace_hook.h"

namespace superslm {

namespace {
SslmTraceHookFn g_trace_hook_fn = nullptr;
void* g_trace_hook_user = nullptr;
}  // namespace

void SslmSetTraceHook(SslmTraceHookFn fn, void* user) {
	g_trace_hook_fn = fn;
	// A stale user pointer must not survive an uninstall (fn == nullptr): the
	// next SslmSetTraceHook(some_fn, ...) call is what supplies the next valid
	// user pointer, never a pointer left over from a previous installation.
	g_trace_hook_user = (fn != nullptr) ? user : nullptr;
}

bool SslmTraceHookInstalled() noexcept { return g_trace_hook_fn != nullptr; }

void SslmEmitChainTrace(const SslmChainTraceRecord& record) {
	if (g_trace_hook_fn == nullptr) return;
	g_trace_hook_fn(&record, nullptr, g_trace_hook_user);
}

void SslmEmitKvLandingTrace(const SslmKvLandingTraceRecord& record) {
	if (g_trace_hook_fn == nullptr) return;
	g_trace_hook_fn(nullptr, &record, g_trace_hook_user);
}

}  // namespace superslm
