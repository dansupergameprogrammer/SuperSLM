/* MUST-ACCEPT (real-engine-headers leg, design Sec11 dim 7, added at the T-2112 mini-fold
 * 2026-08-15; wired as a STANDING build-gate cell at T-2113 B9): sslm_gpu_1p0.h included
 * alongside the real superslm/model.h under `using namespace superslm;` -- the exact shape a
 * real fixture-loading translation unit takes (fixture_common.h's own convention, T-2112 Sec2).
 * Proves the SslmModelView forward-declaration fix (design Sec4.1: `namespace superslm { struct
 * SslmModelView; } using superslm::SslmModelView;`) resolves to the SAME type as the real
 * superslm::SslmModelView this header defines -- no ambiguous-symbol collision (MSVC C2872),
 * which the PRE-fold global-scope stub (`typedef struct SslmModelView SslmModelView;`) produced
 * the first time a real translation unit combined both headers (T-2112 Finding 1). Commissioned
 * against the prefold header (../prefold/sslm_gpu_1p0.h) before this cell was trusted as a
 * standing build gate -- see interface_probe/commission_probe.bat's own real-engine-headers leg,
 * added the same session -- confirming it reproduces the real C2872 defect, not merely compiles
 * against a header it can never fail against. If THIS cell fails to compile against the real,
 * post-fold header, the interface's own namespace-scope fix has regressed -- not a claim this
 * build makes without checking, StandardsDocument.md Sec5.4. Uses sslm_gpu_adapter_map (no
 * by-value config struct parameter) rather than sslm_gpu_model_map, deliberately -- the suite's
 * own canonical header leaves GpuContextConfig/GpuResidencyConfig incomplete by design ("the
 * build seat defines alongside B1/B2"), so a cell whose whole job is proving the SslmModelView
 * fix should not entangle itself with a second, unrelated incomplete-type concern. */
#include "sslm_gpu_1p0.h"
#include "superslm/model.h"
using namespace superslm;

int cell_real_engine_headers(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                              const SslmModelView* adapter_artifact,
                              SslmGpuAdapterHandle** out_adapter) {
	SslmGpuStatus st = sslm_gpu_adapter_map(ctx, model, adapter_artifact, out_adapter);
	return st == SSLM_OK || st == SSLM_ADAPTER_BASE_HASH_MISMATCH || st == SSLM_DEVICE_LOST;
}
