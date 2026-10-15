// Paged-KV plan rev 16.21 §23.8, step T0: unimplemented stubs for the R9 surface the red suite calls
// (pkv_r9.h), so superslm_pkv_r9 links on the 2ebe5e3 engine and each cell fails at its assertion
// instead of at the link.
//
// Each stub is the verb with no behaviour: a refusal, nothing written through any pointer. They
// carry no logic for a builder to keep. The builder defines the real symbol in the library and
// turns the CMake option guarding its stub OFF in the same commit (tests/paged-kv/paged_kv.cmake);
// a stub left ON beside a real definition is a duplicate symbol, so the link refuses the pair.

#include "pkv_r9.h"

extern "C" {

#if defined(PKV_R9_STUB_CONTIGUOUS)
size_t sslm_kv_pool_contiguous_size(sslm_model, uint32_t) { return 0; }

sslm_status sslm_kv_pool_create_contiguous(sslm_model, void*, size_t, uint32_t, sslm_kv_pool*) {
	return SSLM_INVALID_ARGUMENT;
}
#endif

#if defined(PKV_R9_STUB_PIN)
sslm_status KvPagesTestPinPagePositions(sslm_model, int64_t) { return SSLM_INVALID_ARGUMENT; }
#endif

// The geometry seam lands beside the pin at T1 (rev 16.23, D-SLM8437), so the pin's option guards both.
#if defined(PKV_R9_STUB_PIN) && !defined(PKV_PIN_GENERATED_FIXTURES)
sslm_status KvPagesTestPoolGeometry(const void*, size_t, uint32_t, KvPagesTestPoolGeometryOut*) {
	return SSLM_INVALID_ARGUMENT;
}
#endif

}  // extern "C"
