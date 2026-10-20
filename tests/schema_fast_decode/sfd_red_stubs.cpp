// Schema fast-decode plan (rev 1.2.1), slices M1 and J1: unimplemented stubs for the surface the red
// suite calls (sfd_api.h), so the suite links on main (2e12a6e) and each cell fails at its
// assertion instead of at the link.
//
// Each stub is the symbol with no behaviour: nothing written through any pointer, an empty log, and
// for the two verbs a status no cell ever expects of them (SSLM_SCHEMA_UNSATISFIABLE is the
// converter-side compiler rejection, never a verb's), so a refusal cell cannot pass by accident. They
// carry no logic for a builder to keep. The builder defines the real symbols in the library and turns
// SUPERSLM_SFD_RED_STUBS OFF in the same commit (CMakeLists.txt); a stub left ON beside a real
// definition is a duplicate symbol, so the link refuses the pair.

#include "schema_fast_decode/sfd_api.h"

#if defined(SUPERSLM_SFD_RED_STUBS)

extern "C" {

sslm_status sslm_seq_forced_tokens(sslm_model, sslm_seq, int32_t*, int32_t, int32_t*, int32_t*) {
	return SSLM_SCHEMA_UNSATISFIABLE;
}

sslm_status sslm_seq_jump_forward(sslm_model, sslm_seq, int32_t, const sslm_decode_params*, sslm_workspace,
                                  int32_t*, int32_t, int32_t*) {
	return SSLM_SCHEMA_UNSATISFIABLE;
}

}  // extern "C"

namespace superslm {

bool LegalRowFinishExact(size_t) noexcept { return false; }

SslmForwardStatus LogitsSiteLegalRows(const int8_t*, size_t, const int8_t*, size_t, const uint32_t*, size_t,
                                      int64_t*, int32_t*, const sslm_parallel_for*) {
	return SslmForwardStatus::WorkspaceTooSmall;
}

namespace test {
namespace sfd {

void RowsLogReset() noexcept {}
size_t RowsLogCount() noexcept { return 0; }
uint64_t RowsLogAt(size_t) noexcept { return 0; }

void LayerRowsLogReset() noexcept {}
size_t LayerRowsLogCount() noexcept { return 0; }
LayerRowsEntry LayerRowsLogAt(size_t) noexcept { return LayerRowsEntry{}; }

void LastFinishRowReset() noexcept {}
size_t LastFinishRow(int32_t*, size_t) noexcept { return 0; }

void ArmChoice(int32_t) noexcept {}
void DisarmChoice() noexcept {}
bool ChoiceArmed() noexcept { return false; }

void SetExactnessHiddenBound(size_t) noexcept {}
void ResetExactnessHiddenBound() noexcept {}

void WalkCountReset() noexcept {}
uint64_t WalkCount() noexcept { return 0; }

sslm_status ForcedTableEntry(sslm_model, int32_t, uint32_t, int32_t*, uint32_t*) noexcept {
	return SSLM_SCHEMA_UNSATISFIABLE;
}
sslm_status CorruptForcedTableEntry(sslm_model, int32_t, uint32_t, int32_t) noexcept {
	return SSLM_SCHEMA_UNSATISFIABLE;
}

}  // namespace sfd
}  // namespace test
}  // namespace superslm

#endif  // SUPERSLM_SFD_RED_STUBS
