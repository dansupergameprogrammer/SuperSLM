// Schema fast-decode plan (rev 1.3.3), slice J1-X (§5.10, §11.4): unimplemented stubs for the 9-argument
// verb and the J1-X seams (sfd_api.h groups 1x and 5), so the suite links before J1-X exists and each
// cell fails at its assertion instead of at the link.
//
// Each stub is the symbol with no behaviour. The verb writes nothing and returns SSLM_STATUS_NEXT_FREE,
// the sentinel no verb ever returns (sslm_abi.h), so a cell reads it as "exact mode unimplemented" and
// no cell can pass on a status the stub happened to share with a real refusal. The seams do nothing,
// and the reads report nothing. They carry no logic for a builder to keep. The J1-X build defines the
// verb in the library (the ABI headers' 9-argument extern "C" form) and the seams beside the §11.0
// seams, and turns SUPERSLM_SFD_X_RED_STUBS OFF in the same commit (CMakeLists.txt, build.bat).

#include "schema_fast_decode/sfd_api.h"

#if defined(SUPERSLM_SFD_X_RED_STUBS)

sslm_status sslm_seq_jump_forward(sslm_model, sslm_seq, int32_t, const sslm_decode_params*, const sslm_jump_options*,
                                  sslm_workspace, int32_t*, int32_t, int32_t*) {
	return SSLM_STATUS_NEXT_FREE;
}

namespace superslm {
namespace test {
namespace sfd {

void SetVerifyForce(VerifyForce) noexcept {}
void ArmSatRow(int32_t) noexcept {}
void DisarmSatRow() noexcept {}
bool SatRowArmed() noexcept { return false; }
bool RowsLogIsCheck(size_t) noexcept { return false; }
void LastCheckRowReset() noexcept {}
size_t LastCheckRow(int32_t*, size_t) noexcept { return 0; }
void SetDraftWalkLegalityDisabled(bool) noexcept {}
void SetHolderUnmapTailStubbed(bool) noexcept {}

}  // namespace sfd
}  // namespace test
}  // namespace superslm

#endif  // SUPERSLM_SFD_X_RED_STUBS
