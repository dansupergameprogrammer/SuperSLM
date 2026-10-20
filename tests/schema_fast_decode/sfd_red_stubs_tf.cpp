// Schema fast-decode plan (rev 1.3.3), slice TF (§4.5, §11.5): unimplemented stubs for the two seams
// the TF cells call (sfd_api.h group 4), so the suite and the allocation sweep link on the J1 branch
// before TF exists and each cell fails at its assertion instead of at the link.
//
// Each stub is the symbol with no behaviour: S-FINISHEXIT never fails a finish and reports that it
// fired 0 times, and the guard switch does nothing. They carry no logic for a builder to keep. The TF
// build defines the real seams in the library (src/sslm_abi.cpp, beside the §11.0 seams) and turns
// SUPERSLM_SFD_TF_RED_STUBS OFF in the same commit (CMakeLists.txt, build.bat); a stub left ON beside a
// real definition is a duplicate symbol, so the link refuses the pair.

#include "schema_fast_decode/sfd_api.h"

#if defined(SUPERSLM_SFD_TF_RED_STUBS)

namespace superslm {
namespace test {
namespace sfd {

void ArmFinishExit(FinishExit, int32_t) noexcept {}
void DisarmFinishExit() noexcept {}
uint64_t FinishExitFired() noexcept { return 0; }
void SetFinishGuardDisabled(bool) noexcept {}

}  // namespace sfd
}  // namespace test
}  // namespace superslm

#endif  // SUPERSLM_SFD_TF_RED_STUBS
