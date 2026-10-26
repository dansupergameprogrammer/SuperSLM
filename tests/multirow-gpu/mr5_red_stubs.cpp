// Multi-row GPU plan (rev 1.3.2), slice MR5: no-behaviour stand-ins for the MR5 names the library does not define yet
// (mr5_entry.h has the contract). Linked into mr5_cells.exe only. The runner passes -DMR5_HAVE_<NAME> for each name
// src/ already defines, so no stub ever sits next to the real one. Every stub does nothing, and the counter of
// stubbed seams lets a cell say "unimplemented" instead of a misleading verdict.
#include "mr5_entry.h"

namespace {
int32_t g_stubbed = 0;
}

#if !defined(MR5_HAVE_ROWS_PRODUCED)
SslmGpuStatus sslm_gpu_seq_rows_produced(const SslmGpuSequenceHandle*, int32_t*) noexcept { return MR5_STUB_STATUS; }
#endif
#if !defined(MR5_HAVE_SATROW)
extern "C" void ArmGpuSatRowForTest(int32_t) noexcept { g_stubbed |= 1; }
extern "C" int32_t GpuSatRowArmedForTest() noexcept { return 1; }  // never consumed: the cell sees the seam never fired
#endif
#if !defined(MR5_HAVE_NARROW)
extern "C" void ArmGpuFinishNarrowFaultForTest(int32_t) noexcept { g_stubbed |= 2; }
#endif
#if !defined(MR5_HAVE_READBACK)
extern "C" void ArmGpuExactReadbackFaultForTest(int32_t, int32_t) noexcept { g_stubbed |= 4; }
#endif
#if !defined(MR5_HAVE_VERIFY)
extern "C" void SetGpuExactVerifyAlwaysEqualForTest(int32_t) noexcept { g_stubbed |= 8; }
#endif
#if !defined(MR5_HAVE_PASSLOG)
extern "C" void GpuExactPassLogResetForTest() noexcept { g_stubbed |= 16; }
extern "C" int32_t GpuExactPassCountForTest() noexcept { return -1; }
#endif
extern "C" int32_t Mr5StubbedSeamsForTest() noexcept { return g_stubbed; }
