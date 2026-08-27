// T-2272 strike construction -- a must-reject the fold's own instrument never sees.
//
// Shaped after sslm_abi.cpp's own `CatchAllocationFailure<Lambda>` / model.cpp's
// `WrapBadAllocContract<Lambda>` wrappers, which are what produce every
// `?catch$0@?0???$CatchAllocationFailure@...` funclet the four scan roots reach.
// The floating-point operation is placed where a retry/backoff heuristic naturally
// goes: in the allocation-failure handler.  Nothing here is exotic; it is ordinary
// C++ that the real data path produces, and it lands in the catch FUNCLET -- a
// separate COFF symbol invoked by the runtime unwinder through the parent's .xdata
// scope table, reachable by no call and no jmp instruction anywhere in the image.
//
// Build:  cl /nologo /c /O2 /std:c++20 /EHsc funclet_fp.cpp
#include <new>
#include <cstddef>
#include <vector>

volatile double g_retry_backoff = 0.0;
volatile int g_sink = 0;

extern "C" int RegressionParent(std::size_t n) noexcept {
    try {
        std::vector<int> v;
        v.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            v.push_back(static_cast<int>(i));
        }
        g_sink = v.empty() ? 0 : v[0];
        return 0;
    } catch (const std::bad_alloc&) {
        // Retry backoff proportional to the request that failed.
        g_retry_backoff = static_cast<double>(n) / 1024.0 + 0.5;
        return -1;
    }
}
