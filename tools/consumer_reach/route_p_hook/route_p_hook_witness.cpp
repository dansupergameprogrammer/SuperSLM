// Route P hook-state RUNTIME witness (plan rev 6, cell 10.0 H print; C11 witness kind (a)).
// Links the plugin's own SuperSLMFinishHook.cpp, compiled verbatim against UE shims, calls the
// function the plugin installs from (Make), and reads the value it produces. H = "hooked" iff the
// plugin would install a hook at that task count (ShouldInstall) AND the produced `reserved`
// carries the prefill opt-in bit (plan §5.1: bit 0, SSLM_PARALLEL_FOR_PREFILL). Fails closed:
// any other nonzero bit is reported as INVALID (the engine rejects it, G33), never as hooked.
#include "SuperSLMFinishHook.h"
#include <cstdio>
int main() {
    const unsigned kPrefillBit = 1u;
    const int ks[] = {0, 1, 2, 8, 256};
    bool any_hooked = false, any_invalid = false;
    for (int k : ks) {
        sslm_parallel_for h = SuperSLMFinishHook::Make(k);
        bool install = SuperSLMFinishHook::ShouldInstall(k);
        bool invalid = (h.reserved & ~kPrefillBit) != 0;
        bool hooked = install && !invalid && (h.reserved & kPrefillBit);
        any_hooked |= hooked; any_invalid |= invalid && install;
        std::printf("route P: Make(%d) -> reserved=0x%x run=%s max_tasks=%d install=%s prefill hooked=%s%s\n",
                    k, h.reserved, h.run ? "set" : "null", h.max_tasks, install ? "yes" : "no",
                    hooked ? "yes" : "no", invalid ? " (INVALID reserved bits)" : "");
    }
    std::printf("route P: H = %s\n", any_invalid ? "INVALID" : any_hooked ? "hooked" : "not hooked");
    return any_invalid ? 2 : 0;
}
