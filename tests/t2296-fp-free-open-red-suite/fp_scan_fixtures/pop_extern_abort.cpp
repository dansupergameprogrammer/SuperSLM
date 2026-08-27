// T-2342 (Curie) -- Sec4.1 gap (c), the external-edge policy: "import-thunk
// and plain-name renderings vet separately, by design, not by oversight"
// (design's own text, citing check_fp_free_scan.py's own _X86_EXTERN_ALLOW,
// which carries "abort" but not "__imp_abort"). This ONE source file,
// compiled twice under two real, different linkage configurations, produces
// the two different renderings of the identical std::abort() call:
//   - /MT (static CRT): a direct relocation to the plain symbol "abort".
//   - Release config, /MD (dynamic CRT, the real shipping configuration):
//     an INDIRECT call through the import-address-table slot for
//     "__imp_abort" -- a different literal string in the object's own
//     symbol table.
// Confirmed by direct execution this session (both linkages, both
// dumpbin-independent capstone decodes). Neither rendering is a convention
// match or a prefix strip in the built instrument's own check (C)
// (`check_fp_free_scan.py:719-778`'s own `target_sym["name"] not in
// extern_allow` is exact-string set membership, confirmed at source) -- this
// cell pins that the two renderings are independently vetted, not silently
// unified by any future shortcut.

#include <cstdlib>

extern "C" void CallAbort() {
    std::abort();
}
