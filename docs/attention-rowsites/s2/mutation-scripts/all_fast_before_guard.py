# all: the fast increments moved before the guard (counted at dispatch, removed from the bodies).
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('\tsuperslm_test::g_pv_fast_avx2.fetch_add(1, std::memory_order_relaxed);\n', ''); sub('\tsuperslm_test::g_pv_fast_avx512.fetch_add(1, std::memory_order_relaxed);\n', ''); sub('\t\tcase detail::SitesKernel::kAvx2:\n\t\t\tif (ProbVFastPathAdmits', '\t\tcase detail::SitesKernel::kAvx2:\n\t\t\tsuperslm_test::g_pv_fast_avx2.fetch_add(1);  // MUTANT\n\t\t\tif (ProbVFastPathAdmits'); sub('\t\tcase detail::SitesKernel::kAvx512:\n\t\t\tif (ProbVFastPathAdmits', '\t\tcase detail::SitesKernel::kAvx512:\n\t\t\tsuperslm_test::g_pv_fast_avx512.fetch_add(1);  // MUTANT\n\t\t\tif (ProbVFastPathAdmits')
open(p,'w').write(s)
