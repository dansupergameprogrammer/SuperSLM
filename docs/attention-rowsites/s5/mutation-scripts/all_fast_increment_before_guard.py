# All slices: the fast increment moved from the bodies to before the guard.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('\tsuperslm_test::g_q31_row_fast_avx2.fetch_add(1, std::memory_order_relaxed);\n', '\t/* MUTANT */\n')
sub('\tsuperslm_test::g_q31_row_fast_avx512.fetch_add(1, std::memory_order_relaxed);\n', '\t/* MUTANT */\n')
sub('''	if (kernel != detail::SitesKernel::kShipped) {
		if (Q31RowFastPathAdmits(''', '''	if (kernel != detail::SitesKernel::kShipped) {
#ifdef SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT
		(kernel == detail::SitesKernel::kAvx2 ? superslm_test::g_q31_row_fast_avx2 : superslm_test::g_q31_row_fast_avx512)
		    .fetch_add(1, std::memory_order_relaxed); /* MUTANT */
#endif
		if (Q31RowFastPathAdmits(''')
open(p, "w").write(s)
