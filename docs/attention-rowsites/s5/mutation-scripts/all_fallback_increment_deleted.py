# All slices: the fallback increment deleted.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('''		(kernel == detail::SitesKernel::kAvx2 ? superslm_test::g_q31_row_fallback_avx2
		                                      : superslm_test::g_q31_row_fallback_avx512)
		    .fetch_add(1, std::memory_order_relaxed);
''', '		/* MUTANT */\n')
open(p, "w").write(s)
