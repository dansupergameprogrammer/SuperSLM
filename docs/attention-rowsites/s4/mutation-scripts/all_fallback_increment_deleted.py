# Dispatcher: the fallback increment deleted.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('''\t\t\t(kernel == detail::SitesKernel::kAvx2 ? superslm_test::g_softmax_fallback_avx2
\t\t\t                                      : superslm_test::g_softmax_fallback_avx512)
\t\t\t    .fetch_add(1, std::memory_order_relaxed);
''', '\t\t\t/* MUTANT */\n', 1, start='bool SoftmaxRowQ15(', end='// C32 (§5.2')
open(p,"w").write(s)
