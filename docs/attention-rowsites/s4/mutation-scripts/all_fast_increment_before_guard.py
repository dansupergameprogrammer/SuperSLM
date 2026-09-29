# The fast increment moved from the bodies to before the guard.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('\tsuperslm_test::g_softmax_fast_avx2.fetch_add(1, std::memory_order_relaxed);\n', '\t/* MUTANT */\n')
sub('\tsuperslm_test::g_softmax_fast_avx512.fetch_add(1, std::memory_order_relaxed);\n', '\t/* MUTANT */\n')
sub('\t\t\tint64_t peak = 0;\n', '''#ifdef SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT
\t\t\t(kernel == detail::SitesKernel::kAvx2 ? superslm_test::g_softmax_fast_avx2 : superslm_test::g_softmax_fast_avx512)
\t\t\t    .fetch_add(1, std::memory_order_relaxed); /* MUTANT */
#endif
\t\t\tint64_t peak = 0;
''', 1, start='bool SoftmaxRowQ15(', end='// C32 (§5.2')
open(p,"w").write(s)
