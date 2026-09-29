# all: the fallback increments deleted (both tiers).
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('\t\t\tsuperslm_test::g_pv_fallback_avx2.fetch_add(1, std::memory_order_relaxed);\n', '\t\t\t(void)0;  // MUTANT\n'); sub('\t\t\tsuperslm_test::g_pv_fallback_avx512.fetch_add(1, std::memory_order_relaxed);\n', '\t\t\t(void)0;  // MUTANT\n')
open(p,'w').write(s)
