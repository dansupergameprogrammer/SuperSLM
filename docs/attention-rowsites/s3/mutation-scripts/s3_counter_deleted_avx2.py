# (extra) S3 AVX2 body only: its requant_row increment deleted.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('superslm_test::g_requant_row_avx2.fetch_add(1, std::memory_order_relaxed);', '/* MUTANT */', start="size_t RequantRowAvx2(", end="SUPERSLM_INTMATH_AVX512_TARGET")
open(p,'w').write(s)
