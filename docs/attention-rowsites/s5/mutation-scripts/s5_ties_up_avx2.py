# S5: ties toward +inf, AVX2 body (the threshold loses its +1 for negative x).
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm256_add_epi64(_mm256_set1_epi64x(INT64_C(0x3FFFFFFF)), _mm256_srli_epi64(x, 63))', '_mm256_set1_epi64x(INT64_C(0x3FFFFFFF)) /* MUTANT */')
open(p, "w").write(s)
