# S5: ties toward +inf, AVX-512 body.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm512_add_epi64(_mm512_set1_epi64(INT64_C(0x3FFFFFFF)), _mm512_srli_epi64(x, 63))', '_mm512_set1_epi64(INT64_C(0x3FFFFFFF)) /* MUTANT */')
open(p, "w").write(s)
