# Extra, AVX-512 body: floor by a logical shift.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('const __m512i floor = _mm512_srai_epi64(x, 31);', 'const __m512i floor = _mm512_srli_epi64(x, 31); /* MUTANT */')
open(p, "w").write(s)
