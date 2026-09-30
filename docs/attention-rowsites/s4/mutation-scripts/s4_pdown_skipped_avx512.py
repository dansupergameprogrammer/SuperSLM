# AVX-512 body: p downward correction skipped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('const __m512i down = _mm512_srai_epi64(_mm512_sub_epi64(num, prod), 63);', 'const __m512i down = _mm512_setzero_si512(); /* MUTANT */')
open(p,"w").write(s)
