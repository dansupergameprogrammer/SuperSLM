# Extra. AVX-512 body: the clip min(a, 30 q_ln2) dropped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('const __m512i a = _mm512_min_epu64(_mm512_sub_epi64(c.peak, s), c.clip);', 'const __m512i a = _mm512_sub_epi64(c.peak, s); /* MUTANT */')
open(p,"w").write(s)
