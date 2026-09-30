# Extra. AVX2 body: the clip min(a, 30 q_ln2) dropped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('const __m256i a = _mm256_min_epu32(_mm256_or_si256(a_raw, big), c.clip);', 'const __m256i a = a_raw; (void)big; /* MUTANT */')
open(p,"w").write(s)
