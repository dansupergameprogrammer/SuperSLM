# AVX2 body: z upward correction skipped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('const __m256i up = _mm256_cmpgt_epi64(r, c.q_ln2_m1);', 'const __m256i up = _mm256_setzero_si256(); /* MUTANT */')
open(p,"w").write(s)
