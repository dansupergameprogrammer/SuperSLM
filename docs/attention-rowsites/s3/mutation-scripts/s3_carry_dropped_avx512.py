# S3 AVX-512 body only: the carry term (127L + 2^(e-1)) >> 32 dropped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm512_add_epi64(_mm512_mul_epu32(h, c127), carry)', '_mm512_add_epi64(_mm512_mul_epu32(h, c127), _mm512_setzero_si512()) /* MUTANT */', start="size_t RequantRowAvx512(", end="}  // namespace")
open(p,'w').write(s)
