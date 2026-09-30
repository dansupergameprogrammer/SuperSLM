# S3 AVX2 body only: the carry term (127L + 2^(e-1)) >> 32 dropped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm256_add_epi64(_mm256_mul_epu32(h, c127), carry)', '_mm256_add_epi64(_mm256_mul_epu32(h, c127), _mm256_setzero_si256()) /* MUTANT */', start="size_t RequantRowAvx2(", end="SUPERSLM_INTMATH_AVX512_TARGET")
open(p,'w').write(s)
