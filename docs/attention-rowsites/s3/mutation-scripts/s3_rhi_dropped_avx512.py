# (extra) S3 AVX-512 body only: P formed from r's low half alone (wrong only where r = 2^32).
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm512_add_epi64(_mm512_mul_epu32(ax, r_lo), _mm512_slli_epi64(_mm512_mul_epu32(ax, r_hi), 32))', '_mm512_mul_epu32(ax, r_lo) /* MUTANT */', start="size_t RequantRowAvx512(", end="}  // namespace")
open(p,'w').write(s)
