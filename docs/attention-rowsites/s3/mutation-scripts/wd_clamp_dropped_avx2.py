# (withdrawn, §5.3/§9: run to confirm it is equivalent) S3 AVX2 body only: the clamp at 127 dropped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm256_min_epu32(_mm256_or_si256(mag, big), c127);', 'mag; /* MUTANT */', start="size_t RequantRowAvx2(", end="SUPERSLM_INTMATH_AVX512_TARGET")
open(p,'w').write(s)
