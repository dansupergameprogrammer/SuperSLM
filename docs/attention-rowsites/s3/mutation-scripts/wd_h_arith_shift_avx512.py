# (withdrawn, §5.3/§9: run to confirm it is equivalent) S3 AVX-512 body only: H by an arithmetic shift.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm512_srli_epi64(p, 32);', '_mm512_srai_epi64(p, 32); /* MUTANT */', start="size_t RequantRowAvx512(", end="}  // namespace")
open(p,'w').write(s)
