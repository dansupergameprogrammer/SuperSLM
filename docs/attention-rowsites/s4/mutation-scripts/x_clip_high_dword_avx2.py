# Extra. AVX2 body: the high dword not folded into bit 31 before the 32-bit clip.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm256_min_epu32(_mm256_or_si256(a_raw, big), c.clip)', '_mm256_min_epu32(a_raw, c.clip); (void)big /* MUTANT */')
open(p,"w").write(s)
