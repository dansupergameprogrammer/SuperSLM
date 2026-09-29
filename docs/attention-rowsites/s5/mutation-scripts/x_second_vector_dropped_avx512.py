# Extra, AVX-512 body: the second vector of each block (keys 8..15) not accumulated.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('acc[l][1] = _mm512_add_epi32(acc[l][1], _mm512_madd_epi16(k1, b));', '(void)k1; /* MUTANT */')
open(p, "w").write(s)
