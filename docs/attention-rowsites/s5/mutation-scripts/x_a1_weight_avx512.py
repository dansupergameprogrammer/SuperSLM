# Extra, AVX-512 body: a1 recombined at 2^16.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm512_slli_epi64(w[1], 15)', '_mm512_slli_epi64(w[1], 16) /* MUTANT */')
open(p, "w").write(s)
