# Extra, AVX2 body: the pair sums not added (hadd -> unpacklo, only one pair per key).
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm256_hadd_epi32(acc[l][0], acc[l][1])', '_mm256_unpacklo_epi32(acc[l][0], acc[l][1]) /* MUTANT */')
open(p, "w").write(s)
