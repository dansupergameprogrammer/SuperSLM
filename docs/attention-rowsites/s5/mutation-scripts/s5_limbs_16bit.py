# S5: limbs of 16 bits (a0 < 2^16): a0 = w & 0xFFFF, a1 = (w >> 16) & 0xFFFF, a2 = w >> 32, recombined at 2^32 and 2^16 in both bodies.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('const int64_t a[3] = {w & 0x7FFF, (w >> 15) & 0x7FFF, w >> 30};', 'const int64_t a[3] = {w & 0xFFFF, (w >> 16) & 0xFFFF, w >> 32}; /* MUTANT */')
sub('_mm256_slli_epi64(w[2], 30), _mm256_slli_epi64(w[1], 15)', '_mm256_slli_epi64(w[2], 32), _mm256_slli_epi64(w[1], 16) /* MUTANT */')
sub('_mm512_slli_epi64(w[2], 30), _mm512_slli_epi64(w[1], 15)', '_mm512_slli_epi64(w[2], 32), _mm512_slli_epi64(w[1], 16) /* MUTANT */')
open(p, "w").write(s)
