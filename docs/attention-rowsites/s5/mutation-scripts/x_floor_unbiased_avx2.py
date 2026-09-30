# Extra, AVX2 body: floor by a logical shift without the bias (wrong for negative x).
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('_mm256_set1_epi64x(INT64_C(1) << 62)', '_mm256_setzero_si256() /* MUTANT */', 1, start='inline __m256i Q31RoundAvx2(', end='void QkQ31RowAvx2(')
sub('_mm256_set1_epi64x(INT64_C(1) << 31))', '_mm256_setzero_si256())', 1, start='inline __m256i Q31RoundAvx2(', end='void QkQ31RowAvx2(')
open(p, "w").write(s)
