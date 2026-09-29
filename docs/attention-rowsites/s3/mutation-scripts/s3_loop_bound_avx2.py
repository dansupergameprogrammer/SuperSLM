# S3 AVX2 body only: vector loop bound n instead of n - lanes + 1 (writes past the row).
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('for (; i + 4 <= n; i += 4)', 'for (; i < n; i += 4) /* MUTANT */', start="size_t RequantRowAvx2(", end="SUPERSLM_INTMATH_AVX512_TARGET")
open(p,'w').write(s)
