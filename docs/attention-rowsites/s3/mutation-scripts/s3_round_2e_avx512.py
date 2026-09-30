# S3 AVX-512 body only: rounding term 2^e instead of 2^(e-1).
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('uint64_t{1} << (e - 1)', 'uint64_t{1} << e /* MUTANT */', start="size_t RequantRowAvx512(", end="}  // namespace")
open(p,'w').write(s)
