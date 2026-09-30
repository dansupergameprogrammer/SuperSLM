# Dispatch: the AVX-512 kernel runs the AVX2 body.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('i = RequantRowAvx512(x, n, r, s, out);', 'i = RequantRowAvx2(x, n, r, s, out); /* MUTANT */')
open(p,'w').write(s)
