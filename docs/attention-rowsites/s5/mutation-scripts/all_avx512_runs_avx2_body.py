# Dispatch: the AVX-512 kernel runs the AVX2 body.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('				QkQ31RowAvx512(keys, head_dim, width, limbs, out);', '				QkQ31RowAvx2(keys, head_dim, width, limbs, out); /* MUTANT */')
open(p, "w").write(s)
