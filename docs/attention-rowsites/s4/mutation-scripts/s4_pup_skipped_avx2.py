# AVX2 body: p upward correction skipped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('return _mm256_add_epi64(p, _mm256_add_epi64(no_up, c.one));', '(void)no_up; return p; /* MUTANT */')
open(p,"w").write(s)
