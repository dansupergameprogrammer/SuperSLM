# Guard: M <= 2^47 dropped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub(' || !SGe(SFromI64(kSoftmaxRowMaxSafeExponent), m128)', ' /* MUTANT */', 1, start='bool SoftmaxFastGuard(', end='struct SoftmaxFastRow')
open(p,"w").write(s)
