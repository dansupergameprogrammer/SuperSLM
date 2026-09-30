# Guard: width <= 2^14 -> < 2^14.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('if (width > kSoftmaxFastMaxWidth) return false;', 'if (width >= kSoftmaxFastMaxWidth) return false; /* MUTANT */', start='bool SoftmaxFastGuard(', end='struct SoftmaxFastRow')
open(p,"w").write(s)
