# Dispatcher: always fall back.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('if (SoftmaxFastGuard(scores', 'if (false /* MUTANT */ && SoftmaxFastGuard(scores', 1, start='bool SoftmaxRowQ15(', end='// C32 (§5.2')
open(p,"w").write(s)
