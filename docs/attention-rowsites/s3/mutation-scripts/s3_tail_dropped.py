# (extra) S3 dispatcher: the scalar tail after the lanes is skipped.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('for (; i < n; ++i) out[i] = RequantTokenCodeWide(x[i], r, s);  // the tail, or the whole row', 'if (i == 0) for (; i < n; ++i) out[i] = RequantTokenCodeWide(x[i], r, s); /* MUTANT */')
open(p,'w').write(s)
