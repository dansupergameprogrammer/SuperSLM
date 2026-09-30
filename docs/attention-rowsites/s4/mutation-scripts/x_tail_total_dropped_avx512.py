# Extra. AVX-512 body: the tail's e left out of the total.
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('\t\t\ttotal += buf[i];\n', '\t\t\t/* MUTANT */\n', 1, start='void SoftmaxRowAvx512(')
open(p,"w").write(s)
