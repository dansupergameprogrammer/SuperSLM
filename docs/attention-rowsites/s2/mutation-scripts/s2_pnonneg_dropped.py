# S2 guard: p >= 0 dropped.
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('if (p < 0 || p > kProbVMaxP) return false;', 'if (p > kProbVMaxP) return false;  // MUTANT')
open(p,'w').write(s)
