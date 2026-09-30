# S2 guard: Sum p <= 2^15 -> < 2^15.
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('return sum <= kProbVMaxSum;', 'return sum < kProbVMaxSum;  // MUTANT')
open(p,'w').write(s)
