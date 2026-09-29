# S2 guard: head_dim % 16 -> % 32.
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('return head_dim % 16 == 0 && ProbVInt16Condition', 'return head_dim % 32 == 0 && ProbVInt16Condition  /* MUTANT */')
open(p,'w').write(s)
