# S2 guard: p <= 32,767 -> <= 32,768.
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('constexpr int64_t kProbVMaxP = 32767;', 'constexpr int64_t kProbVMaxP = 32768;  // MUTANT')
open(p,'w').write(s)
