# S2 AVX2 body only: odd last key dropped.
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
i = s.index('SUPERSLM_AVX2_TARGET inline void ProbVBlockAvx2'); j = s.index('if (k < width) {  // the unpaired last key', i); s = s[:j] + 'if (false && k < width) {  // MUTANT' + s[j+len('if (k < width) {  // the unpaired last key'):]
open(p,'w').write(s)
