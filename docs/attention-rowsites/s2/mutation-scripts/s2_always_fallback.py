# S2: always fall back (the fast path never admits a row).
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('inline bool ProbVFastPathAdmits(const int64_t* probs, size_t width, size_t head_dim) {\n', 'inline bool ProbVFastPathAdmits(const int64_t* probs, size_t width, size_t head_dim) {\n\treturn false;  // MUTANT\n')
open(p,'w').write(s)
