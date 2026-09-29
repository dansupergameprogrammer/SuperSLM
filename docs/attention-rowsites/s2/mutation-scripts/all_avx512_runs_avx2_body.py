# all: the AVX-512 dispatch runs the AVX2 body.
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('\t\t\t\tProbVAccumulateIntoAvx512(probs, values, width, head_dim, out_ctx);\n', '\t\t\t\tProbVAccumulateIntoAvx2(probs, values, width, head_dim, out_ctx);  // MUTANT\n')
open(p,'w').write(s)
