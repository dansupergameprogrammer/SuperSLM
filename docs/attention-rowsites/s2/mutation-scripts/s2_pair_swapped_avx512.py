# S2 AVX-512 body only: pair order swapped (p_{k+1} with v_k).
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
i = s.index('SUPERSLM_AVX512_TARGET inline void ProbVBlockAvx512'); j = s.index('ProbVPair(probs[k], probs[k + 1])', i); s = s[:j] + 'ProbVPair(probs[k + 1], probs[k]) /* MUTANT */' + s[j+len('ProbVPair(probs[k], probs[k + 1])'):]
open(p,'w').write(s)
