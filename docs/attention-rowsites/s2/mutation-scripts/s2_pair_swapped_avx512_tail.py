# S2 AVX-512 16-dimension tail unit only: pair order swapped (p_{k+1} with v_k).
p='src/matmul.cpp'; s=open(p).read()
i = s.index('SUPERSLM_AVX512_TARGET inline void ProbVTail16Avx512')
j = s.index('ProbVPair(probs[k], probs[k + 1])', i)
s = s[:j] + 'ProbVPair(probs[k + 1], probs[k]) /* MUTANT */' + s[j+len('ProbVPair(probs[k], probs[k + 1])'):]
open(p,'w').write(s)
