# S2 AVX-512 16-dimension tail unit only: odd last key dropped.
p='src/matmul.cpp'; s=open(p).read()
i = s.index('SUPERSLM_AVX512_TARGET inline void ProbVTail16Avx512')
old = 'if (k < width) {  // the unpaired last key'
j = s.index(old, i)
s = s[:j] + 'if (false && k < width) {  // MUTANT' + s[j+len(old):]
open(p,'w').write(s)
