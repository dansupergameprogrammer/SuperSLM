# S2 AVX-512 32-dimension unit only (extra): the in-lane halves stored at each other's dimensions (o + 16 and o swapped).
p='src/matmul.cpp'; s=open(p).read()
old = '''			_mm512_storeu_si512(o, _mm512_add_epi64(_mm512_loadu_si512(o), lo));
			_mm512_storeu_si512(o + 16, _mm512_add_epi64(_mm512_loadu_si512(o + 16), hi));'''
new = '''			_mm512_storeu_si512(o + 16, _mm512_add_epi64(_mm512_loadu_si512(o + 16), lo));  // MUTANT
			_mm512_storeu_si512(o, _mm512_add_epi64(_mm512_loadu_si512(o), hi));'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p,'w').write(s)
