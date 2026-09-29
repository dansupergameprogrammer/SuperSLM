p='src/matmul.cpp'; s=open(p).read(); old='dst + (k / 2 + q) * 32 + h * 16), r[q]);'; new='dst + (k / 2 + q) * 32 + (1 - h) * 16), r[q]);  // MUTANT X3b'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
