p='src/matmul.cpp'; s=open(p).read(); old='constexpr size_t kTiledFlushPairs = 16384;'; new='constexpr size_t kTiledFlushPairs = 66053;  // MUTANT X2'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
