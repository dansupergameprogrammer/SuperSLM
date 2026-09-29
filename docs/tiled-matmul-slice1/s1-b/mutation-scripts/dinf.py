p='src/matmul.cpp'; s=open(p).read(); old='constexpr size_t kTiledMinTokens = 8;'; new='constexpr size_t kTiledMinTokens = SIZE_MAX;  // MUTANT D-inf'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
