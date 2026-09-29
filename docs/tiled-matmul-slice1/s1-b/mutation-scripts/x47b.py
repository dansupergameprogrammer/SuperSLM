p='src/matmul.cpp'; s=open(p).read(); old='\tif (DispatchGemmPath(tier, num_tokens) == GemmPath::kTiled) {'; new='\tif (DispatchGemmPath(tier, num_tokens) == GemmPath::kTiled || true) {  // MUTANT X4.7b'
assert s.count(old)==2, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
