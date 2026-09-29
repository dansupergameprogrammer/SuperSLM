p='src/matmul.cpp'; s=open(p).read(); old='\treturn SelectGemmPath(tier, num_tokens, kTiledAvx512MsvcSwitch, kIsMsvcBuild);'; new='\treturn SelectGemmPath(tier, num_tokens, kTiledAvx512MsvcSwitch, false);  // MUTANT X4.7a'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
