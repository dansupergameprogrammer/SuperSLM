p='src/matmul.cpp'; s=open(p).read(); old='\t\t\treturn (is_msvc_build && msvc_avx512_switch == 0) ? GemmPath::kDotRowLoop : GemmPath::kTiled;'; new='\t\t\treturn GemmPath::kTiled;  // MUTANT XS2: the whole MSVC term deleted'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
