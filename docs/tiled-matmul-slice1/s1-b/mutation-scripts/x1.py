p='src/matmul.cpp'; s=open(p).read(); old="\t\tfor (int m = 0; m < MR; ++m) {  // flush the window's int32 lanes into int64"; new='\t\tif (q_end == pairs) for (int m = 0; m < MR; ++m) {  // MUTANT X1: in-loop flush deleted'
assert s.count(old)==2, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
