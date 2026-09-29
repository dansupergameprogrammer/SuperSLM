p='src/matmul.cpp'; s=open(p).read(); old='\t\t\t\t\tif (n < j_end) out_acc'; new='\t\t\t\t\tif (true) out_acc'
assert s.count(old)==2, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
