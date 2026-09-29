p='src/matmul.cpp'; s=open(p).read(); old='\t    .fetch_add(1, std::memory_order_relaxed);'; new='\t    .fetch_add((j_end - j_begin + 15) / 16, std::memory_order_relaxed);  // MUTANT X11.1a'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
