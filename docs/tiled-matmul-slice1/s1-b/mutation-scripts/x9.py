p='src/matmul.cpp'; s=open(p).read(); old='\t\t\t\tfor (int i = 0; i < 8; ++i) acc64[m * 16 + h * 8 + i] += static_cast<int64_t>(lanes[i]);'; new='\t\t\t\tfor (int i = 0; i < 8; ++i) acc64[m * 16 + h * 8 + i] += static_cast<int64_t>(lanes[q_end == pairs ? i : (i ^ 1)]);  // MUTANT X9'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
s=open(p).read(); old='				for (int i = 0; i < 16; ++i) acc64[m * 32 + p * 16 + i] += static_cast<int64_t>(lanes[i]);'
assert s.count(old)==1; s=s.replace(old,'				for (int i = 0; i < 16; ++i) acc64[m * 32 + p * 16 + i] += static_cast<int64_t>(lanes[q_end == pairs ? i : (i ^ 1)]);'); open('src/matmul.cpp','w').write(s)
