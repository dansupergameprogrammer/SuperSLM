p='src/matmul.cpp'; s=open(p).read(); old='rows[i][kk] : 0;'; new='rows[i][kk] : 1;  // MUTANT X4e (weight pad)'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
p='src/matmul.cpp'; s=open(p).read(); old='		std::vector<int16_t> a16(num_tokens * kp);'
assert s.count(old)==1, s.count(old); s=s.replace(old,'		std::vector<int16_t> a16(num_tokens * kp, int16_t{1});  // MUTANT X4e (activation pad)'); open(p,'w').write(s)
