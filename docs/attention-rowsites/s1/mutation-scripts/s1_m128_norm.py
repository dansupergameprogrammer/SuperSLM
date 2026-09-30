# S1 norm table built for [-127, 127] only (as the SiLU and landing tables are), -128 entry left 0.
p='src/forward/forward_sites.cpp'; s=open(p).read()
old='''		int64_t divided[256];
		for (int c = -128; c <= 127; ++c)'''
assert s.count(old)==1
s=s.replace(old,'''		int64_t divided[256] = {};  // MUTANT
		for (int c = -127; c <= 127; ++c)'''); open(p,'w').write(s)
