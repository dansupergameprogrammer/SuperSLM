# S1 "table built once per call site and reused across rows" (norm).
p='src/forward/forward_sites.cpp'; s=open(p).read()
old='''		int64_t divided[256];
		for (int c = -128; c <= 127; ++c)
			divided[c + 128] = FloorDivI64(static_cast<int64_t>(c) << (2 * kNormFracBits), root);'''
assert s.count(old)==1
s=s.replace(old,'''		static int64_t divided[256];  // MUTANT
		static bool built = false;
		if (!built) { built = true;
		for (int c = -128; c <= 127; ++c)
			divided[c + 128] = FloorDivI64(static_cast<int64_t>(c) << (2 * kNormFracBits), root); }'''); open(p,'w').write(s)
