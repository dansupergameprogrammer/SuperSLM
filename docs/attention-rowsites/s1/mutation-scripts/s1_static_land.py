# S1 "table built once per call site and reused across rows" (landing; one table for both candidates too).
p='src/forward/forward_sites.cpp'; s=open(p).read()
old='''		int64_t landed_table[255];
		bool exceeded_table[255];
		if (use_table) {'''
assert s.count(old)==1
s=s.replace(old,'''		static int64_t landed_table[255];  // MUTANT
		static bool exceeded_table[255];
		static bool built = false;
		if (use_table && !built) { built = true;'''); open(p,'w').write(s)
