# all: "fast increment moved before the guard" (row tables: every call counts as taken).
p='src/forward/forward_sites.cpp'; s=open(p).read(); old='inline void CountRowTableDecision(RowTableSite site, bool taken) {'; assert s.count(old)==1
s=s.replace(old,old+'\n\ttaken = true;  // MUTANT'); open(p,'w').write(s)
