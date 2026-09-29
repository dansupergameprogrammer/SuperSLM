# all: "fallback increment deleted" (row tables: the skipped counters never move).
p='src/forward/forward_sites.cpp'; s=open(p).read(); old='inline void CountRowTableDecision(RowTableSite site, bool taken) {'; assert s.count(old)==1
s=s.replace(old,old+'\n\tif (!taken) return;  // MUTANT'); open(p,'w').write(s)
