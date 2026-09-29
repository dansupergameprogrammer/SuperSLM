# S1 threshold SIZE_MAX (tables never taken).
p='src/forward/forward_sites.cpp'; s=open(p).read(); old='constexpr size_t kRowTableMinWidth = 512;'; assert s.count(old)==1
s=s.replace(old,'constexpr size_t kRowTableMinWidth = SIZE_MAX;  // MUTANT'); open(p,'w').write(s)
