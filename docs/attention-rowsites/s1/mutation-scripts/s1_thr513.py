# S1 threshold n >= 513.
p='src/forward/forward_sites.cpp'; s=open(p).read(); old='constexpr size_t kRowTableMinWidth = 512;'; assert s.count(old)==1
s=s.replace(old,'constexpr size_t kRowTableMinWidth = 513;  // MUTANT'); open(p,'w').write(s)
