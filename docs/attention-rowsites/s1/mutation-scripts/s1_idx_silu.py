# S1 "table indexed by code + 127" (SiLU table, whose own offset is 127): read at code + 126.
p='src/forward/forward_sites.cpp'; s=open(p).read()
old=': sigmoid[static_cast<int>(code) + 127];'; assert s.count(old)==1
s=s.replace(old,': sigmoid[(static_cast<int>(code) + 126 + 255) % 255];  // MUTANT'); open(p,'w').write(s)
