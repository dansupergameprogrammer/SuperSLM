# S1 "table indexed by code + 127" (norm table): every element reads its neighbour's entry.
p='src/forward/forward_sites.cpp'; s=open(p).read()
old='wide[i] = divided[static_cast<int>(h[i]) + 128]'; assert s.count(old)==1
s=s.replace(old,'wide[i] = divided[(static_cast<int>(h[i]) + 127) & 255]  /* MUTANT */'); open(p,'w').write(s)
