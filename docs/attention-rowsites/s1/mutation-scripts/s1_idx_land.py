# S1 "table indexed by code + 127" (landing table, whose own offset is 127): value and flag read at code + 126.
p='src/forward/forward_sites.cpp'; s=open(p).read()
for old in ['landed = landed_table[static_cast<int>(other_code[i]) + 127];','magnitude_exceeded = exceeded_table[static_cast<int>(other_code[i]) + 127];']:
    assert s.count(old)==1
    s=s.replace(old, old.replace('[static_cast<int>(other_code[i]) + 127]','[(static_cast<int>(other_code[i]) + 126 + 255) % 255]')+'  // MUTANT')
open(p,'w').write(s)
