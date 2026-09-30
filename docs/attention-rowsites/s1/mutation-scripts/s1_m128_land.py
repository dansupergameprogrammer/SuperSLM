# S1 "-128 read from the table (entry left unset)": landing tables of 256 entries, entry for -128 never built (value 0, flag false).
p='src/forward/forward_sites.cpp'; s=open(p).read()
rep=[('''		int64_t landed_table[255];
		bool exceeded_table[255];''','''		int64_t landed_table[256] = {};  // MUTANT
		bool exceeded_table[256] = {};'''),
('''				landed_table[c + 127] = LandingRescale(''','''				landed_table[c + 128] = LandingRescale('''),
('''				exceeded_table[c + 127] = exceeded;''','''				exceeded_table[c + 128] = exceeded;'''),
('''			if (use_table && other_code[i] != INT8_MIN) {''','''			if (use_table) {  // MUTANT'''),
('''landed = landed_table[static_cast<int>(other_code[i]) + 127];''','''landed = landed_table[static_cast<int>(other_code[i]) + 128];'''),
('''magnitude_exceeded = exceeded_table[static_cast<int>(other_code[i]) + 127];''','''magnitude_exceeded = exceeded_table[static_cast<int>(other_code[i]) + 128];''')]
for a,b in rep:
    assert s.count(a)==1,a; s=s.replace(a,b)
open(p,'w').write(s)
