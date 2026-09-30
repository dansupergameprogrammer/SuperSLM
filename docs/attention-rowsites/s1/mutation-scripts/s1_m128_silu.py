# S1 "-128 read from the table (entry left unset)": SiLU table of 256 entries, entry for -128 never built (0).
p='src/forward/forward_sites.cpp'; s=open(p).read()
old='''		int32_t sigmoid[255];
		for (int c = -127; c <= 127; ++c)
			sigmoid[c + 127] = SiluSigmoidQ15(sigmoid_lut_table, static_cast<int8_t>(c), gate_scale.m, gate_e);'''
assert s.count(old)==1
s=s.replace(old,'''		int32_t sigmoid[256] = {};  // MUTANT
		for (int c = -127; c <= 127; ++c)
			sigmoid[c + 128] = SiluSigmoidQ15(sigmoid_lut_table, static_cast<int8_t>(c), gate_scale.m, gate_e);''')
old='''			const int32_t sig = code == INT8_MIN ? SiluSigmoidQ15(sigmoid_lut_table, code, gate_scale.m, gate_e)
			                                     : sigmoid[static_cast<int>(code) + 127];'''
assert s.count(old)==1
s=s.replace(old,'''			const int32_t sig = sigmoid[static_cast<int>(code) + 128];  // MUTANT''')
open(p,'w').write(s)
