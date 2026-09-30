# S1 "table built once per call site and reused across rows" (SiLU).
p='src/forward/forward_sites.cpp'; s=open(p).read()
old='''		int32_t sigmoid[255];
		for (int c = -127; c <= 127; ++c)
			sigmoid[c + 127] = SiluSigmoidQ15(sigmoid_lut_table, static_cast<int8_t>(c), gate_scale.m, gate_e);'''
assert s.count(old)==1
s=s.replace(old,'''		static int32_t sigmoid[255];  // MUTANT
		static bool built = false;
		if (!built) { built = true;
		for (int c = -127; c <= 127; ++c)
			sigmoid[c + 127] = SiluSigmoidQ15(sigmoid_lut_table, static_cast<int8_t>(c), gate_scale.m, gate_e); }'''); open(p,'w').write(s)
