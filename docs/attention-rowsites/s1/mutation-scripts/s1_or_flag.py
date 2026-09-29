# S1 "landing flag OR-ed over the whole table": the candidate is refused if any code's entry is flagged.
p='src/forward/forward_sites.cpp'; s=open(p).read()
old='''				exceeded_table[c + 127] = exceeded;
			}
		}'''
assert s.count(old)==1
s=s.replace(old,'''				exceeded_table[c + 127] = exceeded;
			}
			for (int c = 0; c < 255; ++c)  // MUTANT
				if (exceeded_table[c]) {
					candidate.status = SslmForwardStatus::ResidualReconciliationMagnitudeOutOfDomain;
					return candidate;
				}
		}'''); open(p,'w').write(s)
