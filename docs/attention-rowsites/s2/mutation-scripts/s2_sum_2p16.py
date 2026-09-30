# S2 guard: Sum p <= 2^15 -> <= 2^16 (output-equivalent, section 5.2).
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('constexpr int64_t kProbVMaxSum = INT64_C(1) << 15;', 'constexpr int64_t kProbVMaxSum = INT64_C(1) << 16;  // MUTANT')
open(p,'w').write(s)
