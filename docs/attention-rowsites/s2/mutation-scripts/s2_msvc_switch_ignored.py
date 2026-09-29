# 11.2 (extra): the selector ignores the MSVC switch (AVX-512 kernels on in every MSVC build).
import re
p='src/matmul.cpp'; s=open(p).read()
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
sub('return (is_msvc_build && msvc_avx512_switch == 0) ? SitesKernel::kShipped : SitesKernel::kAvx512;', 'return (void)is_msvc_build, (void)msvc_avx512_switch, SitesKernel::kAvx512;  // MUTANT')
open(p,'w').write(s)
