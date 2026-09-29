p='src/matmul.cpp'; s=open(p).read(); old='(avx512 ? superslm_test::g_tiled_entry_invocations_avx512 : superslm_test::g_tiled_entry_invocations_avx2)'; new='(avx512 ? superslm_test::g_tiled_entry_invocations_avx2 : superslm_test::g_tiled_entry_invocations_avx512)  // MUTANT X11.1b'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
