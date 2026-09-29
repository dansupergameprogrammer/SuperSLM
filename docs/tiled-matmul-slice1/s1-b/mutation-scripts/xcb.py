p='src/matmul.cpp'; s=open(p).read(); old='\tinternal::MaybeThrowInjectedTiledGemmAllocFault();  // test seam (cell 5.1); empty in production'; new='\tinternal::MaybeThrowInjectedTiledGemmAllocFault(); j_begin &= ~size_t{15};  // MUTANT XCb'
assert s.count(old)==1, (p, s.count(old)); s=s.replace(old,new); open(p,'w').write(s)
