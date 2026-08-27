// T-2342 (Curie) -- corpus_symbols population: the CALLER half. Calls
// CorpusCallee, defined in pop_corpus_callee.s (a SEPARATE compiled object) --
// an unresolved external reference from THIS object's own point of view,
// exactly the shape design Sec4.1's gap (b) names: "a relocation whose target
// is undefined in that object's own symbol table... is indistinguishable,
// from inside one call [to scan_object], between a genuine third-party
// external and a first-party symbol defined in a sibling translation unit of
// the same corpus." Neither object alone resolves the call; corpus_symbols is
// what is supposed to tell scan_object the target is first-party.

    .text
    .globl CallsCorpusCallee
    .type CallsCorpusCallee,@function
CallsCorpusCallee:
    subq $8, %rsp
    call CorpusCallee
    addq $8, %rsp
    ret
