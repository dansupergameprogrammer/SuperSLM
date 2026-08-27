// T-2342 (Curie) -- corpus_symbols population: the CALLEE half. Defines
// CorpusCallee, an ordinary integer function with NO floating-point
// instruction -- a genuine, first-party, all-integer corpus member, real
// enough that granting the cross-object edge does not "excuse the target
// from its own bytes being checked" (design Sec4.1 gap (b)'s own text):
// this object's own independent scan_object call must ACCEPT it.

    .text
    .globl CorpusCallee
    .type CorpusCallee,@function
CorpusCallee:
    addl %esi, %edi
    movl %edi, %eax
    ret
