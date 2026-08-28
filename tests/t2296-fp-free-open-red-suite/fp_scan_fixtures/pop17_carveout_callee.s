// T-2347 (Curie) -- population seventeen's must-accept half, the CALLEE:
// SiblingCallee, a genuine, first-party, all-integer function -- real enough
// that granting the caller's edge does not excuse SiblingCallee's own bytes
// from being independently checked (identical discipline to
// pop_corpus_callee.s, T-2342's population for gap (b)).

    .text
    .globl SiblingCallee
    .type SiblingCallee,@function
SiblingCallee:
    addl %esi, %edi
    movl %edi, %eax
    ret
