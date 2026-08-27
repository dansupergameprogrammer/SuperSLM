// T-2342 (Curie) -- ci_gate_corpus's own all-ACCEPT leg: HashSite only, no
// FP-carrying symbol anywhere in this object. Paired with pop11_elf_x64.s
// (this directory), which carries HashSite (ACCEPT) AND BodyDivide (REJECT)
// -- one clean object, one dirty object, the minimum pair
// ci_gate_corpus needs to discriminate "every object all-ACCEPT" from "a
// REJECT anywhere in the corpus."

    .text
    .globl HashSite
    .type HashSite,@function
HashSite:
    addl %esi, %edi
    movl %edi, %eax
    ret
