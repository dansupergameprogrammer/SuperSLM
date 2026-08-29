// T-2380 (Curie) -- the genuinely arithmetic float mnemonic family every
// forty-first/forty-second/forty-third population's own control asserts is
// UNAFFECTED by that population's own widening (addps/mulps/divps/sqrtps/
// cvtsi2sd/comiss and their forms, design Sec4.1).
    .text
    .globl ArithFamilyControlReject
    .type ArithFamilyControlReject,@function
ArithFamilyControlReject:
    addps %xmm1, %xmm0
    mulps %xmm1, %xmm0
    divps %xmm1, %xmm0
    sqrtps %xmm1, %xmm0
    cvtsi2sd %eax, %xmm0
    comiss %xmm1, %xmm0
    ret
