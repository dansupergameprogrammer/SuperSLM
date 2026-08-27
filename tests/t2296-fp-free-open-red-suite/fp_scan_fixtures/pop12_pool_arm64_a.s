// T-2275 -- an AArch64 code section carrying a function plus an inline literal
// pool: the EXACT shape fold round 9 disclosed as its own residual (design Sec4.1,
// D-SLM4386 -- "std::_Uhash_compare's own FNV-1a hash-constant literal,
// 0xcbf29ce484222325, stored inline as two 32-bit data words that dumpbin does not
// disassemble as an instruction"), and the shape fold round 10 claims to have
// generalised structurally (D-SLM4404).
//
// BodyDivide's three instructions are byte-identical to the fold-10 fold record's
// own AArch64 leg (Sec3 row 4: "BodyDivide (3 FP-arithmetic instructions:
// fdiv/fmadd/fsub)").  The literal pool follows it in the same .text section, with
// no symbol covering it -- exactly as fold round 9 found it.

    .text
    .globl  BodyDivide
    .type   BodyDivide, %function
BodyDivide:
    fdiv    d2, d0, d1
    fmadd   d1, d0, d1, d2
    fsub    d0, d1, d0
    ret
    .p2align 3
.Lfnv_pool:
    .quad   0xcbf29ce484222325
