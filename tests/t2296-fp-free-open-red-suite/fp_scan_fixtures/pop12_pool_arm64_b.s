// T-2275 -- the same AArch64 code section, with the function order a compiler
// chooses freely: an inline literal pool, then a floating-point body after it.
//
// This is fold round 9's own disclosed residual shape (design Sec4.1, D-SLM4386 --
// "std::_Uhash_compare's own FNV-1a hash-constant literal, 0xcbf29ce484222325,
// stored inline as two 32-bit data words") sitting where MSVC's AArch64 code
// generator actually put it: inside .text, between code.
//
// BodyDivide's three instructions are byte-identical to the fold-10 fold record's
// own Sec3 row-4 leg ("BodyDivide (3 FP-arithmetic instructions: fdiv/fmadd/fsub)").

    .text
    .globl  HashSite
    .type   HashSite, %function
HashSite:
    add     w0, w0, w1
    ret
    .p2align 3
.Lfnv_pool:
    .quad   0xcbf29ce484222325      // fold round 9's own FNV-1a offset basis

    .globl  BodyDivide
    .type   BodyDivide, %function
BodyDivide:
    fdiv    d2, d0, d1
    fmadd   d1, d0, d1, d2
    fsub    d0, d1, d0
    ret
