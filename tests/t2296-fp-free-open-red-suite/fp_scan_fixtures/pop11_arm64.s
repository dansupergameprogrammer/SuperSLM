// T-2326 (Curie) -- population eleven, clang/ELF/AArch64 leg (the leg fold
// round 11's own text names as superseding population ten's own infeasible
// MSVC/dumpbin-rendered construction -- see this campaign's case file for the
// full disposition). Identical shape to the x86-64 legs: an ordinary integer
// function, then an ordinary floating-point function.

    .text
    .globl HashSite
    .type HashSite, %function
HashSite:
    add     w0, w0, w1
    ret

    .globl BodyDivide
    .type BodyDivide, %function
BodyDivide:
    fdiv    d0, d0, d1
    ret
