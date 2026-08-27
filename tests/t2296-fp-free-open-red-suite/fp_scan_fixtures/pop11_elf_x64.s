// T-2326 (Curie) -- Sec7 dimension 11's eleventh commissioning population: the
// byte-accounting law's own toolchain-and-format independence (fold round 10,
// D-SLM4407). One ISA (x86-64), object format (ELF), toolchain (clang) cell of
// four -- see pop11_coff_x64.s, pop11_msvc_x64.asm, pop11_arm64.s for the other
// three, matching this design's own four named production legs exactly
// (Sec4.1: MSVC/COFF/x86-64, clang/COFF/x86-64, clang/ELF/x86-64,
// clang/ELF/AArch64).
//
// Deliberately no inline literal pool here (contrast pop12/pop13, which vary
// decodability of an unaccounted range): this population's own claim is
// narrower -- an ordinary integer function and an ordinary floating-point
// function, on THIS (ISA, format, toolchain) cell, decode and classify exactly
// as they do on every other cell. Must-accept: HashSite (no FP instruction).
// Must-reject: BodyDivide (a genuine divsd).

    .text
    .globl HashSite
    .type HashSite,@function
HashSite:
    addl %esi, %edi
    movl %edi, %eax
    ret

    .globl BodyDivide
    .type BodyDivide,@function
BodyDivide:
    divsd %xmm1, %xmm0
    ret
