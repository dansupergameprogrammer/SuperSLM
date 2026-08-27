// T-2265 fold round 12 -- G12's must-reject construction for clang/COFF/x86-64.
//
// Same construction as pool_elf.s (see that file's header for the pool bytes'
// derivation and the invalidity check performed before authoring), assembled for
// the Windows/COFF object format via clang's integrated assembler instead of
// clang's Linux/ELF target -- GAS syntax, no .type/@function directive (COFF's
// own function-symbol encoding is a different mechanism this construction does
// not need, since a REFUSE leg never reaches per-function attribution).

    .text
    .globl HashSite
HashSite:
    addl %esi, %edi
    movl %edi, %eax
    ret

    .p2align 3
.Lpool:
    .byte 0x06,0x07,0x0E,0x16,0x17,0x1E,0x1F,0xD6

    .globl BodyDivide
BodyDivide:
    divsd %xmm1, %xmm0
    mulsd %xmm1, %xmm0
    subsd %xmm1, %xmm0
    ret
