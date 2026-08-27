// T-2276 strike -- clang/ELF/x86-64 (the real ubuntu-latest CI cell's own object format).
//
// Same construction shape as the folds-11/12 must-reject population
// (Claude/Vitruvius/t2265-fold12-probe/pool_elf.s): an ordinary integer function,
// an 8-byte inline literal pool covered by NO symbol, then a floating-point body,
// all in one code section.
//
// ONE coordinate is varied, and it is the coordinate every construction in that
// population holds fixed: the pool's bytes are DECODABLE as x86-64 instructions
// instead of undecodable. The pool's trailing bytes 0x48 0xB8 begin a 10-byte
// `movabs rax, imm64`, whose 8-byte immediate consumes the pool's own last three
// bytes AND all five bytes of BodyDivide's body.
    .text
    .globl HashSite
    .type HashSite,@function
HashSite:
    addl %esi, %edi         // 01 f7
    movl %edi, %eax         // 89 f8
    ret                     // c3
.Lpool:
    .byte 0x90,0x90,0x90,0x48,0xB8,0xAA,0xBB,0xCC
    .globl BodyDivide
    .type BodyDivide,@function
BodyDivide:
    divsd %xmm1, %xmm0      // f2 0f 5e c1
    ret                     // c3
