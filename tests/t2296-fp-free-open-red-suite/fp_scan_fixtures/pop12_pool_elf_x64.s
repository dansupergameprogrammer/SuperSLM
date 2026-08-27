// T-2265 fold round 12 -- G12's must-reject construction for clang/ELF/x86-64.
//
// Mirrors T-2275's own method (Sec7 dim 11 twelfth population, pool2.s): an
// ordinary integer function, an inline literal pool undecodable as code in this
// ISA/mode, then a floating-point body -- the exact shape clause (0) exists to
// REFUSE on, built here for the leg the twelfth population did not cover.
//
// The pool bytes are eight one-byte opcodes that are INVALID IN X86-64 LONG MODE
// specifically -- 0x06/0x07/0x0E/0x16/0x17/0x1E/0x1F (segment-register push/pop,
// repurposed as REX prefixes or removed in 64-bit mode) and 0xD6 (SALC, removed in
// long mode). Confirmed by direct execution against capstone's own x86-64 decoder
// before this file was written: capstone.Cs(CS_ARCH_X86, CS_MODE_64).disasm()
// returns an empty instruction list for this byte string at any offset -- genuinely
// undecodable in this mode, not merely unusual. No symbol in this object covers
// the pool bytes, so byte_accounting_scan.py's data-in-code classification (a
// DATA-typed symbol's own extent) cannot account for them either -- the two
// structural escapes (recognised NOP padding, symbol-grounded data) both fail,
// and the accounting law's own residue-tracking is what must catch it.

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
