// T-2380 (Curie) -- design Sec7 dim 11, forty-first population's own
// fold-42 replacement control (D-SLM5070): bnd jmp / bnd call, real
// two-token mnemonics capstone renders for a CET/legacy-MPX branch-tracking
// prefix ahead of a real indirect jmp/call. clang's integrated assembler
// does not accept the `bnd` mnemonic directly (MPX support was removed from
// recent LLVM/binutils) -- the bytes are emitted directly via .byte,
// confirmed by direct execution this session to be the real F2-prefixed
// encoding capstone 5.0.7 (this repo's own pinned version) decodes as
// "bnd jmp"/"bnd call".
    .text
    .globl BndJmpProbe
    .type BndJmpProbe,@function
BndJmpProbe:
    .byte 0xf2, 0xff, 0xe0    // bnd jmp rax
    ret

    .globl BndCallProbe
    .type BndCallProbe,@function
BndCallProbe:
    .byte 0xf2, 0xff, 0xd0    // bnd call rax
    ret
