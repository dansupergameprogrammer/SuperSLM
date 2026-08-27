// T-2276 strike -- clang/COFF/x86-64, the second of the design's four named
// production legs. Identical construction to pool_decodable_elf.s; COFF function
// symbols are declared via .def/.type 32 so byte_accounting_scan.read_coff's own
// DTYPE_FUNCTION test sees them.
    .text
    .def HashSite; .scl 2; .type 32; .endef
    .globl HashSite
HashSite:
    addl %esi, %edi
    movl %edi, %eax
    ret
.Lpool:
    .byte 0x90,0x90,0x90,0x48,0xB8,0xAA,0xBB,0xCC
    .def BodyDivide; .scl 2; .type 32; .endef
    .globl BodyDivide
BodyDivide:
    divsd %xmm1, %xmm0
    ret
