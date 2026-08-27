// T-2326 (Curie) -- population eleven, clang/COFF/x86-64 leg. Identical
// construction to pop11_elf_x64.s; see that file's own header. COFF function
// symbols declared via .def/.scl/.type 32/.endef (clang's integrated
// assembler, *-windows-msvc target), matching pop13's own COFF fixture
// convention (Claude/Loki/t2276-probe/pool_decodable_coff.s).

    .text
    .def HashSite; .scl 2; .type 32; .endef
    .globl HashSite
HashSite:
    addl %esi, %edi
    movl %edi, %eax
    ret

    .def BodyDivide; .scl 2; .type 32; .endef
    .globl BodyDivide
BodyDivide:
    divsd %xmm1, %xmm0
    ret
