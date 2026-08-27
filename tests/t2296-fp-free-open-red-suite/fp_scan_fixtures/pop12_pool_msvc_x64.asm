; T-2265 fold round 12 -- G12's must-reject construction for MSVC/COFF/x86-64.
;
; Same construction as pool_elf.s/pool_coff.s (see pool_elf.s's header for the
; pool bytes' derivation and the invalidity check performed before authoring),
; assembled for MSVC's own COFF object format via ml64.exe (MASM) rather than
; clang's integrated assembler -- MASM syntax, since ml64.exe does not accept
; GAS syntax. Same bytes, same shape: an integer function, an inline literal
; pool undecodable as x86-64 code, a floating-point body.

.code

HashSite PROC
    add ecx, edx
    mov eax, ecx
    ret
HashSite ENDP

ALIGN 8
Lpool BYTE 06h, 07h, 0Eh, 16h, 17h, 1Eh, 1Fh, 0D6h

BodyDivide PROC
    divsd xmm0, xmm1
    mulsd xmm0, xmm1
    subsd xmm0, xmm1
    ret
BodyDivide ENDP

END
