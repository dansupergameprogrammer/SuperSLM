; T-2326 (Curie) -- population eleven, MSVC/COFF/x86-64 leg -- "the toolchain
; its whole-corpus clean scan was measured on" (design Sec4.1). ml64.exe
; (MASM), matching pop12/pop13's own MSVC fixture convention.

.code

HashSite PROC
    add ecx, edx
    mov eax, ecx
    ret
HashSite ENDP

BodyDivide PROC
    divsd xmm0, xmm1
    ret
BodyDivide ENDP

END
