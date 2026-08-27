; T-2276 strike -- MSVC/COFF/x86-64 leg, ml64.exe (MASM).
;
; Byte-for-byte the same construction shape as
; Claude/Vitruvius/t2265-fold12-probe/pool_msvc.asm, with exactly one coordinate
; varied: the pool's eight bytes are DECODABLE as x86-64 long-mode instructions
; instead of undecodable. 90 90 90 are three NOPs; 48 B8 begins a ten-byte
; `movabs rax, imm64` whose immediate consumes the pool's last three bytes and
; all five bytes of BodyDivide's body.

.code

HashSite PROC
    add ecx, edx
    mov eax, ecx
    ret
HashSite ENDP

ALIGN 8
Lpool BYTE 090h, 090h, 090h, 048h, 0B8h, 0AAh, 0BBh, 0CCh

BodyDivide PROC
    divsd xmm0, xmm1
    ret
BodyDivide ENDP

END
