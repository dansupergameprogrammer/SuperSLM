; T-2347 (Curie) -- population seventeen, must-reject half: the ORIGINAL S2
; construction (design Sec4.1 fold round 35, D-SLM4886's own carve-out
; boundary) -- a genuine function-pointer call, `jmp QWORD PTR [gp]`, where
; `gp` is a real IN-OBJECT data symbol whose runtime CONTENTS (not its own
; address) determine which function actually runs. This edge's relocation
; resolves to `gp` itself -- the data HOLDER, never the callable entity --
; so it is exactly the shape the carve-out's own text says stays unvettable:
; "does the relocation's own target is not itself the callable entity... a
; data symbol whose runtime contents, not its address, determine which
; function actually runs." Must REJECT before and after the fold-35 fix
; lands -- this is the fix's own boundary, not a regression it introduces.

.data
gp QWORD 0

.code

Helper PROC
    add ecx, edx
    mov eax, ecx
    ret
Helper ENDP

Caller PROC
    jmp QWORD PTR [gp]
Caller ENDP

END
