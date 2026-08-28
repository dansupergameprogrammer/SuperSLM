// T-2347 (Curie) -- population sixteen: the self-zeroing vxorps/vxorpd
// boundary (design Sec4.1 fold round 35, D-SLM4889's own first disposed
// item; the remedy for the fold-34 review's own S1 landed at 8a28460 but
// carries zero cells anywhere in this suite -- Poirot's S2 grep-confirmed
// zero occurrences of "vxorps"/"xorps" in the suite text). Two real,
// VEX-encoded instructions, same mnemonic, only the operand list differs:
// SelfZeroAccept's every operand is the identical register (the
// constant-zero-materialization idiom); GenuineXorReject's is not (a real
// bitwise XOR between two distinct register values) -- genuine floating-point
// arithmetic under this design's own printed classification (a sign-flip/
// XOR-based negation shares this exact mnemonic and encoding shape).

.intel_syntax noprefix
.text
.globl SelfZeroAccept
.type SelfZeroAccept,@function
SelfZeroAccept:
    vxorps xmm0, xmm0, xmm0
    ret

.globl GenuineXorReject
.type GenuineXorReject,@function
GenuineXorReject:
    vxorps xmm0, xmm1, xmm0
    ret
