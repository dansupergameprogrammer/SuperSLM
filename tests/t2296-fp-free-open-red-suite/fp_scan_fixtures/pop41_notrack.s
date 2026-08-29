// T-2380 (Curie) -- design Sec7 dim 11, forty-first population's own
// must-accept (D-SLM5037, D-SLM5058): `notrack jmp`/`notrack call`, the
// CET/legacy-MPX branch-tracking prefix ahead of a real indirect jmp/call
// that -fcf-protection emits. Real GCC-built corpus measurement (D-SLM5032):
// 8 real notrack jmp symbols. Two separate symbols so each gets its own
// ACCEPT/REJECT verdict from scan_object.
    .text
    .globl NotrackJmpAccept
    .type NotrackJmpAccept,@function
NotrackJmpAccept:
    notrack jmp *%rax

    .globl NotrackCallAccept
    .type NotrackCallAccept,@function
NotrackCallAccept:
    notrack call *%rax
