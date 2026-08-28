// T-2347 (Curie) -- population seventeen, must-accept half: check (C)'s
// resolve-then-classify carve-out (design Sec4.1 fold round 35, D-SLM4886).
// IndirectCallSibling performs a genuine INDIRECT (memory-operand) call
// whose relocation resolves to SiblingCallee, a real function defined in a
// SEPARATE compiled object (pop17_carveout_callee.s) -- undefined in THIS
// object's own symbol table, present in `corpus_symbols`. This is the
// "resolves to a symbol, external or in-object" half of the carve-out: the
// relocation names the callable entity itself (SiblingCallee, a function
// symbol), not a data holder whose contents determine the target -- so
// resolve-then-classify must ACCEPT this edge once the relocation is read,
// regardless of the memory-operand encoding. Today (pre-fold-35-fix), check
// (C) rejects on `_operand_is_direct_immediate` before ever reading the
// relocation's own target, so this construction currently REJECTs -- the
// exact defect Poirot's Critical C1 and the re-commissioning both measured.
// Body is entirely integer; no floating-point instruction anywhere.

    .text
    .globl IndirectCallSibling
    .type IndirectCallSibling,@function
IndirectCallSibling:
    subq $8, %rsp
    call *SiblingCallee(%rip)
    addq $8, %rsp
    ret
