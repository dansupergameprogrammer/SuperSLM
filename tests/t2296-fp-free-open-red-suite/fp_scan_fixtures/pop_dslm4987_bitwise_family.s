// T-2366 (Curie) -- D-SLM4987's own eight-mnemonic bitwise-family widening
// (design Sec4.1, fold round 39): orps/orpd/andps/andpd/andnps/andnpd/xorps/
// xorpd and their VEX forms are ruled to perform no floating-point arithmetic
// and to ACCEPT unconditionally, on any operand list -- joining pand/por/
// pandn/pxor on the same footing. Sixteen real, assembled instructions, one
// per symbol, every operand list DELIBERATELY NOT uniformly the same
// register (the "genuine bitwise operation, not the self-zeroing idiom" case
// pop16_vxorps.s already isolates for vxorps alone) -- this fixture is the
// same construction generalized to the remaining seven mnemonics and their
// VEX forms, closing the gap T-2365's own coverage audit named (population
// 35's own text names only three of the eight explicitly).
//
// Every one of these sixteen instructions is a pure Boolean function of its
// operand bits under every x86 encoding (packed-integer or packed-single/
// -double) -- no rounding, no exception, no vendor-dependent behavior -- and
// the pre-fold-39 classifier accepts NONE of them: the six OR/AND/ANDN
// mnemonics (legacy and VEX) are named nowhere in _x86_check_a and fall
// through to its own default REJECT; the two XOR mnemonics (legacy and VEX)
// are named only for the self-zeroing (all-operands-identical) idiom, which
// none of the constructions below is.

.intel_syntax noprefix
.text

.globl Legacy_Orps
.type Legacy_Orps,@function
Legacy_Orps:
    orps xmm0, xmm1
    ret

.globl Legacy_Orpd
.type Legacy_Orpd,@function
Legacy_Orpd:
    orpd xmm0, xmm1
    ret

.globl Legacy_Andps
.type Legacy_Andps,@function
Legacy_Andps:
    andps xmm0, xmm1
    ret

.globl Legacy_Andpd
.type Legacy_Andpd,@function
Legacy_Andpd:
    andpd xmm0, xmm1
    ret

.globl Legacy_Andnps
.type Legacy_Andnps,@function
Legacy_Andnps:
    andnps xmm0, xmm1
    ret

.globl Legacy_Andnpd
.type Legacy_Andnpd,@function
Legacy_Andnpd:
    andnpd xmm0, xmm1
    ret

.globl Legacy_Xorps
.type Legacy_Xorps,@function
Legacy_Xorps:
    xorps xmm0, xmm1
    ret

.globl Legacy_Xorpd
.type Legacy_Xorpd,@function
Legacy_Xorpd:
    xorpd xmm0, xmm1
    ret

.globl Vex_Vorps
.type Vex_Vorps,@function
Vex_Vorps:
    vorps xmm0, xmm1, xmm2
    ret

.globl Vex_Vorpd
.type Vex_Vorpd,@function
Vex_Vorpd:
    vorpd xmm0, xmm1, xmm2
    ret

.globl Vex_Vandps
.type Vex_Vandps,@function
Vex_Vandps:
    vandps xmm0, xmm1, xmm2
    ret

.globl Vex_Vandpd
.type Vex_Vandpd,@function
Vex_Vandpd:
    vandpd xmm0, xmm1, xmm2
    ret

.globl Vex_Vandnps
.type Vex_Vandnps,@function
Vex_Vandnps:
    vandnps xmm0, xmm1, xmm2
    ret

.globl Vex_Vandnpd
.type Vex_Vandnpd,@function
Vex_Vandnpd:
    vandnpd xmm0, xmm1, xmm2
    ret

.globl Vex_Vxorps
.type Vex_Vxorps,@function
Vex_Vxorps:
    vxorps xmm0, xmm1, xmm2
    ret

.globl Vex_Vxorpd
.type Vex_Vxorpd,@function
Vex_Vxorpd:
    vxorpd xmm0, xmm1, xmm2
    ret
