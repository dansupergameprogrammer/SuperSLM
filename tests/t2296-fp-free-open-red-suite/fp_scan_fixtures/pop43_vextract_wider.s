// T-2380 (Curie) -- design Sec7 dim 11, forty-third population's own
// must-accept, the WIDER family specified but not yet measured on a real
// corpus (design's own ruling, D-SLM5037): the f-suffixed sibling of every
// measured i-suffixed form, both suffixes' 32x4/64x2/64x4 AVX-512 forms, and
// their vinsert-mnemonic counterparts.
    .text
    .globl VextractWiderAccept
    .type VextractWiderAccept,@function
VextractWiderAccept:
    vextractf128 $1, %ymm1, %xmm0
    vextracti32x4 $1, %zmm1, %xmm0
    vextracti64x2 $1, %zmm1, %xmm0
    vextractf32x4 $1, %zmm1, %xmm0
    vextractf64x2 $1, %zmm1, %xmm0
    vextractf64x4 $1, %zmm1, %ymm0
    ret

    .globl VinsertWiderAccept
    .type VinsertWiderAccept,@function
VinsertWiderAccept:
    vinserti128 $1, %xmm0, %ymm1, %ymm2
    vinsertf128 $1, %xmm0, %ymm1, %ymm2
    vinserti32x4 $1, %xmm0, %zmm1, %zmm2
    vinserti64x2 $1, %xmm0, %zmm1, %zmm2
    vinserti64x4 $1, %ymm0, %zmm1, %zmm2
    vinsertf32x4 $1, %xmm0, %zmm1, %zmm2
    vinsertf64x2 $1, %xmm0, %zmm1, %zmm2
    vinsertf64x4 $1, %ymm0, %zmm1, %zmm2
    ret
