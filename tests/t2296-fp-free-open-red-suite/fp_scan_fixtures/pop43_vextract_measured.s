// T-2380 (Curie) -- design Sec7 dim 11, forty-third population's own
// must-accept, the subset measured on the real GCC-built corpus (D-SLM5032,
// matmul.o: 1 vextracti128 + 1 vextracti64x4 symbol).
    .text
    .globl VextractMeasuredAccept
    .type VextractMeasuredAccept,@function
VextractMeasuredAccept:
    vextracti128 $1, %ymm1, %xmm0
    vextracti64x4 $1, %zmm1, %ymm0
    ret
