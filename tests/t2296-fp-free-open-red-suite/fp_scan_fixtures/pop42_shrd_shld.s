// T-2380 (Curie) -- design Sec7 dim 11, forty-second population's own
// must-accept (D-SLM5037): shrd/shld, ordinary integer double-precision
// shifts (naming operand width, not a float type) -- the identical semantic
// class as shl/shr/sar, already on the allow-list. Real GCC-built corpus
// measurement (D-SLM5032): 5 shrd + 3 shld symbols.
    .text
    .globl ShrdShldAccept
    .type ShrdShldAccept,@function
ShrdShldAccept:
    shrd %cl, %ebx, %eax
    shld %cl, %ebx, %eax
    ret
