// T-1986 GPU-serial port, B2 (Sec6, Sec11 B2): CheckRoundingDivideByPotExponentDomainGpu.
// Bit-exact port of checked_chain_funnel.cpp:420-435. Output is a local status
// TAG (0=Ok, 3=RoundingDivideByPotExponentOutOfDomain), not the real
// SslmForwardStatus ordinal -- the host side maps the tag to the named C++
// enumerator by switch, never by hand-counted ordinal (StandardsDocument.md
// Sec5.4: exactness verified at source, not by construction/inspection of an
// enum declaration).
#include "funnel_guards.hlsli"

ByteAddressBuffer   In  : register(t0);  // int64 q_B, int64 e_a
RWByteAddressBuffer Out : register(u0);  // int64 status_tag

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t q_B = In.Load<int64_t>(0);
    int64_t e_a = In.Load<int64_t>(8);

    int64_t unused = 0;
    int64_t tag = 0;
    if (!RoundingDivideByPotComposedExponentInDomainGpu(q_B, e_a, unused))
    {
        tag = 3;  // RoundingDivideByPotExponentOutOfDomain
    }
    Out.Store<int64_t>(0, tag);
}
