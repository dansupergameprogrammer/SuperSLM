// T-1986 GPU-serial port, B2 (Sec6, Sec11 B2): RequantChainCheckedGpu.
// Guard-only tier: reproduces RequantChainChecked's own status (steps 0, 1-2,
// 3, 4, 5; checked_chain_funnel.cpp:253-333), never step 6's per-element code
// write (this suite's B2 fixtures assert only the returned status). Step 4's
// own MaxAbsReduceWide is a self-contained, single-thread reduction (this
// file's own, not B5's two-schedule production reduction -- B2 does not
// depend on B5 in Sec11's own dependency graph). Step 4's DynamicScaleReciprocal
// is not computed: its result never gates a rejection, only step 6's per-
// element write does, and step 6 does not run at this tier.
//
// Fixed-capacity input layout (host-packed, unused slots zeroed):
//   offset 0:    int64 n            (wide_row length, <= kMaxRow)
//   offset 8:    int64 n_incoming   (<= kMaxIncoming)
//   offset 16:   wide_row[0..kMaxRow-1]         (kMaxRow * 8 bytes)
//   offset X:    incoming[0..kMaxIncoming-1]    (kMaxIncoming * 16 bytes, m then e)
//   offset Y:    site_constant.m, site_constant.e (16 bytes)
//
// Output: int64 status_tag (0=Ok, 1=CarriedScaleMantissaOutOfDomain,
// 2=ChainInputOutOfDomain) -- a local tag, mapped to the real
// SslmForwardStatus enumerator by name on the host side (see
// check_rdp_exponent.hlsl's own header comment for why).
#include "funnel_guards.hlsli"

static const uint kMaxRow = 32;
static const uint kMaxIncoming = 32;
static const uint kRowOff = 16;
static const uint kIncomingOff = kRowOff + kMaxRow * 8;
static const uint kSiteOff = kIncomingOff + kMaxIncoming * 16;

ByteAddressBuffer   In  : register(t0);
RWByteAddressBuffer Out : register(u0);  // int64 status_tag

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t n = In.Load<int64_t>(0);
    int64_t n_incoming = In.Load<int64_t>(8);

    // Step 0: every incoming factor, then site_constant, must fit int32_t.
    int64_t tag = 0;
    bool rejected = false;
    for (int64_t i = 0; i < n_incoming && !rejected; ++i)
    {
        int64_t m = In.Load<int64_t>(kIncomingOff + (uint)i * 16u + 0u);
        if (!CarriedScaleMantissaFitsInt32Gpu(m))
        {
            tag = 1;  // CarriedScaleMantissaOutOfDomain
            rejected = true;
        }
    }
    int64_t site_m = In.Load<int64_t>(kSiteOff + 0u);
    int64_t site_e = In.Load<int64_t>(kSiteOff + 8u);
    if (!rejected && !CarriedScaleMantissaFitsInt32Gpu(site_m))
    {
        tag = 1;
        rejected = true;
    }

    // Steps 1-2: d_prime = MaxAbsReduceWide(wide_row, n) -- self-contained,
    // single-thread guard-tier reduction (this file's own; not B5's).
    int64_t d_prime = 0;
    if (!rejected)
    {
        uint64_t d = 0;
        for (int64_t i = 0; i < n; ++i)
        {
            int64_t xi = In.Load<int64_t>(kRowOff + (uint)i * 8u);
            uint64_t a = (xi < 0) ? (~(uint64_t)xi + 1ULL) : (uint64_t)xi;
            if (a > d) d = a;
        }
        if (d < 1ULL) d = 1ULL;
        d_prime = (d > 0x7FFFFFFFFFFFFFFFULL) ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)d;

        // Step 3: C29's own domain check.
        if (d_prime > (int64_t(1) << 31))
        {
            tag = 2;  // ChainInputOutOfDomain
            rejected = true;
        }
    }

    // Step 4: NormalizeScale only (its dn/s feed step 5's d_prime_factor;
    // DynamicScaleReciprocal's own result never gates a rejection here).
    int64_t ns_dn = 0;
    int ns_s = 0;
    if (!rejected)
    {
        NormalizeScaleGpu(d_prime, ns_dn, ns_s);
    }

    // Step 5: fold incoming[0..n_incoming-1], then site_constant, then the
    // d_prime_factor {ns.dn, -ns.s}, checking CombineCarriedScale's own
    // int32 precondition after every fold step (380b75f review N1).
    if (!rejected)
    {
        bool have_running = false;
        int64_t run_m = 0, run_e = 0;
        for (int64_t i = 0; i < n_incoming && !rejected; ++i)
        {
            int64_t im = In.Load<int64_t>(kIncomingOff + (uint)i * 16u + 0u);
            int64_t ie = In.Load<int64_t>(kIncomingOff + (uint)i * 16u + 8u);
            if (have_running)
            {
                int64_t new_m, new_e;
                CombineCarriedScaleGpu_(run_m, run_e, im, ie, new_m, new_e);
                run_m = new_m;
                run_e = new_e;
            }
            else
            {
                run_m = im;
                run_e = ie;
                have_running = true;
            }
            if (!CarriedScaleMantissaFitsInt32Gpu(run_m))
            {
                tag = 1;  // CarriedScaleMantissaOutOfDomain
                rejected = true;
            }
        }
        if (!rejected)
        {
            if (have_running)
            {
                int64_t new_m, new_e;
                CombineCarriedScaleGpu_(run_m, run_e, site_m, site_e, new_m, new_e);
                run_m = new_m;
                run_e = new_e;
            }
            else
            {
                run_m = site_m;
                run_e = site_e;
                have_running = true;
            }
            if (!CarriedScaleMantissaFitsInt32Gpu(run_m))
            {
                tag = 1;
                rejected = true;
            }
        }
        if (!rejected)
        {
            int64_t dpf_m = ns_dn;
            int64_t dpf_e = -(int64_t)ns_s;
            int64_t new_m, new_e;
            CombineCarriedScaleGpu_(run_m, run_e, dpf_m, dpf_e, new_m, new_e);
            run_m = new_m;
            run_e = new_e;
            if (!CarriedScaleMantissaFitsInt32Gpu(run_m))
            {
                tag = 1;
                rejected = true;
            }
        }
    }

    Out.Store<int64_t>(0, tag);
}
