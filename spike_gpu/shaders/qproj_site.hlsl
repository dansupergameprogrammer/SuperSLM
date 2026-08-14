// T-1979 GPU walking-skeleton spike: ONE composed ProjectAndFunnel site
// (layer0.q_proj) -- int8 GEMM -> WSC1 per-channel weight-scale fold ->
// optional BIA1 bias reconciliation -> the funnel (MaxAbsReduceWide ->
// NormalizeScale -> DynamicScaleReciprocal -> RequantTokenCodeWide per
// element -> CombineCarriedScale), bit-for-bit the same construction as the
// production CPU code (src/forward/forward_sites.cpp's ProjectAndFunnel and
// src/forward/checked_chain_funnel.cpp's RequantChainChecked;
// src/intmath.cpp's DynamicScaleReciprocal/RequantTokenCodeWide/
// BiasReconcileWide).
//
// Disposable spike shader (branch brunel/t1979-gpu-skeleton). Single thread,
// single dispatch: the walking skeleton's job is to prove the composition is
// bit-exact on real D3D12/DXIL hardware, not to prove a parallel dispatch
// strategy -- that is the port design's own question (see the build log).
//
// 128-bit intermediates are built from two uint64_t (lo, hi), mirroring this
// codebase's own MSVC-path portable facility (intmath.cpp / forward_sites.cpp,
// which use the identical {lo,hi} struct because MSVC has no native __int128)
// -- the same construction is needed here because HLSL has no native 128-bit
// integer type either. umulhi/smulhi below are the SAME idiom the H2
// int64_battery experiment already proved bit-exact against the CPU on this
// exact hardware (Claude/Laplace/gpu-determinism/shaders/int64_battery.hlsl).
//
// Buffers (root signature: 2 x 32-bit root constants, 7 root SRVs, 1 root UAV):
//   b0: HiddenSize (uint), HasBias (uint)
//   t0: In_ActCodes  -- hidden_size x int32 (widened int8), the real
//                       layer0.attn_norm output (RmsNormSite's own real result)
//   t1: In_Weight    -- hidden_size*hidden_size x int32 (widened int8),
//                       row-major [out_channels, in_channels] (WGT1 q_proj)
//   t2: In_FoldId    -- hidden_size x int32 (WSC1 identity flag per channel)
//   t3: In_FoldMult  -- hidden_size x int32 (WSC1 mult per channel)
//   t4: In_FoldShift -- hidden_size x int32 (WSC1 shift per channel)
//   t5: In_Bias      -- hidden_size x int64 (BIA1 bias per channel; ignored if HasBias==0)
//   t6: In_Scalars   -- 4 x int64: in_scale.m, in_scale.e, site_constant.m, site_constant.e
//   u0: Out          -- see the layout comment at the bottom of this file
ByteAddressBuffer   In_ActCodes  : register(t0);
ByteAddressBuffer   In_Weight    : register(t1);
ByteAddressBuffer   In_FoldId    : register(t2);
ByteAddressBuffer   In_FoldMult  : register(t3);
ByteAddressBuffer   In_FoldShift : register(t4);
ByteAddressBuffer   In_Bias      : register(t5);
ByteAddressBuffer   In_Scalars   : register(t6);
RWByteAddressBuffer  Out         : register(u0);
cbuffer C : register(b0) { uint HiddenSize; uint HasBias; };

// --- 64x64 -> 128 multiply, the H2-proven idiom (int64_battery.hlsl) --------
uint64_t umulhi(uint64_t a, uint64_t b) {
    uint64_t aL = a & 0xffffffffULL, aH = a >> 32;
    uint64_t bL = b & 0xffffffffULL, bH = b >> 32;
    uint64_t ll = aL * bL, lh = aL * bH, hl = aH * bL, hh = aH * bH;
    uint64_t mid = (ll >> 32) + (lh & 0xffffffffULL) + (hl & 0xffffffffULL);
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

// --- Unsigned 128-bit facility (mirrors forward_sites.cpp's own U128) ------
struct U128 { uint64_t lo; uint64_t hi; };

U128 UMul64(uint64_t a, uint64_t b) {
    U128 r;
    r.lo = a * b;         // wrapping low 64
    r.hi = umulhi(a, b);  // high 64
    return r;
}

// 128 * small-u64 -> 128; caller guarantees the true product fits 128 bits
// (RequantTokenCodeWide's own use: b == 127, matching forward_sites.cpp's
// U128MulSmall contract exactly).
U128 UMulSmall(U128 a, uint64_t b) {
    U128 lo = UMul64(a.lo, b);
    uint64_t hi = a.hi * b;
    U128 r; r.lo = lo.lo; r.hi = lo.hi + hi;
    return r;
}

U128 UAdd64(U128 a, uint64_t b) {
    uint64_t lo = a.lo + b;
    uint64_t carry = (lo < a.lo) ? 1UL : 0UL;
    U128 r; r.lo = lo; r.hi = a.hi + carry;
    return r;
}

U128 UTwice(U128 a) {
    U128 r; r.lo = a.lo << 1; r.hi = (a.hi << 1) | (a.lo >> 63);
    return r;
}

uint64_t UShrToU64(U128 v, int k) {
    if (k == 0) return v.lo;
    if (k >= 64) return v.hi >> (uint)(k - 64);
    return (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));
}

// --- Signed 128-bit facility (mirrors intmath.cpp's own S128, MSVC path) ---
struct S128 { uint64_t lo; uint64_t hi; };  // two's complement

S128 SFromI64(int64_t v) {
    S128 r; r.lo = (uint64_t)v; r.hi = (v < 0) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
    return r;
}

S128 SMul(int64_t a, int64_t b) {
    uint64_t ua = (a < 0) ? (~(uint64_t)a + 1ULL) : (uint64_t)a;
    uint64_t ub = (b < 0) ? (~(uint64_t)b + 1ULL) : (uint64_t)b;
    U128 m = UMul64(ua, ub);
    S128 r; r.lo = m.lo; r.hi = m.hi;
    if ((a < 0) != (b < 0)) {
        r.lo = ~r.lo;
        r.hi = ~r.hi;
        r.lo = r.lo + 1;
        if (r.lo == 0) r.hi = r.hi + 1;
    }
    return r;
}

S128 SAdd(S128 a, S128 b) {
    uint64_t lo = a.lo + b.lo;
    uint64_t carry = (lo < a.lo) ? 1UL : 0UL;
    S128 r; r.lo = lo; r.hi = a.hi + b.hi + carry;
    return r;
}

S128 SSub(S128 a, S128 b) {
    uint64_t lo = a.lo - b.lo;
    uint64_t borrow = (a.lo < b.lo) ? 1UL : 0UL;
    S128 r; r.lo = lo; r.hi = a.hi - b.hi - borrow;
    return r;
}

S128 STwice(S128 a) { return SAdd(a, a); }

bool SGe(S128 a, S128 b) {
    int64_t ah = (int64_t)a.hi, bh = (int64_t)b.hi;
    if (ah != bh) return ah > bh;
    return a.lo >= b.lo;
}

bool SLt(S128 a, S128 b) { return !SGe(a, b); }

int64_t SShrToI64(S128 v, int k) {
    if (k == 0) return (int64_t)v.lo;
    uint64_t lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));
    return (int64_t)lo;
}

// Full-width arithmetic (floor) right shift, k in [1,63] (caller guards k<=0
// -- matching intmath.cpp's SShrFull, whose own comment notes "64-k" is UB at
// k==0, avoided the same way here).
S128 SShrFull(S128 v, int k) {
    if (k <= 0) return v;
    uint64_t lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));
    int64_t hi = ((int64_t)v.hi) >> k;
    S128 r; r.lo = lo; r.hi = (uint64_t)hi;
    return r;
}

// --- C1/C2/C3 requant primitives (intmath.cpp) ------------------------------
static const int32_t kInt32Max = 2147483647;

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t result = (ab + ((int64_t)1 << 30)) >> 31;
    return (result > (int64_t)kInt32Max) ? kInt32Max : (int32_t)result;
}

int32_t RoundingDivideByPOT_i32(int32_t x, int exponent) {
    if (exponent == 0) return x;
    uint32_t mask = ((uint32_t)1 << exponent) - 1u;
    uint32_t remainder = (uint32_t)x & mask;
    uint32_t threshold = (mask >> 1) + ((x < 0) ? 1u : 0u);
    int32_t shifted = x >> exponent;
    return shifted + ((remainder > threshold) ? 1 : 0);
}

int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t mult, int shift) {
    return RoundingDivideByPOT_i32(SaturatingRoundingDoublingHighMul(x, mult), shift);
}

// C24/C25's WSC1 fold-apply dispatch (forward_sites.cpp's ApplyWeightScaleFold).
int64_t ApplyWeightScaleFold(int64_t acc, int32_t identity, int32_t mult, int32_t shift) {
    if (identity != 0) return acc;
    return (int64_t)MultiplyByQuantizedMultiplier((int32_t)acc, mult, shift);
}

// C19 -- DynamicScaleReciprocal, bit-for-bit the same construction as
// intmath.cpp (3 Newton iterations + 2 fixed correction steps, all
// unconditional -- op-count determinism, matching the CPU's own §14 law).
int64_t DynamicScaleReciprocal(int64_t dn) {
    const int64_t kC32   = (2 * ((int64_t)48 << 31) + 17) / 34;
    const int64_t kC32_2 = (2 * ((int64_t)32 << 31) + 17) / 34;
    int64_t y = kC32 - ((kC32_2 * dn) >> 31);

    // T-1987 fix (T-1983 review M-3): distinct loop-variable names. Under
    // -HV 2018 (this file's own compile path, matching the sibling substrate)
    // a `for`-scoped variable is NOT block-scoped -- it leaks into the
    // enclosing function scope -- so a second `for (int i = ...)` here
    // redeclares the first loop's `i` rather than shadowing an unrelated
    // outer `i`. DXC's only warning on this file sat here, inside the one
    // function whose unconditional, data-independent op count is a
    // documented CPU-side determinism law (build log §14).
    for (int newton_i = 0; newton_i < 3; ++newton_i) {
        int64_t dn_y = SShrToI64(SMul(dn, y), 31);
        int64_t delta = ((int64_t)1 << 32) - dn_y;
        y = SShrToI64(SMul(y, delta), 31);
    }
    for (int correct_i = 0; correct_i < 2; ++correct_i) {
        S128 residual_2x = STwice(SSub(SFromI64((int64_t)1 << 62), SMul(y, dn)));
        if (SGe(residual_2x, SFromI64(dn))) {
            y = y + 1;
        } else if (SLt(residual_2x, SFromI64(-dn))) {
            y = y - 1;
        }
    }
    return y;
}

// C21 -- Clz64/NormalizeScale, bit-for-bit the same construction as intmath.cpp.
int Clz64(uint64_t n) {
    uint hi = (uint)(n >> 32);
    if (hi != 0) return 31 - (int)firstbithigh(hi);
    uint lo = (uint)(n & 0xFFFFFFFFUL);
    if (lo == 0) return 64;  // n==0 is out of this function's own contract; never reached here
    return 32 + (31 - (int)firstbithigh(lo));
}

struct NormalizedScale { int64_t dn; int s; };

NormalizedScale NormalizeScale(int64_t d_prime) {
    int p = 63 - Clz64((uint64_t)d_prime);
    int s = 30 - p;
    int64_t dn = (s >= 0) ? (d_prime << s) : (d_prime >> 1);
    NormalizedScale r; r.dn = dn; r.s = s;
    return r;
}

// C22/F-S3-7 -- RequantTokenCodeWide, bit-for-bit the same 128-bit-intermediate
// composite as intmath.cpp.
int32_t RequantTokenCodeWide(int64_t x_i, int64_t r, int s) {
    int exponent = 62 - s;  // in [32, 63]
    uint64_t abs_x = (x_i < 0) ? (~(uint64_t)x_i + 1ULL) : (uint64_t)x_i;

    U128 prod0 = UMul64(abs_x, (uint64_t)r);
    U128 prod = UMulSmall(prod0, 127ULL);
    U128 numerator = UAdd64(UTwice(prod), ((uint64_t)1 << exponent));
    uint64_t magnitude = UShrToU64(numerator, exponent + 1);

    if (magnitude > 127UL) magnitude = 127UL;
    int32_t q = (int32_t)magnitude;
    return (x_i < 0) ? -q : q;
}

// C26's carried-scale combine (checked_chain_funnel.cpp's CombineCarriedScale).
// The SaturatingAdd64 overflow-guard machinery is not ported -- this spike's
// real operand range (real artifact CarriedScale.e values) never approaches
// int64 overflow, and the guard is a documented simplification (see the build
// log's port-requirements report).
void CombineCarriedScale(int64_t am, int64_t ae, int64_t bm, int64_t be,
                          out int64_t om, out int64_t oe) {
    int32_t ma = (int32_t)am;
    int32_t mb = (int32_t)bm;
    int64_t e = ae + be + 31;
    int64_t m = (int64_t)SaturatingRoundingDoublingHighMul(ma, mb);
    if (m < ((int64_t)1 << 30)) {
        m = m << 1;
        e = e - 1;
    }
    om = m; oe = e;
}

// C28's bias-reconciliation core (intmath.cpp's BiasReconcileWide /
// forward_sites.cpp's BiasReconcile). The composed-exponent domain guard
// (RoundingDivideByPotComposedExponentInDomain) is not ported for the same
// reason CombineCarriedScale's overflow guard is not -- this site's real
// (q_B=30, e_a) is comfortably inside [0,63] on the real artifact.
S128 RoundingDivideByPOTWide(S128 x, int exponent) {
    uint64_t mask = ((uint64_t)1 << exponent) - 1ULL;
    uint64_t x_lo = x.lo;
    bool x_negative = ((int64_t)x.hi) < 0;
    uint64_t remainder = x_lo & mask;
    uint64_t threshold = (mask >> 1) + (x_negative ? 1ULL : 0ULL);
    S128 shifted = SShrFull(x, exponent);
    if (remainder > threshold) return SAdd(shifted, SFromI64(1));
    return shifted;
}

int64_t BiasReconcile(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a) {
    int64_t exponent = q_b + 62 + e_a;
    S128 wide = SMul(b, r_a);
    S128 rounded = RoundingDivideByPOTWide(wide, (int)exponent);
    return (int64_t)rounded.lo;
}

// --- Output layout (all offsets in bytes, HiddenSize substituted at runtime) --
//   [0, HiddenSize*8)                                   : wide_row[j]  (int64, post WSC1+bias, pre-funnel)
//   [HiddenSize*8 + 0,  +8)                              : d_prime      (int64)
//   [HiddenSize*8 + 8,  +8)                              : dn           (int64)
//   [HiddenSize*8 + 16, +8)                              : s            (int64, widened)
//   [HiddenSize*8 + 24, +8)                              : r            (int64)
//   [HiddenSize*8 + 32, +HiddenSize*4)                   : out_codes[j] (int32, widened int8)
//   [HiddenSize*8 + 32 + HiddenSize*4, +8)                : out_scale.m  (int64)
//   [HiddenSize*8 + 32 + HiddenSize*4 + 8, +8)            : out_scale.e  (int64)
[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint hidden_size = HiddenSize;

    int64_t in_scale_m = (int64_t)In_Scalars.Load<uint64_t>(0);
    int64_t in_scale_e = (int64_t)In_Scalars.Load<uint64_t>(8);
    int64_t site_m     = (int64_t)In_Scalars.Load<uint64_t>(16);
    int64_t site_e      = (int64_t)In_Scalars.Load<uint64_t>(24);

    // --- Phase 1: GEMM -> WSC1 fold -> optional BIA1 bias, per output channel.
    int64_t r_a = 0;
    if (HasBias != 0) {
        r_a = DynamicScaleReciprocal(in_scale_m);  // CarriedScaleReciprocal's own body
    }

    for (uint j = 0; j < hidden_size; ++j) {
        int64_t acc = 0;
        uint w_row_base = j * hidden_size * 4;
        for (uint k = 0; k < hidden_size; ++k) {
            int a = In_ActCodes.Load<int>(k * 4);
            int w = In_Weight.Load<int>(w_row_base + k * 4);
            acc += (int64_t)a * (int64_t)w;
        }
        int32_t id    = In_FoldId.Load<int>(j * 4);
        int32_t mult  = In_FoldMult.Load<int>(j * 4);
        int32_t shift = In_FoldShift.Load<int>(j * 4);
        acc = ApplyWeightScaleFold(acc, id, mult, shift);

        if (HasBias != 0) {
            int64_t bias_j = In_Bias.Load<int64_t>(j * 8);
            acc = acc + BiasReconcile(bias_j, 30, r_a, in_scale_e);
        }

        Out.Store<int64_t>(j * 8, acc);
    }

    // --- Phase 2: the funnel. --------------------------------------------
    uint64_t dmax = 0;
    for (uint j2 = 0; j2 < hidden_size; ++j2) {
        int64_t v = Out.Load<int64_t>(j2 * 8);
        uint64_t a = (v < 0) ? (~(uint64_t)v + 1ULL) : (uint64_t)v;
        if (a > dmax) dmax = a;
    }
    if (dmax < 1) dmax = 1;
    int64_t d_prime = (dmax > 0x7FFFFFFFFFFFFFFFULL) ? (int64_t)0x7FFFFFFFFFFFFFFFLL : (int64_t)dmax;

    NormalizedScale ns = NormalizeScale(d_prime);
    int64_t r = DynamicScaleReciprocal(ns.dn);

    int64_t run_m = in_scale_m, run_e = in_scale_e;
    int64_t tmp_m, tmp_e;
    CombineCarriedScale(run_m, run_e, site_m, site_e, tmp_m, tmp_e);
    run_m = tmp_m; run_e = tmp_e;
    CombineCarriedScale(run_m, run_e, ns.dn, (int64_t)(-ns.s), tmp_m, tmp_e);
    run_m = tmp_m; run_e = tmp_e;

    uint out_codes_base = hidden_size * 8 + 32;
    for (uint j3 = 0; j3 < hidden_size; ++j3) {
        int64_t v = Out.Load<int64_t>(j3 * 8);
        int32_t code = RequantTokenCodeWide(v, r, ns.s);
        Out.Store<int>(out_codes_base + j3 * 4, code);
    }

    uint meta_base = hidden_size * 8;
    Out.Store<int64_t>(meta_base + 0,  d_prime);
    Out.Store<int64_t>(meta_base + 8,  ns.dn);
    Out.Store<int64_t>(meta_base + 16, (int64_t)ns.s);
    Out.Store<int64_t>(meta_base + 24, r);

    uint scale_base = out_codes_base + hidden_size * 4;
    Out.Store<int64_t>(scale_base + 0, run_m);
    Out.Store<int64_t>(scale_base + 8, run_e);
}
