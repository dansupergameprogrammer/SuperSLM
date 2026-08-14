// T-1986 GPU-serial port (B1, Sec7.1/Sec7.2): portable 128-bit facility, ported
// bit-exact from src/intmath.cpp's own MSVC (no-native-__int128) branch -- the
// struct-of-two-uint64 two's-complement path, translated 1:1 to HLSL free
// functions since HLSL has neither a native 128-bit integer nor operator
// overloading on user structs at this project's pinned compile target (cs_6_2
// -HV 2018). Every function here has a named C++ sibling in src/intmath.cpp;
// keep the two in lockstep -- this is the single algorithm the design's own
// build.ps1-derived shaders (Claude/Laplace/gpu-determinism/shaders/
// int64_battery.hlsl) already established the pattern for (umulhi/smulhi/
// wideMulShr/swideMulShr), extended here to the full signed/unsigned 128-bit
// add/sub/compare/shift set B1's four primitives need.
#ifndef SSLM_WIDE128_HLSLI
#define SSLM_WIDE128_HLSLI

struct U128 { uint64_t lo; uint64_t hi; };
struct S128 { uint64_t lo; uint64_t hi; };  // two's-complement; hi read as signed for sign

// 64x64 -> 128 unsigned (schoolbook, exact) -- intmath.cpp UMul (MSVC branch).
U128 UMul(uint64_t a, uint64_t b)
{
    uint64_t ll = (a & 0xFFFFFFFFULL) * (b & 0xFFFFFFFFULL);
    uint64_t lh = (a & 0xFFFFFFFFULL) * (b >> 32);
    uint64_t hl = (a >> 32) * (b & 0xFFFFFFFFULL);
    uint64_t hh = (a >> 32) * (b >> 32);
    uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
    U128 r;
    r.lo = (ll & 0xFFFFFFFFULL) | (mid << 32);
    r.hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    return r;
}

// 128 * (small u64) -> 128; caller guarantees the product fits 128 bits --
// intmath.cpp UMulWide.
U128 UMulWide(U128 a, uint64_t b)
{
    U128 lo = UMul(a.lo, b);
    uint64_t hi = a.hi * b;
    U128 r;
    r.lo = lo.lo;
    r.hi = lo.hi + hi;
    return r;
}

U128 UAdd64(U128 a, uint64_t b)
{
    uint64_t lo = a.lo + b;
    uint64_t carry = (lo < a.lo) ? 1ULL : 0ULL;
    U128 r;
    r.lo = lo;
    r.hi = a.hi + carry;
    return r;
}

U128 UTwice(U128 a)
{
    U128 r;
    r.lo = a.lo << 1;
    r.hi = (a.hi << 1) | (a.lo >> 63);
    return r;
}

// Logical right shift of an unsigned 128-bit value; result assumed to fit u64
// (every B1 caller's own contract, matching intmath.cpp UShrToU64).
uint64_t UShrToU64(U128 v, int k)
{
    if (k == 0) return v.lo;
    if (k >= 64) return v.hi >> (uint)(k - 64);
    return (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));
}

bool UFitsI64(U128 v)
{
    return v.hi == 0ULL && v.lo <= 0x7FFFFFFFFFFFFFFFULL;
}

S128 SFromI64(int64_t v)
{
    S128 r;
    r.lo = (uint64_t)v;
    r.hi = (v < 0) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
    return r;
}

// intmath.cpp SMul (MSVC branch): unsigned-magnitude multiply, then two's-
// complement negate if the operand signs differ.
S128 SMul(int64_t a, int64_t b)
{
    uint64_t ua = (a < 0) ? (~(uint64_t)a + 1ULL) : (uint64_t)a;
    uint64_t ub = (b < 0) ? (~(uint64_t)b + 1ULL) : (uint64_t)b;
    U128 m = UMul(ua, ub);
    S128 r;
    r.lo = m.lo;
    r.hi = m.hi;
    if ((a < 0) != (b < 0))
    {
        r.lo = ~r.lo;
        r.hi = ~r.hi;
        r.lo = r.lo + 1ULL;
        if (r.lo == 0ULL) r.hi = r.hi + 1ULL;
    }
    return r;
}

S128 SAdd(S128 a, S128 b)
{
    uint64_t lo = a.lo + b.lo;
    uint64_t carry = (lo < a.lo) ? 1ULL : 0ULL;
    S128 r;
    r.lo = lo;
    r.hi = a.hi + b.hi + carry;
    return r;
}

S128 SSub(S128 a, S128 b)
{
    uint64_t lo = a.lo - b.lo;
    uint64_t borrow = (a.lo < b.lo) ? 1ULL : 0ULL;
    S128 r;
    r.lo = lo;
    r.hi = a.hi - b.hi - borrow;
    return r;
}

S128 STwice(S128 a) { return SAdd(a, a); }

bool SGe(S128 a, S128 b)  // signed a >= b
{
    int64_t ah = (int64_t)a.hi;
    int64_t bh = (int64_t)b.hi;
    if (ah != bh) return ah > bh;
    return a.lo >= b.lo;
}

bool SLt(S128 a, S128 b) { return !SGe(a, b); }

// Arithmetic (floor) right shift; result assumed to fit i64 -- intmath.cpp
// SShrToI64.
int64_t SShrToI64(S128 v, int k)
{
    if (k == 0) return (int64_t)v.lo;
    uint64_t lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));
    return (int64_t)lo;
}

// Full-width arithmetic (floor) right shift, k in [0,63] -- intmath.cpp
// SShrFull (MSVC branch). BiasReconcileWide's own only caller of the
// equivalent C++ function never asks k outside this range (its caller checks
// the composed exponent is in [0,63] first).
S128 SShrFull(S128 v, int k)
{
    if (k <= 0) return v;
    uint64_t lo = (v.lo >> (uint)k) | (v.hi << (uint)(64 - k));
    int64_t hi = ((int64_t)v.hi) >> k;
    S128 r;
    r.lo = lo;
    r.hi = (uint64_t)hi;
    return r;
}

// Representable in int64_t: the high word is exactly the sign-extension of
// the low word's own sign bit -- intmath.cpp SFitsI64.
bool SFitsI64(S128 v)
{
    uint64_t sign_ext = (((int64_t)v.lo) < 0) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
    return v.hi == sign_ext;
}

int64_t SLow64(S128 v) { return (int64_t)v.lo; }

#endif  // SSLM_WIDE128_HLSLI
