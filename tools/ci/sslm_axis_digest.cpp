// S-HARDEN-6 axis harness -- cross-toolchain / cross-optimization bit-identity probe.
//
// Promoted from a disposable Laplace commission instrument
// (Claude/Laplace/sharden6-axis-harness/sslm_axis_digest.cpp) into committed
// CI tooling (design Sec2.1 component 2, T-411/T-410). Built and run by the
// five S-HARDEN-6 digest jobs (linux-x64-gcc-digest, linux-x64-clang-digest,
// windows-msvc-digest, windows-clangcl-digest, macos-arm64-digest) plus the
// SUPERSLM_FORCE_SCALAR_MATMUL axis, compared by axis-digest-compare
// (.github/workflows/tests.yml). Still not a suite test -- it produces a
// digest for a CI job to compare, not a pass/fail assertion of its own
// (local_invariant_failures below is informational, folded into the digest
// so a scalar/SIMD divergence changes the hash rather than only printing).
//
// What it does: drives every Layer-1 arithmetic primitive that lies on the reproducible
// path over a fixed, seeded input population, serializes every result as explicit
// little-endian bytes, and folds those bytes into one SHA-256 per section plus one
// global SHA-256. Two builds of this program agree bit-for-bit iff every primitive
// they contain agrees bit-for-bit on this population.
//
// Design constraints, because the instrument must not be the thing that diverges:
//   * No <iostream> formatting on any digested value -- results are serialized by hand.
//   * No std::random -- the distribution of std::uniform_int_distribution is
//     implementation-defined and would diverge between libstdc++ and MSVC's STL for
//     reasons that have nothing to do with the kernels. A pinned splitmix64 supplies
//     every pseudo-random input.
//   * No undefined behaviour in the driver. Every primitive with a caller-ensures
//     precondition is fed only in-domain values. The i-exp evaluation is gated by
//     construction rather than by discipline since S-HARDEN-0: IExpEvaluate takes a
//     validated IExpConstruction, so there is no ungated form to misuse. On a
//     pre-S-HARDEN-0 pin the legacy arm gates IExpFromConstants through
//     IExpConstantsInDomain, which is the guard that header told callers to use.
//     UB is exactly the license a compiler needs to legally diverge, so a harness that
//     invokes it cannot distinguish "the construction diverges" from "the harness lied".
//   * The SHA-256 used to fold results is the project's own, so the first section
//     self-checks it against the FIPS 180-4 vector before anything else is trusted.
//
// Build (see run_axes.sh / run_axes.ps1):
//   <cxx> -std=c++20 -I<repo>/include <repo>/src/*.cpp sslm_axis_digest.cpp -o sslm_axis_digest

#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/sha256.h"
#include "superslm/silu_lut.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using superslm::Sha256;

// --- the pinned generator ------------------------------------------------------
// splitmix64, integer-only, identical on every conforming C++20 implementation.
struct Rng {
	uint64_t s;
	explicit Rng(uint64_t seed) : s(seed) {}
	uint64_t Next() {
		s += 0x9e3779b97f4a7c15ULL;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		return z ^ (z >> 31);
	}
	// Uniform-enough over [lo, hi] for a determinism probe; the point is that the
	// SAME values are produced everywhere, not that they are well-distributed.
	int64_t InRange(int64_t lo, int64_t hi) {
		const uint64_t span = static_cast<uint64_t>(hi - lo) + 1ULL;
		return span == 0 ? static_cast<int64_t>(Next()) : lo + static_cast<int64_t>(Next() % span);
	}
};

// --- explicit little-endian serialization --------------------------------------
// Every digested value goes through these. Nothing is digested by reinterpreting
// object representation, so struct layout and padding cannot leak into a hash.
struct Sink {
	Sha256 h;
	uint64_t count = 0;

	void U8(uint8_t v) { h.Update(&v, 1); ++count; }
	void U64(uint64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++count;
	}
	void I64(int64_t v) { U64(static_cast<uint64_t>(v)); }
	void I32(int32_t v) { U64(static_cast<uint64_t>(static_cast<int64_t>(v))); }
	void I8(int8_t v) { U64(static_cast<uint64_t>(static_cast<int64_t>(v))); }
	void Bool(bool v) { U8(v ? 1U : 0U); }
	void Bytes(const uint8_t* p, size_t n) {
		h.Update(p, n);
		count += n;
	}
};

struct Section {
	const char* name;
	Sink sink;
	uint8_t digest[32]{};
	bool ok = true;         // section-local invariants (e.g. scalar == SIMD)
	const char* note = "";  // set when ok == false
};

std::vector<Section*>& Registry() {
	static std::vector<Section*> r;
	return r;
}

Section& NewSection(const char* name) {
	Section* s = new Section();
	s->name = name;
	Registry().push_back(s);
	return *s;
}

// --- 1. sha256 self-check ------------------------------------------------------
// The instrument folds every other section through this hash. If it is wrong, every
// downstream digest is wrong in a way that looks like a kernel divergence. Check the
// FIPS 180-4 "abc" vector and a multi-block vector before trusting anything.
void SectionSha256() {
	Section& sec = NewSection("sha256");

	{
		const char* msg = "abc";
		uint8_t d[32];
		superslm::Sha256Hash(reinterpret_cast<const uint8_t*>(msg), 3, d);
		static const char* kExpect = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
		if (superslm::ToHex(d) != kExpect) {
			sec.ok = false;
			sec.note = "FIPS 180-4 'abc' vector mismatch";
		}
		sec.sink.Bytes(d, 32);
	}
	{
		// Multi-block + length-padding edge: 1,000,000 'a' is the FIPS long vector.
		Sha256 h;
		std::vector<uint8_t> chunk(1000, 'a');
		for (int i = 0; i < 1000; ++i) h.Update(chunk.data(), chunk.size());
		uint8_t d[32];
		h.Final(d);
		static const char* kExpect = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
		if (superslm::ToHex(d) != kExpect) {
			sec.ok = false;
			sec.note = "FIPS 180-4 1e6-'a' vector mismatch";
		}
		sec.sink.Bytes(d, 32);
	}
	{
		// Every buffer length across a block boundary, so Update's internal buffering
		// is exercised at each residue rather than only at 0.
		std::vector<uint8_t> buf(200);
		Rng rng(0xA1B2C3D4E5F60718ULL);
		for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(rng.Next() & 0xffU);
		for (size_t n = 0; n <= 200; ++n) {
			uint8_t d[32];
			superslm::Sha256Hash(buf.data(), n, d);
			sec.sink.Bytes(d, 32);
		}
	}
}

// --- 2. C1/C2/C3 requant primitives --------------------------------------------
void SectionRequant() {
	Section& sec = NewSection("c1c2c3_requant");

	static const int32_t kEdge[] = {
	    superslm::kInt32Min, superslm::kInt32Min + 1, -2147483647, -1073741824, -65537, -65536,
	    -32769,              -32768,                  -1024,       -3,          -2,     -1,
	    0,                   1,                       2,           3,           1023,   32767,
	    32768,               65535,                   65536,       1073741823,  1073741824,
	    2147483646,          superslm::kInt32Max};
	constexpr size_t kEdgeN = sizeof(kEdge) / sizeof(kEdge[0]);

	// C2 over the full edge cross-product -- includes (INT32_MIN, INT32_MIN), the one
	// saturating pair the header names.
	for (size_t i = 0; i < kEdgeN; ++i)
		for (size_t j = 0; j < kEdgeN; ++j)
			sec.sink.I32(superslm::SaturatingRoundingDoublingHighMul(kEdge[i], kEdge[j]));

	// C3 int32 over every legal exponent [0,31] x the edge set.
	for (size_t i = 0; i < kEdgeN; ++i)
		for (int e = 0; e <= 31; ++e) sec.sink.I32(superslm::RoundingDivideByPOT(kEdge[i], e));

	// C3 int64 over every legal exponent [0,63].
	Rng rng(0x5151515151515151ULL);
	for (int rep = 0; rep < 512; ++rep) {
		const int64_t x = static_cast<int64_t>(rng.Next());
		for (int e = 0; e <= 63; ++e) sec.sink.I64(superslm::RoundingDivideByPOT(x, e));
	}

	// C1 composition.
	for (int rep = 0; rep < 20000; ++rep) {
		const int32_t x = static_cast<int32_t>(rng.InRange(superslm::kInt32Min, superslm::kInt32Max));
		const int32_t m = static_cast<int32_t>(rng.InRange(1, superslm::kInt32Max));
		const int shift = static_cast<int>(rng.InRange(0, 31));
		sec.sink.I32(superslm::MultiplyByQuantizedMultiplier(x, m, shift));
	}
}

// --- 3. C19/C20/C21/C22 per-token dynamic-scale chain ---------------------------
void SectionDynamicScale() {
	Section& sec = NewSection("c19c22_dynamic_scale");

	// Clz64 over every single-bit value plus every all-ones prefix -- both boundaries
	// of the bit-scan, which is the primitive most likely to sit on a compiler
	// intrinsic (__builtin_clzll vs _BitScanReverse64) and therefore most exposed to a
	// toolchain difference.
	for (int b = 0; b < 64; ++b) {
		sec.sink.I32(superslm::Clz64(1ULL << b));
		sec.sink.I32(superslm::Clz64((1ULL << b) | 1ULL));
		sec.sink.I32(superslm::Clz64(~0ULL >> b));
	}

	// C21 over the full normalized domain boundaries plus a sweep.
	static const int64_t kDp[] = {1,          2,          3,          1023,       1024,
	                              1073741823, 1073741824, 1073741825, 2147483646, 2147483647,
	                              2147483648LL};
	for (int64_t d : kDp) {
		const superslm::NormalizedScale ns = superslm::NormalizeScale(d);
		sec.sink.I64(ns.dn);
		sec.sink.I32(ns.s);
		sec.sink.I64(superslm::DynamicScaleReciprocal(ns.dn));
	}
	Rng rng(0x2222333344445555ULL);
	for (int rep = 0; rep < 20000; ++rep) {
		const int64_t d = rng.InRange(1, 2147483648LL);
		const superslm::NormalizedScale ns = superslm::NormalizeScale(d);
		sec.sink.I64(ns.dn);
		sec.sink.I32(ns.s);
		sec.sink.I64(superslm::DynamicScaleReciprocal(ns.dn));
	}

	// C19 at both ends of the normalized domain exactly -- R = 2^32 at Dn = 2^30 is
	// the pinned extreme.
	sec.sink.I64(superslm::DynamicScaleReciprocal(1073741824LL));
	sec.sink.I64(superslm::DynamicScaleReciprocal(2147483647LL));

	// C20 + C22 end to end over token vectors, including a token containing INT32_MIN
	// (the case the header names as forcing R = 2^32 and a ~2^70 intermediate).
	for (int rep = 0; rep < 400; ++rep) {
		const size_t n = static_cast<size_t>(rng.InRange(1, 257));
		std::vector<int32_t> tok(n);
		for (size_t i = 0; i < n; ++i)
			tok[i] = static_cast<int32_t>(rng.InRange(superslm::kInt32Min, superslm::kInt32Max));
		if (rep % 7 == 0) tok[n / 2] = superslm::kInt32Min;
		if (rep % 11 == 0) tok[0] = 0;

		const int64_t dp = superslm::MaxAbsReduce(tok.data(), n);
		const superslm::NormalizedScale ns = superslm::NormalizeScale(dp);
		const int64_t r = superslm::DynamicScaleReciprocal(ns.dn);
		sec.sink.I64(dp);
		sec.sink.I64(ns.dn);
		sec.sink.I32(ns.s);
		sec.sink.I64(r);
		for (size_t i = 0; i < n; ++i) sec.sink.I8(superslm::RequantTokenCode(tok[i], r, ns.s));
	}

	// An all-zero token: the D' = max(D,1) guard.
	{
		std::vector<int32_t> z(64, 0);
		sec.sink.I64(superslm::MaxAbsReduce(z.data(), z.size()));
	}
}

// --- 4. C4/C5/C6 i-sqrt ---------------------------------------------------------
void SectionISqrt() {
	Section& sec = NewSection("c4c6_isqrt");

	static const int64_t kEdge[] = {0,
	                                1,
	                                2,
	                                3,
	                                4,
	                                8,
	                                9,
	                                15,
	                                16,
	                                17,
	                                1023,
	                                1024,
	                                1025,
	                                4294967295LL,
	                                4294967296LL,
	                                4294967297LL,
	                                9223372036854775806LL,
	                                9223372036854775807LL};
	for (int64_t n : kEdge) {
		sec.sink.I64(superslm::ISqrt(n));
		int64_t it[superslm::I_SQRT_ITERATIONS];
		superslm::ISqrtTrace(n, it);
		for (int i = 0; i < superslm::I_SQRT_ITERATIONS; ++i) sec.sink.I64(it[i]);
	}

	Rng rng(0x7777888899990000ULL);
	for (int rep = 0; rep < 8000; ++rep) {
		const int64_t n = static_cast<int64_t>(rng.Next() >> 1);  // [0, 2^63-1]
		sec.sink.I64(superslm::ISqrt(n));
		// Perfect squares and their neighbours, where the digit recurrence's final
		// restoring decision is tightest.
		const int64_t root = superslm::ISqrt(n);
		if (root > 0 && root < 3037000499LL) {
			const int64_t sq = root * root;
			sec.sink.I64(superslm::ISqrt(sq));
			sec.sink.I64(superslm::ISqrt(sq - 1));
			sec.sink.I64(superslm::ISqrt(sq + 1));
		}
	}
}

// --- 5. C7/C8/C9 i-exp ----------------------------------------------------------
void SectionIExp() {
	Section& sec = NewSection("c7c9_iexp");

	// C9 ShiftByMax.
	Rng rng(0x0F1E2D3C4B5A6978ULL);
	for (int rep = 0; rep < 400; ++rep) {
		const size_t n = static_cast<size_t>(rng.InRange(1, 129));
		std::vector<int64_t> logits(n), out(n);
		for (size_t i = 0; i < n; ++i) logits[i] = rng.InRange(-1000000000LL, 1000000000LL);
		superslm::ShiftByMax(logits.data(), n, out.data());
		for (size_t i = 0; i < n; ++i) sec.sink.I64(out[i]);
	}

	// C7/C8. Every triple is gated through IExpConstantsInDomain -- the header states
	// that predicate IS the caller's obligation, and calling the parent out of domain
	// is UB under NDEBUG. An out-of-domain triple is digested as its predicate answer
	// only, so the population still covers the rejection region without invoking it.
	static const int64_t kQln2[] = {1, 2, 3, 255, 65536, 887904998LL, 1000000000LL, 3000000000LL,
	                                4000000000LL};
	static const int64_t kQb[] = {0, 1, 7, 1733160715LL, 2000000000LL, 3037000499LL};
	static const int64_t kQc[] = {0, 1, 65535, 1000000000LL, 4611686018427387904LL,
	                              9223372036854775807LL};

	for (int64_t ln2 : kQln2) {
		for (int64_t qb : kQb) {
			for (int64_t qc : kQc) {
				// q must be <= 0 (caller-ensures). Sweep both ends the header names as
				// the extremes of |base|, plus interior points.
				const int64_t qs[] = {0, -1, -(ln2 / 2), -(ln2 - 1), -ln2, -(ln2 * 7), -(ln2 * 29),
				                      -(ln2 * 31)};
				for (int64_t q : qs) {
					if (q > 0) continue;
					const bool in_domain = superslm::IExpConstantsInDomain(q, ln2, qb, qc);
					sec.sink.Bool(in_domain);
					// S-HARDEN-0 removed the public IExpShift/IExpBase: they formed the
					// decomposition unguarded, which is the UB this pass itself executed
					// and killed (sharden6-popper-ub-2026-07-21.md 2 and 3). The same two
					// values now come from the checked entry point, and the digest is
					// unchanged -- verified by building both sides and comparing.
					//
					// The legacy arm exists so this harness still builds at PINS PREDATING
					// S-HARDEN-0, above all `f078403`, which is the commit the result
					// packet's published hashes name. An instrument that can no longer
					// measure the state it certified turns that packet into exactly the
					// unreproducible scratch measurement 17.3 cell 8 forbids, so the arm
					// is a reproducibility requirement rather than a compatibility
					// courtesy. Build a pre-S-HARDEN-0 pin with
					// -DSH6_LEGACY_IEXP_ACCESSORS=1.
#if defined(SH6_LEGACY_IEXP_ACCESSORS) && SH6_LEGACY_IEXP_ACCESSORS
					sec.sink.I64(superslm::IExpShift(q, ln2));
					sec.sink.I64(superslm::IExpBase(q, ln2, qb));
					if (in_domain) sec.sink.I64(superslm::IExpFromConstants(q, ln2, qb, qc));
#else
					// `con` is digested unconditionally, refused or not. This population DOES
					// contain refused triples (the F21-strike quadruple q=0, q_ln2=887904998,
					// q_b=1733160715, q_c=INT64_MAX among them, per the header's own documented
					// example) -- but a refused construction does NOT digest as (0, 0):
					// IExpConstruct populates `con.z()`/`con.base()` with the formed z/base
					// BEFORE the representability judgement runs (intmath.cpp's own ordering),
					// so `con`'s digested bytes are identical whether or not the construction is
					// later judged representable. That is the actual reason the cross-arm digest
					// identity with the legacy IExpShift/IExpBase arm holds: both arms form and
					// digest the same z/base regardless of refusal, and only the OPTIONAL third
					// value -- IExpFromConstants/IExpEvaluate, gated on `in_domain` above -- is
					// ever skipped for a refused triple, identically in both arms.
					//
					// S-HARDEN-0 also split the value out of the single-call form, so the
					// evaluation moved inside this arm: `IExpEvaluate` takes the construction
					// (which carries its own q_c) rather than the four constants again.
					superslm::IExpConstruction con;
					superslm::IExpConstruct(q, ln2, qb, qc, &con);
					sec.sink.I64(con.z());
					sec.sink.I64(con.base());
					if (in_domain) sec.sink.I64(superslm::IExpEvaluate(con));
#endif
				}
			}
		}
	}

	// A realistic C30-shaped population: q_b / q_ln2 ~ 1.952, which the header says
	// satisfies the one-call shortcut.
	for (int rep = 0; rep < 4000; ++rep) {
		const int64_t ln2 = rng.InRange(1, 100000);
		const int64_t qb = (ln2 * 1952) / 1000;
		const int64_t qc = rng.InRange(0, 1000000000LL);
		const int64_t q = -rng.InRange(0, ln2 * superslm::I_EXP_CLIP_N + 5);
#if defined(SH6_LEGACY_IEXP_ACCESSORS) && SH6_LEGACY_IEXP_ACCESSORS
		if (!superslm::IExpConstantsInDomain(q, ln2, qb, qc)) {
			sec.sink.Bool(false);
			continue;
		}
		sec.sink.Bool(true);
		sec.sink.I64(superslm::IExpFromConstants(q, ln2, qb, qc));
#else
		// One construct call answers the domain question and carries the values, rather than
		// asking the predicate and then re-deriving the same decomposition inside a second
		// call. Digest-identical to the legacy arm: the predicate IS `kOk`.
		superslm::IExpConstruction con;
		if (superslm::IExpConstruct(q, ln2, qb, qc, &con) != superslm::IExpDomain::kOk) {
			sec.sink.Bool(false);
			continue;
		}
		sec.sink.Bool(true);
		sec.sink.I64(superslm::IExpEvaluate(con));
#endif
	}
}

// --- 6. C11/C13 RoPE ------------------------------------------------------------
void SectionRope() {
	Section& sec = NewSection("c11c13_rope");

	static const int32_t kQ30[] = {-superslm::ROPE_ONE, -superslm::ROPE_ONE + 1, -1073741823,
	                               -536870912,          -1,                     0,
	                               1,                   536870912,              1073741823,
	                               superslm::ROPE_ONE};
	static const int32_t kVal[] = {superslm::kInt32Min, -2147483647, -1073741824, -32768, -1, 0,
	                               1,                   32767,       1073741823,  superslm::kInt32Max};

	for (int32_t x : kVal)
		for (int32_t y : kVal)
			for (int32_t c : kQ30)
				for (int32_t s : kQ30) {
					const superslm::RopePair p = superslm::RopeApplyPair(x, y, c, s);
					sec.sink.I64(p.x);
					sec.sink.I64(p.y);
				}

	Rng rng(0xDEADBEEFCAFEF00DULL);
	for (int rep = 0; rep < 20000; ++rep) {
		const int32_t x = static_cast<int32_t>(rng.InRange(superslm::kInt32Min, superslm::kInt32Max));
		const int32_t y = static_cast<int32_t>(rng.InRange(superslm::kInt32Min, superslm::kInt32Max));
		const int32_t c = static_cast<int32_t>(rng.InRange(-superslm::ROPE_ONE, superslm::ROPE_ONE));
		const int32_t s = static_cast<int32_t>(rng.InRange(-superslm::ROPE_ONE, superslm::ROPE_ONE));
		const superslm::RopePair p = superslm::RopeApplyPair(x, y, c, s);
		sec.sink.I64(p.x);
		sec.sink.I64(p.y);
	}
}

// --- 7. C10 SiLU sigmoid LUT ----------------------------------------------------
void SectionSiluLut() {
	Section& sec = NewSection("c10_silu_lut");

	// A pinned integer Q15 table. Built by an exact integer construction (a monotone
	// rational ramp) rather than a float sigmoid, so the table itself carries no
	// floating-point provenance and cannot be the source of a divergence. The lookup
	// under test is the interpolation, not the table's fidelity.
	constexpr int kEntries = superslm::kSiluLutN + 1;
	std::vector<int32_t> table(kEntries);
	for (int i = 0; i < kEntries; ++i) {
		// s(i) = round(32768 * i^3 / (N^3))  -- monotone, integer, exactly reproducible.
		const int64_t n = superslm::kSiluLutN;
		const int64_t num = 32768LL * static_cast<int64_t>(i) * static_cast<int64_t>(i) *
		                    static_cast<int64_t>(i);
		table[i] = static_cast<int32_t>(num / (n * n * n));
	}

	// The calibrated corpus range the impl documents is shift = e+17 in [-19,-15],
	// i.e. e in [-36,-32]. Sweep well past both edges to reach the saturation and the
	// exact-left-shift branches, staying inside the documented domain limits
	// (left branch exact while shift < 24; right branch legal while -shift <= 63).
	Rng rng(0x1234ABCD5678EF90ULL);
	for (int e = -60; e <= 6; ++e) {
		for (int c = -127; c <= 127; ++c) {
			static const int64_t kM[] = {1, 2, 3, 12345, 1000000, 2147483647LL};
			for (int64_t m : kM)
				sec.sink.I32(superslm::SiluSigmoidQ15(table.data(), static_cast<int8_t>(c), m, e));
		}
	}
	for (int rep = 0; rep < 20000; ++rep) {
		const int8_t c = static_cast<int8_t>(rng.InRange(-127, 127));
		const int64_t m = rng.InRange(1, 2147483647LL);
		const int e = static_cast<int>(rng.InRange(-60, 6));
		sec.sink.I32(superslm::SiluSigmoidQ15(table.data(), c, m, e));
	}
}

// --- 8. C17/C19-C22 matmul: the scalar-vs-SIMD A/B ------------------------------
//
// This is the section S-HARDEN-6's "scalar-forced vs SIMD-enabled" leg exists for.
// On x64 the shipping DotRow ALWAYS takes the SSE2 path; DotRowScalarRef is the
// normative scalar reference. Every case asserts they agree AND digests the SIMD
// result, so a divergence shows up twice: as ok=false locally and as a hash mismatch
// against any build where the two agree.
void SectionMatmul() {
	Section& sec = NewSection("c17_matmul");

	Rng rng(0x0BADC0DE0BADC0DEULL);

	// Shapes chosen against the construction's own seams:
	//   1..17           -- every tail residue mod 8, twice over
	//   131064..131080  -- the design's int32-width bound (131,071) either side
	//   131072          -- exactly kFlushBlocks(16384) * 8: the first flush window
	//   262144, 393224  -- two and three flush windows, with a partial third
	static const size_t kN[] = {1,      2,      3,      4,      5,      6,      7,      8,
	                            9,      15,     16,     17,     63,     64,     65,     127,
	                            128,    129,    1000,   1024,   4095,   4096,   4097,   131064,
	                            131071, 131072, 131073, 131080, 262144, 262145, 393224, 400000};

	for (size_t n : kN) {
		// Worst-case magnitude first: all -127 x -127 drives the accumulator to its
		// extreme and is the case a premature int32 narrowing would break.
		std::vector<int8_t> a(n + 16), w(n + 16);
		for (size_t i = 0; i < a.size(); ++i) a[i] = -127;
		for (size_t i = 0; i < w.size(); ++i) w[i] = -127;
		// Unaligned offsets 0..7 -- the movq loads carry no alignment precondition
		// and the design names the unaligned-pointer cell explicitly.
		for (size_t off = 0; off < 8; ++off) {
			const int64_t simd_via_gemm = [&] {
				int64_t acc = 0;
				superslm::GemmInt8AccumulateRow(a.data() + off, w.data() + off, n, 1, &acc);
				return acc;
			}();
			const int64_t scalar = superslm::DotRowScalarRef(a.data() + off, w.data() + off, n);
			if (simd_via_gemm != scalar) {
				sec.ok = false;
				sec.note = "scalar != SIMD on the saturated -127 population";
			}
			sec.sink.I64(simd_via_gemm);
			sec.sink.I64(scalar);
		}

		// Seeded mixed-sign population, where lane regrouping actually changes the
		// order of partial sums.
		for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>(rng.InRange(-127, 127));
		for (size_t i = 0; i < w.size(); ++i) w[i] = static_cast<int8_t>(rng.InRange(-127, 127));
		for (size_t off = 0; off < 8; ++off) {
			int64_t acc = 0;
			superslm::GemmInt8AccumulateRow(a.data() + off, w.data() + off, n, 1, &acc);
			const int64_t scalar = superslm::DotRowScalarRef(a.data() + off, w.data() + off, n);
			if (acc != scalar) {
				sec.ok = false;
				sec.note = "scalar != SIMD on the mixed-sign population";
			}
			sec.sink.I64(acc);
			sec.sink.I64(scalar);
		}
	}

	// Multi-output-channel and multi-token forms, plus the stacking-equivalence
	// property (row t of the batched call equals the single-row call on row t).
	for (int rep = 0; rep < 60; ++rep) {
		const size_t in_c = static_cast<size_t>(rng.InRange(1, 300));
		const size_t out_c = static_cast<size_t>(rng.InRange(1, 40));
		const size_t toks = static_cast<size_t>(rng.InRange(1, 9));
		std::vector<int8_t> act(toks * in_c), wgt(out_c * in_c);
		for (size_t i = 0; i < act.size(); ++i) act[i] = static_cast<int8_t>(rng.InRange(-127, 127));
		for (size_t i = 0; i < wgt.size(); ++i) wgt[i] = static_cast<int8_t>(rng.InRange(-127, 127));

		std::vector<int64_t> batched(toks * out_c, 0);
		superslm::GemmInt8Accumulate(act.data(), wgt.data(), toks, in_c, out_c, batched.data());

		std::vector<int64_t> single(out_c, 0);
		for (size_t t = 0; t < toks; ++t) {
			superslm::GemmInt8AccumulateRow(act.data() + t * in_c, wgt.data(), in_c, out_c,
			                                single.data());
			for (size_t j = 0; j < out_c; ++j) {
				if (single[j] != batched[t * out_c + j]) {
					sec.ok = false;
					sec.note = "stacking equivalence broken";
				}
				const int64_t ref =
				    superslm::DotRowScalarRef(act.data() + t * in_c, wgt.data() + j * in_c, in_c);
				if (ref != single[j]) {
					sec.ok = false;
					sec.note = "scalar reference != gemm row";
				}
				sec.sink.I64(batched[t * out_c + j]);
			}
		}

		// C17 narrowing, fed only values proven to fit int32 (caller-ensures).
		std::vector<int32_t> narrowed(out_c);
		bool fits = true;
		for (size_t j = 0; j < out_c; ++j)
			if (single[j] > 2147483647LL || single[j] < -2147483648LL) fits = false;
		if (fits) {
			superslm::NarrowAccumulatorToI32(single.data(), out_c, narrowed.data());
			for (size_t j = 0; j < out_c; ++j) sec.sink.I32(narrowed[j]);
		}
	}
}

// --- driver ---------------------------------------------------------------------

void PrintBuildIdentity() {
	// Not digested -- identity is what distinguishes the axes, so folding it into the
	// hash would make every axis differ by construction and prove nothing.
	std::printf("# compiler: ");
#if defined(__clang__)
	std::printf("clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#if defined(_MSC_VER)
	std::printf(" (clang-cl, _MSC_VER=%d)", _MSC_VER);
#endif
#elif defined(_MSC_VER)
	std::printf("msvc _MSC_VER=%d", _MSC_VER);
#elif defined(__GNUC__)
	std::printf("gcc %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
	std::printf("unknown");
#endif
	std::printf("\n# cplusplus: %ld\n", static_cast<long>(__cplusplus));
#if defined(NDEBUG)
	std::printf("# ndebug: 1\n");
#else
	std::printf("# ndebug: 0\n");
#endif
#if defined(_M_X64) || defined(__x86_64__)
	std::printf("# arch: x86_64 (matmul SSE2 path active)\n");
#else
	std::printf("# arch: other (matmul scalar fallback active)\n");
#endif
	std::printf("# int64_digits: %d\n", static_cast<int>(sizeof(long long) * 8));
}

}  // namespace

int main() {
	PrintBuildIdentity();

	SectionSha256();
	SectionRequant();
	SectionDynamicScale();
	SectionISqrt();
	SectionIExp();
	SectionRope();
	SectionSiluLut();
	SectionMatmul();

	Sha256 global;
	int failures = 0;
	for (Section* s : Registry()) {
		s->sink.h.Final(s->digest);
		global.Update(s->digest, 32);
		std::printf("%-24s %s  values=%llu%s%s\n", s->name, superslm::ToHex(s->digest).c_str(),
		            static_cast<unsigned long long>(s->sink.count), s->ok ? "" : "  LOCAL-FAIL: ",
		            s->ok ? "" : s->note);
		if (!s->ok) ++failures;
	}

	uint8_t gd[32];
	global.Final(gd);
	std::printf("GLOBAL %s\n", superslm::ToHex(gd).c_str());
	std::printf("local_invariant_failures %d\n", failures);
	return 0;  // A local failure is data, not a build error -- the runner compares hashes.
}
