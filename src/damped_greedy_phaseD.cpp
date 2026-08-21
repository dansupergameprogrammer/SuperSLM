// damped_greedy_phaseD.cpp -- T-2199 Phase D1 (artifact-carried scale constants) and the
// D2/D3-shared params validation + D2's thin ABI wrapper. Design of record:
// Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md Sec8 Phase D, Sec9 dim2/dim9. Every certified
// primitive this file calls (SslmArtifact, IExpScaleConstants) is reused, never reimplemented.
#include "superslm/sslm_phaseD.h"

#include <cstring>

#include "superslm/intmath.h"

namespace superslm_test_phaseD {

namespace {

// DGC1's own fixed 12-byte layout: int64 scale_mantissa_m (little-endian), int32
// scale_exponent_e (little-endian) -- matching this codebase's own "a converter never emits
// host-endian bytes" convention (sslm_fixtures.h). Kept file-local: no other translation unit
// needs the raw byte shape, only the parsed struct.
constexpr size_t kSectionBytes = 12;

void PutI64LE(std::vector<uint8_t>& b, size_t off, int64_t v) {
	const uint64_t u = static_cast<uint64_t>(v);
	for (int i = 0; i < 8; ++i) b[off + static_cast<size_t>(i)] = static_cast<uint8_t>((u >> (8 * i)) & 0xFF);
}
void PutI32LE(std::vector<uint8_t>& b, size_t off, int32_t v) {
	const uint32_t u = static_cast<uint32_t>(v);
	for (int i = 0; i < 4; ++i) b[off + static_cast<size_t>(i)] = static_cast<uint8_t>((u >> (8 * i)) & 0xFF);
}
int64_t GetI64LE(const uint8_t* p) {
	uint64_t u = 0;
	for (int i = 0; i < 8; ++i) u |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(u);
}
int32_t GetI32LE(const uint8_t* p) {
	uint32_t u = 0;
	for (int i = 0; i < 4; ++i) u |= static_cast<uint32_t>(p[i]) << (8 * i);
	return static_cast<int32_t>(u);
}

// Plan Sec2.3's own real constraint, applied here: a derived (q_b, q_c) is only usable if
// M = q_b^2 + q_c stays within [0, kSoftmaxRowMaxSafeExponent] AND width <= INT64_MAX / M
// (when M != 0) at kRealVocabSizeForDomainCheck. Reuses the certified IExpScaleConstants to
// derive (q_ln2, q_b, q_c) from (m, e) exactly the way Phase B3's runtime derivation would --
// never a second, hand-rolled derivation.
bool ScaleConstantsInDomain(int64_t m, int32_t e) {
	int64_t q_ln2 = 0, q_b = 0, q_c = 0;
	const auto domain = superslm::IExpScaleConstants(m, e, superslm::kIExpLn2Q, 30,
	                                                  superslm::kIExpBQ, 30, superslm::kIExpCaQ, 30,
	                                                  &q_ln2, &q_b, &q_c);
	if (domain != superslm::IExpScaleDomain::kOk) return false;
	const auto width_status = superslm::CheckSoftmaxRowWidthDomain(
	    q_b, q_c, static_cast<size_t>(kRealVocabSizeForDomainCheck));
	return width_status == superslm::SslmForwardStatus::Ok;
}

}  // namespace

RawSection MakeDampedGreedyConstantsSection(const DampedGreedyScaleConstants& constants) {
	RawSection s;
	s.type = static_cast<uint32_t>(superslm::SslmSectionType::DampedGreedyConstants);
	s.dtype = static_cast<uint32_t>(superslm::SslmDtype::Raw);
	s.alignment = 8;
	s.data.assign(kSectionBytes, 0);
	PutI64LE(s.data, 0, constants.scale_mantissa_m);
	PutI32LE(s.data, 8, constants.scale_exponent_e);
	return s;
}

bool ArtifactHasDampedGreedyConstants(const superslm::SslmArtifact& art) noexcept {
	if (!art.DampedGreedyConstantsFlagSet()) return false;
	const superslm::SslmSectionView* sec =
	    art.Section(superslm::SslmSectionType::DampedGreedyConstants);
	if (!sec) return false;
	if (sec->dtype != superslm::SslmDtype::Raw) return false;
	if (sec->byte_size != kSectionBytes) return false;
	return true;
}

bool ReadDampedGreedyScaleConstants(const superslm::SslmArtifact& art,
                                     DampedGreedyScaleConstants* out) noexcept {
	if (!ArtifactHasDampedGreedyConstants(art)) return false;
	const superslm::SslmSectionView* sec =
	    art.Section(superslm::SslmSectionType::DampedGreedyConstants);
	const int64_t m = GetI64LE(sec->data);
	const int32_t e = GetI32LE(sec->data + 8);
	// Sec9 dim2's own hostile-input discipline, applied to this new field: a malformed or
	// out-of-domain (m, e) is a defined rejection, never a silent hand-back of a pair Phase B3's
	// own IExpScaleConstants derivation would misbehave on downstream.
	if (!ScaleConstantsInDomain(m, e)) return false;
	out->scale_mantissa_m = m;
	out->scale_exponent_e = e;
	return true;
}

// --- D2/D3 shared params validation (plan Sec9 dim2) -------------------------------------------
bool ValidateDampedGreedyParams(const DampedGreedyValidationParams& p,
                                 int32_t vocab_size) noexcept {
	if (p.mode != DampedGreedyMode::kDampedGreedy) return true;  // greedy: nothing to validate
	if (p.alpha_q15 < 0) return false;
	if (p.anti_lm_max_order < 1) return false;
	if (p.top_k < 1 || p.top_k > vocab_size) return false;
	return true;
}

// --- D2: thin wrapper (plan Sec8 D1's own "additive-field build" shape, sslm_phaseD_stub.h) --
// Copies the extra fields into the (now mode-aware) sslm_decode_params and calls straight
// through to the ALREADY-PUBLIC sslm_decode_step -- no new internal entry point, matching the
// ruling's own "additive field, no successor struct" shape exactly. sslm_decode_stepImpl's own
// mode-aware branch (src/sslm_abi.cpp) is what actually wires DampedGreedyScoreAndArgmax in.
extern "C" sslm_status sslm_decode_step_damped_greedy(sslm_model model, sslm_seq* seqs, int32_t n,
                                                        const sslm_decode_params_damped_greedy* params,
                                                        sslm_workspace ws, int32_t* out_tokens) {
	if (!params) return SSLM_INVALID_ARGUMENT;
	sslm_decode_params plain{};
	plain.layer_budget = params->layer_budget;
	plain.mode = static_cast<int32_t>(params->mode);
	plain.alpha_q15 = params->alpha_q15;
	plain.anti_lm_max_order = params->anti_lm_max_order;
	plain.top_k = params->top_k;
	plain.q_ln2 = params->q_ln2;
	plain.q_b = params->q_b;
	plain.q_c = params->q_c;
	return sslm_decode_step(model, seqs, n, &plain, ws, out_tokens);
}

}  // namespace superslm_test_phaseD
