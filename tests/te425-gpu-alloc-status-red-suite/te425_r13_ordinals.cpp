// TE-425 R13 (plan Sec3.5 R13, TE-423 F-12): every existing SslmForwardStatus enumerator keeps its
// ordinal, and GpuOperationFailed (plan Sec3.2 E-3) is APPENDED after ParallelForIncomplete, the last
// enumerator at v1.7.1 -- never inserted. tests/t2791 and tests/t2956 cell_status_ordinals.cpp pin
// only SslmGpuStatus; nothing pinned the forward enum.
//
// How it fails, and when:
//   - An enumerator inserted anywhere above ParallelForIncomplete moves an existing ordinal: one of the
//     37 static_asserts below stops the build (the mutant "inserted rather than appended").
//   - GpuOperationFailed present at any ordinal other than 37: the static_assert on OpFailedOrdinal
//     stops the build.
//   - GpuOperationFailed absent (v1.7.1): the program builds, and exits 1 -- the cell's red reading.
//     The enumerator's existence is read through a requires-expression, so the cell compiles at both
//     versions and is red at v1.7.1 for its own reason, not by a compile error.
//   - SslmForwardStatusName has no arm for it (TE-422 W-1: it would return "?"): exits 1.
#include <cstdio>
#include <cstring>

#include "superslm/checked_chain_funnel.h"

using S = superslm::SslmForwardStatus;

#define PIN(name, v) \
	static_assert(static_cast<int>(S::name) == (v), "SslmForwardStatus::" #name " moved from " #v)
PIN(Ok, 0);
PIN(ChainInputOutOfDomain, 1);
PIN(LogitNarrowingOverflow, 2);
PIN(IExpConstantsOutOfDomain, 3);
PIN(CarriedScaleMantissaOutOfDomain, 4);
PIN(SiluCompositionScaleOutOfDomain, 5);
PIN(RoundingDivideByPotExponentOutOfDomain, 6);
PIN(SoftmaxRowWidthOutOfDomain, 7);
PIN(TokenIdOutOfRange, 8);
PIN(PositionOverCap, 9);
PIN(WorkspaceTooSmall, 10);
PIN(KvCapacityExhausted, 11);
PIN(KvPrecisionUnsupported, 12);
PIN(InvalidLayerBudget, 13);
PIN(RopeTableTensorMissing, 14);
PIN(RopeTableExtentExceeded, 15);
PIN(InvalidContextCap, 16);
PIN(HeadDimGeometryMismatch, 17);
PIN(KvHeadGeometryMismatch, 18);
PIN(SequenceAlreadyComplete, 19);
PIN(SoftmaxKernelRefusedAfterGateAccepted, 20);
PIN(ResidualReconciliationMagnitudeOutOfDomain, 21);
PIN(ResidualReconciliationScaleOutOfDomain, 22);
PIN(InvalidHiddenCodes, 23);
PIN(IExpScaleDerivationOutOfDomain, 24);
PIN(BiasReconcileProductOutOfDomain, 25);
PIN(OptionGWideRopeMagnitudeOutOfDomain, 26);
PIN(OptionGFusedLandingExponentOutOfDomain, 27);
PIN(QkNormFusedLandingMagnitudeOutOfDomain, 28);
PIN(GpuAllocationFailed, 29);
PIN(GpuDeviceRemoved, 30);
PIN(GpuGemmGroupArithmeticInvalid, 31);
PIN(GpuLayerWeightsContractViolation, 32);
PIN(InvalidDecodeParams, 33);
PIN(OutputCapacityExceeded, 34);
PIN(GpuShaderBinaryStale, 35);
PIN(ParallelForIncomplete, 36);
#undef PIN

constexpr int kLastAtV171 = 36;  // ParallelForIncomplete

template <class E>
constexpr int OpFailedOrdinal() {
	if constexpr (requires { E::GpuOperationFailed; }) {
		return static_cast<int>(E::GpuOperationFailed);
	} else {
		return -1;
	}
}
static_assert(OpFailedOrdinal<S>() == -1 || OpFailedOrdinal<S>() == kLastAtV171 + 1,
              "SslmForwardStatus::GpuOperationFailed must be appended after ParallelForIncomplete (ordinal 37)");

int main() {
	const int op = OpFailedOrdinal<S>();
	if (op < 0) {
		std::printf("R13 SslmForwardStatus: 37 existing ordinals pinned; GpuOperationFailed absent -> FAIL\n");
		std::printf("VERDICT R13 RED\n");
		return 1;
	}
	const char* name = superslm::SslmForwardStatusName(static_cast<S>(op));
	const bool named = name && std::strcmp(name, "GpuOperationFailed") == 0;
	std::printf("R13 SslmForwardStatus: 37 existing ordinals pinned; GpuOperationFailed=%d (appended); "
	            "SslmForwardStatusName=\"%s\" -> %s\n",
	            op, name ? name : "(null)", named ? "PASS" : "FAIL");
	std::printf("VERDICT R13 %s\n", named ? "GREEN" : "RED");
	return named ? 0 : 1;
}
