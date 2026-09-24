// TE-425 -- the one translation unit that includes the engine's internal harness header, as the
// SuperSLM-Unreal plugin does (SuperSLMGpuSubsystem.cpp: ProbeUsable, QueryLayer1LocalVideoMemoryBytes,
// the configure job), to call harness::GetDevice() directly (plan Sec3.2 E-7, Sec3.5 R1: "a first touch
// that is not an entry point").
//
// SUPERSLM_GPU_ALLOC_FAULT_INJECTION is defined before the include because the library under test is
// the seam build: the header's inline members (Device::TryMakeBuffer among them) must be the same
// definition in this translation unit as in the library's, or the linker's choice between the two
// COMDAT copies would silently decide whether the seam exists.
#define SUPERSLM_GPU_ALLOC_FAULT_INJECTION 1
#include "d3d12_harness.h"

bool Te425DirectGetDeviceFirstTouch(bool* threw, bool* available) {
	*threw = false;
	*available = true;
	try {
		superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
		const bool a = dev.available;  // a plain bool at v1.7.1, std::atomic<bool> at the fix (E-7)
		*available = a;
	} catch (...) {
		*threw = true;
	}
	return true;
}
