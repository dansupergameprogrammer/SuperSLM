// Paged-KV plan (rev 16.2), G1, group g1c: the debug-layer message count of plan §7's grading rule
// ("the debug-layer message count, read by the test from the device's info queue"), for 3.6 and
// 3.6d's scope control. The one translation unit of this group that includes the engine's
// d3d12_harness.h, as tests/t2178-gpu-batched-prefill-red-suite/cell_trust_and_guard.cpp does.
//
// The count is the harness's DrainDebugLayerMessages(): WARNING-or-worse messages since the last
// drain. D3D12 devices are one per adapter in a process, so the singleton's info queue also carries
// what the per-context devices' work raised (the pool create's zero-fill included).
//
// A build that compiles the library with SUPERSLM_GPU_ALLOC_FAULT_INJECTION must compile this file
// with it too (te425_getdevice.cpp's reason: the header's inline members must be the library's).
//
// KNOWN LIMIT, carried as-is: T-2195 found the layer blind to a desynced resume-barrier latch on the
// RTX 2080 SUPER (driver 560.94), so on that device a 0 is "no message", not "no missing barrier";
// 3.6's byte-equal tokens are its second operand for the latch (see g1c_cell_3_6.cpp).
#include "d3d12_harness.h"

#include "g1c_helpers.h"
#include "pkvg_runner.h"

// A state-tracking build (SUPERSLM_GPU_STATE_TRACKING, build_paged_kv_gpu_mingw.sh --state-tracking) also
// counts the host-side resource-state tracker's violations since the last call
// (src/gpu/gpu_resource_state_tracker.h), and reports the count enabled without the layer: on the software
// GPU, which has no D3D12 validation, the tracker stands in for the layer's resource-state class (barrier
// StateBefore, copy and dispatch states) and for nothing else.
size_t g1c::DebugLayerReportable(bool* enabled) {
	superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
	static uint64_t tracked_seen = 0;
	const uint64_t tracked_total = superslm_gpu::harness::StateTrackingViolations().Total();
	const size_t tracked = static_cast<size_t>(tracked_total - tracked_seen);
	tracked_seen = tracked_total;
	*enabled = (dev.debug_layer_enabled && dev.debug_info_queue) || superslm_gpu::harness::kStateTrackingBuild;
	return dev.DrainDebugLayerMessages() + tracked;
}

// pkvg_main.cpp's cell-boundary drain (pkvg_runner.h). It never sets the device up: a cell that left it
// unset (or failed its setup on purpose, 3.6d's first touch) is reported as nothing, and its next GetDevice()
// is still the cell's own.
namespace {
size_t CellBoundaryDrain(bool* enabled) {
	*enabled = false;
	if (superslm_gpu::harness::SingletonLedgerIfConstructed() == nullptr) return 0;
	return g1c::DebugLayerReportable(enabled);
}
[[maybe_unused]] const bool kCellBoundaryDrainRegistered = (pkvg::CellBoundaryDrain() = &CellBoundaryDrain, true);
}  // namespace

#if defined(SUPERSLM_GPU_STATE_TRACKING)
// ---------------------------------------------------------------------------------------------
// tracker/G3:lifetime (tracking builds only): the commissioning of the tracker's lifetime classes
// (released_use, released_recorded, released_in_flight; gpu_resource_state_tracker.h), each by a
// must-reject construction beside its must-accept twin. GPU-based validation resolves a root
// descriptor or a copy by GPU virtual address, so a buffer released while recorded or in-flight work
// can still reach it makes the layer charge that work to whatever buffer the address holds next; the
// host state classes cannot see that, so these classes are the cloud's only view of it.
//
// The constructions drive the tracker's hooks directly with real buffers (their release is what the
// lifetime classes observe) and fake command-list keys, so nothing is recorded or executed on the GPU:
// a list key is only a map key to the tracker and is never dereferenced. Each must-reject construction
// raises exactly one violation of its class; the twins raise none. The cell then restores the counts,
// so the run's summary keeps counting only unplanned violations.
// ---------------------------------------------------------------------------------------------
namespace {
using superslm_gpu::harness::StateTrackingCounts;
namespace st = superslm_gpu::harness::state_tracking;

ID3D12GraphicsCommandList* FakeList(uintptr_t k) { return reinterpret_cast<ID3D12GraphicsCommandList*>(k * 0x40); }

void TrackerLifetime() {
	superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
	G1C_CHECK(dev.available.load(), "SETUP: the process's D3D12 device is not available");
	if (!dev.available.load()) return;
	using Microsoft::WRL::ComPtr;
	auto dst = [&]() {
		return dev.MakeBuffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
	};
	auto src = [&]() {
		return dev.MakeBuffer(256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
	};
	const std::source_location here = std::source_location::current();
	const StateTrackingCounts saved = superslm_gpu::harness::StateTrackingViolations();
	std::printf("  note: the next GPU STATE VIOLATION lines (released_recorded, released_in_flight x2, released_use) "
	            "are this cell's must-reject constructions\n");
	std::fflush(stdout);
	auto delta = [&](uint64_t StateTrackingCounts::*cls) {
		return superslm_gpu::harness::StateTrackingViolations().*cls - saved.*cls;
	};

	// released_recorded. Twin: the recording is reset (discarded) instead of executed.
	{
		ComPtr<ID3D12Resource> a = dst(), b = src(), c = dst(), d = src();
		st::OnCopy(FakeList(1), c.Get(), d.Get(), here);
		c.Reset();
		st::OnReset(FakeList(1));
		G1C_CHECK(delta(&StateTrackingCounts::released_recorded) == 0,
		          "must-accept: a buffer released under a recording that is reset, not executed, was counted");
		st::OnCopy(FakeList(2), a.Get(), b.Get(), here);
		a.Reset();
		st::OnExecute(FakeList(2));
		st::OnSignal(FakeList(2), 1);
		st::OnCompleted(FakeList(2), 1);
		G1C_CHECK(delta(&StateTrackingCounts::released_recorded) == 1,
		          "must-reject: a list executed after a buffer it references was released: released_recorded moved "
		          "by %llu, want 1", static_cast<unsigned long long>(delta(&StateTrackingCounts::released_recorded)));
	}
	// released_in_flight: executed and signalled, not complete; executed, not signalled. Twin: the
	// device saw the submission's fence value complete before the release.
	{
		ComPtr<ID3D12Resource> e = dst(), f = src(), g = dst(), h = src(), i = dst(), j = src();
		st::OnCopy(FakeList(3), g.Get(), h.Get(), here);
		st::OnExecute(FakeList(3));
		st::OnSignal(FakeList(3), 9);
		st::OnCompleted(FakeList(3), 9);
		g.Reset();
		G1C_CHECK(delta(&StateTrackingCounts::released_in_flight) == 0,
		          "must-accept: a buffer released after its submission completed was counted");
		st::OnCopy(FakeList(4), e.Get(), f.Get(), here);
		st::OnExecute(FakeList(4));
		st::OnSignal(FakeList(4), 7);
		e.Reset();
		st::OnCopy(FakeList(5), i.Get(), j.Get(), here);
		st::OnExecute(FakeList(5));
		i.Reset();
		G1C_CHECK(delta(&StateTrackingCounts::released_in_flight) == 2,
		          "must-reject: two buffers released while their submissions could still run: released_in_flight "
		          "moved by %llu, want 2", static_cast<unsigned long long>(delta(&StateTrackingCounts::released_in_flight)));
		st::OnCompleted(FakeList(4), 7);
		st::OnSignal(FakeList(5), 1);
		st::OnCompleted(FakeList(5), 1);
	}
	// released_use. Twin: the same command on a live buffer.
	{
		ComPtr<ID3D12Resource> k = dst(), m = src(), n = dst();
		st::OnCopy(FakeList(6), n.Get(), m.Get(), here);
		st::OnReset(FakeList(6));
		G1C_CHECK(delta(&StateTrackingCounts::released_use) == 0, "must-accept: a command on a live buffer was counted");
		ID3D12Resource* const stale = k.Get();
		k.Reset();  // no buffer is created before the command below, so the address is not reused
		st::OnCopy(FakeList(6), stale, m.Get(), here);
		st::OnReset(FakeList(6));
		G1C_CHECK(delta(&StateTrackingCounts::released_use) == 1,
		          "must-reject: a command recorded on a released buffer: released_use moved by %llu, want 1",
		          static_cast<unsigned long long>(delta(&StateTrackingCounts::released_use)));
	}
	// The default debug name (Device::TryMakeBuffer, debug layer on) is exercised here too, since the
	// software GPU has no debug layer to take that path: the name is set, and nothing else changes.
	{
		ComPtr<ID3D12Resource> named = dst();
		superslm_gpu::harness::NameBufferBySite(named.Get(), here, D3D12_HEAP_TYPE_DEFAULT);
	}
	const StateTrackingCounts now = superslm_gpu::harness::StateTrackingViolations();
	G1C_CHECK(now.barrier_before == saved.barrier_before && now.copy_source == saved.copy_source &&
	              now.copy_dest == saved.copy_dest && now.dispatch_srv == saved.dispatch_srv &&
	              now.dispatch_uav == saved.dispatch_uav,
	          "the constructions moved a state class: they should raise lifetime violations only");
	superslm_gpu::harness::RestoreStateTrackingViolations(saved);
}
}  // namespace

G1C_CELL("tracker/G3:lifetime", "G3", TrackerLifetime);
#endif  // SUPERSLM_GPU_STATE_TRACKING
