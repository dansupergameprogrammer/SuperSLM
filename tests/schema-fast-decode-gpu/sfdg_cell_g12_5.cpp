// Schema fast-decode plan rev 1.4.4 §11.2, cell G12.5 (slice G1-F): an injected allocation failure in the GPU
// table build fails the GPU map with SSLM_GPU_ALLOCATION_FAILED and no handle. The seam is the table-build
// bad-alloc (§11.0): the existing MaybeThrowInjectedForcedTableBuildAllocFault, reached from the GPU map
// through the lifted builder (tests/support/bad_alloc_injection.h, single-shot, this thread). The next map,
// unarmed, succeeds, and the CPU map is unaffected. G2.1 carries the table-less arm.
//
// Needs the seam build (SUPERSLM_ENABLE_BAD_ALLOC_INJECTION). Box only. Run: sfdg_cell_g12_5.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

#include "support/bad_alloc_injection.h"

using namespace sfdg;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, false)) {
		superslm_test::g_inject_forced_table_build_alloc_failure = true;
		SslmGpuModelHandle* m = OutSentinel<SslmGpuModelHandle>();
		const GS ms = sslm_gpu_model_map(rig.ctx(), &rig.env.fx.view, GpuResidencyConfig{}, &m);
		const bool fired = !superslm_test::g_inject_forced_table_build_alloc_failure;
		superslm_test::g_inject_forced_table_build_alloc_failure = false;
		CHECK_MSG(fired, "[G12.5] the GPU map never reached the table build's allocation seam");
		CHECK_MSG(ms == GS::SSLM_GPU_ALLOCATION_FAILED && m == nullptr,
		          "[G12.5] an allocation failure in the table build: %s, handle %p (want SSLM_GPU_ALLOCATION_FAILED, null)",
		          Name(ms), static_cast<void*>(m));
		if (ms == GS::SSLM_OK && Real(m)) sslm_gpu_model_unmap(rig.ctx(), m);
		// Unarmed, the map succeeds and carries the table.
		SslmGpuModelHandle* m2 = nullptr;
		const GS ms2 = sslm_gpu_model_map(rig.ctx(), &rig.env.fx.view, GpuResidencyConfig{}, &m2);
		CHECK_MSG(ms2 == GS::SSLM_OK && m2, "[G12.5] the unarmed map: %s", Name(ms2));
		if (m2) {
			int32_t nf = -9;
			uint32_t fl = 0;
			const GS es = sslm_gpu_test_only_forced_table_entry(m2, 0, 0, &nf, &fl);
			CHECK_MSG(es == GS::SSLM_OK, "[G12.5] the unarmed map has a table: %s", Name(es));
			sslm_gpu_model_unmap(rig.ctx(), m2);
		}
		// The CPU map, unarmed, is unaffected.
		CRig c(rig.raw);
		CHECK_MSG(c.ok, "[G12.5] the CPU map after the GPU's injected failure");
	}
	return Finish("sfdg_cell_g12_5");
}
