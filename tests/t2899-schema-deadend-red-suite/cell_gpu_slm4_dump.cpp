// T-2900 (Curie) -- plan Sec3.10.1's legacy backward-compatibility promise: an 'SLM4' blob (no
// schema tail) produced by the CURRENTLY-SHIPPED (pre-T-2895) `sslm_gpu_seq_save` must still
// restore correctly under T-2895's own v5 restore code. Two separate PROCESSES, mirroring how a
// real caller upgrading in place would encounter this: THIS program links against
// TE-338/T-2866's own already-fixed, UNMODIFIED (pre-T-2895) GPU objects -- the checked-return +
// `ready_for_logits` re-arm fix ALONE, no SLM5 blob format (`build_red_suite_gpu.bat`'s own
// `gpu_fixed_noslm5` step, T-2903 fix; `D:\_te338\gpu_1p0_fixed.cpp` paired with the engine's own
// pristine `superslm_gpu.cpp`) -- and writes a genuine 'SLM4' blob to disk; a plain AS_BUILT link
// cannot serve this role, because the fully unfixed finish bridge never returns `-2` at all, so
// the pre-save dead-end assertion below would be unwinnable. `cell_gpu_slm4_restore.cpp`, a
// separate process linked against the v5-patched (SLM5) objects, reads the blob back.
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED, from `Claude/Vitruvius/t2895-probe/dump_slm4_blob.cpp` --
// already executed and cited by the plan (Sec3.10.1) as the authored-cell source. That probe's
// own header names its linkage identically: "TE-338/T-2866's already-fixed, UNMODIFIED GPU
// objects (fixed/gpu_1p0_fixed.obj, fixed/superslm_gpu.obj) -- this program never sees T-2895's
// own v5 patch."
//
// T-2903 fix: the ADOPTED probe's own construction (prompt prefill, bind schema 0, ONE decode
// call feeding an arbitrary token) is one step short of reaching a genuine dead end. The
// probe's own printf-verdict form never asserted its decode_out value, so the gap was silent;
// this cell's own CHECK on it (T-2900's own conversion to CHECK-asserted cells) is what surfaces
// it. `g5_minimal_one_field` compiles to 2 states and 1 transition (this file's own sibling,
// `cell_gpu_cell2_degenerate.cpp`'s header, citing `make_g5an_fixture.py`): state 0's own row
// admits exactly `t0`, so a decode call issued from state 0 with no schema-content progress
// CONSUMES that one legal transition (state 0 -> state 1, a real produced token, not a dead
// end) rather than dead-ending -- exactly what this cell's own AS_BUILT-labeled run showed
// (`out=0`, a real token, on the object that already carries T-2866's own checked-return fix).
// The fix mirrors Cell 1's own Route R construction and `cell_gpu_slm5_saverestore.cpp`'s own P2
// save point exactly: a schema-content prefill of `{t0}` (the one legal token) drives the walk
// to state 1 -- G-7a's own accepting state, the only state a real int32 row can dead-end at
// (plan Sec3.10.2's own Cell 1 citation) -- BEFORE the decode call that follows.
//
// Run: cell_gpu_slm4_dump.exe --model=PATH --out=BLOBPATH
#include "fixture_common.h"
#include <fstream>

int main(int argc, char** argv) {
	std::string model_path, out_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--out=", 0) == 0) out_path = a.substr(6);
	}
	if (model_path.empty() || out_path.empty()) {
		std::printf("SKIP cell_gpu_slm4_dump -- needs --model=PATH --out=BLOBPATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	GpuModelFixture fx;
	CHECK_MSG(fx.Open(model_path, ctx), "GPU open failed");
	if (GFailures > 0) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}

	// Route R dead end, matching cell_gpu_slm5_saverestore.cpp's own P2 save point construction:
	// prompt prefill, bind schema 0, a schema-content prefill of {t0} (the one legal transition
	// out of state 0) to reach the accepting state, THEN the decode call that dead-ends there.
	IndependentSchema ds;
	CHECK(ds.Build(fx.bytes, "g5_minimal_one_field"));
	CHECK_MSG(ds.entry != nullptr, "independent parse of schema 'g5_minimal_one_field' failed");
	if (GFailures > 0) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}
	const int32_t t0 = ds.FirstLegal(0);
	CHECK_MSG(t0 >= 0, "SETUP: no schema-admitted token found at state 0");

	SslmGpuSequenceHandle* s = nullptr;
	sslm_gpu_seq_create(ctx, fx.model, fx.model_cap, &s);
	const std::vector<int32_t> P = fx.TokensP();
	Prefill(fx, s, P);
	SslmGpuSeqSetSchemaForG5Bridge(ctx, s, 0);
	int32_t consumed = -1;
	CHECK_MSG(SslmGpuSeqPrefillSchemaContentForG5Bridge(ctx, s, &t0, 1, fx.one_layer_budget, &consumed) == SSLM_OK &&
	              consumed == 1,
	          "schema-content prefill {t0} failed");
	int32_t out = -1;
	SslmGpuSeqDecodeStepForG5Bridge(ctx, s, fx.OtherToken(), fx.one_layer_budget, &out);
	CHECK_MSG(out == -2, "pre-save decode should dead-end: out=%d, want -2", out);

	size_t need = 0;
	sslm_gpu_seq_save(ctx, s, nullptr, &need);
	std::vector<uint8_t> blob(need);
	size_t n = need;
	const SslmGpuStatus sv = sslm_gpu_seq_save(ctx, s, blob.data(), &n);
	CHECK_MSG(sv == SSLM_OK, "save returned %s", StatusName(sv));
	CHECK_MSG(n >= 4 && blob[0] == 'S' && blob[1] == 'L' && blob[2] == 'M' && blob[3] == '4',
	          "expected an 'SLM4' magic from the UNMODIFIED (pre-T-2895) save path, got %02x%02x%02x%02x",
	          n >= 1 ? blob[0] : 0, n >= 2 ? blob[1] : 0, n >= 3 ? blob[2] : 0, n >= 4 ? blob[3] : 0);
	std::printf("save=%s blob_bytes=%zu magic=%c%c%c%c\n", StatusName(sv), n, blob[0], blob[1], blob[2], blob[3]);

	std::ofstream out_f(out_path, std::ios::binary);
	out_f.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(n));
	out_f.close();
	CHECK_MSG(static_cast<bool>(out_f), "writing the blob file failed: %s", out_path.c_str());

	sslm_gpu_seq_release(ctx, s);
	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
