// Precached-prefixes plan §7 cells 2.13 and 11.6, their GPU legs [F2]: the schema parser's two refusals
// (a duplicate and a zero name hash) fire at the GPU model map. The GPU map parses the schema section through the CPU's parser (SchemaMasksTable::Parse,
// src/gpu/gpu_1p0.cpp's model map), and a section the parser refuses is a malformed artifact there, which the
// GPU map reports as SSLM_DEVICE_LOST with no handle (its convention for every marshal failure). The parser's
// reason (DuplicateSchemaNameHash, ZeroSchemaNameHash) is graded by the CPU cell (tests/precache/e2_schema.cpp);
// this leg grades that the GPU map refuses the same bytes and maps the renamed pair.
//
// Inputs: the E0 debunker's independently written artifacts (must-reject-two-name.sslm,
// must-reject-zero-hash.sslm, must-accept-renamed-pair.sslm) in $SUPERSLM_PRECACHE_E0_ARTIFACT_DIR, the same
// bytes the CPU leg reads. They are box inputs: a missing directory is a failed check, never a skip.
//
// RED AT F1: no product change is needed for this leg (the parser rule landed at E2 and the GPU map already parses
// through it), so it is green as soon as its inputs exist; it is in F1's suite so F2's gate runs it.
//
// Run: pcg_cell_schema.exe (with SUPERSLM_PRECACHE_E0_ARTIFACT_DIR set).
#include "pcg_helpers.h"

using namespace pcg;

int main() {
	const char* dir = std::getenv("SUPERSLM_PRECACHE_E0_ARTIFACT_DIR");
	CHECK_MSG(dir && *dir, "SUPERSLM_PRECACHE_E0_ARTIFACT_DIR is unset: the E0 artifacts are a box input");
	if (dir && *dir) {
		SslmGpuContext* ctx = nullptr;
		const GS cs = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
		CHECK_MSG(cs == GS::SSLM_OK && ctx, "sslm_gpu_context_create: %s", StatusName(cs));
		if (ctx) {
			struct {
				const char* file;
				bool accept;
			} cases[] = {{"must-reject-two-name.sslm", false}, {"must-reject-zero-hash.sslm", false},
			             {"must-accept-renamed-pair.sslm", true}};
			for (const auto& c : cases) {
				std::vector<uint8_t> bytes;
				const std::string path = std::string(dir) + "/" + c.file;
				if (!ReadFileBytes(path, &bytes)) {
					CHECK_MSG(false, "2.13 GPU: cannot read %s", path.c_str());
					continue;
				}
				superslm::SslmModelView view{};
				std::string err;
				const bool loaded = superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) ==
				                    superslm::SslmModelStatus::Ok;
				CHECK_MSG(loaded, "2.13 GPU: %s does not load as a model view (%s); the leg needs the view", c.file, err.c_str());
				if (!loaded) continue;
				SslmGpuModelHandle* m = OutSentinel<SslmGpuModelHandle>();
				const GS st = sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &m);
				if (c.accept) {
					CHECK_MSG(st == GS::SSLM_OK && Real(m), "2.13 GPU: %s must map: %s", c.file, StatusName(st));
				} else {
					CHECK_MSG(st != GS::SSLM_OK && m == nullptr, "2.13/11.6 GPU: %s mapped on the GPU (%s): the parser rule does not hold there",
					          c.file, StatusName(st));
					std::printf("2.13 GPU: %s refused at map with %s\n", c.file, StatusName(st));
				}
				if (st == GS::SSLM_OK && Real(m)) sslm_gpu_model_unmap(ctx, m);
			}
			sslm_gpu_context_destroy(ctx);
		}
	}
	return FinishSuite("pcg_cell_schema");
}
