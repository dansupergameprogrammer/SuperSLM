// array_s2_shared.h -- the POD interface between the S2 array harness's driver
// (array_s2_main.cpp), its CPU side (array_s2_cpu.cpp, the only TU that includes sslm_abi.h) and
// its GPU side (array_s2_gpu.cpp, the only TU that includes gpu_1p0.h). The two public headers
// declare global-scope enums with shared enumerator names, so neither may share a TU with the
// other (the same split tools/t2132_g5_gpu_parity_shared.h uses).
#ifndef ARRAY_S2_SHARED_H
#define ARRAY_S2_SHARED_H

#include <cstdint>
#include <string>
#include <vector>

// A teacher-forced walk or a rejection: the ids are admitted one per schema-content prefill call.
struct WalkResult {
	int32_t admitted = 0;         // calls that admitted their token
	int32_t refused_at = -1;      // index of the first refused token, -1 when none was refused
	std::string refusal;          // the refusing call's status ("unreachable", or "status N")
	int32_t accepting_end = -1;   // the walk's accepting flag after the last admitted token
	std::vector<uint32_t> states; // walk state after each admitted token (GPU only; CPU has no query)
};

enum class GenMode { Plain, JumpCanonical, JumpExact };

// One constrained greedy generation from a prompt, ending at the schema's dead end or max_tokens.
struct GenResult {
	std::vector<int32_t> ids;    // the document's tokens, in output order (forced tokens included)
	std::string end;             // "dead_end", "max_tokens", or "error:<call> <status>"
	int32_t accepting = -1;      // accepting flag where the run ended
	int64_t forced = 0;          // forced_token_count (canonical jump-forward) at the end
	int64_t jump_calls = 0;      // jump-forward calls that admitted at least one token
	double ms = 0.0;             // wall time from the first decode call to the end
};

// Save after ids[after], restore into a fresh sequence, decode to the end.
struct RestoreResult {
	int32_t after = -1;
	size_t blob_bytes = 0;
	std::vector<int32_t> continuation;
	std::string end;
};

struct MapResult {
	bool ok = false;
	std::string error;
	double map_ms = 0.0;
	int64_t vram_delta_bytes = -1;  // GPU only: DXGI local-segment usage after map minus before
	int64_t schema_bytes = 0;       // SchemaMasks section size the model was mapped from
};

// CPU side. `full_finish` (seam builds only) forces the full-vocabulary token finish.
struct CpuSession;
CpuSession* CpuOpen(const uint8_t* bytes, size_t n, const std::string& schema, bool full_finish,
                    int32_t max_chunk, MapResult* map);
void CpuClose(CpuSession* s);
WalkResult CpuWalk(CpuSession* s, const std::vector<int32_t>& ids);
GenResult CpuGen(CpuSession* s, const std::vector<int32_t>& prompt, int32_t max_tokens, GenMode mode);
// Re-runs the plain generation (its tokens must equal `ref`), saving after each index in `after`,
// then restores every blob and decodes each continuation to the end.
std::vector<RestoreResult> CpuRestores(CpuSession* s, const std::vector<int32_t>& prompt,
                                       const std::vector<int32_t>& ref, const std::vector<int32_t>& after,
                                       int32_t max_tokens, std::string* error);
bool CpuSeamAvailable();

// GPU side (adapter 0).
struct GpuSession;
GpuSession* GpuOpen(const uint8_t* bytes, size_t n, const std::string& schema, MapResult* map);
void GpuClose(GpuSession* s);
WalkResult GpuWalk(GpuSession* s, const std::vector<int32_t>& ids);
GenResult GpuGen(GpuSession* s, const std::vector<int32_t>& prompt, int32_t max_tokens);
std::vector<RestoreResult> GpuRestores(GpuSession* s, const std::vector<int32_t>& prompt,
                                       const std::vector<int32_t>& ref, const std::vector<int32_t>& after,
                                       int32_t max_tokens, std::string* error);

#endif  // ARRAY_S2_SHARED_H
