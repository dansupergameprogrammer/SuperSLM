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
	int32_t admitted = 0;           // calls that admitted their token
	int32_t refused_at = -1;        // index of the first refused token, -1 when none was refused
	std::string refusal;            // the refusing call's status ("unreachable", or "status N")
	int32_t accepting_end = -1;     // the walk's accepting flag after the last admitted token
	std::vector<int32_t> accepting; // accepting flag after each admitted token (CPU and GPU)
	std::vector<uint32_t> states;   // walk state after each admitted token (GPU only; CPU has no query)
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
	// GPU only, one entry per decode-step call (the last is the call that found the dead end):
	// its wall time, and superslm_gpu::LastCallTiming().gpu_busy_ms after it -- the GPU timestamp
	// window of the step's last layer-loop submission, so the whole step's layer loop only at one
	// slice per token, and never the token finish (host or device logits).
	std::vector<double> step_ms;
	std::vector<double> step_gpu_busy_ms;
};

// What a sequence did, token by token: the tokens it decoded (or was teacher-forced), and the
// accepting flag and walk state after each. `states` is filled on GPU only (the CPU ABI has no
// walk-state query); `end` is "dead_end", "max_tokens", "refused" or "error:<call> <status>".
struct Trace {
	std::vector<int32_t> ids;
	std::vector<int32_t> accepting;
	std::vector<uint32_t> states;
	std::string end;
};

// One save point. A save is taken on the unbroken sequence after token `after`, the blob is
// restored into a fresh sequence, and the restored sequence is observed.
struct RestoreResult {
	int32_t after = -1;
	size_t blob_bytes = 0;
	std::string error;             // non-empty when the save point could not be run
	int32_t saved_accepting = -1;  // on the unbroken sequence, at the save
	uint32_t saved_state = 0;      // on the unbroken sequence, at the save (GPU only)
	int32_t restored_accepting = -1;  // on the restored sequence, before it decodes
	uint32_t restored_state = 0;      // on the restored sequence, before it decodes (GPU only)
	Trace continuation;            // the restored sequence's greedy decode
	Trace unbroken;                // teacher-forced only: the unbroken sequence's greedy decode
	Trace forced;                  // teacher-forced only: a second restore, teacher-forced through
	                               // the rest of the document
};

// `state_zero`: the restore-at-state-zero mutant. The harness overwrites the walk state recorded
// in each save blob with 0 before restoring it, so a restore check that cannot tell a wrong walk
// state from the right one passes the mutant.
struct RestoreOptions {
	int32_t max_tokens = 0;  // generated: the whole run's budget; teacher-forced: tokens per continuation
	bool state_zero = false;
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
// Generated save points. Re-runs the plain generation, teacher-checked against `ref`, saving after
// each index in `after`, and records the unbroken run's trace (ids, accepting, states) through the
// dead end. Then restores every blob and decodes each continuation to the end.
std::vector<RestoreResult> CpuRestores(CpuSession* s, const std::vector<int32_t>& prompt,
                                       const std::vector<int32_t>& ref, const std::vector<int32_t>& after,
                                       const RestoreOptions& opt, Trace* unbroken, std::string* error);
// Teacher-forced save points. Walks `doc` after `prompt` one token per call into `walk`; then, for
// each index in `after`, teacher-forces doc[0..after] after the prompt, saves, decodes the unbroken
// continuation, and restores the blob twice: once to decode, once to teacher-force the rest of `doc`.
std::vector<RestoreResult> CpuTfRestores(CpuSession* s, const std::vector<int32_t>& prompt,
                                         const std::vector<int32_t>& doc, const std::vector<int32_t>& after,
                                         const RestoreOptions& opt, Trace* walk, std::string* error);
bool CpuSeamAvailable();

// GPU side (adapter 0). The slice size comes from ARRAY_S2_GPU_LAYERS_PER_SLICE (array_s2_gpu.cpp);
// `head_on_device` maps with SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE (`gputime` only; every check maps
// with default residency, the output head on the host).
struct GpuOptions {
	bool head_on_device = false;
};
struct GpuSession;
GpuSession* GpuOpen(const uint8_t* bytes, size_t n, const std::string& schema, MapResult* map,
                    const GpuOptions& opt = GpuOptions{});
void GpuClose(GpuSession* s);
WalkResult GpuWalk(GpuSession* s, const std::vector<int32_t>& ids);
GenResult GpuGen(GpuSession* s, const std::vector<int32_t>& prompt, int32_t max_tokens);
std::vector<RestoreResult> GpuRestores(GpuSession* s, const std::vector<int32_t>& prompt,
                                       const std::vector<int32_t>& ref, const std::vector<int32_t>& after,
                                       const RestoreOptions& opt, Trace* unbroken, std::string* error);
std::vector<RestoreResult> GpuTfRestores(GpuSession* s, const std::vector<int32_t>& prompt,
                                         const std::vector<int32_t>& doc, const std::vector<int32_t>& after,
                                         const RestoreOptions& opt, Trace* walk, std::string* error);

// The restore-at-state-zero mutant: overwrites the walk state a save blob records with 0. The CPU
// blob ('SSB5'/'SSB6', src/sslm_abi.cpp) holds it as LE32 at byte 48; the GPU blob ('SLM5'/'SLM6',
// src/gpu/superslm_gpu.cpp and src/gpu/gpu_1p0.cpp) at byte 124, after the 120-byte header and the
// bound schema index. Returns false, changing nothing, for any other magic.
bool ZeroBlobWalkState(std::vector<uint8_t>* blob, bool gpu);

#endif  // ARRAY_S2_SHARED_H
