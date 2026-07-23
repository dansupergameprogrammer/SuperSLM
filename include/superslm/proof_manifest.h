// The converter proof manifest and its independent checks (S-HARDEN-3, F13,
// SuperSLM_Plan.md §13 item 7, §17.3 cell 4).
//
// convert_model.py's writer discharges its "must load Ok" contract by invoking
// this project's own C++ loader (SslmModel::Load), not by asserting it. This
// header is that independent check: it re-derives facts about an already-loaded
// artifact (config-vs-tensor geometry, per-tensor extrema and saturation, section
// and artifact hashes) from the artifact's own bytes, never from the Python
// writer's intermediate arrays. `tools/sslm_verify.cpp` is the CLI that drives
// this into a JSON manifest; the functions here are the independently unit-
// tested core so the geometry cross-check (§17.3 cell 4, including its
// zero-boundary rejection cells) is provable without shelling out to Python.
//
// §17.3 cell 4: "Config geometry x tensor shapes... Where these are
// converter-only proofs they are recorded in the §13 item 7 manifest; where
// runtime code depends on them they are rejected at load." No runtime kernel in
// this tree yet consumes num_attention_heads/num_key_value_heads/head_dim (S3's
// forward path does not exist), so this geometry check is NOT wired into
// SslmModel::Load -- doing so today would reject artifacts that satisfy every
// existing S-HARDEN-1/-2 fixture's Cfg1Spec defaults (which do not happen to
// satisfy hidden_size == heads * head_dim) for a relation nothing at runtime yet
// depends on. It lives here, in the manifest path, until S3 gives it a runtime
// consumer -- exactly the sequencing SuperSLM_Plan.md §19's S-HARDEN preamble
// states ("S3's forward path is what turns two of them from parser defects into
// out-of-range model indices").
#ifndef SUPERSLM_PROOF_MANIFEST_H
#define SUPERSLM_PROOF_MANIFEST_H

#include <cstdint>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/model.h"

namespace superslm {

// The config x tensor-shape geometry cross-check (§17.3 cell 4). Operates on raw
// fields rather than a validated SslmModelConfig so it is directly testable
// against the zero-boundary cases the coverage audit names (2026-07-21 §3.3):
// `heads == 0` and, separately, `kv_heads == 0` must each produce a DEFINED
// rejection with a named diagnostic -- never a crash in the modulus that would
// otherwise evaluate `heads % kv_heads` before any guard can fire.
enum class ConfigGeometryStatus {
	Ok = 0,
	ZeroAttentionHeads,      // num_attention_heads == 0 -- checked before any modulus
	ZeroKeyValueHeads,       // num_key_value_heads == 0 -- checked before any modulus
	KvHeadsExceedsHeads,     // num_key_value_heads > num_attention_heads
	HeadsNotDivisibleByKv,   // num_attention_heads % num_key_value_heads != 0
	HiddenSizeGeometryMismatch, // hidden_size != num_attention_heads * head_dim
};

const char* ConfigGeometryStatusName(ConfigGeometryStatus s) noexcept;

struct ConfigGeometryResult {
	ConfigGeometryStatus status = ConfigGeometryStatus::Ok;
	std::string diagnostic;   // human-readable, names the offending values
};

// Checked in the order the coverage audit's zero-boundary note requires: both
// zero cases are rejected explicitly BEFORE `heads % kv_heads` is ever evaluated,
// so the check can never fault on its own input (the F14-shaped guard-vitality
// failure the audit names: "a guard that faults instead of rejecting").
ConfigGeometryResult CheckConfigGeometry(uint32_t hidden_size, uint32_t num_attention_heads,
                                         uint32_t num_key_value_heads, uint32_t head_dim) noexcept;

// Per-tensor extrema and saturation evidence for the proof manifest. Saturation
// is counted against the dtype's own representable range (int8: [-128,127]),
// which is what a coercion-not-proof defect (F13: int16[128,-129] -> int8 silently
// wrapping to [-128,127]) leaves as a population of exactly-boundary values in an
// artifact that should not have any if the source calibration was in range.
struct TensorEvidence {
	std::string name;
	int64_t min_value = 0;
	int64_t max_value = 0;
	uint64_t saturation_lo_count = 0;  // elements exactly at the dtype minimum
	uint64_t saturation_hi_count = 0;  // elements exactly at the dtype maximum
	uint64_t elem_count = 0;
};

// Reads every tensor in a parsed manifest section (WGT1/BIA1/ROP1) and computes
// its evidence. `dtype` fixes the signed range the min/max/saturation counters
// are read against (matches the section's ExpectedDtype).
std::vector<TensorEvidence> ComputeTensorEvidence(const SslmTensorManifest& manifest, SslmDtype dtype);

// WeightScales (WSC1) is itself a tensor manifest of (identity, mult, shift)
// int32 triples, not a single-value tensor -- its evidence is the shift column's
// margin against RoundingDivideByPOT's domain, not a raw int32 min/max (mult is
// intentionally unbounded per D-SLM142; identity is a {0,1} flag). One row per
// (tensor, per-triple index).
struct WeightScaleEvidence {
	std::string tensor_name;
	int32_t shift_min = 0;
	int32_t shift_max = 0;
	uint64_t row_count = 0;
	uint64_t identity_count = 0;   // rows with identity == 1
};

std::vector<WeightScaleEvidence> ComputeWeightScaleEvidence(const SslmTensorManifest& weight_scales);

// SHA-256 of one section's raw bytes, lowercase hex -- independent evidence per
// section, distinct from the whole-artifact integrity hash SslmArtifact already
// verifies (T-129: whole-file integrity proves bytes were not changed; it proves
// nothing about whether they are safe operands, which is what the evidence above
// is for).
std::string HashSectionHex(const SslmSectionView& section);

// Assembles the full proof-manifest JSON body for an already-Load-accepted
// artifact: format version, artifact fingerprint (re-derived from `artifact`,
// independent of whatever the Python writer printed), the geometry check result,
// per-section hash, and per-tensor/weight-scale evidence for every section this
// tool knows how to interpret. Never throws. The caller (tools/sslm_verify.cpp)
// adds the fields only the Python side can supply (source hashes, converter and
// reference commits) and writes the combined document.
std::string BuildProofManifestJson(const SslmArtifact& artifact, const SslmModelView& view);

}  // namespace superslm

#endif  // SUPERSLM_PROOF_MANIFEST_H
