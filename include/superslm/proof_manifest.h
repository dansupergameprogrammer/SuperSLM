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
// runtime code depends on them they are rejected at load." S3.3 gave the
// relation its runtime consumer, so as of b2a3a91 this geometry check IS wired
// into SslmModel::Load, via ValidateConfigGeometryJoin in model.cpp (board
// T-1335, D-SLM425). The functions here remain the independently unit-tested
// core; the loader wraps them rather than re-deriving the arithmetic.
//
// This comment previously recorded that wiring it would reject artifacts built
// from the S-HARDEN-1/-2 Cfg1Spec defaults, which did not satisfy
// hidden_size == heads * head_dim. That forecast was correct: wiring it failed
// 58 checks across ~45 call sites. The defaults were the defect and were
// repaired at 905daf6 -- the check was not weakened (D-SLM426).
#ifndef SUPERSLM_PROOF_MANIFEST_H
#define SUPERSLM_PROOF_MANIFEST_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/model.h"

namespace superslm {

// JSON string escaping -- the manifest carries tensor and section names, which
// are UTF-8 but otherwise untrusted, and (T-1449) rejection diagnostics, which
// routinely quote a section/entry name (e.g. `entry "kv"`). Escapes the ASCII
// control set and the two structural characters so any emitted document is
// valid JSON regardless of what a hostile-but-Load-accepted name, or a
// loader diagnostic naming one, contains. Exposed here (not file-local) so
// every JSON-emitting call site -- BuildProofManifestJson's own fields and
// tools/sslm_verify.cpp's hand-assembled REJECTED-path manifests alike --
// routes through the one implementation instead of a second, unescaped one.
// Throws only std::bad_alloc (S-HARDEN-7, F5; wrapped T-1475 -- being
// declared here, rather than file-local, makes it a member of the "throws
// only std::bad_alloc" contract's derived population).
std::string JsonEscape(std::string_view s);

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
// are read against (matches the section's ExpectedDtype). Throws only
// std::bad_alloc (S-HARDEN-7, F5).
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

// Throws only std::bad_alloc (S-HARDEN-7, F5).
std::vector<WeightScaleEvidence> ComputeWeightScaleEvidence(const SslmTensorManifest& weight_scales);

// SHA-256 of one section's raw bytes, lowercase hex -- independent evidence per
// section, distinct from the whole-artifact integrity hash SslmArtifact already
// verifies (T-129: whole-file integrity proves bytes were not changed; it proves
// nothing about whether they are safe operands, which is what the evidence above
// is for). Throws only std::bad_alloc (S-HARDEN-7, F5).
std::string HashSectionHex(const SslmSectionView& section);

// Assembles the full proof-manifest JSON body for an artifact the caller has
// already confirmed loads Ok (via a separate SslmModel::Load call whose
// SslmModelView is used only for its Ok/rejected STATUS and its plain-value
// Config fields -- never for a pointer-bearing field of that view). This
// function re-derives everything it reports directly from `artifact` itself
// -- Config, geometry, and every tensor's evidence are (re-)parsed HERE, from
// `artifact.Sections()`, not read from a previously-populated SslmModelView.
//
// This is deliberate, not merely independent-for-its-own-sake, but the reason is simpler than a
// prior version of this comment claimed. CORRECTED (T-2104, Poirot 8e07d0c7/7a0b6426 review,
// Significant 5 -- the tree-wide sweep for this exact false claim two rounds ago used the pattern
// `dangle|dangles`, which cannot match "dangling," and missed this site and its sibling in
// src/proof_manifest.cpp): this function's own signature takes only an `SslmArtifact&`, never an
// `SslmModelView` -- there is no already-parsed view available here to read Config's plain-value
// fields from in the first place, so re-parsing Config from `artifact` is the only source this
// function has, not a workaround for a lifetime hazard. (For the record, since the claim was
// asserted as established fact and is now known to be false: `SslmModelView`'s own `backing_`
// member, include/superslm/model.h, OWNS the file bytes every pointer in the view points into: the
// view is valid for its own lifetime, and reading a pointer-bearing field off an already-returned
// view is exactly as safe as reading a plain-value one. There is no latent undefined behavior in
// `SslmModel::Load` to route around, here or anywhere else.) The re-parse below is kept anyway, for
// an independent and real reason: every field this function reports is sourced the same way (fresh
// from `artifact`, never from a second object with a different provenance), which is worth the one
// cheap 84-byte re-parse regardless of what any caller's own view might already hold. Throws only
// std::bad_alloc (S-HARDEN-7, F5).
std::string BuildProofManifestJson(const SslmArtifact& artifact);

}  // namespace superslm

#endif  // SUPERSLM_PROOF_MANIFEST_H
