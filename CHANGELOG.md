# Changelog

All notable changes to SuperSLM (Layer 1) are recorded here. Pre-release; the format
and API are not yet frozen (the ABI freezes at S-FREEZE, `SuperSLM_Plan.md` §19).

## [Unreleased]

### S0 — skeleton + artifact format + loader (complete)

- Repository skeleton mirroring the SuperFAISS core: CMake + `build.bat`, matrix CI
  (Windows/Linux/macOS-ARM, ASan+UBSan), standard-library-only test harness.
- `.sslm` artifact format v1 specified (`docs/sslm_format.md`) and declared
  (`include/superslm/artifact.h`): little-endian, 64-byte header, 40-byte section
  descriptors, aligned sections, SHA-256 integrity over the whole file, reject-over-
  degrade validation. Each section type's dtype is normative and validated.
- In-tree SHA-256 (`include/superslm/sha256.h`) for the integrity hash / fingerprint;
  NIST known-answer vectors green.
- Artifact loader (`OpenFromMemory` / `OpenFromFile`): full trust-boundary validation
  of a hostile file — bounded reads, integer-overflow-guarded, whole-file integrity,
  per-section and cross-section checks, required-`Config`, type↔dtype pairing. Views
  point into an owned buffer; the type is non-copyable so they cannot dangle.
- Suite: 158 checks / 0 failures, green on the full CI matrix (Windows, Linux, Linux
  ASan+UBSan, macOS-ARM). Reviewed and SHIP-verdicted (Poirot).
