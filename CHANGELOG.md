# Changelog

All notable changes to SuperSLM (Layer 1) are recorded here. Pre-release; the format
and API are not yet frozen (the ABI freezes at S-FREEZE, `SuperSLM_Plan.md` §19).

## [Unreleased]

### S0 — skeleton + artifact format (in progress)

- Repository skeleton mirroring the SuperFAISS core: CMake + `build.bat`, matrix CI
  (Windows/Linux/macOS-ARM, ASan/UBSan), standard-library-only test harness.
- `.sslm` artifact format v1 specified (`docs/sslm_format.md`) and declared
  (`include/superslm/artifact.h`): little-endian, 64-byte header, 40-byte section
  descriptors, aligned sections, SHA-256 integrity over the whole file, reject-over-
  degrade validation.
- In-tree SHA-256 (`include/superslm/sha256.h`) for the integrity hash / fingerprint;
  NIST known-answer vectors green.
- Artifact loader: contract declared, validation authored red-first (Curie's S0
  loader suite), implementation pending.
