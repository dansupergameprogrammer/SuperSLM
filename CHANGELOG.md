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

### S1 — tokenizer + converter (complete)

- `TokenizerView` (`include/superslm/tokenizer.h`): an integer-only byte-level BPE
  tokenizer for the Qwen2.5 lineage. `Encode` does special-token splitting, NFC, the
  fixed Qwen/GPT pre-tokenization, byte-level BPE by merge rank; `Decode` reverses it.
  Deterministic and dependency-free — NFC and `\p{L}`/`\p{N}`/`\s` come from pinned
  Unicode tables in the artifact, never a platform Unicode or regex library.
- `.sslm` gains `Tokenizer` (20), `ChatTemplate` (21), and `UnicodeTables` (22)
  sections; the `TOK1`/`UNI1` sub-formats are specified in `docs/sslm_format.md`.
- Offline converter (`tools/convert_tokenizer.py`, `tools/unicode_tables.py`): reads a
  Hugging Face checkpoint, generates the pinned NFC + property tables (Unicode 15.1.0),
  emits the tokenizer artifact + a golden reference pack. The table-driven Unicode is
  proven to reproduce `unicodedata` exhaustively over all 1.1M codepoints.
- Verified bit-for-bit against the upstream HF tokenizer: 0 mismatches across thousands
  of adversarial + multilingual lines (CJK, Hangul NFC, ZWJ/flag emoji, combining
  marks, fullwidth, ligatures). The golden gate runs in CI on every toolchain.
- Suite: 247 checks / 0 failures, green on the full CI matrix. Reviewed and SHIP-
  verdicted (Poirot), after fixing two trust-boundary bounds-check defects in the
  sub-blob parse.

### S2 — integer-arithmetic core + kernels (complete)

- Bit-exact, dependency-free integer-only kernels for the forward pass: i-sqrt and
  the i-exp core (`intmath.h`), `RopeApplyPair` (rotary position embedding),
  a fixed-point SiLU sigmoid LUT (`silu_lut.h`, format v2's `SigmoidLut`/`SIL1`
  section), and the int8 matmul kernel (`matmul.h`) with a scalar reference proven
  identical to its SIMD path. Every kernel proven bit-equal to the Python reference
  implementation over its red-first suite, plus an exhaustive/hostile-input
  population for the arithmetic core.
- `.sslm` gains the `SigmoidLut` section (type 12, `SIL1`) and is required from
  format v2 onward.
- Golden cross-ISA/toolchain hashes committed for the SiLU LUT and matmul kernels
  (scalar == SSE2 == the pinned oracle).

### S3a — walking skeleton: converter geometry + KV landing (in progress)

- Config-vs-tensor geometry cross-check (`CheckConfigGeometry`,
  `include/superslm/proof_manifest.h`) wired into `SslmModel::Load`'s
  `ValidateConfigGeometryJoin`, rejecting a hidden_size/heads/head_dim mismatch at
  load rather than only in the converter-side proof manifest.
  `KvLandingScales`/`KvLandingReciprocals` schema-value domains
  (`ValidateKvLandingScalesDomain`/`ValidateKvLandingReciprocalsDomain`,
  `src/model.cpp`) gate the KV-landing mantissa/exponent/reciprocal ranges at load.
  The checked chain funnel and forward composition sites
  (`src/forward/checked_chain_funnel.cpp`, `src/forward/forward_sites.cpp`) are
  under active development; not yet feature-complete or CI-confirmed end to end
  (see the S-HARDEN entry below for the CI-availability caveat that applies to
  every change after 2026-07-23).

### S-HARDEN 1-8 — converter/CI hardening campaign (in progress)

A numbered series closing findings from the S2/S3a-era spec and coverage audits:
schema-value domain gates at load (S-HARDEN-1, F1/F14/F20/F22/F23/F24); the
tokenizer's TOK1×CFG1 vocabulary join and a strict shared UTF-8 decoder
(S-HARDEN-2, F6/F7/F15/F18); the converter's validate/serialize/verify transaction
and its independent C++ proof manifest, `tools/sslm_verify.cpp`/
`include/superslm/proof_manifest.h` (S-HARDEN-3, F13); the S2.5 acceptance-gate
missing assertion (S-HARDEN-4, F2); provenance verification for vendored reference
files (S-HARDEN-5, `tests/reference/check_provenance.py`); the cross-toolchain/
cross-optimization axis-digest harness (S-HARDEN-6, `tools/ci/sslm_axis_digest.cpp`);
a `std::bad_alloc`-only fault-injection contract across every allocating entry
point (S-HARDEN-7, F5; the last of its nineteen-member derived population,
`proof_manifest.h`'s `JsonEscape`, was wrapped at T-1475 -- confirmed by
`tools/ci/check_bad_alloc_contract.py --list-unwrapped` exiting 0); and
per-file branch-coverage floors (S-HARDEN-8, F12,
`tools/ci/check_branch_coverage_floors.py`).

This campaign's own CI verification is incomplete, both because most of it has
never run in the repository's own Actions workflow, and because the
bad-alloc-membership-check job was itself red until this commit range: its
membership-derivation script pinned `-std=c++17` against a C++20 codebase and
could not parse the tree at all once `trace_hook.h` started using `std::span`
(T-1476), which is what masked the unwrapped `JsonEscape` above (T-1475).
`.github/workflows/tests.yml`'s job history shows the last successful run was
2026-07-23T19:03:55Z (commit `4071828`), and every job added or widened after
that commit -- which includes most of S-HARDEN-3 through -8 -- has never
started (Actions billing limit; see the workflow's own forward-leaf-check job
comment). Suite counts are therefore not restated here per phase: a number
measured only by local, partial execution (no `clang++` on PATH on the
measuring machine for several suites -- a clang 18.1.8 toolchain is installed
there and reached by setting `SUPERSLM_CLANGXX`, D-SLM522) would misrepresent
itself as the "green on the full CI matrix" figures S0/S1 above earned
honestly. What has been proven true by local
execution as of this entry: `python -m pytest tools/ tests/reference/` (69/69)
and `python -m pytest tests/ci/` (79 non-clang-gated cells) both green against a
freshly built `sslm_verify`; with `SUPERSLM_CLANGXX` pointed at the local
clang 18.1.8 install, all 111 of `tests/ci`'s cells run (the 32 that
previously skipped for want of a discoverable clang++ included) and pass, as
of T-1475/T-1476; the S2.1 exhaustive certifier passes over its full
2^30 + 2^31-input domain on this tier.
