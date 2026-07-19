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
