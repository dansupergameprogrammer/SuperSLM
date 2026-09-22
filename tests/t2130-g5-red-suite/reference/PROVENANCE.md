# Vendored reference material for the G5-1 schema compiler test suite

T-2919 (TE-372 S3): every file under this directory was copied, at the date and commit
named below, from a private per-session records worktree (Wizard repo,
`superslm-super-embedder-fixes-c20ddf` branch) into this repo, so the compiler test suite in
`tests/t2130-g5-red-suite/` is self-contained and reproducible from a fresh clone. Before this
fold, five of the suite's seven pytest files resolved these paths against
`D:/Wizard/.claude/worktrees/superslm-super-embedder-fixes-c20ddf`, a path private to one
session and one machine; every cell that touched it raised `FileNotFoundError` once that
worktree was removed, which is the routine end of a session (TE-372 finding S3).

## Reference compiler modules (design history, not maintained going forward)

Each of these four modules is a **fixed historical artifact**: a design-stage Python compiler
prototype, superseded by the next one in the chain, ending at `t2912-probe/`'s value-level
close, which is the design `tools/sslm_convert_schema.py` now ships (T-2915, "the compiler
port landed"). None of the four is edited after vendoring; each carries its own provenance
comment at the top of the file, above its original docstring.

| File | Vendored from (Wizard repo path) | Superseded by |
|---|---|---|
| `t2908-probe/sslm_convert_schema_bytelevel.py` | `Claude/Vitruvius/t2908-probe/sslm_convert_schema_bytelevel.py` | T-2910 |
| `t2910-probe/sslm_convert_schema_bytelevel_boundary.py` | `Claude/Vitruvius/t2910-probe/sslm_convert_schema_bytelevel_boundary.py` | T-2911 |
| `t2911-probe/sslm_convert_schema_close_structural.py` | `Claude/Vitruvius/t2911-probe/sslm_convert_schema_close_structural.py` | T-2912 |
| `t2912-probe/sslm_convert_schema_value_level.py` | `Claude/Vitruvius/t2912-probe/sslm_convert_schema_value_level.py` | ships (`tools/sslm_convert_schema.py`, T-2915) |

`t2912-probe/sslm_convert_schema_value_level.py` loads `t2911-probe/sslm_convert_schema_close_structural.py`
as its own base module at import time, by a path relative to its own `__file__` two levels up
(`../t2911-probe/...`) -- the two directories must stay siblings under `reference/`.

**T-2919's own reconsideration (TE-372 S3's second remedy clause):** `t2912-probe`'s module was,
at authoring time, the target design `tools/sslm_convert_schema.py` had not yet folded in.
T-2915 ported that exact design into the shipped module, so a test that only asserts "the
shipped compiler agrees with this vendored copy" is now comparing the shipped code to a copy
of its own design, not to an independent oracle -- the two are expected to agree because the
same design was transcribed into both, not because either was checked against ground truth.
`t2899_string_leaf_red.py`'s three-oracle framing was corrected to two (the shipped compiler,
checked directly against `json.loads`, RFC 8259's own independent implementation) for exactly
this reason; see that file's own module docstring. Where a comparison against these modules
remains in the suite (`t2908_byte_level_red.py`, `t2910_boundary_red.py`,
`t2912_value_level_red.py`), it is a **mutant** (a fixed, deliberately superseded or historical
construction proving the suite's own discriminating power against a known-wrong shape) or a
**structural audit of one committed module in isolation**, never the sole correctness claim for
a fresh behavior.

## The commissioned answer oracle and its fixtures (hash-pinned, never altered)

| File | Vendored from | Integrity |
|---|---|---|
| `t2912-probe/t2912_answer_oracle.py` | `Claude/Vitruvius/t2912-probe/t2912_answer_oracle.py` | sha256 `5dcdbebc212cba5c11cd00fa27edfd72f150d8ca69b2932044e260cc2e1fb97e`, matching the commissioned instrument's own attestation (`Claude/Vitruvius/t2912-probe/attestations/T2912-answer-value-oracle.json::instrument_sha256`) |
| `t2912-probe/t2912_heldout_prompts.json` | `Claude/Vitruvius/t2912-probe/t2912_heldout_prompts.json` | sha256 `44ffc54de4b62d5ff5f9ff84044b93947dc38bf316489b5e140ed8002e27fa76`, checked in `t2912_value_level_red.py::T2912_V5_HeldoutCensus` |
| `t2912-probe/t2912_oracle_must_accept.json` | `Claude/Vitruvius/t2912-probe/t2912_oracle_must_accept.json` at commit `2fe6a7299b` | byte-identical to the live records-tree file as of 2026-09-21 (verified: no in-flight edit touches this file) |
| `t2912-probe/t2912_oracle_must_reject.json` | `Claude/Vitruvius/t2912-probe/t2912_oracle_must_reject.json` at commit `2fe6a7299b` (5 rows) | **deliberately NOT the live records-tree file**: the live file carries 4 additional rows from an in-flight T-2914/TE-369 fold, out of this ticket's scope (the same "stay out of concurrent work" boundary `t2912_value_level_red.py`'s own `_pinned_json` already documented). A future ticket closing T-2914/TE-369 supersedes this vendored snapshot with the settled row set. |
| `t2912-probe/t2912_canonical_control.ids` | `Claude/Vitruvius/t2912-probe/t2912_canonical_control.ids` | vendored byte-identical; the forced-token-id control `T2912_V5_HeldoutCensus` uses for its "canonical control" arm |

The oracle's own commissioning record (must-accept scored correctly, must-reject scored
correctly, both by a seat independent of the oracle's own author, per
`StandardsDocument.md` Sec5.4) lives at
`Claude/Vitruvius/t2912-probe/attestations/T2912-answer-value-oracle.json` in the Wizard
repo; it is not duplicated here because it is a record about the oracle, not an input the
suite loads.

## The TE-368 prompt set and real-engine harness

| File | Vendored from | Notes |
|---|---|---|
| `te368-probe/te368_prompts.json` | `Claude/Loki/te368-probe/te368_prompts.json` (Wizard repo commit `6c3d177811`) | Only `prompts[1]` is used (`T2912_V1_TE368Prompt01`); the test itself pins the exact string, which is this file's own integrity check. |
| `te368-probe/te366_schema_run.cpp` | `D:/_te368/probe/te366_schema_run.cpp` (local scratch; itself a copy of `D:/_te366/te366_schema_run.cpp`, TE-366's own strike harness, never previously committed anywhere) | The real-engine CPU decode harness `T2912_V1_TE368Prompt01`/`T2912_V5_HeldoutCensus` run. Built via `build_te368_harness.bat` in this same directory (T-2919: previously an uncommitted local `.exe` at `D:/_te368/probe/te368_schema_run.exe`, unbuildable from a fresh clone -- TE-372 S3's third named gap, "put the harness source in-repo"). The harness uses only the public `superslm/sslm_abi.h` C ABI and needs a built engine (include + static lib) to link against; `build_te368_harness.bat` takes that install directory as its own first argument, so which engine build it links (1.5.0, 1.6.0, or any other tag) is an explicit, named choice at build time rather than a path buried in a probe-only script (TE-372 M6 is a separate, disclosed finding about which engine the existing evidence happened to link -- not this ticket's own finding to resolve; vendoring the harness makes that choice visible and explicit for whoever runs it next). |

## Real-model artifact gate

Every cell that needs a real cached checkpoint, the shipped A-EX artifact, the shopkeeper-LoRA
checkpoint, or the TE-368 harness is gated on the `SUPERSLM_G5_REAL_MODEL_TESTS` environment
variable (`t2913_common.py::_require_real_model`, `REAL_MODEL_ENV`): unset, the cell is
skipped (these artifacts are never provisioned on a bare CI runner); set but the artifact is
missing, the cell fails loudly, never silently skips (T-2909's fail-closed rule, preserved: a
box that opts in and then can't find the artifact has a real setup gap). CI never sets the
variable, so these cells do not run there; a release-verification pass, or a local box with
the checkpoint cached, sets it to get full coverage.
