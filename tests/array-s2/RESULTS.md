# S2 results: JSON arrays on a real artifact (CPU and GPU)

2026-10-25 · DESKTOP-T2JJTNM (RTX 2080 SUPER, adapter 0) · branch `claude/project-thread-mq1of4`, sources at `e9d73750`
(no `src/` or `include/` difference from `origin/main`, v1.16.0). Brief: [BRIEF.md](BRIEF.md).

Every step passed. No engine or converter defect was found. Three findings are recorded under **Findings**: a verifier
coverage gap, a canonical jump-forward interaction with arrays, and a stale `build.bat` tripwire. Pre-existing red cells
in the 1.16.0 suites are listed under **Build**.

| Step | Result |
|---|---|
| 1 Build | Release CPU and GPU engine, `sslm_verify`, the test targets and this harness built. The test run is red on 8 pre-existing cells, none of them in array code (see Build). |
| 2 Base | `qwen2.5-0.5b-instruct-combined.sslm` has a TOKENIZER section and no SchemaMasks section |
| 3 Attach | pass: verified, byte-reproducible, counts equal to the cloud measurement |
| 4a Map | CPU +226 ms mean for the schema artifact; schema section VRAM 59,637,760 B |
| 4b Walks | 3/3 pass on CPU, 3/3 on GPU |
| 4c Rejections | 3/3 refused at the first bad token on CPU, 3/3 on GPU |
| 4d Generation | 10/10 valid IntentDoc v2 documents; CPU and GPU streams identical 10/10 (and at every GPU slice size tested) |
| 4e Save/restore | 54/54 on CPU, 54/54 on GPU |
| 4f Engine features | exact jump-forward identical 10/10; full finish identical 10/10; canonical jump-forward valid 10/10 but different 10/10 (cause below) |
| 5 Mutants | 2/2 killed. The verifier at `e9d73750` caught neither: one was caught by (b) and (d), the other at map time. The verifier at `d7ebeaee` refuses offset_swap |

## Build

- **Command:** `build.bat`, run from the worktree root with the machine-local `build.local.bat` (the
  T2199 Phase D model and adapter). The run used a local, untracked copy of `build.bat` that differs in
  one line, `if "%T2139_VERB_COUNT%"=="55"` in place of `"51"` (see Finding 3).
- **Wall time:** 26 min 28 s, from 2026-10-25T23:00:00Z to 23:26:28Z.
- **Harness:** `tests\array-s2\build_harness.bat` builds `out\array_s2.exe` (the production library) and
  `out\array_s2_seam.exe` (the same sources with `/DSUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS`). The
  seam binary is used only for the full-finish comparison in (f).
- **Verifier:** `sslm_verify` was compiled from `tools\sslm_verify.cpp` using `tools\build_verify.bat`'s
  own source list.
- **Suite results.** Every suite before the stop was green: the CMake CPU-only build, the gates, the
  Phase C and Phase D suites with the real model, and the T-2296 red suites.
  - The build stopped at `dim7_capacity_red.cpp`: 12,484 checks, 5 failures. They are GrowableContextMap
    capacity at 9 and 17 contexts (31 against 32, and 62 against 64), and the AntiLmRetainedBytes
    calibration (1.1392x, outside [1.25x, 2.61x]).
  - `out\superslm_tests.exe`, which `build.bat` reaches only after that suite, was run directly with the
    same fixture environment `build.bat` sets: 322,372 checks, 3 failures, all in
    `tests/schema_fast_decode/sfd_cells_x.inc` (cells `[1.X4]`, `[5.X4]` and `[11.X3]`).
  - These 8 failures are in code and tests this branch shares unchanged with `origin/main`. They are not
    caused by arrays, and this session did not investigate them.

## Artifacts

| | Path | SHA-256 |
|---|---|---|
| Base | `D:\hf_cache\superslm_artifacts\qwen2.5-0.5b-instruct-combined.sslm` (518,177,048 B) | `3b48b25b90127b46f2ac038c23362ae38d4450e9ca3a7ab9a4d294180d75db8f` |
| Attached | `D:\_artifacts\SuperSLM\array-s2\qwen05-intentdoc-v2.sslm` (577,791,538 B) | `6c2d8378f8054bd04c44a99651d8110d3d3f63237239a2f7545c88580c23b178` |
| Mutant accept_shift | `D:\_scratch\array-s2\mutants\accept_shift.sslm` | integrity fingerprint `8dc2ff7f…` |
| Mutant offset_swap | `D:\_scratch\array-s2\mutants\offset_swap.sslm` | integrity fingerprint `5a50bac6…` |

- **Tokenizer checkpoint:** `D:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775`.
  The attach command found its tokenizer tables byte-equal to the base's.
- **Attached artifact:** integrity fingerprint `b75950bb9713fe9544b0b6565a7849e50fee43272eea3fc07882ad7ef40ddc55`; SCM1
  section 59,614,386 B. Re-attaching on an idle box wrote a byte-identical file (same SHA-256).

## Measurements

**Attach** (`measure.py`, sampled every 50 ms; [attach_retime.log](results/attach_retime.log)):
- Wall time: 28.91 s and 28.72 s on an idle box. The first run, 30.43 s, overlapped the build.
- Converter (Python) peak working set: 2,146,209,792 B. Verifier child peak: 1,749,508,096 B.
- With the `intents` schema alone: 23.04 s, converter peak 2,003,795,968 B.

| Schema | DFA states | NFA states | Token transitions | Section bytes | Tracked temporary bytes | Recursion depth |
|---|---:|---:|---:|---:|---:|---:|
| `intents` (IntentDoc v2) | 635 | 681 | 3,074,805 | 36,660,995 | 165,514,483 | 6 |
| `intent_gate` (G1) | 399 | 424 | 1,921,739 | 22,953,415 | 133,260,439 | 4 |

IntentDoc v2 equals the cloud measurement: 635 states, 3,074,805 transitions, 36,660,995 bytes and about 165.5 MB
tracked. `minItems`: `/properties/intents` 1, `/properties/intents/items/properties/conditions` 0.

**Map** (three runs each; [map.txt](results/map.txt)):

| | Base | With schemas | Difference |
|---|---:|---:|---:|
| CPU `sslm_model_map` | 360–412 ms (mean 378.3) | 598–613 ms (mean 604.1) | +226 ms |
| GPU `sslm_gpu_model_map` | 649–769 ms (mean 707.0) | 967–990 ms (mean 981.5) | +275 ms |
| GPU local-segment usage after map minus before (DXGI) | 378,798,080 B | 438,435,840 B | **59,637,760 B** |

The schema's VRAM residency, 59,637,760 B, is the 59,614,386 B SCM1 section rounded up to 64 KiB (910 × 65,536).
The GPU model also keeps a host copy of the section.

**Decode speed** (idle box, greedy, the 10 prompts, 767 tokens in total; [gen.txt](results/gen.txt), [slices.txt](results/slices.txt)):

- **CPU:** 38.2 ms/token in one slice per token (`layer_budget` 24, so all 24 layers run in one `sslm_decode_step`
  call).
- **CPU with jump-forward:** canonical 33.5 ms/token over its 291 tokens; exact 37.6 ms/token.
- **GPU through the G5 bridge**, by slices per token. `SslmGpuSeqDecodeStepForG5Bridge` submits and then blocks on
  `sslm_gpu_ready` once per `dispatch_budget`, and one layer is 25 dispatches. The harness reads the slice size from
  `ARRAY_S2_GPU_LAYERS_PER_SLICE` (default 1). Default residency, no `SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE`:

  | Layers per slice | Slices per token | ms/token | Tokens |
  |---:|---:|---:|---|
  | 1 | 24 | 148.7 | identical |
  | 2 | 12 | 84.7 | identical |
  | 6 | 4 | 39.6 | identical |
  | 24 | 1 | 27.1 | identical |

- **Per-slice cost.** Each extra slice costs about 5.3 ms, (148.7 − 27.1) / 23, measured in this cell. The 146.5
  ms/token in (d) is the 24-slice setting. At one slice per token the GPU is 1.41× faster than the CPU.
- **Determinism.** A second run of each variant, and every slice setting, produced the same tokens.

## Step 4: the runtime harness

- **Code.** `array_s2_main.cpp` is the driver. `array_s2_cpu.cpp` uses the public C ABI (`sslm_abi.h`) and
  `array_s2_gpu.cpp` uses the shipped GPU surface and its G5 bridge (`gpu_1p0.h`). They are split across TUs as
  `tools/t2132_g5_gpu_parity` is, because the two headers' global enums collide.
- **Fixtures.** `fixtures.py` builds them with the pinned tokenizer (`tokenizers` 0.22.2), derives each rejection's
  expected token from the byte offset where the document leaves IntentDoc v2, and validates outputs with `jsonschema`
  4.22 (Draft 2020-12).
- **Prompts.** Each prompt is ChatML, with `prompt-intentdoc-v2.txt` as the system turn and one utterance as the user
  turn. The 10 utterances are frozen-examples rows ex005, ex006, ex033, ex034, ex035, ex041, ex042, ex051, ex052 and
  ex053. Nothing named `sealed` was opened.
- **Tokenizing fixed documents.** The converter opens a free-text string only through the bare `"` token (the T-2910
  open-boundary rule, `tools/sslm_convert_schema.py`). So each document is tokenized in segments split around every
  free-text value's opening quote; the model itself emits `":` then `"` there. With plain canonical encoding, every
  walk is refused at its first `":"` (3252) before a free-text value. That is the documented rule, not a defect.

**(b) Teacher-forced walks.** Each document is admitted one token per `SSLM_SPAN_SCHEMA_CONTENT` prefill call.
Every token was admitted, and the walk ended accepting, on CPU and GPU:

| Walk | Tokens | CPU | GPU |
|---|---:|---|---|
| one intent, `conditions: []` | 29 | pass | pass |
| one intent, 1 condition | 62 | pass | pass |
| three intents, 0, 2 and 1 conditions | 175 | pass | pass |

On GPU, every walk state after each token is in [walks_gpu.txt](results/walks_gpu.txt).

**(c) Rejections.** Each case was refused at the expected token (`SSLM_SCHEMA_SPAN_UNREACHABLE` on CPU,
`SSLM_SEQUENCE_REJECTED` on GPU), with every earlier token admitted:

| Document | Expected first bad token | CPU | GPU |
|---|---|---|---|
| trailing comma `},]}` | 28 (`]}`) | refused at 28 | refused at 28 |
| `"intents":[]` | 4 (`[]}`) | refused at 4 | refused at 4 |
| missing `target` | 13 (`conditions`) | refused at 13 | refused at 13 |

**(d) Constrained greedy generation.**
- All 20 runs (10 CPU, 10 GPU) ended at the schema's dead end in an accepting state, and every document parsed and
  validated against IntentDoc v2.
- The CPU and GPU token streams are identical for all 10 prompts, 767 tokens in total.
- The documents hold 1, 2 or 4 intents, and 0, 1 or 4 conditions per intent.

**(e) Save and restore.**
- **Save points.** A save follows every token that contains an array `[` or `]`, or a comma whose innermost
  container is an array. That gives 54 save points across the 10 CPU streams ([saveplan.txt](results/saveplan.txt)):
  - after `[` of `intents`: 10;
  - after `,` between intents: 9;
  - after `[` of `conditions`: 3, plus 16 tokens spelled `[]`;
  - after `,` between conditions: 3;
  - after `]` of `conditions`: 3, plus the same 16 `[]` tokens;
  - after `]` of `intents`: 10.
- **Method.** At each point the unbroken run (checked token by token against the reference) was saved. Each blob was
  restored into a fresh sequence and decoded to the dead end.
- **Result.** CPU 54/54 (`sslm_seq_save` and `sslm_seq_restore`, 201,327,648 B blobs) and GPU 54/54
  (`sslm_gpu_seq_save` and `sslm_gpu_seq_restore`, 201,327,620 B blobs). Every continuation equals the unbroken run
  and ends at the dead end.

**(f) Engine features on an array schema** (CPU; the GPU has no jump-forward verb):
- **Exact jump-forward** (`SSLM_JUMP_EXACT`, NULL draft, one call before each decode with a forced chain): identical
  to plain decode on 10/10.
- **Masked finish against full finish.** `array_s2_seam.exe` sets the exactness hidden bound to 0 before map, which
  clears `legal_row_finish`. Its token streams are identical to plain decode's on 10/10. The GPU path, which runs the
  full finish and masks on the host, is a third independent witness and is also identical 10/10.
- **Canonical jump-forward** (the documented host loop): 10/10 documents are valid, but 10/10 differ from plain
  decode, first at token 3. Plain emits `":[` (8899), or `":` (788) on ex041 and ex042; canonical admits `":[{"`
  (66582). The cause is Finding 2.

## Step 5: mutants

- **Construction.** `make_mutants.py` writes scratch copies of `tools/sslm_attach_schema.py`, patched to rewrite the
  `intents` schema's SCM1 bytes after `serialize_scm1`. Both runs used the attach command's own verifier gate.
- **Results.**

| Mutant | Mutation | `sslm_verify` | Step (b) | Step (d) | Caught by |
|---|---|---|---|---|---|
| accept_shift | accepting state 213 → 214 (of 635; the only accepting id) | accepted | 3/3 walks end non-accepting, CPU and GPU | runs end at a dead end, `accepting=0` | (b) and (d) |
| offset_swap | CSR `state_offsets[1]=2` and `[2]=4` swapped | **accepted** by the verifier at `e9d73750`; refused by the verifier at `d7ebeaee` | `sslm_model_map` refused it (status 4) and `sslm_gpu_model_map` refused it (status 8) | the same map refusal | (b) and (d), at map |

## Findings

1. **`sslm_verify` does not check the SchemaMasks section.**
   - The verifier loads through `SslmModel::Load`, which never parses SCM1. The section is parsed only by
     `SchemaMasksTable::Parse` inside `sslm_model_map` (`src/sslm_abi.cpp:2260`) and in the GPU model map.
   - So `sslm_attach_schema.py` printed "verified: independent loader accepted the artifact" for the offset_swap
     artifact, which both runtime loaders refuse.
   - The attach command's promise that the verifier checks the new artifact does not extend to the one section it
     adds. A correct serializer never produces such an artifact; a corrupted one would ship as verified.
   - **Fixed on this branch by `d7ebeaee`** (pushed from the cloud during this run). Re-checked on the box with
     `sslm_verify` rebuilt from `d7ebeaee`:
     - offset_swap: refused, `SchemaMasksRejected -- BadSchemaStateOffsets`;
     - the attached artifact: accepted, same fingerprint;
     - accept_shift: accepted, as expected, because it is structurally valid. (b) and (d) catch it.
2. **Canonical jump-forward closes arrays early.**
   - Canonical mode admits a forced span on its own: `}` after `false`, and `":[` after `"conditions"`. It then
     leaves the array's continue-or-close choice to a fresh decode step.
   - The model's own tokens bundle the forced byte with that choice (`},{"`, `[]`, `[` then `{"`). When the model is
     given the bare forced token, it closes the array.
   - Across the 10 prompts, canonical produced exactly one intent with `conditions: []` every time. Plain decode
     produced up to 4 intents and 4 conditions.
   - The output stays schema-valid, and the documented contract promises plain-identical output only in exact mode.
     Exact mode is the jump-forward to use with array schemas.
3. **`build.bat`'s C ABI verb count is stale.**
   - The check expects 51 verbs. The inventory (`tests/t2807-api-slot/c_abi_verbs.ps1`) reports 55: the four extra
     are `sslm_prefix_save` and `sslm_prefix_restore` (1.14.0), and `sslm_seq_forced_tokens` and
     `sslm_seq_jump_forward` (1.15.0).
   - `build.bat` on `origin/main` has the same line, so the stock `build.bat` stops there on every tree.

## Not done, or done differently from the brief

- **Worktree location.** The work ran in this session's own worktree (`D:\SuperSLM\.claude\worktrees\hungry-haslett-d11041`,
  detached at `e9d73750`) instead of `D:\SuperSLM\.worktrees\array-s2`, because this session's write hook refuses
  edits outside its own worktree. Commits were pushed to `claude/project-thread-mq1of4` by fast-forward.
- **GPU coverage of (f).** Jump-forward and the masked finish are CPU verbs. On GPU, (f) is covered only as the
  full-finish witness in (d).
- **Build outcome.** The test run is red on the 8 pre-existing cells listed under Build, which this session did not
  investigate.

## Files

- `fixtures.py`: fixture builder, save-plan builder and output checker.
- `array_s2_*.cpp` and `array_s2_shared.h`: the harness.
- `build_harness.bat`: builds the harness.
- `measure.py`: wall time and peak memory.
- `make_mutants.py`: the mutant attach generator.
- `results/`: raw outputs. `fixtures.txt`, `walks_*.txt`, `gen.txt` and `gen_check.json` (every document, with
  validation and comparisons), `saveplan.txt`, `restore_*.txt`, `map.txt`, and the attach logs.
