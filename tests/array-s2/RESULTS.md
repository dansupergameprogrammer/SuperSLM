# S2 results: JSON arrays on a real artifact (CPU and GPU)

2026-10-25 · DESKTOP-T2JJTNM (RTX 2080 SUPER, adapter 0) · branch `claude/project-thread-mq1of4`, sources at `e9d73750`
(no `src/` or `include/` difference from `origin/main`, v1.16.0). Brief: [BRIEF.md](BRIEF.md).

Every step passed. No engine or converter defect was found. Three findings are recorded under **Findings**: a verifier
coverage gap, a canonical jump-forward interaction with arrays, and a stale `build.bat` tripwire. Pre-existing red cells
in the 1.16.0 suites are listed under **Build**.

**S2 fix round (2026-10-26, cloud), pending the box rerun.** The code review (FIX FIRST) and the coverage audit (REFUSE)
found that step 4e's save points mostly sit past the array bytes the brief lists, that no save sits inside a string,
that the restore check compares only tokens and the final dead end, and that some figures below have no raw output.
The harness now has teacher-forced save points exactly at every listed byte and inside two free-text values, a restore
check that compares the accepting flag after every token (and on GPU the walk state), the restore-at-state-zero
mutant, a rejection inside `conditions`, a hardened verifier check, and `box_run.py`, which reruns every step and
writes the logs behind each figure. Until that rerun replaces them, every figure below is the 2026-10-25 run's; those
marked "not recorded in raw output" have no committed log behind them. Fix-round detail: **Fix round** below.

| Step | Result |
|---|---|
| 1 Build | Release CPU and GPU engine, `sslm_verify`, the test targets and this harness built. The test run is red on 8 pre-existing cells, none of them in array code (see Build). |
| 2 Base | `qwen2.5-0.5b-instruct-combined.sslm` has a TOKENIZER section and no SchemaMasks section |
| 3 Attach | pass: verified, byte-reproducible, counts equal to the cloud measurement |
| 4a Map | CPU +226 ms mean for the schema artifact; schema section VRAM 59,637,760 B |
| 4b Walks | 3/3 pass on CPU, 3/3 on GPU |
| 4c Rejections | 3/3 refused at the first bad token on CPU, 3/3 on GPU |
| 4d Generation | 10/10 valid IntentDoc v2 documents; CPU and GPU streams identical 10/10 (and at every GPU slice size tested) |
| 4e Save/restore | 54/54 on CPU, 54/54 on GPU at generated save points; most sit past the array byte, not at it (see (e)). Teacher-forced exact points: not yet run on the box |
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
| Mutant accept_shift | `D:\_scratch\array-s2\mutants\accept_shift.sslm` | integrity fingerprint `8dc2ff7f…` *(not recorded in raw output)* |
| Mutant offset_swap | `D:\_scratch\array-s2\mutants\offset_swap.sslm` | integrity fingerprint `5a50bac6…` *(not recorded in raw output)* |

- **Tokenizer checkpoint:** `D:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775`.
  The attach command found its tokenizer tables byte-equal to the base's.
- **Attached artifact:** integrity fingerprint `b75950bb9713fe9544b0b6565a7849e50fee43272eea3fc07882ad7ef40ddc55`; SCM1
  section 59,614,386 B. Re-attaching on an idle box wrote a byte-identical file (same SHA-256) *(not recorded in raw output)*; the
  committed [attach_retime.log](results/attach_retime.log) shows the same integrity fingerprint on both re-attaches.

## Measurements

**Attach** (`measure.py`, sampled every 50 ms; [attach_retime.log](results/attach_retime.log)):
- Wall time: 28.91 s and 28.72 s on an idle box. The first run, 30.43 s, overlapped the build. Its
  [attach.log](results/attach.log) `peak_bytes=3896426496` line came from a `measure.py` that is not the committed one.
- Converter (Python) peak working set: 2,146,209,792 B. Verifier child peak: 1,749,508,096 B.
- With the `intents` schema alone: 23.04 s, converter peak 2,003,795,968 B *(not recorded in raw output)*.

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
- **Determinism.** Every slice setting produced the same tokens ([slices.txt](results/slices.txt)). A second run of
  each variant produced the same tokens *(not recorded in raw output)*.
- **GPU busy against wall** (one slice per token, head on the host and on the device): not yet measured; the
  `gputime` step of `box_run.py` records both per token. 27.1 ms/token above is wall time.

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
  - The counts by array byte, relabelled in the fix round from the same committed streams (`fixtures.py saves` on
    [gen.txt](results/gen.txt)). *Exact* means the token ends with that byte, so the save sits exactly after it.
    *Bundled* means later bytes of the same token follow it, so the save sits past it. *Crossed* is an `[` closed by
    its `]` in the same token (`[]`): the save is never at the after-`[` state, and it is no longer counted there.

    | After | Exact | Bundled | Crossed |
    |---|---:|---:|---:|
    | `[` of `intents` | 10 | 0 | 0 |
    | `,` between intents | 1 | 8 (`},{"`) | 0 |
    | `[` of `conditions` | 1 | 2 (`":[{"`) | 16 (`[]`) |
    | `,` inside `conditions` | **0** | 3 (`"},{"`) | 0 |
    | `]` of `conditions` | 16 (`[]`) | 3 (`],"`) | 0 |
    | `]` of `intents` | 9 | 1 (`]}`) | 0 |

  - The 2026-10-25 text counted the 16 `[]` tokens as "after `[` of `conditions`" and the bundled tokens as the
    positions themselves. The brief's positions after `,` and inside `conditions` were reached exactly 1, 0 and 1
    times. The fix round's teacher-forced points cover them (see **Fix round**).
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
  (66582). Finding 2 states a hypothesis for the cause; it has not been measured.

## Step 5: mutants

- **Construction.** `make_mutants.py` writes scratch copies of `tools/sslm_attach_schema.py`, patched to rewrite the
  `intents` schema's SCM1 bytes after `serialize_scm1`. Both runs used the attach command's own verifier gate.
- **Results.** Every figure in this table is *(not recorded in raw output)*: no log of either mutant run, or of the
  box re-check of `d7ebeaee`, was committed. The fix round's `mutant_*` and `state_zero_*` steps record them.

| Mutant | Mutation | `sslm_verify` | Step (b) | Step (d) | Caught by |
|---|---|---|---|---|---|
| accept_shift | accepting state 213 → 214 (of 635; the only accepting id) | accepted | 3/3 walks end non-accepting, CPU and GPU | runs end at a dead end, `accepting=0` | (b) and (d) |
| offset_swap | CSR `state_offsets[1]=2` and `[2]=4` swapped | **accepted** by the verifier at `e9d73750`; refused by the verifier at `d7ebeaee` | `sslm_model_map` refused it (status 4) and `sslm_gpu_model_map` refused it (status 8) | the same map refusal | (b) and (d), at map |

## Fix round (2026-10-26, cloud; box rerun pending)

Each item, the code-review finding or audit gap it answers, and what ran in the cloud. Nothing here has run on a GPU.

- **Exact save points** (review 1, audit gap 1). `fixtures.py build` re-tokenizes the three walk documents so a token
  boundary falls exactly after each listed byte: after `[` of `intents`, after each `,` between intents, after `[`
  and after each `,` inside `conditions` (the `"},` boundary included), and after each `]`. That is 19 points: `[` of
  `intents` 3, `,` between intents 2, `[` of `conditions` 5, `,` inside `conditions` 1, `]` of `conditions` 5, `]` of
  `intents` 3. Generated-stream saves are reported as exact, bundled or crossed (table in (e)).
- **Saves inside strings** (audit gap 2): 2 more points, after `"the` in intent 2's target and after `"be` in intent
  3's condition entity. 21 teacher-forced points in all.
- **A restore check that can fail** (audit gap 3). `array_s2 tfrestore` prefills a prompt (ex041), teacher-forces the
  document to each point, saves, decodes 64 greedy tokens, then restores the blob twice: once to decode the same 64
  tokens, once to teacher-force the rest of the document. It requires the accepting flag (and on GPU the walk state)
  at the save and right after restore to equal the one-token-per-call walk's, the restored decode to equal the
  unbroken one token by token (ids, accepting, and on GPU walk state), and the rest of the document to walk exactly
  as the unbroken walk did. `restore` (generated streams) now records the unbroken run's accepting flag and walk state
  through its dead end and checks every continuation token against them. Walks require the accepting flag to be 0
  after every token but the last. `--state-zero` is the restore-at-state-zero mutant: the harness writes 0 over the
  walk state the save blob records (CPU `SSB5` byte 48, GPU `SLM5` byte 124) before restoring. The CPU ABI has no
  walk-state query, so on CPU the check reads the accepting flag and the tokens only.
- **Verifier check** (review 3). `verify_schema_masks_check.py` attaches IntentDoc v2 with `--verifier <binary
  under test>` and checks the exit status, then requires the clean artifact and accept_shift to be accepted, and
  offset_swap and an SCM1 header word to be refused with `reject_status` SchemaMasksRejected and the expected
  reason. pytest collects it as `converter-v2/test_cv2_verify_schema_masks.py`.
- **No pass without checks** (review 5). Every mode fails on a planned row with no reference or no prompt, on a
  result count that differs from the plan, and on zero records checked.
- **Raw output for every figure** (review 4). `box_run.py` writes each step's log under `results/box/` and the raw
  files under `results/`. Figures without a log are marked above.
- **Rejection inside `conditions`**: a trailing comma after the last condition, refused at token 54 (`],"`).
- **GPU timing split**: `array_s2 gputime`, at one slice per token, with the output head on the host and on the
  device, tokens checked against the plain GPU run. Per decode step it logs wall ms and `gpu_busy_ms`, the engine's
  own GPU timestamp window (`superslm_gpu::LastCallTiming`) around the step's layer-loop submission, so wall minus
  busy is the per-token gap, and a `steps` line per prompt gives both against token position. The token finish
  (host logits, or the device logits submission) has no timestamps in the engine and is in wall time only;
  measuring it on the GPU's clock would need a `src/` change.
- **Not done:** review finding 2 (isolating the canonical jump-forward cause). Canonical mode may be removed in
  engine 1.18, a choice on hold with the user. Finding 2 below states its cause as a hypothesis.

**Executed in the cloud (Linux, no GPU):**

| Run | Result |
|---|---|
| `fixtures.py build`, pinned tokenizer | 44 records; the 10 prompt, 3 walk and 3 reject records equal the committed `fixtures.txt`; 21 teacher-forced points, each exactly at its byte |
| `fixtures.py precheck` (IntentDoc v2 compiled by the repository converter) | 31/31: every walk admitted with accepting only at its last token, every rejection refused at its expected token |
| `fixtures.py saves` on the committed `gen.txt` | the same 54 save indices; the exact, bundled and crossed counts in (e), equal to the code review's table |
| CPU harness, gcc `-Wall -Wextra`, linked against `build/libsuperslm.a` with `array_s2_gpu_none.cpp` | builds with no warnings; the GPU TU passes a syntax-only clang check with stand-in DXGI headers |
| CPU harness on a synthetic random-weight artifact (the converter-v2 base at context 1024, both schemas attached) | walks and rejections 10/10; `tfrestore` 21/21; `--state-zero` fails 21/21; accept_shift fails every walk at the per-token accepting check; offset_swap refused at `sslm_model_map` |
| `verify_schema_masks_check.py` | 5/5 with the verifier at the tip; with a verifier built from `e9d73750` both must-reject cases fail |
| pytest, `test_cv2_verify_schema_masks.py` with `test_cv2_attach.py` | 44 passed |

The synthetic model never reaches a dead end in generation (it runs to the token cap inside a string), so the
generated-stream `restore` passing path, the GPU, and every timing run only on the box.

**Re-check follow-ups (`poirot-s2-recheck.md`, FIX FIRST):**

- **gen_check.** Plain CPU, plain GPU, exact jump-forward and the full finish are checked together and must exit 0;
  any difference, invalid document or crash fails the step. Canonical is checked separately as information only
  (`gen_canon.txt`): its differences are recorded, never a pass condition.
- **Mutant kills.** A mutant step passes only on the expected detection: the mutant was applied (`mutated=1` on every
  result, or the `MUTANT` line from the writer), the result count equals the plan, and the failure is an
  accepting-flag, walk-state or token mismatch, or for offset_swap the expected map refusal. A missing row, a restore
  error or a mutant that was never applied fails the step. Checked on canned logs for each case and on the real
  CPU state-zero run (13/13 applied, 13 detections).
- **Completion markers.** Each records the git HEAD and the hashes of the harness and seam binaries; a change reruns
  the step (a binary change reruns the steps after the build). A `--base` or `--ckpt` that differs from `locate.json`
  stops the run with a message; `--redo <step>` reruns that step and every later one.
- **Timing.** `gputime` fails if `gpu_busy_ms` is 0, missing or above wall for any decode step after step 0. Step 0
  after the prompt prefill runs no layer loop (the logits are already ready), so it has no busy value of its own and
  is left out of the check and the means.

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
   - **Hypothesis, not measured:** the model's own tokens bundle the forced byte with that choice (`},{"`, `[]`, `[`
     then `{"`), and when the model is given the bare forced token, it closes the array. No run isolates this:
     canonical decode leaves the plain stream at token 3, so every later array decision is made on a different
     prefix, and plain ex041 decodes a bare `[` and then continues with `{"` instead of closing.
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
- **CI for the verifier pytest: known gap.** No workflow runs `tests/t3094-multicommand-s0/converter-v2/`
  (the pytest runs in `.github/workflows/` cover `tests/ci/`, `tools/`, `tests/reference/` and
  `tests/t2296-fp-free-open-red-suite`; neither CMake nor `build.bat` collects it). The workflows are
  left unedited; the box run's `verify_check` step is the only automated runner of `test_cv2_verify_schema_masks.py`.
- **Build outcome.** The test run is red on the 8 pre-existing cells listed under Build, which this session did not
  investigate.

## Files

- `box_run.py`: the whole box run, resumable, one step at a time (`--list` shows the steps).
- `fixtures.py`: fixture builder, table precheck, save-plan builder and output checker.
- `scm1_mutants.py`: the two SCM1 mutations, and a writer that applies one to an attached artifact.
- `array_s2_gpu_none.cpp`: the GPU side for a CPU-only build (Linux).
- `verify_schema_masks_check.py`: the verifier's SchemaMasks check (also collected by pytest).
- `array_s2_*.cpp` and `array_s2_shared.h`: the harness.
- `build_harness.bat`: builds the harness.
- `measure.py`: wall time and peak memory.
- `make_mutants.py`: the mutant attach generator.
- `results/`: raw outputs. `fixtures.txt`, `walks_*.txt`, `gen.txt` and `gen_check.json` (every document, with
  validation and comparisons), `saveplan.txt`, `restore_*.txt`, `map.txt`, and the attach logs.
