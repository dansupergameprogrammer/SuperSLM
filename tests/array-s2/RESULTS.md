# S2 results: JSON arrays on a real artifact (CPU and GPU)

Branch `claude/project-thread-mq1of4`. Brief: [BRIEF.md](BRIEF.md). Two box runs:

- **Workhorse** (2026-10-26, the fix-round rerun): WORKHORSE, Radeon RX 7900 XT 20 GB (driver 32.0.21030.2001),
  Ryzen 9 5950X, sources at `897b9bb4`. All 79 `box_run.py` steps passed. Every raw file under `results/` except
  `attach.log`, `attach_retime.log` and `slices.txt` is from this run; the step logs are under `results/box/`.
- **Old box** (2026-10-25): DESKTOP-T2JJTNM, RTX 2080 SUPER, sources at `e9d73750` (no `src/` or `include/`
  difference from `origin/main`, v1.16.0). Its raw files are in this file's history at `897b9bb4`;
  `attach.log`, `attach_retime.log` and `slices.txt` are still its own.

No engine or converter defect was found on either box. The CPU and GPU token streams on workhorse are identical, for
all 50 streams, to the old box's (`gen.txt` at `897b9bb4`): the same tokens on an AMD and an NVIDIA GPU, and on two
CPUs. The findings are under **Findings**.

**Workhorse timings are "not quiet".** No other harness, test binary or attach ran during the run (the script's busy
check). But Chrome and the ChatGPT and Codex desktop apps were open. Before the run they used about 1 of 32 hardware
threads over a 5 s sample, with the GPU 3D engine idle. Frozen Hearts and Unreal were not running.

| Step | Old box (2080 SUPER) | Workhorse (7900 XT) |
|---|---|---|
| 1 Build | `build.bat` full build; 8 pre-existing red cells, none in array code | `sslm_verify`, shaders and the two harness binaries (`box_run.py`); `build.bat`'s suites not run |
| 2 Base | `qwen2.5-0.5b-instruct-combined.sslm`, TOKENIZER, no SchemaMasks | the same file, byte-identical (rebuilt, see Artifacts) |
| 3 Attach | verified, byte-reproducible, counts equal to the cloud's | the same: identical attached SHA-256, counts and fingerprint |
| Verifier check | not run (the verifier at `d7ebeaee` was re-checked by hand) | 5/5 (`verify_schema_masks_check.py`) |
| Precheck | not run | 31/31 |
| 4a Map | CPU +226 ms; schema VRAM 59,637,760 B | CPU +191 ms; GPU +160 ms; schema VRAM 59,768,832 B |
| 4b Walks | 3/3 CPU, 3/3 GPU | 3/3 CPU, 3/3 GPU, accepting only at the last token |
| 4c Rejections | 3/3 CPU, 3/3 GPU | 4/4 CPU, 4/4 GPU (adds the trailing comma inside `conditions`) |
| 4d Generation | 10/10 valid; CPU = GPU 10/10 | 10/10 valid; CPU = GPU 10/10; a second run identical 50/50 streams |
| 4e Save/restore, generated | 54/54 CPU, 54/54 GPU | 54/54 CPU, 54/54 GPU, now checking accepting (and GPU walk state) per token |
| 4e Save/restore, teacher-forced | not run | 21/21 CPU, 21/21 GPU, plus 3/3 walk traces each |
| 4f Engine features | exact = plain 10/10; full finish = plain 10/10; canonical valid 10/10, different 10/10 | the same |
| 5 Mutants | 2/2 killed (accept_shift by (b) and (d), offset_swap at map) | 2/2 plus restore-at-state-zero, each killed by its expected detection on CPU and GPU |
| GPU busy vs wall | not measured | measured: see **Decode speed** |

## Build

**Workhorse** (`box_run.py`, MSVC 14.34 from VS 2022 through `VsDevCmd.bat`, Python 3.13.17):
- `tools\build_verify.bat` builds `out\sslm_verify.exe` (11 s).
- The shaders: `dxc -T cs_6_2 -E main -O3 -HV 2018 -WX` on every `src\gpu\shaders\*.hlsl`, into `out\shaders`. This
  box has dxc 10.0.26100.0, not the 10.0.19041.0 that `build.bat` pins. The GPU token streams are identical to the
  old box's.
- `tests\array-s2\build_harness.bat` builds `out\array_s2.exe` and `out\array_s2_seam.exe`.
- `build.bat` and its suites were not run on workhorse; this round did not ask for them.

**Old box:** `build.bat` with the machine-local `build.local.bat`, 26 min 28 s, from a local copy of `build.bat` whose
verb-count check reads 55 instead of 51 (Finding 3).
- `dim7_capacity_red.cpp` had 12,484 checks with 5 failures: GrowableContextMap capacity at 9 and 17 contexts (31
  against 32, 62 against 64), and the AntiLmRetainedBytes calibration (1.1392x, outside [1.25x, 2.61x]).
- `out\superslm_tests.exe` had 322,372 checks with 3 failures, in `tests/schema_fast_decode/sfd_cells_x.inc` (`[1.X4]`,
  `[5.X4]`, `[11.X3]`).
- All 8 failures are in code this branch shares unchanged with `origin/main`. None of them was investigated.

## Artifacts

| | Path (workhorse) | SHA-256 |
|---|---|---|
| Base | `D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-combined.sslm` (518,177,048 B) | `3b48b25b90127b46f2ac038c23362ae38d4450e9ca3a7ab9a4d294180d75db8f` |
| Attached | `D:\_artifacts\SuperSLM\array-s2\qwen05-intentdoc-v2.sslm` (577,791,538 B) | `6c2d8378f8054bd04c44a99651d8110d3d3f63237239a2f7545c88580c23b178` |
| Mutant accept_shift | `D:\_scratch\array-s2\mutants\` | `e642e8a75debe4302ace659d6ebb54cb8adb9d192888e0e8ee75b90c6bf20401` |
| Mutant offset_swap | `D:\_scratch\array-s2\mutants\` | `7f9627dc3a81c0fb700dbfbb66cb8046beb589c773dda4a71ad140e902e90e98` |

([artifacts.txt](results/artifacts.txt), [mutants_artifacts.txt](results/mutants_artifacts.txt))

- **Tokenizer checkpoint:** the Qwen2.5-0.5B-Instruct snapshot `7ae557604adf67be50417f59c2c2f167def9a775`.
  - On workhorse it is at `D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-7ae5576`, copied from the old box's
    `paging-1p2\amd-kit\models\hf\hub` share.
  - On the old box it was `D:\hf_cache\hub\...\7ae5576...`.
- **The base on workhorse.** Workhorse had no tokenizer-bearing base: its `qwen2.5-0.5b-instruct.sslm` has no
  TOKENIZER section, and the `aex-pr` artifact already carries SchemaMasks.
  - A scratch script (`D:\_scratch\array-s2\make_combined.py`, outside the repository) rebuilt the base. It copies
    the plain artifact's sections and flags byte-for-byte, then appends `convert_tokenizer.py`'s TOKENIZER,
    UNICODE_TABLES and CHAT_TEMPLATE sections, as `tools/t2139_build_combined_fixture.py` composes them, through
    `sslm_format.write_artifact`.
  - The result is byte-identical to the old box's base (the SHA-256 above).
- **Python.** The tools need Python 3.13: `unicode_tables.py` requires unicodedata 15.1.0, and Python 3.12 has
  15.0.0. The workhorse run used Python 3.13.17 with `tokenizers` 0.23.3, `jsonschema` 4.26.0, `psutil` 7.2.2 and
  `numpy` (the old box used `tokenizers` 0.22.2 and `jsonschema` 4.22). `fixtures.py build` wrote the 16 prompt,
  walk and reject records the old box wrote, plus the fix round's 28 teacher-forced records.
- **The attached artifact** has integrity fingerprint `b75950bb9713fe9544b0b6565a7849e50fee43272eea3fc07882ad7ef40ddc55`
  and a 59,614,386 B SCM1 section on both boxes. A re-attach on workhorse wrote the same SHA-256
  ([box/reattach.log](results/box/reattach.log)).

## Measurements

**Attach** (`measure.py`, sampled every 50 ms):

| | Old box | Workhorse |
|---|---:|---:|
| Wall, both schemas | 28.91 s, 28.72 s (30.43 s while the build ran) | 23.27 s, 23.79 s |
| Converter peak working set | 2,146,209,792 B | 2,148,151,296 B |
| Verifier child peak | 1,749,508,096 B | 1,750,355,968 B |
| `intents` alone: wall, converter peak | 23.04 s, 2,003,795,968 B *(not in raw output)* | 18.80 s, 2,005,708,800 B |

Workhorse figures: [box/attach.measure.log](results/box/attach.measure.log), `reattach.measure.log` and
`attach_intents_only.measure.log`.

| Schema | DFA states | NFA states | Token transitions | Section bytes | Tracked temporary bytes | Recursion depth |
|---|---:|---:|---:|---:|---:|---:|
| `intents` (IntentDoc v2) | 635 | 681 | 3,074,805 | 36,660,995 | 165,514,483 | 6 |
| `intent_gate` (G1) | 399 | 424 | 1,921,739 | 22,953,415 | 133,260,439 | 4 |

The counts are identical on both boxes and equal the cloud measurement. `minItems`: `/properties/intents` 1,
`/properties/intents/items/properties/conditions` 0.

**Map** (three runs each; [map.txt](results/map.txt) is workhorse's):

| | Old box base | Old box schema | Workhorse base | Workhorse schema |
|---|---:|---:|---:|---:|
| CPU `sslm_model_map`, mean ms | 378.3 | 604.1 (+226) | 325.9 | 516.8 (+191) |
| GPU `sslm_gpu_model_map`, mean ms | 707.0 | 981.5 (+275) | 520.8 | 680.7 (+160) |
| GPU local-segment usage, after map minus before | 378,798,080 B | 438,435,840 B | 379,060,224 B | 438,829,056 B |

The schema's VRAM residency is 59,637,760 B on the old box (910 × 64 KiB) and 59,768,832 B on workhorse (912 × 64 KiB),
for the 59,614,386 B SCM1 section. The GPU model also keeps a host copy of the section.

**Decode speed** (greedy, the 10 prompts, 767 tokens; workhorse's: [box/gen_*.log](results/box/), [gputime.txt](results/gputime.txt)):

| | Old box ms/token | Workhorse ms/token |
|---|---:|---:|
| CPU plain (`layer_budget` 24, one slice per token) | 38.2 | 31.8 (run 2: 32.4) |
| CPU canonical jump-forward (291 tokens) | 33.5 | 28.1 |
| CPU exact jump-forward | 37.6 | 33.8 |
| CPU full finish (`array_s2_seam.exe`) | not recorded | 44.5 |
| GPU, G5 bridge, 24 slices per token (the default `ARRAY_S2_GPU_LAYERS_PER_SLICE=1`) | 146.5–148.7 | 210.1 (run 2: 212.1) |
| GPU, 1 slice per token, head on host | 27.1 | 29.2 |
| GPU, 1 slice per token, head on device | not measured | 18.0 |

The old box's slice sweep ([slices.txt](results/slices.txt), old box only; tokens identical at every setting):
24 slices 148.7, 12 slices 84.7, 4 slices 39.6, 1 slice 27.1 ms/token. Each extra slice cost about 5.3 ms there. On
workhorse it costs about 7.9 ms, (210.1 − 29.2) / 23. Each slice ends in a host wait on `sslm_gpu_ready`, so this
cost is the CPU-GPU round trip, not GPU work.

**GPU busy against wall** (workhorse only; `array_s2 gputime`, one slice per token, tokens checked equal to the
plain GPU run for 10/10 prompts in both modes). `gpu_busy_ms` is the engine's own GPU timestamp window
(`superslm_gpu::LastCallTiming`) around each decode step's layer-loop submission. Means are over the 767 decode steps
after step 0 (step 0 runs no layer loop). The token finish (host logits, or the device-head submission) has no GPU
timestamps, so it is inside wall time and the gap only.

| Head | Wall ms/token | GPU busy ms/token | Wall − busy | tok/s |
|---|---:|---:|---:|---:|
| on host (default residency) | 29.07 | 12.72 | 16.35 | 34.3 |
| on device (`SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE`) | 18.01 | 9.84 | 8.18 | 55.5 |

- Per prompt the spread is small: on host, wall 28.6–29.4 and busy 12.5–13.0 ms; on device, wall 17.8–18.4 and
  busy 9.6–10.3 ms.
- **The 27 ms against 6.6 ms question.** The 27.1 ms/token of the old box is wall time with the head on the host. On
  workhorse, in that mode, the GPU is busy for 44 % of the wall time (12.7 of 29.1 ms). The other 16.4 ms is the
  host-side token finish over the full vocabulary, the readback and the sync wait. Moving the head to the device
  removes half of that gap (16.4 → 8.2 ms) and gives 18.0 ms/token, 55 tok/s.
- No measurement on either box shows 6.6 ms of GPU busy time per token on this artifact; the old box never recorded
  busy time. This file has no source for the 6.6 ms figure.
- The busy window itself is 2.9 ms shorter with the head on the device (9.84 against 12.72 ms). The harness does not
  show why: the window should hold the same layer loop in both modes. It needs a look at what the host-head path
  submits inside `LastCallTiming`'s window.

## Step 4: the runtime harness

- **Code.** `array_s2_main.cpp` is the driver. `array_s2_cpu.cpp` uses the public C ABI (`sslm_abi.h`) and
  `array_s2_gpu.cpp` uses the shipped GPU surface and its G5 bridge (`gpu_1p0.h`). They are split across TUs as
  `tools/t2132_g5_gpu_parity` is, because the two headers' global enums collide.
- **Fixtures.** `fixtures.py` builds them with the pinned tokenizer. It derives each rejection's expected token from
  the byte offset where the document leaves IntentDoc v2, and validates outputs with `jsonschema` (Draft 2020-12).
- **Prompts.** Each prompt is ChatML, with `prompt-intentdoc-v2.txt` as the system turn and one utterance as the user
  turn: frozen-examples rows ex005, ex006, ex033, ex034, ex035, ex041, ex042, ex051, ex052 and ex053. Nothing named
  `sealed` was opened.
- **Tokenizing fixed documents.** The converter opens a free-text string only through the bare `"` token (the T-2910
  open-boundary rule, `tools/sslm_convert_schema.py`). So each document is tokenized in segments split around every
  free-text value's opening quote. The model itself emits `":` and then `"` there.
- **Precheck** ([precheck.txt](results/precheck.txt), workhorse): 31/31 against the repository converter's tables.
  Every walk is admitted with accepting only at its last token, every rejection is refused at its expected token,
  and every teacher-forced save point sits exactly at its byte.

**(b) Teacher-forced walks** ([walks_cpu.txt](results/walks_cpu.txt), [walks_gpu.txt](results/walks_gpu.txt)). Each
token is admitted by one `SSLM_SPAN_SCHEMA_CONTENT` prefill call. The fix round requires the accepting flag to be 0
after every token but the last.

| Walk | Old box tokens | Old box CPU/GPU | Workhorse tokens (fix-round tokenization) | Workhorse CPU/GPU |
|---|---:|---|---:|---|
| one intent, `conditions: []` | 29 | pass / pass | 31 | pass / pass |
| one intent, 1 condition | 62 | pass / pass | 65 | pass / pass |
| three intents, 0, 2 and 1 conditions | 175 | pass / pass | 184 | pass / pass |

The workhorse documents are re-tokenized so that a token boundary falls exactly after each listed array byte, which
is why they are a few tokens longer.

**(c) Rejections.** Each case is refused at the expected token (`SSLM_SCHEMA_SPAN_UNREACHABLE` on CPU,
`SSLM_SEQUENCE_REJECTED` on GPU), with every earlier token admitted:

| Document | Expected first bad token | Old box CPU/GPU | Workhorse CPU/GPU |
|---|---|---|---|
| trailing comma `},]}` | 28 (`]}`) | 28 / 28 | 28 / 28 |
| `"intents":[]` | 4 (`[]}`) | 4 / 4 | 4 / 4 |
| missing `target` | 13 (`conditions`) | 13 / 13 | 13 / 13 |
| trailing comma after the last condition | 54 (`],"`) | not run | 54 / 54 |

**(d) Constrained greedy generation** ([gen.txt](results/gen.txt), [gen_check.json](results/gen_check.json)):
- On workhorse, all 20 runs (10 CPU, 10 GPU) ended at the schema's dead end in an accepting state. Every document
  parsed and validated against IntentDoc v2 (`gen_check`: 0 failures).
- The CPU and GPU token streams are identical for all 10 prompts, 767 tokens in total.
- A second run of every variant (`gen_run2.txt`) is identical: 50/50 streams.
- All 50 streams are also identical to the old box's.
- The documents hold 1, 2 or 4 intents, and 0, 1 or 4 conditions per intent.

**(e) Save and restore.**
- **Generated save points** ([saveplan.txt](results/saveplan.txt), [restore_cpu.txt](results/restore_cpu.txt),
  [restore_gpu.txt](results/restore_gpu.txt)). These are the same 54 points on both boxes: after every token that
  contains an array `[` or `]`, or a comma whose innermost container is an array.
  - Each blob is restored into a fresh sequence and decoded to the dead end.
  - On workhorse, every continuation token is checked against the unbroken run's token, accepting flag and (on GPU)
    walk state.
  - Result: 54/54 on CPU (201,327,648 B blobs) and 54/54 on GPU (201,327,620 B), on both boxes. Workhorse also
    passed the 10 per-prompt trace lines on each backend.
  - Most generated points sit past the array byte, not exactly at it:

    | After | Exact | Bundled | Crossed |
    |---|---:|---:|---:|
    | `[` of `intents` | 10 | 0 | 0 |
    | `,` between intents | 1 | 8 (`},{"`) | 0 |
    | `[` of `conditions` | 1 | 2 (`":[{"`) | 16 (`[]`) |
    | `,` inside `conditions` | **0** | 3 (`"},{"`) | 0 |
    | `]` of `conditions` | 16 (`[]`) | 3 (`],"`) | 0 |
    | `]` of `intents` | 9 | 1 (`]}`) | 0 |

- **Teacher-forced save points** (workhorse only; [tfrestore_cpu.txt](results/tfrestore_cpu.txt),
  [tfrestore_gpu.txt](results/tfrestore_gpu.txt)). There are 21 points, each exactly at its byte:
  - after `[` of `intents` (3), each `,` between intents (2), `[` of `conditions` (5), `,` inside `conditions` (1),
    `]` of `conditions` (5) and `]` of `intents` (3);
  - inside two free-text strings (2).

  For each point, `array_s2 tfrestore` does the following on prompt ex041:
  1. Teacher-forces the document to the point, saves, and decodes 64 tokens.
  2. Restores twice: once to decode the same 64 tokens, and once to walk the rest of the document.

  The accepting flag (and GPU walk state) must match at the save, after restore, and on every later token. Result:
  21/21 on CPU and 21/21 on GPU, plus 3/3 walk traces on each.

**(f) Engine features on an array schema** (CPU; the GPU has no jump-forward verb):
- **Exact jump-forward** (`SSLM_JUMP_EXACT`): identical to plain decode on 10/10, on both boxes.
- **Masked finish against full finish.** `array_s2_seam.exe` sets the exactness hidden bound to 0 before map, which
  clears `legal_row_finish`. Its streams are identical to plain decode's on 10/10, on both boxes. The GPU (full
  finish, host mask) is a third witness, also 10/10.
- **Canonical jump-forward** (information only, [gen_check_canon.json](results/gen_check_canon.json)): 10/10
  documents are valid, but 10/10 differ from plain decode at token 3, on both boxes. Plain emits `":[` (8899), or
  `":` (788) on ex041 and ex042; canonical admits `":[{"` (66582). Canonical produces 28–30 tokens per prompt. See
  Finding 2.

## Step 5: mutants

[mutants.txt](results/mutants.txt) and [state_zero.txt](results/state_zero.txt) (workhorse) hold the raw rows. A step
passes only on its expected detection, with the mutant applied on every result and no setup error.

| Mutant | Mutation | `sslm_verify` (tip) | CPU | GPU |
|---|---|---|---|---|
| accept_shift | accepting state 213 → 214 (of 635; the only accepting id) | accepted, as expected: structurally valid | walks: 6/6 caught at the per-token accepting check; gen: 2/2 end at a dead end with `accepting=0` | the same, 6/6 and 2/2 |
| offset_swap | CSR `state_offsets[1]=2` and `[2]=4` swapped | refused, `SchemaMasksRejected -- BadSchemaStateOffsets`; the attach command wrote nothing | refused at `sslm_model_map` | refused at `sslm_gpu_model_map` |
| state zero | 0 written over the save blob's walk state (CPU `SSB5` byte 48, GPU `SLM5` byte 124) before restore | n/a | teacher-forced 13/13 caught, gen 7/7 caught | teacher-forced 13/13 caught, gen 7/7 caught |

On the old box the verifier at `e9d73750` accepted both mutants (Finding 1). Old-box mutant figures were not
recorded in raw output.

## Fix round (2026-10-26, cloud, then the workhorse rerun)

What the code review (FIX FIRST) and the coverage audit (REFUSE) asked for, and where it now stands. All of it ran on
workhorse.

- **Exact save points and saves inside strings:** the 21 teacher-forced points in (e).
- **A restore check that can fail:** a per-token check of the accepting flag and GPU walk state, the state-zero mutant,
  and walks that refuse an early accepting flag.
- **Verifier check:** `verify_schema_masks_check.py`, 5/5 with the tip's verifier. pytest collects it as
  `converter-v2/test_cv2_verify_schema_masks.py`.
- **No pass without checks:** every mode fails on a planned row with no reference, a count different from the plan, or
  zero records checked.
- **Raw output for every figure:** `box_run.py` writes each step's log under `results/box/`.
- **GPU timing split:** the `gputime` table above.
- **Not done:** review finding 2, isolating the canonical jump-forward cause. Canonical mode may be removed in 1.18;
  that choice is on hold with the user.

**Executed in the cloud before the rerun (Linux, no GPU):** `fixtures.py build` (44 records), `precheck` 31/31,
`saves` (the same 54 indices), the gcc `-Wall -Wextra` CPU build with no warnings, the CPU harness on a synthetic
artifact (walks and rejections 10/10, `tfrestore` 21/21, `--state-zero` fails 21/21, both mutants caught),
`verify_schema_masks_check.py` 5/5, and pytest with 44 passed.

## Findings

1. **`sslm_verify` did not check the SchemaMasks section** (old box, verifier at `e9d73750`).
   - The verifier loaded through `SslmModel::Load`, which never parses SCM1. So the attach command reported
     "verified" for offset_swap, which both runtime loaders refuse.
   - Fixed on this branch by `d7ebeaee`. On workhorse the tip's verifier refuses offset_swap and an SCM1 header-word
     corruption, and accepts the clean artifact and accept_shift (5/5).
2. **Canonical jump-forward closes arrays early** (both boxes, same streams).
   - Canonical mode admits a forced span on its own (`}` after `false`, `":[` after `"conditions"`), then leaves the
     array's continue-or-close choice to a fresh decode step.
   - **Hypothesis, not measured:** the model's own tokens bundle the forced byte with that choice (`},{"`, `[]`), and
     when it is given the bare forced token, it closes the array.
   - Across the 10 prompts, canonical produced exactly one intent with `conditions: []` every time. The output stays
     schema-valid, and the contract promises plain-identical output only in exact mode. Use exact mode with array
     schemas.
3. **`build.bat`'s C ABI verb count is stale** (old box).
   - The check expects 51 verbs; the inventory (`tests/t2807-api-slot/c_abi_verbs.ps1`) reports 55. The four extra
     are `sslm_prefix_save` and `sslm_prefix_restore` (1.14.0), and `sslm_seq_forced_tokens` and
     `sslm_seq_jump_forward` (1.15.0).
   - `build.bat` on `origin/main` has the same line, so the stock `build.bat` stops there on every tree.
4. **The GPU decode wall time is mostly outside the GPU** (workhorse).
   - With the head on the host, 56 % of each token's wall time is outside the layer loop's GPU window.
   - With the head on the device, wall time drops to 18.0 ms/token (1.6× faster).
   - At the default 24 slices per token, every slice adds about 7.9 ms of round trip on the 7900 XT (5.3 ms on the
     2080 SUPER).
5. **The workhorse toolchain differs from `build.bat`'s pins** (workhorse). The run used dxc 10.0.26100.0 rather than
   10.0.19041.0, and Python 3.13 is required. The outputs are unaffected: the token streams equal the old box's.

## Not done, or done differently from the brief

- **Old box worktree.** That run used its session's own worktree (`D:\SuperSLM\.claude\worktrees\hungry-haslett-d11041`,
  detached at `e9d73750`) because its write hook refused other paths. Workhorse used `D:\SuperSLM\.worktrees\array-s2`.
- **Workhorse base and checkpoint.** `D:\_artifacts\SuperSLM\aex` had no tokenizer folder and no qualifying base. Both
  were brought over and rebuilt as described under Artifacts.
- **`build.bat` suites** were not run on workhorse.
- **GPU coverage of (f).** Jump-forward and the masked finish are CPU verbs. On GPU, (f) is covered only as the
  full-finish witness in (d).
- **CI for the verifier pytest: known gap.** No workflow runs `tests/t3094-multicommand-s0/converter-v2/`. The box
  run's `verify_check` step is its only automated runner.

## Files

- `box_run.py`: the whole box run, resumable, one step at a time (`--list` shows the steps).
- `fixtures.py`: the fixture builder, table precheck, save-plan builder and output checker.
- `scm1_mutants.py`: the two SCM1 mutations, and a writer that applies one to an attached artifact.
- `array_s2_gpu_none.cpp`: the GPU side for a CPU-only build (Linux).
- `verify_schema_masks_check.py`: the verifier's SchemaMasks check (also collected by pytest).
- `array_s2_*.cpp` and `array_s2_shared.h`: the harness.
- `build_harness.bat`: builds the harness.
- `measure.py`: wall time and peak memory.
- `make_mutants.py`: the mutant attach generator.
- `results/`: raw outputs; `results/box/` holds each step's log and completion record.
