# SuperSLM 1.2 release-candidate review packet

This packet defines the candidate, its public claim, the evidence already executed, and the two
independent gates that remain before release. Greedy remains the default. Damped greedy is an
explicitly selected quality tradeoff for loop-prone free-text generation.

## Index

1. [Candidate and decision](#1-candidate-and-decision)
2. [Public contract](#2-public-contract)
3. [Executed evidence](#3-executed-evidence)
4. [Independent gates remaining](#4-independent-gates-remaining)
5. [Adversarial review brief](#5-adversarial-review-brief)
6. [Release procedure](#6-release-procedure)

## 1. Candidate and decision

- Candidate branch: `codex/t2199-b0-calibration`
- Review range: `main...codex/t2199-b0-calibration`
- Version prepared in-tree: `1.2.0`
- Ruled defaults: `alpha=2` (`alpha_q15=65536`), anti-LM order `n=2`, `top_k=6`
- Compatibility decision: `sslm_decode_step` and ordinary conversion remain greedy. Damped
  greedy requires both `convert_model.py --enable-damped-greedy` and an explicit runtime mode.

## 2. Public contract

The production converter emits a DGC1 scale section and flag when requested. The CLI accepts
`--decode-mode damped-greedy`; alpha/order/top-k are optional and default to the ruled row. The C
ABI exposes `sslm_decode_params_init`, which fills a complete v2 parameter block and derives the
fixed-point scale triple from the mapped artifact. An absent, malformed, or out-of-domain DGC1
section is a defined rejection.

Damped selection runs after schema masking, has per-sequence anti-LM state, participates in
reset/save/restore/prefix adoption, works with runtime adapters, contributes its selected tokens
to the decode digest, and preserves the concurrent-release contract. Greedy output is graded
against the independent direct-engine loop, not against a second call through the same ABI path.

The public quality statement is deliberately bounded: the measured corpus often reads more
coherently and its observed loop locks disappear, while legitimate repeated structure can also
be penalized. `list_primed_00` is the concrete counterexample: damped greedy stopped the runaway
list but also changed the list's formatting. Format-sensitive work should use greedy or a schema.

## 3. Executed evidence

### 3.1 Calibration and real generation

Phase B0 evaluated the 63-row primary grid after all commissioning controls passed. Phase E then
ran 192 fresh greedy and 192 damped generations: 48 prompts × two model sizes × two token
ceilings. Greedy reproduced the frozen baseline exactly in all 192 cases. Damped greedy matched
the B0 reach ceiling in all four cells, moved all three known 0.5B lock cases, and introduced no
new lock. Full tables and real text are in
[`t2199-phase-e-confirmation.md`](../calibration/t2199-phase-e-confirmation.md).

### 3.2 Current candidate checks

| Cell | Population | Result |
|---|---|---|
| Production converters | default, DGC1, Option-G+DGC flag composition; adapter conversion, diagnostics, magnitude warning, and notice suites | 65 passed |
| Phase D1 | artifact flag and DGC1 rejection paths | 18 checks, 0 failures |
| Phase D2 | independent greedy oracle, damped wiring/lifecycle/digest/default initializer, schema-mask-first composition | 234 checks, 0 failures on hermetic S8 |
| Real schema composition | Qwen2.5 0.5B DGC+SCM1 artifact, independent DFA replay | 259 checks, 0 failures; adapter-only cell skipped |
| Phase D2a | full selector plus isolated renormalizer against a real forward | 29 checks, 0 failures |
| Phase D3 | 200 concurrent teardown trials | 1,805 checks, 0 failures |
| CLI defaulted damped mode | Qwen2.5 0.5B, parameters omitted | successful 8-token production generation |
| Real adapter composition | Qwen2.5 1.5B base + 28-layer rank-16 runtime adapter | 248 checks, 0 failures; schema-only cell skipped |
| T-2138 real-artifact ABI suite | base, adapter, foreign shape, variant, and tokenizer artifacts supplied | 567 checks, 0 failures, 0 skips across 11 dimensions |
| Production CLI + adapter | paired greedy/damped 1.5B generation, damped parameters omitted | both modes loaded the adapter and completed the same 12-token response |
| Root build | all mandatory compile, ABI, Phase A-D, core, and structural gates | PASS; 34,189 core checks, 0 failures |

The hermetic S8 fixture is regenerated on every root build, carries DGC1 and a compiled schema,
and is sized to the older T-2138 ABI suite's actual population (`vocab_size=128`,
`context_cap=64`). Root `build.bat` now builds and executes all eleven T-2138 binaries instead
of merely compiling unrelated header probes. Optional real adapter/tokenizer/foreign/variant
paths activate those artifact-specific cells.

### 3.3 Cost interpretation

The release-candidate selector uses caller workspace, performs no transient allocation, and
measured 0.2453%–0.2461% of a real Qwen2.5 0.5B forward step across three runs. A controlled cost follow-up used the same
DGC1 artifact, prompt, executable, and exactly 30 generated tokens, alternating arm order across
six pairs: greedy averaged 4.276667 seconds and damped greedy 4.277000 seconds (1.000078x). The
0.333 ms difference is below the CLI timer's 1 ms reporting resolution. The original Phase E
separate-run timing ratios are retained only as raw evidence; differing lengths and non-
interleaved arms make them invalid as decoder-cost measurements.

### 3.4 Final verification fill-in

The release owner fills this block from the final clean run; reviewers treat a blank value as an
open gate, never as a pass.

- Full `build.bat`: **PASS** — exit 0; 34,189 core checks, 0 failures; integrated Phase D and
  hermetic T-2138 runs green. The optional records-repo ABI inventory check skipped because the
  sandbox user could not trust that external worktree's ownership; the in-tree 36-verb inventory
  gate ran and passed.
- T-2138 complete real-artifact run: **PASS** — 567 checks, 0 failures, 0 skips across 11 cells.
- Runtime-adapter + damped composition: **PASS** — real adapter conversion produced the runtime
  artifact; Phase D2 reported 248 checks, 0 failures; production CLI loaded the adapter in both
  decode modes. At the merged candidate tip, T-2213's current converter/report population passed
  62 tests; its retired pooled quality gate is not represented as evidence.
- Real 0.5B selector ratio after the allocation fix: **PASS** — 0.2453%–0.2461%; fixed-work six-pair
  end-to-end comparison unresolved at 1 ms timer resolution (1.000078x measured ratio).
- `git diff --check`: **PASS**

### 3.5 Independent review disposition

The independent review recorded in `4c8cd2d-superslm-1p2-release-candidate.md` returned
**FIX-THEN-SHIP**. This candidate closes every filed item; the reviewer must verify this closure
before the remaining gate is marked passed.

| ID | Closure | Regression evidence |
|---|---|---|
| C1 | Replaced the vocabulary-sized vector and full-range `partial_sort` with a caller-scratch, `k`-element heap; the ABI workspace now owns the index and Q15 scratch. Anti-LM read paths and top-k renormalization also avoid transient allocation. | Damped ABI call at the 151,936-token target vocabulary records zero hot-path allocations; Phase C equivalence/adversarial cells remain green. |
| S1 | Removed the confounded Phase E ratios as a cost claim and added an alternating, fixed-work six-pair measurement. | 30 tokens per arm per pair: greedy 4.276667 s, damped 4.277000 s, 1.000078x; difference below the 1 ms timer resolution. |
| S2 | Save and restore both enforce `anti_lm_history_count <= saved context_length`; the model sizing bound remains sufficient. | Malformed history/context rejection plus state-size-capacity save/restore with live history. |
| S3 | SSB3 remains the write format; restore explicitly accepts shipped SSB2 and reconstructs its absent anti-LM state as empty. | Hand-converted SSB2 compatibility cell restores and continues. |
| S4 | Restore again treats `n` as capacity and accepts trailing bytes after the exact encoded blob. | `sslm_seq_state_size`-sized buffer restores a shorter live-state blob. |
| S5 | The v2 ABI rejects damped mode without DGC1 before forward/state mutation and validates the fixed-point scale triple at the boundary. | Missing-DGC1 and hostile-scale cells pin status and failure atomicity. |
| M1 | Corrected the public ABI count to 36. | Existing 36-verb inventory gate. |
| M2 | Both parameter-domain failures now return `InvalidDecodeParams`. | Phase D loop suite. |
| M3 | Documented the Phase D loop's input, output, and mutation preconditions beside its declaration. | Header/source contract review. |
| O1 | Added the two missing coverage shapes identified by the reviewer. | C1 and S2 regression cells above. |

### 3.6 Reproduction commands

Run from the repository root. Angle-bracketed artifact paths are supplied by the reviewer:

```bat
cmd /d /c build.bat
python -m pytest ^
  tools\test_convert_model_damped_greedy.py ^
  tools\test_sslm_convert_adapter.py ^
  tools\test_sslm_convert_adapter_bf16.py ^
  tools\test_sslm_convert_adapter_b3_diagnostic.py ^
  tools\test_sslm_convert_adapter_b3_magnitude.py ^
  tools\test_sslm_convert_adapter_b3_notice_e2e.py -q

tests\t2199-damped-greedy-red-suite\obj_green_D\phaseD2_wiring_red.exe --model=<dgc-model.sslm> --adapter=<adapter.sslm>
tests\t2199-damped-greedy-red-suite\obj_green_D\phaseD2a_cost_ratio_red.exe --model=<dgc-model.sslm>

set T2138_ADAPTER=<adapter.sslm>
set T2138_FOREIGN_MODEL=<different-shape-model.sslm>
set T2138_MODEL_VARIANT=<same-shape-different-model.sslm>
set T2138_MODEL_TOK=<model-and-tokenizer.sslm>
call tests\t2138-abi-red-suite\run_green.bat <base-model.sslm>
```

The CLI's public-default path is:

```bat
build\sslm_generate <dgc-model.sslm> <tokenizer.sslm> "<prompt>" --max-new 32 --decode-mode damped-greedy
```

## 4. Independent gates remaining

The Public Strike is complete and recorded in
[`superslm-1.2-public-strike.md`](superslm-1.2-public-strike.md). Exactly one
judgment gate remains:

1. Independent adversarial code/evidence review. This must be performed by a reviewer who did not
   author the candidate changes.

No tag, release date, or “current release 1.2” statement lands before the remaining gate passes.
This is why the changelog remains under `Unreleased` and the README calls 1.2 a release candidate.

## 5. Adversarial review brief

Review the whole `main...codex/t2199-b0-calibration` range, with extra attention to these attack
surfaces:

1. Prove legacy `sslm_decode_step` cannot read the extended struct or select damped mode.
2. Attack DGC1 flag/section/dtype/size/value combinations and header/library skew.
3. Attack `sslm_decode_params_init` output validity, failure atomicity, small-vocabulary behavior,
   and stale model handles.
4. Find a path where schema masking occurs after candidate selection, or where jump-forward and
   anti-LM history disagree.
5. Attack reset, restore, prefix adoption, adapter changes, batched sequences, and concurrent
   release for state leakage or use-after-free.
6. Compare every timing and quality claim with the exact population that produced it. In
   particular, do not let the selector microbenchmark stand in for Phase E wall time or the
   measured prose sample stand in for a universal quality claim.
7. Run the documented quickstart commands with and without `--enable-damped-greedy`; confirm the
   default path remains greedy and the opt-in path works without manually supplied alpha/n/k.

## 6. Release procedure

After both independent gates pass:

1. Resolve every finding in the candidate and rerun the affected cell plus full `build.bat`.
2. Move the 1.2 changelog material from `Unreleased` to `## [1.2.0] - YYYY-MM-DD`.
3. Change README's “1.2 release candidate” line to `Current release: **1.2**`.
4. Commit the reviewed public-strike result and release-doc finalization.
5. Merge, tag `v1.2.0`, and publish from that exact commit.
