# Brief: S2, JSON arrays on a real model (old box, CPU and GPU)

2026-10-25 · conductor (cloud thread "JSON array support") · seat: the builder, on DESKTOP-T2JJTNM

## Why

S1, the array-capable schema converter and the `sslm_attach_schema.py` command, is built and code-reviewed in the cloud. The array suite passes 1378 of 1378, and every schema that works today compiles byte-identically. Plan of record step S2 proves it on a real artifact, which only a box can do.

S2 is in the multi-command plan, §8 S2 row (Wizard `Claude/Plans/superslm-multicommand-array/plan.md`): "Use the general command to serialize the IntentDoc v2 MaskPages into an actual complete .sslm; the independent verifier and both CPU/GPU loaders accept it. Generate across CPU and GPU on a real 0.5B int8 artifact, verify accepts/dead ends, token parity and saves/restores at every array and nested-array position. Measure actual section bytes, converter peak memory, map time and GPU residency."

This feature ships as engine 1.19.

## Where

- **Repository:** `D:\SuperSLM`. Run `git fetch origin claude/project-thread-mq1of4`, then work in a worktree at `D:\SuperSLM\.worktrees\array-s2` on that branch. The tip is `07c448fb` (v1.16.0 main merged in).
- **Scratch:** `D:\_scratch\array-s2`. Put anything a committed test reads under `D:\_artifacts\SuperSLM\array-s2`.
- **Hardware:** the only GPU is the RTX 2080 SUPER, adapter 0. There are no AMD legs here, so don't raise that as a question.
- **Shared-branch rules:** never rebase, reset, amend or force-push `develop`, `master` or `main`. Commit only to `claude/project-thread-mq1of4`, by fast-forward pushes of your own commits. Every commit message carries `[skip ci]`.
- **Box rules:**
  - Keep each step short (a few minutes), resumable, and saved as it goes. A crash should cost only the step in progress.
  - Before timing anything, check that no other session's run is using the box.
  - When you finish, stop every background task you started.

## Steps

1. **Build.** Build the Release CPU and GPU engine, `sslm_verify` and the test targets with the repository's own Windows build. Record the build command.
2. **Choose a base artifact.** Use a Qwen2.5-0.5B-Instruct int8 `.sslm` that has no SchemaMasks section. The attach command refuses a base that already has one. If none exists under `D:\hf_cache\superslm_artifacts\`, convert one from `D:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775` with the repository's converter. Record the path and SHA-256.
3. **Attach.** Run `python tools\sslm_attach_schema.py --base <base> --ckpt <that snapshot> --schema intents:tests\t3094-multicommand-s0\v2\intentdoc-v2.schema.json --schema intent_gate:tests\t3094-multicommand-s0\v2\g1.schema.json --out D:\_artifacts\SuperSLM\array-s2\qwen05-intentdoc-v2.sslm`.
   - Record wall time, converter peak memory, and the per-schema `resource_counts` (states, transitions, section bytes).
   - The cloud measured IntentDoc v2 at 635 DFA states, 3,074,805 transitions, 36,660,995 section bytes and about 165.5 MB tracked.
4. **Write a runtime harness in `tests/array-s2/`.** Use the public C ABI, as the existing schema tests do; see `docs/api.md` → schema-constrained generation. The harness covers:
   - (a) **Map:** map time, and GPU residency of the schema section.
   - (b) **Teacher-forced walks** of fixed IntentDoc v2 documents tokenized with the pinned tokenizer. Cover at least: one intent with `conditions: []`; one intent with 1 condition; three intents with 0, 2 and 1 conditions. Each walk must be accepted token by token and end accepting.
   - (c) **Rejections:** a trailing comma, `"intents":[]`, and a missing field must each be refused at the first bad token, on CPU and on GPU.
   - (d) **Constrained greedy generation, CPU and GPU.** Use about 10 prompts made from `tests\t3094-multicommand-s0\v2\frozen-examples.jsonl` utterances, with the baseline prompt from the same folder. Never open anything named `sealed`. Every output must parse as JSON and validate against IntentDoc v2. The CPU and GPU token streams must be identical.
   - (e) **Save and restore** at every array position: after `[`, after each `,` between intents, inside `conditions` after `[` and after `,`, and after each `]`. A restored sequence must continue with the same tokens as the unbroken run, on CPU and GPU.
   - (f) **Engine features on an array schema:** jump-forward in canonical and exact modes, and the masked finish. Each must give the same documents as plain decode, or report a difference with its cause. These features were built before arrays existed: a cyclic DFA is new to them.
5. **Mutant check.**
   - Build two bad artifacts by patching a scratch copy of the attach command: one shifts an accepting-state id by one, and one swaps two state offsets in the SCM1 section.
   - The verifier, step (b), or step (d) must refuse each bad artifact, or fail on it. Record which one caught it.
6. **Results.** Commit the harness and a results file, `tests/array-s2/RESULTS.md`, then push the branch. The results file holds:
   - the build;
   - the artifact hashes;
   - the measurements;
   - each step's pass or fail, with counts;
   - the mutant kills.

## Report back

Reply to this thread's cloud session with send_message, in under 30 lines:
- the commit;
- for each step: pass or fail, with numbers;
- each mismatch, with its first differing token and position;
- anything you couldn't do.

Stop at the first real defect in the engine or the converter: record it, push what you have, and report. Don't fix engine code in this session.
