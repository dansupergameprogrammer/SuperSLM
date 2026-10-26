# GB rules, engine 1.17.0 (schema fast-decode plan rev 1.4.5 Sec. 10: GB2, GK2 and the RTX 5060's exactness step)

Frozen 2026-10-25, before any GB2-shaped run on any card. Its sha256 is recorded with every reading graded against
it. A change means a new file with a new name, never an edit to this one.

## Inputs
- Model: the 0.5B potion artifact named by the run (`qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm`); its sha256 is
  recorded beside the CPU and GPU streams, and both sides must name the same sha256.
- Prompts: B7b's 41: B2's prompt ("Hello! Could I get two health potions, please?") and the 40 lines of
  `potion-population-v1.jsonl` (sha256 d846b9e301bdd4369ff3fc51fca43af6c06de0acd97cb09489eaac107e5a67da), in that
  order. Schema `potion_shop_order`. `max_new` 64.
- CPU reference: GB2c's streams (`GB2c-b2.json`, `GB2c-pop.json`), CPU off and CPU canonical, one thread, made at the
  same engine commit as the GPU binaries, or at a commit whose CPU decode path is unchanged from it.

## Rules (per card, per prompt, token for token)
1. **Off.** The GPU off stream equals the CPU off stream.
2. **Canonical.** Every GPU canonical arm equals the CPU canonical stream: head on the device at `chunk_budget` 1, 2
   and 4; head on the host at 1.
3. Equality is of the whole token stream, including its length and its stop. A digest match stands for stream
   equality only when the digest is over the full stream.
4. **Verdict.** All 41 prompts pass both rules in every arm: UN-REFUTED on that card. Any mismatch: REFUTED on that
   card, with the first differing prompt, arm and token index reported; the release waits for a ruling (plan K19). A
   step that timed out, crashed or is incomplete is NO EVIDENCE, never a pass.
5. Diagnostics only, never graded: pool statistics, timings, the debug layer's value (a card without Graphics Tools
   runs at 0 and records layer-off, plan K23).
6. **Scope of the claim.** Stream equality is claimed only on the cards where this file graded a complete run, named
   by adapter, vendor/device id and driver version. No determinism is claimed for any other GPU.
