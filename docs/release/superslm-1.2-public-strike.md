# SuperSLM 1.2 Public Strike

Date: 2026-08-21

Verdict: **PASS after corrections**

## Population

The strike inspected every 1.2 public claim and documented command in:

- `README.md`
- `CHANGELOG.md`
- `docs/api.md`
- `docs/quickstart.md`
- `docs/platform-support.md`
- `docs/sslm_format.md`
- `docs/release/superslm-1.2-review-packet.md`

Claims were checked against the Phase B0 and Phase E records, final Phase D summaries, the
zero-skip T-2138 real-artifact run, production converter/CLI execution, and the artifact/ABI
implementation at the candidate tip. The strike also checked that 1.1 remains named as the
current release and that 1.2 remains `Unreleased` pending independent adversarial review.

## Findings resolved

1. README said to use greedy for “exact format/style continuity.” Greedy is the unchanged
   baseline but does not guarantee style or formatting. The text now says to *prefer* greedy when
   learned continuity matters, and reserves validity guarantees for compiled schemas.
2. Quickstart described `top_k=6` without the small-vocabulary clamp and did not say that
   `--alpha-q15` takes Q15 units. It now documents `top_k=min(6, vocab_size)` and maps
   `alpha=2` to `--alpha-q15 65536` explicitly. The CLI was aligned with the ABI initializer so
   an omitted `--top-k` clamps while an explicitly invalid override is still rejected.
3. Platform support said the review packet carried verification commands, but the packet only
   carried results. The packet now includes the root build, Python regression, Phase D2/D2a,
   real-artifact T-2138, and public CLI command shapes.
4. After current `main` was integrated, the packet still called the old B3 pooled accept/reject
   result evidence. T-2213 retired that non-discriminating gate. The packet now names the real
   artifact conversion separately from the merged-tip 62-test converter/report population and
   explicitly refuses to treat the retired gate as evidence.

## Claims deliberately retained

- “Often/many outputs read more coherently” remains explicitly scoped to the measured corpus and
  is paired with the `list_primed_00` formatting counterexample. It is not a universal quality or
  semantic-fidelity claim.
- The original Phase E separate-run timing ratios remain raw evidence, not a cost claim. The
  fixed-work alternating-arm follow-up measured 1.000078x over six 30-token pairs; its 0.333 ms
  mean difference is below the CLI timer's 1 ms reporting resolution. The allocation-free
  selector measured 0.2453%–0.2461% of a real 0.5B forward step across three runs.
- Greedy remains the runtime and conversion default. Damped greedy requires a DGC-enabled
  artifact and explicit mode selection.

## Remaining release gate

Independent adversarial code/evidence review is still required. Until it passes, the changelog
stays under `Unreleased`, README continues to name 1.1 as current, and no `v1.2.0` tag is created.
