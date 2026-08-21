# T-2199 Phase B0 ruling

**Ruled by Dan, 2026-08-21.**

- Greedy remains the default decoder.
- Damped greedy ships as an explicit opt-in decoder.
- The operating point carried into Phase E is `alpha=2`, `n=2`, `k=6`, with artifact scale
  `q=(493, 964, 487361)` derived from `m=2883584, e=-36`.
- Damped greedy is an anti-loop policy, not a semantic-equivalence mode. It may improve coherence
  where greedy falls into repetition, while changing otherwise valid wording, structure, list
  formatting, termination, or other model intent.
- Format-sensitive output should continue to use greedy or schema-constrained decoding unless the
  caller explicitly chooses the damped-greedy tradeoff.

## Basis

The full 63-row calibration table and genuine autoregressive confirmation are recorded in
`t2199-b0-primary.*` and `t2199-b0-decision-packet.*` in this directory.

At the ruled point, genuine 100-token rollouts produced:

| Checkpoint | Reach | Greedy locks | Damped locks | Greedy mean rep-3 | Damped mean rep-3 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B int8 | 41/48 | 3 | 0 | 0.09063 | 0.00172 |
| Qwen2.5 1.5B int8 | 43/48 | 0 | 0 | 0.01873 | 0.00087 |

The higher-strength `alpha=3, n=2, k=6` challenger achieved no additional genuine reach on the
0.5B confirmation cell, completed two fewer generations by EOS, and retained less model-governance
margin. `k=10` added selector cost without additional replay reach. These observations support the
ruled opt-in point; they do not justify changing the default decoder.

## Phase E input

Phase E confirms the production, certified implementation at the ruled operating point. Its report
must present real greedy and damped-greedy text side by side and preserve the distinction between
anti-loop effectiveness and format/semantic fidelity.
