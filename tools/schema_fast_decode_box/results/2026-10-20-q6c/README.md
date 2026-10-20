# Schema fast-decode §16 Q6 C re-time, 2026-10-20

The span readings at the commit that splits short jump-forward spans (`36fe4b37d6d6cd715182884d6b22d43a68682cdb`,
"split only where a one-row pass threads"), and the off/on digest check, on the same box and model as
`../2026-10-20/`.

- **Box:** AMD Ryzen 9 3950X, Windows 10, High performance. The bench was rebuilt at 36fe4b3 (bench sha256
  `8c9c69fc…`, library `5b3692d2…`).
- **Model:** `qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm`, sha256 `9ccf7e378bbe47aa0ffd3810aa82875926cbe51af1bdfa08393c3fae657463da`.
- **Runs:** one bench process per part, 45 s apart, 15 pairs after one warm-up (`run-q6c.ps1`, `run-status.txt`).
  The first driver process died after part 1 had written its file (its status log was held open by a reader);
  the rerun skipped part 1 and ran parts 2-6. No part ran twice.

## What the bench can and cannot measure here

The split lives only in `sslm_seq_jump_forward` (`split_max_rows = 7`). `pass` times `sslm_prefill`, which the
change does not touch: it is the batched reference, and the path the verb still takes for a span of 8 or more.
`spans` times the verb itself on the spans the example prompt produces (M = 3, 5, 6, 5, 2), against the same tokens
as M one-token threaded admissions. No bench mode drives the verb at an arbitrary M, so M = 4 and 7 through the
verb are not measured.

## Readings (median, n = 15 sign interval)

**The verb against one-token admissions (`spans`), D = verb − one-token, ms.**

| Span M | 4 tasks: verb / one-token | D [interval] | 2 tasks: verb / one-token | D [interval] |
|---|---|---|---|---|
| 3 | 63.3 / 63.6 | +0.06 [−1.46, +3.18] | 68.9 / 69.8 | −0.88 [−1.76, +0.39] |
| 5 | 105.7 / 104.7 | +0.90 [−5.02, +3.47] | 113.3 / 114.3 | −1.28 [−2.44, +0.96] |
| 6 | 124.9 / 124.0 | −0.81 [−2.62, +4.05] | 137.6 / 136.7 | +0.27 [−0.47, +1.46] |
| 5 | 103.7 / 104.1 | −0.89 [−3.02, +1.08] | 113.3 / 114.4 | −0.59 [−3.38, +0.93] |
| 2 | 40.6 / 41.6 | −0.71 [−3.23, +0.02] | 45.4 / 45.8 | −0.13 [−0.46, +0.29] |

Every interval contains 0. At 79e5b92 the same spans at 4 tasks cost +5.3 to +8.9 ms more than one-token admissions,
every interval above 0 (`../2026-10-20/B3.json`).

**The batched pass (`pass`, the prefill path), D = batched − M one-row, ms.**

| M | 4 tasks: batched / one-row | D [interval] | 2 tasks: batched / one-row | D [interval] |
|---|---|---|---|---|
| 1 | 21.2 / 21.6 | +0.3 [−0.8, +1.2] | 23.2 / 23.1 | −0.0 [−0.3, +0.5] |
| 2 | 57.1 / 44.1 | +12.8 [+10.7, +15.8] | 56.5 / 46.4 | +10.5 [+9.6, +11.0] |
| 3 | 80.6 / 64.4 | +17.2 [+14.1, +18.2] | 81.6 / 69.0 | +12.7 [+11.3, +13.2] |
| 4 | 105.9 / 83.7 | +22.2 [+15.9, +24.2] | 104.1 / 91.6 | +12.5 [+11.7, +14.9] |
| 5 | 130.1 / 105.4 | +25.5 [+21.8, +27.2] | 129.1 / 114.1 | +15.2 [+14.6, +18.1] |
| 6 | 158.0 / 130.3 | +27.5 [+21.1, +29.8] | 154.1 / 137.0 | +16.5 [+16.1, +19.7] |
| 7 | 179.2 / 153.1 | +27.1 [+24.9, +31.6] | 178.9 / 159.9 | +19.5 [+17.5, +22.7] |
| 8 | 132.3 / 169.0 | −36.3 [−41.1, −34.3] | 131.4 / 181.6 | −50.1 [−50.5, −49.4] |
| 9 | 151.9 / 193.1 | −40.0 [−48.3, −35.5] | 146.8 / 204.9 | −57.9 [−61.4, −56.4] |

At 2 and 4 tasks a batched pass of 2-7 rows loses to one-row threaded passes, and 8 or more wins by 36-58 ms, so
k = 7 holds at both task counts.

**Off/on (`extract`, the example order), generation ms and digests.**

| | off | on | off / on [interval] |
|---|---|---|---|
| 1 thread | 904.8 | 547.7 | 1.650 [1.640, 1.664] |
| 4 tasks | 563.7 | 422.1 | 1.338 [1.325, 1.390] |

Digests are equal in every run: off `7e46330608d11704`, on `8ad5ea7961656cef`, the same as at 79e5b92.

**Open: the on arm at 1 thread.** It took 547.7 ms here against 470.7 ms at 79e5b92 (`../2026-10-20/B2.json`), with
the off arm unchanged (904.8 against 924.6) and the same digest. At 1 thread no hook is installed, so `MatvecThreading`
returns no row splitter and the split cannot engage; the code path is the one 79e5b92 ran. The readings are from two
sessions, before and after a reboot, not one interleaved comparison, and the box moved between them on code that did
not change: the prefill path's D at 4 tasks for M = 2..7 was +5.4 to +8.6 ms in `../2026-10-20/B3.json` and is
+12.8 to +27.5 ms here. A same-session A/B of the two commits' benches would settle it.

## The same-session A/B of the on arm at 1 thread (`ab/`)

The open item above, settled in one session: B2's jump-forward-on extraction at 1 thread, the bench built at 79e5b92
(fresh build, `sfd-ab-79e5b92/build-box`, sha256 `4fcf3c76…`) against the bench built at 36fe4b3 (sha256 `8c9c69fc…`),
one process per arm per pair after one discarded warm-up pair, the first arm alternating, 15 measured pairs, pooled by
the bench's own `pool` mode (`ab/ab-on-t1.json`, `ab/run-ab.ps1`).

| | 79e5b92 | 36fe4b3 |
|---|---|---|
| on, generation ms, median [interval] | 453.4 [453.0, 453.9] | 548.1 [547.2, 550.6] |
| prefill ms, median | 518.2 | 524.8 |
| digest | `8ad5ea7961656cef` | `8ad5ea7961656cef` (all runs equal) |

Ratio 79e5b92 / 36fe4b3 = 0.827 [0.823, 0.829]: the on arm at 1 thread is about 21 % slower at 36fe4b3, with the same
tokens. This is a regression, not drift between sessions. The source does not explain it: at 1 thread the bench installs
no hook (`tasks < 2`), so `MatvecThreading` returns no row splitter, `pass_rows` equals `admit_count`, and
`PrefillWholeTokensImpl` makes the same single batched call 79e5b92 made. One difference between the two binaries is not
yet excluded: the 36fe4b3 bench is an incremental build over the 79e5b92 tree's `build-box`, the 79e5b92 bench a fresh
build. A clean rebuild of 36fe4b3, measured against the same 79e5b92 bench, would separate a build artefact from a code
effect.

### Clean rebuild and bisect (`ab/ab-clean/`, `ab/ab-bisect/`)

Same measurement, run by `ab/run-abn.ps1`: each arm a commit with its own worktree and a fresh build folder (an existing
folder is refused), except the 79e5b92 bench above, which was itself a fresh build. One process per arm per round, the
arm order rotated per round, 15 rounds after one discarded warm-up. Every process was passed the same model explicitly
and logged it: `D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm`, sha256
`9ccf7e378bbe47aa0ffd3810aa82875926cbe51af1bdfa08393c3fae657463da`, 24 layers, tasks 0 (32 runs in each two-arm A/B,
48 in the bisect).

| Arm | Build | Bench sha256 | on, generation ms [interval] | 79e5b92 / arm [interval] | Digest |
|---|---|---|---|---|---|
| 79e5b92 | fresh | `4fcf3c76…` | 455.9 [451.5, 457.0] (clean A/B); 452.5 [451.7, 453.9] (bisect) | 1 | `8ad5ea7961656cef` |
| 36fe4b3 | fresh | `57949688…` | 548.9 [546.0, 549.9] | 0.831 [0.824, 0.836] | `8ad5ea7961656cef` |
| b75b5a6 (red cells only) | fresh | `b3da4adc…` | 452.5 [451.8, 453.9] | 1.001 [0.998, 1.002] | `8ad5ea7961656cef` |
| d7ec81b (the split) | fresh | `ce1e3ff6…` | 545.1 [543.9, 546.5] | 0.831 [0.830, 0.831] | `8ad5ea7961656cef` |

The slowdown is not a build artefact (a fresh 36fe4b3 is as slow as the incremental one) and not the model (one file,
logged by every run). It enters at d7ec81b, the commit that adds the split to `PrefillWholeTokensImpl`; b75b5a6, its
test-only parent, is as fast as 79e5b92, and 36fe4b3 keeps it. At one thread the bench installs no hook, so by the source
both d7ec81b's condition (`threading.row != nullptr`) and 36fe4b3's keep `pass_rows = admit_count` and make one batched
call, as before; the off arm and the prompt prefill, which also run `PrefillWholeTokensImpl`, are unchanged. So on this box
the jump-forward-on arm costs about 93 ms more from d7ec81b on, with the same tokens, by a cause the source reading does not
show. Not yet examined: whether a span really takes the one-row passes at one thread (the `SfdLogLayerRows` record would
show the rows per pass), and the generated code of `PrefillWholeTokensImpl` before and after.
