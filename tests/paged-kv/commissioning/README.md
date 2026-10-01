# Paged-KV commissioning constructions (plan rev 16.2, §8)

The debunker's must-accept and must-reject constructions for the four test-side instruments
of the paged-KV plan. Until an instrument is commissioned, its readings are quarantined (§8). These
constructions are what `instrument-commission.ps1 -Commission` runs to lift that quarantine. They
were authored blind: from §3.1, §3.4, §8 and §10 R9 of the plan and from the instruments' own
code. The builder's self-test and the cells' mutant notes were not read.

Nothing here is part of the red suite. `paged_kv.cmake` never globs this directory.

| File | What it is |
|---|---|
| `cm_cells.cpp` | Cells that call the oracle, the legacy-create count and the two fill probes exactly as a suite cell would. They link with the unmodified runner `pkv_main.cpp`. |
| `mutants/debunk_mutants.patch` | Mutations of the code under test (`src/kv_pages.{h,cpp}`, `src/sslm_abi.cpp`, `src/forward/forward_sites.cpp`, a new `src/debunk_mutant.h`). Each one is gated on `PKV_DEBUNK_MUTANT`, so one mutant library serves every construction. With the variable unset, the mutant library behaves as the unmutated one (constructions `*.A2`). |
| `run_commissioning.py` | The runner for both platforms. It exports HEAD into `<scratch>/{pristine,mutant}/tree` and applies the patch to the mutant copy only. It appends the commissioning targets to the copy's `paged_kv.cmake`, builds, runs the constructions and grades them. `--list` prints every set. |
| `run_commissioning.ps1` | The box wrapper. It enters the developer shell, generates the fixtures if they are missing, then runs the runner. |

## How a construction is graded

Each construction is graded by the instrument's own verdict: the runner's per-cell `RED`/`green`
line, and the `FAIL` reason the instrument printed. A must-reject counts as **FIRED** only when
every selected cell is RED for its stated reason. A cell can also be green (**ACCEPTED**), RED for
another reason (**WRONG-REASON**), or never report because of a crash (**NOT-REPORTED**). None of
those three counts as fired.

The exit status follows `instrument-commission.ps1`, where non-zero means REJECTED:

| Selection | Exit status |
|---|---|
| A `*-accept` set | 0 only when every construction was ACCEPTED |
| A `*-reject` or `*-noresult` set (or both) | 1 only when every construction drew its required verdict, else 0 |

A must-reject that is accepted therefore writes the registry entry **DEAD** and never a false
COMMISSIONED. The same fail-safe applies to infrastructure failures (a failed build, no fixtures,
a box that is not quiet): they exit 0 for reject sets and 2 for accept sets. **Run every set by
hand once before `-Commission`.** Through the registry, an infrastructure failure can only read
as DEAD or REJECTS_HEALTHY.

Logs go to `<scratch>/logs/<construction>.run<N>.log`. The constructions run on HEAD, and a dirty
`src/`, `include/`, `tests/paged-kv/` or `tools/` is refused.

## The state the page-count constructions probe

The pool is a page pool on `pkv_def` (B = 16, ceil(cap/B) = 256) of `free + 7` pages:

1. A budget sequence of budget 96 holds R(96) = 7 pages for the state's whole life (§3.4).
2. One 2-page create (budget B) is made and released.

On the correct build exactly `free` pages are left. The mutants act on that release, through the
real release path:

| Mutant | Effect on the first release in each pool |
|---|---|
| `leak1` | Leaks one reserve page: a **one-page deficit**. |
| `dfree1` | Pushes one reserve page twice: a **one-page excess**, a double free. |

## 1. Byte-equality oracle (`RunScenario` + `ExpectMatchesReference`, `Pin` + `RefLookup`; v1.11.0)

Claim: "these tokens and K/V bytes equal v1.11.0's".

| Id | Construction | Required | Cloud (2026-10-01) |
|---|---|---|---|
| oracle.A1 | Unmutated build. All 8 scenarios on all 4 fixtures (2,063 checks), plus both pins (persist, saturating) restored, re-saved and continued | ACCEPT | ACCEPTED |
| oracle.A2 | Same, on the mutant library with the mutant off | ACCEPT | ACCEPTED |
| oracle.R1 | `kv_flip`: one V byte (layer 1, KV head 1, position 17, d 0) lands XOR 1 in `LandTokenKVRow` | REJECT, "K/V rows differ" | FIRED, on all 4 fixtures |
| oracle.R2 | `tok_flip`: the token returned by the step that leaves L = 102 is +1; K/V untouched | REJECT, "tokens differ" | FIRED |
| oracle.R3 | `blob_hdr`: every save flips bit 0 of blob byte 100 (`forced_token_count`); tokens and rows untouched | REJECT, "blob differs" | FIRED |
| oracle.R4 | Fixture tamper: `pkv_def.sslm` with one weight byte flipped (offset len·5/40) and its integrity hash restamped. It still maps, so this is a host that builds different fixture bytes | REJECT, "was recorded against fixture" | FIRED |
| oracle.R5 | Reference tamper: one hex digit of `rows=` in `lifecycle prefill100+decode4` | REJECT, "K/V rows differ" | FIRED |
| oracle.R6 | Reference tamper: one extra record `persist ghost` in `v1.11.0_pkv_qk.ref` | REJECT, "records, reference has" | FIRED ("4 records, reference has 5") |
| oracle.R7 | Pin tamper: `v1.11.0_pkv_def_persist_saved` with one K byte (layer 0, head 0, position 5) flipped, restored on the unmutated build | REJECT, rows/blob/tokens differ | FIRED ("K/V rows differ") |

Note on R7: `Pin()` itself decodes a flipped pin without complaint. It never checks a pin against
the `blob=` digest of its own `.ref` record. A corrupted pin is caught only where a cell compares
the restored state against the reference, as R7 does.

To run it: `--set oracle-accept`, then `--set oracle-reject`.

## 2. Legacy-create admission count (`CountLegacyCreates` at P and P − 1)

Claim: it resolves a one-page deficit at P and a one-page excess at P − 1. The procedure is §8's:
free pages = n·256 at P, expect n at P and n − 1 at P − 1, for n = 1, 2, 3.

| Id | Construction | Required | Cloud |
|---|---|---|---|
| count.A1 | Exact states, unmutated | ACCEPT | ACCEPTED |
| count.A2 | Exact states, mutant library with the mutant off | ACCEPT | ACCEPTED |
| count.R1 | `leak1` (deficit) | REJECT at P | FIRED: n − 1 at P for n = 1, 2, 3; the P − 1 leg passed (one direction only, as §8 says) |
| count.R2 | `dfree1` (excess) | REJECT at P − 1 | FIRED: n at P − 1; the P leg passed |

## 3a. Two-sided fill probe (`ProbeExactlyFree`)

Claim: "exactly k pages are free". It must reject a one-page excess, a one-page deficit and a
double free. k ∈ {2, 3, 16, 17, 255, 256, 257, 258, 511, 512, 513, 600}, with no excess form at
k = 2.

| Id | Construction | Required | Cloud |
|---|---|---|---|
| probe2.A1 / A2 | Exact states (unmutated / mutant off) | ACCEPT | ACCEPTED (12 states each) |
| probe2.R1 | Claim k + 1 on k free (deficit) | REJECT, "were not all admitted" | FIRED (12/12) |
| probe2.R2 | Claim k − 1 on k free (excess) | REJECT, "were all admitted" | FIRED (11/11) |
| probe2.R3 | `leak1` (deficit) | REJECT | FIRED (12/12) |
| probe2.R4 | `dfree1` (excess, double free) | REJECT | FIRED (12/12) |
| **probe2.R5** | `budget_edge`: the budgeted create also refuses budget ≥ cap − B with `SSLM_INVALID_ARGUMENT` (a domain check one page short). Claim k − 1 on k free, k = 256 and 258 | REJECT | **ACCEPTED: the instrument is DEAD on this construction** |

Why R5 is accepted: the refusal leg counts any refusal as "at most k free". It never checks that
the refusal was `SSLM_KV_POOL_EXHAUSTED`. With k = 258 free, the claim 257 splits as [255, 2] and
is admitted. The refusal leg's [256, 2] includes a ceil(cap/B)-page create (budget 4080), which this
build refuses as an invalid argument, whatever the pool holds. The probe therefore reports
"exactly 257 free" while 258 are free. k = 256 (claim 255: [255] admitted, [256] refused) behaves
the same way. The defect needs only one wrong refusal anywhere in the refusal leg, at any create
size the split uses.

## 3b. One-state fill probe (`ProbeExactlyFreeOneState`)

k ∈ {3, 4, 16, 17, 256, 257, 258, 513, 600}.

| Id | Construction | Required | Cloud |
|---|---|---|---|
| probe1.A1 / A2 | Exact states | ACCEPT | ACCEPTED (9 states each) |
| probe1.R1 / R2 | Claim k + 1 / k − 1 | REJECT, "one-state fill probe at k=" | FIRED (9/9, 8/8) |
| probe1.R3 / R4 | `leak1` / `dfree1` (double free) | REJECT | FIRED (9/9 each) |
| probe1.R5 | `budget_edge`, claim k − 1, k = 256, 258 | REJECT | FIRED. Its refusal step is a 2-page create, which `budget_edge` does not touch. For k = 258 it also failed step 1, because the admission leg contains a 256-page create |

## 4. Timing harness (cells 7.9 and 7.4; `c6_dim7_throughput.cpp`, `c6_dim7_reset_adopt_timing.cpp`)

Claim: "the lifecycle verb's cost and the decode rate changed by at least the reported amount",
each result with its resolving power. An effect below the resolving power is no result.

Slowdowns are injected by the `decode_slow` mutant. On a paged holder (B < cap) only,
`sslm_decode_step` spins f × its own elapsed time, so the paged decode rate is 1/(1 + f) of the
unmutated rate. f = 0.25 makes paged decode 20 % slower, 4× the 5 % bar. `reset_capscale` makes
`sslm_seq_reset` write context_cap × 64 bytes.

**Box constructions.** These need the real artifacts and a quiet box:

- no `D:\_ssu_build_lock`;
- none of UnrealEditor, cl, link, dotnet, UnrealBuildTool, ninja, cmake or msbuild running, checked
  before and after each construction;
- no other timing leg (such as `run_paged_kv_c6_box.ps1`) running.

A construction that finds the box busy reads NOT-IDLE and counts as not drawing its verdict.

| Id | Construction | Required |
|---|---|---|
| timing79.box.A1 | Unmutated pair, A/A. Run 1 writes `SUPERSLM_PAGED_KV_79_OUT`. Run 2, also unmutated, is graded against it with `..._TIMING_COMMISSIONED=1` | ACCEPT: not called a slowdown. Also needs the box's resolving power < 5 %, or the harness fails it as "cannot resolve" |
| timing79.box.R1 | Unmutated baseline, then `decode_slow` f = 0.25 graded against it | REJECT, "paged decode is … slower" |
| timing79.box.N1 | Unmutated baseline, then `decode_slow` f = `-SubresFactor` (default 0.01, about 1 %). The effect is below the run's resolving power. The runner reads INCONCLUSIVE when the harness reports a resolving power ≤ the injected effect; rerun with a smaller factor | **NO RESULT**: neither a pass nor a fail |
| timing74.box.A1 | Unmutated: reset and adopt on the 0.5B (cap 4096) against the 1.5B (cap 32768), bound `-Max74Ratio` (default 2.0; calibration's number, §8) | ACCEPT |
| timing74.box.R1 | `reset_capscale`, same bound | REJECT, "reset: ratio" |

**Cloud twins.** These use fixtures and run here, with `--allow-busy`. The cloud is noisy, so the
readings are recorded but grade nothing:

| Id | Construction | Observed (7.9: O1 and R1 4 runs, N1 3 runs; 7.4: 2 runs) |
|---|---|---|
| timing79.cloud.O1 | Unmutated, pkv_def (B = 16) against pkv_odd (one page) | slowdown 14.0 / 20.2 / 6.8 / 5.9 %, resolving power 13.6 / 49.6 / 16.8 / 44.3 %. The cell fails **both** "cannot resolve the 5 % bar" and "paged decode is X % slower", even when X is far below the resolving power |
| timing79.cloud.R1 | `decode_slow` f = 0.25 | FIRED: "paged decode is 30.6 / 34.9 / 21.6 / 33.0 % slower" |
| timing79.cloud.N1 | `decode_slow` f = 0.01 | NOT AS REQUIRED: slowdown −10.2 / −9.1 / −16.4 %, resolving power 16–35 %, reported as a FAIL ("cannot resolve"), never as no result |
| timing74.cloud.A1 | Unmutated, bound 2.0 | ACCEPTED: ratios 0.60–1.07 |
| timing74.cloud.R1 | `reset_capscale`, bound 2.0 | FIRED: reset ratio 15.6 / 17.4 (budget), 17.8 / 20.9 (whole_reserve) |

**What the harness cannot do, by its code:**

- **7.9 has no "no result" verdict.** `Grade()` has two outcomes:
  - **Pass** when `resolving < 5 %` and the point estimate `slowdown ≤ 5 %`. That includes a
    sub-resolution effect (slowdown 1 %, resolving 3 %), and an estimate of 4 % with a 4 %
    resolving power, whose upper bound of 8 % is above the bar.
  - **Fail.** When the resolving power is ≥ 5 %, the run is failed as "cannot resolve", and the
    point estimate is graded and headlined as a slowdown anyway.

  timing79.box.N1 can therefore never draw NO RESULT, and the registration below, which includes
  it, will read **DEAD** on the box. That reading is the finding, not an infrastructure fault.
- **7.4 reports no resolving power at all.** It compares a ratio of medians (201 calls) with a
  caller-supplied bound. The "each result with its resolving power" half of the claim has no
  instrument behind it for lifecycle verbs, so no no-result construction can be stated for 7.4.
- **The A/B timing is decode only.** §8 says the harness times
  create/reset/adopt/save/restore/release on the v1.9.0 (v1.11.0) binary and on the paged binary.
  Only decode tok/s (7.9) runs on both. 7.4 compares two caps inside the paged binary, so "the
  lifecycle verb's cost changed by at least the reported amount", against the reference binary,
  is measured by no cell.

## Commands

### Cloud (Linux)

The cloud results above were all produced with this command:

```sh
python3 tests/paged-kv/commissioning/run_commissioning.py --scratch /tmp/claude-0/b/debunk-cm \
    --fixtures /tmp/claude-0/b/pkv1 --set oracle-accept   # and each set of --list
python3 tests/paged-kv/commissioning/run_commissioning.py ... --allow-busy --set timing79-observe-cloud \
    --set timing79-reject-cloud --set timing79-noresult-cloud --set timing74-accept-cloud --set timing74-reject-cloud
```

### Box: run each set by hand first

Use the checkout holding this branch; `D:\SuperSLM` is shown. Scratch is `D:\_scratch\pkv-commission`.
The timing sets need the artifacts directory that `tools\run_paged_kv_c6_box.ps1` makes
(`D:\_scratch\pkv-c6\artifacts`), so run that script once, or pass `-ArtifactDir`.

```powershell
$W = 'D:\SuperSLM\tests\paged-kv\commissioning\run_commissioning.ps1'
pwsh -NoProfile -File $W -BuildOnly
foreach ($s in 'oracle-accept','oracle-reject','count-accept','count-reject','probe2-accept','probe2-reject',
               'probe1-accept','probe1-reject') { pwsh -NoProfile -File $W -Set $s; "$s exit $LASTEXITCODE" }
# quiet box only, one at a time:
foreach ($s in 'timing79-accept-box','timing79-reject-box,timing79-noresult-box','timing74-accept-box','timing74-reject-box') {
    pwsh -NoProfile -File $W -Set $s; "$s exit $LASTEXITCODE" }
```

Expected exits: accept sets 0. For the reject sets:

| Set | Expected exit |
|---|---|
| oracle, count, probe1 | 1 (fired) |
| probe2-reject | **0**: probe2.R5 accepted, DEAD |
| timing79 reject + noresult | **0**: timing79.box.N1 cannot draw "no result", DEAD |
| timing74-reject-box | 1 |

### Registration

`-Register` and `-Commission` are run from the Wizard checkout that holds the instrument registry.
The registry change is committed with that tree's own commit tool. `BuiltBy` and `ConstructedBy`
must name different seats; replace the two seat labels below with the ones the records use.

```powershell
$T  = 'D:\Wizard\Claude\Tooling\instrument-commission.ps1'
$R  = 'D:\SuperSLM'
$Run = "pwsh -NoProfile -File `"$R\tests\paged-kv\commissioning\run_commissioning.ps1`" -Scratch D:\_scratch\pkv-commission"

pwsh -NoProfile -File $T -Register -Id PKV-ORACLE-R0 -Instrument "$R\tests\paged-kv\pkv_common.h" `
  -Decides 'Whether a paged-KV run''s tokens and K/V bytes equal the v1.11.0 reference (RunScenario + ExpectMatchesReference, Pin + RefLookup).' `
  -BuiltBy 'builder-pkv' -ConstructedBy 'debunker-pkv' `
  -MustAccept "$Run -Set oracle-accept" -MustReject "$Run -Set oracle-reject" `
  -MustRejectMagnitude 'One V byte at one position, one returned token, one blob header bit, one fixture byte, one reference digest digit, one extra reference record, one pinned K byte: each the smallest difference in the decided quantity.' `
  -ProducibleBy 'A landing, gather or argmax defect in the paged engine produces each byte or token difference; a host that builds different fixture bytes, or a damaged reference or pin, produces the rest.'
pwsh -NoProfile -File $T -Commission -Id PKV-ORACLE-R0

pwsh -NoProfile -File $T -Register -Id PKV-LEGACY-COUNT-C4 -Instrument "$R\tests\paged-kv\pkv_common.h" `
  -Decides 'Whether a pool admits exactly the page count a C4 cell states, by the legacy-create count at P and P - 1.' `
  -BuiltBy 'builder-pkv' -ConstructedBy 'debunker-pkv' `
  -MustAccept "$Run -Set count-accept" -MustReject "$Run -Set count-reject" `
  -MustRejectMagnitude 'A one-page deficit (one leaked page) and a one-page excess (one page freed twice): the smallest defect in a page count.' `
  -ProducibleBy 'The page module''s real release path, mutated to leak or double-free one reserve page.'
pwsh -NoProfile -File $T -Commission -Id PKV-LEGACY-COUNT-C4

pwsh -NoProfile -File $T -Register -Id PKV-FILL-PROBE-C5 -Instrument "$R\tests\paged-kv\pkv_common.h" `
  -Decides 'Whether exactly k pages are free (two-sided fill probe, ProbeExactlyFree).' `
  -BuiltBy 'builder-pkv' -ConstructedBy 'debunker-pkv' `
  -MustAccept "$Run -Set probe2-accept" -MustReject "$Run -Set probe2-reject" `
  -MustRejectMagnitude 'A one-page deficit, a one-page excess and a double free, as claims off by one and as release mutants; and a one-page excess on a build whose budgeted create refuses a full-cap budget for a non-capacity reason.' `
  -ProducibleBy 'The page module''s real release path (leak, double free) and the budgeted create''s domain check (off by one page) in the code under test.'
pwsh -NoProfile -File $T -Commission -Id PKV-FILL-PROBE-C5

pwsh -NoProfile -File $T -Register -Id PKV-FILL-PROBE-ONESTATE-C5 -Instrument "$R\tests\paged-kv\pkv_common.h" `
  -Decides 'Whether exactly k pages are free on one live, non-rebuildable state (ProbeExactlyFreeOneState).' `
  -BuiltBy 'builder-pkv' -ConstructedBy 'debunker-pkv' `
  -MustAccept "$Run -Set probe1-accept" -MustReject "$Run -Set probe1-reject" `
  -MustRejectMagnitude 'A one-page deficit, a one-page excess and a double free; and a one-page excess on the budget-domain mutant.' `
  -ProducibleBy 'The page module''s real release path and the budgeted create''s domain check, mutated.'
pwsh -NoProfile -File $T -Commission -Id PKV-FILL-PROBE-ONESTATE-C5

pwsh -NoProfile -File $T -Register -Id PKV-TIMING-79-C6 -Instrument "$R\tests\paged-kv\c6_dim7_throughput.cpp" `
  -Decides 'Whether CPU paged decode is more than 5% slower than the reference, with its resolving power; an effect below it is no result (cell 7.9, R9).' `
  -BuiltBy 'builder-pkv' -ConstructedBy 'debunker-pkv' `
  -MustAccept "$Run -Set timing79-accept-box" -MustReject "$Run -Set timing79-reject-box,timing79-noresult-box" `
  -MustRejectMagnitude 'Paged decode slowed to 1/1.25 of its rate (20%, four times the 5% bar); and a 1% slowdown below the run''s resolving power, which must read no result.' `
  -ProducibleBy 'A per-page attention cost in the paged decode path (R9), injected as a self-timed spin on the paged holder''s decode step.'
pwsh -NoProfile -File $T -Commission -Id PKV-TIMING-79-C6

pwsh -NoProfile -File $T -Register -Id PKV-TIMING-74-C6 -Instrument "$R\tests\paged-kv\c6_dim7_reset_adopt_timing.cpp" `
  -Decides 'Whether reset and adopt scale with the cap (cell 7.4: cap-32768 / cap-4096 median ratio against the commissioned bound).' `
  -BuiltBy 'builder-pkv' -ConstructedBy 'debunker-pkv' `
  -MustAccept "$Run -Set timing74-accept-box -Max74Ratio 2.0" -MustReject "$Run -Set timing74-reject-box -Max74Ratio 2.0" `
  -MustRejectMagnitude 'A reset that writes cap x 64 bytes: about 8x more work at cap 32768 than at cap 4096, the old whole-block cost shape.' `
  -ProducibleBy 'A reset that touches cap-sized memory (the pre-paging memset) in the code under test.'
pwsh -NoProfile -File $T -Commission -Id PKV-TIMING-74-C6
```

Expected `-Commission` outcomes, given the cloud results:

| Entry | Expected |
|---|---|
| PKV-ORACLE-R0, PKV-LEGACY-COUNT-C4, PKV-FILL-PROBE-ONESTATE-C5 | COMMISSIONED |
| PKV-FILL-PROBE-C5 | **DEAD** (probe2.R5) |
| PKV-TIMING-79-C6 | **DEAD** (timing79.box.N1), and REJECTS_HEALTHY instead if the box's own resolving power is ≥ 5 % on the A/A pair |
| PKV-TIMING-74-C6 | COMMISSIONED only for "a cap-proportional reset is flagged". It carries no resolving power, so the claim's resolving-power half stays unsupported |
