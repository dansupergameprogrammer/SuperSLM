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

| Id | Construction | Required | Cloud (2026-10-01, at 751e441) |
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
| oracle.R8 | The same flipped pin, graded by `Pin()` itself | REJECT, "does not match its reference" | FIRED ("pinned blob … does not match its reference: sha256 …") |

Note on the pins: since 751e441 `Pin()` checks each pin's SHA-256 against the `blob=` digest of its
own `.ref` record when it loads (R8). Before that it decoded a flipped pin silently, and a corrupted
pin was caught only where a cell went on to compare the restored state with the reference (R7, which
still fires as well).

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
| probe2.R5 | `budget_edge`: the budgeted create also refuses budget ≥ cap − B with `SSLM_INVALID_ARGUMENT` (a domain check one page short). Claim k − 1 on k free, k = 256 and 258 | REJECT, "were all admitted" | FIRED ("… were all admitted or refused for a reason other than capacity (a 256-page create was refused with SSLM_INVALID_ARGUMENT (1)) …") |

History of R5: at 7132d14 the instrument ACCEPTED it, and was DEAD on this construction. Its
refusal leg counted any refusal as "at most k free". With 258 free, the claim 257 splits as
[255, 2] and is admitted. The refusal leg's [256, 2] contains a 256-page create (budget 4080),
which this build refuses as an invalid argument whatever the pool holds, so the probe reported
"exactly 257 free". Since 751e441 a refusal proves "at most" only when it is
`SSLM_KV_POOL_EXHAUSTED`, and R5 fires.

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

Since 751e441 both cells grade three ways, with a resolving power R per result:

- **7.9**, for each target, with slowdown s = 1 − paged / reference, prints two verdict lines:
  - `7.9 <target> effect: verdict NO RESULT|RESOLVED (…)`: NO RESULT when |s| < R. Never asserted.
  - `7.9 <target> bar: verdict PASS|FAIL|NO RESULT (…)`: PASS when s + R ≤ 5 %, FAIL when
    s − R > 5 %, NO RESULT otherwise. Only the bar is asserted, and only a PASS or a FAIL; a NO
    RESULT leaves the cell green with nothing asserted.

  R is the larger relative spread, (max − min) / median, of the two sides' 5 runs.
- **7.4**, for each (holder mode, verb), prints `7.4 <mode> <verb>: verdict PASS|FAIL|NO RESULT (…)`
  for the ratio cap 32768 / cap 4096 against the bound. The test is d = ratio / bound − 1: PASS when
  d ≤ −R, FAIL when d > R, NO RESULT in between. R is now taken from 5 interleaved batches of 41
  calls each.

How the runner grades them:

| Role | Graded on |
|---|---|
| reject | The asserted FAIL message (cell RED). A green cell whose bar or 7.4 line reads NO RESULT is reported as `NOT-FIRED (instrument NO RESULT)`, not as an accept. Either way it has not fired. |
| noresult | The **effect** lines only: every graded target's effect must read NO RESULT. A bar NO RESULT can come with a resolved effect (s = 24.8 %, R = 22.3 % does), so it is never taken as one. |
| accept | Cell green, plus the verdict lines a construction names (`verdicts=`). See below. |

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
| timing79.box.A1 | Unmutated pair, A/A. Run 1 writes `SUPERSLM_PAGED_KV_79_OUT`. Run 2, also unmutated, is graded against it with `..._TIMING_COMMISSIONED=1` | ACCEPT, and **every target's bar reads PASS and its effect reads NO RESULT**. See below |
| timing79.box.R1 | Unmutated baseline, then `decode_slow` f = 0.25 graded against it | REJECT: bar FAIL, "paged decode is … slower …" |
| timing79.box.N1 | Unmutated baseline, then `decode_slow` f = `-SubresFactor` (default 0.01, about 1 %). The runner reads INCONCLUSIVE when the reported R is at or below the injected effect; rerun with a smaller factor | **effect NO RESULT** on every target |
| timing74.box.A1 | Unmutated: reset and adopt on the 0.5B (cap 4096) against the 1.5B (cap 32768), bound `-Max74Ratio` (default 2.0; calibration's number, §8) | ACCEPT, every ratio PASS |
| timing74.box.R1 | `reset_capscale`, same bound | REJECT: FAIL, "reset: ratio … above the commissioned …" |

**Why timing79.box.A1 requires bar PASS.** A green cell is not enough, because a bar NO RESULT
also leaves the cell green, and it asserts nothing. The point of the bar is that an unchanged
build can meet it, so the must-accept must show that happen: bar PASS, which needs R < 5 % on the
box. It also requires effect NO RESULT, because an A/A pair whose identical builds read a RESOLVED
effect has a resolving power that understates the run-to-run noise. If the box cannot get R under
5 %, this construction reads REJECTED (REJECTS_HEALTHY through the registry). That is the
truthful outcome: the harness cannot then show the bar met on that box. Only the box can run it;
the cloud twin compares two fixtures, not a build with itself, and its R is 24–40 %.

**Cloud twins.** These use fixtures and run here, with `--allow-busy`. The cloud is noisy, and its
7.9 compares pkv_def with pkv_odd, which also carries their real difference, so its 7.9 rows are
readings and grade nothing. 7.4's cloud twin is a valid construction (two caps of one model).
Three runs each:

| Id | Construction | Observed |
|---|---|---|
| timing79.cloud.O1 | Unmutated, pkv_def (B = 16) against pkv_odd (one page) | s = −5.1 / −7.5 / 7.5 %, R = 40.0 / 37.8 / 24.2 %. Effect NO RESULT and bar NO RESULT in every run; cell green, nothing asserted |
| timing79.cloud.R1 | `decode_slow` f = 0.25 | 1 of 3 FIRED ("paged decode is 32.68 % slower … beyond the 5 % bar by more than the resolving power 27.06 %"). The other 2 read bar NO RESULT, so they did not fire: R was too large for a 20-point effect. Box only |
| timing79.cloud.N1 | `decode_slow` f ≈ 0.01 | effect NO RESULT in all 3 (s = 10.8 / 22.1 / 13.5 %, R = 32.7 / 42.5 / 31.0 %). Readings only: the twin's s is not the injected 1 % |
| timing74.cloud.A1 | Unmutated, bound 2.0 | ACCEPTED, 3/3: all four ratios PASS |
| timing74.cloud.R1 | `reset_capscale`, bound 2.0 | FIRED, 3/3: budget reset ratio 16.1 / 16.9 / 16.8, R 6.7 / 9.7 / 13.3 % |

**What remains, by the code:**

- **Only decode runs A/B.** §8 says the harness times create/reset/adopt/save/restore/release on
  the v1.9.0 (v1.11.0) binary and on the paged binary. 7.9 is the only cell that runs on both. 7.4
  compares two caps inside the paged binary. (A `c6_lifecycle_timing.cpp` cell now exists at
  751e441; it is not constructed against here.)
- **7.4 grades only against a bound.** By its own comment, it has no "effect below the resolving
  power" verdict. Its claim is the ratio's place against the bound. So no no-result construction
  is stated for 7.4.

## Commands

### Cloud (Linux)

The cloud results above were all produced with this command:

```sh
python3 tests/paged-kv/commissioning/run_commissioning.py --scratch /tmp/claude-0/b/debunk2-cm \
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
| oracle, count, probe2, probe1 | 1 (fired; probe2.R5 fires since 751e441) |
| timing79-reject-box,timing79-noresult-box | 1 on a box whose R is under about 15 % (bar FAIL for a 20 % slowdown needs s − R > 5 %) and above the injected 1 % |
| timing74-reject-box | 1 |

timing79-accept-box exits 0 only when every target reads bar PASS and effect NO RESULT. That needs
R < 5 % on the box's A/A pair.

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
| PKV-FILL-PROBE-C5 | COMMISSIONED (probe2.R5 now fires: a refusal proves "at most" only when it is SSLM_KV_POOL_EXHAUSTED) |
| PKV-TIMING-79-C6 | COMMISSIONED on a quiet box whose A/A resolving power is under 5 % (timing79.box.A1 needs bar PASS). REJECTS_HEALTHY if the box's R is 5 % or more. DEAD if R1 or N1 do not draw their verdicts on that box |
| PKV-TIMING-74-C6 | COMMISSIONED. It now reports a resolving power and grades three-way, so A1 needs every ratio PASS and R1 must FAIL on the budget reset ratio |
