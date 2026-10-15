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

## 4. Current C6 timing instruments

The 7.4 cell is `=7.4/C6` on the box and `=7.4/C6:fixtures` for a same-geometry fixture construction. Both compare cap 4096 with cap 32768 on the **same model**. They grade 205 paired calls for each holder mode and verb at the fixed ratio bound 2.0, reporting `PASS`, `FAIL`, or `NO RESULT` with ranks `r(86)` and `r(120)`. A `FAIL` is a standing failure even if another verb is unresolved. A timer vitality refusal is diagnostic 7.4t output and does not block the direct 7.4 grade (D-SLM8226).

`run_commissioning.py --list` exposes only the current `timing74-{accept,reject}-{cloud,box}` constructions. The reject construction mutates the engine reset path through `reset_capscale`; the runner expects the actual `ratio above 2.0` failure. The box constructions require the two 0.5B artifact names in one `--artifacts` directory. They are timing runs and require a quiet box. No box timing verdict was obtained in T-3114.

7.9 and 7.9b use `tools/run_paged_kv_pairs.py` and `superslm_pkv_pair.exe` linked to the production library. The retired `7.9/C6` and `7.9/C6:fixtures` cells and `c6_dim7_throughput.cpp` are not commissioning targets. The pair runner requires a successor receipt for `PKV-TIMING-79-C6-R168` or `PKV-TIMING-79B-C6`, respectively, before any 508-pair result stands. A short `--mechanism` run remains ungraded.

### Independent source binding

A successor receipt entry must include `source_binding` with `manifest_sha256`, `git_head`, `git_tree`, `build_recipe_sha256`, `verifier_sha256`, and `measurement_env`; a loaded 7.9 entry also requires `load_probe_sha256`. The independent commissioner supplies this JSON claim at registration with `instrument-commission.ps1 -Register ... -SourceBindingJson PATH`. Register after the verifier containing that parameter reaches the official `D:/Wizard/Claude/Tooling/instrument-commission.ps1` path, because `verifier_sha256` pins its exact bytes. The commissioner records these values independently after building and executing must-accept and must-reject constructions. `manifest_sha256` is the SHA-256 of the exact `<exe>.source-set.json` bytes. `build_recipe_sha256` is the SHA-256 of canonical JSON (`sort_keys=True`, separators `(',', ':')`) for its `build_recipe` object. `measurement_env` is the exact environment map selected by `tools/verify_paged_kv_binding.py`. The checker compares every source with Git at the recorded HEAD, requires a clean tracked tree, checks compiler identity and all Ninja compile/link commands, and hashes every invoked executable, library and script. A relink or any measurement hook change requires a new independently commissioned successor receipt. The link manifest alone never confers standing.

For a loaded 7.9 construction, the same source binding also pins `load_probe_sha256`. The CLI load generator requires `--calibration-probe`; its record carries that probe path and hash, and the pair runner checks the live file against the independent receipt before a loaded result can stand.

The eventual commissioning attestation covers the registered claim alongside the construction commands and instrument hash.

The box runner's `timing74` and `lifecycle` names map to successor IDs ending `-R168`. The old IDs remain historical and cannot make the current legs stand. Lifecycle has no performance bar; its fixed 64 ms reset mutant reports an observed transfer ratio against the old baseline, never a guaranteed rejection sized from that baseline. The historical `lifecycle-v1.11.0` binary is cited; it has no source-set manifest and must not be rerun as a standing leg.
