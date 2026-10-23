# 1.15.0 box run: results (2026-10-23, home box, 3950X)

The run follows `../README.md` with `run-1p15-box.ps1`, unchanged, on `claude/sfd-1p15` at `5c83f65e`. Its engine
paths are identical to `e82d7d40`, and the tree is clean. `run-wrap.ps1` loads the BuildTools x64 environment,
MSVC 19.44.35214 (VCToolsVersion 14.44.35207), the toolset of B7's build, and then calls the driver. The driver does
not set up MSVC itself. Scratch is `D:\_scratch\sfd-1p15`. The version bump patch was not applied.

| Step | Result |
|---|---|
| build | PASS: exit 0 |
| pinmap | PASS: `GemmInt8AccumulateCols` mod64=0, `DotRowAvx2` 16, `TiledGemmAvx2` 32; `PIN OK` |
| b7b-t1 | PASS: `41 prompts, 123 exact arms, 0 not identical to off`; B2 off digest `7e46330608d11704`, 28 tokens |
| b7b-t4 | PASS: the same line |
| abp | PASS: 7-token one-thread prefill, ref 148.2545 ms, release 148.2068 ms, release/ref **1.0008 [0.9971, 1.0088]** (band 0.97–1.03). The ref bench is `fcba88db…`, B7's, on disk; the release bench is `f6cc7d61…` |

The driver ended with `end PASS`, exit 0, between 12:47 and 13:00 PDT.

**Box load** (`load-snapshots.txt`): snapshots were taken during B7b, at the start of abp, during abp and after it.
- No SuperEmbedder binary, UnrealEditor or build tool was running.
- The only heavy process was the run's own bench, and total CPU was 0.5–9.8%.
- The background apps (Creative Cloud, Spotify, TourBox) were idle.
- The A/B reading is therefore taken on a quiet box and stands.

Files: `status.txt`, `pinmap.log`, `b7b-t1/summary.json`, `b7b-t4/summary.json`, `abp/summary.json`,
`load-snapshots.txt`, `run-wrap.ps1`, `wrap-console.txt`.
