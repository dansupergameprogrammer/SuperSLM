# MSVC: what moved the one-thread 7-token prefill, and the pin, 2026-10-21

## What MSVC does differently

The bisect (`../2026-10-21-prefill7-bisect/`) found the one-thread 7-token prefill at about 145 ms or about 185 ms,
flipping on commits that do not change the path. It is code **placement**, not code generation:

- **The kernel's code is the same.** Disassembling `forward_sites.cpp.obj` (dumpbin /disasm) in a fast build
  (a4692482) and a slow one (f4f69efb): the batched kernel is 2330 instructions in the fast build and 2339 in the slow
  one's `<false>` instantiation (the J1-X template; `<true>`, the exact path, is 2368). It calls the same compute
  functions; the only differences are a few vector-allocation helpers taken out of line. Every other function in the
  file has the same instruction count in both builds. `sslm_abi.cpp`'s prefill path (`PrefillWholeTokensImpl`) does
  not change between the two commits at all.
- **Below 8 tokens the GEMM runs the DotRow loop.** `GemmInt8Accumulate` takes `GemmPath::kDotRowLoop` when
  `num_tokens < kTiledMinTokens` (8) and the tiled kernel from 8 up: a 7-token pass runs the DotRow loop, a 16-token
  pass the tiled kernel, which is why the 16-token row did not move.
- **The DotRow loop moves by 16 bytes mod 64 with the size of the code linked before it.** Relinking each bench with
  CMake's own link line plus `/MAP` (`run-maps.ps1`): within `matmul.obj` every function keeps its relative offset,
  but the block lands 16 bytes later mod 64 in the slow builds.

| Function | fast builds (a4692482, 2af41659), address mod 64 | slow builds (f4f69efb, bfdec2a6), mod 64 |
|---|---|---|
| `DotRowAvx2` | 16 | 32 |
| `GemmInt8AccumulateCols` (the DotRow loop) | 0 | 16 |
| `GemmInt8AccumulateRow` | 0 | 16 |
| `TiledGemmAvx2` | 32 | 48 |

`matmul.cpp` is the same in all of them; what changes is how much code from other files the linker places ahead of
it. MSVC has no loop- or function-alignment switch (cl /? lists none; the linker aligns whole image sections only).

## The pin (branch `claude/msvc-batched-pin`, from f4f69efb)

`src/matmul.cpp`, MSVC only (`#if defined(_MSC_VER) && !defined(__clang__)`):

1. All of the file's code goes in its own image section, `.sslmgem` (`#pragma code_seg`). An image section starts on a
   page boundary, so the kernels' placement depends on `matmul.cpp` alone.
2. A 16-byte head function in `.sslmgem$a`, which the linker sorts ahead of `.sslmgem$m` (the kernels), puts the
   kernels at the placement every fast build had. It is never called; `/INCLUDE` keeps it.

Step 1 alone fixed the placement at `DotRowAvx2` 0 mod 64, which is slow (`pin-section-only/`); step 2 moves it to 16.
After step 2 the map shows `ActiveGemmTier` 32, `GemmInt8AccumulateCols` 0, `DotRowAvx2` 16, `TiledGemmAvx2` 32 mod 64,
identical to the fast builds (`run-pinmap.ps1`). The diff is `pin.patch` (34 lines).

## Readings (one thread, aex-pr `9ccf7e37…`, MSVC 19.44 Release, fresh builds, 15 interleaved rounds)

| Arm | Row | ms | arm / 2af41659 [interval] | Digests |
|---|---|---|---|---|
| section only (83bcc050) | jump-forward on | 550.0 | 1.211 [1.205, 1.213] | `8ad5ea7961656cef` |
| section only (83bcc050) | 7-token prefill | 177.9 | 1.227 [1.224, 1.233] | — |
| **pin (43e06b25)** | **jump-forward on** | 457.5 | **1.001 [0.998, 1.014]** | `8ad5ea7961656cef` (= A) |
| **pin (43e06b25)** | **7-token prefill** | 145.8 | **1.002 [0.997, 1.008]** | — |
| pin (43e06b25) | 16-token prefill | 237.8 | 1.005 [1.002, 1.012] | — |
| bfdec2a6 | 7-token prefill | 185.5 | 1.281 [1.280, 1.287] | — |
| bfdec2a6 + the same pin | 7-token prefill | 144.8 | 1.002 [1.001, 1.003] | — |
| 83a82c20 (decode-only bench) | 7-token prefill | 185.1 | 1.279 [1.277, 1.282] | — |
| 83a82c20 + the same pin | 7-token prefill | 144.5 | 1.000 [0.997, 1.001] | — |

Both rows reach a ratio of 1.01 or less at f4f69efb with the pin, and the same patch, applied to TF (bfdec2a6) and to
the precache tip (83a82c20), takes each from about 1.28 to 1.00. The pinned placement is fixed by `matmul.cpp`'s own
content, so a later edit to `matmul.cpp` itself can still move it; the map check (`run-pinmap.ps1`) shows where it
lands. The patched bfdec2a6 and 83a82c20 builds were local commits for this check and are not pushed.
