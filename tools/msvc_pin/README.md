# MSVC GEMM placement pin

Under MSVC, `src/matmul.cpp` puts all of its code in its own image section, `.sslmgem` (`#pragma code_seg`), with a
16-byte head function, `superslm_gemm_section_head`, in `.sslmgem$a`, which the linker sorts ahead of the kernels in
`.sslmgem$m`. `/INCLUDE` keeps the head; nothing calls it. An image section starts on a page boundary, so the GEMM
kernels' placement depends on `matmul.cpp` alone, not on how much code other files put ahead of it. Other compilers
align loops themselves and do not see the pin.

## The check

`run-pinmap.ps1 -Worktree <source tree> -BuildDir <MSVC Ninja build directory> [-MapDir <dir>]` builds
`sslm_decode_threading_bench`, relinks it with `/MAP`, and reads each kernel's address mod 64. That bench drives the
C ABI's prefill and decode, so it runs the whole forward pass and links the forward kernels of `matmul.cpp`, as the
bench the pin was measured on did. A narrower binary (a GEMM-only bench) loses functions to `/OPT:REF` and lays the
section out differently. The pinned placement is:

| Function | address mod 64 |
|---|---|
| `GemmInt8AccumulateCols` (the DotRow loop) | 0 |
| `DotRowAvx2` | 16 |
| `TiledGemmAvx2` | 32 |

It exits 0 when all three match, and 1 when one moved or is missing from the map.

**These offsets still need a box run on this binary.** They were read on 2026-10-21 from the forward pass of
another line's C ABI bench, which is not in this tree. Until a box run of this script on `sslm_decode_threading_bench`
reads them, together with the one-thread 7-token prefill A/B, they are expected values, not measured ones. If
MSVC inlines `DotRowAvx2` or `TiledGemmAvx2` into this binary, the check reports them MISSING, and that also needs a
box reading.

**Re-run it after any change to `src/matmul.cpp` or to the MSVC toolchain.** The pin fixes the placement against
edits elsewhere, but an edit to `matmul.cpp` itself, or a different compiler, can still move the kernels.

## Why (the 2026-10-21 bisect)

A one-thread 7-token prefill under MSVC flipped between about 145 ms and 185 ms (about 1.28x) on commits that did not
touch its path: the DotRow loop moved 16 bytes mod 64 with the code linked ahead of it, and the pin restores the fast placement (1.00x).
