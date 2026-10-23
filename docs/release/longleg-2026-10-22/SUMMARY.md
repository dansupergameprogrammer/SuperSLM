# SuperSLM 1.14.0 long CPU legs: drive kit summary

Written 2026-10-22 15:58 on Intel(R) Core(TM) i5-14400F, 32 GB, Microsoft Windows 11 Home build 26300.
Source: `d129b5fad074faf31efdacc8176d73e3e4c5d523` (claude/precache-1p14-post-m1), cloned from the drive's bundle; `--is-shallow-repository` = false. Work root: `C:\_scratch\longleg`.

## Builds

| toolchain | status | compiler | superslm_precache_e1 SHA-256 | superslm_precache_unpinned SHA-256 |
|---|---|---|---|---|
| msvc | ok | Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31948 for x64 | 77e428dc781b2e3854c63f22e0a2860149aaa48d35737b750c3be3892e53add9 | 0b5c2879eb4dafcc19b732f0e0f176919e4de71ae7052bee3963f9bcf8a109a9 |
| clangcl | ok | clang version 15.0.1; Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31948 for x64 | 4819bcbc22235f195c01d060413ee510a74eeafc3a1d3f43fcbb6bc60aa703f3 | d092dca9c2fade2dc58bf4a2b0a7eff38fa1f7cbd6ba5ee4e2761565f45723fc |
| msvc-asan | ok | Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31948 for x64 | 3370578f295acc497c50fa611c485eeaa1056b9a60f4ac435380f2873b2f654c | 65644397eb9aecba383b23925b3dbc3cd15bbb3f928b23eda511dca8000cbcaa |

## Pieces

| piece | cell | toolchain | verdict | checks | failed | hours (last attempt) | attempts | finished |
|---|---|---|---|---|---|---|---|---|
| msvc__4.2-E3 | 4.2/E3 | msvc | **GREEN** | 3134 | 0 | 4.51 | 1 | 2026-10-22T14:36:58 |
| msvc__4.1-E2-synthetic-shapes | 4.1/E2:synthetic-shapes | msvc | **GREEN** | 2262 | 0 | 4.45 | 1 | 2026-10-22T14:33:17 |
| msvc__4.7-E2-synthetic1p5 | 4.7/E2:synthetic1p5 | msvc | **GREEN** | 4098 | 0 | 2.09 | 1 | 2026-10-22T12:11:24 |
| msvc__9.8-E2-synthetic1p5 | 9.8/E2:synthetic1p5 | msvc | **GREEN** | 1987 | 0 | 1.93 | 1 | 2026-10-22T12:02:21 |
| msvc__4.7-E3-synthetic1p5 | 4.7/E3:synthetic1p5 | msvc | **GREEN** | 1046 | 0 | 0.01 | 1 | 2026-10-22T12:02:43 |
| msvc__9.8-E3-synthetic1p5 | 9.8/E3:synthetic1p5 | msvc | **GREEN** | 1288 | 0 | 0.90 | 1 | 2026-10-22T12:56:48 |
| msvc-asan__4.2-E3 | 4.2/E3 | msvc-asan | **CRASHED** |  |  | 0.01 | 2 | 2026-10-22T12:12:06 |
| clangcl__4.2-E3 | 4.2/E3 | clangcl | **GREEN** | 3134 | 0 | 3.77 | 1 | 2026-10-22T15:58:07 |
| msvc-asan__4.7-E2-synthetic1p5 | 4.7/E2:synthetic1p5 | msvc-asan | **CRASHED** |  |  | 0.01 | 2 | 2026-10-22T12:57:30 |
| msvc-asan__9.8-E2-synthetic1p5 | 9.8/E2:synthetic1p5 | msvc-asan | **CRASHED** |  |  | 0.01 | 2 | 2026-10-22T12:58:11 |
| msvc-asan__4.7-E3-synthetic1p5 | 4.7/E3:synthetic1p5 | msvc-asan | **CRASHED** |  |  | 0.01 | 2 | 2026-10-22T12:58:53 |
| clangcl__4.7-E2-synthetic1p5 | 4.7/E2:synthetic1p5 | clangcl | **GREEN** | 4098 | 0 | 1.75 | 1 | 2026-10-22T14:43:43 |
| clangcl__9.8-E2-synthetic1p5 | 9.8/E2:synthetic1p5 | clangcl | **GREEN** | 1987 | 0 | 1.41 | 1 | 2026-10-22T15:58:09 |
| msvc-asan__9.8-E3-synthetic1p5 | 9.8/E3:synthetic1p5 | msvc-asan | **CRASHED** |  |  | 0.01 | 2 | 2026-10-22T14:37:40 |
| clangcl__4.7-E3-synthetic1p5 | 4.7/E3:synthetic1p5 | clangcl | **GREEN** | 1046 | 0 | 0.01 | 1 | 2026-10-22T14:38:01 |
| clangcl__9.8-E3-synthetic1p5 | 9.8/E3:synthetic1p5 | clangcl | **GREEN** | 1288 | 0 | 0.77 | 1 | 2026-10-22T15:24:05 |

## Failures (first lines; the full logs are in results\)

### msvc-asan__4.2-E3

```
piece: msvc-asan__4.2-E3
cell: 4.2/E3
verdict: CRASHED
exit: -1073741502
checks: 
failures: 
```

### msvc-asan__4.7-E2-synthetic1p5

```
piece: msvc-asan__4.7-E2-synthetic1p5
cell: 4.7/E2:synthetic1p5
verdict: CRASHED
exit: -1073741502
checks: 
failures: 
```

### msvc-asan__9.8-E2-synthetic1p5

```
piece: msvc-asan__9.8-E2-synthetic1p5
cell: 9.8/E2:synthetic1p5
verdict: CRASHED
exit: -1073741502
checks: 
failures: 
```

### msvc-asan__4.7-E3-synthetic1p5

```
piece: msvc-asan__4.7-E3-synthetic1p5
cell: 4.7/E3:synthetic1p5
verdict: CRASHED
exit: -1073741502
checks: 
failures: 
```

### msvc-asan__9.8-E3-synthetic1p5

```
piece: msvc-asan__9.8-E3-synthetic1p5
cell: 9.8/E3:synthetic1p5
verdict: CRASHED
exit: -1073741502
checks: 
failures: 
```

## Not run by this kit

- **TSan 4.7/E2:pkv_32k and 9.8/E2:pkv_32k**: Linux only. ThreadSanitizer has no Windows runtime for MSVC or clang-cl. Each cell needs more than 2 hours under TSan; run them on Linux (GCC 13, -fsanitize=thread) with no time limit.
- **E4: 4.3, 6.6/E4, 6.6/E4:rule-b, 10.1-10.3/E4, real_two_process**: No GPU needed, but not ready: the E4 re-derivation to the rule B (10.1, 10.2) is not written, the real-artifact references at the post-M1 anchor are not regenerated or licensed (box B3), the timing instrument (10.3) is not commissioned, and 4.3 needs the three compilers' blobs. The drive carries the real 0.5B and 1.5B models for when they are.

**All pieces are done.** Bring this file back.
