# SuperSLM 1.14.0 long CPU legs: drive kit summary

Written 2026-10-22 22:10 on Intel(R) Core(TM) i5-14400F, 32 GB, Microsoft Windows 11 Home build 26300.
Source: `d129b5fad074faf31efdacc8176d73e3e4c5d523` (claude/precache-1p14-post-m1), cloned from the drive's bundle; `--is-shallow-repository` = false. Work root: `C:\_scratch\longleg`.

## Builds

| toolchain | status | compiler | superslm_precache_e1 SHA-256 | superslm_precache_unpinned SHA-256 |
|---|---|---|---|---|
| msvc | ok | Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31948 for x64 | 77e428dc781b2e3854c63f22e0a2860149aaa48d35737b750c3be3892e53add9 | 0b5c2879eb4dafcc19b732f0e0f176919e4de71ae7052bee3963f9bcf8a109a9 |
| clangcl | ok | clang version 15.0.1; Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31948 for x64 | 4819bcbc22235f195c01d060413ee510a74eeafc3a1d3f43fcbb6bc60aa703f3 | d092dca9c2fade2dc58bf4a2b0a7eff38fa1f7cbd6ba5ee4e2761565f45723fc |
| msvc-asan | failed | Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31948 for x64 | 3370578f295acc497c50fa611c485eeaa1056b9a60f4ac435380f2873b2f654c | 65644397eb9aecba383b23925b3dbc3cd15bbb3f928b23eda511dca8000cbcaa |
| msvc-e4 | ok | Microsoft (R) C/C++ Optimizing Compiler Version 19.34.31948 for x64 | 38396bf7984d877a6c98aa08a0599d248226acc2acda45e3b467910a367f1de1 |  |

## Pieces

| piece | cell | toolchain | verdict | checks | failed | hours (last attempt) | attempts | finished |
|---|---|---|---|---|---|---|---|---|
| msvc-e4__10.3-E4-worst | 10.3/E4:worst | msvc-e4 | **GREEN** | 27 | 0 | 1.08 | 1 | 2026-10-22T22:10:47 |

## Not run by this kit

- **TSan 4.7/E2:pkv_32k and 9.8/E2:pkv_32k**: Linux only. ThreadSanitizer has no Windows runtime for MSVC or clang-cl. Each cell needs more than 2 hours under TSan; run them on Linux (GCC 13, -fsanitize=thread) with no time limit.
- **E4: 4.3, 6.6/E4, 6.6/E4:rule-b, 10.1-10.3/E4, real_two_process**: Run on the home box under the E4 run sheet (timings need its commissioned receipt). Only W1, the 1.5B cap-length capture, runs here, with -E4Worst.

**All pieces are done.** Bring this file back.
