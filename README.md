# SuperSLM

A deterministic, integer-only runtime for small language models — Layer 1 of the
SuperSLM stack. It loads a converted, quantized `.sslm` artifact and runs it with
**bit-identical output across machines, compilers, and (later) GPUs**: the same
prompt produces the same tokens and the same hashes everywhere.

Layer 1 is independently embeddable — standard library only, no third-party runtime
dependency, no engine in its graph (DecisionLog D-SLM13). Layer 2 (`SuperSLM-Unreal`)
wraps it for Unreal Engine and ships in lockstep.

> **Status: pre-release, all rights reserved.** This repository is under active
> construction and is not yet public. The intended public license is permissive
> (see `LICENSE`); it is withheld only until the first public release (build
> sequence S6).

## What ships in an artifact

A `.sslm` file is a versioned, integrity-checked container: a header, a section
table, and aligned sections (weights, scales, RoPE tables, biases, tokenizer, schema
masks, golden reference hashes). The offline converter (build-time Python) reads a
Hugging Face checkpoint and emits the artifact; the runtime maps and validates it.
The format is specified in [`docs/sslm_format.md`](docs/sslm_format.md).

The loader is a **trust boundary**: it treats the whole file as hostile input and
validates every field against declared bounds before reading a section byte. Any
deviation — bad magic, unsupported version, an out-of-bounds or overlapping section,
a hash mismatch — is a rejection with a versioned diagnostic, never a silent partial
load.

## Building

```
cmake -B build && cmake --build build && ctest --test-dir build
```

On Windows with Visual Studio installed, `build.bat` is a one-shot MSVC build + test.
The test suite is standard-library only and prints a `checks / failures` summary; a
nonzero exit means a failure.

## Layout

| Path                     | What                                                    |
|--------------------------|---------------------------------------------------------|
| `include/superslm/`      | public headers — the embeddable API                     |
| `src/`                   | implementation                                          |
| `tests/`                 | the test suite (one harness, no third-party framework)  |
| `tools/`                 | build-time tooling                                      |
| `docs/`                  | format and design specifications                        |

## Build sequence

The runtime is built in strictly dependency-ordered slots, each closing on its proof
(`SuperSLM_Plan.md` §19): **S0** skeleton + artifact format + loader, **S1** tokenizer
+ converter, **S2** integer kernel set, **S3** single-sequence end-to-end + the first
determinism matrix, **S4** many-sequence machinery, **S4g** GPU backend, **S5**
constraint engine, **S6** calibration + docs + publish. This repository currently has
**S0** open.
