# SuperSLM 1.3 release review record

This record defines the candidate, its public claim, the executed evidence, the independent
review, and the release disposition. 1.3.0 carries one change of substance: the library's load
path no longer performs floating-point arithmetic, and a CI gate now decides that by
disassembly rather than by inspection.

## Index

1. [Candidate and decision](#1-candidate-and-decision)
2. [Public contract](#2-public-contract)
3. [Executed evidence](#3-executed-evidence)
4. [Independent-gate disposition](#4-independent-gate-disposition)
5. [Adversarial review brief](#5-adversarial-review-brief)
6. [Release procedure](#6-release-procedure)

## 1. Candidate and decision

- Candidate branch: `brunel/t2348-fp-scan-build`
- Review range: `main...brunel/t2348-fp-scan-build`
- Version prepared in-tree: `1.3.0` (`CMakeLists.txt`, `README.md`, `CHANGELOG.md`)
- Scope ruling: 1.3.0 carries **one** consumer-driven ask and nothing else. Four sibling asks were
  approved at the same time and are deliberately excluded; they remain open 1.x work.
- Ship-blocking prerequisite this release satisfies: a downstream consumer's load path traps at
  handle-open time on some toolchains when the six floating-point exception mask bits are cleared,
  because a standard-library hash container performs a floating-point divide during bucket sizing.
  That consumer cannot open a handle against any engine older than this tag.

**What changed, mechanically.** Ten `std::unordered_map`/`std::unordered_set` instances reachable
pre-token from tokenizer-open, model-load, and session-create/session-restore are replaced by a
fixed-population, integer-arithmetic open-addressing construction (`src/detail/int_hash.h`,
`src/detail/context_hash.h`). Separately, seven compiled symbols are restructured from a `switch`
to a chain of direct conditional branches, removing a compiler-emitted jump table from each
symbol's own extent.

## 2. Public contract

**The guarantee, stated exactly.** The `superslm` CMake target's compiled object output — every
member of the static archive it links into — contains no floating-point arithmetic instruction.
"Floating-point arithmetic instruction" means an instruction whose semantics compute a numeric
result under IEEE-754 rules: addition, subtraction, multiplication, division, square root, fused
multiply-add, rounding conversion, or a numeric comparison that reads operand bits as a float.

**What the guarantee does not assert.** It says nothing about arithmetic an external callee — the
CRT, the STL, a consumer-installed callback — might itself perform. The gate's call-edge
classifier is retained as a non-gating diagnostic only. This is narrower than the wording several
earlier drafts of this work used ("no floating-point operation of any kind"), and the narrowing is
deliberate: the earlier wording claimed something the mechanism does not establish.

**Why the narrower claim still covers the motivating defect.** The bucket-sizing divide that
started this work is inlined by the compiler directly into SuperSLM's own objects rather than
called out of line, so it is arithmetic present in the scanned corpus. The call-edge classifier
was catching a different and weaker concern.

**How it is decided.** Every archive member is disassembled. Checks (A) and (B) allow-list known
safe move, bitwise-logical, and integer mnemonics and reject everything else that touches the
vector or floating-point register file. The shape is deliberately allow-list rather than
deny-list: an unknown mnemonic — a new ISA extension, a codegen change — is rejected rather than
passed, so the gate fails closed. A member whose bytes cannot be fully accounted for is a REFUSE,
which blocks the gate exactly as a REJECT does, with no partial credit.

**ABI.** No ABI change. One public header is modified and the change is comment-only: the
`AntiLmRetainedBytes` footprint calibration is retracted as measured-false against the new
containers and replaced with measured lower-bound ranges. No declaration changed. Verified by
reading `git diff main...HEAD -- include/`.

## 3. Executed evidence

Every figure below was executed. The machine is an AMD Ryzen 9 3950X (Zen 2, sixteen cores, **no
AVX-512**) unless stated otherwise, and the commit is named per row because the branch moved
during this work. Rows are marked by who ran them.

### 3.1 The gate, in CI's own build configuration

Run at `61bdcdd`, **conductor-executed**, against a build produced by the `fp-free-scan-gate`
job's own two commands (`cmake -B <dir>`, then `cmake --build <dir> --target superslm --config
Release`) rather than by the repository's hand-written batch build:

```
17 object(s); 2096 symbol(s) ACCEPT, 0 REJECT, 0 object(s) REFUSE (checks (A)/(B), gating);
32 symbol(s) reject under check (C) alone (non-gating diagnostic)
PASS  — exit 0
```

This configuration had never been exercised before this release round, and it is the one CI uses.

### 3.2 Suites

| Suite | Result | Commit | Run by |
|---|---|---|---|
| `tests/t2296-fp-free-open-red-suite` vs the CI-configuration build | 189 passed / 1 skipped / 3 xfailed / 0 failed | `61bdcdd` | conductor |
| `tests/t2296-fp-free-open-red-suite` vs the repository's canonical build | 189 passed / 1 skipped / 3 xfailed / 0 failed | `61bdcdd` | test-author round |
| `pytest tools/ tests/reference/` (the `converter-validate` job) | 1914 passed / 9 skipped / 13 deselected / 0 failed | `61bdcdd` | conductor |
| `pytest tests/ci` | 377 passed, exit 0 | `af4e841` | build round |
| `build.bat` end to end | exit 0; 34,207 checks, 0 failures | `af4e841` | build round |

The engine regression is cited at `af4e841` rather than at the branch tip because the only files
that changed between `af4e841` and `61bdcdd` are three Python test files — verified by
`git diff --name-only`, zero C++, header, CMake, or batch files — so no compiled surface moved.

### 3.3 The gate's own behaviour was proven unchanged by the release round

The release round modified the gate driver (a docstring correction, a de-duplicated candidate-path
list, and the removal of a function with no callers). Before that change was committed, the
modified driver and the pristine prior driver were both run against two independent real archives:
byte-identical stdout, exit 0 in both, 0 REJECT and 0 REFUSE in both, and a byte-identical
missing-archive error path. No verdict moved.

### 3.4 What is NOT executed, stated rather than implied

The workflow has 29 jobs. Only the Windows and Python-reachable subset above has been run on this
branch. **The Linux, macOS, sanitizer, forced-ISA, and cross-toolchain digest legs have never run
on this candidate**, because the branch is unpushed and the repository's remote carries a single
head. Those legs are proven by the matrix after the push and by nothing before it.

Separately, the AVX-512 code paths are not exercised by any figure in this document: the machine
that produced them has no AVX-512.

## 4. Independent-gate disposition

**The instrument that decides the guarantee is independently commissioned.** Its per-symbol
verdict was validated by a seat other than its builder, on three toolchain legs, at a floor of a
**single** floating-point instruction — the construction fires whether that instruction is the
first or the last member of a seventeen-member archive. Commissioning was re-run rather than
inherited each time the classifier changed; four earlier commissionings returned DEAD and killed
the readings they covered.

**The arc's adversarial history is not decoration.** The design was folded 43 times; adversary
strikes fractured its acceptance form repeatedly, including on its own stated corollary, and the
fold that answered each was itself re-struck when it moved what the gate accepts. Two code-review
gates ran on the built instrument and both returned fix-then-ship; the second found all three of
its blocking findings inside the first round's own remedy.

**Two defects were found in this release round after the arc was called closed**, both by
pre-flight rather than by review, and both would have been red legs on the tagged commit:

1. A test file importing a module absent from HEAD aborted the `converter-validate` job **at
   collection**, taking 1923 uncollected tests with it. It had been invisible for 26 days because
   the branch it lived on was never pushed.
2. A red-suite cell asserted that a specific object still carried one specific mnemonic. Under
   CI's build configuration the compiler selects a different member of the same instruction family
   for the same source, so the cell died on its own fixture precondition and never reached the
   boundary it exists to grade.

Both are closed and re-verified by execution. They are recorded here because they bear on how much
weight the phrase "the arc is closed" carries: it described the design and its instrument
accurately, and it did not describe the release.

**Reviews of this candidate.** Two independent reviews read this same commit. One is an in-house
code review briefed blind — given the artifact and the standard, and deliberately not given the
findings above or any suspicion about where to look. The other is an outside cross-model review
run by the maintainer against this file. Neither is shown the other's findings.

## 5. Adversarial review brief

The claims worth attacking, in the order a defect in them would cost most:

1. **The guarantee in §2 is either wider than the mechanism establishes, or the mechanism does not
   establish it.** The allow-list is asserted to be fail-closed on an unknown mnemonic. Is it? A
   prior strike on an earlier version of this gate found that for a large fraction of the
   mnemonics one check accepted, a deny-list was in fact deciding — the opposite of the claimed
   shape.
2. **The corpus is the archive the build produced.** If a compiled unit that ships can be absent
   from that archive, the guarantee has a hole that no amount of scanning inside it will find.
3. **REFUSE is claimed to block exactly as REJECT does.** Verify that in the driver rather than in
   its documentation.
4. **The replacement containers are asserted behaviourally identical to the standard-library ones
   they replace**, including on erase, rehash, iteration order dependence, and exhaustion. The
   load path's correctness now rests on that.
5. **Seven symbols were restructured from `switch` to conditional chains and called
   "behaviourally identical in every case."** That is a claim about seven functions' full input
   domains.
6. **§3's numbers.** Each names a commit and a runner. Any figure whose stated cell does not
   support the conclusion drawn from it is a finding, whether or not the figure is correct.

## 6. Release procedure

1. Land any blocking finding from either review, and re-run §3's evidence at the new commit.
2. Fast-forward `main` to the candidate and push. The push also carries two older local `main`
   commits that were never pushed — a test probe and a converter model-label fix with its own
   tests — both read and both presentable.
3. The full 29-job matrix runs on the release commit. It must be green before the tag; §3.4 names
   what the matrix, and only the matrix, can establish.
4. Annotated `v1.3.0` tag on that commit, pushed.
5. Release page published from the tag.
