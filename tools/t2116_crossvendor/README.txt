SuperSLM 1.0 GPU core -- cross-vendor certification package (T-2116)
======================================================================

SOURCE OF TRUTH. This file, run_crossvendor.ps1, and pins/pins.txt are committed at
tools/t2116_crossvendor/ in the SuperSLM repository (branch brunel/t2116-crossvendor) --
this is the canonical copy; the flash-drive/scratch package copies these three files
verbatim. bin/ (prebuilt .exe files, built from this same commit via build.bat) and
artifacts/ (the real .sslm model/adapter files) are NOT committed -- they are build
products and large binary artifacts respectively, assembled into the package alongside
these three files but reproducible from the commit named in pins/pins.txt.

WHAT THIS IS
------------
A run-only package that certifies the SuperSLM 1.0 GPU core on a D3D12 GPU other than
the NVIDIA RTX 2080 SUPER this project develops on. It checks three things, per adapter:

  (a) GPU decode output is bit-identical to the CPU-oracle reference values recorded
      on the 2080 SUPER (determinism).
  (b) The GPU decode path produces the identical result no matter how the work is
      sliced across dispatch calls (slice-invariance).
  (c) Tokens/second, measured and reported per cell, labeled by adapter and surface --
      NOT gated pass/fail, since throughput is expected to differ across vendors.

NO BUILD STEP. Every .exe in bin\ is prebuilt (MSVC 2022, this project's own build.bat,
SuperSLM main @ ce3e47c plus the adapter-selection change this ticket added -- see
CHANGES.txt-equivalent notes in the build log this package's own commit message points
at). Nothing on the target machine needs Visual Studio, the Windows SDK, or dxc.exe.

WHAT TO DO
----------
1. Copy this whole folder to the target machine (a flash drive is fine -- everything
   here is self-contained; nothing is written outside this folder except the results\
   subfolder this script creates).
2. Open PowerShell (either Windows PowerShell 5.1 or PowerShell 7/pwsh -- both work) in
   this folder.
3. Run:
       powershell -File run_crossvendor.ps1
   or, from inside an already-open PowerShell/pwsh window:
       .\run_crossvendor.ps1

   By default this enumerates every D3D12 hardware adapter on the machine (skipping
   software/WARP adapters) and requires ALL of them to pass. If the machine also has an
   integrated GPU you do not want certified, restrict the run to the card you care
   about:
       .\run_crossvendor.ps1 -OnlyAdapterNameLike "*Radeon*"
       .\run_crossvendor.ps1 -OnlyAdapterIndex 1
   (The script prints every adapter it finds, with its index, at the very start of every
   run -- use that list to pick the right filter.)

EXPECTED DURATION
------------------
Roughly 8-10 minutes per adapter on hardware comparable to the reference card (an
NVIDIA RTX 2080 SUPER). The two batch/thread-safety cells (b7, b8) are the slowest,
each 1-3 minutes; every other cell is under a minute. A machine with more than one
adapter certified runs the whole battery once per adapter, serially -- never in
parallel (this is a GPU-resident workload; two cells racing for the same GPU would
invalidate both).

WHAT A PASS LOOKS LIKE
------------------------
The script prints one line per battery cell per adapter (PASS or FAIL, with the
measured value), then one verdict line per adapter ("ADAPTER [n] <name> : PASS" or
"FAIL"), then a final line:

    CROSS-VENDOR GATE: PASS -- every requested adapter matched the RTX 2080 SUPER
    reference pins on every determinism check.

and exits with code 0. A full, timestamped, plain-text record of the run (every check,
every value, every adapter's driver version) is written to
results\<timestamp>\SUMMARY.txt, with a further per-tool raw log under
results\<timestamp>\adapter_<n>_<name>\<tool>.log.

WHAT A FAIL LOOKS LIKE
------------------------
Any of the following produces "CROSS-VENDOR GATE: FAIL" and a non-zero exit code:
  - A missing .exe, missing .sslm artifact, or missing pins file (checked before
    anything runs -- this is a loud, immediate failure, never a silently-skipped cell).
  - A determinism or slice-invariance check count, failure count, or exit code that
    does not match the pinned value from the reference card (pins\pins.txt).
  - A CPU/GPU divergence on the C5 oracle cell (the argmax token or the
    bit-identical/DIVERGENCE verdict).
Look at the specific FAIL lines the script prints, then the matching
results\<timestamp>\adapter_<n>_<name>\<tool>.log for the full raw output of that cell
(every battery tool prints the exact line and file where a check failed).

Tokens/second is NEVER part of the pass/fail verdict -- it is measured and reported,
labeled by adapter, cell, and surface, purely for comparison across vendors.

PACKAGE CONTENTS
-----------------
  README.txt              -- this file
  run_crossvendor.ps1      -- the entry script (ASCII-only, PowerShell 5.1 + 7 compatible)
  bin\                      -- 12 prebuilt tools + their compiled shaders (bin\shaders\*.cso)
  artifacts\                -- the real .sslm model/adapter files the battery runs against
                                (1.5B instruct, 0.5B instruct, the shopkeeper LoRA adapter)
  pins\pins.txt             -- reference values recorded on the RTX 2080 SUPER; the
                                cross-vendor pass compares every adapter against these
  results\                  -- created by the script on first run; timestamped per run

THE BATTERY, IN ORDER
-----------------------
  c5    -- t2039_c5_harness: single-token CPU-vs-GPU bit-identity oracle across the full
           SequenceLayerState surface plus final_norm+logits+argmax
  b1    -- t2113_b1_context_smoke: context handle lifecycle
  b2    -- t2113_b2_model_smoke: model handle map/unmap, base-hash validation
  b3    -- t2113_b3_sequence_smoke: sequence handle lifecycle, 24-step decode
  b35   -- t2113_b35_embed_smoke: embed-token path, prefill+decode bit-identity
  b5    -- t2113_b5_async_smoke: the async 1.0 API, slice-invariance across dispatch
           granularities (1/2/7/28 calls per token), plus throughput
  b6    -- t2113_b6_adapter_smoke: adapter (LoRA) residency and guard checks
  b6b   -- t2113_b6b_adapter_delta_smoke: adapter-bound GPU decode bit-identity vs the
           adapter-bound CPU oracle, at every slicing granularity, plus the
           adapter-bound-vs-base-only divergence proof, plus throughput with and
           without the adapter bound
  b7    -- t2113_b7_batch_smoke: multi-sequence batched decode, mixed adapter
           composition, batch-vs-serial and batch-vs-CPU-oracle bit-identity, plus
           aggregate throughput at batch widths 1/2/4
  b8    -- t2113_b8_thread_smoke: concurrent decode across two threads under handle
           churn, bit-identity vs the CPU oracle
  t2100 -- t2100_gpu_throughput: the dispatch-path benchmark -- 64 decode steps,
           CPU vs GPU per-step equality, GPU tokens/second, plus a full per-dispatch-site
           GPU-busy time breakdown

Every cell above ran clean (0 failures) on the RTX 2080 SUPER before this package was
assembled -- see pins\pins.txt for the exact pinned counts.

ADAPTER SELECTION -- WHY IT MATTERS
--------------------------------------
A machine with more than one D3D12 adapter (a discrete GPU plus an integrated one, or
two discrete GPUs) has no reliable "the right one" without being told. This project's
battery tools did not have a way to select an adapter before this ticket; the change is
one environment variable, SSLM_GPU_ADAPTER_INDEX, read once at device-init time
(src\gpu\d3d12_harness.h) -- when unset, behavior is byte-identical to before (picks
whichever adapter DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE ranks first). run_crossvendor.ps1
sets this variable once per adapter it certifies, in the same raw index order a
dedicated enumeration tool (t2116_list_adapters.exe, bin\) reports, and labels every
result with the adapter it ran on. A requested index that does not exist, is a software
adapter, or fails device creation is a loud tool-level failure, never a silent fallback
to a different adapter.
