# Adapter peak bounds: the allocation-site census

Streamed adapters plan (rev 1.2) §3.1 and step L0. `sslm_adapter_map_peak_bound` and
`sslm_gpu_adapter_prepare_peak_bound` return `superslm_adapter::AdapterPeakTermsFor(...).Sum()`
(`src/adapter_core.h`). This file lists every function on the call graph of `sslm_adapter_map` and
of `superslm_adapter::PrepareAdapterCore` once the prescreen has passed, every heap allocation site
in it, and the term that covers the site. `tools/ci/check_adapter_peak_bound.py` holds it to the
source: every term below has a `PEAK-BOUND <term>` tag at its sites, every function listed exists,
and every term the bound sums appears here.

Census anchored on SuperSLM main 5599f7e4 (1.15.0 plus CI fixes), 2026-10-24.

## The terms

| Term | Covers | Size |
|---|---|---|
| `copy` | `SslmArtifact::bytes_` (`OpenFromMemory`'s `bytes_.assign`) | `artifact_bytes` + O |
| `sections` | `OpenFromMemory`'s `placed` rows and `sections_` table, both reserved to `section_count` <= 6 | 6 x (48 + `sizeof(SslmSectionView)`) |
| `manifests` | `SslmTensorManifest::Parse` (Weights) and `SslmAmplifyingFoldScaleView::Parse` (DFS1, UFS1): the entry table, the ranges and the `FixedIntSet` slots, each reserved to `tensor_count`, which the prescreen caps at 14 (Weights) and 7 (each fold kind) per base layer | per manifest n x (entry + 16) + 4n x 24 + 3 O |
| `name` | `AdapterMeta::source_adapter_name.assign` (prescreen: Provenance <= 68 + 4,096) | (4,096 \| 15) + 1 + O = 4,113 + O |
| `layers` | `layer_adapters.assign(num_hidden_layers)` | L x `sizeof(LayerAdapter)` + O |
| `modules` | `TargetModulesFromMask`'s vector, seven pushes (libstdc++ capacities 1, 2, 4, 8; MSVC 1, 2, 3, 4, 6, 9) | 15 x `sizeof(string_view)` |
| `messages` | `artifact.cpp`'s `Reject` messages (fixed text and numbers; at most three temporaries); `ValidateQkChannelTableAdditive`'s two per-layer probe names. Every name-carrying message is built only for a non-null error string, and both adapter paths pass null | 3 KiB + 128 |
| `handle` | CPU: `new sslm_adapter_s`. GPU: the prepared object | `sizeof` |
| `packed` | GPU prepare only: `ab_bytes` and `fold_bytes`, each reserved to its exact final size; every byte is a copy of a tensor or fold row the artifact holds, in non-overlapping ranges | `artifact_bytes` + 2 + 2 O |
| `slots` | GPU prepare only: the `[layer][7]` slot table | L x 7 x `sizeof(AdapterProjSlot)` + O |

O is `kAllocatorBigBlockOverhead`: what `std::allocator` adds to a block of 4,096 bytes or more on
the target compiler (MSVC x86/x64: 47 bytes, 55 in a debug build; libstdc++ and libc++: 0). The
info figures count every allocator block the same way (`AllocatorBlockBytes`).

The terms are summed as though all were live at once, so the bound over-approximates: at
pkv_def's two layers the measured slack is 10 to 22 KiB (cell 7.2's table). Only `copy` (and, for
the prepare, `packed`) is larger than every other term's slack combined, so only those two per-term
mutants are killed (cell `7.2/L2:bound-term-mutants`); the others are reported as survivors.

## The call graph and its sites

| Function | File | Sites | Term |
|---|---|---|---|
| `sslm_adapter_map` | `src/sslm_abi.cpp` | `new (std::nothrow) sslm_adapter_s` | handle |
| `PrescreenAdapterArtifact` | `src/adapter_core.h` | none | — |
| `SslmModel::Load` / `LoadImpl` | `src/model.cpp` | none of its own (null `err`: no message) | — |
| `SslmArtifact::OpenFromMemory` / `OpenFromMemoryImpl` | `src/artifact.cpp` | `placed.reserve`, `bytes_.assign`, `sections_.reserve`, `Reject` messages | sections, copy, messages |
| `ParseConfig` / `ParseConfigImpl` | `src/model.cpp` | none (null `err`) | — |
| `ParseSigmoidLut` / `ParseSigmoidLutImpl` | `src/model.cpp` | none (null `err`) | — |
| `SslmTensorManifest::Parse` / `ParseImpl` | `src/model.cpp` | `tensors.reserve`, `ranges.reserve`, `seen_names` | manifests |
| `SslmAmplifyingFoldScaleView::Parse` / `ParseImpl` | `src/model.cpp` | `entries.reserve`, `ranges.reserve`, `seen_names` | manifests |
| `ValidateSectionValues` | `src/model.cpp` | none of its own | — |
| `ValidateAmplifyingFoldScalesDomain` | `src/model.cpp` | message only under a non-null `err` | — |
| `ValidateTokenizerVocabSizeJoin` | `src/model.cpp` | message only under a non-null `err` | — |
| `ValidateConfigGeometryJoin` | `src/model.cpp` | message only under a non-null `err` | — |
| `ValidateFusedKHeadDim` | `src/model.cpp` | message only under a non-null `err` | — |
| `ValidateQkChannelTableAdditive` | `src/model.cpp` | per-layer probe names | messages |
| `ValidateNoLegacyFusedKKeys`, `ValidateFusedKCompositionDomains`, `ValidateQkChannelScaleRelations` | `src/model.cpp` | return at once: an adapter's header flags never set the QK-table bit (the prescreen refuses the table it requires) | — |
| `PopulateAdapterFromView` | `include/superslm/adapter_marshal.h` | `source_adapter_name.assign`, `layer_adapters.assign`, `TargetModulesFromMask`; its validators are handed a message string only for a non-null `err` | name, layers, modules |
| `ParseLayerProjName` (view form) | `include/superslm/adapter_marshal.h` | none | — |
| `FindTensorWithSuffix` | `include/superslm/adapter_marshal.h` | none | — |
| `ValidateAmplifyingFoldProjection`, `ValidateAmplifyingFoldDimension`, `ValidateAmplifyingFoldBaseHash` | `src/model.cpp` | message only under a non-null `err` | — |
| `PrepareAdapterCore` / `PrepareAdapterCoreFromView` | `src/adapter_core.h` | the local `layers` table (populate's) | layers |
| `PackAdapterSlots` | `src/adapter_core.h` | `slots.assign`, `ab_bytes.reserve`, `fold_bytes.reserve` | slots, packed |
| `CoveredLayers` | `src/adapter_core.h` | none | — |

Not on the graph after the prescreen: `TokenizerView::Open` (its sections are refused), every
other section parser (`Biases`, `RopeTables`, `WeightScales`, the keyed constants), and the
QK-table validators' table reads.
