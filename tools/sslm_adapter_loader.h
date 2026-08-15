// sslm_adapter_loader.h -- T-2113 (B6, D-SLM3368): compatibility shim. This file's own body
// (LoadAdapterArtifact/AdapterHandle/ApplyAdapterToLayers, T-2102) relocated verbatim to
// include/superslm/adapter_marshal.h, so the 1.0 GPU API's own sslm_gpu_adapter_map
// (src/gpu/gpu_1p0.cpp) can reuse it -- the identical move D-SLM3351 made for sslm_marshal.h at
// B2. Every existing includer of this file (sslm_generate.cpp, sslm_lora_switch_demo.cpp,
// test_main.cpp) keeps compiling unchanged.
#ifndef SUPERSLM_TOOLS_SSLM_ADAPTER_LOADER_H
#define SUPERSLM_TOOLS_SSLM_ADAPTER_LOADER_H

#include "superslm/adapter_marshal.h"

#endif  // SUPERSLM_TOOLS_SSLM_ADAPTER_LOADER_H
