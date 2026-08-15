// sslm_marshal.h -- compatibility shim (T-2113 B2). The real content moved to
// include/superslm/layer_marshal.h so a production translation unit (src/gpu/gpu_1p0.cpp's
// own sslm_gpu_model_map, design Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md
// Sec10 B2) can include the marshaling logic too -- this file's own original comment said
// "tools-only... never included from src/ or include/", which stopped being true the moment
// a production call site needed it. Kept here, forwarding, so every existing
// `#include "sslm_marshal.h"` call site (tests/test_main.cpp, tools/sslm_adapter_loader.h,
// tools/sslm_generate.cpp, tools/sslm_layer_trace.cpp, tools/sslm_lora_switch_demo.cpp,
// tools/t2039_c5_harness.cpp, tools/t2100_gpu_throughput.cpp) keeps compiling unchanged --
// zero call sites touched by this move.
#pragma once
#include "superslm/layer_marshal.h"
