// Force-included (after d3d12.h) by the multirow-gpu suite's mingw-w64 builds only. mingw-w64 11's
// d3d12.h (Ubuntu 24.04's mingw-w64-x86-64-dev 11.0.1) predates enumerators the engine uses; this
// supplies them with the Windows SDK's values. MSVC builds never include this file.
#pragma once
#if defined(__MINGW64_VERSION_MAJOR) && __MINGW64_VERSION_MAJOR < 12
static constexpr D3D12_RESOURCE_STATES D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE =
    static_cast<D3D12_RESOURCE_STATES>(0x400000);
#endif
