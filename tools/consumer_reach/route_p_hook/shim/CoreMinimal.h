// Shim for compiling SuperSLMFinishHook.cpp outside Unreal (route P runtime hook witness, plan rev 6).
// Only what the file uses: int32 and TEXT. Nothing here changes the value Make() assigns.
#pragma once
#include <cstdint>
typedef int32_t int32;
#define TEXT(x) x
