#pragma once
// Shim: serial ParallelFor with UE's (name, count, min batch, body) shape.
template <class F> inline void ParallelFor(const char*, int32 n, int32, F body) { for (int32 i = 0; i < n; ++i) body(i); }
