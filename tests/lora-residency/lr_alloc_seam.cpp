// The allocation seam (lr_alloc_seam.h). Every block carries a 16-byte header just below the pointer
// handed out: the malloc'd base and the requested size, the size's top bit marking a counted block.
// Aligned forms over-allocate by the alignment and place the header below the aligned pointer the
// same way, so every delete form frees through the same header.
#include "lr_alloc_seam.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

std::atomic<int64_t> g_live{0};
std::atomic<int64_t> g_hwm{0};
std::atomic<int64_t> g_count{0};
thread_local bool t_armed = false;

constexpr uint64_t kCountedBit = uint64_t{1} << 63;
constexpr size_t kHeader = 16;

void* Allocate(size_t size, size_t align) noexcept {
	if (align < alignof(std::max_align_t)) align = alignof(std::max_align_t);
	const size_t extra = kHeader + align;
	void* base = std::malloc(size + extra);
	if (!base) return nullptr;
	uintptr_t p = reinterpret_cast<uintptr_t>(base) + kHeader;
	p = (p + align - 1) / align * align;
	uint64_t meta = size;
	if (t_armed) {
		meta |= kCountedBit;
		g_count.fetch_add(1, std::memory_order_relaxed);
		const int64_t live = g_live.fetch_add(static_cast<int64_t>(size), std::memory_order_relaxed) +
		                     static_cast<int64_t>(size);
		int64_t hwm = g_hwm.load(std::memory_order_relaxed);
		while (live > hwm && !g_hwm.compare_exchange_weak(hwm, live, std::memory_order_relaxed)) {
		}
	}
	std::memcpy(reinterpret_cast<void*>(p - 16), &base, sizeof(base));
	std::memcpy(reinterpret_cast<void*>(p - 8), &meta, sizeof(meta));
	return reinterpret_cast<void*>(p);
}

void Release(void* ptr) noexcept {
	if (!ptr) return;
	const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
	void* base = nullptr;
	uint64_t meta = 0;
	std::memcpy(&base, reinterpret_cast<void*>(p - 16), sizeof(base));
	std::memcpy(&meta, reinterpret_cast<void*>(p - 8), sizeof(meta));
	if (meta & kCountedBit) g_live.fetch_sub(static_cast<int64_t>(meta & ~kCountedBit), std::memory_order_relaxed);
	std::free(base);
}

void* AllocateOrThrow(size_t size, size_t align) {
	if (void* p = Allocate(size ? size : 1, align)) return p;
	throw std::bad_alloc{};
}

}  // namespace

namespace lr_seam {
void Arm() { t_armed = true; }
void Disarm() { t_armed = false; }
int64_t Live() { return g_live.load(); }
int64_t Hwm() { return g_hwm.load(); }
int64_t Count() { return g_count.load(); }
void ResetHwm() { g_hwm.store(g_live.load()); }

Scope::Scope() : live0_(Live()), count0_(Count()), was_armed_(t_armed) {
	ResetHwm();
	Arm();
}
Scope::~Scope() { t_armed = was_armed_; }
int64_t Scope::PeakAboveEntry() const { return Hwm() - live0_; }
int64_t Scope::NetLive() const { return Live() - live0_; }
int64_t Scope::Allocations() const { return Count() - count0_; }
}  // namespace lr_seam

void* operator new(std::size_t size) { return AllocateOrThrow(size, 0); }
void* operator new[](std::size_t size) { return AllocateOrThrow(size, 0); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return Allocate(size ? size : 1, 0); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return Allocate(size ? size : 1, 0); }
void* operator new(std::size_t size, std::align_val_t a) { return AllocateOrThrow(size, static_cast<size_t>(a)); }
void* operator new[](std::size_t size, std::align_val_t a) { return AllocateOrThrow(size, static_cast<size_t>(a)); }
void* operator new(std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
	return Allocate(size ? size : 1, static_cast<size_t>(a));
}
void* operator new[](std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
	return Allocate(size ? size : 1, static_cast<size_t>(a));
}
void operator delete(void* p) noexcept { Release(p); }
void operator delete[](void* p) noexcept { Release(p); }
void operator delete(void* p, std::size_t) noexcept { Release(p); }
void operator delete[](void* p, std::size_t) noexcept { Release(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { Release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { Release(p); }
void operator delete(void* p, std::align_val_t) noexcept { Release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { Release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { Release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { Release(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { Release(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { Release(p); }
