// Multi-row GPU plan (rev 1.3.2) §3.4: the host-only part of a context's multi-row state, with no device types, so
// a native build can run it under ThreadSanitizer (cell 3.3, tests/multirow-gpu/mr3_tsan.cpp) while the engine uses
// the same code (GpuRowsContextState, multirow_wide.h).
//
// RowSetPool keeps the context's free list of row sets and the count of chunks open, under one mutex. A chunk takes a
// free set when its first slice is recorded (or makes one, then counts it with MadeOne), and returns it when the chunk
// closes, so after warm-up no slice allocates a set and the sets never outnumber the most chunks open at once.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace superslm_gpu::rows_host {

template <class RowSet>
class RowSetPool {
public:
	struct Stats {
		uint32_t sets_total = 0;  // sets made for this context: on the free list plus held by open chunks
		uint32_t free = 0;        // of those, on the free list
		uint32_t chunks_open = 0;
		uint32_t chunks_open_peak = 0;
	};

	// A set from the free list, or null when it is empty (the caller makes one and calls MadeOne).
	std::unique_ptr<RowSet> Take() {
		const std::lock_guard<std::mutex> lock(mu_);
		if (free_.empty()) return nullptr;
		std::unique_ptr<RowSet> set = std::move(free_.back());
		free_.pop_back();
		return set;
	}
	void MadeOne() {
		const std::lock_guard<std::mutex> lock(mu_);
		++sets_total_;
	}
	// A set back onto the free list (a null set is ignored).
	void Return(std::unique_ptr<RowSet> set) {
		if (set == nullptr) return;
		const std::lock_guard<std::mutex> lock(mu_);
		free_.push_back(std::move(set));
	}
	void ChunkOpened() {
		const std::lock_guard<std::mutex> lock(mu_);
		++chunks_open_;
		chunks_open_peak_ = std::max(chunks_open_peak_, chunks_open_);
	}
	void ChunkClosed() {
		const std::lock_guard<std::mutex> lock(mu_);
		if (chunks_open_ > 0) --chunks_open_;
	}
	// A chunk's close in one step: its set (if any) back onto the free list and the chunk count down.
	void ChunkClosed(std::unique_ptr<RowSet> set) {
		const std::lock_guard<std::mutex> lock(mu_);
		if (set != nullptr) free_.push_back(std::move(set));
		if (chunks_open_ > 0) --chunks_open_;
	}
	// A consistent snapshot.
	Stats stats() const {
		const std::lock_guard<std::mutex> lock(mu_);
		Stats s;
		s.sets_total = sets_total_;
		s.free = static_cast<uint32_t>(free_.size());
		s.chunks_open = chunks_open_;
		s.chunks_open_peak = chunks_open_peak_;
		return s;
	}

private:
	mutable std::mutex mu_;
	std::vector<std::unique_ptr<RowSet>> free_;
	uint32_t sets_total_ = 0;
	uint32_t chunks_open_ = 0, chunks_open_peak_ = 0;
};

}  // namespace superslm_gpu::rows_host
