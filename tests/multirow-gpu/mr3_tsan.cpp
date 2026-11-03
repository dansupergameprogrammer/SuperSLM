// Multi-row GPU plan (rev 1.3.2), cell 3.3's TSan leg: the host chunk state and the row-buffer free list under
// ThreadSanitizer, with a device stub. Built natively (g++ -fsanitize=thread) by mr3_wine.sh, not under mingw.
//
// CONTRACT FOR THE BUILDER (the test author's proposal; one edit here if the builder chooses other names). The plan
// asks for TSan "with a device stub", and the engine's chunk state is typed by D3D12 (multirow_wide.h), which no
// native build can compile. So the host-only part (the per-context free list of row sets and the open-chunk counts
// that GpuRowsContextState keeps under its mutex today) is expected in a portable header with no device types:
//
//   src/gpu/multirow_rows_host.h, namespace superslm_gpu::rows_host:
//     template <class RowSet> class RowSetPool {
//      public:
//       std::unique_ptr<RowSet> Take();                 // a free set, or nullptr (the caller makes one: MadeOne())
//       void MadeOne();                                 // count a set the caller made (sets_total + 1)
//       void Return(std::unique_ptr<RowSet> set);       // back onto the free list
//       void ChunkOpened();                             // chunks_open + 1, the peak kept
//       void ChunkClosed();                             // chunks_open - 1
//       struct Stats { uint32_t sets_total, free, chunks_open, chunks_open_peak; };
//       Stats stats() const;                            // a consistent snapshot
//     };
//   and the engine's GpuRowsContextState uses it (so the TSan run exercises the shipped code, not a copy).
//
// The run: 8 threads each open a chunk, take or make a set, read stats, return it and close, 20,000 times, beside a
// reader thread. Graded: TSan reports nothing (the runner fails on any report), and at the end sets_total <= the
// peak of open chunks, free == sets_total, chunks_open == 0.
// The other half of 3.3 (sslm_gpu_ready against rows_abandon on one handle, another handle uncorrupted) needs the
// engine and runs under Wine as mr3_cells' 3.3w.
#include <atomic>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#if __has_include("multirow_rows_host.h")
#include "multirow_rows_host.h"

struct StubSet {
	int bytes[16] = {};
};

int main() {
	superslm_gpu::rows_host::RowSetPool<StubSet> pool;
	std::atomic<bool> stop{false};
	std::atomic<int> bad{0};
	std::thread reader([&] {
		while (!stop) {
			const auto s = pool.stats();
			if (s.free > s.sets_total || s.chunks_open > s.chunks_open_peak) ++bad;
		}
	});
	std::vector<std::thread> th;
	for (int t = 0; t < 8; ++t)
		th.emplace_back([&, t] {
			for (int i = 0; i < 20000; ++i) {
				pool.ChunkOpened();
				std::unique_ptr<StubSet> s = pool.Take();
				if (!s) {
					s = std::make_unique<StubSet>();
					pool.MadeOne();
				}
				s->bytes[(t + i) % 16] += 1;  // the set is this chunk's alone while it is out
				pool.Return(std::move(s));
				pool.ChunkClosed();
			}
		});
	for (auto& x : th) x.join();
	stop = true;
	reader.join();
	const auto s = pool.stats();
	const bool ok = bad == 0 && s.sets_total <= s.chunks_open_peak && s.free == s.sets_total && s.chunks_open == 0;
	std::printf("%s 3.3 tsan: sets %u free %u open %u peak %u, %d inconsistent snapshots\n", ok ? "PASS" : "RED", s.sets_total, s.free,
	            s.chunks_open, s.chunks_open_peak, bad.load());
	return ok ? 0 : 1;
}
#else
int main() {
	std::printf("RED 3.3 tsan: red-unimplemented (src/gpu/multirow_rows_host.h, the portable host chunk state, does not exist)\n");
	return 1;
}
#endif
