// T-2851 row 3: compile the documented run itself. The probe mode retains
// the earlier pool's shared dispatch members so ThreadSanitizer must reject it.
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include "docs/parallel_for_reference.hpp"

namespace {
struct TaskContext {
    std::array<std::atomic<int>, 238> visits{};
    int count = 0;
    std::atomic<int> out_of_range{0};
    void Reset(int n) {
        count = n;
        out_of_range.store(0);
        for (auto& v : visits) v.store(0);
    }
};
void Task(void* opaque, int32_t i) {
    auto& ctx = *static_cast<TaskContext*>(opaque);
    if (i < 0 || i >= ctx.count) ctx.out_of_range.fetch_add(1);
    else ctx.visits[static_cast<size_t>(i)].fetch_add(1);
}

// The probe's original Pool shape: workers read shared fn_/ctx_/count_ after
// unlocking, and the next dispatch can overwrite those members. Its defect
// population is independent of the documented ReferenceParallelFor class.
class ProbePool {
public:
    explicit ProbePool(int participants) {
        for (int i = 1; i < participants; ++i) workers_.emplace_back([this] { Loop(); });
    }
    ~ProbePool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    static void Run(void* host, int32_t count, sslm_task_fn task, void* task_ctx) {
        static_cast<ProbePool*>(host)->Dispatch(count, task, task_ctx);
    }
private:
    void Dispatch(int32_t count, sslm_task_fn task, void* task_ctx) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            task_ctx_ = task_ctx;
            count_ = count;
            next_.store(0);
            remaining_.store(count);
            ++generation_;
        }
        ready_.notify_all();
        Drain();
        while (remaining_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
    void Drain() {
        for (int32_t i = next_.fetch_add(1); i < count_; i = next_.fetch_add(1)) {
            task_(task_ctx_, i);
            remaining_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    void Loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
            }
            Drain();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_;
    uint64_t generation_ = 0;
    bool stop_ = false;
    sslm_task_fn task_ = nullptr;
    void* task_ctx_ = nullptr;
    int32_t count_ = 0;
    std::atomic<int32_t> next_{0};
    std::atomic<int32_t> remaining_{0};
};
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const bool probe = argv[1][0] == 'p' && argv[1][1] == '\0';
    if (!probe && !(argv[1][0] == 'r' && argv[1][1] == '\0')) return 2;
    ReferenceParallelFor reference(3);
    ProbePool old_pool(4);
    const auto hook = reference.Hook(238);
    TaskContext contexts[2];
    for (int dispatch = 0; dispatch < 10000; ++dispatch) {
        TaskContext& ctx = contexts[dispatch & 1];
        const int count = dispatch & 1 ? 238 : 3;
        ctx.Reset(count);
        if (probe) ProbePool::Run(&old_pool, count, &Task, &ctx);
        else hook.run(hook.host_ctx, count, &Task, &ctx);
        if (ctx.out_of_range.load()) return 3;
        for (int i = 0; i < count; ++i) if (ctx.visits[i].load() != 1) return 4;
    }
    std::printf("PASS %s 10000 alternating task_ctx/count dispatches\n",
                probe ? "probe" : "reference");
    return 0;
}
