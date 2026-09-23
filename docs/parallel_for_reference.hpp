// Reference `run` for superslm/parallel_for.h, for a host with no job system of its own.
// docs/api.md ("The token finish's parallel-for hook") points here. It is not a library API:
// SuperSLM never creates a thread, and a host with a job system should route `run` through that
// system instead. It is a single header, compiled as it stands by the test suite.
//
// Usage:
//   ReferenceParallelFor pool(3);                          // 3 workers + the calling thread
//   sslm_parallel_for hook = pool.Hook(4);                 // max_tasks = 4
//   sslm_workspace_set_parallel_for(ws, &hook);            // or sslm_gpu_context_set_host_parallel_for
//   ... decode ...
//   sslm_workspace_set_parallel_for(ws, NULL);             // clear before `pool` is destroyed
//
// How it meets parallel_for.h's precondition ("return only after every invocation has returned,
// and never touch task or task_ctx after that"):
// - Each dispatch is one Job record on the calling thread's stack: task, task_ctx, task_count, a
//   generation number, and the record's own next-index and participant counters.
// - A worker joins a dispatch only under the pool mutex, only while that record is the current
//   one, and only once per generation; joining increments the record's participant count.
// - Workers and the caller claim indices from the record's own counter, so each index runs once.
// - The caller withdraws the record under the mutex, so no worker can join after that point, and
//   returns only when the participant count reaches zero: every worker that took the record has
//   released it. No worker reads the record, `task` or `task_ctx` after `run` returns.
// One pool serves one `run` at a time; concurrent callers are serialized.
#ifndef SUPERSLM_DOCS_PARALLEL_FOR_REFERENCE_HPP
#define SUPERSLM_DOCS_PARALLEL_FOR_REFERENCE_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "superslm/parallel_for.h"

class ReferenceParallelFor {
public:
	explicit ReferenceParallelFor(int worker_count) {
		for (int i = 0; i < worker_count; ++i) workers_.emplace_back([this] { WorkerLoop(); });
	}
	~ReferenceParallelFor() {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stop_ = true;
		}
		work_ready_.notify_all();
		for (std::thread& t : workers_) t.join();
	}
	ReferenceParallelFor(const ReferenceParallelFor&) = delete;
	ReferenceParallelFor& operator=(const ReferenceParallelFor&) = delete;

	sslm_parallel_for Hook(int32_t max_tasks) {
		sslm_parallel_for pf{};
		pf.run = &ReferenceParallelFor::Run;
		pf.host_ctx = this;
		pf.max_tasks = max_tasks;
		return pf;
	}

	static void Run(void* host_ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		static_cast<ReferenceParallelFor*>(host_ctx)->Dispatch(task_count, task, task_ctx);
	}

private:
	struct Job {
		sslm_task_fn task;
		void* task_ctx;
		int32_t task_count;
		uint64_t generation;
		std::atomic<int32_t> next{0};
		int32_t participants = 0;  // guarded by mutex_
	};

	static void Drain(Job& job) {
		for (int32_t i = job.next.fetch_add(1); i < job.task_count; i = job.next.fetch_add(1)) {
			job.task(job.task_ctx, i);
		}
	}

	void Dispatch(int32_t task_count, sslm_task_fn task, void* task_ctx) {
		std::lock_guard<std::mutex> one_run_at_a_time(run_mutex_);
		Job job{task, task_ctx, task_count, 0};
		{
			std::lock_guard<std::mutex> lock(mutex_);
			job.generation = ++generation_;
			job.participants = 1;  // the calling thread
			current_ = &job;
		}
		work_ready_.notify_all();
		Drain(job);
		std::unique_lock<std::mutex> lock(mutex_);
		current_ = nullptr;  // withdrawn: no worker can join this record from here on
		job.participants -= 1;
		job_released_.wait(lock, [&] { return job.participants == 0; });
	}

	void WorkerLoop() {
		uint64_t seen = 0;
		for (;;) {
			Job* job = nullptr;
			{
				std::unique_lock<std::mutex> lock(mutex_);
				work_ready_.wait(lock, [&] {
					return stop_ || (current_ != nullptr && current_->generation != seen);
				});
				if (stop_) return;
				job = current_;
				seen = job->generation;
				job->participants += 1;
			}
			Drain(*job);
			{
				std::lock_guard<std::mutex> lock(mutex_);
				job->participants -= 1;
				if (job->participants == 0) job_released_.notify_all();
			}
		}
	}

	std::mutex run_mutex_;
	std::mutex mutex_;
	std::condition_variable work_ready_;
	std::condition_variable job_released_;
	Job* current_ = nullptr;
	uint64_t generation_ = 0;
	bool stop_ = false;
	std::vector<std::thread> workers_;
};

#endif  // SUPERSLM_DOCS_PARALLEL_FOR_REFERENCE_HPP
