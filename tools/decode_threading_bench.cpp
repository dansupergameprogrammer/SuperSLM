// Decode-threading plan (rev 1.2) §7: the bench for the box steps B0 and B1, headless. It drives the
// C ABI with the reference `run` of docs/parallel_for_reference.hpp (a std::thread pool, the calling
// thread included) and prints plain `key value` lines a driver can parse.
//
//   decode_threading_bench ARTIFACT verify [--prompt 300] [--decode 64] [--tasks 1,2,3,4,8]
//       B0: greedy tokens and the final SSB5 blob with no hook, then with SSLM_PARALLEL_FOR_MATVEC
//       at each max_tasks; every run must be byte-equal to no-hook. Prints the `run` calls per decode
//       token at each max_tasks (97 expected on Qwen2.5-0.5B at 4 and 256 KiB). Exit 1 on a mismatch.
//   decode_threading_bench ARTIFACT time [--contexts 16,300] [--decode 32] [--tasks 2,4,8]
//                                         [--pairs 15] [--min-row-bytes 65536,262144,1048576]
//       B1: decode tokens per second with the hook installed and bit 1 off against bit 1 on, as n
//       interleaved off/on pairs per setting; prints median and IQR of each side. --min-row-bytes
//       needs the seam build (sslm_decode_threading_bench_seams); the production build reads the
//       library's constant and refuses the flag.
//   decode_threading_bench ARTIFACT onetoken [--prompt 64] [--tasks 4] [--pairs 15]
//       B1's last reading: one-token prompt calls, milliseconds per token, bit 1 off against on.
//   decode_threading_bench - groups [--tasks 2,4] [--layers 24] [--pairs 15]
//       No artifact: the five one-row groups of one decode layer at Qwen2.5-0.5B's shapes (hidden
//       896, k + v 2 x 128, intermediate 4864), on synthetic weights for `--layers` layers (so the
//       weights stream from memory, not cache), serial against the split at the library's minimum
//       work, through the reference pool. The mechanism's cost alone, off the box; not a B-step.
//
// Every figure is indicative outside the box: the plan's gate readings are the box's (§7).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "superslm/layer_marshal.h"
#include "superslm/matmul.h"
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "../docs/parallel_for_reference.hpp"
#include "../src/forward/parallel_split.h"  // MatvecGroupSplit, MatvecGroupParallel (groups mode)

namespace {

std::vector<int> ParseList(const char* s) {
	std::vector<int> out;
	for (const char* p = s; *p;) {
		out.push_back(std::atoi(p));
		while (*p && *p != ',') ++p;
		if (*p == ',') ++p;
	}
	return out;
}

// Counts `run` calls, forwarding each to the reference pool.
struct CountingPool {
	ReferenceParallelFor* pool = nullptr;
	std::atomic<long long> calls{0};
	static void Run(void* host_ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		CountingPool& c = *static_cast<CountingPool*>(host_ctx);
		c.calls.fetch_add(1, std::memory_order_relaxed);
		ReferenceParallelFor::Run(c.pool, task_count, task, task_ctx);
	}
};

struct Engine {
	uint8_t* bytes = nullptr;
	size_t size = 0;
	sslm_model model = nullptr;
	int32_t vocab = 0;
	int32_t layers = 0;
	void* pool_buf = nullptr;
	sslm_kv_pool pool = nullptr;
	void* ws_buf = nullptr;
	sslm_workspace ws = nullptr;

	bool Open(const char* path, int32_t chunk) {
		std::vector<uint8_t> raw;
		if (!superslm_marshal::ReadFile(path, raw) || raw.empty()) {
			std::fprintf(stderr, "cannot read %s\n", path);
			return false;
		}
		size = raw.size();
		bytes = static_cast<uint8_t*>(::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
		std::memcpy(bytes, raw.data(), size);
		if (sslm_model_map(bytes, size, &model) != SSLM_OK) {
			std::fprintf(stderr, "sslm_model_map failed\n");
			return false;
		}
		sslm_decode_params p{};
		for (int32_t l = 256; l >= 1; --l) {
			if (sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, l, &p) == SSLM_OK) {
				layers = l;
				break;
			}
		}
		superslm::SslmModelView view;
		std::string err;
		if (superslm::SslmModel::Load(bytes, size, view, &err) != superslm::SslmModelStatus::Ok) return false;
		vocab = static_cast<int32_t>(view.config.vocab_size);
		const uint32_t blocks = 1;
		const size_t psize = blocks * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, blocks);
		pool_buf = ::operator new(psize, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (sslm_kv_pool_create(model, pool_buf, psize, blocks, &pool) != SSLM_OK) return false;
		const sslm_config cfg{1, chunk, layers, 0u};
		const size_t wsz = sslm_workspace_size(model, &cfg);
		ws_buf = ::operator new(wsz, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		return sslm_workspace_create(model, &cfg, ws_buf, wsz, &ws) == SSLM_OK;
	}
	~Engine() {
		if (ws) sslm_workspace_destroy(ws);
		if (pool) sslm_kv_pool_destroy(pool);
		if (model) sslm_model_unmap(model);
		if (ws_buf) ::operator delete(ws_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (pool_buf) ::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (bytes) ::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	std::vector<int32_t> Prompt(int n) const {
		std::vector<int32_t> p;
		for (int i = 0; i < n; ++i) p.push_back(static_cast<int32_t>((i * 37 + 11) % vocab));
		return p;
	}
	void Install(const sslm_parallel_for* pf) {
		if (sslm_workspace_set_parallel_for(ws, pf) != SSLM_OK) {
			std::fprintf(stderr, "hook refused (reserved %#x): this library lacks SSLM_PARALLEL_FOR_MATVEC?\n",
			             pf ? pf->reserved : 0u);
			std::exit(2);
		}
	}
	// Prefills `prompt` (chunk `chunk`), then decodes `n` tokens. Returns the decode seconds.
	double Run(const std::vector<int32_t>& prompt, int32_t chunk, int n, std::vector<int32_t>* toks,
	           std::vector<uint8_t>* blob) {
		sslm_seq seq = nullptr;
		if (sslm_seq_create(model, &pool, &seq) != SSLM_OK) std::exit(3);
		for (size_t at = 0; at < prompt.size();) {
			const int32_t k = static_cast<int32_t>(std::min<size_t>(chunk, prompt.size() - at));
			int32_t consumed = 0;
			if (sslm_prefill(model, seq, prompt.data() + at, k, k, SSLM_SPAN_PROMPT, ws, &consumed) != SSLM_OK)
				std::exit(4);
			at += static_cast<size_t>(consumed);
		}
		sslm_decode_params p{};
		sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, layers, &p);
		int32_t tok = -1;
		sslm_decode_step_v2(model, &seq, 1, &p, ws, &tok);  // the prompt's own finish, untimed
		if (toks) toks->push_back(tok);
		const auto t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < n; ++i) {
			if (sslm_decode_step_v2(model, &seq, 1, &p, ws, &tok) != SSLM_OK) std::exit(5);
			if (toks) toks->push_back(tok);
		}
		const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		if (blob) {
			blob->assign(sslm_seq_state_size(model), 0);
			size_t len = blob->size();
			sslm_seq_save(seq, blob->data(), &len);
			blob->resize(len);
		}
		sslm_seq_release(seq);
		return s;
	}
};

struct Stats {
	double median, q1, q3;
};
Stats Summarize(std::vector<double> v) {
	std::sort(v.begin(), v.end());
	auto at = [&](double q) {
		const double pos = q * static_cast<double>(v.size() - 1);
		const size_t i = static_cast<size_t>(pos);
		const double f = pos - static_cast<double>(i);
		return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
	};
	return Stats{at(0.5), at(0.25), at(0.75)};
}

const char* Arg(int argc, char** argv, const char* name, const char* def) {
	for (int i = 3; i + 1 < argc; ++i)
		if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
	return def;
}

int Verify(Engine& e, int argc, char** argv) {
	const int prompt_n = std::atoi(Arg(argc, argv, "--prompt", "300"));
	const int decode_n = std::atoi(Arg(argc, argv, "--decode", "64"));
	const std::vector<int> tasks = ParseList(Arg(argc, argv, "--tasks", "1,2,3,4,8"));
	const std::vector<int32_t> prompt = e.Prompt(prompt_n);
	std::vector<int32_t> ref_toks;
	std::vector<uint8_t> ref_blob;
	e.Install(nullptr);
	e.Run(prompt, 64, decode_n, &ref_toks, &ref_blob);
	int bad = 0;
	for (int mt : tasks) {
		ReferenceParallelFor pool(std::max(0, mt - 1));
		CountingPool counting;
		counting.pool = &pool;
		const sslm_parallel_for pf{&CountingPool::Run, &counting, mt, SSLM_PARALLEL_FOR_MATVEC};
		e.Install(&pf);
		std::vector<int32_t> toks;
		std::vector<uint8_t> blob;
		counting.calls = 0;
		e.Run(prompt, 64, decode_n, &toks, &blob);
		const long long calls = counting.calls.load();
		// One run over the prompt finish plus `decode_n` tokens; the prompt's prefill (chunks of 64) makes none.
		const bool equal = toks == ref_toks && blob == ref_blob;
		bad += equal ? 0 : 1;
		std::printf("verify max_tasks %d tokens_equal %d blob_equal %d run_calls %lld run_calls_per_decode_token %.2f\n",
		            mt, toks == ref_toks ? 1 : 0, blob == ref_blob ? 1 : 0, calls,
		            decode_n ? static_cast<double>(calls - (mt >= 2 ? 1 : 0)) / decode_n : 0.0);
		e.Install(nullptr);
	}
	std::printf("verify complete %d mismatches\n", bad);
	return bad == 0 ? 0 : 1;
}

int Time(Engine& e, int argc, char** argv) {
	const std::vector<int> contexts = ParseList(Arg(argc, argv, "--contexts", "16,300"));
	const int decode_n = std::atoi(Arg(argc, argv, "--decode", "32"));
	const std::vector<int> tasks = ParseList(Arg(argc, argv, "--tasks", "2,4,8"));
	const int pairs = std::atoi(Arg(argc, argv, "--pairs", "15"));
	const char* mrb = Arg(argc, argv, "--min-row-bytes", nullptr);
	std::vector<int> thresholds = mrb ? ParseList(mrb) : std::vector<int>{0};
#if !defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
	if (mrb) {
		std::fprintf(stderr, "--min-row-bytes needs the seam build (sslm_decode_threading_bench_seams)\n");
		return 2;
	}
#endif
	for (int threshold : thresholds) {
#if defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
		if (threshold > 0) superslm::test::SetMinRowBytesPerTask(static_cast<size_t>(threshold));
		else superslm::test::ResetMinRowBytesPerTask();
#endif
		for (int ctx : contexts) {
			const std::vector<int32_t> prompt = e.Prompt(ctx);
			for (int mt : tasks) {
				ReferenceParallelFor pool(std::max(0, mt - 1));
				const sslm_parallel_for off = pool.Hook(mt);
				sslm_parallel_for on = off;
				on.reserved = SSLM_PARALLEL_FOR_MATVEC;
				std::vector<double> tps_off, tps_on;
				for (int i = 0; i < pairs; ++i) {
					e.Install(&off);
					tps_off.push_back(decode_n / e.Run(prompt, 64, decode_n, nullptr, nullptr));
					e.Install(&on);
					tps_on.push_back(decode_n / e.Run(prompt, 64, decode_n, nullptr, nullptr));
				}
				e.Install(nullptr);
				const Stats a = Summarize(tps_off), b = Summarize(tps_on);
				std::printf("time min_row_bytes %d context %d max_tasks %d n %d off_tok_s_median %.3f off_iqr %.3f..%.3f "
				            "on_tok_s_median %.3f on_iqr %.3f..%.3f ratio %.3f\n",
				            threshold, ctx, mt, pairs, a.median, a.q1, a.q3, b.median, b.q1, b.q3, b.median / a.median);
			}
		}
	}
	std::printf("time complete\n");
	return 0;
}

int OneToken(Engine& e, int argc, char** argv) {
	const int prompt_n = std::atoi(Arg(argc, argv, "--prompt", "64"));
	const std::vector<int> tasks = ParseList(Arg(argc, argv, "--tasks", "4"));
	const int pairs = std::atoi(Arg(argc, argv, "--pairs", "15"));
	const std::vector<int32_t> prompt = e.Prompt(prompt_n);
	for (int mt : tasks) {
		ReferenceParallelFor pool(std::max(0, mt - 1));
		const sslm_parallel_for off = pool.Hook(mt);
		sslm_parallel_for on = off;
		on.reserved = SSLM_PARALLEL_FOR_MATVEC;
		std::vector<double> ms_off, ms_on;
		for (int i = 0; i < pairs; ++i) {
			for (int side = 0; side < 2; ++side) {
				e.Install(side ? &on : &off);
				sslm_seq seq = nullptr;
				if (sslm_seq_create(e.model, &e.pool, &seq) != SSLM_OK) return 3;
				const auto t0 = std::chrono::steady_clock::now();
				for (int32_t tok : prompt) {
					int32_t consumed = 0;
					if (sslm_prefill(e.model, seq, &tok, 1, 1, SSLM_SPAN_PROMPT, e.ws, &consumed) != SSLM_OK) return 4;
				}
				const double ms =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / prompt_n;
				(side ? ms_on : ms_off).push_back(ms);
				sslm_seq_release(seq);
			}
		}
		e.Install(nullptr);
		const Stats a = Summarize(ms_off), b = Summarize(ms_on);
		std::printf("onetoken max_tasks %d n %d off_ms_per_token_median %.3f off_iqr %.3f..%.3f on_ms_per_token_median "
		            "%.3f on_iqr %.3f..%.3f\n",
		            mt, pairs, a.median, a.q1, a.q3, b.median, b.q1, b.q3);
	}
	std::printf("onetoken complete\n");
	return 0;
}

int Groups(int argc, char** argv) {
	const std::vector<int> tasks = ParseList(Arg(argc, argv, "--tasks", "2,4"));
	const int layers = std::atoi(Arg(argc, argv, "--layers", "24"));
	const int pairs = std::atoi(Arg(argc, argv, "--pairs", "15"));
	struct Group {
		const char* name;
		size_t k;
		std::vector<size_t> rows;
	};
	const Group groups[] = {{"q", 896, {896}},
	                        {"k+v", 896, {128, 128}},
	                        {"o", 896, {896}},
	                        {"gate+up", 896, {4864, 4864}},
	                        {"down", 4864, {896}}};
	// One weight buffer per (layer, group, part); deterministic bytes.
	std::vector<std::vector<std::vector<int8_t>>> w(static_cast<size_t>(layers) * 5);
	uint32_t seed = 12345;
	for (int l = 0; l < layers; ++l) {
		for (size_t g = 0; g < 5; ++g) {
			for (size_t rows : groups[g].rows) {
				std::vector<int8_t> m(rows * groups[g].k);
				for (int8_t& b : m) {
					seed = seed * 1664525u + 1013904223u;
					b = static_cast<int8_t>(static_cast<int>(seed >> 24) - 128);
				}
				w[static_cast<size_t>(l) * 5 + g].push_back(std::move(m));
			}
		}
	}
	std::vector<int8_t> x(4864);
	for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<int8_t>(static_cast<int>(i * 37 % 255) - 127);
	std::vector<int64_t> out_a(9728), out_b(9728);
	// One pass over every layer's five groups; returns milliseconds per layer and checks equality.
	auto pass = [&](const sslm_parallel_for* pf, long long* calls, bool* equal) {
		const auto t0 = std::chrono::steady_clock::now();
		for (int l = 0; l < layers; ++l) {
			for (size_t g = 0; g < 5; ++g) {
				const auto& parts = w[static_cast<size_t>(l) * 5 + g];
				size_t total = 0;
				for (size_t r : groups[g].rows) total += r;
				superslm::MatvecPart mp[2];
				size_t off = 0;
				for (size_t i = 0; i < parts.size(); ++i) {
					mp[i] = superslm::MatvecPart{parts[i].data(), groups[g].rows[i], out_b.data() + off};
					off += groups[g].rows[i];
				}
				const superslm::ColumnSplit split = superslm::MatvecGroupSplit(pf, groups[g].k, total);
				if (split.task_count >= 2) {
					++*calls;
					superslm::MatvecGroupParallel(*pf, split, x.data(), groups[g].k, mp, parts.size());
				} else {
					for (size_t i = 0; i < parts.size(); ++i)
						superslm::GemmInt8AccumulateRow(x.data(), mp[i].weight, groups[g].k, mp[i].rows, mp[i].out);
				}
				if (equal && l == layers - 1) {
					size_t o2 = 0;
					for (size_t i = 0; i < parts.size(); ++i) {
						superslm::GemmInt8AccumulateRow(x.data(), parts[i].data(), groups[g].k, groups[g].rows[i],
						                                out_a.data() + o2);
						o2 += groups[g].rows[i];
					}
					*equal = *equal && std::equal(out_a.begin(), out_a.begin() + total, out_b.begin());
				}
			}
		}
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / layers;
	};
	for (int mt : tasks) {
		ReferenceParallelFor pool(std::max(0, mt - 1));
		const sslm_parallel_for pf = pool.Hook(mt);
		std::vector<double> serial, split;
		long long calls = 0;
		bool equal = true;
		for (int i = 0; i < pairs; ++i) {
			long long dummy = 0;
			serial.push_back(pass(nullptr, &dummy, nullptr));
			calls = 0;
			split.push_back(pass(&pf, &calls, &equal));
		}
		const Stats a = Summarize(serial), b = Summarize(split);
		std::printf("groups max_tasks %d layers %d n %d serial_ms_per_layer_median %.4f iqr %.4f..%.4f "
		            "split_ms_per_layer_median %.4f iqr %.4f..%.4f run_calls_per_layer %.2f rows_equal %d "
		            "projected_ms_saved_per_token_24_layers %.2f\n",
		            mt, layers, pairs, a.median, a.q1, a.q3, b.median, b.q1, b.q3,
		            static_cast<double>(calls) / layers, equal ? 1 : 0, 24 * (a.median - b.median));
	}
	std::printf("groups complete\n");
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s ARTIFACT verify|time|onetoken [options], or %s - groups [options]\n", argv[0], argv[0]);
		return 2;
	}
	if (std::strcmp(argv[2], "groups") == 0) return Groups(argc, argv);
	Engine e;
	if (!e.Open(argv[1], 64)) return 2;
#if defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
	std::printf("build seams 1\n");
#else
	std::printf("build seams 0\n");
#endif
	std::printf("artifact %s layers %d vocab %d\n", argv[1], e.layers, e.vocab);
	const std::string mode = argv[2];
	if (mode == "verify") return Verify(e, argc, argv);
	if (mode == "time") return Time(e, argc, argv);
	if (mode == "onetoken") return OneToken(e, argc, argv);
	std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
	return 2;
}
