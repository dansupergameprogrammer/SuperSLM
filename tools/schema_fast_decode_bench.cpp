// Schema fast-decode plan (rev 1.2.1) §9 (slice J1) and §10 (box steps B1-B3): the bench, headless. It
// drives the public C ABI only and prints plain `key value` lines a driver can parse. Every figure it
// prints off the box is indicative; the plan's readings are the box's (§10).
//
//   schema_fast_decode_bench ARTIFACT extract SCHEMA [--prompt TEXT] [--budget 64] [--pairs 15]
//                            [--tasks 0] [--max-new 256]
//       B2 (and B3 with --tasks 4): the extraction from a fixed prompt, jump-forward off (the plain
//       greedy decode loop) against on (the §5.6 host loop: drain the chain with sslm_seq_jump_forward
//       until sslm_seq_forced_tokens reports 0, then decode), as n interleaved off/on pairs. Prints
//       each arm's token count, layer passes, finishes, a stream digest (FNV-1a 64 over LE32 ids) and
//       the median and IQR of its milliseconds. Off and on may legitimately differ in stream (the
//       contract choice of §5.2); both must end at the -2 dead end or the cap. --tasks N installs the
//       reference std::thread pool with SSLM_PARALLEL_FOR_MATVEC at max_tasks N (0: no hook).
//   schema_fast_decode_bench ARTIFACT spans SCHEMA [--prompt TEXT] [--pairs 15] [--tasks 0]
//       B3's span table: for each span the on-arm admits, its size M and the median milliseconds of
//       the verb call that admitted it, beside M one-token prefills of the same tokens (§16 Q6).
//   schema_fast_decode_bench ARTIFACT pass [--sizes 1,2,3,4,5,6,7,8] [--depth 100] [--pairs 15]
//                            [--tasks 0]
//       B1: the real batched pass (every site) for M tokens at prompt depth `depth`, through the
//       prefill path the verb uses, median and IQR of n runs per M, and r(M) = t(M) / t(1).
//   schema_fast_decode_bench ARTIFACT finish SCHEMA [--prompt TEXT] [--pairs 15] [--tasks 0]
//       B1's finish: the whole finish (rows, re-mask, argmax) of a finish-only decode, greedy and
//       damped greedy (when the artifact carries DGC1): unbound (all V rows), and at every choice the
//       jump-forward run reaches, the last being the dead end (0 rows). The row count of a bound state
//       is not observable through the ABI; each finish is printed with its index and the token.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "superslm/layer_marshal.h"
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "../docs/parallel_for_reference.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
	return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Stats {
	double median, q1, q3;
};
Stats Summarize(std::vector<double> v) {
	if (v.empty()) return Stats{0, 0, 0};
	std::sort(v.begin(), v.end());
	auto at = [&](double q) {
		const double pos = q * static_cast<double>(v.size() - 1);
		const size_t i = static_cast<size_t>(pos);
		const double f = pos - static_cast<double>(i);
		return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
	};
	return Stats{at(0.5), at(0.25), at(0.75)};
}

std::vector<int> ParseList(const char* s) {
	std::vector<int> out;
	for (const char* p = s; *p;) {
		out.push_back(std::atoi(p));
		while (*p && *p != ',') ++p;
		if (*p == ',') ++p;
	}
	return out;
}

const char* Arg(int argc, char** argv, int first, const char* name, const char* def) {
	for (int i = first; i + 1 < argc; ++i)
		if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
	return def;
}

uint64_t Digest(const std::vector<int32_t>& ids) {
	uint64_t h = 1469598103934665603ull;
	for (int32_t x : ids) {
		const uint32_t u = static_cast<uint32_t>(x);
		for (int b = 0; b < 4; ++b) {
			h ^= (u >> (8 * b)) & 0xFFu;
			h *= 1099511628211ull;
		}
	}
	return h;
}

struct Engine {
	uint8_t* bytes = nullptr;
	size_t size = 0;
	sslm_model model = nullptr;
	int32_t layers = 0;
	int32_t cap_positions = 0;
	void* pool_buf = nullptr;
	sslm_kv_pool pool = nullptr;
	void* ws_buf = nullptr;
	sslm_workspace ws = nullptr;
	ReferenceParallelFor* threads = nullptr;
	sslm_parallel_for pf{};

	bool Open(const char* path, int tasks) {
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
		const uint32_t blocks = 2;
		const size_t psize = blocks * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, blocks);
		pool_buf = ::operator new(psize, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (sslm_kv_pool_create(model, pool_buf, psize, blocks, &pool) != SSLM_OK) return false;
		const sslm_config cfg{1, 64, layers, 0u};
		const size_t wsz = sslm_workspace_size(model, &cfg);
		ws_buf = ::operator new(wsz, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (sslm_workspace_create(model, &cfg, ws_buf, wsz, &ws) != SSLM_OK) return false;
		if (tasks >= 2) {
			threads = new ReferenceParallelFor(tasks - 1);
			pf = sslm_parallel_for{&ReferenceParallelFor::Run, threads, tasks, SSLM_PARALLEL_FOR_MATVEC};
			if (sslm_workspace_set_parallel_for(ws, &pf) != SSLM_OK) {
				std::fprintf(stderr, "hook refused\n");
				return false;
			}
		}
		return true;
	}
	~Engine() {
		if (ws) sslm_workspace_destroy(ws);
		delete threads;
		if (pool) sslm_kv_pool_destroy(pool);
		if (model) sslm_model_unmap(model);
		if (ws_buf) ::operator delete(ws_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (pool_buf) ::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (bytes) ::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	sslm_decode_params Greedy() const {
		sslm_decode_params p{};
		sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, layers, &p);
		return p;
	}
	bool Damped(sslm_decode_params* p) const {
		return sslm_decode_params_init(model, SSLM_DECODE_MODE_DAMPED_GREEDY, layers, p) == SSLM_OK;
	}
	std::vector<int32_t> Tokenize(const char* text) const {
		int32_t n = 0;
		sslm_tokenize(model, text, nullptr, &n);
		std::vector<int32_t> t(static_cast<size_t>(n > 0 ? n : 1));
		if (sslm_tokenize(model, text, t.data(), &n) != SSLM_OK) {
			std::fprintf(stderr, "sslm_tokenize failed\n");
			std::exit(3);
		}
		t.resize(static_cast<size_t>(n));
		return t;
	}
	// A sequence bound to `schema` (NULL: unbound) with `prompt` prefilled.
	sslm_seq Start(const char* schema, const std::vector<int32_t>& prompt) {
		sslm_seq s = nullptr;
		if (sslm_seq_create(model, &pool, &s) != SSLM_OK) std::exit(4);
		if (schema) {
			sslm_schema sc = nullptr;
			if (sslm_schema_lookup(model, schema, &sc) != SSLM_OK || sslm_seq_set_schema(s, sc) != SSLM_OK) {
				std::fprintf(stderr, "schema %s not found or not bindable\n", schema);
				std::exit(4);
			}
		}
		for (size_t at = 0; at < prompt.size();) {
			const int32_t k = static_cast<int32_t>(std::min<size_t>(64, prompt.size() - at));
			int32_t consumed = 0;
			if (sslm_prefill(model, s, prompt.data() + at, k, k, SSLM_SPAN_PROMPT, ws, &consumed) != SSLM_OK) std::exit(4);
			at += static_cast<size_t>(consumed);
		}
		return s;
	}
};

struct Arm {
	std::vector<int32_t> stream;
	std::vector<int> spans;          // M per verb admission (pending token included)
	std::vector<double> span_ms;     // the verb call's milliseconds, per admission
	std::vector<std::vector<int32_t>> span_tokens;  // the tokens each admission embedded
	int layer_passes = 0;            // batched passes + layer-running decodes
	int finishes = 0;
	bool dead_end = false;
	double ms = 0;
};

// Jump-forward off: the plain greedy loop until -2 or `max_new` tokens.
Arm RunOff(Engine& e, const char* schema, const std::vector<int32_t>& prompt, int max_new) {
	Arm a;
	sslm_seq s = e.Start(schema, prompt);
	const sslm_decode_params g = e.Greedy();
	const auto t0 = Clock::now();
	for (int i = 0; i <= max_new; ++i) {
		int32_t t = -100;
		if (sslm_decode_step_v2(e.model, &s, 1, &g, e.ws, &t) != SSLM_OK) break;
		++a.finishes;
		if (i > 0) ++a.layer_passes;  // the first decode finishes the prompt's last token only
		if (t == -2) {
			a.dead_end = true;
			break;
		}
		if (t < 0) break;
		a.stream.push_back(t);
	}
	a.ms = MsSince(t0);
	sslm_seq_release(s);
	return a;
}

// Jump-forward on: the §5.6 host loop.
Arm RunOn(Engine& e, const char* schema, const std::vector<int32_t>& prompt, int32_t budget, int max_new) {
	Arm a;
	sslm_seq s = e.Start(schema, prompt);
	const sslm_decode_params g = e.Greedy();
	std::vector<int32_t> buf(static_cast<size_t>(budget));
	bool pending = false;  // the last decode's token is not yet admitted
	const auto t0 = Clock::now();
	while (static_cast<int>(a.stream.size()) <= max_new) {
		for (;;) {
			int32_t count = 0, accepting = 0;
			if (sslm_seq_forced_tokens(e.model, s, nullptr, 0, &count, &accepting) != SSLM_OK || count == 0) break;
			int32_t forced = 0;
			const auto v0 = Clock::now();
			const sslm_status st = sslm_seq_jump_forward(e.model, s, budget, nullptr, e.ws, buf.data(), budget, &forced);
			const double v_ms = MsSince(v0);
			if (st != SSLM_OK) {
				std::fprintf(stderr, "jump_forward status %d\n", static_cast<int>(st));
				goto done;
			}
			const int m = forced + (pending ? 1 : 0);
			if (m > 0) {
				a.spans.push_back(m);
				a.span_ms.push_back(v_ms);
				std::vector<int32_t> toks;
				if (pending) toks.push_back(a.stream.back());
				toks.insert(toks.end(), buf.begin(), buf.begin() + forced);
				a.span_tokens.push_back(toks);
				++a.layer_passes;
			}
			pending = false;
			a.stream.insert(a.stream.end(), buf.begin(), buf.begin() + forced);
		}
		{
			int32_t t = -100;
			if (sslm_decode_step_v2(e.model, &s, 1, &g, e.ws, &t) != SSLM_OK) break;
			++a.finishes;
			if (pending) ++a.layer_passes;  // a choice after a choice embeds the pending token here
			if (t == -2) {
				a.dead_end = true;
				break;
			}
			if (t < 0) break;
			a.stream.push_back(t);
			pending = true;
		}
	}
done:
	a.ms = MsSince(t0);
	sslm_seq_release(s);
	return a;
}

void PrintArm(const char* name, const Arm& a, const std::vector<double>& ms) {
	const Stats st = Summarize(ms);
	std::printf("%s tokens %zu layer_passes %d finishes %d dead_end %d digest %016llx ms_median %.3f ms_q1 %.3f ms_q3 %.3f\n",
	            name, a.stream.size(), a.layer_passes, a.finishes, a.dead_end ? 1 : 0,
	            static_cast<unsigned long long>(Digest(a.stream)), st.median, st.q1, st.q3);
}

int Extract(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const std::vector<int32_t> prompt = e.Tokenize(Arg(argc, argv, 4, "--prompt", "Customer: I would like to buy a health potion, please.\nOrder:"));
	const int32_t budget = std::atoi(Arg(argc, argv, 4, "--budget", "64"));
	const int pairs = std::atoi(Arg(argc, argv, 4, "--pairs", "15"));
	const int max_new = std::atoi(Arg(argc, argv, 4, "--max-new", "256"));
	std::vector<double> off_ms, on_ms;
	Arm off, on;
	for (int i = 0; i < pairs; ++i) {
		off = RunOff(e, schema, prompt, max_new);
		on = RunOn(e, schema, prompt, budget, max_new);
		off_ms.push_back(off.ms);
		on_ms.push_back(on.ms);
	}
	std::printf("schema %s prompt_tokens %zu budget %d pairs %d\n", schema, prompt.size(), budget, pairs);
	PrintArm("off", off, off_ms);
	PrintArm("on", on, on_ms);
	std::printf("speedup_median %.3f\n", Summarize(on_ms).median > 0 ? Summarize(off_ms).median / Summarize(on_ms).median : 0.0);
	std::printf("spans");
	for (int m : on.spans) std::printf(" %d", m);
	std::printf("\n");
	return (off.dead_end || static_cast<int>(off.stream.size()) >= max_new) && on.dead_end ? 0 : 1;
}

int Spans(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const std::vector<int32_t> prompt = e.Tokenize(Arg(argc, argv, 4, "--prompt", "Customer: I would like to buy a health potion, please.\nOrder:"));
	const int pairs = std::atoi(Arg(argc, argv, 4, "--pairs", "15"));
	std::vector<std::vector<double>> per_span;
	Arm on;
	for (int i = 0; i < pairs; ++i) {
		on = RunOn(e, schema, prompt, 64, 256);
		if (per_span.empty()) per_span.resize(on.span_ms.size());
		for (size_t k = 0; k < on.span_ms.size() && k < per_span.size(); ++k) per_span[k].push_back(on.span_ms[k]);
	}
	// The same spans as M one-token prefills of the same tokens, in order on one sequence (prompt spans:
	// the walk is not the point here, the M one-row passes are).
	std::vector<std::vector<double>> per_span_one(per_span.size());
	for (int i = 0; i < pairs; ++i) {
		sslm_seq s = e.Start(nullptr, prompt);
		for (size_t k = 0; k < on.span_tokens.size() && k < per_span_one.size(); ++k) {
			const auto t0 = Clock::now();
			for (int32_t tok : on.span_tokens[k]) {
				int32_t consumed = 0;
				sslm_prefill(e.model, s, &tok, 1, 1, SSLM_SPAN_PROMPT, e.ws, &consumed);
			}
			per_span_one[k].push_back(MsSince(t0));
		}
		sslm_seq_release(s);
	}
	for (size_t k = 0; k < per_span.size(); ++k) {
		const Stats a = Summarize(per_span[k]), b = Summarize(per_span_one[k]);
		std::printf("span %zu M %d verb_ms_median %.3f one_token_ms_median %.3f\n", k, on.spans[k], a.median, b.median);
	}
	return 0;
}

int Pass(Engine& e, int argc, char** argv) {
	const std::vector<int> sizes = ParseList(Arg(argc, argv, 3, "--sizes", "1,2,3,4,5,6,7,8"));
	const int depth = std::atoi(Arg(argc, argv, 3, "--depth", "100"));
	const int pairs = std::atoi(Arg(argc, argv, 3, "--pairs", "15"));
	std::vector<int32_t> prompt;
	for (int i = 0; i < depth; ++i) prompt.push_back((i * 37 + 11) % 1000 + 100);
	double t1 = 0;
	for (int m : sizes) {
		std::vector<int32_t> span;
		for (int i = 0; i < m; ++i) span.push_back((i * 53 + 7) % 1000 + 100);
		std::vector<double> ms;
		for (int i = 0; i < pairs; ++i) {
			sslm_seq s = e.Start(nullptr, prompt);
			int32_t consumed = 0;
			const auto t0 = Clock::now();
			sslm_prefill(e.model, s, span.data(), m, m, SSLM_SPAN_PROMPT, e.ws, &consumed);
			ms.push_back(MsSince(t0));
			sslm_seq_release(s);
		}
		const Stats st = Summarize(ms);
		if (m == 1) t1 = st.median;
		std::printf("pass M %d depth %d ms_median %.3f ms_q1 %.3f ms_q3 %.3f r %.3f\n", m, depth, st.median, st.q1, st.q3,
		            t1 > 0 ? st.median / t1 : 0.0);
	}
	return 0;
}

int Finish(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const std::vector<int32_t> prompt = e.Tokenize(Arg(argc, argv, 4, "--prompt", "Customer: I would like to buy a health potion, please.\nOrder:"));
	const int pairs = std::atoi(Arg(argc, argv, 4, "--pairs", "15"));
	sslm_decode_params modes[2];
	modes[0] = e.Greedy();
	const bool damped = e.Damped(&modes[1]);
	for (int mi = 0; mi < (damped ? 2 : 1); ++mi) {
		const char* mname = mi == 0 ? "greedy" : "damped";
		// Unbound: all V rows. Each sample is a fresh prefill and one finish-only decode.
		std::vector<double> ms;
		for (int i = 0; i < pairs; ++i) {
			sslm_seq s = e.Start(nullptr, prompt);
			int32_t t = -100;
			const auto t0 = Clock::now();
			sslm_decode_step_v2(e.model, &s, 1, &modes[mi], e.ws, &t);
			ms.push_back(MsSince(t0));
			sslm_seq_release(s);
		}
		const Stats st = Summarize(ms);
		std::printf("finish %s unbound ms_median %.3f ms_q1 %.3f ms_q3 %.3f\n", mname, st.median, st.q1, st.q3);
		// Bound: every finish-only decode of the jump-forward run, by index.
		std::vector<std::vector<double>> per;
		std::vector<int32_t> toks;
		for (int i = 0; i < pairs; ++i) {
			sslm_seq s = e.Start(schema, prompt);
			std::vector<int32_t> buf(64);
			for (size_t k = 0;; ++k) {
				for (;;) {
					int32_t count = 0, acc = 0, forced = 0;
					if (sslm_seq_forced_tokens(e.model, s, nullptr, 0, &count, &acc) != SSLM_OK || count == 0) break;
					if (sslm_seq_jump_forward(e.model, s, 64, mi ? &modes[mi] : nullptr, e.ws, buf.data(), 64, &forced) != SSLM_OK) break;
					if (forced == 0) break;
				}
				int32_t t = -100;
				const auto t0 = Clock::now();
				const sslm_status st2 = sslm_decode_step_v2(e.model, &s, 1, &modes[mi], e.ws, &t);
				const double d = MsSince(t0);
				if (per.size() <= k) per.resize(k + 1);
				per[k].push_back(d);
				if (i == 0) toks.push_back(t);
				if (st2 != SSLM_OK || t < 0 || k > 256) break;
			}
			sslm_seq_release(s);
		}
		for (size_t k = 0; k < per.size(); ++k) {
			const Stats b = Summarize(per[k]);
			std::printf("finish %s %s index %zu token %d ms_median %.3f\n", mname, schema, k, k < toks.size() ? toks[k] : -100,
			            b.median);
		}
	}
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s ARTIFACT extract|spans|finish SCHEMA [options], or %s ARTIFACT pass [options]\n",
		             argv[0], argv[0]);
		return 2;
	}
	const std::string mode = argv[2];
	const bool needs_schema = mode != "pass";
	if (needs_schema && argc < 4) {
		std::fprintf(stderr, "%s needs a schema name\n", mode.c_str());
		return 2;
	}
	const int tasks = std::atoi(Arg(argc, argv, 3, "--tasks", "0"));
	Engine e;
	if (!e.Open(argv[1], tasks)) return 2;
	std::printf("artifact %s layers %d tasks %d\n", argv[1], e.layers, tasks);
	if (mode == "extract") return Extract(e, argc, argv);
	if (mode == "spans") return Spans(e, argc, argv);
	if (mode == "pass") return Pass(e, argc, argv);
	if (mode == "finish") return Finish(e, argc, argv);
	std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
	return 2;
}
