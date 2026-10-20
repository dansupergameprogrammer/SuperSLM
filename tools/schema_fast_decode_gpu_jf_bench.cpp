// Schema fast-decode plan §7.1, GX.3 follow-up, headless: jump-forward off against on, on the D3D12 GPU backend, at
// whole-token depth. The example's extraction (potion_shop_order, the scene's default utterance) through the GPU G5
// bridges, n interleaved pairs after `warmup` unmeasured ones, the arm that runs first alternating per pair.
//
//   off  SslmGpuSeqDecodeStepForG5Bridge once per token (embed, full depth in one slice, finish), until the schema walk
//        accepts, the -2 dead end, or max-new tokens.
//   on   the §5.6 host loop on the GPU: whenever the walk state has a forced chain, the chain goes to the device in one
//        SslmGpuSeqPrefillSchemaContentForG5Bridge call (sub-chunks of at most 4 tokens) and the next decode step only
//        finishes its last token. A token the finish produced is walked but not yet run; when a chain follows it, it
//        runs first through SslmGpuSeqPrefillPromptForG5Bridge, because the schema-content bridge checks every token
//        against the walk state and has no pre-walked prefix (the CPU verb's `prewalked_prefix`).
//
// The forced chain is the library's own: the CPU model's forced-next table (the §11.0 read seam ForcedTableEntry, built
// at map from SCM1 and TOK1), followed through the SCM1 transitions, from the GPU sequence's walk state. The CPU model is
// only read: no CPU decode runs. The prompt is tokenized by the CPU ABI (sslm_tokenize), as the CPU bench does.
//
// Timed: generation only, from the first call after the prompt prefill to the stop (the prompt prefill is untimed and
// identical in both arms). Per run: wall ms, the sum of superslm_gpu::LastCallTiming().gpu_busy_ms over the run's
// layer-loop submissions (the finish's logits are not in it), submissions, tokens, ms per token, the token stream and
// its FNV-1a 64 digest. Per pair, R = off / on wall.
//
// Usage: sslm_schema_fast_decode_gpu_jf_bench <model.sslm> [--pairs 15] [--warmup 1] [--max-new 64] [--device-head] [--json PATH]
// --device-head maps the model with SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE (the packaged example's residency): each finish
// computes the logits row on the device. Without it the host computes it (LogitsSiteParallel over all V rows).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/schema_masks.h"
#include "superslm/sslm_abi.h"
#include "schema_fast_decode/sfd_api.h"


namespace {

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

const char* Arg(int argc, char** argv, const char* name, const char* def) {
	for (int i = 2; i + 1 < argc; ++i)
		if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
	return def;
}
[[noreturn]] void Die(const char* what, int st) {
	std::fprintf(stderr, "%s failed: status %d\n", what, st);
	std::exit(3);
}
void Ok(SslmGpuStatus st, const char* what) {  // GPU statuses are qualified: sslm_abi.h has its own SSLM_OK
	if (st != SslmGpuStatus::SSLM_OK) Die(what, static_cast<int>(st));
}

struct Stat {
	size_t n = 0;
	double median = 0, q1 = 0, q3 = 0, lo = 0, hi = 0;
};
double Quantile(std::vector<double> v, double q) {
	std::sort(v.begin(), v.end());
	const double pos = q * static_cast<double>(v.size() - 1);
	const size_t i = static_cast<size_t>(pos);
	const double f = pos - static_cast<double>(i);
	return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
}
Stat Summarize(const std::vector<double>& v) {
	Stat s;
	s.n = v.size();
	if (v.empty()) return s;
	s.median = Quantile(v, 0.5);
	s.q1 = Quantile(v, 0.25);
	s.q3 = Quantile(v, 0.75);
	std::vector<double> sorted = v;
	std::sort(sorted.begin(), sorted.end());
	if (sorted.size() == 15) {  // the n = 15 sign-test interval [R(4), R(12)]
		s.lo = sorted[3];
		s.hi = sorted[11];
	} else {
		s.lo = s.q1;
		s.hi = s.q3;
	}
	return s;
}
std::string JStat(const Stat& s) {
	char b[256];
	std::snprintf(b, sizeof b, "{\"n\":%zu,\"median\":%.4f,\"q1\":%.4f,\"q3\":%.4f,\"sign_lo\":%.4f,\"sign_hi\":%.4f}", s.n,
	              s.median, s.q1, s.q3, s.lo, s.hi);
	return b;
}
uint64_t Fnv(const std::vector<int32_t>& v) {
	uint64_t h = 1469598103934665603ull;
	for (int32_t x : v) {
		for (int b = 0; b < 4; ++b) {
			h ^= static_cast<uint8_t>((static_cast<uint32_t>(x) >> (8 * b)) & 0xFF);
			h *= 1099511628211ull;
		}
	}
	return h;
}
std::string JIds(const std::vector<int32_t>& v) {
	std::string s = "[";
	for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s + "]";
}

struct Run {
	double wall = 0, gpu = 0;
	int submits = 0;
	std::vector<int32_t> stream;
	std::string stop;
};

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [--pairs 15] [--warmup 1] [--max-new 64] [--json PATH]\n", argv[0]);
		return 2;
	}
	const int pairs = std::atoi(Arg(argc, argv, "--pairs", "15"));
	const int warmup = std::atoi(Arg(argc, argv, "--warmup", "1"));
	const int max_new = std::atoi(Arg(argc, argv, "--max-new", "64"));
	const char* json_path = Arg(argc, argv, "--json", nullptr);
	bool device_head = false;
	for (int i = 2; i < argc; ++i)
		if (std::strcmp(argv[i], "--device-head") == 0) device_head = true;

	// The artifact, once, aligned: the CPU ABI maps it (tokenizer, forced table) and the GPU maps its view.
	std::FILE* f = std::fopen(argv[1], "rb");
	if (!f) {
		std::fprintf(stderr, "cannot open %s\n", argv[1]);
		return 2;
	}
	std::fseek(f, 0, SEEK_END);
	const size_t size = static_cast<size_t>(std::ftell(f));
	std::fseek(f, 0, SEEK_SET);
	auto* bytes = static_cast<uint8_t*>(::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
	if (std::fread(bytes, 1, size, f) != size) {
		std::fprintf(stderr, "short read\n");
		return 2;
	}
	std::fclose(f);
	sslm_model cpu = nullptr;
	if (sslm_model_map(bytes, size, &cpu) != SSLM_OK) Die("sslm_model_map", 0);
	superslm::SslmModelView view;
	std::string err;
	if (superslm::SslmModel::Load(bytes, size, view, &err) != superslm::SslmModelStatus::Ok) {
		std::fprintf(stderr, "model load failed: %s\n", err.c_str());
		return 2;
	}
	const superslm::SslmSectionView* scm = view.Section(superslm::SslmSectionType::SchemaMasks);
	superslm::SchemaMasksTable table;
	if (!scm || !superslm::SchemaMasksTable::Parse(scm->data, scm->byte_size, view.config.vocab_size, table, &err)) {
		std::fprintf(stderr, "no schema table: %s\n", err.c_str());
		return 2;
	}
	size_t schema_index = 0;
	const superslm::SchemaEntry* entry = table.ByName("potion_shop_order", &schema_index);
	if (!entry) {
		std::fprintf(stderr, "potion_shop_order not in the artifact\n");
		return 2;
	}

	const std::string prompt_text =
	    "<|im_start|>system\nYou work the counter of a potion shop. Record the customer's order as JSON.<|im_end|>\n"
	    "<|im_start|>user\nHello! Could I get two health potions, please?<|im_end|>\n<|im_start|>assistant\n";
	int32_t n = 0;
	sslm_tokenize(cpu, prompt_text.c_str(), nullptr, &n);
	std::vector<int32_t> prompt(static_cast<size_t>(n > 0 ? n : 1));
	if (sslm_tokenize(cpu, prompt_text.c_str(), prompt.data(), &n) != SSLM_OK) Die("sslm_tokenize", 0);
	prompt.resize(static_cast<size_t>(n));

	// The forced chain from walk state `w`: next_forced(w) from the library's table, then the SCM1 transition.
	auto chain_from = [&](uint32_t w) {
		std::vector<int32_t> out;
		for (size_t guard = 0; guard < 4096; ++guard) {
			int32_t tok = -1;
			uint32_t f_len = 0;
			const sslm_status st = superslm::test::sfd::ForcedTableEntry(cpu, static_cast<int32_t>(schema_index), w, &tok, &f_len);
			if (st != SSLM_OK) Die("ForcedTableEntry", static_cast<int>(st));
			if (tok < 0) break;
			out.push_back(tok);
			uint32_t nx = 0;
			if (!table.Transition(*entry, w, static_cast<uint32_t>(tok), &nx)) break;
			w = nx;
		}
		return out;
	};

	GpuContextConfig cfg{};
	GpuResidencyConfig rcfg{};
	if (device_head) rcfg.flags |= SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
	SslmGpuContext* ctx = nullptr;
	Ok(sslm_gpu_context_create(cfg, &ctx), "sslm_gpu_context_create");
	SslmGpuModelHandle* model = nullptr;
	Ok(sslm_gpu_model_map(ctx, &view, rcfg, &model), "sslm_gpu_model_map");
	const int32_t gpu_schema = SslmGpuSchemaLookupForG5Bridge(model, "potion_shop_order");
	if (gpu_schema != static_cast<int32_t>(schema_index)) {
		std::fprintf(stderr, "schema index differs: GPU %d, SCM1 %zu\n", gpu_schema, schema_index);
		return 2;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	Ok(sslm_gpu_seq_create(ctx, model, static_cast<int64_t>(view.config.context_cap), &seq), "sslm_gpu_seq_create");
	constexpr uint32_t kWhole = 0xFFFFFFFFu;

	auto start = [&]() {
		Ok(sslm_gpu_seq_reset(ctx, seq), "reset");
		Ok(SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, gpu_schema), "set schema");
		Ok(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt.data(), static_cast<int32_t>(prompt.size()), kWhole), "prompt");
	};
	auto accepting = [&]() {
		int32_t a = 0;
		Ok(SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, seq, &a), "accepting");
		return a != 0;
	};
	auto timing = [&](Run& r) {
		r.gpu += superslm_gpu::LastCallTiming().gpu_busy_ms;
		++r.submits;
	};
	// The off arm: one decode step per token. The first finishes the prompt's last token.
	auto run_off = [&]() {
		Run r;
		start();
		int32_t next = 0;
		const auto t0 = Clock::now();
		for (int i = 0; i < max_new; ++i) {
			int32_t tok = -1;
			Ok(SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, next, kWhole, &tok), "off decode");
			if (i > 0) timing(r);  // the first call only finishes the prompt's last token: no layer submission
			if (tok == -2) {
				r.stop = "deadend";
				break;
			}
			r.stream.push_back(tok);
			next = tok;
			if (accepting()) {
				r.stop = "accepting";
				break;
			}
		}
		r.wall = MsSince(t0);
		if (r.stop.empty()) r.stop = "max-new";
		return r;
	};
	// The on arm: a forced chain goes to the device in one schema-content call; the decode step then only finishes.
	auto run_on = [&]() {
		Run r;
		start();
		int32_t pending = -1;  // a produced token that is walked but not yet run
		const auto t0 = Clock::now();
		for (int i = 0; i < max_new && static_cast<int>(r.stream.size()) < max_new;) {
			const std::vector<int32_t> chain = chain_from(SslmGpuSeqWalkStateForG5Bridge(seq));
			int32_t next = pending;
			if (!chain.empty()) {
				if (pending >= 0) {
					Ok(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, &pending, 1, kWhole), "on pending");
					timing(r);
				}
				int32_t consumed = 0;
				Ok(SslmGpuSeqPrefillSchemaContentForG5Bridge(ctx, seq, chain.data(), static_cast<int32_t>(chain.size()), kWhole,
				                                              &consumed),
				   "on chain");
				if (consumed != static_cast<int32_t>(chain.size())) Die("on chain consumed", consumed);
				r.submits += (static_cast<int>(chain.size()) + 3) / 4 - 1;  // sub-chunks beyond the one timing() counts
				timing(r);
				for (int32_t t : chain) r.stream.push_back(t);
				if (accepting()) {
					r.stop = "accepting";
					pending = -1;
					break;
				}
			}
			int32_t tok = -1;
			const bool runs_layers = chain.empty() && pending >= 0;
			Ok(SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, next < 0 ? 0 : next, kWhole, &tok), "on decode");
			if (runs_layers) timing(r);
			++i;
			if (tok == -2) {
				r.stop = "deadend";
				break;
			}
			r.stream.push_back(tok);
			pending = tok;
			if (accepting()) {
				r.stop = "accepting";
				break;
			}
		}
		r.wall = MsSince(t0);
		if (r.stop.empty()) r.stop = "max-new";
		return r;
	};

	std::vector<Run> offs, ons;
	std::vector<double> ratio;
	for (int p = 0; p < warmup + pairs; ++p) {
		Run a, b;
		if (p % 2 == 0) {
			a = run_off();
			b = run_on();
		} else {
			b = run_on();
			a = run_off();
		}
		if (p < warmup) continue;
		offs.push_back(a);
		ons.push_back(b);
		ratio.push_back(a.wall / b.wall);
	}
	sslm_gpu_seq_release(ctx, seq);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	auto arm = [&](const char* label, const std::vector<Run>& runs) {
		std::vector<double> wall, gpu, per_tok, gpu_per_tok, submits;
		bool equal = true;
		for (const Run& r : runs) {
			wall.push_back(r.wall);
			gpu.push_back(r.gpu);
			per_tok.push_back(r.wall / static_cast<double>(r.stream.size()));
			gpu_per_tok.push_back(r.gpu / static_cast<double>(r.stream.size()));
			submits.push_back(r.submits);
			if (r.stream != runs.front().stream) equal = false;
		}
		const Stat w = Summarize(wall), g = Summarize(gpu), pt = Summarize(per_tok), gt = Summarize(gpu_per_tok);
		char d[32];
		std::snprintf(d, sizeof d, "%016llx", static_cast<unsigned long long>(Fnv(runs.front().stream)));
		std::printf("%s: tokens %zu stop %s submits %d | query wall %.3f [%.3f, %.3f] ms gpu %.3f ms | per token wall %.3f gpu %.3f ms | "
		            "digest %s all_equal %d\n",
		            label, runs.front().stream.size(), runs.front().stop.c_str(), runs.front().submits, w.median, w.lo, w.hi,
		            g.median, pt.median, gt.median, d, equal ? 1 : 0);
		return std::string("{\"label\":\"") + label + "\",\"tokens\":" + std::to_string(runs.front().stream.size()) +
		       ",\"stop\":\"" + runs.front().stop + "\",\"submits\":" + std::to_string(runs.front().submits) +
		       ",\"query_wall_ms\":" + JStat(w) + ",\"query_gpu_busy_ms\":" + JStat(g) + ",\"per_token_wall_ms\":" + JStat(pt) +
		       ",\"per_token_gpu_busy_ms\":" + JStat(gt) + ",\"digest\":\"" + d + "\",\"streams_all_equal\":" +
		       (equal ? "true" : "false") + ",\"stream\":" + JIds(runs.front().stream) + "}";
	};
	std::printf("prompt tokens %zu, schema potion_shop_order (index %zu), whole-token depth, logits head on the %s, %d pairs after "
	            "%d warm-up\n",
	            prompt.size(), schema_index, device_head ? "device" : "host", pairs, warmup);
	const std::string ja = arm("off", offs), jb = arm("on", ons);
	const Stat r = Summarize(ratio);
	std::printf("off / on wall: %.4f [%.4f, %.4f]\n", r.median, r.lo, r.hi);
	if (json_path) {
		std::FILE* o = std::fopen(json_path, "wb");
		if (!o) return 4;
		const std::string js = std::string("{\"mode\":\"gpu-jf\",\"logits_head\":\"") + (device_head ? "device" : "host") +
		                       "\",\"pairs\":" + std::to_string(pairs) + ",\"warmup\":" +
		                       std::to_string(warmup) + ",\"prompt_tokens\":" + std::to_string(prompt.size()) +
		                       ",\"prompt\":" + JIds(prompt) + ",\"arms\":[" + ja + "," + jb + "],\"ratio_off_over_on\":" +
		                       JStat(r) + "}";
		std::fwrite(js.data(), 1, js.size(), o);
		std::fclose(o);
	}
	return 0;
}
