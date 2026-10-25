// Schema fast-decode plan (rev 1.2.x) §9 (slice J1) and §10 (box steps B1-B4): the bench, headless. It
// drives the public C ABI (plus, in `finishrows` only, the library's public finish primitives) and
// prints plain `key value` lines; with --json PATH every mode also writes its readings as one JSON
// object, which tools/schema_fast_decode_box/run_box_step.ps1 wraps with provenance. Every figure it
// prints off the box is indicative; the plan's readings are the box's (§10).
//
//   schema_fast_decode_bench ARTIFACT extract SCHEMA [prompt options] [--pair A,B] [--budget 64]
//                            [--pairs 15] [--warmup 0] [--tasks 0] [--max-new 256]
//       B2 (B3 with --tasks 4): the extraction from a fixed prompt as n interleaved pairs of two arms
//       (or n runs of one) after `warmup` unmeasured pairs; measured pair k runs arm A first when k is
//       even and arm B first when it is odd. Arms: off (the plain greedy decode loop), on (the §5.6
//       host loop: drain the chain with sslm_seq_jump_forward until sslm_seq_forced_tokens reports 0,
//       then decode), on:B (on at chunk budget B; on:1 is the CB1 control) and off2 (the off run twice
//       back to back per sample; off2,off is the 2X control; off,off is AA). Only generation is timed
//       (the first call after the prompt prefill to the stop); the prefill is timed apart. Every
//       measured run is recorded (stream, digest, text, schema validity, field values), with per-arm
//       median, IQR and digest equality, and R_i = t(A)/t(B) per pair with the n = 15 sign-test
//       interval [R(4), R(12)]. --arms both|off|on is the older spelling of --pair off,on|off|on.
//   schema_fast_decode_bench - pool [--warmup 1] [--labels main,m1] [--first ABBA...] A0 B0 A1 B1 ...
//       B2/B3's `main` against M1, one process per arm per pair: pools one-arm, one-run extract --json
//       files as extract would report two arms, dropping the first `warmup` pairs; --first says
//       which arm ran first in each measured pair.
//   schema_fast_decode_bench ARTIFACT spans SCHEMA [prompt options] [--pairs 15] [--tasks 0]
//       B3's span table: for each span the on-arm admits, its size M and the milliseconds of the verb
//       call that admitted it, interleaved per pair with M one-token prefills of the same tokens (the
//       one-token threaded admission under --tasks), and D = verb - one-token per pair (§16 Q6).
//   schema_fast_decode_bench ARTIFACT pass [--example | --depth 100] [--sizes 1,2,3,4,5,6,7,8]
//                            [--pairs 15] [--warmup 1] [--tasks 0] [--no-one-row]
//       B1 (B3 with --tasks 4): the real batched pass (every site) for M tokens through the prefill
//       path the verb uses, at the example prompt's depth (--example) or `depth` synthetic ids;
//       rounds outer and the sizes rotated by one per round. Per size: median, IQR and sign interval
//       of T(M), of r(M) = T(M)/T(1) per round, of the same M tokens as M one-token prefills
//       (interleaved), and of D = T(M) - one-row per round; S = r(3) + 2 r(6) + r(5) + r(2) per round.
//   schema_fast_decode_bench ARTIFACT finish SCHEMA [prompt options] [--pairs 15] [--tasks 0]
//       The whole finish of a finish-only decode through the ABI, greedy and damped greedy (when the
//       artifact carries DGC1): unbound (all V rows), and at every choice the jump-forward run
//       reaches, the last being the dead end (0 rows).
//   schema_fast_decode_bench ARTIFACT finishrows [--rows 0,1,22,1000,V] [--pairs 15] [--warmup 1]
//                            [--tasks 0]
//       B1's finish at a chosen legal-row count: the library's own finish primitives on the artifact's
//       head (LogitsSiteLegalRows over n evenly spaced ids, then the re-mask and argmax; damped greedy
//       adds its V-wide masking loop and the scorer over an anti-LM state fed 16 tokens), rounds outer
//       and the row counts rotated per round; and `main`'s finish (LogitsSiteParallel over all V rows,
//       then the re-mask and argmax) for comparison. The final hidden codes are a fixed pattern: the
//       cost does not depend on their values.
//   schema_fast_decode_bench ARTIFACT quality SCHEMA --population FILE --kind potion|prompt_result
//                            [--from 0] [--count 20] [--max-new 64] [--special-from 151643]
//                            [--sealed PATH]
//       B4: each prompt of the population, jump-forward off and on (greedy), stopping when the schema
//       walk accepts, at the -2 dead end or at max-new tokens. potion: a JSONL file, one object per
//       line with the line (`customer_line`, `utterance` or `text`, wrapped in the example's
//       extraction prompt) or the wrapped `prompt` (both: they must agree), and the labels (`intent`,
//       `item`, `quantity`, `polite`, under `labels`/`expected` or at top level); each arm's output is
//       scored per field against the labels, byte for byte, and a run that is not schema-valid JSON
//       scores every field wrong. prompt_result: a JSON array of strings, wrapped in Qwen2.5's default
//       chat template (what the te368 run's ids were made with); each arm is graded by clean accept
//       (the walk accepted, and the value is answer-shaped with no special token, as the te368
//       census defines it). With --sealed, the on arm's records and every off-vs-on readout
//       (agreement, same fields with different bytes) go to PATH only, so the off arm can be read
//       first.
//   schema_fast_decode_bench - summarize X1 X2 ...
//       The median and IQR (linear interpolation, as every mode reports) of the given samples.
//
// Prompt options: --example (the example scene's extraction prompt, ChatML, with --utterance TEXT,
// default the scene's default utterance), --prompt TEXT (raw text, default the fixture's prompt).
//
// SFD_BENCH_DECODE_ONLY: built against a library without J1 (main), the bench keeps the modes that
// need decode only (extract with the off and off2 arms, pass, pool, summarize).
#include <algorithm>
#include <cctype>
#include <climits>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "superslm/layer_marshal.h"
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "../docs/parallel_for_reference.hpp"
#if !defined(SFD_BENCH_DECODE_ONLY)
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/schema_masks.h"
#if defined(SFD_BENCH_DRAFT_TABLE_BUILDER)
#include "schema_fast_decode/sfd_api.h"  // ForcedTableEntry: the library's forced flag per state (the §11.0 read seam)
#endif
#include "superslm/sslm_damped_greedy.h"
#endif

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

std::vector<int> ParseList(const char* s, int v_value = 0) {
	std::vector<int> out;
	for (const char* p = s; *p;) {
		out.push_back((*p == 'V' || *p == 'v') ? v_value : std::atoi(p));
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
bool Flag(int argc, char** argv, int first, const char* name) {
	for (int i = first; i < argc; ++i)
		if (std::strcmp(argv[i], name) == 0) return true;
	return false;
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

std::string Hex64(uint64_t v) {
	char b[17];
	std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
	return b;
}

// ---- JSON out ----------------------------------------------------------------------------------

std::string JStr(const std::string& s) {
	std::string o = "\"";
	for (unsigned char c : s) {
		switch (c) {
		case '"': o += "\\\""; break;
		case '\\': o += "\\\\"; break;
		case '\n': o += "\\n"; break;
		case '\r': o += "\\r"; break;
		case '\t': o += "\\t"; break;
		default:
			if (c < 0x20) {
				char b[8];
				std::snprintf(b, sizeof b, "\\u%04x", c);
				o += b;
			} else {
				o += static_cast<char>(c);
			}
		}
	}
	return o + "\"";
}
std::string JNum(double v) {
	char b[64];
	std::snprintf(b, sizeof b, "%.4f", v);
	return b;
}
std::string JInts(const std::vector<int>& v) {
	std::string o = "[";
	for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
	return o + "]";
}
std::string JIds(const std::vector<int32_t>& v) {
	std::string o = "[";
	for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
	return o + "]";
}
std::string JNums(const std::vector<double>& v) {
	std::string o = "[";
	for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + JNum(v[i]);
	return o + "]";
}
std::string JStats(const std::vector<double>& samples) {
	const Stats s = Summarize(samples);
	return "{\"n\":" + std::to_string(samples.size()) + ",\"median\":" + JNum(s.median) + ",\"q1\":" + JNum(s.q1) +
	       ",\"q3\":" + JNum(s.q3) + ",\"iqr\":" + JNum(s.q3 - s.q1) + ",\"samples\":" + JNums(samples) + "}";
}

// The --json file: the mode's object, written once at the end.
bool WriteJson(int argc, char** argv, const std::string& body) {
	const char* path = Arg(argc, argv, 2, "--json", nullptr);
	if (!path) return true;
	std::ofstream f(path, std::ios::binary);
	f << body << "\n";
	if (!f) {
		std::fprintf(stderr, "cannot write %s\n", path);
		return false;
	}
	return true;
}

// ---- JSON in (the populations) -----------------------------------------------------------------

struct JMember;  // a std::pair member would need JVal complete; std::vector does not
struct JVal {
	enum Kind { kNull, kBool, kNum, kStr, kArr, kObj } kind = kNull;
	bool b = false;
	double num = 0;
	std::string str;
	std::vector<JVal> arr;
	std::vector<JMember> obj;  // in document order
	const JVal* Get(const char* key) const;
};
struct JMember {
	std::string first;
	JVal second;
};
const JVal* JVal::Get(const char* key) const {
	if (kind != kObj) return nullptr;
	for (const JMember& kv : obj)
		if (kv.first == key) return &kv.second;
	return nullptr;
}

struct JParser {
	const char* p;
	const char* end;
	bool ok = true;
	void Ws() {
		while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
	}
	static void PutUtf8(std::string& o, uint32_t cp) {
		if (cp < 0x80) {
			o += static_cast<char>(cp);
		} else if (cp < 0x800) {
			o += static_cast<char>(0xC0 | (cp >> 6));
			o += static_cast<char>(0x80 | (cp & 0x3F));
		} else if (cp < 0x10000) {
			o += static_cast<char>(0xE0 | (cp >> 12));
			o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
			o += static_cast<char>(0x80 | (cp & 0x3F));
		} else {
			o += static_cast<char>(0xF0 | (cp >> 18));
			o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
			o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
			o += static_cast<char>(0x80 | (cp & 0x3F));
		}
	}
	bool Hex4(uint32_t* out) {
		if (end - p < 4) return false;
		uint32_t v = 0;
		for (int i = 0; i < 4; ++i, ++p) {
			const char c = *p;
			v <<= 4;
			if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
			else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
			else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
			else return false;
		}
		*out = v;
		return true;
	}
	bool Str(std::string* out) {
		if (p >= end || *p != '"') return false;
		++p;
		while (p < end && *p != '"') {
			if (*p == '\\') {
				if (++p >= end) return false;
				const char c = *p++;
				switch (c) {
				case '"': *out += '"'; break;
				case '\\': *out += '\\'; break;
				case '/': *out += '/'; break;
				case 'b': *out += '\b'; break;
				case 'f': *out += '\f'; break;
				case 'n': *out += '\n'; break;
				case 'r': *out += '\r'; break;
				case 't': *out += '\t'; break;
				case 'u': {
					uint32_t cp = 0;
					if (!Hex4(&cp)) return false;
					if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
						p += 2;
						uint32_t lo = 0;
						if (!Hex4(&lo)) return false;
						cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
					}
					PutUtf8(*out, cp);
					break;
				}
				default: return false;
				}
			} else {
				*out += *p++;
			}
		}
		if (p >= end) return false;
		++p;
		return true;
	}
	bool Value(JVal* v, int depth = 0) {
		if (depth > 64) return false;
		Ws();
		if (p >= end) return false;
		if (*p == '{') {
			v->kind = JVal::kObj;
			++p;
			Ws();
			if (p < end && *p == '}') return ++p, true;
			for (;;) {
				Ws();
				std::string k;
				if (!Str(&k)) return false;
				Ws();
				if (p >= end || *p != ':') return false;
				++p;
				JVal child;
				if (!Value(&child, depth + 1)) return false;
				v->obj.push_back(JMember{std::move(k), std::move(child)});
				Ws();
				if (p < end && *p == ',') { ++p; continue; }
				if (p < end && *p == '}') return ++p, true;
				return false;
			}
		}
		if (*p == '[') {
			v->kind = JVal::kArr;
			++p;
			Ws();
			if (p < end && *p == ']') return ++p, true;
			for (;;) {
				JVal child;
				if (!Value(&child, depth + 1)) return false;
				v->arr.push_back(std::move(child));
				Ws();
				if (p < end && *p == ',') { ++p; continue; }
				if (p < end && *p == ']') return ++p, true;
				return false;
			}
		}
		if (*p == '"') {
			v->kind = JVal::kStr;
			return Str(&v->str);
		}
		auto lit = [&](const char* w) {
			const size_t n = std::strlen(w);
			if (static_cast<size_t>(end - p) >= n && std::memcmp(p, w, n) == 0) return p += n, true;
			return false;
		};
		if (lit("true")) return v->kind = JVal::kBool, v->b = true, true;
		if (lit("false")) return v->kind = JVal::kBool, v->b = false, true;
		if (lit("null")) return v->kind = JVal::kNull, true;
		char* e = nullptr;
		const std::string tail(p, static_cast<size_t>(std::min<ptrdiff_t>(end - p, 64)));
		const double d = std::strtod(tail.c_str(), &e);
		if (e == tail.c_str()) return false;
		p += e - tail.c_str();
		v->kind = JVal::kNum;
		v->num = d;
		return true;
	}
};

// Parses one whole JSON text; trailing bytes other than whitespace fail it.
bool ParseJson(const std::string& text, JVal* out) {
	JParser jp{text.data(), text.data() + text.size()};
	if (!jp.Value(out)) return false;
	jp.Ws();
	return jp.p == jp.end;
}

bool ReadText(const char* path, std::string* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	std::ostringstream ss;
	ss << f.rdbuf();
	*out = ss.str();
	if (out->size() >= 3 && static_cast<unsigned char>((*out)[0]) == 0xEF &&
	    static_cast<unsigned char>((*out)[1]) == 0xBB && static_cast<unsigned char>((*out)[2]) == 0xBF) {
		out->erase(0, 3);
	}
	return true;
}

// ---- prompts -----------------------------------------------------------------------------------

const char* kFixturePrompt = "Customer: I would like to buy a health potion, please.\nOrder:";
// The example scene's extraction prompt (SuperSLMExampleDemo.cpp ExtractionPromptText) and its
// default utterance (SuperSLMExampleDemo.h DefaultUtterance).
const char* kExampleUtterance = "Hello! Could I get two health potions, please?";
std::string ExamplePrompt(const std::string& utterance) {
	return "<|im_start|>system\nYou work the counter of a potion shop. Record the customer's order as JSON.<|im_end|>\n"
	       "<|im_start|>user\n" + utterance + "<|im_end|>\n<|im_start|>assistant\n";
}
// Qwen2.5's default chat template for one user turn (apply_chat_template with no system message),
// which is what the te368 population's ids were made with.
[[maybe_unused]] std::string QwenDefaultPrompt(const std::string& user) {
	return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
	       "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
}
std::string PromptText(int argc, char** argv) {
	if (Flag(argc, argv, 3, "--example")) return ExamplePrompt(Arg(argc, argv, 3, "--utterance", kExampleUtterance));
	return Arg(argc, argv, 3, "--prompt", kFixturePrompt);
}

// ---- engine ------------------------------------------------------------------------------------

struct Engine {
	uint8_t* bytes = nullptr;
	size_t size = 0;
	sslm_model model = nullptr;
	int32_t layers = 0;
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
	std::vector<int32_t> Tokenize(const std::string& text) const {
		int32_t n = 0;
		sslm_tokenize(model, text.c_str(), nullptr, &n);
		std::vector<int32_t> t(static_cast<size_t>(n > 0 ? n : 1));
		if (sslm_tokenize(model, text.c_str(), t.data(), &n) != SSLM_OK) {
			std::fprintf(stderr, "sslm_tokenize failed\n");
			std::exit(3);
		}
		t.resize(static_cast<size_t>(n));
		return t;
	}
	std::string Detokenize(const std::vector<int32_t>& ids) const {
		if (ids.empty()) return std::string();
		sslm_detok_state st = {};
		std::vector<char> buf(ids.size() * 64 + 64);
		int32_t n = static_cast<int32_t>(buf.size());
		if (sslm_detokenize_stream(model, &st, ids.data(), static_cast<int32_t>(ids.size()), buf.data(), &n) != SSLM_OK) {
			return std::string();
		}
		return std::string(buf.data(), static_cast<size_t>(n));
	}
	bool Accepting(sslm_seq s) const {
		sslm_stats_out stats{};
		return sslm_stats(model, s, &stats) == SSLM_OK && stats.schema_accepting;
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
	const char* stop = "budget";     // "deadend", "accepting", "budget" or "status"
	bool accepting_end = false;      // the schema walk accepts where the run stopped
	double prefill_ms = 0;           // the prompt prefill (sslm_prefill), timed apart
	double ms = 0;                   // generation only: the first call after the prefill to the stop
};

// Jump-forward off: the plain greedy loop until -2 or `max_new` tokens (or, with stop_on_accept, the
// first decode after which the schema walk accepts, as the te368 harness stops).
Arm RunOff(Engine& e, const char* schema, const std::vector<int32_t>& prompt, int max_new, bool stop_on_accept = false) {
	Arm a;
	const auto p0 = Clock::now();
	sslm_seq s = e.Start(schema, prompt);
	a.prefill_ms = MsSince(p0);
	const sslm_decode_params g = e.Greedy();
	const auto t0 = Clock::now();
	for (int i = 0; i <= max_new; ++i) {
		if (static_cast<int>(a.stream.size()) >= max_new) break;
		int32_t t = -100;
		if (sslm_decode_step_v2(e.model, &s, 1, &g, e.ws, &t) != SSLM_OK) {
			a.stop = "status";
			break;
		}
		++a.finishes;
		if (i > 0) ++a.layer_passes;  // the first decode finishes the prompt's last token only
		if (t == -2) {
			a.dead_end = true;
			a.stop = "deadend";
			break;
		}
		if (t < 0) {
			a.stop = "status";
			break;
		}
		a.stream.push_back(t);
		if (stop_on_accept && e.Accepting(s)) {
			a.stop = "accepting";
			break;
		}
	}
	a.ms = MsSince(t0);
	a.accepting_end = schema && e.Accepting(s);
	sslm_seq_release(s);
	return a;
}

// Exact mode's draft (the arms `on:exact`, `on:exact:learned`, `on:exact:wrong`).
enum class DraftKind { kCanonical, kLearned, kWrong };

#if !defined(SFD_BENCH_DECODE_ONLY)
// ---- Exact mode's draft sources (plan rev 1.3.4 §5.10.5, the B7 steps) -------------------------------------
// The host tracks the schema walk itself (the public ABI has no walk-state query): from state 0 at the first
// generated token, one SCM1 transition per token the run appends. The drafts:
//   learned  the frozen D2 table (drafttable, below): from the state after the pending token, the table's most
//            frequent token at each state while the running product of those tokens' shares stays >= 0.5;
//            stop at a state the table never saw or the library marks unforced, and clip to the span budget.
//            Nothing stored for the state: an empty draft (no verb call), never the canonical chain.
//   wrong    a legal token other than off's at each forced state with two or more legal tokens, stopping at a
//            state with one legal token (or an unforced one). Off's stream comes from one unmeasured off run in
//            the same process; after the first substitution the draft follows its own path, any legal token.
struct SchemaWalker {
	superslm::SslmModelView view;
	superslm::SchemaMasksTable table;
	const superslm::SchemaEntry* entry = nullptr;
	size_t index = 0;
	bool Open(const Engine& e, const char* schema) {
		std::string err;
		if (superslm::SslmModel::Load(e.bytes, e.size, view, &err) != superslm::SslmModelStatus::Ok) return false;
		const superslm::SslmSectionView* scm = view.Section(superslm::SslmSectionType::SchemaMasks);
		if (!scm || !superslm::SchemaMasksTable::Parse(scm->data, scm->byte_size, view.config.vocab_size, table, &err)) return false;
		entry = table.ByName(schema, &index);
		return entry != nullptr;
	}
	bool Step(uint32_t state, int32_t tok, uint32_t* next) const {
		return tok >= 0 && table.Transition(*entry, state, static_cast<uint32_t>(tok), next);
	}
	// The state's legal tokens, in CSR order (ascending ids), with their next states.
	void Legal(uint32_t state, std::vector<std::pair<int32_t, uint32_t>>* out) const {
		using superslm::schema_masks_detail::ReadLE32;
		out->clear();
		const uint32_t b = ReadLE32(entry->state_offsets_le + static_cast<size_t>(state) * 4);
		const uint32_t en = ReadLE32(entry->state_offsets_le + static_cast<size_t>(state + 1) * 4);
		for (uint32_t i = b; i < en; ++i)
			out->push_back({static_cast<int32_t>(ReadLE32(entry->transitions_le + static_cast<size_t>(i) * 8)),
			                ReadLE32(entry->transitions_le + static_cast<size_t>(i) * 8 + 4)});
	}
};
struct DraftTable {
	std::vector<uint8_t> forced;                                     // per state: the library's forced flag
	std::vector<std::vector<std::pair<int32_t, int64_t>>> counts;  // per state: (token, count)
	bool Load(const char* path) {
		std::ifstream in(path);
		std::string line;
		if (!in || !std::getline(in, line) || line.rfind("SFD-DRAFT-TABLE/1 ", 0) != 0) return false;
		while (std::getline(in, line)) {
			std::istringstream ls(line);
			std::string tag;
			ls >> tag;
			if (tag == "states") {
				size_t n = 0;
				ls >> n;
				forced.assign(n, 0);
				counts.assign(n, {});
			} else if (tag == "forced") {
				size_t st = 0;
				while (ls >> st)
					if (st < forced.size()) forced[st] = 1;
			} else if (tag == "S") {
				size_t st = 0;
				ls >> st;
				std::string tc;
				while (ls >> tc) {
					const size_t c = tc.find(':');
					if (st < counts.size() && c != std::string::npos)
						counts[st].push_back({std::atoi(tc.substr(0, c).c_str()), std::atoll(tc.substr(c + 1).c_str())});
				}
			}
		}
		return !forced.empty();
	}
};
struct DraftCtx {
	const SchemaWalker* walker = nullptr;
	const DraftTable* table = nullptr;
	const std::vector<int32_t>* off_stream = nullptr;
};
DraftCtx g_draft;

// The draft from walk state `w`, the run's next stream position being `pos`, at most `cap` tokens.
std::vector<int32_t> MakeDraft(DraftKind kind, uint32_t w, size_t pos, int32_t cap) {
	std::vector<int32_t> d;
	const SchemaWalker& wk = *g_draft.walker;
	if (kind == DraftKind::kLearned) {
		const DraftTable& t = *g_draft.table;
		double product = 1.0;
		uint32_t s = w;
		while (static_cast<int32_t>(d.size()) < cap && s < t.counts.size() && t.forced[s] && !t.counts[s].empty()) {
			int64_t total = 0, best = -1;
			int32_t tok = -1;
			for (const auto& tc : t.counts[s]) {
				total += tc.second;
				if (tc.second > best) best = tc.second, tok = tc.first;
			}
			product *= static_cast<double>(best) / static_cast<double>(total);
			if (product < 0.5) break;
			uint32_t nx = 0;
			if (!wk.Step(s, tok, &nx)) break;
			d.push_back(tok);
			s = nx;
		}
	} else if (kind == DraftKind::kWrong) {
		const DraftTable& t = *g_draft.table;
		const std::vector<int32_t>& off = *g_draft.off_stream;
		std::vector<std::pair<int32_t, uint32_t>> legal;
		uint32_t s = w;
		bool diverged = false;
		while (static_cast<int32_t>(d.size()) < cap && s < t.forced.size() && t.forced[s]) {
			wk.Legal(s, &legal);
			if (legal.size() < 2) break;
			const size_t at = pos + d.size();
			const int32_t avoid = !diverged && at < off.size() ? off[at] : -1;
			size_t pick = 0;
			while (pick < legal.size() && legal[pick].first == avoid) ++pick;
			if (pick == legal.size()) break;
			d.push_back(legal[pick].first);
			diverged = diverged || legal[pick].first != avoid;
			s = legal[pick].second;
		}
	}
	return d;
}

// Jump-forward on: the §5.6 host loop.
// `exact`: the verb in exact mode (sslm_jump_options, SSLM_JUMP_EXACT). It needs no host loop of its own
// (docs/api.md, exact mode): this loop calls it after each decode step, while a token is pending, and it keeps
// only the drafted tokens the model would itself produce. `draft` picks the draft: the canonical chain (a NULL
// draft), the learned table, or the always-wrong draft (above).
Arm RunOn(Engine& e, const char* schema, const std::vector<int32_t>& prompt, int32_t budget, int max_new,
          bool stop_on_accept = false, bool exact = false, DraftKind draft = DraftKind::kCanonical) {
	Arm a;
	sslm_jump_options exact_opts{};
	exact_opts.struct_size = sizeof(sslm_jump_options);
	exact_opts.mode = SSLM_JUMP_EXACT;
	exact_opts.draft = nullptr;
	exact_opts.draft_count = 0;
	const bool host_draft = exact && draft != DraftKind::kCanonical;
	uint32_t walk = 0;  // the schema walk after the stream so far (tracked only for a host-made draft)
	bool walk_ok = true;
	auto advance = [&](int32_t tok) {
		uint32_t nx = 0;
		if (host_draft && walk_ok) walk_ok = g_draft.walker->Step(walk, tok, &nx), walk = nx;
	};
	std::vector<int32_t> draft_buf;
	const auto p0 = Clock::now();
	sslm_seq s = e.Start(schema, prompt);
	a.prefill_ms = MsSince(p0);
	const sslm_decode_params g = e.Greedy();
	std::vector<int32_t> buf(static_cast<size_t>(budget));
	bool pending = false;  // the last decode's token is not yet admitted
	const auto t0 = Clock::now();
	while (static_cast<int>(a.stream.size()) < max_new) {
		for (;;) {
			int32_t count = 0, accepting = 0;
			if (sslm_seq_forced_tokens(e.model, s, nullptr, 0, &count, &accepting) != SSLM_OK || count == 0) break;
			int32_t forced = 0;
			const int32_t room = std::min<int32_t>(budget, static_cast<int32_t>(max_new - a.stream.size()) + (pending ? 1 : 0));
			if (room < 1 || static_cast<int>(a.stream.size()) >= max_new) break;  // the cap is reached
			if (host_draft) {
				// The draft starts at the walk state after the pending token; with no pending token, or an empty
				// draft, the verb would do nothing, so the decode step takes it from here.
				if (!pending || !walk_ok) break;
				draft_buf = MakeDraft(draft, walk, a.stream.size(), room - 1);
				if (draft_buf.empty()) break;
				exact_opts.draft = draft_buf.data();
				exact_opts.draft_count = static_cast<int32_t>(draft_buf.size());
			}
			const auto v0 = Clock::now();
			const sslm_status st = sslm_seq_jump_forward(e.model, s, room, nullptr, exact ? &exact_opts : nullptr, e.ws, buf.data(),
			                                             budget, &forced);
			const double v_ms = MsSince(v0);
			if (st != SSLM_OK) {
				std::fprintf(stderr, "jump_forward status %d\n", static_cast<int>(st));
				a.stop = "status";
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
			for (int32_t i = 0; i < forced; ++i) advance(buf[static_cast<size_t>(i)]);
			// Drain until the walk reports no forced token: a call that admitted only the pending token (a
			// chunk budget of 1) has not drained the chain; one that admitted nothing cannot progress.
			if (m == 0) break;
		}
		if (stop_on_accept && !a.stream.empty() && e.Accepting(s)) {
			a.stop = "accepting";
			break;
		}
		if (static_cast<int>(a.stream.size()) >= max_new) break;
		{
			int32_t t = -100;
			if (sslm_decode_step_v2(e.model, &s, 1, &g, e.ws, &t) != SSLM_OK) {
				a.stop = "status";
				break;
			}
			++a.finishes;
			if (pending) ++a.layer_passes;  // a choice after a choice embeds the pending token here
			if (t == -2) {
				a.dead_end = true;
				a.stop = "deadend";
				break;
			}
			if (t < 0) {
				a.stop = "status";
				break;
			}
			a.stream.push_back(t);
			advance(t);
			pending = true;
			if (stop_on_accept && e.Accepting(s)) {
				a.stop = "accepting";
				break;
			}
		}
	}
done:
	a.ms = MsSince(t0);
	a.accepting_end = schema && e.Accepting(s);
	sslm_seq_release(s);
	return a;
}
#endif

// ---- pairs ------------------------------------------------------------------------------------

std::vector<double> PairDiff(const std::vector<double>& a, const std::vector<double>& b) {
	std::vector<double> d;
	for (size_t i = 0; i < a.size() && i < b.size(); ++i) d.push_back(a[i] - b[i]);
	return d;
}

// The acceptance rules' decision interval for n = 15 pairs: the 4th and 12th order statistics, the
// sign-test interval for the median (coverage 96.48%). Reported beside the median and IQR; for any
// other n it is null.
std::string JPaired(const std::vector<double>& v) {
	std::string o = JStats(v);
	o.pop_back();
	if (v.size() == 15) {
		std::vector<double> s = v;
		std::sort(s.begin(), s.end());
		o += ",\"sign_lo\":" + JNum(s[3]) + ",\"sign_hi\":" + JNum(s[11]);
	} else {
		o += ",\"sign_lo\":null,\"sign_hi\":null";
	}
	return o + "}";
}

// potion_shop_order as the example's validator table has it: the four keys in this order, the enums,
// and `polite` a JSON boolean.
bool PotionValid(const JVal& v) {
	static const char* kIntent[] = {"buy", "sell", "ask_price", "haggle", "leave"};
	static const char* kItem[] = {"health_potion", "mana_potion", "rope", "torch", "lantern", "none"};
	static const char* kQuantity[] = {"one", "two", "three", "several", "none"};
	auto in = [](const JVal& x, const char* const* set, size_t n) {
		if (x.kind != JVal::kStr) return false;
		for (size_t i = 0; i < n; ++i)
			if (x.str == set[i]) return true;
		return false;
	};
	return v.kind == JVal::kObj && v.obj.size() == 4 && v.obj[0].first == "intent" && in(v.obj[0].second, kIntent, 5) &&
	       v.obj[1].first == "item" && in(v.obj[1].second, kItem, 6) && v.obj[2].first == "quantity" &&
	       in(v.obj[2].second, kQuantity, 5) && v.obj[3].first == "polite" && v.obj[3].second.kind == JVal::kBool;
}

std::string JDump(const JVal& v) {
	switch (v.kind) {
	case JVal::kNull: return "null";
	case JVal::kBool: return v.b ? "true" : "false";
	case JVal::kNum: {
		if (v.num == static_cast<double>(static_cast<long long>(v.num)) && v.num < 1e15 && v.num > -1e15)
			return std::to_string(static_cast<long long>(v.num));
		return JNum(v.num);
	}
	case JVal::kStr: return JStr(v.str);
	case JVal::kArr: {
		std::string o = "[";
		for (size_t i = 0; i < v.arr.size(); ++i) o += (i ? "," : "") + JDump(v.arr[i]);
		return o + "]";
	}
	case JVal::kObj: {
		std::string o = "{";
		for (size_t i = 0; i < v.obj.size(); ++i) o += (i ? "," : "") + JStr(v.obj[i].first) + ":" + JDump(v.obj[i].second);
		return o + "}";
	}
	}
	return "null";
}

// An arm of a timed comparison: `off` (jump-forward off), `off2` (the off-arm extraction twice back to
// back per sample: the rules' 2X control), `on` (jump-forward on at the default chunk budget) or `on:B`
// (on at chunk budget B: `on:1` is the CB1 control).
struct ArmSpec {
	std::string name;
	bool on = false;
	bool exact = false;  // `on:exact`: jump-forward on in exact mode
	DraftKind draft = DraftKind::kCanonical;  // `on:exact:learned`, `on:exact:wrong` (both need --draft-table)
	int32_t budget = 64;
	int repeat = 1;
};
bool ParseArmSpec(const std::string& t, int32_t default_budget, ArmSpec* out) {
	out->name = t;
	out->budget = default_budget;
	if (t == "off") return true;
	if (t == "off2") return out->repeat = 2, true;
	if (t == "on") return out->on = true, true;
	if (t == "on:exact") return out->on = true, out->exact = true, true;
	if (t == "on:exact:learned") return out->on = true, out->exact = true, out->draft = DraftKind::kLearned, true;
	if (t == "on:exact:wrong") return out->on = true, out->exact = true, out->draft = DraftKind::kWrong, true;
	if (t.compare(0, 3, "on:") == 0 && t.size() > 3) {
		out->on = true;
		out->budget = std::atoi(t.c_str() + 3);
		return out->budget >= 1;
	}
	return false;
}

// One sample of an arm: the generation milliseconds (summed over off2's two runs), the prefill
// milliseconds beside it, and the record of the (last) run.
struct ArmRecord {
	std::string label;
	std::vector<std::string> runs;  // one JSON object per measured sample
	std::vector<double> gen_ms, prefill_ms;
	std::vector<std::string> digests;
};

std::string JRunRecord(const Engine& e, const Arm& a, const char* schema, double gen_ms, double prefill_ms, bool repeat_equal) {
	const std::string text = e.Detokenize(a.stream);
	JVal obj;
	const bool parsed = ParseJson(text, &obj) && obj.kind == JVal::kObj;
	const bool potion = schema && std::strcmp(schema, "potion_shop_order") == 0;
	std::string o = "{\"gen_ms\":" + JNum(gen_ms) + ",\"prefill_ms\":" + JNum(prefill_ms) + ",\"tokens\":" +
	                std::to_string(a.stream.size()) + ",\"layer_passes\":" + std::to_string(a.layer_passes) +
	                ",\"finishes\":" + std::to_string(a.finishes) + ",\"dead_end\":" + (a.dead_end ? "true" : "false") +
	                ",\"stop\":" + JStr(a.stop) + ",\"accepting_end\":" + (a.accepting_end ? "true" : "false") +
	                ",\"digest\":" + JStr(Hex64(Digest(a.stream))) + ",\"repeat_equal\":" + (repeat_equal ? "true" : "false") +
	                ",\"spans\":" + JInts(a.spans) + ",\"stream\":" + JIds(a.stream) + ",\"text\":" + JStr(text) +
	                ",\"json\":" + (parsed ? "true" : "false");
	if (potion) o += std::string(",\"schema_valid\":") + (parsed && PotionValid(obj) ? "true" : "false");
	o += ",\"fields\":" + (parsed ? JDump(obj) : std::string("null")) + "}";
	return o;
}

// The arms, which arm ran first in each pair, and (two arms) R_i = t(arm 0) / t(arm 1) per pair: the
// baseline is listed first, so R > 1 means the second arm is faster (for 2X, list off2 first).
std::string JComparison(const std::vector<ArmRecord>& arms, const std::vector<int>& first) {
	std::string j = ",\"arms\":[";
	for (size_t k = 0; k < arms.size(); ++k) {
		const ArmRecord& r = arms[k];
		bool eq = true;
		for (const std::string& d : r.digests) eq = eq && d == r.digests.front();
		j += std::string(k ? "," : "") + "{\"label\":" + JStr(r.label) + ",\"gen_ms\":" + JPaired(r.gen_ms) +
		     ",\"prefill_ms\":" + JStats(r.prefill_ms) + ",\"digests_equal\":" + (eq ? "true" : "false") +
		     ",\"digest\":" + JStr(r.digests.empty() ? "" : r.digests.front()) + ",\"runs\":[";
		for (size_t i = 0; i < r.runs.size(); ++i) j += (i ? "," : "") + r.runs[i];
		j += "]}";
		const Stats st = Summarize(r.gen_ms);
		std::printf("arm %s n %zu gen_ms_median %.3f q1 %.3f q3 %.3f prefill_ms_median %.3f digest %s digests_equal %d\n",
		            r.label.c_str(), r.gen_ms.size(), st.median, st.q1, st.q3, Summarize(r.prefill_ms).median,
		            r.digests.empty() ? "-" : r.digests.front().c_str(), eq ? 1 : 0);
	}
	j += "],\"first\":[";
	for (size_t i = 0; i < first.size(); ++i) j += std::string(i ? "," : "") + JStr(arms[static_cast<size_t>(first[i])].label);
	j += "]";
	if (arms.size() == 2) {
		std::vector<double> ratio;
		for (size_t i = 0; i < arms[0].gen_ms.size() && i < arms[1].gen_ms.size(); ++i)
			ratio.push_back(arms[1].gen_ms[i] > 0 ? arms[0].gen_ms[i] / arms[1].gen_ms[i] : 0.0);
		const Stats st = Summarize(ratio);
		std::vector<double> sorted = ratio;
		std::sort(sorted.begin(), sorted.end());
		std::printf("ratio %s/%s median %.4f q1 %.4f q3 %.4f", arms[0].label.c_str(), arms[1].label.c_str(), st.median, st.q1,
		            st.q3);
		if (sorted.size() == 15) std::printf(" sign_lo %.4f sign_hi %.4f", sorted[3], sorted[11]);
		std::printf("\n");
		j += ",\"ratio\":" + JPaired(ratio);
	}
	return j;
}

// B2/B3: n interleaved pairs of two arms (or n runs of one) after `warmup` unmeasured pairs. The warm-up
// pair runs arm 0 first; measured pair k (from 0) runs arm 0 first when k is even and arm 1 first when
// it is odd (the rules' ABBA). Only generation is timed; the prefill is timed apart.
int Extract(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const std::string text = PromptText(argc, argv);
	const std::vector<int32_t> prompt = e.Tokenize(text);
	const int32_t budget = std::atoi(Arg(argc, argv, 4, "--budget", "64"));
	const int pairs = std::atoi(Arg(argc, argv, 4, "--pairs", "15"));
	const int warmup = std::atoi(Arg(argc, argv, 4, "--warmup", "0"));
	const int max_new = std::atoi(Arg(argc, argv, 4, "--max-new", "256"));
	std::string pair = Arg(argc, argv, 4, "--pair", "");
	if (pair.empty()) {
		const std::string arms = Arg(argc, argv, 4, "--arms", "both");
		pair = arms == "both" ? "off,on" : arms;
	}
	std::vector<ArmSpec> specs;
	for (size_t at = 0; at <= pair.size();) {
		const size_t comma = pair.find(',', at);
		const std::string t = pair.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
		ArmSpec sp;
		if (!ParseArmSpec(t, budget, &sp)) {
			std::fprintf(stderr, "bad arm '%s' (off, off2, on, on:B)\n", t.c_str());
			return 2;
		}
#if defined(SFD_BENCH_DECODE_ONLY)
		if (sp.on) {
			std::fprintf(stderr, "this bench is built without J1: jump-forward arms are not available\n");
			return 2;
		}
#endif
		specs.push_back(sp);
		if (comma == std::string::npos) break;
		at = comma + 1;
	}
	if (specs.empty() || specs.size() > 4 || pairs < 1 || warmup < 0) {
		std::fprintf(stderr, "--pair takes one to four arms (more than two run in the order listed, with no ratio); --pairs >= 1\n");
		return 2;
	}
#if !defined(SFD_BENCH_DECODE_ONLY)
	// A host-made draft (learned or wrong) needs the walk, the frozen table and, for wrong, off's stream.
	static SchemaWalker walker;
	static DraftTable table;
	static std::vector<int32_t> off_stream;
	bool need_table = false, need_off = false;
	for (const ArmSpec& sp : specs) {
		need_table = need_table || sp.draft != DraftKind::kCanonical;
		need_off = need_off || sp.draft == DraftKind::kWrong;
	}
	if (need_table) {
		const char* path = Arg(argc, argv, 4, "--draft-table", nullptr);
		if (!path || !table.Load(path) || !walker.Open(e, schema) || table.forced.size() != walker.entry->state_count) {
			std::fprintf(stderr, "on:exact:learned and on:exact:wrong need --draft-table PATH, a table for this schema\n");
			return 2;
		}
		g_draft.walker = &walker;
		g_draft.table = &table;
		std::printf("draft_table %s states %zu\n", path, table.forced.size());
	}
	if (need_off) {
		off_stream = RunOff(e, schema, prompt, max_new).stream;  // unmeasured: off's tokens for the wrong draft
		g_draft.off_stream = &off_stream;
	}
#endif
	std::vector<ArmRecord> rec(specs.size());
	for (size_t k = 0; k < specs.size(); ++k) rec[k].label = specs[k].name;
	if (specs.size() == 2 && specs[0].name == specs[1].name) rec[0].label += "#A", rec[1].label += "#B";
	std::vector<int> first;
	bool ok = true;
	for (int i = 0; i < warmup + pairs; ++i) {
		const int k = i - warmup;
		const bool a_first = k < 0 || k % 2 == 0;
		for (size_t slot = 0; slot < specs.size(); ++slot) {
			const size_t w = specs.size() == 2 && !a_first ? 1 - slot : slot;
			const ArmSpec& sp = specs[w];
			Arm a, prev;
			double gen = 0, pre = 0;
			bool repeat_equal = true;
			for (int r = 0; r < sp.repeat; ++r) {
#if !defined(SFD_BENCH_DECODE_ONLY)
				a = sp.on ? RunOn(e, schema, prompt, sp.budget, max_new, /*stop_on_accept=*/false, sp.exact, sp.draft)
				          : RunOff(e, schema, prompt, max_new);
#else
				a = RunOff(e, schema, prompt, max_new);
#endif
				gen += a.ms;
				pre += a.prefill_ms;
				if (r > 0) repeat_equal = repeat_equal && a.stream == prev.stream;
				prev = a;
			}
			ok = ok && std::strcmp(a.stop, "status") != 0 &&
			     (a.dead_end || static_cast<int>(a.stream.size()) >= max_new);
			if (k < 0) continue;
			rec[w].runs.push_back(JRunRecord(e, a, schema, gen, pre, repeat_equal));
			rec[w].gen_ms.push_back(gen);
			rec[w].prefill_ms.push_back(pre);
			rec[w].digests.push_back(Hex64(Digest(a.stream)));
		}
		if (k >= 0) first.push_back(a_first ? 0 : 1);
	}
	std::printf("schema %s prompt_tokens %zu budget %d pairs %d warmup %d pair %s\n", schema, prompt.size(), budget, pairs,
	            warmup, pair.c_str());
	std::string j = "{\"mode\":\"extract\",\"schema\":" + JStr(schema) + ",\"prompt\":" + JStr(text) +
	                ",\"prompt_tokens\":" + std::to_string(prompt.size()) + ",\"budget\":" + std::to_string(budget) +
	                ",\"pairs\":" + std::to_string(pairs) + ",\"warmup\":" + std::to_string(warmup) +
	                ",\"max_new\":" + std::to_string(max_new) + ",\"tasks\":" + Arg(argc, argv, 3, "--tasks", "0") +
	                ",\"pair\":" + JStr(pair) + JComparison(rec, first) + ",\"ok\":" + (ok ? "true" : "false") + "}";
	if (!WriteJson(argc, argv, j)) return 2;
	return ok ? 0 : 1;
}

// B2/B3's `main` against M1: two benches on two libraries cannot share a process, so each pair is two
// processes (the driver alternates which runs first, as Extract does) and this pools them. FILES are
// extract --json outputs of one arm and one measured run each, listed per pair as A0 B0 A1 B1 ...; the
// first `warmup` pairs are dropped unread.
int Pool(int argc, char** argv) {
	const int warmup = std::atoi(Arg(argc, argv, 3, "--warmup", "1"));
	const std::string labels = Arg(argc, argv, 3, "--labels", "main,m1");
	const std::string order = Arg(argc, argv, 3, "--first", "");  // per pair, A or B (as run)
	std::vector<std::string> files;
	for (int i = 3; i < argc; ++i) {
		if (!std::strcmp(argv[i], "--json") || !std::strcmp(argv[i], "--warmup") || !std::strcmp(argv[i], "--labels") ||
		    !std::strcmp(argv[i], "--first")) {
			++i;
			continue;
		}
		files.push_back(argv[i]);
	}
	if (files.empty() || files.size() % 2 != 0) {
		std::fprintf(stderr, "pool takes an even number of files (A0 B0 A1 B1 ...)\n");
		return 2;
	}
	std::vector<ArmRecord> rec(2);
	const size_t comma = labels.find(',');
	rec[0].label = labels.substr(0, comma);
	rec[1].label = comma == std::string::npos ? "b" : labels.substr(comma + 1);
	std::vector<int> first;
	std::string schema, prompt_text;
	for (size_t f = 0; f < files.size(); ++f) {
		const size_t pair_index = f / 2;
		std::string text;
		JVal v;
		if (!ReadText(files[f].c_str(), &text) || !ParseJson(text, &v) || v.kind != JVal::kObj) {
			std::fprintf(stderr, "cannot read %s as JSON\n", files[f].c_str());
			return 2;
		}
		const JVal* arms = v.Get("arms");
		const JVal* ok = v.Get("ok");
		if (!arms || arms->kind != JVal::kArr || arms->arr.size() != 1 || !ok || !ok->b) {
			std::fprintf(stderr, "%s is not a one-arm extract that completed\n", files[f].c_str());
			return 1;
		}
		const JVal* runs = arms->arr[0].Get("runs");
		if (!runs || runs->kind != JVal::kArr || runs->arr.size() != 1) {
			std::fprintf(stderr, "%s does not hold exactly one measured run\n", files[f].c_str());
			return 2;
		}
		const JVal* sc = v.Get("schema");
		const JVal* pr = v.Get("prompt");
		if (f == 0) {
			schema = sc ? sc->str : "";
			prompt_text = pr ? pr->str : "";
		} else if (!sc || sc->str != schema || !pr || pr->str != prompt_text) {
			std::fprintf(stderr, "%s has a different schema or prompt\n", files[f].c_str());
			return 2;
		}
		if (pair_index < static_cast<size_t>(warmup)) continue;
		const JVal& run = runs->arr[0];
		const JVal* g = run.Get("gen_ms");
		const JVal* p = run.Get("prefill_ms");
		const JVal* d = run.Get("digest");
		ArmRecord& r = rec[f % 2];
		r.runs.push_back(JDump(run));
		r.gen_ms.push_back(g ? g->num : 0);
		r.prefill_ms.push_back(p ? p->num : 0);
		r.digests.push_back(d ? d->str : "");
		if (f % 2 == 0) {
			const size_t m = pair_index - static_cast<size_t>(warmup);
			first.push_back(m < order.size() && order[m] == 'B' ? 1 : 0);
		}
	}
	bool all_equal = true;
	for (const ArmRecord& r : rec)
		for (const std::string& d : r.digests) all_equal = all_equal && d == rec[0].digests.front();
	std::printf("pool pairs %zu warmup %d digests_all_equal %d\n", rec[0].gen_ms.size(), warmup, all_equal ? 1 : 0);
	const std::string j = "{\"mode\":\"pool\",\"schema\":" + JStr(schema) + ",\"prompt\":" + JStr(prompt_text) +
	                      ",\"pairs\":" + std::to_string(rec[0].gen_ms.size()) + ",\"warmup\":" + std::to_string(warmup) +
	                      JComparison(rec, first) + ",\"digests_all_equal\":" + (all_equal ? "true" : "false") + "}";
	return WriteJson(argc, argv, j) ? 0 : 2;
}

#if !defined(SFD_BENCH_DECODE_ONLY)
int Spans(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const std::vector<int32_t> prompt = e.Tokenize(PromptText(argc, argv));
	const int pairs = std::atoi(Arg(argc, argv, 4, "--pairs", "15"));
	const int32_t budget = std::atoi(Arg(argc, argv, 4, "--budget", "64"));
	std::vector<std::vector<double>> per_span, per_span_one;
	Arm on;
	// Interleaved per pair: the on run (each admission's verb call timed), then the same admissions as
	// M one-token prefills of the same tokens, in order on one sequence (threaded under --tasks: the
	// one-token threaded admission the rules name as the comparator; prompt spans, since the walk is
	// not the point here, the M one-row passes are).
	for (int i = 0; i < pairs; ++i) {
		on = RunOn(e, schema, prompt, budget, 256);
		if (per_span.empty()) per_span.resize(on.span_ms.size()), per_span_one.resize(on.span_ms.size());
		for (size_t k = 0; k < on.span_ms.size() && k < per_span.size(); ++k) per_span[k].push_back(on.span_ms[k]);
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
	std::string j = "{\"mode\":\"spans\",\"schema\":" + JStr(schema) + ",\"pairs\":" + std::to_string(pairs) + ",\"spans\":[";
	for (size_t k = 0; k < per_span.size(); ++k) {
		const Stats a = Summarize(per_span[k]), b = Summarize(per_span_one[k]);
		std::printf("span %zu M %d verb_ms_median %.3f one_token_ms_median %.3f\n", k, on.spans[k], a.median, b.median);
		j += std::string(k ? "," : "") + "{\"index\":" + std::to_string(k) + ",\"M\":" + std::to_string(on.spans[k]) +
		     ",\"verb_ms\":" + JStats(per_span[k]) + ",\"one_token_ms\":" + JStats(per_span_one[k]) +
		     ",\"d_ms\":" + JPaired(PairDiff(per_span[k], per_span_one[k])) + "}";
	}
	j += "]}";
	return WriteJson(argc, argv, j) ? 0 : 2;
}
#endif

// B1 (B3 with --tasks 4): rounds outer, the sizes inner and rotated by one each round (round r starts
// at size index r mod |sizes|), so every M sees every position in the round. Per round and size: the
// batched pass, then (unless --no-one-row) the same M tokens as M one-token prefills, interleaved.
// --example prefixes the example's real extraction prompt (the depth the extraction reaches) instead
// of `depth` synthetic ids. Per round: r(M) = T(M)/T(1), and S = r(3) + 2 r(6) + r(5) + r(2) when
// those sizes are present (the rules' K3 composite); per size: D = batched - one-row.
int Pass(Engine& e, int argc, char** argv) {
	const std::vector<int> sizes = ParseList(Arg(argc, argv, 3, "--sizes", "1,2,3,4,5,6,7,8"));
	const int pairs = std::atoi(Arg(argc, argv, 3, "--pairs", "15"));
	const int warmup = std::atoi(Arg(argc, argv, 3, "--warmup", "1"));
	const bool one_row = !Flag(argc, argv, 3, "--no-one-row");
	std::vector<int32_t> prompt;
	if (Flag(argc, argv, 3, "--example")) {
		prompt = e.Tokenize(ExamplePrompt(Arg(argc, argv, 3, "--utterance", kExampleUtterance)));
	} else {
		const int depth = std::atoi(Arg(argc, argv, 3, "--depth", "100"));
		for (int i = 0; i < depth; ++i) prompt.push_back((i * 37 + 11) % 1000 + 100);
	}
	const size_t n = sizes.size();
	std::vector<std::vector<double>> ms(n), ones(n);
	for (int round = 0; round < warmup + pairs; ++round) {
		for (size_t q = 0; q < n; ++q) {
			const size_t si = (q + static_cast<size_t>(round)) % n;
			const int m = sizes[si];
			std::vector<int32_t> span;
			for (int i = 0; i < m; ++i) span.push_back((i * 53 + 7) % 1000 + 100);
			{
				sslm_seq s = e.Start(nullptr, prompt);
				int32_t consumed = 0;
				const auto t0 = Clock::now();
				sslm_prefill(e.model, s, span.data(), m, m, SSLM_SPAN_PROMPT, e.ws, &consumed);
				const double d = MsSince(t0);
				if (round >= warmup) ms[si].push_back(d);
				sslm_seq_release(s);
			}
			if (one_row) {
				sslm_seq s = e.Start(nullptr, prompt);
				const auto t0 = Clock::now();
				for (int32_t tok : span) {
					int32_t consumed = 0;
					sslm_prefill(e.model, s, &tok, 1, 1, SSLM_SPAN_PROMPT, e.ws, &consumed);
				}
				const double d = MsSince(t0);
				if (round >= warmup) ones[si].push_back(d);
				sslm_seq_release(s);
			}
		}
	}
	auto index_of = [&](int m) -> int {
		for (size_t i = 0; i < n; ++i)
			if (sizes[i] == m) return static_cast<int>(i);
		return -1;
	};
	const int i1 = index_of(1);
	std::string j = "{\"mode\":\"pass\",\"depth\":" + std::to_string(prompt.size()) + ",\"example\":" +
	                (Flag(argc, argv, 3, "--example") ? "true" : "false") + ",\"tasks\":" + Arg(argc, argv, 3, "--tasks", "0") +
	                ",\"pairs\":" + std::to_string(pairs) + ",\"warmup\":" + std::to_string(warmup) +
	                ",\"order\":\"rounds outer, sizes rotated by one per round\",\"sizes\":[";
	for (size_t si = 0; si < n; ++si) {
		const int m = sizes[si];
		std::vector<double> r;
		if (i1 >= 0)
			for (size_t k = 0; k < ms[si].size(); ++k) r.push_back(ms[static_cast<size_t>(i1)][k] > 0 ? ms[si][k] / ms[static_cast<size_t>(i1)][k] : 0);
		const Stats st = Summarize(ms[si]), so = Summarize(ones[si]), sr = Summarize(r);
		std::printf("pass M %d depth %zu ms_median %.3f ms_q1 %.3f ms_q3 %.3f r_median %.3f", m, prompt.size(), st.median, st.q1,
		            st.q3, sr.median);
		if (one_row) std::printf(" one_row_ms_median %.3f d_ms_median %.3f", so.median, Summarize(PairDiff(ms[si], ones[si])).median);
		std::printf("\n");
		j += std::string(si ? "," : "") + "{\"M\":" + std::to_string(m) + ",\"batched_ms\":" + JPaired(ms[si]) +
		     ",\"r\":" + JPaired(r);
		if (one_row) j += ",\"one_row_ms\":" + JPaired(ones[si]) + ",\"d_ms\":" + JPaired(PairDiff(ms[si], ones[si]));
		j += "}";
	}
	j += "]";
	const int i2 = index_of(2), i3 = index_of(3), i5 = index_of(5), i6 = index_of(6);
	if (i1 >= 0 && i2 >= 0 && i3 >= 0 && i5 >= 0 && i6 >= 0) {
		std::vector<double> sc;
		for (size_t k = 0; k < ms[static_cast<size_t>(i1)].size(); ++k) {
			const double t1 = ms[static_cast<size_t>(i1)][k];
			auto rr = [&](int i) { return t1 > 0 ? ms[static_cast<size_t>(i)][k] / t1 : 0.0; };
			sc.push_back(rr(i3) + 2 * rr(i6) + rr(i5) + rr(i2));
		}
		std::printf("pass S_composite_median %.3f\n", Summarize(sc).median);
		j += ",\"s_composite\":" + JPaired(sc);
	}
	j += "}";
	return WriteJson(argc, argv, j) ? 0 : 2;
}

#if !defined(SFD_BENCH_DECODE_ONLY)
int Finish(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const std::vector<int32_t> prompt = e.Tokenize(PromptText(argc, argv));
	const int pairs = std::atoi(Arg(argc, argv, 4, "--pairs", "15"));
	sslm_decode_params modes[2];
	modes[0] = e.Greedy();
	const bool damped = e.Damped(&modes[1]);
	std::string j = "{\"mode\":\"finish\",\"schema\":" + JStr(schema) + ",\"pairs\":" + std::to_string(pairs) + ",\"arms\":[";
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
		j += std::string(mi ? "," : "") + "{\"params\":" + JStr(mname) + ",\"unbound_ms\":" + JStats(ms) + ",\"bound\":[";
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
					if (sslm_seq_jump_forward(e.model, s, 64, mi ? &modes[mi] : nullptr, /*opts=*/nullptr, e.ws, buf.data(), 64,
					                          &forced) != SSLM_OK) break;
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
			const int32_t tok = k < toks.size() ? toks[k] : -100;
			std::printf("finish %s %s index %zu token %d ms_median %.3f\n", mname, schema, k, tok, b.median);
			j += std::string(k ? "," : "") + "{\"index\":" + std::to_string(k) + ",\"token\":" + std::to_string(tok) +
			     ",\"ms\":" + JStats(per[k]) + "}";
		}
		j += "]}";
	}
	j += "]}";
	return WriteJson(argc, argv, j) ? 0 : 2;
}

// B1's finish at chosen legal-row counts, on the artifact's own head and the library's own finish
// primitives (the ones sslm_decode_step_v2 calls), so the row count is the bench's choice and not the
// schema's.
int FinishRows(int argc, char** argv) {
	const int tasks = std::atoi(Arg(argc, argv, 3, "--tasks", "0"));
	const int pairs = std::atoi(Arg(argc, argv, 3, "--pairs", "15"));
	const int warmup = std::atoi(Arg(argc, argv, 3, "--warmup", "1"));
	std::vector<uint8_t> raw;
	if (!superslm_marshal::ReadFile(argv[1], raw) || raw.empty()) {
		std::fprintf(stderr, "cannot read %s\n", argv[1]);
		return 2;
	}
	uint8_t* bytes = static_cast<uint8_t*>(::operator new(raw.size(), std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
	std::memcpy(bytes, raw.data(), raw.size());
	int rc = 0;
	{
		superslm::SslmModelView view;
		std::string err;
		if (superslm::SslmModel::Load(bytes, raw.size(), view, &err) != superslm::SslmModelStatus::Ok) {
			std::fprintf(stderr, "SslmModel::Load failed: %s\n", err.c_str());
			::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
			return 2;
		}
		const size_t hidden = view.config.hidden_size;
		const int32_t vocab = static_cast<int32_t>(view.config.vocab_size);
		const superslm::SslmTensorView* head = view.config.tie_word_embeddings ? view.weights.Tensor("embed")
		                                                                       : view.weights.Tensor("lm_head");
		if (!head || head->elem_count != static_cast<uint64_t>(hidden) * static_cast<uint64_t>(vocab)) {
			std::fprintf(stderr, "no head tensor of %zu x %d\n", hidden, vocab);
			::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
			return 2;
		}
		const int8_t* head_w = reinterpret_cast<const int8_t*>(head->data);
		const std::vector<int> rows = ParseList(Arg(argc, argv, 3, "--rows", "0,1,22,1000,V"), vocab);
		std::vector<int8_t> codes(hidden);
		for (size_t i = 0; i < hidden; ++i) codes[i] = static_cast<int8_t>(static_cast<int>((i * 37 + 5) % 201) - 100);
		std::vector<int64_t> wide(static_cast<size_t>(vocab));
		std::vector<int32_t> logits(static_cast<size_t>(vocab)), keep(static_cast<size_t>(vocab));
		ReferenceParallelFor* threads = tasks >= 2 ? new ReferenceParallelFor(tasks - 1) : nullptr;
		const sslm_parallel_for pf{threads ? &ReferenceParallelFor::Run : nullptr, threads, tasks >= 2 ? tasks : 1, 0u};
		// Damped greedy's constants, from the artifact (decode_params_init), and a fed anti-LM state.
		sslm_model model = nullptr;
		sslm_decode_params dp{};
		bool damped = false;
		if (sslm_model_map(bytes, raw.size(), &model) == SSLM_OK) {
			int32_t layers = 0;
			sslm_decode_params p{};
			for (int32_t l = 256; l >= 1; --l)
				if (sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, l, &p) == SSLM_OK) { layers = l; break; }
			damped = layers > 0 && sslm_decode_params_init(model, SSLM_DECODE_MODE_DAMPED_GREEDY, layers, &dp) == SSLM_OK;
		}
		superslm::AntiLmState* alm = damped ? superslm::AntiLmCreate(dp.anti_lm_max_order) : nullptr;
		for (int i = 0; alm && i < 16; ++i) superslm::AntiLmUpdate(alm, (i * 977 + 13) % vocab);
		std::vector<int32_t> idx(damped ? static_cast<size_t>(dp.top_k) : 1);
		std::vector<int64_t> qth(damped ? static_cast<size_t>(dp.top_k) : 1);
		std::printf("finishrows hidden %zu vocab %d tasks %d pairs %d damped %d\n", hidden, vocab, tasks, pairs, damped ? 1 : 0);
		std::string j = "{\"mode\":\"finishrows\",\"hidden\":" + std::to_string(hidden) + ",\"vocab\":" + std::to_string(vocab) +
		                ",\"tasks\":" + std::to_string(tasks) + ",\"pairs\":" + std::to_string(pairs) +
		                ",\"warmup\":" + std::to_string(warmup) + ",\"order\":\"rounds outer, row counts rotated by one per round\"" +
		                ",\"damped_available\":" + (damped ? "true" : "false") + ",\"rows\":[";
		// Rounds outer, the row counts inner and rotated by one per round, so the per-round differences the
		// rules take (f(V) - f(1000), f_d(n) - f_d(V)) are paired within a round.
		const size_t R = rows.size();
		std::vector<std::vector<uint32_t>> ids(R);
		std::vector<std::vector<uint8_t>> masks(R);
		std::vector<int> ns(R);
		for (size_t ri = 0; ri < R; ++ri) {
			const int n = std::max(0, std::min(rows[ri], vocab));
			ns[ri] = n;
			ids[ri].resize(static_cast<size_t>(n));
			masks[ri].assign((static_cast<size_t>(vocab) + 7) / 8, 0);
			for (int k = 0; k < n; ++k) {
				const uint32_t id = static_cast<uint32_t>((static_cast<uint64_t>(k) * static_cast<uint64_t>(vocab)) / static_cast<uint64_t>(n));
				ids[ri][static_cast<size_t>(k)] = id;
				masks[ri][id >> 3] = static_cast<uint8_t>(masks[ri][id >> 3] | (1u << (id & 7)));
			}
		}
		std::vector<std::vector<double>> rows_ms(R), greedy_ms(R), damped_ms(R);
		std::vector<int32_t> gtok(R, -1), dtok(R, -1);
		std::vector<char> drefused(R, 0);
		for (int round = 0; round < warmup + pairs; ++round) {
			for (size_t q = 0; q < R; ++q) {
				const size_t ri = (q + static_cast<size_t>(round)) % R;
				const int n = ns[ri];
				const uint8_t* mask = masks[ri].data();
				const bool keep_it = round >= warmup;
				// Greedy: rows, then the re-mask and argmax over V.
				auto t0 = Clock::now();
				if (superslm::LogitsSiteLegalRows(codes.data(), hidden, head_w, static_cast<size_t>(vocab), ids[ri].data(),
				                                  static_cast<size_t>(n), wide.data(), logits.data(), &pf) !=
				    superslm::SslmForwardStatus::Ok) {
					rc = 1;
				}
				const double rows_d = MsSince(t0);
				keep = logits;
				superslm::ApplyMaskAndArgmax(logits.data(), mask, vocab, &gtok[ri]);
				const double greedy_d = MsSince(t0);
				if (keep_it) rows_ms[ri].push_back(rows_d), greedy_ms[ri].push_back(greedy_d);
				if (damped) {
					// Damped: the same rows, then the decode step's V-wide masking loop and the scorer.
					logits = keep;
					t0 = Clock::now();
					if (superslm::LogitsSiteLegalRows(codes.data(), hidden, head_w, static_cast<size_t>(vocab), ids[ri].data(),
					                                  static_cast<size_t>(n), wide.data(), logits.data(), &pf) !=
					    superslm::SslmForwardStatus::Ok) {
						rc = 1;
					}
					for (int32_t t = 0; t < vocab; ++t)
						if (!((mask[static_cast<size_t>(t) >> 3] >> (t & 7)) & 1u)) logits[static_cast<size_t>(t)] = INT32_MIN;
					bool refused = false;
					if (!superslm::DampedGreedyScoreAndArgmaxWithScratch(logits.data(), mask, vocab, dp.top_k, alm, dp.alpha_q15,
					                                                      dp.q_ln2, dp.q_b, dp.q_c, idx.data(), qth.data(),
					                                                      &dtok[ri], &refused)) {
						refused = true;
					}
					if (refused) drefused[ri] = 1;
					if (keep_it) damped_ms[ri].push_back(MsSince(t0));
				}
			}
		}
		for (size_t ri = 0; ri < R; ++ri) {
			const int n = ns[ri];
			const Stats a = Summarize(rows_ms[ri]), g = Summarize(greedy_ms[ri]), d = Summarize(damped_ms[ri]);
			std::printf("finishrows n %d rows_ms_median %.3f greedy_ms_median %.3f greedy_q1 %.3f greedy_q3 %.3f", n, a.median,
			            g.median, g.q1, g.q3);
			if (damped) std::printf(" damped_ms_median %.3f damped_q1 %.3f damped_q3 %.3f damped_refused %d", d.median, d.q1, d.q3,
			                        drefused[ri] ? 1 : 0);
			std::printf("\n");
			j += std::string(ri ? "," : "") + "{\"n\":" + std::to_string(n) + ",\"rows_ms\":" + JPaired(rows_ms[ri]) +
			     ",\"greedy_ms\":" + JPaired(greedy_ms[ri]) + ",\"greedy_token\":" + std::to_string(gtok[ri]);
			if (damped) j += ",\"damped_ms\":" + JPaired(damped_ms[ri]) + ",\"damped_refused\":" + (drefused[ri] ? "true" : "false");
			j += "}";
		}
		// The full finish, as `main` runs it under a bound schema (and as unbound decode runs its rows):
		// every row through LogitsSiteParallel, then the re-mask and argmax over an all-ones page.
		std::vector<double> full_ms, full_greedy_ms;
		const std::vector<uint8_t> all_ones((static_cast<size_t>(vocab) + 7) / 8, 0xFF);
		int32_t ftok = -1;
		for (int i = 0; i < warmup + pairs; ++i) {
			const auto t0 = Clock::now();
			if (superslm::LogitsSiteParallel(codes.data(), hidden, head_w, static_cast<size_t>(vocab), wide.data(), logits.data(),
			                                 &pf) != superslm::SslmForwardStatus::Ok) {
				rc = 1;
			}
			const double rows_d = MsSince(t0);
			superslm::ApplyMaskAndArgmax(logits.data(), all_ones.data(), vocab, &ftok);
			if (i >= warmup) full_ms.push_back(rows_d), full_greedy_ms.push_back(MsSince(t0));
		}
		const Stats f = Summarize(full_ms), fg = Summarize(full_greedy_ms);
		std::printf("finishrows full_rows_ms_median %.3f q1 %.3f q3 %.3f full_greedy_ms_median %.3f q1 %.3f q3 %.3f\n", f.median,
		            f.q1, f.q3, fg.median, fg.q1, fg.q3);
		j += "],\"full_rows_ms\":" + JPaired(full_ms) + ",\"full_greedy_ms\":" + JPaired(full_greedy_ms) +
		     ",\"full_greedy_token\":" + std::to_string(ftok) + "}";
		superslm::AntiLmDestroy(alm);
		if (model) sslm_model_unmap(model);
		delete threads;
		if (!WriteJson(argc, argv, j)) rc = 2;
	}
	::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	return rc;
}

// ---- B4: quality -------------------------------------------------------------------------------

std::string LabelString(const JVal* v) {
	if (!v) return std::string();
	if (v->kind == JVal::kStr) return v->str;
	if (v->kind == JVal::kBool) return v->b ? "true" : "false";
	if (v->kind == JVal::kNum) return JNum(v->num);
	return std::string();
}

// The te368 census's looks_like_answer (t2911_final_census.py), over UTF-8 bytes: any byte >= 0x80
// counts as alphanumeric, which Python's str.isalnum also reads for the letters of other scripts.
bool LooksLikeAnswer(std::string s) {
	const std::string pre = "<unparsed> ";
	if (s.compare(0, pre.size(), pre) == 0) s.erase(0, pre.size());
	if (s.empty()) return false;
	if (std::strchr(":{>,[}", s[0])) return false;
	for (unsigned char c : s)
		if (std::isalnum(c) || c >= 0x80) return true;
	return false;
}

int Quality(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const char* pop_path = Arg(argc, argv, 4, "--population", nullptr);
	const std::string kind = Arg(argc, argv, 4, "--kind", "potion");
	const int from = std::atoi(Arg(argc, argv, 4, "--from", "0"));
	const int count = std::atoi(Arg(argc, argv, 4, "--count", "20"));
	const int max_new = std::atoi(Arg(argc, argv, 4, "--max-new", kind == "potion" ? "64" : "300"));
	const int32_t special_from = std::atoi(Arg(argc, argv, 4, "--special-from", "151643"));
	const int32_t budget = std::atoi(Arg(argc, argv, 4, "--budget", "64"));
	std::string text;
	if (!pop_path || !ReadText(pop_path, &text)) {
		std::fprintf(stderr, "cannot read --population\n");
		return 2;
	}
	struct Item {
		std::string id, prompt;
		std::vector<std::pair<std::string, std::string>> labels;  // field -> expected text
	};
	std::vector<Item> items;
	static const char* kFields[] = {"intent", "item", "quantity", "polite"};
	if (kind == "potion") {
		std::istringstream ls(text);
		std::string line;
		int lineno = 0;
		while (std::getline(ls, line)) {
			++lineno;
			if (!line.empty() && line.back() == '\r') line.pop_back();
			if (line.find_first_not_of(" \t") == std::string::npos) continue;
			JVal v;
			if (!ParseJson(line, &v) || v.kind != JVal::kObj) {
				std::fprintf(stderr, "population line %d is not a JSON object\n", lineno);
				return 2;
			}
			Item it;
			it.id = LabelString(v.Get("id"));
			if (it.id.empty()) it.id = std::to_string(items.size());
			// The utterance (`customer_line`, `utterance` or `text`) is wrapped in the example's extraction
			// prompt; a `prompt` field is the already-wrapped string and must equal that wrapping.
			const JVal* u = v.Get("customer_line");
			if (!u) u = v.Get("utterance");
			if (!u) u = v.Get("text");
			const JVal* pr = v.Get("prompt");
			if ((u && u->kind != JVal::kStr) || (pr && pr->kind != JVal::kStr) || (!u && !pr)) {
				std::fprintf(stderr, "population line %d has no customer_line/utterance/text or prompt string\n", lineno);
				return 2;
			}
			it.prompt = u ? ExamplePrompt(u->str) : pr->str;
			if (u && pr && pr->str != it.prompt) {
				std::fprintf(stderr, "population line %d: its prompt is not the example's wrapping of its line\n", lineno);
				return 2;
			}
			const JVal* lab = v.Get("labels");
			if (!lab) lab = v.Get("expected");
			if (!lab) lab = &v;
			for (const char* f : kFields) {
				const JVal* x = lab->Get(f);
				if (!x) {
					std::fprintf(stderr, "population line %d has no label %s\n", lineno, f);
					return 2;
				}
				it.labels.emplace_back(f, LabelString(x));
			}
			items.push_back(std::move(it));
		}
	} else if (kind == "prompt_result") {
		JVal v;
		if (!ParseJson(text, &v) || v.kind != JVal::kArr) {
			std::fprintf(stderr, "population is not a JSON array\n");
			return 2;
		}
		for (size_t i = 0; i < v.arr.size(); ++i) {
			if (v.arr[i].kind != JVal::kStr) {
				std::fprintf(stderr, "population entry %zu is not a string\n", i);
				return 2;
			}
			items.push_back(Item{std::to_string(i), QwenDefaultPrompt(v.arr[i].str), {}});
		}
	} else {
		std::fprintf(stderr, "--kind must be potion or prompt_result\n");
		return 2;
	}
	if (from < 0 || count < 1 || static_cast<size_t>(from) >= items.size()) {
		std::fprintf(stderr, "--from %d --count %d outside a population of %zu\n", from, count, items.size());
		return 2;
	}
	const size_t last = std::min(items.size(), static_cast<size_t>(from) + static_cast<size_t>(count));
	// --sealed PATH: the on arm's records, its tallies and every off-vs-on readout go to PATH and not to
	// --json or the console, so the off arm's outputs can be read (the rules' B4 commissioning controls)
	// before the on arm's are unsealed.
	const char* sealed = Arg(argc, argv, 4, "--sealed", nullptr);
	struct Tally {
		int prompts = 0, parsed = 0, schema_valid = 0, all_fields = 0, accepting = 0, answer = 0, clean = 0;
		int field_ok[4] = {0, 0, 0, 0};
	};
	Tally tally[2];
	int agree = 0, same_fields = 0, same_fields_diff_bytes = 0;
	std::string rows_json[2];
	std::string cross_json;
	auto show = [&](int a) { return a == 0 || !sealed; };
	for (size_t i = static_cast<size_t>(from); i < last; ++i) {
		const Item& it = items[i];
		const std::vector<int32_t> prompt = e.Tokenize(it.prompt);
		Arm arm[2];
		arm[0] = RunOff(e, schema, prompt, max_new, /*stop_on_accept=*/true);
		arm[1] = RunOn(e, schema, prompt, budget, max_new, /*stop_on_accept=*/true);
		const bool same = arm[0].stream == arm[1].stream;
		if (same) ++agree;
		std::string got_fields[2];
		std::string texts[2];
		std::printf("item %zu id %s", i, it.id.c_str());
		for (int a = 0; a < 2; ++a) {
			Tally& t = tally[a];
			++t.prompts;
			const std::string out = e.Detokenize(arm[a].stream);
			texts[a] = out;
			bool special = false;
			for (int32_t id : arm[a].stream) special = special || id >= special_from;
			JVal obj;
			const bool parsed = ParseJson(out, &obj) && obj.kind == JVal::kObj;
			const bool accepted = std::strcmp(arm[a].stop, "accepting") == 0;
			std::string rec = "{\"index\":" + std::to_string(i) + ",\"id\":" + JStr(it.id) + ",\"prompt_tokens\":" +
			                  std::to_string(prompt.size()) + ",\"prompt_digest\":" + JStr(Hex64(Digest(prompt))) +
			                  ",\"stop\":" + JStr(arm[a].stop) + ",\"accepting_end\":" + (arm[a].accepting_end ? "true" : "false") +
			                  ",\"tokens\":" + std::to_string(arm[a].stream.size()) + ",\"digest\":" +
			                  JStr(Hex64(Digest(arm[a].stream))) + ",\"stream\":" + JIds(arm[a].stream) +
			                  ",\"special\":" + (special ? "true" : "false") + ",\"text\":" + JStr(out) +
			                  ",\"json\":" + (parsed ? "true" : "false");
			if (parsed) ++t.parsed;
			if (accepted) ++t.accepting;
			if (kind == "potion") {
				const bool valid = parsed && PotionValid(obj);
				if (valid) ++t.schema_valid;
				int ok = 0;
				rec += std::string(",\"schema_valid\":") + (valid ? "true" : "false") + ",\"fields\":{";
				for (size_t f = 0; f < it.labels.size(); ++f) {
					// A run that is not schema-valid JSON scores every field wrong (the rules' §7.2).
					const JVal* gv = valid ? obj.Get(it.labels[f].first.c_str()) : nullptr;
					const std::string got = gv ? LabelString(gv) : std::string();
					const bool hit = gv && got == it.labels[f].second;
					if (hit) ++ok, ++t.field_ok[f];
					got_fields[a] += got + "|";
					rec += std::string(f ? "," : "") + JStr(it.labels[f].first) + ":{\"want\":" + JStr(it.labels[f].second) +
					       ",\"got\":" + (gv ? JStr(got) : std::string("null")) + ",\"ok\":" + (hit ? "true" : "false") + "}";
				}
				rec += "}";
				if (ok == static_cast<int>(it.labels.size())) ++t.all_fields;
				if (show(a)) std::printf(" %s fields %d stop %s", a ? "on" : "off", ok, arm[a].stop);
			} else {
				std::string value;
				const JVal* pr = parsed ? obj.Get("Prompt_Result") : nullptr;
				if (pr && pr->kind == JVal::kStr) {
					value = pr->str;
				} else {
					const std::string key = "\"Prompt_Result\":";
					const size_t at = out.find(key);
					std::string rawv = at == std::string::npos ? out : out.substr(at + key.size());
					if (!rawv.empty() && rawv[0] == '"') rawv.erase(0, 1);
					value = "<unparsed> " + rawv;
				}
				const bool answer = LooksLikeAnswer(value) && !special;
				if (answer) ++t.answer;
				if (answer && accepted) ++t.clean;
				rec += ",\"value\":" + JStr(value) + ",\"answer\":" + (answer ? "true" : "false") +
				       ",\"clean_accept\":" + (answer && accepted ? "true" : "false");
				if (show(a)) std::printf(" %s clean %d stop %s", a ? "on" : "off", answer && accepted ? 1 : 0, arm[a].stop);
			}
			rows_json[a] += std::string(rows_json[a].empty() ? "" : ",") + rec + "}";
		}
		std::printf("\n");
		const bool sf = kind == "potion" && got_fields[0] == got_fields[1];
		if (sf) ++same_fields;
		if (sf && texts[0] != texts[1]) ++same_fields_diff_bytes;
		cross_json += std::string(cross_json.empty() ? "" : ",") + "{\"index\":" + std::to_string(i) + ",\"id\":" + JStr(it.id) +
		              ",\"agree\":" + (same ? "true" : "false") + ",\"same_fields\":" + (sf ? "true" : "false") +
		              ",\"same_bytes\":" + (texts[0] == texts[1] ? "true" : "false") + "}";
	}
	auto tally_json = [&](int a) {
		const Tally& t = tally[a];
		std::string o = "{\"prompts\":" + std::to_string(t.prompts) + ",\"parsed\":" + std::to_string(t.parsed) +
		                ",\"accepted\":" + std::to_string(t.accepting);
		if (show(a)) std::printf("%s prompts %d parsed %d accepted %d", a ? "on" : "off", t.prompts, t.parsed, t.accepting);
		if (kind == "potion") {
			o += ",\"schema_valid\":" + std::to_string(t.schema_valid) + ",\"all_fields\":" + std::to_string(t.all_fields) +
			     ",\"field_correct\":{";
			for (int f = 0; f < 4; ++f) o += std::string(f ? "," : "") + JStr(kFields[f]) + ":" + std::to_string(t.field_ok[f]);
			o += "}";
			if (show(a))
				std::printf(" schema_valid %d all_fields %d intent %d item %d quantity %d polite %d", t.schema_valid, t.all_fields,
				            t.field_ok[0], t.field_ok[1], t.field_ok[2], t.field_ok[3]);
		} else {
			o += ",\"answer\":" + std::to_string(t.answer) + ",\"clean_accept\":" + std::to_string(t.clean);
			if (show(a)) std::printf(" answer %d clean_accept %d", t.answer, t.clean);
		}
		if (show(a)) std::printf("\n");
		return o + ",\"items\":[" + rows_json[a] + "]}";
	};
	const std::string head = "{\"mode\":\"quality\",\"schema\":" + JStr(schema) + ",\"kind\":" + JStr(kind) + ",\"from\":" +
	                         std::to_string(from) + ",\"count\":" + std::to_string(last - static_cast<size_t>(from)) +
	                         ",\"max_new\":" + std::to_string(max_new) + ",\"budget\":" + std::to_string(budget) +
	                         ",\"special_from\":" + std::to_string(special_from) + ",\"population_size\":" +
	                         std::to_string(items.size());
	const std::string off_json = tally_json(0);
	const std::string on_json = tally_json(1);
	const std::string cross = ",\"agree\":" + std::to_string(agree) + ",\"same_fields\":" + std::to_string(same_fields) +
	                          ",\"same_fields_different_bytes\":" + std::to_string(same_fields_diff_bytes) +
	                          ",\"pairs\":[" + cross_json + "]";
	if (!sealed) std::printf("agree %d of %zu\n", agree, last - static_cast<size_t>(from));
	std::string j = head + ",\"sealed\":" + (sealed ? "true" : "false") + ",\"arms\":{\"off\":" + off_json;
	if (!sealed) j += ",\"on\":" + on_json + "}" + cross + "}";
	else j += "}}";
	if (sealed) {
		std::ofstream f(sealed, std::ios::binary);
		f << head << ",\"arms\":{\"on\":" << on_json << "}" << cross << "}\n";
		if (!f) {
			std::fprintf(stderr, "cannot write %s\n", sealed);
			return 2;
		}
	}
	return WriteJson(argc, argv, j) ? 0 : 2;
}
#endif

int SummarizeMode(int argc, char** argv) {
	std::vector<double> v;
	for (int i = 3; i < argc; ++i) {
		if (std::strcmp(argv[i], "--json") == 0) {
			++i;
			continue;
		}
		v.push_back(std::atof(argv[i]));
	}
	const Stats s = Summarize(v);
	std::printf("summary n %zu median %.4f q1 %.4f q3 %.4f\n", v.size(), s.median, s.q1, s.q3);
	return WriteJson(argc, argv, JStats(v)) ? 0 : 2;
}

}  // namespace

#if defined(SFD_BENCH_DRAFT_TABLE_BUILDER)
// `drafttable SCHEMA --streams FILE --out PATH`: the D2 table, built offline (plan rev 1.3.4 §5.10.5, B7). FILE is a
// quality-mode result (readings.quality_off.arms.off.items[].stream, or arms.off.items[].stream): each off stream is
// walked through the schema from state 0, counting the model's token at every state. Every state's forced flag is
// the library's own (next_forced >= 0, through the read seam), so the timed bench, built without seams, can stop a
// learned draft at an unforced state. This build is a test build of the library; it times nothing.
int DraftTableBuild(Engine& e, int argc, char** argv) {
	const char* schema = argv[3];
	const char* streams = Arg(argc, argv, 4, "--streams", nullptr);
	const char* out_path = Arg(argc, argv, 4, "--out", nullptr);
	if (!streams || !out_path) {
		std::fprintf(stderr, "drafttable SCHEMA --streams FILE --out PATH\n");
		return 2;
	}
	SchemaWalker wk;
	if (!wk.Open(e, schema)) {
		std::fprintf(stderr, "schema %s not in the artifact\n", schema);
		return 2;
	}
	std::ifstream in(streams, std::ios::binary);
	const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	JVal root;
	if (!ParseJson(text, &root)) {
		std::fprintf(stderr, "cannot parse %s\n", streams);
		return 2;
	}
	const JVal* arms = root.Get("arms");
	if (const JVal* r = root.Get("readings"))
		if (const JVal* q = r->Get("quality_off")) arms = q->Get("arms");
	const JVal* off = arms ? arms->Get("off") : nullptr;
	const JVal* items = off ? off->Get("items") : nullptr;
	if (!items || items->kind != JVal::kArr) {
		std::fprintf(stderr, "no arms.off.items in %s\n", streams);
		return 2;
	}
	const uint32_t n_states = wk.entry->state_count;
	std::vector<std::vector<std::pair<int32_t, int64_t>>> counts(n_states);
	size_t walked = 0, tokens = 0, cut = 0;
	for (const JVal& item : items->arr) {
		const JVal* st = item.Get("stream");
		if (!st || st->kind != JVal::kArr) continue;
		++walked;
		uint32_t s = 0;
		for (const JVal& t : st->arr) {
			const int32_t tok = static_cast<int32_t>(t.num);
			auto& c = counts[s];
			auto it = std::find_if(c.begin(), c.end(), [&](const std::pair<int32_t, int64_t>& p) { return p.first == tok; });
			if (it == c.end()) c.push_back({tok, 1}); else ++it->second;
			++tokens;
			uint32_t nx = 0;
			if (!wk.Step(s, tok, &nx)) { ++cut; break; }
			s = nx;
		}
	}
	std::vector<uint32_t> forced;
	for (uint32_t s = 0; s < n_states; ++s) {
		int32_t nf = -1;
		uint32_t f_len = 0;
		if (superslm::test::sfd::ForcedTableEntry(e.model, static_cast<int32_t>(wk.index), s, &nf, &f_len) != SSLM_OK) {
			std::fprintf(stderr, "ForcedTableEntry failed at state %u\n", s);
			return 3;
		}
		if (nf >= 0) forced.push_back(s);
	}
	std::ofstream o(out_path, std::ios::binary);
	o << "SFD-DRAFT-TABLE/1 schema " << schema << " streams " << walked << " tokens " << tokens << "\n";
	o << "states " << n_states << "\n";
	o << "forced";
	for (uint32_t s : forced) o << ' ' << s;
	o << "\n";
	size_t seen = 0;
	for (uint32_t s = 0; s < n_states; ++s) {
		if (counts[s].empty()) continue;
		++seen;
		std::sort(counts[s].begin(), counts[s].end());
		o << "S " << s;
		for (const auto& tc : counts[s]) o << ' ' << tc.first << ':' << tc.second;
		o << "\n";
	}
	std::printf("drafttable %s: %zu streams, %zu tokens, %zu states seen of %u, %zu forced states, %zu streams cut\n", out_path,
	            walked, tokens, seen, n_states, forced.size(), cut);
	return 0;
}
#endif

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr,
		             "usage: %s ARTIFACT extract|spans|finish|quality SCHEMA [options], %s ARTIFACT pass|finishrows "
		             "[options], or %s - summarize X...\n",
		             argv[0], argv[0], argv[0]);
		return 2;
	}
	const std::string mode = argv[2];
	if (mode == "summarize") return SummarizeMode(argc, argv);
	if (mode == "pool") return Pool(argc, argv);
#if !defined(SFD_BENCH_DECODE_ONLY)
	if (mode == "finishrows") return FinishRows(argc, argv);
#endif
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
	if (mode == "pass") return Pass(e, argc, argv);
#if !defined(SFD_BENCH_DECODE_ONLY)
	if (mode == "spans") return Spans(e, argc, argv);
	if (mode == "finish") return Finish(e, argc, argv);
	if (mode == "quality") return Quality(e, argc, argv);
#endif
#if defined(SFD_BENCH_DRAFT_TABLE_BUILDER)
	if (mode == "drafttable") return DraftTableBuild(e, argc, argv);
#endif
	std::fprintf(stderr, "unknown mode %s (or not in this build)\n", mode.c_str());
	return 2;
}
