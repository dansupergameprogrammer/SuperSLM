// Schema fast-decode plan rev 1.4.4 §11.3, P2.16's oracle (rev 1.4.1, audit G-2; rev 1.4.2, strike B4): a headless
// reference loop over the GPU engine that writes, before plugin P2 is built, the four expected values P2.16 compares
// for each (status, mode, point): the token stream, each id's kind (forced or produced), ForcedTokenCount and the
// ending. Built in G1-J; run on the box in GB-J (step g) on the fixture P2.16 maps. Box only: it needs a hardware
// D3D12 device. The cloud compiles and links it (tests/schema-fast-decode-gpu/build_sfd_gpu_mingw.sh).
//
// The loop is plan §6.1's whole-token path in Canonical, at R_max rows per advance (--r-max, default 2: P2.16's
// verb is called only at R_max >= 2, §6.1 *At R_max = 1*), with §6.1's clip (R_max, the context room, the allowance
// plus the pending row) and its drain. The `Rest` shape picks the verb's argument and the finishing call:
//   Primed       (after the prompt)       a start chain: the verb with -1; otherwise the decode bridge finishes it;
//   TokenInHand  (after any finish)       a chain after the token: the verb with that token; otherwise the decode
//                                         bridge embeds and finishes it;
//   Ready        (an action that ended    a chain: the verb with -1; otherwise the decode bridge's shortcut.
//                 with no decode: drain)
// After an admitting verb call the action decodes through the shortcut unless the chain is still open (the drain)
// or the allowance is spent (MaxNewTokens).
//
// The injection (P2.16): at the point, the verb call is SKIPPED and the status is returned with out_forced_count 0
// -- the engine's rest-at-entry and h = 0 state, which G12.7 and G12.6 pin. The points: `first` (the first action
// after the prompt, Primed with a start chain), `choice` (the first TokenInHand action with a chain) and `midchain`
// (the first Ready action with a chain open). Then §6.1's status map:
//   SSLM_GPU_CONTEXT_CAP_EXCEEDED, SSLM_GPU_KV_BUDGET_EXCEEDED   the committed rows (0) are delivered and the
//                                    generation ends at the limit (endings ContextCap, KvBudgetExceeded);
//   SSLM_GPU_FORCED_TABLE_ABSENT, SSLM_BUSY, an unknown value,
//   SSLM_GPU_ALLOCATION_FAILED, SSLM_SEQUENCE_REJECTED            the action runs off's plain decode with the same
//                                    token, and the generation continues in off's loop (the hybrid stream, C17);
//   SSLM_DEVICE_LOST                 off's probe and loss path: the generation ends with a fault at the same id
//                                    count. `healthy` (the probe passes: RecoverablePerSequenceRejection) and
//                                    `terminal` (TerminalDeviceLost) share the stream and differ in the ending.
// The baselines `off` (off's loop throughout) and `canonical` (no injection) are written too.
//
// Endings: SchemaDeadEnd (a -2), MaxNewTokens, ContextCap, KvBudgetExceeded,
// Fault:RecoverablePerSequenceRejection, Fault:TerminalDeviceLost, and Error:<status> for a status the loop did not
// inject (a defect: the tool exits 1). Kinds: one letter per id, F forced, P produced.
//
// Output: JSON (--out=FILE, default P2_16_expected.json in the working directory): the inputs (model sha256, schema,
// prompt ids, R_max, MaxNewTokens), the cases, and "cases_sha256", the sha256 of the cases array's exact text.
//
// Run: schema_fast_decode_gpu_p2_ref.exe [--fixture-dir=DIR] [--model=FILE] [--schema=NAME] [--prompt=TEXT]
//                                        [--r-max=N] [--max-new=N] [--out=FILE]
// --model defaults to DIR/fsfd.sslm, --schema to potion_shop_order, --prompt to the potion prompt. The tool runs
// with whatever SSLM_GPU_ENABLE_DEBUG_LAYER the driver sets; it sets none.
#include "sfdg_helpers.h"

#include <algorithm>

using namespace sfdg;

namespace {

struct Args {
	std::string model, schema = kPotion, prompt = kPromptPotionText, out = "P2_16_expected.json";
	int32_t r_max = 2, max_new = 64;
};

std::string Opt(int argc, char** argv, const char* key) {
	const size_t n = std::strlen(key);
	for (int i = 1; i < argc; ++i)
		if (std::strncmp(argv[i], key, n) == 0) return argv[i] + n;
	return {};
}

enum class Rest { Primed, TokenInHand, Ready };

struct Injection {
	bool on = false;
	const char* point = "";  // first | choice | midchain
	GS status = GS::SSLM_OK;
	bool terminal = false;  // SSLM_DEVICE_LOST only
};

struct Case {
	std::string name, status, point;
	std::vector<int32_t> stream;
	std::string kinds;
	int64_t forced = 0;
	std::string ending;
	bool reached = true;   // the injection point was met
	bool defect = false;   // a status the loop did not inject
};

const char* EndingForLimit(GS st) {
	return st == GS::SSLM_GPU_KV_BUDGET_EXCEEDED ? "KvBudgetExceeded" : "ContextCap";
}

// One generation. `off` runs off's loop throughout; otherwise Canonical with `inj`.
Case Run(GRig& rig, const Args& a, const std::vector<int32_t>& prompt, bool off, const Injection& inj) {
	Case c;
	GSeq s = rig.New(a.schema.c_str());
	if (!s.h) {
		c.defect = true;
		c.ending = "Error:create";
		return c;
	}
	const GS ps = rig.Prompt(s, prompt);
	if (ps != GS::SSLM_OK) {
		c.defect = true;
		c.ending = std::string("Error:prompt:") + Name(ps);
		rig.ReleaseSeqs();
		return c;
	}
	const int64_t cap = static_cast<int64_t>(rig.env.fx.model_cap);
	Rest rest = Rest::Primed;
	int32_t pending = -1;   // the produced token in hand (TokenInHand)
	bool off_loop = off;    // §6.1: after a healthy failure, the generation runs off's loop from here
	bool injected = false;
	auto deliver = [&](int32_t id, bool forced) {
		c.stream.push_back(id);
		c.kinds += forced ? 'F' : 'P';
		if (forced) ++c.forced;
	};
	// Off's plain decode from the current shape: the decode bridge embeds the token in hand, or finishes the primed
	// row, or takes the shortcut on a ready row. Returns false when the generation ended.
	auto decode = [&]() -> bool {
		if (static_cast<int64_t>(c.stream.size()) >= a.max_new) {
			c.ending = "MaxNewTokens";
			return false;
		}
		if (rest == Rest::TokenInHand && CtxLen(s.h) >= cap) {  // ContextHasRoom: the embed needs a position
			c.ending = "ContextCap";
			return false;
		}
		if (rest == Rest::TokenInHand) s.last = pending;
		GS st = GS::SSLM_OK;
		const int32_t t = Token(rig.env, s, &st);
		if (st != GS::SSLM_OK) {
			c.defect = true;
			c.ending = std::string("Error:decode:") + Name(st);
			return false;
		}
		if (t == -2) {
			c.ending = "SchemaDeadEnd";
			return false;
		}
		deliver(t, false);
		pending = t;
		rest = Rest::TokenInHand;
		return true;
	};
	for (int action = 0; action < 4096; ++action) {
		if (static_cast<int64_t>(c.stream.size()) >= a.max_new) {
			c.ending = "MaxNewTokens";
			break;
		}
		int32_t chain = 0;
		if (!off_loop) {
			const GQuery q = QueryGpu(rig.ctx(), s.h, 0);
			if (q.st != GS::SSLM_OK) {
				c.defect = true;
				c.ending = std::string("Error:query:") + Name(q.st);
				break;
			}
			chain = q.count;
		}
		const int32_t pend = rest == Rest::TokenInHand ? 1 : 0;
		// The clip: R_max, the context room, and the allowance plus the pending row (already issued).
		const int64_t room = cap - CtxLen(s.h);
		const int64_t allowance = a.max_new - static_cast<int64_t>(c.stream.size()) + pend;
		const int32_t budget =
		    static_cast<int32_t>(std::max<int64_t>(0, std::min<int64_t>({a.r_max, room, allowance})));
		if (chain == 0 || budget - pend < 1) {
			if (!decode()) break;
			continue;  // no forced row survives the clip: off's plain decode
		}
		const char* point = rest == Rest::Primed ? "first" : rest == Rest::TokenInHand ? "choice" : "midchain";
		JumpOut j;
		if (inj.on && !injected && std::strcmp(point, inj.point) == 0) {
			injected = true;
			j.st = inj.status;  // the call is skipped: rest at entry, out_forced_count 0
		} else {
			j = Jump(rig, s, rest == Rest::TokenInHand ? pending : -1, budget);
		}
		if (j.st == GS::SSLM_OK) {
			for (int32_t id : j.forced) deliver(id, true);
			if (j.count > 0 || pend > 0) rest = Rest::Ready;  // admitted: the ready flag is set
			if (QueryGpu(rig.ctx(), s.h, 0).count > 0) continue;  // the drain: the chain is still open
			if (!decode()) break;  // the shortcut finishes the last admitted row
			continue;
		}
		if (j.st == GS::SSLM_GPU_CONTEXT_CAP_EXCEEDED || j.st == GS::SSLM_GPU_KV_BUDGET_EXCEEDED) {
			for (int32_t id : j.forced) deliver(id, true);
			c.ending = EndingForLimit(j.st);
			break;
		}
		if (j.st == GS::SSLM_DEVICE_LOST) {
			c.ending = inj.terminal ? "Fault:TerminalDeviceLost" : "Fault:RecoverablePerSequenceRejection";
			break;
		}
		// Every other status maps to off's loop: SSLM_GPU_FORCED_TABLE_ABSENT, SSLM_BUSY, the healthy failures
		// (SSLM_GPU_ALLOCATION_FAILED, SSLM_SEQUENCE_REJECTED) and an unknown value (audit N-4).
		if (!(inj.on && injected && j.st == inj.status)) {
			c.defect = true;  // a real verb call failed: not part of the oracle
			c.ending = std::string("Error:verb:") + Name(j.st);
			break;
		}
		off_loop = true;
		if (!decode()) break;  // off's plain decode with the same token
	}
	c.reached = !inj.on || injected;
	rig.ReleaseSeqs();
	return c;
}

std::string JsonIds(const std::vector<int32_t>& v) {
	std::string s = "[";
	for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s + "]";
}

std::string HexSha(const std::string& text) {
	superslm::Sha256 h;
	h.Update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
	uint8_t out[32];
	h.Final(out);
	static const char* hex = "0123456789abcdef";
	std::string s;
	for (uint8_t b : out) {
		s += hex[b >> 4];
		s += hex[b & 15];
	}
	return s;
}

}  // namespace

int main(int argc, char** argv) {
	Args a;
	a.model = Opt(argc, argv, "--model=");
	if (!Opt(argc, argv, "--schema=").empty()) a.schema = Opt(argc, argv, "--schema=");
	if (!Opt(argc, argv, "--prompt=").empty()) a.prompt = Opt(argc, argv, "--prompt=");
	if (!Opt(argc, argv, "--out=").empty()) a.out = Opt(argc, argv, "--out=");
	if (!Opt(argc, argv, "--r-max=").empty()) a.r_max = std::atoi(Opt(argc, argv, "--r-max=").c_str());
	if (!Opt(argc, argv, "--max-new=").empty()) a.max_new = std::atoi(Opt(argc, argv, "--max-new=").c_str());
	CHECK_MSG(a.r_max >= 2 && a.r_max <= 4, "--r-max=%d: P2.16 needs 2..4 (the verb is called only at R_max >= 2)",
	          a.r_max);
	CHECK_MSG(a.max_new >= 1, "--max-new=%d", a.max_new);
	if (a.model.empty()) {
		const std::string dir = FixtureDir(argc, argv);
		if (!dir.empty()) a.model = dir + "/fsfd.sslm";
	}
	GRig rig;
	if (a.model.empty() || !rig.env.Open(a.model, false)) {
		CHECK_MSG(false, "the model did not open: %s", a.model.c_str());
		return Finish("schema_fast_decode_gpu_p2_ref");
	}
	rig.raw = rig.env.fx.bytes;
	CHECK_MSG(rig.Index(a.schema.c_str()) >= 0, "schema %s not in the model", a.schema.c_str());
	CRig cpu(rig.raw);
	const std::vector<int32_t> prompt = cpu.Tokenize(a.prompt.c_str());
	CHECK_MSG(!prompt.empty(), "the prompt tokenized to nothing");

	struct StatusArm {
		const char* name;
		GS st;
		bool terminal;
	};
	const StatusArm arms[] = {
	    {"SSLM_DEVICE_LOST:healthy", GS::SSLM_DEVICE_LOST, false},
	    {"SSLM_DEVICE_LOST:terminal", GS::SSLM_DEVICE_LOST, true},
	    {"SSLM_GPU_ALLOCATION_FAILED", GS::SSLM_GPU_ALLOCATION_FAILED, false},
	    {"SSLM_SEQUENCE_REJECTED", GS::SSLM_SEQUENCE_REJECTED, false},
	    {"SSLM_GPU_KV_BUDGET_EXCEEDED", GS::SSLM_GPU_KV_BUDGET_EXCEEDED, false},
	    {"SSLM_GPU_CONTEXT_CAP_EXCEEDED", GS::SSLM_GPU_CONTEXT_CAP_EXCEEDED, false},
	    {"SSLM_GPU_FORCED_TABLE_ABSENT", GS::SSLM_GPU_FORCED_TABLE_ABSENT, false},
	    {"SSLM_BUSY", GS::SSLM_BUSY, false},
	    {"unknown:9999", static_cast<GS>(9999), false},
	};
	const char* points[] = {"first", "choice", "midchain"};

	std::vector<Case> cases;
	{
		Case off = Run(rig, a, prompt, true, Injection{});
		off.name = "baseline";
		off.status = "none";
		off.point = "off";
		cases.push_back(off);
		Case canon = Run(rig, a, prompt, false, Injection{});
		canon.name = "baseline";
		canon.status = "none";
		canon.point = "canonical";
		CHECK_MSG(canon.forced > 0, "the canonical baseline forced nothing: the fixture has no chain");
		cases.push_back(canon);
	}
	for (const StatusArm& arm : arms) {
		for (const char* p : points) {
			Injection inj;
			inj.on = true;
			inj.point = p;
			inj.status = arm.st;
			inj.terminal = arm.terminal;
			Case c = Run(rig, a, prompt, false, inj);
			c.name = "inject";
			c.status = arm.name;
			c.point = p;
			CHECK_MSG(c.reached, "[%s at %s] the point was never reached", arm.name, p);
			cases.push_back(c);
		}
	}
	for (const Case& c : cases)
		CHECK_MSG(!c.defect, "[%s %s %s] %s", c.name.c_str(), c.status.c_str(), c.point.c_str(), c.ending.c_str());

	std::string arr = "[\n";
	for (size_t i = 0; i < cases.size(); ++i) {
		const Case& c = cases[i];
		arr += "    {\"case\": \"" + c.name + "\", \"mode\": \"" + (c.point == "off" ? "Off" : "Canonical") +
		       "\", \"status\": \"" + c.status + "\", \"point\": \"" + c.point + "\", \"stream\": " +
		       JsonIds(c.stream) + ", \"kinds\": \"" + c.kinds + "\", \"forced_token_count\": " +
		       std::to_string(c.forced) + ", \"ending\": \"" + c.ending + "\", \"reached\": " +
		       (c.reached ? "true" : "false") + "}" + (i + 1 < cases.size() ? ",\n" : "\n");
	}
	arr += "  ]";
	std::string schema_escaped;
	for (char ch : a.schema) {
		if (ch == '"' || ch == '\\') schema_escaped += '\\';
		schema_escaped += ch;
	}
	std::string json = "{\n  \"tool\": \"tools/schema_fast_decode_gpu_p2_ref.cpp\",\n  \"plan\": \"rev 1.4.4 P2.16\",\n";
	json += "  \"model_sha256\": \"" + HexSha(std::string(rig.raw.begin(), rig.raw.end())) + "\",\n";
	json += "  \"schema\": \"" + schema_escaped + "\",\n  \"prompt_ids\": " + JsonIds(prompt) + ",\n";
	json += "  \"r_max\": " + std::to_string(a.r_max) + ",\n  \"max_new_tokens\": " + std::to_string(a.max_new) + ",\n";
	json += "  \"cases\": " + arr + ",\n  \"cases_sha256\": \"" + HexSha(arr) + "\"\n}\n";
	std::ofstream f(a.out, std::ios::binary);
	f << json;
	CHECK_MSG(f.good(), "could not write %s", a.out.c_str());
	std::printf("wrote %s: %zu cases, cases_sha256 %s\n", a.out.c_str(), cases.size(), HexSha(arr).c_str());
	return Finish("schema_fast_decode_gpu_p2_ref");
}
