// t2132_g5_gpu_parity.cpp -- G5-5 (T-2132, Brunel): the executed CPU-vs-GPU parity proof for
// schema-constrained decode and jump-forward (design Claude/Vitruvius/
// t2119-g5-constrained-decoding-design-2026-08-16.md Sec6, Slot G5-5 -- REQUIRED 1.0 gate,
// D-SLM3443). Real D3D12 hardware, the real G5 fixture (tools/t2132_build_g5_fixture.py's own
// output: real Qwen2.5-1.5B-Instruct weights/tokenizer + a real, compiled SchemaMasks section,
// 594 states, design Sec9.10.1's own cited reference-compiler figure).
//
// This driver (main()) includes NEITHER sslm_abi.h NOR gpu_1p0.h -- see
// t2132_g5_gpu_parity_shared.h for why (a real global-scope enum collision between the two
// headers). It loads the artifact bytes once, calls RunCpuGates (t2132_g5_gpu_parity_cpu.cpp)
// then RunGpuGates (t2132_g5_gpu_parity_gpu.cpp) against the SAME bytes and the SAME schema, and
// compares:
//
//   Gate 1 (masked decode): the CPU and GPU produced-token streams, under the identical compiled
//       schema and prompt, are identical token-for-token for every decoded token, and their
//       SHA-256 digests (design Sec6 G5-5's own "hash-equal" framing) match.
//   Gate 2 (jump-forward): the first kForcedChainLen tokens Gate 1's CPU decode loop actually
//       produced ARE a real schema-legal forced chain from the prompt's own end state (every one
//       of them passed the mask when it was produced) -- fed to both paths' own schema-content
//       prefill entry point, against a FRESH sequence re-primed with the same prompt;
//       forced_consumed and the ONE decode step run immediately after the chain must match.
//
// Usage: t2132_g5_gpu_parity <g5-fixture.sslm> "<prompt>" [schema_name] [num_decode_steps]
// Exits 0 on pass (every check above holds), 1 on any failure or setup error.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/model.h"
#include "superslm/schema_masks.h"
#include "superslm/sha256.h"
#include "t2132_g5_gpu_parity_shared.h"

namespace {

bool ReadFile(const char* path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	const std::streamoff size = f.tellg();
	if (size < 0) return false;
	f.seekg(0, std::ios::beg);
	out->resize(static_cast<size_t>(size));
	if (size > 0) f.read(reinterpret_cast<char*>(out->data()), size);
	return static_cast<bool>(f) || f.eof();
}

std::string HexDigest(const uint8_t d[32]) {
	static const char* hex = "0123456789abcdef";
	std::string s(64, '0');
	for (int i = 0; i < 32; ++i) {
		s[2 * i] = hex[(d[i] >> 4) & 0xF];
		s[2 * i + 1] = hex[d[i] & 0xF];
	}
	return s;
}

std::string TokenStreamDigest(const std::vector<int32_t>& tokens) {
	superslm::Sha256 h;
	for (int32_t t : tokens) {
		uint8_t le[4] = {static_cast<uint8_t>(t), static_cast<uint8_t>(t >> 8),
		                  static_cast<uint8_t>(t >> 16), static_cast<uint8_t>(t >> 24)};
		h.Update(le, 4);
	}
	uint8_t digest[32];
	h.Final(digest);
	return HexDigest(digest);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr,
		              "usage: %s <path-to-g5-fixture.sslm> \"<prompt>\" [schema_name] "
		              "[num_decode_steps]\n",
		              argv[0]);
		return 1;
	}
	const std::string artifact_path = argv[1];
	const std::string prompt = argv[2];
	const std::string schema_name = argc >= 4 ? argv[3] : "shopkeeper_intent_extraction";
	const int32_t kNumDecodeSteps = argc >= 5 ? std::atoi(argv[4]) : 24;
	const int32_t kForcedChainLen = 6;  // must be < kNumDecodeSteps -- Gate 2's own chain length.

	std::vector<uint8_t> bytes;
	if (!ReadFile(artifact_path.c_str(), &bytes)) {
		std::fprintf(stderr, "could not read %s\n", artifact_path.c_str());
		return 1;
	}

	std::printf("=== T-2132 G5-5 GPU PARITY ===\n");
	std::printf("artifact: %s\nprompt: \"%s\"\nschema: %s\ndecode_steps: %d, forced_chain_len: %d\n\n",
	            artifact_path.c_str(), prompt.c_str(), schema_name.c_str(), kNumDecodeSteps,
	            kForcedChainLen);

	std::vector<int32_t> prompt_tokens;
	const G5ParityPathResult cpu = RunCpuGates(bytes.data(), bytes.size(), prompt, schema_name,
	                                            kNumDecodeSteps, kForcedChainLen, &prompt_tokens);
	std::printf("[CPU] setup_ok=%s checks=%d failures=%d%s\n", cpu.setup_ok ? "true" : "false",
	            cpu.checks, cpu.failures,
	            cpu.last_error.empty() ? "" : (" last_error=" + cpu.last_error).c_str());
	if (!cpu.setup_ok || cpu.failures > 0) {
		std::fprintf(stderr, "CPU path failed its own internal checks -- aborting before the GPU "
		                      "path runs (nothing to compare against).\n");
		return 1;
	}

	const std::vector<int32_t> forced_chain(cpu.decoded_tokens.begin(),
	                                         cpu.decoded_tokens.begin() +
	                                             (static_cast<int32_t>(cpu.decoded_tokens.size()) >=
	                                                      kForcedChainLen
	                                                  ? kForcedChainLen
	                                                  : 0));
	const G5ParityPathResult gpu = RunGpuGates(bytes.data(), bytes.size(), prompt_tokens, schema_name,
	                                            kNumDecodeSteps, forced_chain);
	std::printf("[GPU] setup_ok=%s checks=%d failures=%d%s\n", gpu.setup_ok ? "true" : "false",
	            gpu.checks, gpu.failures,
	            gpu.last_error.empty() ? "" : (" last_error=" + gpu.last_error).c_str());
	if (!gpu.setup_ok) {
		std::fprintf(stderr, "GPU path failed to set up -- see last_error above.\n");
		return 1;
	}

	int total_checks = 0, total_failures = 0;
	auto Check = [&](bool cond, const char* msg) {
		++total_checks;
		if (!cond) {
			++total_failures;
			std::fprintf(stderr, "FAIL: %s\n", msg);
		}
	};

	// ---- DIAGNOSTIC (not a gate): independently replay both token streams through the SAME
	// schema table (parsed fresh, here, from the artifact bytes -- schema_masks.h is collision-
	// free, no sslm_abi.h/gpu_1p0.h involved) to localize a divergence to either "the two paths
	// disagreed on which state they were in" (a walk-state bug) or "the two paths agreed on
	// state but picked different tokens" (a hidden-state/logits numeric divergence). ----
	{
		superslm::SslmModelView diag_view;
		std::string diag_err;
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), diag_view, &diag_err) ==
		    superslm::SslmModelStatus::Ok) {
			const superslm::SslmSectionView* sec =
			    diag_view.Section(superslm::SslmSectionType::SchemaMasks);
			if (sec) {
				superslm::SchemaMasksTable table;
				std::string perr;
				if (superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size,
				                                       diag_view.config.vocab_size, table, &perr)) {
					size_t idx = 0;
					const superslm::SchemaEntry* entry = table.ByName(schema_name, &idx);
					if (entry) {
						auto Replay = [&](const std::vector<int32_t>& toks) {
							std::vector<uint32_t> states;
							uint32_t state = 0;
							for (int32_t t : toks) {
								states.push_back(state);
								uint32_t next = state;
								table.Transition(*entry, state, static_cast<uint32_t>(t), &next);
								state = next;
							}
							return states;
						};
						const std::vector<uint32_t> cpu_states = Replay(cpu.decoded_tokens);
						const std::vector<uint32_t> gpu_states = Replay(gpu.decoded_tokens);
						std::printf("\n[diag] walk-state replay (state BEFORE each token, both "
						            "paths' own produced tokens fed through the SAME schema table):\n");
						const size_t n = std::min(cpu_states.size(), gpu_states.size());
						for (size_t i = 0; i < n; ++i) {
							const bool state_match = cpu_states[i] == gpu_states[i];
							const bool tok_match =
							    cpu.decoded_tokens[i] == gpu.decoded_tokens[i];
							std::printf(
							    "  step=%2zu cpu_state=%3u gpu_state=%3u %s  cpu_tok=%-6d gpu_tok=%-6d %s\n",
							    i, cpu_states[i], gpu_states[i], state_match ? "SAME " : "DIFF ",
							    cpu.decoded_tokens[i], gpu.decoded_tokens[i],
							    tok_match ? "SAME" : "DIFF");
						}
					}
				}
			}
		}
	}

	// ---- Gate 1: token-for-token + SHA-256 digest of the produced-token stream. ----
	Check(gpu.failures == 0, "Gate 1/2: GPU path reported internal check failures");
	Check(cpu.decoded_tokens.size() == gpu.decoded_tokens.size(),
	      "Gate 1: CPU/GPU decoded-token-count mismatch");
	const size_t common_n = std::min(cpu.decoded_tokens.size(), gpu.decoded_tokens.size());
	bool all_tokens_match = true;
	for (size_t i = 0; i < common_n; ++i) {
		if (cpu.decoded_tokens[i] != gpu.decoded_tokens[i]) {
			all_tokens_match = false;
			std::fprintf(stderr, "  step=%zu cpu_token=%d gpu_token=%d\n", i, cpu.decoded_tokens[i],
			              gpu.decoded_tokens[i]);
		}
	}
	Check(all_tokens_match, "Gate 1: CPU/GPU produced token diverged at one or more steps");

	const std::string cpu_digest = TokenStreamDigest(cpu.decoded_tokens);
	const std::string gpu_digest = TokenStreamDigest(gpu.decoded_tokens);
	Check(cpu_digest == gpu_digest, "Gate 1: CPU/GPU produced-token-stream SHA-256 digests diverged");
	std::printf("\n[Gate 1] masked decode x%d: CPU digest=%s GPU digest=%s %s\n", kNumDecodeSteps,
	            cpu_digest.c_str(), gpu_digest.c_str(), cpu_digest == gpu_digest ? "MATCH" : "DIVERGE");
	std::printf("[Gate 1] CPU tokens: ");
	for (int32_t t : cpu.decoded_tokens) std::printf("%d ", t);
	std::printf("\n[Gate 1] GPU tokens: ");
	for (int32_t t : gpu.decoded_tokens) std::printf("%d ", t);
	std::printf("\n");

	// ---- Gate 2: forced_consumed + the one post-chain decode step. ----
	Check(cpu.forced_consumed == gpu.forced_consumed,
	      "Gate 2: CPU/GPU forced-chain consumed counts diverged");
	Check(cpu.post_forced_token == gpu.post_forced_token,
	      "Gate 2: CPU/GPU post-forced-chain token diverged");
	std::printf(
	    "\n[Gate 2] jump-forward x%d forced tokens: CPU consumed=%d GPU consumed=%d, "
	    "post-chain token CPU=%d GPU=%d %s\n",
	    kForcedChainLen, cpu.forced_consumed, gpu.forced_consumed, cpu.post_forced_token,
	    gpu.post_forced_token,
	    (cpu.forced_consumed == gpu.forced_consumed && cpu.post_forced_token == gpu.post_forced_token)
	        ? "MATCH"
	        : "DIVERGE");

	// ---- Gate 3 (S6, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): CPU/GPU ready_for_logits
	// parity on the PARTIAL-REJECTION path. Reuses Gate 2's own real, schema-legal forced_chain
	// and appends ONE token that is not a legal transition from the state that chain leaves the
	// walk in -- found by re-parsing the same SchemaMasks section (schema_masks.h, collision-free
	// against both sslm_abi.h and gpu_1p0.h) and scanning the reached state's own mask page for
	// the first CLEAR bit (an illegal token by construction: the loader's own mask/transition
	// cross-check, Sec13.3, guarantees clear-bit <=> no CSR row entry). ----
	int32_t illegal_token = -1;
	{
		superslm::SslmModelView diag_view;
		std::string diag_err;
		if (!forced_chain.empty() &&
		    superslm::SslmModel::Load(bytes.data(), bytes.size(), diag_view, &diag_err) ==
		        superslm::SslmModelStatus::Ok) {
			const superslm::SslmSectionView* sec =
			    diag_view.Section(superslm::SslmSectionType::SchemaMasks);
			if (sec) {
				superslm::SchemaMasksTable table;
				std::string perr;
				if (superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size,
				                                       diag_view.config.vocab_size, table, &perr)) {
					size_t idx = 0;
					const superslm::SchemaEntry* entry = table.ByName(schema_name, &idx);
					if (entry) {
						uint32_t state = 0;
						for (int32_t t : forced_chain) {
							uint32_t next = state;
							table.Transition(*entry, state, static_cast<uint32_t>(t), &next);
							state = next;
						}
						const uint8_t* page = entry->mask_pages +
						                       static_cast<size_t>(state) * table.MaskPageBytes();
						const int32_t vocab = static_cast<int32_t>(diag_view.config.vocab_size);
						for (int32_t t = 0; t < vocab; ++t) {
							if (!((page[t >> 3] >> (t & 7)) & 1u)) {
								illegal_token = t;
								break;
							}
						}
					}
				}
			}
		}
	}

	if (illegal_token < 0) {
		std::fprintf(stderr,
		             "\n[Gate 3] SKIPPED -- could not find a clear mask bit (illegal token) at "
		             "the forced chain's own end state; the reference schema's own reachable "
		             "states may all be fully dense at this depth. Not a Gate 3 failure -- no "
		             "input to run it against.\n");
	} else {
		std::vector<int32_t> chain_with_illegal_tail = forced_chain;
		chain_with_illegal_tail.push_back(illegal_token);

		const G5Gate3Result cpu3 =
		    RunCpuGate3(bytes.data(), bytes.size(), prompt_tokens, schema_name, chain_with_illegal_tail);
		const G5Gate3Result gpu3 =
		    RunGpuGate3(bytes.data(), bytes.size(), prompt_tokens, schema_name, chain_with_illegal_tail);

		Check(cpu3.setup_ok && cpu3.failures == 0, "Gate 3: CPU path reported internal check failures");
		Check(gpu3.setup_ok && gpu3.failures == 0, "Gate 3: GPU path reported internal check failures");
		Check(cpu3.rejected_as_expected, "Gate 3: CPU path did not partially reject as expected");
		Check(gpu3.rejected_as_expected, "Gate 3: GPU path did not partially reject as expected");
		Check(cpu3.forced_consumed == gpu3.forced_consumed,
		      "Gate 3: CPU/GPU partial-rejection consumed counts diverged");
		Check(cpu3.post_reject_token == gpu3.post_reject_token,
		      "Gate 3: CPU/GPU post-partial-rejection token diverged (the S6 ready_for_logits "
		      "parity bug this gate exists to catch)");
		std::printf(
		    "\n[Gate 3] partial-rejection ready_for_logits parity: illegal_token=%d, CPU "
		    "consumed=%d/%zu rejected=%s post_token=%d; GPU consumed=%d/%zu rejected=%s "
		    "post_token=%d %s\n",
		    illegal_token, cpu3.forced_consumed, chain_with_illegal_tail.size(),
		    cpu3.rejected_as_expected ? "true" : "false", cpu3.post_reject_token, gpu3.forced_consumed,
		    chain_with_illegal_tail.size(), gpu3.rejected_as_expected ? "true" : "false",
		    gpu3.post_reject_token,
		    (cpu3.forced_consumed == gpu3.forced_consumed &&
		     cpu3.post_reject_token == gpu3.post_reject_token && cpu3.rejected_as_expected &&
		     gpu3.rejected_as_expected)
		        ? "MATCH"
		        : "DIVERGE");
	}

	std::printf("\n=== T-2132 G5-5 GPU PARITY: %s (checks=%d failures=%d, CPU checks=%d/%d, "
	            "GPU checks=%d/%d) ===\n",
	            total_failures ? "FAIL" : "PASS", total_checks, total_failures, cpu.checks,
	            cpu.failures, gpu.checks, gpu.failures);
	return total_failures ? 1 : 0;
}
