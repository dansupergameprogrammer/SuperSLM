// damped_greedy_phaseD_loop.cpp -- T-2199 Phase D3: RunGreedyOrDampedGreedyDecodeLoop, the
// free-text CLI/RunGreedyDecodeLoop entry point's own damped-greedy-aware sibling (plan Sec8
// D3, forward_sites.cpp's own RunGreedyDecodeLoop). A new, parallel function rather than an
// edit to RunGreedyDecodeLoop itself -- zero risk to that already-shipped, heavily-tested
// entry point (Poirot-style no-regression discipline); both call the SAME certified per-token
// primitives (EmbedEntry, RunLayerLoop, RmsNormSite, LogitsSite, ArgmaxLowestIndexTieBreak) and
// the SAME shared decode-step primitive DampedGreedyScoreAndArgmax that sslm_decode_stepImpl
// (Phase D2, src/sslm_abi.cpp) calls -- "one implementation, never two" applies to the
// scoring primitive, not to the surrounding per-entry-point loop scaffolding, matching this
// codebase's own existing precedent (RunGreedyDecodeLoop and sslm_decode_stepImpl already have
// separately-written, structurally-parallel loop bodies calling the same site functions).
#include "superslm/sslm_phaseD.h"

#include <cstring>
#include <vector>

#include "superslm/sslm_damped_greedy.h"

// T-2199 Phase D review fix S5: relocated into namespace superslm (see sslm_phaseD.h's own top
// comment for the full reasoning and the suite-link argument).
namespace superslm {

SslmForwardStatus RunGreedyOrDampedGreedyDecodeLoop(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    const int32_t* prompt_tokens, size_t prompt_len, const int8_t* embed_weights,
    superslm::CarriedScale embed_site_constant, const int32_t* final_norm_gain,
    superslm::CarriedScale final_norm_site_constant, const int8_t* head_weights, int32_t vocab_size,
    const int32_t* stop_ids, size_t stop_count, size_t max_new_tokens, uint8_t* workspace,
    size_t workspace_size, int32_t* out_tokens, int32_t* out_logit_rows, size_t out_tokens_capacity,
    size_t* out_tokens_produced, superslm::SslmDecodeStopReason* out_stop_reason,
    superslm::SslmKvPrecision kv_precision, bool option_g_fused_k_landing, DampedGreedyMode mode,
    int32_t alpha_q15, int32_t anti_lm_max_order, int32_t top_k, int64_t q_ln2, int64_t q_b,
    int64_t q_c) {
	using superslm::SslmForwardStatus;

	// T-2199 closing round, production bug found by the suite owner's strengthened N8 symmetry
	// cell (Claude/Curie/t2199-phaseD-red-2026-08-20.md's closing-confirmation round): `mode` was
	// validated only INSIDE `if (mode == kDampedGreedy)`, below -- an unknown mode value (neither
	// kGreedy nor kDampedGreedy) skipped validation entirely and fell through to the real forward
	// path, unlike sslm_decode_stepImpl's own D2 entry point (S2's fix), which validates `mode`
	// UNCONDITIONALLY before anything else runs. Moved here, to the top of this function, before
	// any branch -- same shared validator, same status, same symmetry S3/N8 already established
	// for the DAMPED-GREEDY-mode validation failure case, now covering the unknown-mode case too.
	// `DampedGreedyValidationParams` doubles as "just the mode" when the other fields are
	// zero-valued and mode is neither real value -- ValidateDampedGreedyParams's own FIRST check
	// (mode in {kGreedy, kDampedGreedy}) rejects before it ever reads alpha_q15/anti_lm_max_order/
	// top_k, so passing zeros for those here is safe regardless of what the caller actually set.
	{
		const DampedGreedyValidationParams vp{mode, alpha_q15, anti_lm_max_order, top_k};
		if (!ValidateDampedGreedyParams(vp, vocab_size)) {
			// T-2199 Phase D review fix S3: InvalidDecodeParams, not TokenIdOutOfRange -- gives
			// this entry point's rejection the SAME outcome as sslm_decode_stepImpl's identical
			// invalid-parameter check (both map to SSLM_INVALID_ARGUMENT, see MapForwardStatus,
			// sslm_abi.cpp), and names the actual cause instead of borrowing an unrelated status.
			return SslmForwardStatus::InvalidDecodeParams;
		}
	}

	if (kv_precision == superslm::SslmKvPrecision::Int16) {
		return SslmForwardStatus::KvPrecisionUnsupported;
	}
	if (seq.hidden_codes == nullptr) {
		return SslmForwardStatus::InvalidHiddenCodes;
	}
	(void)out_tokens_capacity;

	for (size_t i = 0; i < stop_count; ++i) {
		if (stop_ids[i] < 0 || stop_ids[i] >= vocab_size) return SslmForwardStatus::TokenIdOutOfRange;
	}
	for (size_t i = 0; i < prompt_len; ++i) {
		if (prompt_tokens[i] < 0 || prompt_tokens[i] >= vocab_size) {
			return SslmForwardStatus::TokenIdOutOfRange;
		}
	}

	std::vector<int8_t> embed_codes(hidden_size);
	std::vector<int8_t> final_codes(hidden_size);
	const size_t vocab_size_z = static_cast<size_t>(vocab_size);
	std::vector<int64_t> wide_logits(vocab_size_z);
	std::vector<int32_t> logit_row(vocab_size_z);
	superslm::CarriedScale embed_scale{};

	auto RunWholeToken = [&](int32_t token) -> SslmForwardStatus {
		const SslmForwardStatus est = superslm::EmbedEntry(token, vocab_size, embed_weights,
		                                                    hidden_size, embed_site_constant,
		                                                    embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = embed_codes[i];
		seq.hidden_scale = embed_scale;
		seq.layer_index = 0;
		return superslm::RunLayerLoop(
		    seq, layers, num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
		    head_dim, num_key_value_heads, intermediate_size, context_cap, rope_tables, workspace,
		    workspace_size,
		    option_g_fused_k_landing ? superslm::OptionGKLandingMode::kFused
		                              : superslm::OptionGKLandingMode::kLegacy);
	};

	for (size_t i = 0; i + 1 < prompt_len; ++i) {
		const SslmForwardStatus st = RunWholeToken(prompt_tokens[i]);
		if (st != SslmForwardStatus::Ok) return st;
	}

	std::vector<int32_t> produced_tokens;
	std::vector<int32_t> produced_logit_rows;
	produced_tokens.reserve(max_new_tokens);
	produced_logit_rows.reserve(max_new_tokens * vocab_size_z);

	int32_t current_token = prompt_tokens[prompt_len - 1];
	superslm::SslmDecodeStopReason stop_reason = superslm::SslmDecodeStopReason::MaxTokensReached;

	// Sec7.2: a warm-object anti-LM, scoped to this ONE call (one whole generation) -- created
	// only when damped-greedy mode is selected, destroyed on every return path below.
	superslm::AntiLmState* antilm = nullptr;
	if (mode == DampedGreedyMode::kDampedGreedy) {
		antilm = superslm::AntiLmCreate(anti_lm_max_order);
		// AntiLmCreate returns nullptr only for anti_lm_max_order < 1 (AntiLmState.h), already
		// excluded by ValidateDampedGreedyParams above -- unreachable in practice; mapped to the
		// same "no dedicated status" domain-rejection convention as the !ok check below rather
		// than asserting, since a future relaxation of that precondition should fail defined,
		// not crash.
		if (!antilm) return SslmForwardStatus::InvalidDecodeParams;
	}
	auto DestroyAntilm = [&] {
		if (antilm) superslm::AntiLmDestroy(antilm);
	};

	// Sec6: free-text mode's mask is all-ones by construction -- built once, reused every step
	// (this loop has no schema/masking of its own, matching RunGreedyDecodeLoop's own free-text-
	// only scope).
	std::vector<uint8_t> free_text_mask;
	if (mode == DampedGreedyMode::kDampedGreedy) {
		free_text_mask.assign((vocab_size_z + 7) / 8, 0xFF);
	}

	while (produced_tokens.size() < max_new_tokens) {
		SslmForwardStatus st = RunWholeToken(current_token);
		if (st != SslmForwardStatus::Ok) {
			DestroyAntilm();
			return st;
		}

		superslm::CarriedScale final_scale{};
		st = superslm::RmsNormSite(seq.hidden_codes, final_norm_gain, hidden_size, seq.hidden_scale,
		                           final_norm_site_constant, final_codes.data(), &final_scale,
		                           "final_norm");
		if (st != SslmForwardStatus::Ok) {
			DestroyAntilm();
			return st;
		}

		st = superslm::LogitsSite(final_codes.data(), hidden_size, head_weights, vocab_size,
		                          wide_logits.data(), logit_row.data());
		if (st != SslmForwardStatus::Ok) {
			DestroyAntilm();
			return st;
		}

		int32_t token;
		if (mode == DampedGreedyMode::kDampedGreedy) {
			int32_t produced_dg = -1;
			bool refused = false;
			const bool ok = superslm::DampedGreedyScoreAndArgmax(
			    logit_row.data(), free_text_mask.data(), vocab_size, top_k, antilm, alpha_q15,
			    q_ln2, q_b, q_c, &produced_dg, &refused);
			if (!ok) {
				DestroyAntilm();
				return SslmForwardStatus::InvalidDecodeParams;
			}
			if (refused) {
				DestroyAntilm();
				return SslmForwardStatus::SoftmaxKernelRefusedAfterGateAccepted;
			}
			superslm::AntiLmUpdate(antilm, produced_dg);
			token = produced_dg;
		} else {
			token = superslm::ArgmaxLowestIndexTieBreak(logit_row.data(), vocab_size_z);
		}

		produced_tokens.push_back(token);
		produced_logit_rows.insert(produced_logit_rows.end(), logit_row.begin(), logit_row.end());

		bool matched = false;
		for (size_t i = 0; i < stop_count; ++i) {
			if (stop_ids[i] == token) {
				matched = true;
				break;
			}
		}
		if (matched) {
			stop_reason = superslm::SslmDecodeStopReason::StopTokenMatched;
			break;
		}
		if (produced_tokens.size() >= max_new_tokens) {
			stop_reason = superslm::SslmDecodeStopReason::MaxTokensReached;
			break;
		}
		current_token = token;
	}

	DestroyAntilm();

	for (size_t i = 0; i < produced_tokens.size(); ++i) out_tokens[i] = produced_tokens[i];
	for (size_t i = 0; i < produced_logit_rows.size(); ++i) {
		out_logit_rows[i] = produced_logit_rows[i];
	}
	*out_tokens_produced = produced_tokens.size();
	*out_stop_reason = stop_reason;
	return SslmForwardStatus::Ok;
}

}  // namespace superslm
