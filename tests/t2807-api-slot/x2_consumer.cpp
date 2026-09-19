// T-2814, T-2825 (Curie) -- X2's consumer DLL (plan Sec3.7 item 5, TE-266; D-SLM7313, widened to the whole
// C ABI by the T-2823 fold). Compiled with /DSUPERSLM_API=__declspec(dllimport) and linked against
// sslm_engine.dll's import library ONLY, so every engine symbol it calls must cross the DLL boundary. It calls
// every symbol of X2's expected set at least once and nothing else from the engine -- the 25 C++ entries of
// consumed_symbols.txt and the 36 C verbs of c_abi_verbs.txt, 61 at v1.5.0: build_x2.ps1 checks that this
// DLL's import table from sslm_engine.dll EQUALS that set, so a call silently dropped here fails X2 (the
// harness mutant, /DX2_DROP_CHUNK_BATCHED, proves it).
//
// Known answers, per C++ symbol (25):
//   Sha256::Reset / Update / Final, Sha256Hash    SHA-256("abc") both ways, equal to each other and to the
//                                                 FIPS 180-4 digest ba7816bf...f20015ad
//   SslmStatusName, SslmModelStatusName,          "Ok" for each enum's Ok
//   SslmForwardStatusName
//   SslmArtifact::OpenFromMemory                  null data -> SslmStatus::NullData (artifact.h: rejected
//                                                 explicitly before any other check)
//   SslmArtifact::Section                         an empty artifact has no sections -> nullptr
//   SslmModel::Load                               null data -> SslmModelStatus::ArtifactRejected
//   SslmTensorManifest::Tensor                    an empty manifest -> nullptr
//   SslmKeyedConstants::Entry                     empty constants -> nullptr
//   SslmKeyedConstants::Value                     a hand-built entry holding -123456789 as one LE int64
//   TokenizerView(), ~TokenizerView(),            construct two, move-assign one into the other, destroy;
//   operator=(TokenizerView&&)
//   TokenizerView::VocabSize                      an unopened view -> 0
//   TokenizerView::Encode                         an unopened view -> an empty vector
//   TokenizerView::Open                           an artifact with no tokenizer sections -> false
//   RmsNormSite                                   RMSNorm of the zero vector is the zero vector: Ok, all codes 0
//   EmbedEntry                                    token -1 -> TokenIdOutOfRange (checked at entry)
//   RunLayerLoop (both overloads)                 layer_budget 0 -> InvalidLayerBudget (checked at entry)
//   RunLayerLoopChunkBatched                      chunk_tokens 0 -> InvalidLayerBudget (checked at entry)
//   ParseConfig                                   an empty section view -> BadConfigSize (the exact-size check
//                                                 is the first check, src/model.cpp ParseConfigImpl)
// The C verbs (36), each called with null handles and null out-pointers, which every verb's contract refuses
// at entry: the 30 that return sslm_status return SSLM_INVALID_ARGUMENT; the six sizing verbs
// (sslm_kv_block_size, sslm_kv_pool_overhead_size, sslm_seq_state_size, sslm_workspace_size,
// sslm_adapter_residency, sslm_schema_count) return 0 for a null model or adapter. Executed on all 36 by the
// planner's probe (T-2823 P2) and asserted here per verb.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "superslm/sslm_abi.h"
#include "superslm/tokenizer.h"

using namespace superslm;

static int g_fail = 0;
static int g_checks = 0;
#define X2_CHECK(cond, what)                                             \
	do {                                                                 \
		++g_checks;                                                      \
		if (!(cond)) {                                                   \
			++g_fail;                                                    \
			std::printf("  X2 FAIL %s: %s\n", what, #cond);              \
		} else {                                                         \
			std::printf("  X2 ok   %s\n", what);                         \
		}                                                                \
	} while (0)

extern "C" __declspec(dllexport) int x2_run() {
	g_fail = 0;
	g_checks = 0;
	static const uint8_t kAbc[32] = {0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
	                                 0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
	                                 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
	const uint8_t* abc = reinterpret_cast<const uint8_t*>("abc");
	uint8_t d1[32] = {}, d2[32] = {};
	Sha256Hash(abc, 3, d1);
	Sha256 h;  // the inline constructor calls Reset()
	h.Reset();
	h.Update(abc, 3);
	h.Final(d2);
	X2_CHECK(std::memcmp(d1, kAbc, 32) == 0, "Sha256Hash(\"abc\") == FIPS 180-4 digest");
	X2_CHECK(std::memcmp(d2, kAbc, 32) == 0, "Sha256::Reset/Update/Final(\"abc\") == FIPS 180-4 digest");

	X2_CHECK(std::strcmp(SslmStatusName(SslmStatus::Ok), "Ok") == 0, "SslmStatusName(Ok)");
	X2_CHECK(std::strcmp(SslmModelStatusName(SslmModelStatus::Ok), "Ok") == 0, "SslmModelStatusName(Ok)");
	X2_CHECK(std::strcmp(SslmForwardStatusName(SslmForwardStatus::Ok), "Ok") == 0, "SslmForwardStatusName(Ok)");

	SslmArtifact art;
	SslmError aerr{};
	X2_CHECK(SslmArtifact::OpenFromMemory(nullptr, 0, art, &aerr) == SslmStatus::NullData,
	         "SslmArtifact::OpenFromMemory(null) == NullData");
	X2_CHECK(art.Section(SslmSectionType::SchemaMasks) == nullptr, "SslmArtifact::Section on an empty artifact");

	{
		SslmModelView view{};
		std::string err;
		X2_CHECK(SslmModel::Load(nullptr, 0, view, &err) == SslmModelStatus::ArtifactRejected,
		         "SslmModel::Load(null) == ArtifactRejected");
	}
	{
		SslmTensorManifest mf;
		X2_CHECK(mf.Tensor("embed") == nullptr, "SslmTensorManifest::Tensor on an empty manifest");
		SslmKeyedConstants kc;
		X2_CHECK(kc.Entry("embed") == nullptr, "SslmKeyedConstants::Entry on empty constants");
		const int64_t want = -123456789;
		uint8_t le[8];
		for (int i = 0; i < 8; ++i) le[i] = static_cast<uint8_t>(static_cast<uint64_t>(want) >> (8 * i));
		SslmConstantEntry e;
		e.name = "x";
		e.values = le;
		e.value_words = 1;
		X2_CHECK(SslmKeyedConstants::Value(e, 0) == want, "SslmKeyedConstants::Value reads one LE int64");
	}
	{
		TokenizerView a;
		TokenizerView b;
		a = std::move(b);
		X2_CHECK(a.VocabSize() == 0, "TokenizerView ctor/move-assign; VocabSize of an unopened view == 0");
		X2_CHECK(a.Encode("abc").empty(), "TokenizerView::Encode on an unopened view is empty");
		std::string err;
		X2_CHECK(!TokenizerView::Open(art, a, &err), "TokenizerView::Open on an artifact with no tokenizer");
	}  // ~TokenizerView x2

	{
		const int8_t zero[4] = {0, 0, 0, 0};
		const int32_t gain[4] = {1, 1, 1, 1};
		int8_t out[4] = {7, 7, 7, 7};
		CarriedScale os{};
		const SslmForwardStatus st =
		    RmsNormSite(zero, gain, 4, CarriedScale{}, CarriedScale{INT64_C(1) << 30, -30}, out, &os, "x2");
		X2_CHECK(st == SslmForwardStatus::Ok && out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0,
		         "RmsNormSite(zero vector) == Ok, all codes 0");
	}
	{
		int8_t codes[4] = {};
		CarriedScale s{};
		X2_CHECK(EmbedEntry(-1, 4, codes, 4, CarriedScale{INT64_C(1) << 30, -30}, codes, &s) ==
		             SslmForwardStatus::TokenIdOutOfRange,
		         "EmbedEntry(token -1) == TokenIdOutOfRange");
	}
	{
		int8_t codes[4] = {};
		SequenceLayerState seq{};
		seq.hidden_codes = codes;
		SslmTensorManifest rope;
		uint8_t ws[16] = {};
		X2_CHECK(RunLayerLoop(seq, nullptr, 1, 0, 4, 4, 1, 4, 8, rope, ws, sizeof ws) ==
		             SslmForwardStatus::InvalidLayerBudget,
		         "RunLayerLoop(layer_budget 0) == InvalidLayerBudget");
		X2_CHECK(RunLayerLoop(seq, nullptr, 1, 0, 4, 4, 1, 4, 8, rope, ws, sizeof ws, OptionGKLandingMode::kLegacy) ==
		             SslmForwardStatus::InvalidLayerBudget,
		         "RunLayerLoop(OptionGKLandingMode, layer_budget 0) == InvalidLayerBudget");
#ifndef X2_DROP_CHUNK_BATCHED
		CarriedScale scales[1] = {};
		uint64_t sat = 0;
		X2_CHECK(RunLayerLoopChunkBatched(codes, scales, 0, nullptr, 1, 4, 4, 1, 4, 8, 0, rope, ws, sizeof ws, false,
		                                  &sat) == SslmForwardStatus::InvalidLayerBudget,
		         "RunLayerLoopChunkBatched(chunk_tokens 0) == InvalidLayerBudget");
#endif
	}
	{
		SslmSectionView view{};
		SslmModelConfig mc{};
		std::string err;
		X2_CHECK(ParseConfig(view, mc, &err) == SslmModelStatus::BadConfigSize,
		         "ParseConfig(empty section view) == BadConfigSize");
	}

	// The C ABI, every verb (T-2823).
	{
		sslm_model m = nullptr;
		sslm_seq s = nullptr;
		sslm_prefix p = nullptr;
		sslm_adapter a = nullptr;
		sslm_kv_pool kp = nullptr;
		sslm_workspace ws = nullptr;
		sslm_schema sc = nullptr;
		sslm_config cfg{};
		sslm_detok_state ds{};
		sslm_decode_params dp{};
		sslm_stats_out so{};
		int32_t n = 0;
		size_t sz = 0;
		char buf[8] = {};
		int32_t tok[4] = {0, 0, 0, 0};
#define X2_INVALID(call) X2_CHECK((call) == SSLM_INVALID_ARGUMENT, #call " == SSLM_INVALID_ARGUMENT")
#define X2_ZERO(call) X2_CHECK((call) == 0, #call " == 0")
		X2_INVALID(sslm_model_map(nullptr, 0, nullptr));
		X2_INVALID(sslm_model_unmap(m));
		X2_ZERO(sslm_kv_block_size(m));
		X2_ZERO(sslm_kv_pool_overhead_size(m, 1));
		X2_ZERO(sslm_seq_state_size(m));
		X2_INVALID(sslm_kv_pool_create(m, nullptr, 0, 1, nullptr));
		X2_INVALID(sslm_kv_pool_destroy(kp));
		X2_INVALID(sslm_workspace_destroy(ws));
		X2_INVALID(sslm_prefix_begin(m, nullptr, nullptr));
		X2_INVALID(sslm_prefix_prefill(m, p, tok, 1, 1, SSLM_SPAN_PROMPT, ws, &n));
		X2_INVALID(sslm_prefix_freeze(p));
		X2_INVALID(sslm_prefix_release(p));
		X2_INVALID(sslm_seq_create(m, nullptr, nullptr));
		X2_INVALID(sslm_seq_release(s));
		X2_INVALID(sslm_seq_reset(s));
		X2_INVALID(sslm_seq_adopt_prefix(s, p));
		X2_INVALID(sslm_seq_save(s, nullptr, &sz));
		X2_INVALID(sslm_seq_restore(m, nullptr, nullptr, 0, nullptr));
		X2_INVALID(sslm_seq_set_adapter(s, a));
		X2_INVALID(sslm_adapter_map(nullptr, 0, m, nullptr));
		X2_INVALID(sslm_adapter_release(a));
		X2_ZERO(sslm_adapter_residency(a));
		X2_INVALID(sslm_prefill(m, s, tok, 1, 1, SSLM_SPAN_PROMPT, ws, &n));
		X2_INVALID(sslm_decode_step(m, &s, 1, &dp, ws, tok));
		X2_INVALID(sslm_tokenize(m, "a", tok, &n));
		X2_INVALID(sslm_stats(m, s, &so));
		X2_INVALID(sslm_schema_lookup(m, "x", &sc));
		X2_ZERO(sslm_schema_count(m));
		sz = sizeof buf;
		X2_INVALID(sslm_schema_name(m, 0, buf, &sz));
		X2_INVALID(sslm_seq_set_schema(s, sc));
		X2_INVALID(sslm_prefix_set_schema(p, sc));
		X2_INVALID(sslm_decode_step_v2(m, &s, 1, &dp, ws, tok));
		X2_INVALID(sslm_decode_params_init(m, 0, 1, &dp));
		X2_ZERO(sslm_workspace_size(m, &cfg));
		X2_INVALID(sslm_workspace_create(m, &cfg, nullptr, 0, &ws));
		X2_INVALID(sslm_detokenize_stream(m, &ds, tok, 1, buf, &n));
#undef X2_INVALID
#undef X2_ZERO
	}
	std::printf("X2 consumer: checks=%d failures=%d\n", g_checks, g_fail);
	return g_fail;
}
