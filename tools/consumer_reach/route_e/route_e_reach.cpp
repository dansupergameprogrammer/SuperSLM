// route_e_reach.cpp -- the seed of cell 10.0's route E leg (plan rev 5, K1 closure).
// Drives SuperEmbedder's own encode path (semb_interim_encode_ps_n: seam on the chunk-batched
// leg at chunk_budget 0 = whole document, then PS-N select and dequantize) with TOKEN IDS, on a
// synthetic 0.6B-width artifact. This is "the embedder's Encode shape, driven by ids", NOT the
// `semb query` CLI path (the synthetic artifact's vocab is 256; the Qwen tokenizer refuses it).
// It reads the tiled-entry counter from the engine's test seam, which exists only when the
// engine was built with the seam (plan §11.R legs i/ii). The header is included from the
// CANDIDATE source (the reference), never from the leg's own clone.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "superembedder/encoder_interim_abi.h"
#include "support/matmul_dispatch_instrument.h"

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s model.sslm\n", argv[0]); return 2; }
  SembInterimEncoder* enc = nullptr;
  const char* st = nullptr;
  if (semb_interim_encoder_open(argv[1], /*appended_token_id=*/0, &enc, &st) != kSembInterimOk) {
    std::printf("open: refused (%s)\n", st ? st : "?");
    return 3;
  }
  SembInterimGeometry g{};
  g.struct_size = sizeof g;
  semb_interim_encoder_geometry(enc, &g);
  std::printf("open: ok hidden=%u layers=%u vocab=%u context_cap=%u\n", g.hidden_size, g.num_hidden_layers,
              g.vocab_size, g.context_cap);
  int rc = 0;
  for (size_t n_content : {size_t{27}, size_t{275}}) {  // + the appended terminator = 28 and 276 (G40)
    std::vector<int32_t> ids(n_content);
    for (size_t i = 0; i < n_content; ++i) ids[i] = static_cast<int32_t>((i * 7 + 3) % g.vocab_size);
    std::vector<float> v(g.hidden_size);
    size_t req = 0;
    SembWaPoolStatus dq{};
    superslm_test::g_tiled_entry_invocations.store(0);
    const SembInterimStatus s = semb_interim_encode_ps_n(enc, ids.data(), n_content, /*chunk_budget=*/0, v.data(),
                                                         v.size() * sizeof(float), &req, &dq, &st);
    const long long te = superslm_test::g_tiled_entry_invocations.load();
    std::printf("encode: tokens_reaching_gemm=%zu status=%d tiled_entries=%lld\n", n_content + 1, int(s), te);
    if (s != kSembInterimOk && s != kSembInterimDequantizeRefused) rc = 4;
  }
  semb_interim_encoder_close(enc);
  return rc;
}
