#include "superslm/model.h"
#include "superslm/sslm_abi.h"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {
bool ReadFile(const char* path, std::vector<uint8_t>* bytes) {
    FILE* file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || !file) return false;
    std::fseek(file, 0, SEEK_END); const long n = std::ftell(file); std::fseek(file, 0, SEEK_SET);
    bytes->resize(n > 0 ? static_cast<size_t>(n) : 0);
    const bool ok = n >= 0 && (n == 0 || std::fread(bytes->data(), 1, bytes->size(), file) == bytes->size());
    std::fclose(file); return ok;
}
uint32_t Le32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint32_t Layer(sslm_seq seq) {
    size_t n = 0; if (sslm_seq_save(seq, nullptr, &n) != SSLM_BUFFER_TOO_SMALL) return UINT32_MAX;
    std::vector<uint8_t> blob(n); if (sslm_seq_save(seq, blob.data(), &n) != SSLM_OK || n < 72) return UINT32_MAX;
    return Le32(blob.data() + 68);
}
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const uint32_t target = static_cast<uint32_t>(std::stoul(argv[2]));
    std::vector<uint8_t> bytes; if (!ReadFile(argv[1], &bytes)) return 2;
    superslm::SslmModelView view; std::string error;
    if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &error) !=
        superslm::SslmModelStatus::Ok) return 2;
    sslm_model model = nullptr; if (sslm_model_map(bytes.data(), bytes.size(), &model) != SSLM_OK) return 2;
    const size_t block = sslm_kv_block_size(model); const size_t overhead = sslm_kv_pool_overhead_size(model, 2);
    std::vector<uint8_t> storage(block * 2 + overhead + 63); void* aligned = storage.data(); size_t space = storage.size();
    if (!std::align(64, block * 2 + overhead, aligned, space)) return 2;
    sslm_kv_pool pool = nullptr; if (sslm_kv_pool_create(model, aligned, block * 2 + overhead, 2, &pool) != SSLM_OK) return 2;
    bool held = false; int32_t chosen = -1; sslm_status reset_status = SSLM_INVALID_ARGUMENT;
    for (int32_t token = 0; token < static_cast<int32_t>(view.config.vocab_size) && !held; ++token) {
        sslm_seq seq = nullptr; if (sslm_seq_create(model, &pool, &seq) != SSLM_OK) return 2;
        int32_t consumed = 0; if (sslm_prefill(model, seq, &token, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) != SSLM_OK) { sslm_seq_release(seq); continue; }
        sslm_decode_params p{}; p.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers); int32_t first = -9;
        if (sslm_decode_step(model, &seq, 1, &p, nullptr, &first) != SSLM_OK || first < 0) { sslm_seq_release(seq); continue; }
        int32_t second = -9; const sslm_status refusal = sslm_decode_step(model, &seq, 1, &p, nullptr, &second);
        if (refusal != SSLM_OK && Layer(seq) == target) {
            chosen = token; reset_status = sslm_seq_reset(seq); consumed = 0;
            const sslm_status reuse = sslm_prefill(model, seq, &token, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed);
            held = reset_status == SSLM_OK && Layer(seq) == 0 && reuse == SSLM_OK && consumed == 1;
        }
        sslm_seq_release(seq);
    }
    std::printf("CELL cpu_terminal_reset layer=%u token=%d reset=%d held=%d\n", target, chosen, int(reset_status), held);
    sslm_kv_pool_destroy(pool); sslm_model_unmap(model); return chosen < 0 ? 2 : held ? 0 : 1;
}
