// T-2851 rows 4 and 6: the wide and narrowed row must equal serial LogitsSite.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/parallel_for.h"

namespace superslm {
SslmForwardStatus LogitsSiteParallel(const int8_t*, size_t, const int8_t*, size_t,
                                     int64_t*, int32_t*, const sslm_parallel_for*);
}

namespace {
struct Invocation {
    int calls = 0;
    int bad_count = 0;
    int max_tasks = 0;
};
void Reverse(void* host, int32_t count, sslm_task_fn task, void* ctx) {
    auto& state = *static_cast<Invocation*>(host);
    ++state.calls;
    if (count < 1 || count > state.max_tasks) ++state.bad_count;
    for (int32_t i = count - 1; i >= 0; --i) task(ctx, i);
}

std::vector<uint8_t> Read(const char* path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") || !f) return {};
    _fseeki64(f, 0, SEEK_END);
    auto n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> b(static_cast<size_t>(n));
    bool ok = fread(b.data(), 1, b.size(), f) == b.size();
    fclose(f);
    return ok ? b : std::vector<uint8_t>{};
}

bool Compare(const int8_t* head, size_t vocab, size_t hidden, int vectors) {
    constexpr std::array<int, 9> settings{0, 1, 2, 3, 4, 7, 64, 255, 256};
    std::vector<int8_t> x(hidden);
    std::vector<int64_t> reference_wide(vocab), candidate_wide(vocab);
    std::vector<int32_t> reference(vocab), candidate(vocab);
    uint32_t random = 0x2956abcdu;
    for (int vector = 0; vector < vectors; ++vector) {
        for (size_t j = 0; j < hidden; ++j) {
            if (vector == 0) x[j] = 127;
            else if (vector == 1) x[j] = j & 1 ? -127 : 127;
            else if (vector == 2) x[j] = -127;
            else {
                random = 1664525u * random + 1013904223u;
                x[j] = static_cast<int8_t>(static_cast<int>((random >> 16) % 255) - 127);
            }
        }
        if (superslm::LogitsSite(x.data(), hidden, head, vocab,
                reference_wide.data(), reference.data()) != superslm::SslmForwardStatus::Ok)
            return false;
        for (int tasks : settings) {
            Invocation inv{};
            inv.max_tasks = tasks;
            sslm_parallel_for pf{};
            pf.run = &Reverse;
            pf.host_ctx = &inv;
            pf.max_tasks = tasks;
            const sslm_parallel_for* hook = tasks ? &pf : nullptr;
            candidate_wide.assign(vocab, INT64_MIN);
            candidate.assign(vocab, INT32_MIN);
            const auto status = superslm::LogitsSiteParallel(x.data(), hidden, head, vocab,
                                      candidate_wide.data(), candidate.data(), hook);
            if (status != superslm::SslmForwardStatus::Ok || inv.bad_count ||
                (tasks > 1 && !inv.calls) || reference_wide != candidate_wide ||
                reference != candidate) {
                std::fprintf(stderr, "FAIL rows vector=%d max_tasks=%d status=%d calls=%d\n",
                             vector, tasks, static_cast<int>(status), inv.calls);
                return false;
            }
            if (vocab == 70 && vector == 0 &&
                superslm::ArgmaxLowestIndexTieBreak(candidate.data(), vocab) != 0)
                return false;
        }
    }
    return true;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto bytes = Read(argv[1]);
    if (bytes.empty()) return 3;
    superslm::SslmModelView model;
    std::string error;
    if (superslm::SslmModel::Load(bytes.data(), bytes.size(), model, &error) !=
        superslm::SslmModelStatus::Ok) return 4;
    const char* head_name = model.config.tie_word_embeddings ? "embed" : "lm_head";
    const auto* head = model.weights.Tensor(head_name);
    if (!head || head->rank != 2 || head->shape[0] != model.config.vocab_size ||
        head->shape[1] != model.config.hidden_size) return 5;
    if (!Compare(reinterpret_cast<const int8_t*>(head->data), model.config.vocab_size,
                 model.config.hidden_size, 103)) return 6;
    // A tie across the 64-row partition boundary exercises the serial tail.
    constexpr size_t V = 70, H = 8;
    std::vector<int8_t> synthetic(V * H, 0);
    for (size_t k = 0; k < H; ++k) synthetic[k] = synthetic[64 * H + k] = 1;
    if (!Compare(synthetic.data(), V, H, 3)) return 7;
    std::printf("PASS row_identity vectors=103 settings=9 synthetic_tie=0,64\n");
    return 0;
}
