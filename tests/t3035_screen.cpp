// Probe candidate real-text prompts before pinning T-3035 references.
#define T3022_NO_MAIN
#include "t3022_precache_reference.cpp"

#include <sstream>

namespace {
std::vector<int32_t> ReadIds(const char* path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("prompt ids open failed");
    std::vector<int32_t> ids;
    int32_t id = 0;
    while (input >> id) ids.push_back(id);
    if (!input.eof() || ids.empty()) throw std::runtime_error("prompt ids parse failed");
    return ids;
}

void Screen(std::ofstream& out, Input& in, Pool& pool, const char* path,
            const std::vector<int32_t>& ids, bool damped) {
    Prefix prefix;
    Check(sslm_prefix_begin_budgeted(in.model, &pool.pool, int32_t(ids.size()), &prefix.p), "prefix_begin");
    Prefill(in.model, prefix.p, ids, SSLM_SPAN_PROMPT);
    Check(sslm_prefix_freeze(prefix.p), "prefix_freeze");
    for (int mode = 0; mode < (damped ? 2 : 1); ++mode) {
        sslm_seq sequence = nullptr;
        Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &sequence), "seq_create");
        Check(sslm_seq_adopt_prefix(sequence, prefix.p), "seq_adopt_prefix");
        sslm_decode_params params{};
        if (mode) Check(sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &params), "params_init");
        else params.layer_budget = 1;
        out << path << " L=" << ids.size() << " mode=" << (mode ? "damped" : "greedy") << " tokens=";
        sslm_seq batch[1] = {sequence};
        for (int i = 0; i < 64; ++i) {
            int32_t token = -1;
            sslm_status status = SSLM_OK;
            for (int guard = 0; token == -1 && guard < 4096; ++guard) {
                status = mode ? sslm_decode_step_v2(in.model, batch, 1, &params, nullptr, &token)
                              : sslm_decode_step(in.model, batch, 1, &params, nullptr, &token);
                if (status != SSLM_OK) break;
            }
            if (status != SSLM_OK || token < 0) {
                out << " terminal_status=" << int(status) << " terminal_token=" << token;
                break;
            }
            if (i) out << ',';
            out << token;
            if (token == 151643 || token == 151645) {
                out << " eos_index=" << i;
                break;
            }
        }
        out << '\n';
        Check(sslm_seq_release(sequence), "seq_release");
    }
    std::cerr << path << " done\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: t3035_screen MODEL OUT IDS...");
        Input in(argv[1]);
        Pool pool(in.model, in.cap);
        sslm_decode_params probe{};
        const bool damped = sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &probe) == SSLM_OK;
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("output open failed");
        for (int i = 3; i < argc; ++i) Screen(output, in, pool, argv[i], ReadIds(argv[i]), damped);
        output.flush();
        if (!output) throw std::runtime_error("output write failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "T-3035 screen: " << e.what() << '\n';
        return 1;
    }
}
