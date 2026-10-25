// T-3035 real-model prefill references. No precache or prefix restore verb.
#define T3022_NO_MAIN
#include "t3022_precache_reference.cpp"

#include "superslm/sha256.h"

#include <array>
#include <filesystem>

#ifndef T3035_ENGINE_ANCHOR
#error T3035_ENGINE_ANCHOR must be computed by CMake at build time
#endif
#ifndef T3035_HARNESS_COMMIT
#error T3035_HARNESS_COMMIT must be computed by CMake at build time
#endif

namespace {
namespace fs = std::filesystem;

std::vector<int32_t> ReadIds(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("prompt ids open failed: " + path.string());
    std::vector<int32_t> ids;
    int32_t id = 0;
    while (input >> id) ids.push_back(id);
    if (!input.eof() || ids.empty()) throw std::runtime_error("prompt ids parse failed: " + path.string());
    return ids;
}

std::string HexSha(const uint8_t* data, size_t size) {
    uint8_t digest[32]{};
    superslm::Sha256Hash(data, size, digest);
    return superslm::ToHex(digest);
}

std::string FileSha(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("hash input open failed: " + path.string());
    superslm::Sha256 sha;
    std::array<char, 1 << 16> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = input.gcount();
        if (got > 0) sha.Update(reinterpret_cast<const uint8_t*>(buffer.data()), size_t(got));
    }
    if (!input.eof()) throw std::runtime_error("hash input read failed: " + path.string());
    uint8_t digest[32]{};
    sha.Final(digest);
    return superslm::ToHex(digest);
}

struct CaptureResult {
    std::vector<int32_t> tokens;
    int terminal_status = 0;
    int terminal_token = -1;
    int eos_index = -1;
};

CaptureResult CaptureStream(Input& in, Pool& pool, sslm_prefix prefix, bool budget, bool damped) {
    sslm_seq seq = nullptr;
    if (budget) Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &seq), "seq_create_budgeted");
    else Check(sslm_seq_create(in.model, &pool.pool, &seq), "seq_create");
    try {
        Check(sslm_seq_adopt_prefix(seq, prefix), "seq_adopt_prefix");
        sslm_decode_params params{};
        if (damped) Check(sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &params), "params_init");
        else params.layer_budget = 1;
        CaptureResult result;
        sslm_seq batch[1] = {seq};
        for (int i = 0; i < 64; ++i) {
            int32_t token = -1;
            sslm_status status = SSLM_OK;
            for (int guard = 0; token == -1 && guard < 4096; ++guard) {
                status = damped ? sslm_decode_step_v2(in.model, batch, 1, &params, nullptr, &token)
                                : sslm_decode_step(in.model, batch, 1, &params, nullptr, &token);
                if (status != SSLM_OK) break;
            }
            if (status != SSLM_OK || token < 0) {
                result.terminal_status = int(status);
                result.terminal_token = token;
                break;
            }
            result.tokens.push_back(token);
            if (token == 151643 || token == 151645) {
                result.eos_index = i;
                break;
            }
        }
        Check(sslm_seq_release(seq), "seq_release");
        return result;
    } catch (...) {
        sslm_seq_release(seq);
        throw;
    }
}

void WriteRow(std::ofstream& out, Input& in, Pool& pool, sslm_prefix prefix,
              const std::string& id, size_t length, size_t page_positions, size_t origin,
              int adopter, bool damped_available) {
    const bool budget = adopter != 1 && adopter != 3;
    const bool damped = damped_available && (adopter == 2 || adopter == 3);
    const CaptureResult stream = CaptureStream(in, pool, prefix, budget, damped);
    out << "prefix_id=" << id << " L=" << length << " L_mod_B=" << length % page_positions
        << " origin=" << origin << " walk=0 adopter=" << adopter
        << " adopter_mode=" << (budget ? "budget" : "legacy")
        << " decode_mode=" << (damped ? "damped" : "greedy")
        << " schema_name=none schema_hash=0000000000000000 adapter=none"
        << " terminal_status=" << stream.terminal_status
        << " terminal_token=" << stream.terminal_token << " eos_index=" << stream.eos_index
        << " tokens=";
    for (size_t i = 0; i < stream.tokens.size(); ++i) {
        if (i) out << ',';
        out << stream.tokens[i];
    }
    out << '\n';
    if (stream.tokens.size() != 64 || stream.eos_index != -1 || stream.terminal_status != 0)
        throw std::runtime_error("real stream ended before token 64: " + id);
}

// Plan rev 1.18.13 §15.33 (a): the discrimination record a header binds is named per stem by the caller, with no
// default, and exactly that file is hashed. RECORD is a path holding "{id}" once, replaced by the stem (flat00,
// nested03, ...), e.g. ...\discrimination-v2\P1-real0p5-flat\{id}.stream.json. Which record is licensed is the
// run sheet's call, never this program's.
fs::path RecordPath(const std::string& pattern, const std::string& id) {
    const size_t at = pattern.find("{id}");
    if (at == std::string::npos || pattern.find("{id}", at + 1) != std::string::npos)
        throw std::runtime_error("RECORD must contain {id} exactly once: " + pattern);
    std::string path = pattern;
    path.replace(at, 4, id);
    if (!fs::is_regular_file(path)) throw std::runtime_error("discrimination record missing: " + path);
    return fs::path(path);
}

void CapturePrefix(Input& in, Pool& pool, sslm_prefix prefix, const fs::path& prompt_dir,
                   const fs::path& output_dir, const std::string& id, size_t length,
                   size_t page_positions, size_t origin, bool damped_available,
                   const std::string& model_sha, const std::string& tokenizer,
                   const std::string& id_suffix, const std::string& run_label,
                   const std::string& record_pattern) {
    const fs::path source = prompt_dir / (id + ".txt");
    const fs::path ids = prompt_dir / (id + "." + id_suffix);
    const fs::path discrimination = RecordPath(record_pattern, id);
    const fs::path output = output_dir / (id + "-" + run_label + ".ref");
    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("reference output open failed: " + output.string());
    out << "# anchor=" << T3035_ENGINE_ANCHOR << " harness=" << T3035_HARNESS_COMMIT
        << " model_sha256=" << model_sha << " B=" << page_positions << " cap=" << in.cap
        << " decode_modes_available=" << int(damped_available)
        << " prompt_source_sha256=" << FileSha(source)
        << " prompt_ids_sha256=" << FileSha(ids)
        << " tokenizer=" << tokenizer
        << " discrimination_record_sha256=" << FileSha(discrimination);
    if (origin) {
        out << " world_source_sha256=" << FileSha(prompt_dir / "world.txt")
            << " world_ids_sha256=" << FileSha(prompt_dir / ("world." + id_suffix))
            << " persona_source_sha256=" << FileSha(prompt_dir / ("persona" + id.substr(6) + ".txt"))
            << " persona_ids_sha256=" << FileSha(prompt_dir / ("persona" + id.substr(6) + "." + id_suffix));
    }
    out << '\n';
    for (int adopter = 0; adopter < 5; ++adopter)
        WriteRow(out, in, pool, prefix, id, length, page_positions, origin, adopter,
                 damped_available);
    out.flush();
    if (!out) throw std::runtime_error("reference output write failed: " + output.string());
    std::cerr << id << " captured\n";
}
}

#ifndef T3035_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc != 9 && argc != 10) throw std::runtime_error("usage: t3035_reference MODEL PROMPT_DIR OUT_DIR flat|nested run1|run2 TOKENIZER ID_SUFFIX RECORD [COUNT]  (RECORD: record path with {id} for the stem, no default)");
        const fs::path prompt_dir(argv[2]), output_dir(argv[3]);
        const std::string shape(argv[4]), run_label(argv[5]), tokenizer(argv[6]), id_suffix(argv[7]);
        const std::string record_pattern(argv[8]);
        if (shape != "flat" && shape != "nested") throw std::runtime_error("invalid shape");
        if (run_label != "run1" && run_label != "run2") throw std::runtime_error("invalid run label");
        const int count = argc == 10 ? std::stoi(argv[9]) : 10;
        if (count < 1 || count > 10) throw std::runtime_error("invalid prefix count");
        Input in(argv[1]);
        Pool pool(in.model, in.cap);
        const size_t page_positions = size_t(sslm_kv_page_positions(in.model));
        if (!page_positions) throw std::runtime_error("zero page positions");
        const std::string model_sha = HexSha(in.bytes->p, in.size);
        sslm_decode_params probe{};
        const bool damped = sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &probe) == SSLM_OK;
        if (shape == "flat") {
            for (int i = 0; i < count; ++i) {
                const std::string id = "flat" + (i < 10 ? std::string("0") : std::string()) + std::to_string(i);
                const auto ids = ReadIds(prompt_dir / (id + "." + id_suffix));
                Prefix prefix;
                Check(sslm_prefix_begin_budgeted(in.model, &pool.pool, int32_t(ids.size()), &prefix.p), "prefix_begin");
                Prefill(in.model, prefix.p, ids, SSLM_SPAN_PROMPT);
                Check(sslm_prefix_freeze(prefix.p), "prefix_freeze");
                CapturePrefix(in, pool, prefix.p, prompt_dir, output_dir, id, ids.size(),
                              page_positions, 0, damped, model_sha, tokenizer, id_suffix, run_label,
                              record_pattern);
            }
        } else {
            const auto world_ids = ReadIds(prompt_dir / ("world." + id_suffix));
            Prefix world;
            Check(sslm_prefix_begin_budgeted(in.model, &pool.pool, int32_t(world_ids.size()), &world.p), "world_begin");
            Prefill(in.model, world.p, world_ids, SSLM_SPAN_PROMPT);
            Check(sslm_prefix_freeze(world.p), "world_freeze");
            for (int i = 0; i < count; ++i) {
                const std::string suffix = (i < 10 ? std::string("0") : std::string()) + std::to_string(i);
                const std::string id = "nested" + suffix;
                const auto persona_ids = ReadIds(prompt_dir / ("persona" + suffix + "." + id_suffix));
                Prefix child;
                Check(sslm_prefix_begin_from(world.p, int32_t(persona_ids.size()), &child.p), "prefix_begin_from");
                Prefill(in.model, child.p, persona_ids, SSLM_SPAN_PROMPT);
                Check(sslm_prefix_freeze(child.p), "child_freeze");
                CapturePrefix(in, pool, child.p, prompt_dir, output_dir, id,
                              world_ids.size() + persona_ids.size(), page_positions,
                              world_ids.size(), damped, model_sha, tokenizer, id_suffix, run_label,
                              record_pattern);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "T-3035 reference: " << e.what() << '\n';
        return 1;
    }
}
#endif
