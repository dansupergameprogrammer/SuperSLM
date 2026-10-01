// T-3035 prefill references for progressed schemas and the real LoRA composition.
#define T3035_NO_MAIN
#include "t3035_reference.cpp"

#include <iomanip>
#include <sstream>

namespace {

uint64_t NameHash(const std::string& name) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char byte : name) hash = (hash ^ byte) * 1099511628211ull;
    return hash;
}

std::string HashText(const std::string& value) {
    return HexSha(reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

std::string HashName(const std::string& name) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << NameHash(name);
    return out.str();
}

CaptureResult Configured(Input& in, Pool& pool, sslm_prefix prefix, bool budget,
                         bool damped, sslm_schema schema, sslm_adapter adapter) {
    sslm_seq seq = nullptr;
    if (budget) Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &seq), "seq_create_budgeted");
    else Check(sslm_seq_create(in.model, &pool.pool, &seq), "seq_create");
    try {
        if (schema) Check(sslm_seq_set_schema(seq, schema), "seq_set_schema");
        Check(sslm_seq_adopt_prefix(seq, prefix), "seq_adopt_prefix");
        if (adapter) Check(sslm_seq_set_adapter(seq, adapter), "seq_set_adapter");
        sslm_decode_params params{};
        if (damped) Check(sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &params), "params_init_damped");
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
            if (in.vocab > 100000 && (token == 151643 || token == 151645)) {
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

void PrintTokens(std::ofstream& out, const CaptureResult& result) {
    out << " terminal_status=" << result.terminal_status
        << " terminal_token=" << result.terminal_token << " eos_index=" << result.eos_index
        << " tokens=";
    for (size_t i = 0; i < result.tokens.size(); ++i) {
        if (i) out << ',';
        out << result.tokens[i];
    }
    out << '\n';
}

int32_t DeriveFirstSchemaToken(Input& in, Pool& pool, sslm_schema schema) {
    sslm_seq seq = nullptr;
    Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &seq), "derive_create");
    try {
        Check(sslm_seq_set_schema(seq, schema), "derive_set_schema");
        const int32_t seed = 0;
        int32_t consumed = 0;
        Check(sslm_prefill(in.model, seq, &seed, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed), "derive_seed");
        if (consumed != 1) throw std::runtime_error("derive seed not consumed");
        sslm_decode_params params{};
        params.layer_budget = 1;
        int32_t token = -1;
        sslm_seq batch[1] = {seq};
        for (int guard = 0; token == -1 && guard < 4096; ++guard)
            Check(sslm_decode_step(in.model, batch, 1, &params, nullptr, &token), "derive_decode");
        if (token < 0) throw std::runtime_error("derive produced no token");
        Check(sslm_seq_release(seq), "derive_release");
        return token;
    } catch (...) {
        sslm_seq_release(seq);
        throw;
    }
}

uint32_t SavedWalk(Input& in, Pool& pool, sslm_prefix prefix, sslm_schema schema) {
    sslm_seq seq = nullptr;
    Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &seq), "walk_create");
    try {
        Check(sslm_seq_set_schema(seq, schema), "walk_set_schema");
        Check(sslm_seq_adopt_prefix(seq, prefix), "walk_adopt");
        size_t size = 0;
        if (sslm_seq_save(seq, nullptr, &size) != SSLM_BUFFER_TOO_SMALL || size < 52)
            throw std::runtime_error("walk save size");
        std::vector<uint8_t> blob(size);
        Check(sslm_seq_save(seq, blob.data(), &size), "walk_save");
        if (std::memcmp(blob.data(), "SSB6", 4)) throw std::runtime_error("walk not SSB6");
        const uint32_t walk = U32(blob.data() + 48);
        Check(sslm_seq_release(seq), "walk_release");
        return walk;
    } catch (...) {
        sslm_seq_release(seq);
        throw;
    }
}

void SchemaCapture(const char* model_path, const fs::path& prompts,
                   const fs::path& out_dir, const fs::path& discrimination,
                   const std::string& run, const std::string& ids_suffix,
                   const std::string& tokenizer, bool all_schemas) {
    Input in(model_path);
    Pool pool(in.model, in.cap);
    sslm_decode_params probe{};
    const bool damped = sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &probe) == SSLM_OK;
    const size_t n = sslm_schema_count(in.model);
    if (n < (all_schemas ? 3u : 1u)) throw std::runtime_error("insufficient schemas");
    const size_t schema_count = all_schemas ? 3 : 1;
    const auto ids = ReadIds(prompts / ("flat00." + ids_suffix));
    const std::string model_sha = HexSha(in.bytes->p, in.size);
    for (size_t i = 0; i < schema_count; ++i) {
        const std::string name = SchemaName(in.model, int(i));
        sslm_schema schema = nullptr;
        Check(sslm_schema_lookup(in.model, name.c_str(), &schema), "schema_lookup");
        Prefix prefix;
        Check(sslm_prefix_begin_budgeted(in.model, &pool.pool, int32_t(ids.size() + 1), &prefix.p), "schema_prefix_begin");
        Check(sslm_prefix_set_schema(prefix.p, schema), "schema_prefix_set");
        Prefill(in.model, prefix.p, ids, SSLM_SPAN_PROMPT);
        const int32_t forced = DeriveFirstSchemaToken(in, pool, schema);
        Prefill(in.model, prefix.p, std::vector<int32_t>{forced}, SSLM_SPAN_SCHEMA_CONTENT);
        Check(sslm_prefix_freeze(prefix.p), "schema_prefix_freeze");
        const uint32_t walk = SavedWalk(in, pool, prefix.p, schema);
        if (!walk) throw std::runtime_error("schema walk did not progress");
        const std::string id = "schema" + std::to_string(i);
        const fs::path record = discrimination / (id + ".discrimination.json");
        const fs::path output = out_dir / (id + "-" + run + ".ref");
        std::ofstream out(output, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("schema output open failed");
        out << "# anchor=" << T3035_ENGINE_ANCHOR << " harness=" << T3035_HARNESS_COMMIT
            << " model_sha256=" << model_sha
            << " B=" << sslm_kv_page_positions(in.model) << " cap=" << in.cap
            << " decode_modes_available=" << int(damped)
            << " prompt_source_sha256=" << FileSha(prompts / "flat00.txt")
            << " prompt_ids_sha256=" << FileSha(prompts / ("flat00." + ids_suffix))
            << " forced_content_id=" << forced
            << " forced_content_ids_sha256=" << HashText(std::to_string(forced) + "\n")
            << " tokenizer=" << tokenizer << " discrimination_record_sha256=" << FileSha(record) << '\n';
        for (int config = 0; config < 4; ++config) {
            const bool budget = !(config & 1), dg = damped && config >= 2;
            if (!damped && config >= 2) break;
            const auto result = Configured(in, pool, prefix.p, budget, dg, schema, nullptr);
            out << "prefix_id=" << id << " L=" << ids.size() + 1
                << " L_mod_B=" << (ids.size() + 1) % size_t(sslm_kv_page_positions(in.model))
                << " origin=0 walk=" << walk << " adopter=" << config
                << " adopter_mode=" << (budget ? "budget" : "legacy")
                << " decode_mode=" << (dg ? "damped" : "greedy")
                << " schema_name=" << name << " schema_hash=" << HashName(name)
                << " adapter=none";
            PrintTokens(out, result);
            if (result.tokens.size() != 64 || result.eos_index != -1)
                throw std::runtime_error("schema stream did not reach 64 tokens");
        }
        for (size_t bound = 0; bound <= n; ++bound) {
            if (bound == i) continue;
            sslm_seq seq = nullptr;
            Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &seq), "disposition_create");
            const std::string other = bound == n ? "none" : SchemaName(in.model, int(bound));
            if (bound != n) {
                sslm_schema other_schema = nullptr;
                Check(sslm_schema_lookup(in.model, other.c_str(), &other_schema), "other_lookup");
                Check(sslm_seq_set_schema(seq, other_schema), "other_set");
            }
            const sslm_status status = sslm_seq_adopt_prefix(seq, prefix.p);
            out << "disposition prefix_id=" << id << " bound_schema=" << other
                << " bound_schema_hash=" << (bound == n ? "0000000000000000" : HashName(other))
                << " status=" << int(status) << '\n';
            Check(sslm_seq_release(seq), "disposition_release");
        }
        out.flush();
        if (!out) throw std::runtime_error("schema write failed");
        std::cerr << id << " captured walk=" << walk << "\n";
    }
}

void AdapterCapture(const char* model_path, const fs::path& adapter_path,
                    const fs::path& prompts, const fs::path& out_dir,
                    const fs::path& discrimination, const std::string& run,
                    const std::string& ids_suffix, const std::string& tokenizer,
                    int second_index) {
    Input in(model_path);
    Pool pool(in.model, in.cap);
    const auto adapter_size = fs::file_size(adapter_path);
    Aligned adapter_bytes(adapter_size);
    std::ifstream file(adapter_path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(adapter_bytes.p), std::streamsize(adapter_size)))
        throw std::runtime_error("adapter read failed");
    sslm_adapter adapter = nullptr;
    Check(sslm_adapter_map(adapter_bytes.p, adapter_size, in.model, &adapter), "adapter_map");
    try {
        sslm_decode_params probe{};
        const bool damped = sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &probe) == SSLM_OK;
        for (int index : {0, second_index}) {
            const std::string prompt_id = "flat" + (index < 10 ? std::string("0") : std::string()) + std::to_string(index);
            const std::string id = "adapter" + std::to_string(index == 0 ? 0 : 1);
            const auto ids = ReadIds(prompts / (prompt_id + "." + ids_suffix));
            Prefix prefix;
            Check(sslm_prefix_begin_budgeted(in.model, &pool.pool, int32_t(ids.size()), &prefix.p), "adapter_prefix_begin");
            Prefill(in.model, prefix.p, ids, SSLM_SPAN_PROMPT);
            Check(sslm_prefix_freeze(prefix.p), "adapter_prefix_freeze");
            std::ofstream out(out_dir / (id + "-" + run + ".ref"), std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("adapter output open failed");
            out << "# anchor=" << T3035_ENGINE_ANCHOR << " harness=" << T3035_HARNESS_COMMIT
                << " model_sha256=" << HexSha(in.bytes->p, in.size)
                << " adapter_sha256=" << FileSha(adapter_path)
                << " B=" << sslm_kv_page_positions(in.model) << " cap=" << in.cap
                << " decode_modes_available=" << int(damped)
                << " prompt_source_sha256=" << FileSha(prompts / (prompt_id + ".txt"))
                << " prompt_ids_sha256=" << FileSha(prompts / (prompt_id + "." + ids_suffix))
                << " tokenizer=" << tokenizer
                << " discrimination_record_sha256=" << FileSha(discrimination / (id + ".discrimination.json")) << '\n';
            for (int config = 0; config < 4; ++config) {
                const bool budget = !(config & 1), on = config < 2;
                const auto result = Configured(in, pool, prefix.p, budget, false, nullptr, on ? adapter : nullptr);
                out << "prefix_id=" << id << " L=" << ids.size()
                    << " L_mod_B=" << ids.size() % size_t(sslm_kv_page_positions(in.model))
                    << " origin=0 walk=0 adopter=" << config
                    << " adopter_mode=" << (budget ? "budget" : "legacy")
                    << " decode_mode=greedy schema_name=none schema_hash=0000000000000000"
                    << " adapter=" << (on ? "bound" : "off");
                PrintTokens(out, result);
                if (result.tokens.size() != 64 || result.eos_index != -1)
                    throw std::runtime_error("adapter stream did not reach 64 tokens");
            }
            out.flush();
            if (!out) throw std::runtime_error("adapter output write failed");
            std::cerr << id << " captured\n";
        }
        Check(sslm_adapter_release(adapter), "adapter_release");
    } catch (...) {
        sslm_adapter_release(adapter);
        throw;
    }
}

void WalkProbe(const char* model_path, const fs::path& prompts, const fs::path& out_dir,
               const std::string& ids_suffix, bool all_schemas) {
    Input in(model_path);
    Pool pool(in.model, in.cap);
    const size_t schema_count = all_schemas ? 3 : 1;
    const auto ids = ReadIds(prompts / ("flat00." + ids_suffix));
    for (size_t i = 0; i < schema_count; ++i) {
        const std::string name = SchemaName(in.model, int(i));
        const size_t available = sslm_schema_count(in.model);
        const std::string other_name = SchemaName(in.model, int((i + 1) % available));
        sslm_schema schema = nullptr;
        Check(sslm_schema_lookup(in.model, name.c_str(), &schema), "walk_schema_lookup");
        Prefix prefix;
        Check(sslm_prefix_begin_budgeted(in.model, &pool.pool, int32_t(ids.size() + 1), &prefix.p), "walk_prefix_begin");
        Check(sslm_prefix_set_schema(prefix.p, schema), "walk_prefix_schema");
        Prefill(in.model, prefix.p, ids, SSLM_SPAN_PROMPT);
        const int32_t forced = DeriveFirstSchemaToken(in, pool, schema);
        Prefill(in.model, prefix.p, std::vector<int32_t>{forced}, SSLM_SPAN_SCHEMA_CONTENT);
        Check(sslm_prefix_freeze(prefix.p), "walk_prefix_freeze");
        const auto baseline = Configured(in, pool, prefix.p, true, false, schema, nullptr);
        sslm_seq seq = nullptr;
        Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &seq), "walk_seq_create");
        Check(sslm_seq_set_schema(seq, schema), "walk_seq_schema");
        Check(sslm_seq_adopt_prefix(seq, prefix.p), "walk_seq_adopt");
        size_t size = 0;
        if (sslm_seq_save(seq, nullptr, &size) != SSLM_BUFFER_TOO_SMALL || size < 52)
            throw std::runtime_error("walk save size query failed");
        std::vector<uint8_t> blob(size);
        Check(sslm_seq_save(seq, blob.data(), &size), "walk_seq_save");
        Check(sslm_seq_release(seq), "walk_seq_release");
        if (std::memcmp(blob.data(), "SSB6", 4)) throw std::runtime_error("walk probe needs SSB6");
        const uint32_t old_walk = U32(blob.data() + 48);
        if (!old_walk) throw std::runtime_error("walk probe baseline did not progress");
        const uint64_t other_hash = NameHash(other_name);
        if (available > 1)
            for (int byte = 0; byte < 8; ++byte) blob[40 + byte] = uint8_t(other_hash >> (8 * byte));
        for (int byte = 0; byte < 4; ++byte) blob[48 + byte] = 0;
        sslm_seq restored = nullptr;
        const sslm_status restore_status = sslm_seq_restore(in.model, &pool.pool, blob.data(), blob.size(), &restored);
        std::vector<int32_t> mutant;
        int terminal_status = int(restore_status);
        if (restore_status == SSLM_OK) {
            sslm_decode_params params{};
            params.layer_budget = 1;
            sslm_seq batch[1] = {restored};
            for (int token_index = 0; token_index < 64; ++token_index) {
                int32_t token = -1;
                sslm_status status = SSLM_OK;
                for (int guard = 0; token == -1 && guard < 4096; ++guard) {
                    status = sslm_decode_step(in.model, batch, 1, &params, nullptr, &token);
                    if (status != SSLM_OK) break;
                }
                if (status != SSLM_OK || token < 0) {
                    terminal_status = int(status);
                    break;
                }
                mutant.push_back(token);
                if (in.vocab > 100000 && (token == 151643 || token == 151645)) break;
                if (size_t(token_index) >= baseline.tokens.size() || token != baseline.tokens[size_t(token_index)]) break;
            }
            Check(sslm_seq_release(restored), "walk_restored_release");
        }
        int first_diff = -1;
        const size_t common = std::min(mutant.size(), baseline.tokens.size());
        for (size_t t = 0; t < common; ++t)
            if (mutant[t] != baseline.tokens[t]) { first_diff = int(t); break; }
        if (first_diff < 0 && mutant.size() != baseline.tokens.size()) first_diff = int(common);
        const fs::path output = out_dir / ("schema" + std::to_string(i) + ".walkprobe.txt");
        std::ofstream out(output, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("walk probe output open failed");
        out << "schema=" << name << " schema_hash=" << HashName(name)
            << " donor_schema=" << other_name << " donor_schema_hash=" << HashName(other_name)
            << " original_walk=" << old_walk << " mutant_walk=0"
            << " forced_content_id=" << forced << " length=" << ids.size() + 1
            << " baseline_tokens=" << baseline.tokens.size()
            << " restore_status=" << int(restore_status)
            << " terminal_status=" << terminal_status
            << " mutant_tokens=" << mutant.size()
            << " killed=" << int(first_diff >= 0)
            << " first_diff=" << first_diff << '\n';
        out.flush();
        if (!out) throw std::runtime_error("walk probe write failed");
        std::cerr << "schema" << i << " walk probe complete\n";
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "walk") {
            fs::create_directories(argv[4]);
            WalkProbe(argv[2], argv[3], argv[4], argv[5], false);
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "walk3") {
            fs::create_directories(argv[4]);
            WalkProbe(argv[2], argv[3], argv[4], argv[5], true);
            return 0;
        }
        if (argc != 10 && argc != 11) throw std::runtime_error("usage: t3035_composition schema|schema3|adapter MODEL ADAPTER_OR_- PROMPTS OUT DISCRIM run1|run2 ID_SUFFIX TOKENIZER [SECOND_INDEX]");
        const std::string kind(argv[1]), run(argv[7]);
        if (run != "run1" && run != "run2") throw std::runtime_error("invalid run");
        fs::create_directories(argv[5]);
        if (kind == "schema" || kind == "schema3")
            SchemaCapture(argv[2], argv[4], argv[5], argv[6], run, argv[8], argv[9], kind == "schema3");
        else if (kind == "adapter") {
            const int second_index = argc == 11 ? std::stoi(argv[10]) : 9;
            if (second_index < 1 || second_index > 9) throw std::runtime_error("invalid second adapter prefix index");
            AdapterCapture(argv[2], argv[3], argv[4], argv[5], argv[6], run, argv[8], argv[9], second_index);
        }
        else throw std::runtime_error("unknown kind");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "T-3035 composition: " << e.what() << '\n';
        return 1;
    }
}
