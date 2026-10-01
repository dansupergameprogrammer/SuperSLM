// Quarantined T-3035 shipped-verb SSB6 mutation probe.
#define T3022_NO_MAIN
#include "t3022_precache_reference.cpp"

#include <array>
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

struct Geometry {
    size_t hidden = 0, runs = 0, row_bytes = 0, page_positions = 0;
    explicit Geometry(const Input& in) {
        const uint8_t* h = in.bytes->p;
        const uint32_t count = U32(h + 12);
        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t* row = h + 64 + size_t(i) * 40;
            if (U32(row) != 0) continue;
            const uint8_t* c = h + U64(row + 8);
            hidden = U32(c + 8);
            runs = size_t(U32(c + 12)) * 2 * U32(c + 20);
            row_bytes = size_t(U32(c + 24)) * (U32(c + 44) == 1 ? 2 : 1);
            page_positions = size_t(sslm_kv_page_positions(in.model));
            break;
        }
        if (!hidden || !runs || !row_bytes || !page_positions) throw std::runtime_error("bad CFG1 geometry");
    }
};

std::vector<uint8_t> SaveAdopted(Input& in, Pool& pool, const std::vector<int32_t>& ids) {
    Prefix prefix;
    Check(sslm_prefix_begin_budgeted(in.model, &pool.pool, int32_t(ids.size()), &prefix.p), "prefix_begin");
    Prefill(in.model, prefix.p, ids, SSLM_SPAN_PROMPT);
    Check(sslm_prefix_freeze(prefix.p), "prefix_freeze");
    sslm_seq sequence = nullptr;
    Check(sslm_seq_create_budgeted(in.model, &pool.pool, 128, &sequence), "seq_create");
    Check(sslm_seq_adopt_prefix(sequence, prefix.p), "seq_adopt_prefix");
    size_t size = 0;
    if (sslm_seq_save(sequence, nullptr, &size) != SSLM_BUFFER_TOO_SMALL || !size)
        throw std::runtime_error("save size query failed");
    std::vector<uint8_t> blob(size);
    Check(sslm_seq_save(sequence, blob.data(), &size), "seq_save");
    blob.resize(size);
    Check(sslm_seq_release(sequence), "seq_release");
    if (blob.size() < 172 || std::memcmp(blob.data(), "SSB6", 4))
        throw std::runtime_error("budget adopter did not save SSB6");
    return blob;
}

struct Layout {
    size_t length = 0, rows_offset = 0;
    Layout(const std::vector<uint8_t>& blob, const Geometry& g) {
        const size_t history = size_t(U64(blob.data() + 112));
        const size_t count_offset = 172 + g.hidden + history * 4;
        if (count_offset + 8 > blob.size()) throw std::runtime_error("short SSB6 count");
        length = size_t(U64(blob.data() + count_offset));
        rows_offset = count_offset + 8;
        if (length == 0 || rows_offset + length * g.runs * g.row_bytes != blob.size())
            throw std::runtime_error("SSB6 rows geometry mismatch");
    }
    size_t Row(size_t run, size_t position, const Geometry& g) const {
        if (run >= g.runs || position >= length) throw std::runtime_error("row index out of bounds");
        return rows_offset + (run * length + position) * g.row_bytes;
    }
};

std::vector<int32_t> Decode(Input& in, Pool& pool, const std::vector<uint8_t>& blob, bool damped,
                            const std::vector<int32_t>* expected = nullptr) {
    sslm_seq sequence = nullptr;
    Check(sslm_seq_restore(in.model, &pool.pool, blob.data(), blob.size(), &sequence), "seq_restore");
    sslm_decode_params params{};
    if (damped) Check(sslm_decode_params_init(in.model, SSLM_DECODE_MODE_DAMPED_GREEDY, 1, &params), "params_init");
    else params.layer_budget = 1;
    sslm_seq batch[1] = {sequence};
    std::vector<int32_t> tokens;
    for (int i = 0; i < 64; ++i) {
        int32_t token = -1;
        sslm_status status = SSLM_OK;
        for (int guard = 0; token == -1 && guard < 4096; ++guard) {
            status = damped ? sslm_decode_step_v2(in.model, batch, 1, &params, nullptr, &token)
                            : sslm_decode_step(in.model, batch, 1, &params, nullptr, &token);
            if (status != SSLM_OK) break;
        }
        if (status != SSLM_OK || token < 0) break;
        tokens.push_back(token);
        if (expected && (size_t(i) >= expected->size() || token != (*expected)[size_t(i)])) break;
        if (token == 151643 || token == 151645) break;
    }
    Check(sslm_seq_release(sequence), "seq_release");
    return tokens;
}

int FirstDiff(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return int(i);
    return a.size() == b.size() ? -1 : int(n);
}

void CopyRows(std::vector<uint8_t>& target, const std::vector<uint8_t>& donor,
              const Layout& t, const Layout& d, const Geometry& g, size_t first, size_t count) {
    if (first + count > t.length || first + count > d.length) throw std::runtime_error("donor too short");
    for (size_t run = 0; run < g.runs; ++run)
        for (size_t position = first; position < first + count; ++position)
            std::memcpy(target.data() + t.Row(run, position, g),
                        donor.data() + d.Row(run, position, g), g.row_bytes);
}

void Report(std::ofstream& out, const char* name, Input& in, Pool& pool,
            const std::vector<uint8_t>& blob, const std::vector<int32_t>& baseline, bool damped) {
    const auto tokens = Decode(in, pool, blob, damped, &baseline);
    out << name << " killed=" << int(tokens != baseline)
        << " first_diff=" << FirstDiff(baseline, tokens)
        << " tokens=" << tokens.size() << '\n';
    out.flush();
    if (!out) throw std::runtime_error("output write failed");
    std::cerr << name << " done\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7) throw std::runtime_error("usage: t3035_mutant_probe MODEL TARGET_IDS DONOR_IDS OUT greedy|damped [swap]");
        const std::string mode = argv[5];
        if (mode != "greedy" && mode != "damped") throw std::runtime_error("invalid decode mode");
        const bool damped = mode == "damped";
        const bool swap_only = argc == 7 && std::string(argv[6]) == "swap";
        if (argc == 7 && !swap_only) throw std::runtime_error("unknown probe selector");
        Input in(argv[1]);
        Pool pool(in.model, in.cap);
        const Geometry geometry(in);
        const auto target = SaveAdopted(in, pool, ReadIds(argv[2]));
        const auto donor = SaveAdopted(in, pool, ReadIds(argv[3]));
        const Layout t(target, geometry), d(donor, geometry);
        if (d.length < t.length) throw std::runtime_error("donor must cover target positions");
        std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("output open failed");
        const auto baseline = Decode(in, pool, target, damped);
        output << "# L=" << t.length << " B=" << geometry.page_positions
               << " runs=" << geometry.runs << " row_bytes=" << geometry.row_bytes
               << " mode=" << mode << " baseline_tokens=" << baseline.size() << " tokens=";
        for (size_t i = 0; i < baseline.size(); ++i) {
            if (i) output << ',';
            output << baseline[i];
        }
        output << '\n';
        const size_t B = geometry.page_positions, L = t.length;
        const size_t full_pages = L / B;
        if (!swap_only) {
        Report(output, "M-res", in, pool, [&] {
            auto b = target;
            std::memcpy(b.data() + 172, donor.data() + 172, geometry.hidden);
            return b;
        }(), baseline, damped);
        const std::array<size_t, 7> positions{0, 1, B - 1, B, L / 2, L - B - 1, L - 1};
        for (size_t position : positions) {
            auto b = target;
            CopyRows(b, donor, t, d, geometry, position, 1);
            const std::string name = "M-row(" + std::to_string(position) + ")";
            Report(output, name.c_str(), in, pool, b, baseline, damped);
        }
        const std::array<size_t, 3> pages{0, full_pages / 2, full_pages - 1};
        for (size_t page : pages) {
            auto b = target;
            CopyRows(b, donor, t, d, geometry, page * B, B);
            const std::string name = "M-page(" + std::to_string(page) + ")";
            Report(output, name.c_str(), in, pool, b, baseline, damped);
        }
        if (L % B) {
            auto b = target;
            CopyRows(b, donor, t, d, geometry, full_pages * B, L % B);
            Report(output, "M-tail", in, pool, b, baseline, damped);
        }
        }
        if (full_pages >= 2) {
            auto b = target;
            const size_t first = (full_pages - 2) * B;
            const size_t other = (full_pages - 1) * B;
            for (size_t run = 0; run < geometry.runs; ++run)
                for (size_t position = 0; position < B; ++position)
                    for (size_t byte = 0; byte < geometry.row_bytes; ++byte)
                        std::swap(b[t.Row(run, first + position, geometry) + byte],
                                  b[t.Row(run, other + position, geometry) + byte]);
            const std::string name = "M-swap(" + std::to_string(first / B) + "," +
                                     std::to_string(other / B) + ")";
            Report(output, name.c_str(), in, pool, b, baseline, damped);
        }
        output.flush();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "T-3035 mutant probe: " << e.what() << '\n';
        return 1;
    }
}
