// Precached prefixes, plan §7 9.5 [F4]: across releases, a blob written by 1.14.0 is refused by 1.16.0 with
// SSLM_RESTORE_PRODUCER_MISMATCH ("run at F4 on both binaries; the pinned 1.14.0 blob of §8").
//
//   9.5/F4               the four pinned 1.14.0 'SPX1' blobs (§8, rev 1.18.14 §15.34: pkv_def, pkv_odd, pkv_32k and
//                        pkv_qk, each at the rule B, L = B + 13 so the last page is partial), read from
//                        tests/precache/reference/pins-9.5 and checked against their manifest, are each refused by this
//                        build's sslm_prefix_restore with SSLM_RESTORE_PRODUCER_MISMATCH, a null out-handle, a zero
//                        shared-page count, and the pool unchanged: every page still free, read by the one-state fill
//                        probe. The refusal must be the producer's, so the pin's every other field is checked first
//                        (backend 0, CPU adapter ids 0:0, its digest, its model hash equal to this fixture's).
//   9.5/F4:must-reject   the instrument's must-reject (§8: a cell's verdict counts only once it refuses a wrong case):
//                        the same pin relabelled to this build's own version (1.16.0) and re-digested is accepted by the
//                        restore, and the 9.5 grade of it must come out false. If it came out true, the grade would
//                        pass a blob 1.16.0 accepts, and 9.5 would grade nothing.
//
// Unpinned (§7 preamble): 1.14.0 wrote the pins at each fixture's production rule B, so each fixture is its own mapping
// at the rule (MapOwn, RequireRuleB), and its pool is a rule-B page pool.
//
// "Both binaries": this cell is the GCC/Clang half in the cloud, and the MSVC half on the box (the same source, the
// superslm_precache_unpinned target built with MSVC; release-1p16/f3-f4-cells-2026-10-24.md has the command).

#include "unpinned_common.h"
#include "superslm/sslm_version.h"

#include <fstream>
#include <map>
#include <sstream>

#ifndef PC_REFERENCE_DIR
#error "PC_REFERENCE_DIR must name tests/precache/reference (precache.cmake defines it)"
#endif

namespace {
using namespace pc;
using namespace pc::unpinned;

constexpr uint32_t kPoolPages = 8;  // a blob of L = B + 13 needs 2 pages; the probe reads all 8 free after a refusal

struct Pin {
    std::string stem, blob, blob_sha, fixture_sha;
    int64_t B = 0, L = 0;
    size_t size = 0;
};

std::map<std::string, std::string> Words(const std::string& line) {
    std::map<std::string, std::string> f;
    std::istringstream in(line);
    std::string w;
    while (in >> w) {
        const size_t eq = w.find('=');
        if (eq != std::string::npos) f[w.substr(0, eq)] = w.substr(eq + 1);
    }
    return f;
}

// The manifest tools/build_precache_9p5_pins.sh wrote: a header naming writer=1.14.0, then one line per geometry.
std::map<std::string, Pin> ReadManifest(std::string* writer) {
    std::map<std::string, Pin> pins;
    const std::string path = std::string(PC_REFERENCE_DIR) + "/pins-9.5/manifest.txt";
    std::ifstream in(path, std::ios::binary);
    PKV_CHECK_MSG(bool(in), "9.5: the pin manifest %s is absent (plan §8 pins it at E5)", path.c_str());
    std::string line;
    while (in && GetTextLine(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto f = Words(line);
            if (f.count("writer")) *writer = f.at("writer");
            continue;
        }
        std::istringstream s(line);
        Pin p;
        s >> p.stem;
        const auto f = Words(line);
        p.blob = f.count("blob") ? f.at("blob") : "";
        p.blob_sha = f.count("blob_sha256") ? f.at("blob_sha256") : "";
        p.fixture_sha = f.count("fixture_sha256") ? f.at("fixture_sha256") : "";
        p.B = f.count("B") ? std::atoll(f.at("B").c_str()) : 0;
        p.L = f.count("L") ? std::atoll(f.at("L").c_str()) : 0;
        p.size = f.count("size") ? size_t(std::strtoull(f.at("size").c_str(), nullptr, 10)) : 0;
        pins[p.stem] = p;
    }
    return pins;
}

uint32_t Le32(const std::vector<uint8_t>& b, size_t o) {
    uint32_t v = 0;
    if (o + 4 <= b.size()) std::memcpy(&v, b.data() + o, 4);
    return v;
}
uint64_t Le64(const std::vector<uint8_t>& b, size_t o) {
    uint64_t v = 0;
    if (o + 8 <= b.size()) std::memcpy(&v, b.data() + o, 8);
    return v;
}
void Put32(std::vector<uint8_t>& b, size_t o, uint32_t v) {
    if (o + 4 <= b.size()) std::memcpy(b.data() + o, &v, 4);
}

// The pin, read and checked against its manifest line and against this fixture: everything a restore checks before
// the producer (size, magic, digest, backend) and the field right after it (the model hash), so that a refusal here
// can only be the producer's.
bool LoadPin(const Pin& p, const Fixture& fx, std::vector<uint8_t>* blob) {
    const std::string path = std::string(PC_REFERENCE_DIR) + "/pins-9.5/" + p.blob;
    const bool read = !p.blob.empty() && ReadAll(path, blob) && !blob->empty();
    PKV_CHECK_MSG(read, "9.5 %s: the pinned blob %s is missing", p.stem.c_str(), path.c_str());
    if (!read) return false;
    const std::string sha = Sha(blob->data(), blob->size());
    PKV_CHECK_MSG(sha == p.blob_sha && blob->size() == p.size, "9.5 %s: %s is %zu bytes, sha256 %s; the manifest has %zu, %s",
                  p.stem.c_str(), p.blob.c_str(), blob->size(), sha.c_str(), p.size, p.blob_sha.c_str());
    PKV_CHECK_MSG(p.fixture_sha == fx.sha, "9.5 %s: the pin was written on fixture %s; this fixture is %s", p.stem.c_str(),
                  p.fixture_sha.c_str(), fx.sha.c_str());
    PKV_CHECK_MSG(p.B == fx.B(), "9.5 %s: pinned at B = %lld; this fixture's rule B is %lld", p.stem.c_str(),
                  static_cast<long long>(p.B), static_cast<long long>(fx.B()));
    if (sha != p.blob_sha || blob->size() != p.size || p.fixture_sha != fx.sha || p.B != fx.B()) return false;
    std::vector<uint8_t> redigested(*blob);
    Digest(redigested);
    const bool shape = blob->size() >= kHeader && std::memcmp(blob->data(), "SPX1", 4) == 0 && redigested == *blob &&
                       Le32(*blob, 4) == 0 && Le32(*blob, 20) == 0 && Le32(*blob, 24) == 0 &&
                       int64_t(Le64(*blob, 112)) == p.L && p.L % fx.B() != 0;
    PKV_CHECK_MSG(shape, "9.5 %s: the pin is not a valid CPU SPX1 of L = %lld (magic, digest, backend 0, adapter 0:0, "
                  "a partial last page)", p.stem.c_str(), static_cast<long long>(p.L));
    const uint32_t major = Le32(*blob, 8), minor = Le32(*blob, 12), patch = Le32(*blob, 16);
    PKV_CHECK_MSG(major == 1 && minor == 14 && patch == 0, "9.5 %s: the pin's producer is %u.%u.%u, not 1.14.0",
                  p.stem.c_str(), major, minor, patch);
    // The model hash is the check right after the producer (§3.4 step 7): it must match, or a broken producer check
    // would surface as SSLM_RESTORE_MODEL_MISMATCH and still not be a 1.16.0 acceptance.
    std::vector<uint8_t> own_blob;
    {
        PagePool pool(fx.model, kPoolPages);
        Handles h;
        sslm_prefix p0 = nullptr;
        Status(sslm_prefix_begin_budgeted(fx.model, &pool.pool, 1, &p0), SSLM_OK, "9.5 own L = 0 prefix");
        if (p0) {
            h.prefixes.push_back(p0);
            Status(sslm_prefix_freeze(p0), SSLM_OK, "9.5 own L = 0 freeze");
            own_blob = Save(p0);
        }
    }
    const bool model = own_blob.size() >= kHeader && std::memcmp(own_blob.data() + 60, blob->data() + 60, 32) == 0;
    PKV_CHECK_MSG(model, "9.5 %s: the pin's model hash is not this fixture's", p.stem.c_str());
    return shape && major == 1 && minor == 14 && patch == 0 && model;
}

// The 9.5 grade of one blob: refused SSLM_RESTORE_PRODUCER_MISMATCH, a null out-handle, no shared pages, and every
// page of the pool still free. `report` false runs it silently (the must-reject's dry run).
bool Grade95(const Fixture& fx, const std::vector<uint8_t>& blob, const std::string& what, bool report) {
    PagePool pool(fx.model, kPoolPages);
    if (report) PKV_CHECK_EQ(pool.status, SSLM_OK);
    if (pool.status != SSLM_OK) return false;
    sslm_prefix out = nullptr;
    uint32_t shared = 0xffffffffu;
    const sslm_status st = Restore(fx, &pool.pool, blob, nullptr, &out, &shared);
    const bool refused = st == SSLM_RESTORE_PRODUCER_MISMATCH && out == nullptr && shared == 0;
    if (st == SSLM_OK && out && out != OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    if (report)
        PKV_CHECK_MSG(refused, "9.5 %s: restore returned %s (%d), out-handle %s, shared %u; want "
                      "SSLM_RESTORE_PRODUCER_MISMATCH, null, 0", what.c_str(), AbiStatusName(st), int(st),
                      out == nullptr ? "null" : (out == OutSentinel<sslm_prefix>() ? "untouched" : "set"), shared);
    bool unchanged = false;
    if (refused) {
        unchanged = report ? ProbeExactlyFreeOneState(fx, &pool.pool, kPoolPages) : true;
        if (report) PKV_CHECK_MSG(unchanged, "9.5 %s: the refusal changed the pool (not every page free)", what.c_str());
    }
    return refused && unchanged;
}

const Shape* kShapes[] = {&kPkvDef, &kPkvOdd, &kPkv32k, &kPkvQk};

void Release95() {
    std::string writer;
    const auto pins = ReadManifest(&writer);
    PKV_CHECK_MSG(writer == "1.14.0", "9.5: the manifest's writer is '%s', not 1.14.0", writer.c_str());
    PKV_CHECK_EQ(pins.size(), size_t(4));
    PKV_CHECK_MSG(SUPERSLM_VERSION_MAJOR == 1 && SUPERSLM_VERSION_MINOR == 16,
                  "9.5: this build is %d.%d.%d; the cell grades 1.16's refusal", SUPERSLM_VERSION_MAJOR,
                  SUPERSLM_VERSION_MINOR, SUPERSLM_VERSION_PATCH);
    int graded = 0;
    for (const Shape* s : kShapes) {
        const auto it = pins.find(s->name);
        PKV_CHECK_MSG(it != pins.end(), "9.5: the manifest has no pin for %s", s->name);
        if (it == pins.end()) continue;
        Mapped m = MapOwn(*s);
        if (!m.fx || !RequireRuleB(*m.fx, *s, false)) continue;
        std::vector<uint8_t> blob;
        if (!LoadPin(it->second, *m.fx, &blob)) continue;
        if (Grade95(*m.fx, blob, it->second.blob, true)) ++graded;
        std::printf("9.5 %s: 1.14.0 pin %s (L = %lld, B = %lld) on %d.%d.%d: refused SSLM_RESTORE_PRODUCER_MISMATCH\n",
                    s->name, it->second.blob.c_str(), static_cast<long long>(it->second.L),
                    static_cast<long long>(it->second.B), SUPERSLM_VERSION_MAJOR, SUPERSLM_VERSION_MINOR,
                    SUPERSLM_VERSION_PATCH);
    }
    PKV_CHECK_EQ(graded, 4);
}
PKV_CELL("9.5/F4", "F4", Release95);

// The must-reject: each pin relabelled to this build's version and re-digested. The restore must accept it (so the
// construction is a blob 1.16.0 really takes), and the 9.5 grade must refuse to pass it.
void Release95MustReject() {
    std::string writer;
    const auto pins = ReadManifest(&writer);
    int rejected = 0;
    for (const Shape* s : kShapes) {
        const auto it = pins.find(s->name);
        if (it == pins.end()) continue;
        Mapped m = MapOwn(*s);
        if (!m.fx || !RequireRuleB(*m.fx, *s, false)) continue;
        std::vector<uint8_t> blob;
        if (!LoadPin(it->second, *m.fx, &blob)) continue;
        Put32(blob, 8, SUPERSLM_VERSION_MAJOR);
        Put32(blob, 12, SUPERSLM_VERSION_MINOR);
        Put32(blob, 16, SUPERSLM_VERSION_PATCH);
        Digest(blob);
        {
            PagePool pool(m.fx->model, kPoolPages);
            sslm_prefix out = nullptr;
            const sslm_status st = Restore(*m.fx, &pool.pool, blob, nullptr, &out);
            PKV_CHECK_MSG(st == SSLM_OK, "9.5 must-reject %s: the pin relabelled %d.%d.%d is not accepted (%s): the "
                          "construction is not a blob this build takes", s->name, SUPERSLM_VERSION_MAJOR,
                          SUPERSLM_VERSION_MINOR, SUPERSLM_VERSION_PATCH, AbiStatusName(st));
            if (st == SSLM_OK && out && out != OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
            if (st != SSLM_OK) continue;
        }
        const bool passed = Grade95(*m.fx, blob, std::string("must-reject ") + s->name, false);
        PKV_CHECK_MSG(!passed, "9.5 must-reject %s: the grade passed a blob this build accepts", s->name);
        if (!passed) ++rejected;
    }
    PKV_CHECK_EQ(rejected, 4);
}
PKV_CELL("9.5/F4:must-reject", "F4", Release95MustReject);

}  // namespace
