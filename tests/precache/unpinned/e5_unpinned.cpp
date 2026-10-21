// Precached prefixes, plan rev 1.18.1: the unpinned E5 cell.
//
//   9.9/E5   1.13.1's CPU sequence saves on the release tree (PC-13; audit G14 (c)).
//
// The predecessor is v1.13.1 (d697927, tag object 034a76a). Its saves are captured once on a v1.13.1 build by
// tools/build_precache_v1p13p1_captures.sh (tests/precache/reference/capture_v1p13p1_seq_saves.cpp, the cases of
// pc_seq_saves.h) and pinned with their SHA-256s and the 64-token continuation each decodes to on that build. Here,
// on the tree under work, each pinned blob restores through sslm_seq_restore and, for SSB6, through
// sslm_seq_restore_shared (the nested adopter with its matching hint, the nested chain rebuilt on this tree; the
// root and mid-turn states have no prefix to hint, so they take the NULL hint the verb documents as the plain
// restore), re-saves byte-identical to its pin, and decodes the pinned continuation. The oracle is the 1.13.1
// build's own output.
//
// Unpinned (§7 preamble): 1.13.1 saved at pkv_def's production page rule (B = 128), so the restore pool here runs
// at that rule too; the e1 target's B = 16 pin would put the restore at another page size than the capture's. The
// cell maps its own pkv_def instance (MapOwn) and checks the rule first, as every unpinned cell does.
//
// Red-first (plan §9 E5: the capture runs at E5, on d697927): until the capture is pinned, the cell is red on the
// capture's absence, the reason it names.

#include "unpinned_common.h"
#include "precache/reference/pc_seq_saves.h"

#include <fstream>
#include <map>
#include <sstream>

#ifndef PC_REFERENCE_DIR
#error "PC_REFERENCE_DIR must name tests/precache/reference (precache.cmake defines it)"
#endif

namespace {
using namespace pc;
using namespace pc::unpinned;

constexpr const char* kTag = "v1.13.1";
constexpr const char* kCommit = "d697927f6d5ce5fa73ecbb8660020a425518b923";

struct Capture {
    std::string magic, blob_sha;
    size_t size = 0;
    std::vector<int32_t> tokens;
};

struct CaptureFile {
    bool ok = false;
    std::map<std::string, std::string> header;
    std::map<std::string, Capture> cases;
    std::vector<std::string> failed;
};

std::map<std::string, std::string> Fields(const std::string& line) {
    std::map<std::string, std::string> f;
    std::istringstream in(line);
    std::string word;
    while (in >> word) {
        const size_t eq = word.find('=');
        if (eq != std::string::npos) f[word.substr(0, eq)] = word.substr(eq + 1);
    }
    return f;
}

CaptureFile ReadCaptures(const std::string& path) {
    CaptureFile c;
    std::ifstream in(path, std::ios::binary);
    if (!in) return c;
    std::string line;
    if (!GetTextLine(in, line) || line.rfind("# ", 0) != 0) return c;
    c.header = Fields(line.substr(2));
    while (GetTextLine(in, line)) {
        if (line.empty()) continue;
        std::istringstream words(line);
        std::string name, second;
        words >> name >> second;
        if (second == "FAILED") {
            c.failed.push_back(line);
            continue;
        }
        const auto f = Fields(line);
        Capture k;
        k.magic = f.count("magic") ? f.at("magic") : "";
        k.blob_sha = f.count("blob") ? f.at("blob") : "";
        k.size = f.count("size") ? size_t(std::strtoull(f.at("size").c_str(), nullptr, 10)) : 0;
        if (f.count("tokens")) {
            std::istringstream t(f.at("tokens"));
            std::string tok;
            while (std::getline(t, tok, ','))
                if (!tok.empty()) k.tokens.push_back(int32_t(std::atoi(tok.c_str())));
        }
        c.cases[name] = k;
    }
    c.ok = true;
    return c;
}

// The pinned blob of one case, decoded and checked against its record's digest and size.
bool PinnedBlob(const pcsave::Case& c, const Capture& k, std::vector<uint8_t>* blob) {
    const std::string path = std::string(PC_REFERENCE_DIR) + "/pins/" + kTag + "_pkv_def_" + c.name + ".zrl";
    std::vector<uint8_t> enc;
    const bool read = ReadAll(path, &enc) && ZrlDecode(enc, blob);
    PKV_CHECK_MSG(read, "9.9 %s: pinned blob %s missing or malformed", c.name, path.c_str());
    if (!read) return false;
    const std::string sha = Sha(blob->data(), blob->size());
    PKV_CHECK_MSG(sha == k.blob_sha && blob->size() == k.size,
                  "9.9 %s: pin %s is %zu bytes, sha256 %s; its record has %zu, %s", c.name, path.c_str(),
                  blob->size(), sha.c_str(), k.size, k.blob_sha.c_str());
    PKV_CHECK_MSG(IsMagic(*blob, c.magic) && k.magic == c.magic, "9.9 %s: pinned magic is not %s", c.name,
                  c.magic);
    return sha == k.blob_sha && blob->size() == k.size && IsMagic(*blob, c.magic) && k.magic == c.magic;
}

// One restore path: the restored state re-saves to the pinned bytes and decodes the pinned continuation.
void GradeRestored(pcsave::Run& r, const pcsave::Case& c, const char* path, sslm_seq seq,
                   const std::vector<uint8_t>& pinned, const Capture& k) {
    std::vector<uint8_t> again;
    const bool saved = pcsave::SaveSeq(r, seq, &again);
    PKV_CHECK_MSG(saved, "9.9 %s via %s: re-save failed (%s)", c.name, path, r.fail.c_str());
    if (saved) {
        size_t first = 0;
        while (first < again.size() && first < pinned.size() && again[first] == pinned[first]) ++first;
        PKV_CHECK_MSG(again == pinned, "9.9 %s via %s: re-save is %zu bytes and differs from the 1.13.1 pin "
                      "(%zu bytes) from offset %zu", c.name, path, again.size(), pinned.size(), first);
    }
    const std::vector<int32_t> tokens = pcsave::Continue(r, seq, pcsave::kContinue);
    PKV_CHECK_MSG(tokens == k.tokens,
                  "9.9 %s via %s: the continuation differs from 1.13.1's pinned one", c.name, path);
}

void SeqSaves99() {
    const std::string ref = std::string(PC_REFERENCE_DIR) + "/" + kTag + "_pkv_def_seqsaves.ref";
    const CaptureFile cap = ReadCaptures(ref);
    PKV_CHECK_MSG(cap.ok, "9.9: the v1.13.1 sequence-save capture is not pinned: %s is absent (plan §9 E5 captures "
                  "it on a v1.13.1 build, d697927, with tools/build_precache_v1p13p1_captures.sh)", ref.c_str());
    if (!cap.ok) return;
    const auto header = [&](const char* key) {
        const auto it = cap.header.find(key);
        return it == cap.header.end() ? std::string() : it->second;
    };
    PKV_CHECK_MSG(header("writer") == kTag && header("commit") == kCommit,
                  "9.9: the capture was written by %s at %s, not %s at %s", header("writer").c_str(),
                  header("commit").c_str(), kTag, kCommit);
    for (const auto& line : cap.failed) PKV_CHECK_MSG(false, "9.9: the capture recorded a failure: %s", line.c_str());
    Mapped m = MapOwn(kPkvDef);
    if (!m.fx || !RequireRuleB(*m.fx, kPkvDef, false)) return;
    const Fixture& fx = *m.fx;
    PKV_CHECK_MSG(header("sha256") == fx.sha, "9.9: the capture's pkv_def is %s; this fixture is %s",
                  header("sha256").c_str(), fx.sha.c_str());
    PKV_CHECK_MSG(header("B") == std::to_string(fx.B()), "9.9: 1.13.1 captured at B = %s; this tree's rule is %lld",
                  header("B").c_str(), static_cast<long long>(fx.B()));
    if (header("sha256") != fx.sha || header("B") != std::to_string(fx.B())) return;
    PKV_CHECK_EQ(cap.cases.size(), pcsave::kCaseCount);
    for (const pcsave::Case& c : pcsave::kCases) {
        const auto it = cap.cases.find(c.name);
        PKV_CHECK_MSG(it != cap.cases.end(), "9.9: the capture has no record for %s", c.name);
        if (it == cap.cases.end()) continue;
        const Capture& k = it->second;
        PKV_CHECK_MSG(k.tokens.size() == size_t(pcsave::kContinue) && (c.schema || pcsave::AllTokens(k.tokens)),
                      "9.9 %s: 1.13.1's own continuation is %zu outcomes%s, not 64 tokens", c.name, k.tokens.size(),
                      pcsave::AllTokens(k.tokens) ? "" : " with a dead end or refusal");
        std::vector<uint8_t> pinned;
        if (!PinnedBlob(c, k, &pinned)) continue;
        PagePool pool(fx.model, pcsave::kPoolPages);
        PKV_CHECK_EQ(pool.status, SSLM_OK);
        if (pool.status != SSLM_OK) continue;
        pcsave::Run r;
        r.model = fx.model;
        r.pool = pool.pool;
        {
            sslm_seq seq = nullptr;
            Status(sslm_seq_restore(fx.model, &pool.pool, pinned.data(), pinned.size(), &seq), SSLM_OK,
                   "9.9 sslm_seq_restore of a 1.13.1 save");
            if (seq) {
                GradeRestored(r, c, "sslm_seq_restore", seq, pinned, k);
                Status(sslm_seq_release(seq), SSLM_OK, "9.9 release");
            }
        }
        if (std::strcmp(c.magic, "SSB6") != 0) continue;
        sslm_prefix parent = nullptr, child = nullptr;
        if (c.kind == pcsave::Kind::kNestedAdopter && !pcsave::NestedChain(r, c.schema, &parent, &child)) {
            PKV_CHECK_MSG(false, "9.9 %s: the hint's nested chain did not build here (%s)", c.name, r.fail.c_str());
            if (child) sslm_prefix_release(child);
            if (parent) sslm_prefix_release(parent);
            continue;
        }
        // The hint (the child, whose length is the blob's origin) shares its full pages: floor(origin / B).
        const int64_t origin = pcsave::kParentPrompt + pcsave::kChildPrompt;
        const uint32_t want_shared = c.kind == pcsave::Kind::kNestedAdopter ? uint32_t(origin / fx.B()) : 0u;
        sslm_seq seq = nullptr;
        uint32_t shared = 0xffffffffu;
        Status(sslm_seq_restore_shared(fx.model, &pool.pool, pinned.data(), pinned.size(), child, &seq, &shared),
               SSLM_OK, "9.9 sslm_seq_restore_shared of a 1.13.1 save");
        PKV_CHECK_MSG(shared == want_shared, "9.9 %s: restore_shared shared %u pages; floor(origin/B) is %u",
                      c.name, shared, want_shared);
        if (seq) {
            GradeRestored(r, c, "sslm_seq_restore_shared", seq, pinned, k);
            Status(sslm_seq_release(seq), SSLM_OK, "9.9 release shared");
        }
        if (child) sslm_prefix_release(child);
        if (parent) sslm_prefix_release(parent);
    }
}
PKV_CELL("9.9/E5", "E5", SeqSaves99);

}  // namespace
