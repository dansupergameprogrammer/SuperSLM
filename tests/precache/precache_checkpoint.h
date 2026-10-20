#pragma once
// Per-case checkpointing for long precache grid cells.
//
// A grid cell wraps each case in RunCase. With SUPERSLM_PRECACHE_CHECKPOINT_DIR
// unset and SUPERSLM_PRECACHE_CASES unset, RunCase only calls the case body:
// the cell's checks, failures, output and verdict are unchanged.
//
// With SUPERSLM_PRECACHE_CHECKPOINT_DIR set, each completed case appends one
// record to a per-cell journal and forces it to disk before the next case
// starts. The record binds the SHA-256 of the running executable, the cell id,
// the case id and the SHA-256 of every fixture the case ran against, and holds
// the case's check count, failure count, every failure line exactly as written
// to stderr, and whether the case stopped its cell. A rerun of the same binary
// replays a matching record instead of running the case: it adds the recorded
// counts to the run totals and rewrites the recorded failure lines, so the
// cell's totals and verdict are those of an uninterrupted run. A record from
// another binary or against other fixture bytes never matches; journals are
// also named by the binary hash. A torn or corrupt record is ignored and its
// case runs again.
//
// Cases that share a pool across iterations pass a CaseScope naming those
// pools. Such a case is replayed only while every earlier case of the scope
// was replayed and each recorded case left every scope pool with the free and
// shared page counts it found (read through sslm_kv_pool_stats, never as a
// check). Once one case runs live, the rest of the scope runs live, so later
// cases see the same pool history as in an uninterrupted run.
//
// SUPERSLM_PRECACHE_CASES selects cases by comma-separated id prefix. A case it
// does not select is neither run nor recorded and is named on stdout; such a
// run is not a full-cell verdict.
//
// RunCase must wrap only bodies that read no once-per-process cached state
// whose first use counts checks (GetFixture, SyntheticFixture): replaying the
// case that first touched the cache would move those checks to a later case.

#include "precache_common.h"
#include "superslm/sha256.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#endif

namespace pc {

enum class CaseEnd { Next, StopCell };

// The pools a run of cases shares. One scope per pool lifetime.
struct CaseScope {
    std::vector<sslm_kv_pool> pools;
    bool live = false;  // a case of this scope has run live in this process
};

namespace checkpoint_detail {

[[noreturn]] inline void Refuse(const std::string& why) {
    std::fflush(stdout);
    std::fprintf(stderr, "precache checkpoint refused: %s\n", why.c_str());
    std::fflush(stderr);
    std::exit(2);
}

inline std::string ExecutablePath() {
#if defined(_WIN32)
    char* path = nullptr;
    if (_get_pgmptr(&path) == 0 && path && *path) return path;
    return {};
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string path(size, '\0');
    if (_NSGetExecutablePath(path.data(), &size) != 0) return {};
    path.resize(std::strlen(path.c_str()));
    return path;
#else
    std::string path(4096, '\0');
    const ssize_t n = ::readlink("/proc/self/exe", path.data(), path.size());
    if (n <= 0 || size_t(n) >= path.size()) return {};
    path.resize(size_t(n));
    return path;
#endif
}

inline std::string FileSha(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    superslm::Sha256 hash;
    std::vector<char> buffer(size_t(1) << 20);
    while (in) {
        in.read(buffer.data(), std::streamsize(buffer.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) hash.Update(reinterpret_cast<const uint8_t*>(buffer.data()), size_t(got));
    }
    if (!in.eof()) return {};
    uint8_t digest[32];
    hash.Final(digest);
    return Hex(digest, 32);
}

inline std::string Escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '\t') out += "\\t";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out.push_back(c);
    }
    return out;
}

inline bool Unescape(const std::string& s, std::string* out) {
    out->clear();
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\') { out->push_back(s[i]); continue; }
        if (++i == s.size()) return false;
        if (s[i] == '\\') out->push_back('\\');
        else if (s[i] == 't') out->push_back('\t');
        else if (s[i] == 'n') out->push_back('\n');
        else if (s[i] == 'r') out->push_back('\r');
        else return false;
    }
    return true;
}

inline bool ParseCount(const std::string& s, int* out) {
    int value = -1;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec != std::errc() || end != s.data() + s.size() || value < 0) return false;
    *out = value;
    return true;
}

inline std::string StringSha(const std::string& s) {
    uint8_t digest[32];
    superslm::Sha256Hash(reinterpret_cast<const uint8_t*>(s.data()), s.size(), digest);
    return Hex(digest, 32);
}

struct Settings {
    bool checkpoint = false;
    std::string dir;
    std::string binary_path;
    std::string binary_sha;
    bool filtered = false;
    std::vector<std::string> case_prefixes;
};

inline const Settings& Config() {
    static const Settings settings = [] {
        Settings s;
        if (const char* cases = std::getenv("SUPERSLM_PRECACHE_CASES"); cases && *cases) {
            s.filtered = true;
            std::string item;
            for (const char* p = cases;; ++p) {
                if (*p == ',' || *p == '\0') {
                    if (!item.empty()) s.case_prefixes.push_back(item);
                    item.clear();
                    if (*p == '\0') break;
                } else {
                    item.push_back(*p);
                }
            }
            if (s.case_prefixes.empty()) Refuse("SUPERSLM_PRECACHE_CASES names no case prefix");
            std::printf("precache cases: SUPERSLM_PRECACHE_CASES=%s selects a subset; "
                        "this run is not a full-cell verdict\n", cases);
        }
        const char* dir = std::getenv("SUPERSLM_PRECACHE_CHECKPOINT_DIR");
        if (!dir || !*dir) {
            std::fflush(stdout);
            return s;
        }
        s.checkpoint = true;
        s.dir = dir;
        std::error_code ec;
        std::filesystem::create_directories(s.dir, ec);
        if (!std::filesystem::is_directory(s.dir, ec))
            Refuse("checkpoint directory " + s.dir + " cannot be created");
        s.binary_path = ExecutablePath();
        if (s.binary_path.empty()) Refuse("the running executable's path is unavailable");
        s.binary_sha = FileSha(s.binary_path);
        if (s.binary_sha.size() != 64) Refuse("cannot hash the running executable " + s.binary_path);
        std::printf("precache checkpoint: binary %s sha256 %s; journals in %s\n",
                    s.binary_path.c_str(), s.binary_sha.c_str(), s.dir.c_str());
        std::fflush(stdout);
        return s;
    }();
    return settings;
}

inline bool Selected(const Settings& s, const std::string& case_id) {
    for (const std::string& prefix : s.case_prefixes)
        if (case_id.compare(0, prefix.size(), prefix) == 0) return true;
    return false;
}

enum class PoolWitness { None, Restored, Changed };

inline const char* WitnessName(PoolWitness w) {
    return w == PoolWitness::None ? "none" : w == PoolWitness::Restored ? "restored" : "changed";
}

struct CaseRecord {
    int checks = 0;
    int failures = 0;
    CaseEnd end = CaseEnd::Next;
    PoolWitness pools = PoolWitness::None;
    std::vector<std::string> lines;
};

struct Journal {
    std::string path;
    std::map<std::string, CaseRecord> done;  // key: case id, 0x1f, fixture SHA-256 list
    std::set<std::string> seen;              // case ids run or replayed in this process
    bool torn_tail = false;                  // the file ends inside a record
};

// One record per line, tab-separated, every free-text field escaped:
// PCJ1 binary cell case fixtures checks failures end pools n line_1..line_n sha
// where sha is the SHA-256 of everything before its own tab.
inline std::string FormatRecord(const std::string& binary, const std::string& cell,
                                const std::string& case_id, const std::string& fixtures,
                                const CaseRecord& r) {
    std::string body = "PCJ1\t" + binary + "\t" + Escape(cell) + "\t" + Escape(case_id) + "\t" +
                       Escape(fixtures) + "\t" + std::to_string(r.checks) + "\t" +
                       std::to_string(r.failures) + "\t" +
                       (r.end == CaseEnd::StopCell ? "stop" : "next") + "\t" +
                       WitnessName(r.pools) + "\t" + std::to_string(r.lines.size());
    for (const std::string& line : r.lines) body += "\t" + Escape(line);
    return body + "\t" + StringSha(body) + "\n";
}

inline bool ParseRecord(const std::string& line, const std::string& binary,
                        const std::string& cell, std::string* key, CaseRecord* r) {
    const size_t last_tab = line.rfind('\t');
    if (last_tab == std::string::npos) return false;
    if (StringSha(line.substr(0, last_tab)) != line.substr(last_tab + 1)) return false;
    std::vector<std::string> f;
    size_t start = 0;
    for (;;) {
        const size_t tab = line.find('\t', start);
        f.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
        if (tab == std::string::npos) break;
        start = tab + 1;
    }
    int n = 0;
    if (f.size() < 11 || f[0] != "PCJ1" || !ParseCount(f[9], &n) || f.size() != size_t(n) + 11)
        return false;
    std::string record_cell, case_id, fixtures;
    if (f[1] != binary || !Unescape(f[2], &record_cell) || record_cell != cell ||
        !Unescape(f[3], &case_id) || !Unescape(f[4], &fixtures) ||
        !ParseCount(f[5], &r->checks) || !ParseCount(f[6], &r->failures) || r->failures != n)
        return false;
    if (f[7] == "next") r->end = CaseEnd::Next;
    else if (f[7] == "stop") r->end = CaseEnd::StopCell;
    else return false;
    if (f[8] == "none") r->pools = PoolWitness::None;
    else if (f[8] == "restored") r->pools = PoolWitness::Restored;
    else if (f[8] == "changed") r->pools = PoolWitness::Changed;
    else return false;
    r->lines.resize(size_t(n));
    for (int i = 0; i < n; ++i)
        if (!Unescape(f[size_t(10 + i)], &r->lines[size_t(i)])) return false;
    *key = case_id + '\x1f' + fixtures;
    return true;
}

inline std::string JournalName(const std::string& cell, const std::string& binary) {
    std::string name;
    for (char c : cell) {
        const bool plain = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                           (c >= '0' && c <= '9') || c == '.' || c == '-';
        name.push_back(plain ? c : '_');
    }
    return name + "." + binary.substr(0, 16) + ".pcj";
}

inline Journal& JournalFor(const Settings& s, const std::string& cell) {
    static std::map<std::string, Journal> journals;
    auto found = journals.find(cell);
    if (found != journals.end()) return found->second;
    Journal& j = journals[cell];
    j.path = (std::filesystem::path(s.dir) / JournalName(cell, s.binary_sha)).string();
    int loaded = 0, ignored = 0;
    std::ifstream in(j.path, std::ios::binary);
    if (in) {
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        size_t start = 0;
        while (start < text.size()) {
            const size_t end = text.find('\n', start);
            if (end == std::string::npos) {  // torn final record
                ++ignored;
                j.torn_tail = true;
                break;
            }
            std::string key;
            CaseRecord r;
            if (ParseRecord(text.substr(start, end - start), s.binary_sha, cell, &key, &r)) {
                if (j.done.emplace(key, std::move(r)).second) ++loaded;
            } else {
                ++ignored;
            }
            start = end + 1;
        }
    }
    std::printf("precache checkpoint: [%s] journal %s: %d completed case(s) loaded, %d record(s) ignored\n",
                cell.c_str(), j.path.c_str(), loaded, ignored);
    std::fflush(stdout);
    return j;
}

inline void AppendDurably(const std::string& path, const std::string& line) {
    bool ok = false;
#if defined(_WIN32)
    int fd = -1;
    if (_sopen_s(&fd, path.c_str(), _O_WRONLY | _O_APPEND | _O_CREAT | _O_BINARY, _SH_DENYNO,
                 _S_IREAD | _S_IWRITE) == 0 && fd >= 0) {
        ok = _write(fd, line.data(), unsigned(line.size())) == int(line.size()) && _commit(fd) == 0;
        _close(fd);
    }
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd >= 0) {
        ok = ::write(fd, line.data(), line.size()) == ssize_t(line.size()) && ::fsync(fd) == 0;
        ::close(fd);
    }
    // The directory entry of a newly created journal must reach disk as well.
    const std::string dir = std::filesystem::path(path).parent_path().string();
    const int dfd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY);
    if (dfd >= 0) {
        ok = ok && ::fsync(dfd) == 0;
        ::close(dfd);
    } else {
        ok = false;
    }
#endif
    if (!ok) Refuse("cannot durably append a case record to " + path);
}

// Free and shared page counts of each scope pool; empty when any read fails.
inline std::vector<uint32_t> PoolCounts(const CaseScope& scope) {
    std::vector<uint32_t> counts;
    for (sslm_kv_pool pool : scope.pools) {
        sslm_kv_pool_stats_out stats{};
        stats.struct_size = sizeof(stats);
        if (!pool || sslm_kv_pool_stats(pool, &stats) != SSLM_OK) return {};
        counts.push_back(stats.free_pages);
        counts.push_back(stats.shared_pages);
    }
    return counts;
}

}  // namespace checkpoint_detail

// Runs one case of the current cell. case_id is unique within the cell;
// fixtures are every model the case maps or decodes; scope is null for a case
// that shares no pool with another case.
inline CaseEnd RunCase(const std::string& case_id, std::initializer_list<const Fixture*> fixtures,
                       CaseScope* scope, const std::function<CaseEnd()>& body) {
    using namespace checkpoint_detail;
    const Settings& cfg = Config();
    pkv::RunState& state = pkv::State();
    if (cfg.filtered && !Selected(cfg, case_id)) {
        std::printf("[case] %s %s: not selected by SUPERSLM_PRECACHE_CASES\n", state.cell, case_id.c_str());
        std::fflush(stdout);
        return CaseEnd::Next;
    }
    if (!cfg.checkpoint) return body();

    std::string identity;
    for (const Fixture* fx : fixtures) {
        if (!fx || fx->sha.size() != 64) Refuse("case " + case_id + " names a fixture without a SHA-256");
        identity += (identity.empty() ? "" : ",") + fx->sha;
    }
    Journal& journal = JournalFor(cfg, state.cell);
    if (!journal.seen.insert(case_id).second)
        Refuse("case id " + case_id + " repeats within cell " + state.cell);
    const std::string key = case_id + '\x1f' + identity;
    const auto found = journal.done.find(key);
    const bool replayable = found != journal.done.end() &&
        (!scope || (!scope->live && found->second.pools == PoolWitness::Restored));
    if (replayable) {
        const CaseRecord& r = found->second;
        state.checks += r.checks;
        state.failures += r.failures;
        state.cell_failures += r.failures;
        for (const std::string& line : r.lines) {
            std::fputs(line.c_str(), stderr);
            std::fputc('\n', stderr);
        }
        std::fflush(stderr);
        std::printf("[case] %s %s: replayed from checkpoint (%d checks, %d failed)\n",
                    state.cell, case_id.c_str(), r.checks, r.failures);
        std::fflush(stdout);
        return r.end;
    }

    if (state.failure_lines) Refuse("case " + case_id + " is nested inside another checkpointed case");
    if (scope) scope->live = true;
    const std::vector<uint32_t> before = scope ? PoolCounts(*scope) : std::vector<uint32_t>{};
    CaseRecord r;
    const int checks0 = state.checks, failures0 = state.failures;
    state.failure_lines = &r.lines;
    r.end = body();
    state.failure_lines = nullptr;
    r.checks = state.checks - checks0;
    r.failures = state.failures - failures0;
    if (scope) {
        const std::vector<uint32_t> after = PoolCounts(*scope);
        r.pools = !before.empty() && before == after ? PoolWitness::Restored : PoolWitness::Changed;
    }
    if (size_t(r.failures) != r.lines.size())
        Refuse("case " + case_id + " counted a failure that Fail did not write");
    if (found == journal.done.end()) {
        // A torn tail is terminated first, so this record starts its own line.
        AppendDurably(journal.path, (journal.torn_tail ? "\n" : "") +
                      FormatRecord(cfg.binary_sha, state.cell, case_id, identity, r));
        journal.torn_tail = false;
        std::printf("[case] %s %s: recorded (%d checks, %d failed, pools %s)\n", state.cell,
                    case_id.c_str(), r.checks, r.failures, WitnessName(r.pools));
        journal.done.emplace(key, r);
    } else {
        const CaseRecord& old = found->second;
        const bool same = old.checks == r.checks && old.failures == r.failures &&
                          old.end == r.end && old.lines == r.lines;
        std::printf("[case] %s %s: ran live because its pool scope cannot be replayed past "
                    "this point; journal record %s\n", state.cell, case_id.c_str(),
                    same ? "matches" : "DIFFERS (the journal keeps its first record)");
    }
    std::fflush(stdout);
    return r.end;
}

}  // namespace pc
