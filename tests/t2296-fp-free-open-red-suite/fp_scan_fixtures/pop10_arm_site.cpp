// T-2273 strike construction: SuperSLM's own site 1, verbatim in shape.
//
// src/tokenizer.cpp:162 at v1.2.1 declares
//     std::unordered_map<uint64_t, std::pair<int32_t,int32_t>> merges;
// and populates it with reserve()+emplace() -- the container whose bucket-array sizing
// multiplies by max_load_factor(), a float, and divides by it. That divide is the whole
// reason T-2265 exists. This TU is that container and that population loop, compiled for
// the ISA the 29-job matrix's macos-arm64 leg builds the CPU core library on.
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

extern "C" __declspec(dllexport)
void BuildMerges(const uint64_t* keys, const int32_t* a, const int32_t* b, size_t n,
                 std::pair<int32_t, int32_t>* out) {
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges;
    merges.reserve(n);                       // <-- the float divide (site 1)
    for (size_t i = 0; i < n; ++i)
        merges.emplace(keys[i], std::make_pair(a[i], b[i]));
    for (size_t i = 0; i < n; ++i)
        out[i] = merges[keys[i]];
}

extern "C" __declspec(dllexport)
size_t DedupNames(const std::string_view* names, size_t n) {
    std::unordered_set<std::string_view> seen_names;   // model.cpp site 5-7
    for (size_t i = 0; i < n; ++i) seen_names.insert(names[i]);
    return seen_names.size();
}

// A bare, unmistakable floating-point body -- nothing to do with containers. If the
// deciding instrument cannot reject THIS, it cannot reject anything.
extern "C" __declspec(dllexport)
double BodyDivide(double x, double y) { return x / y + x * y - x; }

extern "C" __declspec(dllexport)
int BodyConvertCompare(long long n) {
    double d = static_cast<double>(n) / 3.7;
    return d > 1.5 ? 1 : 0;
}
