// src/detail/context_hash.h (damped_greedy_antilm.cpp's own transparent-lookup
// container) -- a growable, open-addressing, INSERT-ONLY map keyed by a
// variable-length int32 sequence, with heterogeneous ("transparent") lookup by a
// non-owning view. No erase -- AntiLmState::tables_ is appended-to, never erased.
//
// BucketCountFor and the Slot/Grow shape below are int_hash.h's own.
// #include "int_hash.h" (no path prefix -- the two files sit in the same
// src/detail/ directory, the identical same-directory quote-include convention
// src/sslm_abi.cpp already uses for its own sibling src/bad_alloc_wrap.h).
#ifndef SUPERSLM_SRC_DETAIL_CONTEXT_HASH_H
#define SUPERSLM_SRC_DETAIL_CONTEXT_HASH_H

#include "int_hash.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace superslm::detail {

struct ContextView { const int32_t* data; std::size_t size; };

// Identical mixing to the original VecHash (damped_greedy_antilm.cpp) -- FNV-1a-64,
// integer-only already; this design does not change the hash, only the bucket-array
// sizing/growth machinery around it.
inline uint64_t HashContext(ContextView v) {
    uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < v.size; ++i) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(v.data[i]));
        h *= 1099511628211ull;
    }
    return h;
}

template <typename Value>
class GrowableContextMap {
public:
    explicit GrowableContextMap(uint64_t initial_capacity_hint = 8)
        : mask_(BucketCountFor(initial_capacity_hint ? initial_capacity_hint : 1) - 1),
          slots_(mask_ + 1) {}

    // Non-allocating lookup by view -- mirrors table.find(ctx_view) exactly, called
    // on every order for every token in both AntiLmUpdate and AntiLmPenalize. Non-const
    // overload, for a caller (AntiLmUpdate) holding a non-const table.
    Value* Find(ContextView v) {
        return const_cast<Value*>(FindConst(v));
    }

    // Const overload: AntiLmPenalize takes `const AntiLmState* state` and binds
    // `tables_` through `const auto& table` at both its own read call sites -- a const
    // lookup on a const table is exactly the read context every one of AntiLmPenalize's
    // two call sites uses, and with no const overload the non-const Find above cannot
    // be called through it at all. Both overloads share one implementation (FindConst)
    // so there is exactly one lookup body to keep correct.
    const Value* Find(ContextView v) const {
        return FindConst(v);
    }

    // Allocates and inserts a persistent copy of v ONLY when v is genuinely new --
    // mirrors "a vector is allocated only when a genuinely new context becomes
    // persistent table state" (damped_greedy_antilm.cpp's own comment).
    Value& FindOrEmplace(ContextView v) {
        if ((live_ + 1) * 2 > slots_.size()) Grow();
        uint64_t i = HashContext(v) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (!slots_[i].occupied) {
                slots_[i] = {std::vector<int32_t>(v.data, v.data + v.size), Value{}, true};
                ++live_;
                return slots_[i].value;
            }
            if (Eq(slots_[i].key, v)) return slots_[i].value;
            i = (i + 1) & mask_;
        }
        std::abort();  // unreachable: Grow() below always keeps an empty slot on the probe path
    }

private:
    struct Slot { std::vector<int32_t> key; Value value{}; bool occupied = false; };
    // Shared by both Find overloads above -- exactly one lookup body.
    const Value* FindConst(ContextView v) const {
        uint64_t i = HashContext(v) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (!slots_[i].occupied) return nullptr;
            if (Eq(slots_[i].key, v)) return &slots_[i].value;
            i = (i + 1) & mask_;
        }
        return nullptr;
    }
    static bool Eq(const std::vector<int32_t>& k, ContextView v) {
        return k.size() == v.size && (v.size == 0 || std::equal(k.begin(), k.end(), v.data));
    }
    void Grow() {
        std::vector<Slot> old = std::move(slots_);
        uint64_t new_mask = BucketCountFor((live_ + 1) * 2) - 1;
        slots_.assign(new_mask + 1, Slot{});
        mask_ = new_mask;
        live_ = 0;
        for (auto& s : old) if (s.occupied)
            FindOrEmplace(ContextView{s.key.data(), s.key.size()}) = std::move(s.value);
    }
    uint64_t mask_;
    uint64_t live_ = 0;
    std::vector<Slot> slots_;
};

}  // namespace superslm::detail

#endif  // SUPERSLM_SRC_DETAIL_CONTEXT_HASH_H
