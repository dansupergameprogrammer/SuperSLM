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
#include <cassert>
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
    // LAZY (fold round 27, T-2306 Sec8.1, adopted). Every prior fold's constructor
    // allocated slots_ unconditionally -- T-2306 measured 93.1%-97.2% of AntiLmCreate's
    // own whole construction-time footprint (zero tokens, no order ever queried) is
    // this table's own eager, unconditional per-order allocation, fired once per order
    // regardless of whether that order is ever used. This is the identical
    // population-independent lever GrowableIntMap's own fold round 25 fix already
    // applies one level in: deferring the allocation changes only its TIMING for any
    // table eventually touched, and eliminates it entirely for a table constructed and
    // never queried before the run ends. pending_hint_ records the hint and nothing
    // allocates until the first FindOrEmplace.
    explicit GrowableContextMap(uint64_t initial_capacity_hint = 8)
        : pending_hint_(initial_capacity_hint ? initial_capacity_hint : 1) {}
    // mask_ = 0, live_ = 0, slots_ = {} (default member-initializers below) is the
    // identical deferred-population shape GrowableIntMap's own fold-round-25
    // constructor established, restated here for this type's own mask_/slots_ pair;
    // this type has no tombstone member to defer alongside them.

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
        if (slots_.empty()) {
            // First insert this table has ever received (fold round 27, mirrors
            // GrowableIntMap::operator[]): allocate NOW, sized from the hint recorded
            // at construction, never from Grow()'s own BucketCountFor(live_+1) formula
            // below, which ignores the hint entirely.
            mask_ = BucketCountFor(pending_hint_) - 1;
            slots_.assign(mask_ + 1, Slot{});
        } else if ((live_ + 1) * 2 > slots_.size()) {
            Grow();
        }
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
        std::abort();  // unreachable: the allocate-on-first-insert branch and Grow() above
                        // always leave at least one empty slot on the probe path
    }

private:
    struct Slot { std::vector<int32_t> key; Value value{}; bool occupied = false; };
    // Shared by both Find overloads above -- exactly one lookup body.
    const Value* FindConst(ContextView v) const {
        if (slots_.empty()) return nullptr;  // fold round 27: never-touched table, a miss BY
                                              // CONSTRUCTION -- mirrors GrowableIntMap::Find;
                                              // falling through would index slots_[i] into a
                                              // zero-length vector, undefined behavior
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
        // Precondition (fold round 27, mirrors GrowableIntMap::Grow()'s own): slots_ is
        // never empty on entry. FindOrEmplace above routes the slots_.empty() case to
        // the allocate-from-hint branch and never falls through to this function in
        // that state, so this formula is never asked to size the table's FIRST
        // allocation.
        assert(!slots_.empty());
        std::vector<Slot> old = std::move(slots_);
        // Corrected fold round 26 (D-SLM4767/D-SLM4768), identical defect and identical
        // fix as GrowableIntMap::Grow() (Sec3.6, full derivation and termination proof
        // there): BucketCountFor already bakes in the <=50% headroom (Sec3.1); the
        // pre-fold-26 line here read BucketCountFor((live_+1)*2), doubling an already-
        // doubled argument (a quadrupling, uncommented at this call site through fold
        // round 26). live_ is read before the reset two lines down -- still the
        // pre-Grow() live count -- and this type never erases, so there is no tombstone
        // term to account for, unlike Sec3.5's GrowableIntSet::Grow().
        uint64_t new_mask = BucketCountFor(live_ + 1) - 1;
        slots_.assign(new_mask + 1, Slot{});
        mask_ = new_mask;
        live_ = 0;
        for (auto& s : old) if (s.occupied)
            FindOrEmplace(ContextView{s.key.data(), s.key.size()}) = std::move(s.value);
    }
    uint64_t pending_hint_;    // fold round 27: the hint, held until the first
                               // FindOrEmplace allocates from it; unused thereafter
    uint64_t mask_ = 0;        // fold round 27: was always constructor-set before this
                               // fold; 0 is now a real, reachable pre-first-insert state
    uint64_t live_ = 0;
    std::vector<Slot> slots_;  // fold round 27: was always constructor-allocated before
                               // this fold; empty is now a real, reachable pre-first-
                               // insert state, checked explicitly by FindConst/
                               // FindOrEmplace/Grow() above
};

}  // namespace superslm::detail

#endif  // SUPERSLM_SRC_DETAIL_CONTEXT_HASH_H
