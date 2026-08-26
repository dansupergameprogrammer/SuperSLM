// src/detail/int_hash.h
// A fixed-population, open-addressing associative container. No floating-point
// operation anywhere in this file: bucket-count sizing is a bit-shift, hashing is
// integer multiply/xor/shift, probing is integer addition modulo a power of two.
// Built ONCE from a known element count (every caller in this codebase knows its
// count before the first insert -- see tokenizer.cpp/model.cpp population sites);
// no resize, no rehash, no incremental growth. No erase -- no caller needs one
// (grep swept clean, T-2265 design Sec2.2).
//
// This file also carries GrowableIntSet (Sec3.5) and GrowableIntMap (Sec3.6) --
// two further, distinct constructions built from the same integer-only primitives
// (BucketCountFor's bit-shift sizing, MixKey64) for sites whose population is not
// known upfront: GrowableIntSet adds tombstone-based erase for the one site
// (g_live_seqs) that is both unbounded and erased; GrowableIntMap adds insert-only
// doubling growth with no tombstone machinery for the one site (ContextEntry::counts)
// that is unbounded but never erased.
#ifndef SUPERSLM_SRC_DETAIL_INT_HASH_H
#define SUPERSLM_SRC_DETAIL_INT_HASH_H

#include "superslm/intmath.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

// sslm_seq_s is the C-ABI opaque handle type sslm_abi.h:22 forward-declares
// (`typedef struct sslm_seq_s* sslm_seq;`, extern "C", global namespace). Forward-declared
// again here, at GLOBAL scope -- NOT inside superslm::detail below, since HashSeq's
// parameter type must name the SAME type sslm_abi.cpp's own g_live_seqs declaration
// (Sec3.5) names, and that type is ::sslm_seq_s, not superslm::detail::sslm_seq_s. A
// repeated forward declaration of an already-incomplete type in the same scope is a
// legal redeclaration, so this line carries no dependency on include order relative to
// sslm_abi.h.
struct sslm_seq_s;

namespace superslm::detail {

// Splitmix64 finalizer -- integer-only avalanche, used only to spread keys across
// buckets. Not a security hash; this table is never fed attacker-chosen keys where
// clustering is a threat model SuperSLM defends against (the model artifact is
// already trust-boundary-verified before these tables are populated).
inline uint64_t MixKey64(uint64_t k) {
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27; k *= 0x94d049bb133111ebULL;
    k ^= k >> 31;
    return k;
}
inline uint64_t MixKey32(uint32_t k) { return MixKey64(uint64_t(k)); }

// This wrapper exists solely to give the one signed-keyed site (ContextEntry::counts,
// site 10) a hash function of the exact required pointer type: GrowableIntMap's
// non-type template parameter has type uint64_t(*)(Key), and Key=int32_t requires
// uint64_t(*)(int32_t) exactly -- a non-type function-pointer template parameter does
// not convert across a signedness difference, so MixKey32 (uint64_t(*)(uint32_t))
// cannot bind there even though the two types share a common width.
// static_cast<uint32_t>(int32_t) is value-preserving reinterpretation under
// two's-complement (mandatory since C++20, [conv.integral]), so this changes zero
// mixing behavior relative to MixKey32 itself -- it is a type-system adapter, not a
// different hash.
inline uint64_t MixKeyS32(int32_t k) { return MixKey32(static_cast<uint32_t>(k)); }

// FixedIntSet<std::string_view, HashSV> instantiates sites 5-7 (seen_names). Pure-integer
// FNV-1a-64 over the view's bytes -- no float, no double, no floating-point operand of
// any kind, matching this design's own no-FP invariant by the same construction as
// every hash above.
inline uint64_t HashSV(std::string_view s) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a-64 offset basis
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ull;             // FNV-1a-64 prime
    }
    return h;
}

// HashSeq -- a pointer-key hash for site 8 (GrowableIntSet<sslm_seq_s*, HashSeq>).
// sslm_seq_s is forward-declared above, not defined, so this function never
// dereferences the pointer it hashes.
//
// Provenance and mixing: the key is the pointer VALUE, not anything it points to -- two
// live sslm_seq_s* handles are the same registry entry iff they are the same pointer,
// exactly std::unordered_set<sslm_seq_s*>'s own pointer-identity semantics, so a hash
// over the address bits is both the only input available and the only one needed. The
// bits are reinterpreted as a uint64_t (a pointer-to-integer conversion,
// [expr.reinterpret.cast], implementation-defined and never a floating-point or
// arithmetic operation) and passed through MixKey64 -- the SAME multiply-xor-shift
// avalanche finalizer every other hash in this file already uses, rather than a second
// mixing scheme invented for this one site. Reusing MixKey64's own full-bit-avalanche
// mix (every output bit a function of every input bit) rather than masking the address
// directly is what keeps bucket selection from clustering on the structural zero run
// every live heap pointer's low alignof(sslm_seq_s) bits carry.
inline uint64_t HashSeq(sslm_seq_s* p) {
    return MixKey64(static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(p)));
}

// bucket_count = next power of two >= 2*n, computed by Clz64 (already the engine's
// own portable bit-scan primitive, intmath.h) -- pure integer, no float divide, no
// float multiply, no float of any kind. Load factor is therefore always <= 0.5 by
// construction, which is what keeps expected-probe-length O(1)-class without ever
// touching max_load_factor() or any other float-typed STL knob.
//
// n == 0 is handled WITHOUT calling Clz64 at all: Clz64's own documented contract
// (intmath.h/intmath.cpp, "Domain [1, 2^64-1]; n == 0 is out of contract and never
// reached") is violated by n == 0 under the original need = (n<1?1:n*2) formula,
// which reduces to Clz64(0) -- a real, artifact-reachable input; several of the seven
// sites' counts are read from the artifact with no lower-bound check. For every
// n >= 1, need = n*2 >= 2, so Clz64(need-1) receives an argument >= 1, inside
// Clz64's contracted domain -- only n == 0 ever reached the violation, and this is
// the one branch that needed a separate return.
inline uint64_t BucketCountFor(uint64_t n) {
    if (n == 0) return 1;
    uint64_t need = n * 2;
    // Debug-mode precondition, encoding Clz64's own documented domain [1, 2^64-1] as a
    // local check rather than leaving it implicit: for every n >= 1, need = n*2 >= 2, so
    // need-1 >= 1, always inside Clz64's contract. No BEHAVIORAL check can force this
    // divergence: std::countl_zero(0) == 64 (C++20's defined value) makes the reverted
    // formula's OUTPUT byte-identical to this function's -- bits = 64-64 = 0, return
    // 1 << 0 = 1, matching the early return exactly.
    assert(need >= 2);
    int bits = 64 - superslm::Clz64(need - 1);   // ceil(log2(need)); need-1 >= 1
    return uint64_t(1) << bits;
}

template <typename Key, typename Value, uint64_t (*Hash)(Key)>
class FixedIntMap {
public:
    // Deferred-population state: a default-constructed FixedIntMap holds mask_ = 0, an
    // empty slots_, and ready_ = false. This is the state a member of `Impl` is in
    // between `std::make_unique<Impl>()` (no arguments -- the count is not known yet)
    // and the point the count is read and Init is called.
    FixedIntMap() = default;

    // capacity_hint: the EXACT final element count, known to every caller before
    // the first Insert. One allocation; never resized. Retained for any
    // site whose declaration context has the count in hand at declaration time
    // (function-locals) -- delegates to Init so both construction paths
    // share one sizing implementation.
    explicit FixedIntMap(uint64_t capacity_hint) { Init(capacity_hint); }

    // One-shot: sizes the table from a now-known capacity_hint and marks it ready
    // for Insert/Find/InsertOrAssign. Calling Init a second time, or
    // calling it after the capacity-taking constructor already ran, is a caller bug --
    // hardened release-safe (`std::abort()`, independent of NDEBUG), the identical
    // diagnosable-rather-than-silent shape the probe-exhaustion aborts below already
    // use for a capacity/population desync.
    void Init(uint64_t capacity_hint) {
        if (ready_) std::abort();  // Init runs exactly once -- release-safe
        mask_ = BucketCountFor(capacity_hint) - 1;
        slots_.assign(mask_ + 1, Slot{});
        size_ = 0;
        ready_ = true;
    }

    // Insert-only. size_ tracks live occupancy so the debug assert actually exists.
    // The probe is additionally bounded to slots_.size() steps and
    // hard-aborts on exhaustion -- release-safe, independent of NDEBUG -- so a
    // capacity_hint/population desync that reaches this code fails immediately and
    // diagnosably rather than looping forever with no exit code. Every existing
    // caller already bounds population by the artifact's own declared count.
    // A pre-Init call is a second, distinct misuse this design hardens against
    // release-safe, for the identical diagnosable-not-a-hang reason.
    void Insert(Key k, Value v) {
        if (!ready_) std::abort();  // Init() must run before any Insert -- release-safe
        assert(size_ < slots_.size());
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (!slots_[i].occupied) {
                slots_[i] = {k, std::move(v), true};
                ++size_;
                return;
            }
            i = (i + 1) & mask_;
        }
        std::abort();  // capacity_hint/population desync -- diagnosable, not a hang
    }

    // Returns nullptr on miss -- mirrors the *it/it==end() call-site pattern with
    // one dereference instead of two comparisons.
    //
    // Bounded and hard-aborting: every current call site's capacity_hint always
    // equals its actual insert count, which guarantees at least one empty slot
    // remains after population and this loop always terminates without ever
    // reaching the bound. That guarantee is unstated and unenforced in code,
    // though -- nothing asserts capacity_hint == insert count -- so Find is
    // hardened to the SAME bounded-probe-then-abort shape Insert/InsertUnique
    // (above) and GrowableIntSet::Contains/Erase already use, rather than left
    // as the one lookup in this file whose safety rests on an invariant no code
    // checks. Pre-Init hardening, same reason as Insert above.
    const Value* Find(Key k) const {
        if (!ready_) std::abort();  // Init() must run before any Find -- release-safe
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (!slots_[i].occupied) return nullptr;
            if (slots_[i].key == k) return &slots_[i].value;
            i = (i + 1) & mask_;
        }
        std::abort();  // capacity_hint/population desync -- diagnosable, not a hang
    }

    // InsertOrAssign -- overwrites an existing key's value in place, exactly
    // std::unordered_map::operator[]='s repeated-key semantics (last-wins). Insert()
    // above implements first-wins -- a repeated key occupies a SECOND probe-chain slot,
    // and Find returns whichever slot the probe reaches first, which is the
    // FIRST-inserted value. InsertOrAssign is presence-checked FIRST -- same probe Find
    // uses -- so a repeated key never occupies a second slot; only a genuinely new key
    // falls through to the placement branch. Same bounded-probe/hard-abort hardening as
    // Insert/Find above, for the identical capacity_hint/population-desync reason.
    // Pre-Init hardening, same reason.
    void InsertOrAssign(Key k, Value v) {
        if (!ready_) std::abort();  // Init() must run before any InsertOrAssign -- release-safe
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (slots_[i].occupied && slots_[i].key == k) {
                slots_[i].value = std::move(v);
                return;
            }
            if (!slots_[i].occupied) {
                assert(size_ < slots_.size());
                slots_[i] = {k, std::move(v), true};
                ++size_;
                return;
            }
            i = (i + 1) & mask_;
        }
        std::abort();  // capacity_hint/population desync -- diagnosable, not a hang
    }

private:
    struct Slot { Key key{}; Value value{}; bool occupied = false; };
    uint64_t mask_ = 0;
    uint64_t size_ = 0;
    bool ready_ = false;
    std::vector<Slot> slots_;
};

template <typename Key, uint64_t (*Hash)(Key)>
class FixedIntSet {
public:
    // Same deferred-population shape as FixedIntMap above.
    FixedIntSet() = default;

    explicit FixedIntSet(uint64_t capacity_hint) { Init(capacity_hint); }

    void Init(uint64_t capacity_hint) {
        if (ready_) std::abort();  // one-shot -- release-safe
        mask_ = BucketCountFor(capacity_hint) - 1;
        slots_.assign(mask_ + 1, Slot{});
        size_ = 0;
        ready_ = true;
    }

    // Returns false if k was ALREADY present (mirrors seen_names.insert(name).second
    // exactly), true if newly inserted. Same size_/assert/bounded-probe/
    // abort() shape as FixedIntMap::Insert above, for the identical reason. Pre-Init
    // hardening, same reason as FixedIntMap.
    bool InsertUnique(Key k) {
        if (!ready_) std::abort();  // Init() must run before any InsertUnique -- release-safe
        assert(size_ < slots_.size());
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (!slots_[i].occupied) {
                slots_[i] = {k, true};
                ++size_;
                return true;
            }
            if (slots_[i].key == k) return false;
            i = (i + 1) & mask_;
        }
        std::abort();  // capacity_hint/population desync -- diagnosable, not a hang
    }

private:
    struct Slot { Key key{}; bool occupied = false; };
    uint64_t mask_ = 0;
    uint64_t size_ = 0;
    bool ready_ = false;
    std::vector<Slot> slots_;
};

}  // namespace superslm::detail

// A growable, open-addressing associative container with tombstone-based erase. Bucket-
// count growth is a bit-shift doubling (BucketCountFor, unchanged above) -- no
// floating-point operation anywhere, same as FixedIntMap/FixedIntSet. Unlike those types,
// this one has no upfront population precondition: it is sized lazily and grows on demand,
// for the one site (g_live_seqs, site 8) whose population is genuinely unknown in
// advance and which is erased, not only inserted, over its lifetime.
namespace superslm::detail {

template <typename Key, uint64_t (*Hash)(Key)>
class GrowableIntSet {
public:
    // A small initial capacity -- most engine processes hold a handful of concurrent
    // sequences, not thousands; growth handles the tail without over-allocating the
    // common case. Never zero (BucketCountFor(1) rounds up to a valid power of two).
    explicit GrowableIntSet(uint64_t initial_capacity_hint = 8)
        : mask_(BucketCountFor(initial_capacity_hint ? initial_capacity_hint : 1) - 1),
          slots_(mask_ + 1) {}

    // Inserts k if not already present (mirrors g_live_seqs.insert(h) at sslm_seq_create).
    // Grows -- by doubling, an integer bit-shift, never a float divide -- before the load
    // factor (including tombstones, which count against capacity until reclaimed by a
    // fresh probe over them) would exceed 0.5, the same fixed ratio BucketCountFor already
    // bakes in for the fixed types.
    bool InsertOrReclaim(Key k) {
        if ((live_ + tombstones_ + 1) * 2 > slots_.size()) Grow();
        uint64_t i = Hash(k) & mask_;
        int64_t first_tombstone = -1;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (slots_[i].state == State::kOccupied) {
                if (slots_[i].key == k) return false;  // already present
            } else if (slots_[i].state == State::kTombstone) {
                if (first_tombstone < 0) first_tombstone = static_cast<int64_t>(i);
            } else {  // kEmpty -- end of this key's probe sequence
                uint64_t dst = first_tombstone >= 0 ? static_cast<uint64_t>(first_tombstone) : i;
                if (slots_[dst].state == State::kTombstone) --tombstones_;
                slots_[dst] = {k, State::kOccupied};
                ++live_;
                return true;
            }
            i = (i + 1) & mask_;
        }
        std::abort();  // unreachable: Grow() above always keeps an empty slot on the probe path
    }

    // Erases k if present (mirrors g_live_seqs.erase(seq) at sslm_seq_release). Marks a
    // tombstone rather than kEmpty -- a live probe sequence through this slot for a
    // DIFFERENT key must not stop here, or a still-present key placed past this slot by
    // a prior collision becomes unfindable.
    //
    // Two distinct exits, not one. Hitting kEmpty WITHIN the probe (the `return false`
    // inside the loop) is the ordinary, expected "key never present" result:
    // InsertOrReclaim always terminates a key's probe sequence at the first kEmpty slot
    // it finds (there is no reason to place k past one), so a present key's own probe
    // sequence never crosses a kEmpty slot, and absence is therefore correctly decided
    // the instant one is seen -- calling this a desync would reject a legitimate,
    // frequent, non-buggy call (erasing a key that was never inserted, or was already
    // erased). Exhausting all mask_+1 steps WITHOUT ever seeing a kEmpty slot is the
    // different case: InsertOrReclaim's own pre-check
    // (`(live_ + tombstones_ + 1) * 2 > slots_.size()`) guarantees at least one kEmpty
    // slot exists in the table at all times before this call runs, so a full sweep
    // finding none is the identical capacity/population invariant violation
    // Insert/Find/InsertOrAssign and InsertOrReclaim (above) already hard-abort on -- a
    // genuine bug, not a "not found." `std::abort()` on the mask_+1-exhausted path, same
    // as Insert/Find/InsertOrAssign/InsertOrReclaim, diagnosable rather than silently wrong.
    bool Erase(Key k) {
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (slots_[i].state == State::kEmpty) return false;  // never present -- ordinary miss
            if (slots_[i].state == State::kOccupied && slots_[i].key == k) {
                slots_[i].state = State::kTombstone;
                --live_; ++tombstones_;
                return true;
            }
            i = (i + 1) & mask_;
        }
        std::abort();  // capacity/population desync -- diagnosable, not a silent false
    }

    // Same two-exit reconciliation as Erase above, for the identical reason: kEmpty
    // within the probe is an ordinary "not present" miss; exhausting the full probe
    // without ever seeing kEmpty is the capacity/population desync every other lookup in
    // this file hard-aborts on.
    bool Contains(Key k) const {
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (slots_[i].state == State::kEmpty) return false;  // never present -- ordinary miss
            if (slots_[i].state == State::kOccupied && slots_[i].key == k) return true;
            i = (i + 1) & mask_;
        }
        std::abort();  // capacity/population desync -- diagnosable, not a silent false
    }

private:
    enum class State : uint8_t { kEmpty, kOccupied, kTombstone };
    struct Slot { Key key{}; State state = State::kEmpty; };

    void Grow() {
        std::vector<Slot> old = std::move(slots_);
        uint64_t new_mask = BucketCountFor((live_ + 1) * 2) - 1;  // integer doubling via
                                                                    // BucketCountFor's own
                                                                    // bit-shift sizing
        slots_.assign(new_mask + 1, Slot{});
        mask_ = new_mask;
        live_ = 0; tombstones_ = 0;
        for (auto& s : old) {
            if (s.state == State::kOccupied) InsertOrReclaim(s.key);  // rehash; drops tombstones
        }
    }

    uint64_t mask_;
    uint64_t live_ = 0;
    uint64_t tombstones_ = 0;
    std::vector<Slot> slots_;
};

}  // namespace superslm::detail

// A growable, open-addressing, INSERT-ONLY map keyed by an integer scalar, with
// get-or-default-and-mutate access. No erase -- AntiLmState::ContextEntry::counts is
// appended-to and incremented, never erased. Same integer-only bucket-count
// doubling (BucketCountFor, above) as GrowableIntSet; no tombstone machinery is
// needed, since nothing is ever removed.
namespace superslm::detail {

template <typename Key, typename Value, uint64_t (*Hash)(Key)>
class GrowableIntMap {
public:
    // NOT explicit. ContextEntry aggregate-initializes this member from an empty brace
    // list at two independent points: the shipped source's own value-init
    // (`AntiLmState::ContextEntry{}`) and this design's own `slots_[i] = {k, Value{}, true}`/
    // `FindOrEmplace`'s `Value{}` (context_hash.h). Both are copy-list-initialization of
    // the `counts` member from an empty initializer list ([dcl.init.aggr]p9), which
    // excludes an explicit default constructor from overload resolution
    // ([over.match.list]). No call site anywhere in this design or in the engine relies
    // on GrowableIntMap rejecting an implicit uint64_t -> GrowableIntMap conversion: it
    // is a new, internal-only type with no such call site to protect, so dropping
    // `explicit` here changes no ABI and no behavior -- it only widens which
    // initialization forms the compiler accepts for a type nothing outside this file
    // constructs.
    //
    // LAZY (fold round 25, T-2300/D-SLM4763, S1). Every prior constructor allocated
    // `slots_` unconditionally in its own member-initializer list -- the same shape
    // GrowableIntSet's own constructor still uses (unaffected; GrowableIntSet is not
    // touched by this fix). Because ContextEntry (damped_greedy_antilm.cpp) holds a
    // GrowableIntMap BY VALUE, and GrowableContextMap<ContextEntry>'s own Slot holds
    // its Value BY VALUE, every slot of the outer GrowableContextMap<ContextEntry> --
    // occupied AND empty, at least half of every table by the outer type's own <= 50%
    // load invariant -- default-constructed a live ContextEntry, which
    // default-constructed a live, 16-slot-allocated GrowableIntMap: every EMPTY slot of
    // the outer table paid a real heap allocation nothing had ever looked up. The
    // remedy makes ONLY this type lazy -- the outer GrowableContextMap is not touched
    // (a separate, larger change to its own construction, not taken here), and
    // ContextEntry's own `counts` member is not indirected (a pointer/unique_ptr member
    // would also remove the per-empty-slot cost, but changes ContextEntry's layout and
    // every aggregate-init call site above; not adopted unless this narrower fix proves
    // insufficient). `pending_hint_` records the hint and nothing allocates until the
    // first Insert -- via operator[] below, the only mutator this type has.
    GrowableIntMap(uint64_t initial_capacity_hint = 8)
        : pending_hint_(initial_capacity_hint ? initial_capacity_hint : 1) {}
    // mask_ = 0, live_ = 0, slots_ = {} (default member-initializers below) is the state
    // a default-constructed table is in until its first Insert -- a state this type
    // never had before this fold, since every prior constructor allocated
    // unconditionally. Get-or-default access has no natural point to defer TO other
    // than the first insert itself, so the deferred state is recognized structurally
    // (`slots_.empty()`) rather than by a second boolean flag.

    // Returns a reference to k's stored value, default-constructing Value{} and
    // inserting if k is not yet present -- exactly std::unordered_map::operator[]'s
    // own semantics, which is what entry.counts[token] += 1 (damped_greedy_antilm.cpp)
    // needs.
    Value& operator[](Key k) {
        if (slots_.empty()) {
            // First insert this table has ever received (fold round 25, S1): allocate
            // NOW, sized from the hint recorded at construction -- not from Grow()'s
            // own (live_+1)*2 doubling formula below, which ignores the hint entirely
            // and would always produce a 4-slot table regardless of what the caller
            // asked for. This is the only call to BucketCountFor over pending_hint_'s
            // own value; every allocation after this one goes through Grow(), exactly
            // as before this fold -- only the TIMING of the first allocation changed,
            // not its size (BucketCountFor(8) = 16, identical to the pre-fold-25
            // constructor's own eager result) or the growth curve after it.
            mask_ = BucketCountFor(pending_hint_) - 1;
            slots_.assign(mask_ + 1, Slot{});
        } else if ((live_ + 1) * 2 > slots_.size()) {
            Grow();
        }
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (!slots_[i].occupied) {
                slots_[i] = {k, Value{}, true};
                ++live_;
                return slots_[i].value;
            }
            if (slots_[i].key == k) return slots_[i].value;
            i = (i + 1) & mask_;
        }
        std::abort();  // unreachable: the allocate-on-first-insert branch and Grow()
                        // above always leave at least one empty slot on the probe path
    }

    // Non-mutating lookup -- mirrors entry.counts.find(token) != entry.counts.end().
    // A table that has never received an Insert (slots_.empty(), fold round 25, S1) is
    // a miss BY CONSTRUCTION and returns immediately, before touching mask_/slots_ at
    // all. This is the one place the lazy state is a genuinely new condition this
    // method did not have to handle before this fold: falling through to the probe
    // loop at mask_ == 0 with slots_ of size 0 would index slots_[Hash(k) & 0] into a
    // zero-length vector -- undefined behavior, not the correct "not present" answer --
    // rather than the miss this method has always correctly returned for every OTHER
    // unrepresented key.
    const Value* Find(Key k) const {
        if (slots_.empty()) return nullptr;
        uint64_t i = Hash(k) & mask_;
        for (uint64_t steps = 0; steps <= mask_; ++steps) {
            if (!slots_[i].occupied) return nullptr;
            if (slots_[i].key == k) return &slots_[i].value;
            i = (i + 1) & mask_;
        }
        return nullptr;
    }

private:
    struct Slot { Key key{}; Value value{}; bool occupied = false; };
    void Grow() {
        // Precondition (fold round 25, S1): slots_ is never empty on entry. operator[]
        // above routes the slots_.empty() case to the allocate-from-hint branch and
        // never falls through to this function in that state, so Grow()'s own doubling
        // formula ((live_+1)*2) is never asked to size the table's FIRST allocation --
        // it only ever re-sizes an already-populated one, exactly its pre-fold-25 role.
        assert(!slots_.empty());
        std::vector<Slot> old = std::move(slots_);
        uint64_t new_mask = BucketCountFor((live_ + 1) * 2) - 1;  // integer doubling,
        slots_.assign(new_mask + 1, Slot{});                       // BucketCountFor's
        mask_ = new_mask;                                          // own bit-shift
        live_ = 0;                                                 // sizing
        for (auto& s : old) if (s.occupied) (*this)[s.key] = std::move(s.value);
    }
    uint64_t pending_hint_;    // fold round 25, S1: the hint, held until the first
                               // Insert allocates from it; unused thereafter (Grow()'s
                               // own doubling formula does not consult it, unchanged
                               // from every prior fold -- only the FIRST allocation
                               // ever honored the hint, before and after this fold)
    uint64_t mask_ = 0;        // fold round 25: was always constructor-set before this
                               // fold; 0 is now a real, reachable pre-first-insert state
    uint64_t live_ = 0;
    std::vector<Slot> slots_;  // fold round 25: was always constructor-allocated before
                               // this fold; empty is now a real, reachable pre-first-
                               // insert state, checked explicitly by every method above
};

}  // namespace superslm::detail

#endif  // SUPERSLM_SRC_DETAIL_INT_HASH_H
