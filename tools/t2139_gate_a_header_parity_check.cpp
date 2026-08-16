// t2139_gate_a_header_parity_check.cpp -- Gate A (Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec9): declaration-compatibility, checked by the
// compiler with both declarations in one translation unit. MUST-ACCEPT construction: never
// linked, compile-only. Covers the whole of include/superslm/sslm_abi_functions_g5_comparable.inc
// -- every C1/C2 verb this design's own header shares a name with tests/t2130-g5-red-suite/
// sslm_g5.h@760b12b -- from C1 forward (design Sec9: "Gate A has no reason to exempt any slot").
//
// Construction (design Sec9, literal): include tools/t2139_sslm_g5_ref.h (the promoted copy of
// sslm_g5.h) ONCE, unmodified, at its real names -- the one definition of every shared opaque
// handle and sslm_status in this TU. Then, for each verb, macro-rename this library's own
// declaration to a distinct identifier before including a functions-only extract (no type
// redefinitions -- reuses the types t2139_sslm_g5_ref.h already made visible), then
// static_assert(is_same_v<decltype(&real_name), decltype(&renamed_impl)>) per verb. The
// "functions-only extract" is include/superslm/sslm_abi_functions_g5_comparable.inc itself --
// the SAME file sslm_abi.h includes to declare its own real surface, not a hand-copied stand-in
// (see that file's own header comment).
#include <type_traits>

#include "t2139_sslm_g5_ref.h"

#define sslm_model_map          t2139_gate_a_lib_sslm_model_map
#define sslm_model_unmap        t2139_gate_a_lib_sslm_model_unmap
#define sslm_seq_state_size     t2139_gate_a_lib_sslm_seq_state_size
#define sslm_prefix_begin       t2139_gate_a_lib_sslm_prefix_begin
#define sslm_prefix_prefill     t2139_gate_a_lib_sslm_prefix_prefill
#define sslm_prefix_freeze      t2139_gate_a_lib_sslm_prefix_freeze
#define sslm_prefix_release     t2139_gate_a_lib_sslm_prefix_release
#define sslm_seq_create         t2139_gate_a_lib_sslm_seq_create
#define sslm_seq_release        t2139_gate_a_lib_sslm_seq_release
#define sslm_seq_reset          t2139_gate_a_lib_sslm_seq_reset
#define sslm_seq_adopt_prefix   t2139_gate_a_lib_sslm_seq_adopt_prefix
#define sslm_seq_save           t2139_gate_a_lib_sslm_seq_save
#define sslm_seq_restore        t2139_gate_a_lib_sslm_seq_restore
#define sslm_seq_set_adapter    t2139_gate_a_lib_sslm_seq_set_adapter
#define sslm_decode_step        t2139_gate_a_lib_sslm_decode_step
#define sslm_stats              t2139_gate_a_lib_sslm_stats
// Every other verb in the g5-comparable .inc (sslm_kv_block_size, sslm_kv_pool_overhead_size,
// sslm_kv_pool_create/_destroy, sslm_workspace_destroy, sslm_adapter_map/_release/_residency,
// sslm_prefill, sslm_tokenize) has NO counterpart in t2139_sslm_g5_ref.h by this design's own
// invention (Sec4/Sec7) -- renaming them too is harmless (nothing to collide with) but adds no
// checkable assertion, so they are left un-renamed and simply declared as this library's own
// symbols alongside t2139_sslm_g5_ref.h's real ones (no name collision, since none of those
// names appear in t2139_sslm_g5_ref.h at all).

extern "C" {
#include "superslm/sslm_abi_functions_g5_comparable.inc"
}

#undef sslm_model_map
#undef sslm_model_unmap
#undef sslm_seq_state_size
#undef sslm_prefix_begin
#undef sslm_prefix_prefill
#undef sslm_prefix_freeze
#undef sslm_prefix_release
#undef sslm_seq_create
#undef sslm_seq_release
#undef sslm_seq_reset
#undef sslm_seq_adopt_prefix
#undef sslm_seq_save
#undef sslm_seq_restore
#undef sslm_seq_set_adapter
#undef sslm_decode_step
#undef sslm_stats

static_assert(std::is_same_v<decltype(&sslm_model_map), decltype(&t2139_gate_a_lib_sslm_model_map)>,
              "sslm_model_map: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_model_unmap), decltype(&t2139_gate_a_lib_sslm_model_unmap)>,
              "sslm_model_unmap: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_state_size), decltype(&t2139_gate_a_lib_sslm_seq_state_size)>,
              "sslm_seq_state_size: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_prefix_begin), decltype(&t2139_gate_a_lib_sslm_prefix_begin)>,
              "sslm_prefix_begin: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_prefix_prefill), decltype(&t2139_gate_a_lib_sslm_prefix_prefill)>,
              "sslm_prefix_prefill: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_prefix_freeze), decltype(&t2139_gate_a_lib_sslm_prefix_freeze)>,
              "sslm_prefix_freeze: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_prefix_release), decltype(&t2139_gate_a_lib_sslm_prefix_release)>,
              "sslm_prefix_release: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_create), decltype(&t2139_gate_a_lib_sslm_seq_create)>,
              "sslm_seq_create: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_release), decltype(&t2139_gate_a_lib_sslm_seq_release)>,
              "sslm_seq_release: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_reset), decltype(&t2139_gate_a_lib_sslm_seq_reset)>,
              "sslm_seq_reset: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_adopt_prefix), decltype(&t2139_gate_a_lib_sslm_seq_adopt_prefix)>,
              "sslm_seq_adopt_prefix: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_save), decltype(&t2139_gate_a_lib_sslm_seq_save)>,
              "sslm_seq_save: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_restore), decltype(&t2139_gate_a_lib_sslm_seq_restore)>,
              "sslm_seq_restore: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_seq_set_adapter), decltype(&t2139_gate_a_lib_sslm_seq_set_adapter)>,
              "sslm_seq_set_adapter: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_decode_step), decltype(&t2139_gate_a_lib_sslm_decode_step)>,
              "sslm_decode_step: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");
static_assert(std::is_same_v<decltype(&sslm_stats), decltype(&t2139_gate_a_lib_sslm_stats)>,
              "sslm_stats: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");

int main() { return 0; }
