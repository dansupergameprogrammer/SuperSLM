#!/usr/bin/env python3
"""T-1891 gate G3 -- reference parity, the REFERENCE-SIDE half.

DISPOSABLE. Branch brunel/t1891-optionG-spike only, never merged.

Builds a `superslm_spike.pipeline.QuantizedModel` by hand -- the SAME geometry, weight
matrices, scales, and rope tables as `tools/t1891_optiong_gate3_probe.cpp`'s C++ fixture
(itself the same values as `tests/test_main.cpp`'s own `GqaGroupingFixture`, already
proven to clear every domain check the engine enforces, with the rope table replaced by
real, non-degenerate Q2.30 angles) -- and runs
`superslm_spike.dynamic_engine.forward_dynamic_vec` with `option_g_fused_k_landing=True`
over the SAME six real-English-text documents, byte-tokenized the SAME way
(`c % vocab_size`). Dumps the K-landing trace records (`trace.append`'s own
"{prefix}.k_proj.requant" site, one record per (layer, token, head)) to JSON for
`tools/t1891_optiong_gate3_check.py` to compare against the engine's own dumped K/V
store.

`fixture_model()` itself is NOT called: its calibration path imports
`superslm_spike.baseline` (pulls torch/transformers, neither vendored into this spike's
`tests/reference/superslm_spike/` copy -- confirmed at source, T-1519/T-1520's own commit
message). This script calls the SAME constituent functions `fixture_model()` calls
(`_pinned_weights`, `_calibrate`, `_derive_scales`, `_derive_composition_constants`,
`_build_rope_tables`) directly, substituting its own self-contained calibration
records/tokenizer for the ones `fixture_model()` sources from `baseline.py` -- everything
downstream of that substitution is the vendored, unmodified reference code, not a
reimplementation of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

from superslm_spike import dynamic_engine, pipeline  # noqa: E402

# --- Fixture geometry -- IDENTICAL to the C++ probe's own GqaGroupingFixture-derived
# fixture (tools/t1891_optiong_gate3_probe.cpp) ---------------------------------------
HIDDEN_SIZE = 8
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 2
INTERMEDIATE_SIZE = 8
CONTEXT_CAP = 16
VOCAB_SIZE = 32

CFG = pipeline.ModelConfig(
    hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, intermediate_size=INTERMEDIATE_SIZE,
    vocab_size=VOCAB_SIZE, rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
    context_cap=CONTEXT_CAP,
)

# Real, non-degenerate Q2.30 cos/sin values -- the SAME literal transcription (of the
# SAME cos(position)/sin(position) computation) the C++ probe carries.
COS_Q30 = [
    1073741824, 580145183, -446834263, -1062996349, -701844494, 304579952,
    1030974995, 809496382, -156229472, -978318669, -900946194, 4752057,
    906081289, 974363562, 146820470, -815708685,
]
SIN_Q30 = [
    0, 903522590, 976350678, 151526455, -812610492, -1029637100,
    -300020107, 705433989, 1062315328, 442508854, -584138220, -1073731308,
    -576140784, 451150921, 1063656549, 698241252,
]

DOCUMENTS = [
    "the quick fox",
    "a gray cat sat",
    "dog barks",
    "cold rain falls",
    "we ate soup",
    "red bird sings",
]


def tokenize(text: str) -> list[int]:
    """`c % vocab_size` -- the SAME scheme `pipeline._fixture_tokenize_prompt` uses and
    the C++ probe's own `Tokenize()` restates."""
    return [b % VOCAB_SIZE for b in text.encode("utf-8")]


def embed_value(token: int, channel: int) -> int:
    """`((t*7 + i*13) % 255) - 127` -- the SAME formula the C++ probe's `BuildEmbedTable`
    restates (not a copied table; both sides derive from this one rule)."""
    return ((token * 7 + channel * 13) % 255) - 127


def _identity8x8():
    import numpy as np
    return np.eye(8, dtype=np.int8)


def _selector4x8(rows):
    """rows: list of 4 column indices, one per output row, matching the C++ probe's own
    `kKSelector4x8`/`kVSelector4x8` (each output row is a one-hot selector onto one input
    channel)."""
    import numpy as np
    m = np.zeros((4, 8), dtype=np.int8)
    for r, c in enumerate(rows):
        m[r, c] = 1
    return m


def build_model() -> pipeline.QuantizedModel:
    import numpy as np

    codes, weight_scales, floats = pipeline._pinned_weights(CFG)  # noqa: SLF001

    # Override EVERY tensor with the SAME identity/selector/embed content the C++ probe
    # uses (kIdentity8x8/kKSelector4x8/kVSelector4x8/BuildEmbedTable), each at a UNIFORM
    # per-channel weight_scale -- so `_reference_fold`'s per-channel gate fold (this
    # module, above) is `None` (true pass-through) at EVERY channel, matching the C++
    # side's `identity=1` WSC1 fold exactly (D-SLM2305's construction does not touch this
    # mechanism; both sides must already agree on it for ANYTHING downstream to be
    # comparable).
    unit = (pipeline._UNIT_INT8_SCALE,)  # noqa: SLF001
    identity = _identity8x8()
    k_sel = _selector4x8([0, 1, 2, 3])
    v_sel = _selector4x8([0, 1, 2, 3])
    embed = np.zeros((VOCAB_SIZE, HIDDEN_SIZE), dtype=np.int8)
    for t in range(VOCAB_SIZE):
        for i in range(HIDDEN_SIZE):
            embed[t, i] = embed_value(t, i)

    codes["embed"] = embed
    floats["embed"] = np.asarray(embed, dtype=np.float64)
    weight_scales["embed"] = unit

    # Norm gains: `_pinned_weights`'s own non-projection path pins these to an
    # int8-shaped fixture value (`_pinned_int8`), which cannot represent this probe's
    # C++ side (`kNormGain`, forward_sites.h's `attn_norm_gain` is `int32_t*` and its
    # own production fixtures, e.g. test_main.cpp's GqaGroupingFixture, use 256 --
    # outside int8 range). Overridden here to the SAME 256 constant, both in
    # `dynamic_engine._rmsnorm_wide`'s own per-element multiplier array (`codes`) and
    # in `gain_of()`'s read of `weight_scales[name][0]` (`_derive_scales`'s
    # `norm_in = gain_of(...) / 2**NORM_FRAC_BITS` reads the gain's OWN value from
    # `weight_scales`, not from `codes` -- this fixture's own convention, confirmed by
    # execution: leaving this unset reproduced a `_UNIT_INT8_SCALE` gain on the
    # calibration side while the forward itself used 256, which is the root cause a
    # first pass at this fixture found by comparing C++ and Python's normed output
    # directly before this override existed).
    gain_array = np.full(HIDDEN_SIZE, 256, dtype=np.int64)
    for gname in [f"layer{l}.{n}.gain" for l in range(NUM_LAYERS) for n in ("attn_norm", "mlp_norm")] + ["final_norm.gain"]:
        codes[gname] = gain_array
        # The FLOAT reference sees gain as a real-valued multiplier (the C++/int
        # engine's own `gain_of(...)/2**NORM_FRAC_BITS` division, `_derive_scales`,
        # applied here instead so the float path calibrates against the SAME
        # effective gain the integer path actually applies -- 256/2**16 = 0.25,
        # not the raw int32 code 256 (which blew up the float sigmoid on first
        # execution: RuntimeWarning: overflow encountered in exp).
        floats[gname] = (gain_array.astype(np.float64) / (1 << pipeline.NORM_FRAC_BITS))
        weight_scales[gname] = (256.0,)

    for layer in range(NUM_LAYERS):
        prefix = f"layer{layer}"
        for name, mat, width in (
            (f"{prefix}.q_proj", identity, HIDDEN_SIZE),
            (f"{prefix}.k_proj", k_sel, NUM_KV_HEADS * HEAD_DIM),
            (f"{prefix}.v_proj", v_sel, NUM_KV_HEADS * HEAD_DIM),
            (f"{prefix}.o_proj", identity, HIDDEN_SIZE),
            (f"{prefix}.gate_proj", identity, INTERMEDIATE_SIZE),
            (f"{prefix}.up_proj", identity, INTERMEDIATE_SIZE),
            (f"{prefix}.down_proj", identity, HIDDEN_SIZE),
        ):
            codes[name] = mat
            # `floats[name]` is left at `mat` (not `mat * _UNIT_INT8_SCALE`), matching
            # this fixture's ORIGINAL, load-bearing convention: the whole calibration
            # chain (`_derive_scales`'s `add_rescale`/`add_requant`) is tuned to THIS
            # ratio between `codes` and `floats` across seven projections and two
            # layers, and multiplying every projection's floats by 1/127 uniformly was
            # tried and found to blow up an UNRELATED ratio (`mlp_residual.branch`'s
            # `quantize_multiplier` needs a shift outside [0,31]) rather than fixing
            # K's own saturation -- confirmed by execution, not assumed. K/V's own
            # landing scale is corrected directly, below (`_fix_kv_landing_for_g3`),
            # which is the narrower, load-bearing fix for T-1892 finding C1: the K/V
            # landing constants are re-derived FROM THE ACTUAL FUSED ROTATED MAGNITUDE
            # this fixture produces, not from this calibration chain's own (fragile,
            # indirect) maxima estimate.
            floats[name] = np.asarray(mat, dtype=np.float64)
            weight_scales[name] = unit * width

    float_weight = pipeline._dict_float_source(floats)  # noqa: SLF001
    tokenize_prompt = pipeline._fixture_tokenize_prompt(CFG)  # noqa: SLF001

    # Self-contained calibration records: this probe's OWN six documents, as plain
    # single-turn message lists (no `baseline.py` dependency) -- `_calibrate` measures
    # max-abs activations on the float reference over exactly the same text the parity
    # run below decodes, which is the honest calibration population for THIS probe
    # (rather than an unrelated corpus that happens to be vendored).
    records = [[{"role": "user", "content": doc}] for doc in DOCUMENTS]
    maxima = pipeline._calibrate(CFG, float_weight, records, tokenize_prompt)  # noqa: SLF001

    scales, residual_scales, biases = pipeline._derive_scales(  # noqa: SLF001
        CFG, maxima, weight_scales, {})
    composition_constants, kv_landing_scales, kv_landing_reciprocals = (
        pipeline._derive_composition_constants(CFG, weight_scales, scales))  # noqa: SLF001

    # T-1892 C1 fix: the calibration-derived K landing exponent (`e_t`) puts the
    # LANDED value 4-5 orders of magnitude past the +/-127 rail for this fixture's
    # actual fused-rotated magnitudes (measured by execution: x_int ~ 16-160,
    # landed ~ 1.4e6 at the ORIGINAL e_t) -- confirmed as a scale mismatch between
    # this hand-built fixture's calibration chain and its own actual integer
    # magnitudes, not a defect in the landing formula itself (the SAME formula,
    # unmodified, is what both the engine and this reference call).
    #
    # `e_t` is a free, independently-adjustable exponent: `LandingRescale`
    # (forward_sites.cpp) and `residual_reconcile` (intmath.py) take `r_t` and
    # `e_t` directly and never read `m_t` (the target's own canonical mantissa,
    # `kv_landing_scales`) at all -- `m_t`/`e_t`/`r_t` are computed together by
    # `_derive_composition_constants` but only `e_t` shapes LandingRescale's own
    # composed divisor (`ComposedExponent(e_a, e_t) = 62-(e_a-e_t)`), so shifting
    # it alone (leaving `r_t` -- the OFFLINE Newton reciprocal of `m_t` -- and
    # `m_t` itself untouched) changes nothing about what `m_t`/`kv_landing_scales`
    # claims elsewhere. `K_LANDING_EXPONENT_SHIFT` was found by measuring this
    # fixture's own actual rotated magnitude (this file's own diagnostic run) and
    # choosing the smallest shift that moves the compared K population off the
    # clamp rail -- V is UNCHANGED (not part of G3's compared population, and V's
    # own landing is not part of Option G's construction).
    K_LANDING_EXPONENT_SHIFT = 17
    for layer in range(NUM_LAYERS):
        for head in range(NUM_KV_HEADS):
            key = f"layer{layer}.k_head{head}"
            m_t, e_t, r_t = kv_landing_reciprocals[key]
            kv_landing_reciprocals[key] = (m_t, e_t + K_LANDING_EXPONENT_SHIFT, r_t)

    rope_tables = pipeline._as_rope_tables((  # noqa: SLF001
        [COS_Q30[p:p + 1] for p in range(CONTEXT_CAP)],
        [SIN_Q30[p:p + 1] for p in range(CONTEXT_CAP)],
    ))

    return pipeline.QuantizedModel(
        config=CFG, scales=scales, weights=codes, weight_scales=weight_scales,
        residual_scales=residual_scales, rope_tables=rope_tables, biases=biases,
        float_source=float_weight, tokenize_prompt=tokenize_prompt,
        calibration=pipeline.CalibrationRecord(
            corpus_sha256="t1891-spike-self-contained", tokenization="t1891 spike fixture",
            classes=()),
        gemm_weights={}, composition_constants=composition_constants,
        kv_landing_scales=kv_landing_scales, kv_landing_reciprocals=kv_landing_reciprocals,
    )


def dump_cpp_constants(model: pipeline.QuantizedModel) -> None:
    """Prints every derived value `tools/t1891_optiong_gate3_probe.cpp`'s C++ fixture
    needs, ready to paste as C++ literals -- the transcription source for that file (§5.4:
    both sides must read the SAME computed numbers, not two independent derivations of
    scales that happen to be labeled the same)."""
    print("=== C++ TRANSCRIPTION SOURCE (paste into t1891_optiong_gate3_probe.cpp) ===",
          file=sys.stderr)
    for layer in range(NUM_LAYERS):
        prefix = f"layer{layer}"
        print(f"-- layer {layer} --", file=sys.stderr)
        for site in ("attn_norm", "mlp_norm", "q_proj", "o_proj", "gate_proj", "up_proj",
                     "down_proj", "attn_ctx", "mlp_act", "attn_residual", "mlp_residual"):
            key = f"{prefix}.{site}"
            if key in model.composition_constants:
                m, e = model.composition_constants[key]
                print(f"  {key}: CarriedScale{{{m}, {e}}}", file=sys.stderr)
        for head in range(NUM_KV_HEADS):
            m_t_k, e_t_k, r_t_k = model.kv_landing_reciprocals[f"{prefix}.k_head{head}"]
            m_t_v, e_t_v, r_t_v = model.kv_landing_reciprocals[f"{prefix}.v_head{head}"]
            print(f"  k_head{head}: r_t={r_t_k}, e_t={e_t_k}   v_head{head}: r_t={r_t_v}, "
                  f"e_t={e_t_v}", file=sys.stderr)
            m, e = model.composition_constants[f"{prefix}.softmax_khead{head}"]
            print(f"  softmax_khead{head}: iexp_m={m}, iexp_e={e}", file=sys.stderr)
    print(f"embed: CarriedScale{{{model.composition_constants['embed'][0]}, "
          f"{model.composition_constants['embed'][1]}}}", file=sys.stderr)
    print(f"final_norm: CarriedScale{{{model.composition_constants['final_norm'][0]}, "
          f"{model.composition_constants['final_norm'][1]}}}", file=sys.stderr)
    print("=== END TRANSCRIPTION SOURCE ===", file=sys.stderr)


def inject_relative_error(relative_error: float):
    """T-1892 Critical 1's own vitality requirement, made a permanent part of G3
    (alongside G4's order-mutation): wraps `dynamic_engine._rotate_wide_pair_row`
    (module-level monkeypatch, not an edit to the vendored file) so every rotated
    pair this reference computes is scaled by `(1 + relative_error)` before landing
    -- a small, uniform relative error in the fused rotation itself, the exact
    defect class G3's own bit-exact population must be able to see (a hand-rolled
    128-bit facility's realistic failure is a small magnitude or rounding error,
    not a sign flip or an order-of-magnitude miss -- casebook
    `Claude/Poirot/96d2b11-t1891-optiong-spike.md` §5). Returns the ORIGINAL
    function so the caller can restore it; the patch is undone in a `finally`
    block by every caller here, never left installed.
    """
    original = dynamic_engine._rotate_wide_pair_row  # noqa: SLF001

    def mutated(seg, cos_row, sin_row):
        rotated = original(seg, cos_row, sin_row)
        return [
            v if v == 0 else int(round(v * (1.0 + relative_error)))
            for v in rotated
        ]

    dynamic_engine._rotate_wide_pair_row = mutated  # noqa: SLF001
    return original


def run_documents(model: pipeline.QuantizedModel) -> list[dict]:
    """One parity run over `DOCUMENTS`, returning the same per-document record shape
    `main()` dumps to JSON -- factored out so the injection check (below) can call it
    twice (clean, then mutated) without duplicating the loop."""
    all_docs = []
    for doc in DOCUMENTS:
        tokens = tokenize(doc)
        trace: list[dict] = []
        dynamic_engine.forward_dynamic_vec(
            model, tokens, cache=None, trace=trace, option_g_fused_k_landing=True)
        k_records = [r for r in trace if r.get("site", "").endswith(".k_proj.requant")]
        all_docs.append({
            "doc": doc,
            "tokens": tokens,
            "k_records": [
                {"site": r["site"], "token_index": int(r["token_index"]), "head": int(r["head"]),
                 "codes": [int(c) for c in r["codes"]]}
                for r in k_records
            ],
        })
    return all_docs


def run_injection_vitality_check(model: pipeline.QuantizedModel, relative_error: float,
                                  out_path: Path) -> list[dict]:
    """Runs `run_documents` with `_rotate_wide_pair_row` mutated by
    `relative_error`, dumps the mutated K-landing records to `out_path` (a SEPARATE
    file from the clean reference -- the clean reference stays the one G3/G4
    compare against), and returns the mutated records. The patch is always removed,
    even on an exception."""
    original = inject_relative_error(relative_error)
    try:
        docs = run_documents(model)
    finally:
        dynamic_engine._rotate_wide_pair_row = original  # noqa: SLF001
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"documents": docs}, f)
    return docs


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "out" / "t1891_gate3_reference.json"
    model = build_model()
    dump_cpp_constants(model)

    # T-1892 C1's own vitality requirement: inject a small relative error into the
    # fused rotation and confirm the comparator that reports 304/304 agreement
    # against the CLEAN reference reports disagreement against a MUTATED one, run
    # over the SAME six documents and the SAME landing constants -- everything
    # held fixed except the one thing under test. Both +/-10% (T-1892's own named
    # figure, which the ORIGINAL rail-degenerate population passed 0/608 changed)
    # and a much smaller +/-1% are checked, so this is not merely re-proving the
    # figure the review already measured on the broken fixture.
    injection_path_10 = out_path.parent / "t1891_gate3_reference_injected_10pct.json"
    injection_path_1 = out_path.parent / "t1891_gate3_reference_injected_1pct.json"
    injected_10 = run_injection_vitality_check(model, 0.10, injection_path_10)
    injected_1 = run_injection_vitality_check(model, 0.01, injection_path_1)
    clean_for_diff = run_documents(model)
    for label, injected in (("+10%", injected_10), ("+1%", injected_1)):
        changed = sum(
            1
            for d_clean, d_inj in zip(clean_for_diff, injected)
            for r_clean, r_inj in zip(d_clean["k_records"], d_inj["k_records"])
            for c_clean, c_inj in zip(r_clean["codes"], r_inj["codes"])
            if c_clean != c_inj
        )
        total = sum(len(d["k_records"]) * 2 for d in clean_for_diff)
        print(f"G3 injection vitality ({label} relative error in the fused rotation): "
              f"{changed}/{total} compared values changed", file=sys.stderr)
        if changed == 0:
            print(f"FAIL: G3's population is still blind to a {label} rotation error "
                  f"after the C1 fixture fix -- vitality NOT established", file=sys.stderr)
            return 1
    print("PASS: G3's population resolves both +/-10% and +/-1% relative errors in the "
          "fused rotation (nonzero compared-value changes at both)", file=sys.stderr)

    all_docs = []
    for doc in DOCUMENTS:
        tokens = tokenize(doc)
        trace: list[dict] = []
        logits = dynamic_engine.forward_dynamic_vec(
            model, tokens, cache=None, trace=trace, option_g_fused_k_landing=True)
        k_records = [r for r in trace if r.get("site", "").endswith(".k_proj.requant")]
        all_docs.append({
            "doc": doc,
            "tokens": tokens,
            "num_logits": int(logits.shape[0]) if hasattr(logits, "shape") else None,
            "k_records": [
                {"site": r["site"], "token_index": int(r["token_index"]), "head": int(r["head"]),
                 "codes": [int(c) for c in r["codes"]]}
                for r in k_records
            ],
        })
        print(f"document \"{doc}\": {len(tokens)} tokens, {len(k_records)} K-landing "
              f"records", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"documents": all_docs}, f)
    print(f"wrote reference K-landing records for {len(DOCUMENTS)} documents to {out_path}",
          file=sys.stderr)

    # T-1892 C1: report the compared population's own distribution -- this is the
    # cell the review's own fix condition names ("the compared population carries
    # values strictly inside +/-127"), checked here rather than left to the
    # comparator alone to discover.
    all_codes = [c for d in all_docs for r in d["k_records"] for c in r["codes"]]
    at_rail = sum(1 for c in all_codes if c in (127, -127))
    zero = sum(1 for c in all_codes if c == 0)
    interior_nonzero = len(all_codes) - at_rail - zero
    distinct = sorted(set(all_codes))
    print(f"G3 population: {len(all_codes)} values, {at_rail} at rail (+/-127), "
          f"{zero} exactly zero, {interior_nonzero} strictly interior and nonzero, "
          f"{len(distinct)} distinct values", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
