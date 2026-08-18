"""Plan §15 — the simulated integer pipeline: a Qwen2.5-shaped decode over the proven primitives.

§15: "offline Python; simulated integer pipeline in torch — static scales, pinned requant
semantics, i-exp/i-sqrt constructions, RoPE tables — no runtime code needed". The
primitives (`intmath.py`, `rope.py`) are green and proven. This suite pins the forward
pass that wires them into a decode, so the spike measures the pipeline that ships rather
than a plausible neighbour of it.

Three problems this artifact has that the scalar primitives did not, and the cells that
pin them:

1. **Vectorization recreates §6.2's SIMD discipline.** A scalar Python pipeline over 600
   records x ~40 tokens x 28 layers will not finish, so the implementation must vectorize.
   §6.2 already governs: "Every SIMD specialization must produce bit-equal results to the
   scalar reference on the exhaustive edge corpus (§13.1)". The sim's vectorized kernels
   stand in that same relation to `intmath.py`/`rope.py`. Every `vec_*` cell here drives
   the edge corpus, not nominal data — nominal data is where a broken rounding hides.

2. **float64 as an exact integer carrier.** Ruled ACCEPTED, with an envelope guard that
   raises rather than silently losing a bit. Reasoning in the test-design record §3; the
   short form is that int8xint8 over these dot lengths is exactly representable, so the
   carrier is exact AND order-independent — but the margin at a real context cap is
   0.78%, so the guard is load-bearing rather than ceremonial.

3. **Forge W6 — the reason this harness exists.** "torch's default round-half-even
   measures a different pipeline than the one that ships." Every requant goes through
   `intmath.py`. Two cells enforce it: a behavioural tie fixture (decisive, catches the
   substitution however it is spelled) and a monkeypatch sentinel (catches the library
   call at its origin).

§6.8 (D-SLM35) is normative and binds this suite: naming a construction does not
determine its bits. Where §6.8 and a §6 subsection could be read apart, §6.8 governs, and
the cells cite the table row.

**§9 — G-2, real weights, and the spike-grade pipeline (2026-07-15).** The 2026-07-15
record ruled G-2's upstream-fidelity oracle "owed at the spike run" because a self-authored
fixture model has no upstream to be faithful to. **This is the spike run**, so it comes due:
§9's cells pin the float path against HF `transformers` on real Qwen2.5 weights, before the
integer path is allowed to attribute anything to quantization. Without them, "integer
quantization damaged the task" and "we implemented Qwen wrong" are the same number.

Test-design record: Claude/Curie/superslm-t066-pipeline-test-design-2026-07-15.md
"""

import json
import os
import pathlib

import pytest

from conftest import CORPUS_SHA256, api, in_cap_tokenize, require

MODULE = "reference_pipeline.pipeline"
INTMATH = "reference_pipeline.intmath"
ROPE = "reference_pipeline.rope"

np = pytest.importorskip("numpy")

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1
EXACT_LIMIT = 2 ** 53

# The three §16 nominees, bench-confirmed (plan §5). OLMo-2 is full MHA — the degenerate
# grouping the op set must accept EXPLICITLY, not by accident.
NOMINEES = {
    "Qwen2.5-1.5B": dict(hidden_size=1536, num_hidden_layers=28, num_attention_heads=12,
                         num_key_value_heads=2, vocab_size=151936, rope_theta=1000000.0,
                         tie_word_embeddings=True),
    "OLMo-2-1B": dict(hidden_size=2048, num_hidden_layers=16, num_attention_heads=16,
                      num_key_value_heads=16, vocab_size=100352, rope_theta=500000.0,
                      tie_word_embeddings=False),
    "SmolLM2-360M": dict(hidden_size=960, num_hidden_layers=32, num_attention_heads=15,
                         num_key_value_heads=5, vocab_size=49152, rope_theta=100000.0,
                         tie_word_embeddings=True),
}


# --- the edge corpus the vectorized kernels must be bit-equal on -------------------

def requant_edge_corpus():
    """§13.1's "exhaustive edge corpora": saturation boundaries, INT32_MIN, rounding ties
    in both directions. Nominal data cannot see a wrong tie rule."""
    values = [INT32_MIN, INT32_MIN + 1, -(2 ** 30), -12345, -3, -2, -1, 0,
              1, 2, 3, 12345, 2 ** 30, INT32_MAX - 1, INT32_MAX]
    for k in range(-8, 9):                       # exact .5 ties, both signs, at shift 2
        values.append(k * 4 + 2)
        values.append(k * 4 - 2)
    for exponent in (1, 2, 3, 15, 30, 31):       # exact ties at each shift width
        half = 1 << (exponent - 1)
        for k in (-3, -1, 0, 1, 3):
            candidate = k * (1 << exponent) + half
            if INT32_MIN <= candidate <= INT32_MAX:
                values.append(candidate)
    return sorted(set(values))


def fixture_config_kwargs():
    """The §11 fixture model's shape, as plain data so a subprocess can rebuild it."""
    return dict(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def fixture_config(pipeline):
    """The §11 fixture model's shape: "a few-layer toy transformer with pinned weights".

    Tiny, self-authored, zero license surface. §11: "determinism is a property of the
    arithmetic, not of whether the model is any good" — so every cell that does not need
    a real checkpoint uses this.
    """
    return pipeline.ModelConfig(**fixture_config_kwargs())


# ==============================================================================
# 1. Vectorization — §6.2's SIMD discipline, on the edge corpus
# ==============================================================================

def test_vec_rounding_divide_by_pot_is_bit_equal_to_the_scalar_reference():
    """§6.2: "Every SIMD specialization must produce bit-equal results to the scalar
    reference on the exhaustive edge corpus." The vectorized kernel stands in exactly that
    relation to `intmath.rounding_divide_by_pot`, so it is held to exactly that bar.

    Driven on the edge corpus, not nominal data: a kernel that rounds correctly on
    ordinary activations and wrongly on ties is green on nominal data and ships a
    different pipeline.
    """
    vec = api(MODULE, "vec_rounding_divide_by_pot")
    scalar = api(INTMATH, "rounding_divide_by_pot")
    corpus = requant_edge_corpus()
    for exponent in (0, 1, 2, 3, 15, 30, 31):
        expected = [scalar(x, exponent) for x in corpus]
        got = vec(np.array(corpus, dtype=np.int64), exponent)
        assert [int(v) for v in got] == expected, f"exponent={exponent}"


def test_vec_saturating_rounding_doubling_high_mul_is_bit_equal_to_the_scalar_reference():
    """Same bar. §6.8 C2 pins this primitive's ties toward +infinity — the direction that
    differs from its sibling, and the one a vectorized rewrite is most likely to
    homogenise."""
    vec = api(MODULE, "vec_saturating_rounding_doubling_high_mul")
    scalar = api(INTMATH, "saturating_rounding_doubling_high_mul")
    corpus = requant_edge_corpus()
    for multiplier in (1, 2 ** 30, INT32_MAX, INT32_MIN, -1):
        expected = [scalar(x, multiplier) for x in corpus]
        got = vec(np.array(corpus, dtype=np.int64), multiplier)
        assert [int(v) for v in got] == expected, f"multiplier={multiplier}"


def test_vec_srdhm_reproduces_the_int32_min_saturation_case():
    """The one input pair whose defined result exceeds INT32_MAX. A vectorized kernel that
    drops the saturation branch wraps here and is bit-equal nowhere else that matters."""
    vec = api(MODULE, "vec_saturating_rounding_doubling_high_mul")
    got = vec(np.array([INT32_MIN], dtype=np.int64), INT32_MIN)
    assert int(got[0]) == INT32_MAX


def test_vec_multiply_by_quantized_multiplier_is_bit_equal_to_the_scalar_reference():
    """The composed requant (§6.8 C1) — both primitives, in order, with both roundings."""
    vec = api(MODULE, "vec_multiply_by_quantized_multiplier")
    scalar = api(INTMATH, "multiply_by_quantized_multiplier")
    corpus = requant_edge_corpus()
    for multiplier, shift in [(1073741824, 0), (1073741824, 7), (INT32_MAX, 31), (1, 0), (12345678, 3)]:
        expected = [scalar(x, multiplier, shift) for x in corpus]
        got = vec(np.array(corpus, dtype=np.int64), multiplier, shift)
        assert [int(v) for v in got] == expected, f"multiplier={multiplier} shift={shift}"


def test_vec_i_sqrt_is_bit_equal_to_the_scalar_reference():
    """§6.8 C4/C5/C6. The digit recurrence vectorizes as 32 unconditional masked steps; the
    per-digit branch becomes a select. Driven at the radix boundaries and the int64
    ceiling, where a vectorized shift width or a wrong mask lands first."""
    vec = api(MODULE, "vec_i_sqrt")
    scalar = api(INTMATH, "i_sqrt")
    domain = [0, 1, 2, 3, 4, 15, 16, 17, 63, 64, 255, 256, 10 ** 6,
              1085241185, 2 ** 31, 2 ** 62, 2 ** 63 - 1]
    for k in range(0, 32):
        domain.extend([4 ** k - 1, 4 ** k, 4 ** k + 1])
    domain = sorted({n for n in domain if 0 <= n <= 2 ** 63 - 1})
    expected = [scalar(n) for n in domain]
    got = vec(np.array(domain, dtype=object))
    assert [int(v) for v in got] == expected


def test_vec_i_exp_is_bit_equal_to_the_scalar_reference():
    """§6.3's i-exp on max-shifted logits, vectorized. Driven across the branch boundary
    (`q_ln2`) and into the clipped far tail, where the `>> z` becomes a per-lane variable
    shift — the step most likely to diverge from the scalar form."""
    vec = api(MODULE, "vec_i_exp")
    scalar, quantum = api(INTMATH, "i_exp", "i_exp_ln2_quantum")
    for scale in (0.005, 0.01, 0.02):
        q_ln2 = quantum(scale)
        domain = [0, -1, -q_ln2 + 1, -q_ln2, -q_ln2 - 1, -3 * q_ln2, -10 * q_ln2,
                  -30 * q_ln2, -31 * q_ln2, -10 ** 6]
        expected = [scalar(q, scale)[0] for q in domain]
        got_q, got_scale = vec(np.array(domain, dtype=object), scale)
        assert [int(v) for v in got_q] == expected, f"scale={scale}"
        assert got_scale == scalar(0, scale)[1]


def test_vec_rope_apply_pair_is_bit_equal_to_the_scalar_reference():
    """§6.4's rotation, vectorized. §6.8 C13 pins the single rounding as
    `RoundingDivideByPOT`; the fixtures include the exact-tie and int64-overflow cases the
    scalar suite uses, because a vectorized rewrite that narrows to int32 mid-stream passes
    on nominal magnitudes."""
    vec = api(MODULE, "vec_rope_apply_pair")
    scalar = api(ROPE, "rope_apply_pair")
    one = 1 << 30
    cases = [(1, 1, 1 << 29, 1 << 28), (127, 127, one, one), (1, 0, 1 << 29, 0),
             (-1, 0, 1 << 29, 0), (127, -128, one, 0), (0, 0, one, 0),
             (-128, 127, -one, 1 << 20), (12345, -6789, 12345678, -987654)]
    xs, ys, cs, ss = (np.array([c[i] for c in cases], dtype=np.int64) for i in range(4))
    got_x, got_y = vec(xs, ys, cs, ss)
    expected = [scalar(*c) for c in cases]
    assert [(int(a), int(b)) for a, b in zip(got_x, got_y)] == expected


def test_vectorized_kernels_agree_with_scalar_on_a_full_forward():
    """The composition claim, not the per-kernel one. Every kernel being individually
    bit-equal does not prove the pipeline that calls them is — an op could be wired to the
    wrong scale, or a reshape could transpose a head. The fixture model is small enough to
    run scalar, so both paths run and must agree bit-for-bit.

    This is the cell that catches a vectorized pipeline that is fast, self-consistent, and
    not the scalar reference.
    """
    forward, forward_scalar, fixture_model = api(
        MODULE, "forward", "forward_scalar_reference", "fixture_model")
    model = fixture_model(fixture_config(require(MODULE)))
    tokens = [1, 7, 3, 3, 11, 0, 31, 5]
    fast = forward(model, tokens)
    slow = forward_scalar(model, tokens)
    assert [int(v) for v in np.asarray(fast).ravel()] == [int(v) for v in np.asarray(slow).ravel()]


# ==============================================================================
# 2. float64 as an exact integer carrier — proven, not assumed
# ==============================================================================

def test_exact_int_limit_is_the_float64_mantissa_bound():
    """float64 represents every integer up to 2^53 exactly. That bound is the whole
    licence for using it as a carrier, so it is pinned rather than left as a comment."""
    limit = api(MODULE, "EXACT_INT_LIMIT")
    assert limit == 2 ** 53 == 9007199254740992
    assert float(limit) == limit                      # representable
    assert float(limit + 1) == float(limit)           # the first integer that is not


def test_int_matmul_is_bit_identical_to_an_exact_integer_reference():
    """The achievement claim for the carrier, oracled against exact Python ints — not
    against another float path.

    int8 x int8 over dot length D peaks at D*127*127. For every §5 shape that is far under
    2^53, so every partial sum is an exactly-representable integer and the float
    accumulation is exact.
    """
    int_matmul = api(MODULE, "int_matmul")
    rng = np.random.default_rng(20260715)
    for rows, dot, cols in [(3, 1536, 4), (2, 8960, 3), (5, 128, 5), (1, 960, 2)]:
        a = rng.integers(-128, 128, size=(rows, dot), dtype=np.int64)
        b = rng.integers(-128, 128, size=(dot, cols), dtype=np.int64)
        got = np.asarray(int_matmul(a, b))
        expected = a.astype(object) @ b.astype(object)
        assert got.shape == expected.shape
        assert [int(v) for v in got.ravel()] == [int(v) for v in expected.ravel()]


def test_int_matmul_is_exact_at_the_worst_case_magnitude_both_signs():
    """The worst case the envelope must carry: every operand at full int8 magnitude, so
    every product is 127*127 or -128*127 and every term adds rather than cancels. A
    random fixture never reaches this; a real activation distribution might.

    Both signs, because -128 is one larger in magnitude than +127 and an implementation
    that bounds on 127 alone is wrong at exactly the input that matters.
    """
    int_matmul = api(MODULE, "int_matmul")
    for dot in (1536, 8960):
        for a_val, b_val in [(127, 127), (-128, 127), (127, -128), (-128, -128)]:
            a = np.full((1, dot), a_val, dtype=np.int64)
            b = np.full((dot, 1), b_val, dtype=np.int64)
            got = int(np.asarray(int_matmul(a, b)).ravel()[0])
            assert got == dot * a_val * b_val, f"dot={dot} a={a_val} b={b_val}"


def test_int_matmul_is_order_independent_because_it_is_exact():
    """Exactness buys order-independence, and that is the property worth having.

    Every partial sum is an exactly-representable integer, so float addition of two of them
    is exact, so ANY summation order yields the same value. That is why a threaded BLAS
    GEMM — normally the enemy of determinism — is safe here. The claim is asserted rather
    than reasoned: the same dot product, permuted, must give identical bits.
    """
    int_matmul = api(MODULE, "int_matmul")
    rng = np.random.default_rng(11)
    a = rng.integers(-128, 128, size=(1, 8960), dtype=np.int64)
    b = rng.integers(-128, 128, size=(8960, 1), dtype=np.int64)
    baseline = int(np.asarray(int_matmul(a, b)).ravel()[0])
    for _ in range(4):
        perm = rng.permutation(8960)
        permuted = int(np.asarray(int_matmul(a[:, perm], b[perm, :])).ravel()[0])
        assert permuted == baseline


def test_assert_exact_accumulation_accepts_every_real_shape():
    """The §5 op set's actual contractions, all far inside the envelope."""
    assert_exact = api(MODULE, "assert_exact_accumulation")
    for dot in (128, 960, 1536, 2048, 8960, 11008):
        assert_exact(dot_length=dot, max_abs_lhs=128, max_abs_rhs=128)   # returns, does not raise


def test_assert_exact_accumulation_raises_rather_than_silently_losing_a_bit():
    """The guard's entire reason. A shape outside the exactly-representable range must
    RAISE — a float64 accumulation that quietly drops a low bit produces a plausible number
    and a wrong pipeline, which is the same failure class as the torch-default rounding
    this harness exists to prevent.
    """
    assert_exact, exceeded = api(MODULE, "assert_exact_accumulation", "ExactnessEnvelopeExceeded")
    with pytest.raises(exceeded):
        assert_exact(dot_length=2 ** 40, max_abs_lhs=128, max_abs_rhs=128)   # 2^54
    with pytest.raises(exceeded):
        assert_exact(dot_length=33027, max_abs_lhs=2 ** 31 - 1, max_abs_rhs=127)


def test_assert_exact_accumulation_boundary_is_the_true_break_point():
    """The guard is exact, not conservative-by-a-round-number. int32-wide probabilities
    against int8 values break at dot length 33,026 — and Qwen2.5's own context cap of
    32,768 sits at 99.2% of it, a 0.78% margin (test-design record §3).

    A guard that is off by a factor of two here either rejects a legal shape or admits an
    inexact one, and the inexact one is silent.
    """
    assert_exact, exceeded = api(MODULE, "assert_exact_accumulation", "ExactnessEnvelopeExceeded")
    prob_max = 2 ** 31 - 1
    break_point = EXACT_LIMIT // (prob_max * 127)
    assert break_point == 33026
    assert_exact(dot_length=break_point, max_abs_lhs=prob_max, max_abs_rhs=127)
    with pytest.raises(exceeded):
        assert_exact(dot_length=break_point + 1, max_abs_lhs=prob_max, max_abs_rhs=127)


def test_int_matmul_does_not_round_at_its_exit(monkeypatch):
    """**P-5: the carrier's float→int conversion is a cast, not a round.**

    If the accumulation is exact, a rounding step is a no-op. If it is not exact, the
    rounding step *hides* the very inexactness `assert_exact_accumulation` exists to
    raise — the guard is silently disarmed by the operation that looks like safety.

    So `int_matmul`'s exit must be a cast. Enforced by making every library rounding raise
    and requiring `int_matmul` to complete: this is the focused cell for the carrier seam,
    where `test_no_library_rounding_is_called_on_the_reproducible_path` covers the whole
    forward.
    """
    int_matmul = api(MODULE, "int_matmul")

    def forbidden(name):
        def _raise(*a, **k):
            raise AssertionError(
                f"{name} was called at the carrier's exit; an exact accumulation needs no "
                f"rounding, and an inexact one must raise ExactnessEnvelopeExceeded rather "
                f"than be rounded into looking correct"
            )
        return _raise

    for attr in ("round", "rint", "round_"):
        monkeypatch.setattr(np, attr, forbidden(f"numpy.{attr}"), raising=False)

    a = np.full((2, 1536), 127, dtype=np.int64)
    b = np.full((1536, 2), -128, dtype=np.int64)
    got = np.asarray(int_matmul(a, b))
    assert [int(v) for v in got.ravel()] == [1536 * 127 * -128] * 4


def test_int_matmul_enforces_the_envelope_on_its_own_operands():
    """The guard is wired into the carrier, not merely available beside it. An
    `int_matmul` that computes the envelope but never checks it is a comment.
    """
    int_matmul, exceeded = api(MODULE, "int_matmul", "ExactnessEnvelopeExceeded")
    a = np.full((1, 40000), 2 ** 31 - 1, dtype=np.int64)   # past the 33,026 break point
    b = np.full((40000, 1), 127, dtype=np.int64)
    with pytest.raises(exceeded):
        int_matmul(a, b)


# ==============================================================================
# 3. Forge W6 — no library rounding touches the reproducible path
# ==============================================================================

def test_requant_ties_follow_gemmlowp_not_round_half_even():
    """**The highest-value cell in this suite.**

    §15/W6: "torch's default round-half-even measures a different pipeline than the one
    that ships." The fixture is built so the two disagree: at shift 2, gemmlowp's
    `RoundingDivideByPOT` ties away from zero and returns [1, 3, -1, -3]; round-half-even
    returns [0, 2, 0, -2]. Behavioural, so it catches the substitution however it is
    spelled — `torch.round`, `np.round`, `np.rint`, a `.to(int8)` cast, or a hand-rolled
    banker's rounding.

    This defect produces a plausible number that answers a different question, which is the
    worst shape a defect can take in a measurement harness.
    """
    vec = api(MODULE, "vec_rounding_divide_by_pot")
    ties = np.array([2, 10, -2, -10], dtype=np.int64)      # +0.5, +2.5, -0.5, -2.5 at shift 2
    got = [int(v) for v in vec(ties, 2)]
    assert got == [1, 3, -1, -3], f"got {got}; round-half-even would give [0, 2, 0, -2]"


def test_srdhm_ties_toward_positive_infinity_not_away_from_zero():
    """§6.8 C2, and the correction D-SLM36 folded into the plan: §15's gloss said "ties
    away from zero" for both requant primitives and was WRONG for this one. The two
    primitives tie in different directions, so a vectorized rewrite that homogenises them
    is a defect the plan's own prose would have endorsed.

    -0.5 must give 0 (toward +infinity), not -1 (away from zero).
    """
    vec = api(MODULE, "vec_saturating_rounding_doubling_high_mul")
    got = int(vec(np.array([-(1 << 30)], dtype=np.int64), 1)[0])
    assert got == 0, f"got {got}; away-from-zero would give -1"


def test_no_library_rounding_is_called_on_the_reproducible_path(monkeypatch):
    """The W6 sentinel: catch the library call at its origin, not only at its symptom.

    Every rounding on the reproducible path goes through `intmath.py`'s gemmlowp
    primitives. If a forward pass reaches `np.round`, `np.rint`, or `torch.round`, this
    fails — regardless of whether that particular input happened to land on a tie.

    Pairs with the behavioural tie cells above: those catch a wrong result, this catches a
    wrong call that a given fixture did not happen to expose.
    """
    module = require(MODULE)
    forward, fixture_model = api(MODULE, "forward", "fixture_model")

    def forbidden(name):
        def _raise(*a, **k):
            raise AssertionError(
                f"{name} was called on the reproducible path; §15/W6 requires every "
                f"requant to go through intmath.py's gemmlowp primitives"
            )
        return _raise

    monkeypatch.setattr(np, "round", forbidden("numpy.round"), raising=False)
    monkeypatch.setattr(np, "rint", forbidden("numpy.rint"), raising=False)
    monkeypatch.setattr(np, "round_", forbidden("numpy.round_"), raising=False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "round", forbidden("torch.round"), raising=False)

    model = fixture_model(fixture_config(module))
    forward(model, [1, 5, 9, 2])          # must complete without touching library rounding


# ==============================================================================
# 4. Config is read from config.json, never the library (§6.8 C15, D-SLM38)
# ==============================================================================

def write_config(tmp_path, **overrides):
    body = dict(hidden_size=1536, num_hidden_layers=28, num_attention_heads=12,
                num_key_value_heads=2, intermediate_size=8960, vocab_size=151936,
                rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
                max_position_embeddings=32768)
    body.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_load_config_reads_rope_theta_from_config_json(tmp_path):
    """§6.8 C15 / D-SLM38: θ is read from the checkpoint's `config.json`, per model."""
    load_config = api(MODULE, "load_config")
    cfg = load_config(write_config(tmp_path, rope_theta=1000000.0))
    assert cfg.rope_theta == 1000000.0


def test_load_config_rejects_an_absent_rope_theta(tmp_path):
    """§11/D-SLM38, measured at transformers 5.13.1: `cfg.rope_theta` returns **None** —
    5.x moved it to `cfg.rope_parameters`. A converter on the 4.x idiom reads None
    silently and emits §6.4's tables on a wrong θ.

    §11: the result is "the worst shape a defect can take here: a model that loads, runs,
    generates fluent text, is fully deterministic, and is not the source model. Every gate
    in §13 except one ships it green."

    So absence is a hard rejection, never a default. This cell is cheap, needs no
    checkpoint and no threshold, and it catches at the source the defect G-2 catches at the
    symptom (test-design record §4).
    """
    load_config, config_error = api(MODULE, "load_config", "ConfigError")
    body = json.loads(write_config(tmp_path).read_text(encoding="utf-8"))
    del body["rope_theta"]
    (tmp_path / "config.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(config_error):
        load_config(tmp_path / "config.json")


def test_load_config_rejects_a_null_rope_theta(tmp_path):
    """The exact shape transformers 5.13.1 produces: present-but-None. A loader that treats
    None as "use the default" is the D-SLM38 defect verbatim."""
    load_config, config_error = api(MODULE, "load_config", "ConfigError")
    with pytest.raises(config_error):
        load_config(write_config(tmp_path, rope_theta=None))


def test_load_config_never_defaults_rope_theta(tmp_path):
    """A default is worse than a crash here, because it produces a model that runs. Two
    different configs must never collapse onto one θ."""
    load_config = api(MODULE, "load_config")
    assert load_config(write_config(tmp_path / "a", rope_theta=1000000.0)).rope_theta == 1000000.0
    assert load_config(write_config(tmp_path / "b", rope_theta=500000.0)).rope_theta == 500000.0


def test_load_config_does_not_import_transformers(tmp_path, monkeypatch):
    """§11: "The library's Config object is **not** a permitted source." The deeper reason
    is a reproducibility property — the Config object's attribute surface is
    version-unstable across majors, so reading model math through it makes the artifact a
    function of a dependency's release, which §11's reproducibility formula cannot express.

    Enforced by making the import fail: a loader that reaches for `transformers` cannot
    complete.
    """
    import builtins

    load_config = api(MODULE, "load_config")
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("load_config imported transformers; §11/D-SLM38 forbids "
                                 "reading model math through the library Config object")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    cfg = load_config(write_config(tmp_path))
    assert cfg.rope_theta == 1000000.0


def test_load_config_rejects_a_non_utf8_body(tmp_path):
    """T-1931 (Poirot casebook `f10db52e3b-shard-confirmation-round-2.md` §6.1, finding
    D1): a fourth production remedy in the M1 lineage arrived with no pin. T-1929 added
    `except UnicodeDecodeError` to THIS function (`pipeline.py:273-274`) in the same round
    that added the identical clause to `_load_index_json` -- the sharded suite's
    `test_load_index_json_rejects_a_non_utf8_index_body`
    (`tests/test_sharded_checkpoint.py`) pins the second function; nothing pinned the
    first, the original one T-1924's M1 named. `load_config` is not part of the sharded
    checkpoint feature, so this cell lives beside its five siblings above rather than in
    the sharded suite -- the natural home the confirmation casebook itself names.

    Falsifying mutation, confirmed both directions during authoring (test-design record):
    delete `pipeline.py:273-274` and this cell goes red with a raw `UnicodeDecodeError`
    escaping uncaught; restored, it is green again with `ConfigError` naming the failure.
    """
    load_config, config_error = api(MODULE, "load_config", "ConfigError")
    (tmp_path / "config.json").write_bytes(b'{"rope_theta": 10000.0, "bad": "\xff\xfe"}')
    with pytest.raises(config_error, match="(?i)utf-8"):
        load_config(tmp_path / "config.json")


@pytest.mark.parametrize("name", sorted(NOMINEES))
def test_load_config_reads_each_nominee_theta_from_its_real_config(name):
    """Against the real cached checkpoints, skipping cleanly when absent.

    θ differs per checkpoint — 1e6 / 5e5 / 1e5 — which is what makes D-SLM38 load-bearing
    rather than tidy: a defaulted θ is wrong for at least two of the three, and wrong
    silently.
    """
    load_config = api(MODULE, "load_config")
    path = cached_config_path(name)
    if path is None:
        pytest.skip(f"{name} not cached; run with HF_HOME set to exercise this cell")
    cfg = load_config(path)
    assert cfg.rope_theta == NOMINEES[name]["rope_theta"]
    assert cfg.num_attention_heads == NOMINEES[name]["num_attention_heads"]
    assert cfg.num_key_value_heads == NOMINEES[name]["num_key_value_heads"]


def cached_config_path(name):
    import os
    import pathlib
    repo = {"Qwen2.5-1.5B": "models--Qwen--Qwen2.5-1.5B-Instruct",
            "OLMo-2-1B": "models--allenai--OLMo-2-0425-1B-Instruct",
            "SmolLM2-360M": "models--HuggingFaceTB--SmolLM2-360M-Instruct"}[name]
    # T-2152 (outside strike item 6): no private filesystem path fallback -- HF_HOME is the
    # standard HuggingFace cache location env var; an unset one means this opt-in test skips.
    for root in [os.environ.get("HF_HOME")]:
        if not root:
            continue
        found = sorted(pathlib.Path(root).glob(f"hub/{repo}/snapshots/*/config.json"))
        if found:
            return found[0]
    return None


# ==============================================================================
# 5. The op set — §5 / D-SLM8, and the degenerate grouping
# ==============================================================================

@pytest.mark.parametrize("name", sorted(NOMINEES))
def test_attention_group_size_for_each_nominee(name, tmp_path):
    """§5's op set mandates **grouped**-query attention. Group size is n_q // n_kv."""
    load_config, group_size = api(MODULE, "load_config", "attention_group_size")
    cfg = load_config(write_config(tmp_path, **NOMINEES[name]))
    expected = NOMINEES[name]["num_attention_heads"] // NOMINEES[name]["num_key_value_heads"]
    assert group_size(cfg) == expected


def test_full_mha_is_accepted_as_the_named_degenerate_grouping():
    """Plan §5 (2026-07-14, bench): "**`num_key_value_heads == num_attention_heads` is an
    accepted, named configuration — group size 1 — and the converter recognizes it as
    such.**"

    OLMo-2-1B is full MHA (16/16). MHA *is* GQA at group size 1, but "compatible" is not
    "handled": a pipeline that mandates GQA and then silently tolerates n_kv == n_q is
    passing a case it never decided to pass, and "an accidental pass here is
    indistinguishable from a missing check".

    So the cell drives it explicitly and asserts the pipeline *names* it — group size 1,
    reported, not inferred by the reader from the absence of a crash.
    """
    load_config, group_size, is_mha = api(
        MODULE, "load_config", "attention_group_size", "is_degenerate_grouping")
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(write_config(pathlib.Path(d), **NOMINEES["OLMo-2-1B"]))
    assert group_size(cfg) == 1
    assert is_mha(cfg) is True


def test_gqa_is_not_reported_as_the_degenerate_grouping(tmp_path):
    """The negative control for the cell above. Without it, `is_degenerate_grouping`
    returning True unconditionally would pass."""
    load_config, is_mha = api(MODULE, "load_config", "is_degenerate_grouping")
    for name in ("Qwen2.5-1.5B", "SmolLM2-360M"):
        cfg = load_config(write_config(tmp_path / name, **NOMINEES[name]))
        assert is_mha(cfg) is False


def test_a_head_count_that_does_not_divide_is_rejected(tmp_path):
    """§11's reject-over-degrade discipline. A config whose q heads are not a whole
    multiple of its kv heads has no grouping, and a pipeline that floor-divides its way
    past that is silently computing a different attention."""
    load_config, unsupported = api(MODULE, "load_config", "UnsupportedOpSet")
    cfg = load_config(write_config(tmp_path, num_attention_heads=12, num_key_value_heads=5))
    group_size = api(MODULE, "attention_group_size")
    with pytest.raises(unsupported):
        group_size(cfg)


def test_tied_embeddings_reuse_the_embedding_matrix(tmp_path):
    """§5/D-SLM8's op set includes tied embeddings, and two of the three nominees tie
    (Qwen2.5 and SmolLM2 True, OLMo-2 False). A pipeline that ignores the flag and keeps a
    separate lm_head silently uses uninitialised or duplicated weights."""
    module = require(MODULE)
    fixture_model, lm_head_weight, embedding_weight = api(
        MODULE, "fixture_model", "lm_head_weight", "embedding_weight")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    assert cfg.tie_word_embeddings is True
    assert np.asarray(lm_head_weight(model)).shape == np.asarray(embedding_weight(model)).shape
    assert np.array_equal(np.asarray(lm_head_weight(model)), np.asarray(embedding_weight(model)))


# ==============================================================================
# 6. Static scales are CONSTANTS (D-SLM5)
# ==============================================================================

def test_static_scales_are_offline_constants_of_the_model():
    """D-SLM5 baseline: "static per-tensor activation scales, calibrated offline... every
    scale is a constant in the artifact; the runtime never computes a scale."

    The scales a forward pass uses must be the model's, identically, for every input.

    AMENDED for the dual-arm artifact (Curie 2026-07-16, design record A-7 addendum):
    the original `first == model.scales` was a TOTALITY proxy from the static-arm-only
    era — "the static forward reads everything the artifact carries". The artifact now
    also carries the DYNAMIC arm's constants: the C27 per-head KV landing entries
    (`layer{L}.{k,v}_head{h}.scale`, the A-3-pinned calibration surface), which the
    static forward correctly never reads. The totality claim survives with the
    complement NAMED: the static read set equals the artifact minus exactly that
    surface — asserted non-empty so the subset claim cannot go vacuous, and any OTHER
    unread constant still fails the cell (a stray artifact constant nothing consumes
    is still a defect).
    """
    import re

    module = require(MODULE)
    fixture_model, scales_used = api(MODULE, "fixture_model", "scales_used_by_forward")
    model = fixture_model(fixture_config(module))
    first = scales_used(model, [1, 2, 3, 4])
    second = scales_used(model, [31, 30, 29, 28, 27, 26])     # different tokens, different range
    assert first == second, "the scales moved with the data; that is the dynamic fallback, not the baseline"

    # The static forward reads every requant and rescale constant in the artifact.
    assert first.requant == model.scales.requant
    assert sorted(first.rescale) == sorted(model.scales.rescale)

    # Nonlinear: everything except the named dynamic-arm complement, exactly.
    kv_head = re.compile(r"^layer\d+\.[kv]_head\d+\.scale$")
    artifact = dict(model.scales.nonlinear)
    dynamic_only = {name for name in artifact if kv_head.match(name)}
    assert dynamic_only, (
        "the artifact carries no per-head KV landing entries — the dynamic arm's "
        "pinned calibration surface (C27/D-SLM58) is missing, and this cell's named "
        "complement would be vacuous")
    assert dict(first.nonlinear) == {
        name: value for name, value in artifact.items() if name not in dynamic_only
    }, (
        "the static forward's nonlinear read set does not equal the artifact minus "
        "the named dynamic-arm surface — either a static constant went unread (dead "
        "artifact data) or the static path read a dynamic-arm constant")


def test_a_forward_pass_does_not_compute_a_scale(monkeypatch):
    """The cell that catches the designed fallback masquerading as the baseline.

    §6.2's fallback — dynamic per-token scales from an integer max-abs reduction — is
    "designed now and switched on only by spike evidence". If the baseline silently
    computes scales instead of reading them, the spike measures the fallback and reports it
    as the baseline: the format-lock answer would be for a pipeline we did not decide to
    ship.

    Enforced structurally: the baseline forward must never call the scale calibrator.
    """
    module = require(MODULE)
    forward, fixture_model, forbidden = api(
        MODULE, "forward", "fixture_model", "DynamicScaleForbidden")
    model = fixture_model(fixture_config(module))

    def trap(*a, **k):
        raise forbidden("the baseline forward computed an activation scale; D-SLM5 pins "
                        "static scales as offline constants and makes dynamic scales the "
                        "spike-gated fallback")

    monkeypatch.setattr(module, "compute_activation_scale", trap, raising=False)
    forward(model, [1, 5, 9, 2])


def test_scales_are_integer_multiplier_shift_pairs():
    """§6.2's requant consumes a quantized multiplier and a shift, not a float scale. A
    float scale on the path is the trap T-065's prosecution caught in the "integer-only"
    literature.

    **Covers BOTH site lists.** §6.2's integer requant form governs every requant on the
    reproducible path, not only the ones that happen to have a weight. When the site list
    split (2026-07-15), this cell iterated `requant_sites()` alone and its reach shrank to
    the seven projections while its green stayed — a float leaking into a weightless
    rescale would have been invisible. The property is §6.2's; the scope is the path.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))

    def check(where, multiplier, shift):
        assert type(multiplier) is int, f"{where}: multiplier is {type(multiplier).__name__}"
        assert type(shift) is int, f"{where}: shift is {type(shift).__name__}"
        assert 0 <= shift <= 31, f"{where}: shift {shift} outside RoundingDivideByPOT's range"

    for name in model.scales.requant_sites():
        multipliers, shifts = model.scales.requant_for(name)
        for channel, (multiplier, shift) in enumerate(zip(multipliers, shifts)):
            check(f"{name}[{channel}]", multiplier, shift)

    for name in model.scales.rescale_sites():
        multiplier, shift = model.scales.rescale_for(name)
        check(name, multiplier, shift)


def test_the_site_lists_partition_the_requant_path():
    """**The anti-regression cell for the split** (2026-07-15).

    The site list split into `requant_sites()` (the seven §6.1 W8 projections, which have a
    weight and therefore a channel index) and `rescale_sites()` (the weightless rescales,
    which do not). The split is right — see `test_a_rescale_site_has_no_weight_scales` for
    what makes it meaningful — but it is the mechanism by which a cell's reach can shrink
    while its green does not move, which is this session's recurring pattern.

    So the lists are pinned as a partition: disjoint, both non-empty, and the projection
    count follows from §5's op set rather than from whatever the implementation happens to
    register. A refactor that moves a site out of one list and forgets to add it to the
    other drops it from every cell that iterates either — and fails here instead.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    projections, rescales = model.scales.requant_sites(), model.scales.rescale_sites()

    assert set(projections).isdisjoint(rescales), (
        f"a site is in both lists: {sorted(set(projections) & set(rescales))}"
    )
    assert projections and rescales
    # §5 / D-SLM8's op set: q, k, v, o, gate, up, down — seven W8 projections per layer.
    assert len(projections) == 7 * cfg.num_hidden_layers, (
        f"{len(projections)} projection sites for {cfg.num_hidden_layers} layers; §5's op "
        f"set has seven W8 projections per layer (q/k/v/o/gate/up/down)"
    )
    assert len(set(projections)) == len(projections)
    assert len(set(rescales)) == len(rescales)


def test_a_rescale_site_has_no_weight_scales():
    """What makes the split mean something: the lists differ by **the presence of a
    weight**, and nothing else.

    A weightless rescale has no weight and therefore no channel index. Giving it a
    fictional `S_w = 1.0` vector would invent a fact *and* reproduce the exact
    vector-of-identical-scales shape §3a's behavioural cells exist to reject.

    This is also the negative control for the partition cell above: without it,
    `rescale_sites()` returning the projections would satisfy every count and disjointness
    assertion that matters.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))

    for name in model.scales.rescale_sites():
        with pytest.raises(KeyError):
            model.scales.weight_scales(name)
        multiplier, shift = model.scales.rescale_for(name)
        assert isinstance(multiplier, int) and isinstance(shift, int), (
            f"{name}: a weightless rescale's requant is a scalar pair — there is no channel "
            f"index for it to carry"
        )

    for name in model.scales.requant_sites():
        with pytest.raises(KeyError):
            model.scales.rescale_for(name)


def test_the_tied_embedding_is_not_a_per_channel_projection():
    """§6.1 says "per-output-channel" for W8; §6.5 says the final logits "stay int32" and
    are selected by argmax. **For the lm_head those two cannot both hold**, and §6.5 wins.

    `real_logit[j] = acc[j] * S_x * S_w[j]`. §6.5 pins argmax over the raw int32 `acc`, with
    no requant to fold a scale into. That selection is correct only if every vocab channel
    shares one scale: measured, a per-output-channel lm_head scale selects a **different
    token in ~80% of random draws**. So per-channel scales on the lm_head would not merely
    be unnecessary — they would make §6.5's decode wrong.

    The tied case is stronger still. One tensor serves two uses with **orthogonal output
    axes** — as the embedding lookup the output channel is the hidden dim; as the lm_head
    (`x @ E.T`) it is the vocab index — so "per output channel" cannot hold on both at once.

    §6.1 does not state this exemption, which is finding P-11. The cell pins the reading
    §6.5 forces.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    sites = set(model.scales.requant_sites()) | set(model.scales.rescale_sites())
    for name in sites:
        assert "embed" not in name and "lm_head" not in name, (
            f"{name}: the embedding/lm_head is registered as a scale site. §6.5 selects on "
            f"the raw int32 logits, so the lm_head carries no requant that could hold a "
            f"per-channel weight scale"
        )


# ==============================================================================
# 6a. Weight scales are PER-OUTPUT-CHANNEL (§6.1); activation scales are PER-TENSOR (§6.2)
# ==============================================================================
#
# The plan splits the two granularities and the 2026-07-15 contract collapsed them:
#
#   §6.1 — "W8: symmetric int8, **per-output-channel** static scales, computed offline."
#   §6.2 — "Baseline: static **per-tensor** activation scales, calibrated offline."
#
# `StaticScales.as_multiplier_shift_pairs() -> dict[str, tuple[int, int]]` returned one
# scalar pair per tensor, which forces the weight scale to be folded per-tensor. The
# builder followed the contract faithfully; the defect is the contract's.
#
# **Why it fakes the answer.** Per-tensor weight quantization is materially worse than
# per-channel: one outlier channel dominates the shared scale and crushes the resolution
# of every other channel. That is why §6.1 specifies per-channel. So the sim measured a
# WORSE pipeline than the one that ships, and the error runs toward a **false FAIL** on
# §15's 2-point bar — W8A8 lands low, and "integer quantization damaged the task" and "the
# harness quantized wrong" are indistinguishable at the output. That sends the plan down
# §15's fallback ladder (integer dynamic scales -> QAT -> the encoder pivot) for a defect
# that is not in the design.
#
# The arithmetic §6.1 and §6.2 state together:
#
#     acc[j]   = SUM_i x[i] * W[i][j]                     int32 (C17)
#     real[j] ~= acc[j] * S_x * S_w[j]                    S_x per-tensor, S_w per-channel
#     q_out[j] = requant(acc[j], M[j]), M[j] = S_x * S_w[j] / S_out
#
# so the composed requant multiplier is **per output channel**, and both the multiplier
# and the shift are vectors of length out_channels. See the test-design record §3a for the
# shape's S2 reasoning.


def heterogeneous_weight():
    """A weight whose output channels have deliberately different dynamic ranges.

    Column 0 is an outlier at 1270.0; the rest sit at 1.0. Per-channel scales give every
    column the full int8 grid. A single per-tensor scale is set by column 0 and quantizes
    every other column to zero — which is the resolution collapse §6.1 exists to prevent,
    and it is invisible on a weight matrix whose channels happen to share a range.
    """
    weight = np.ones((8, 6), dtype=np.float64)
    weight[:, 0] = 1270.0
    weight[1::2, :] *= -1.0
    return weight


def test_weight_scales_are_per_output_channel():
    """§6.1. One scale per output channel — a tensor's scale list is not a scalar and not
    length 1."""
    quantize = api(MODULE, "quantize_weight_per_channel")
    weight = heterogeneous_weight()
    _, scales = quantize(weight, output_axis=1)
    assert len(scales) == weight.shape[1] == 6, (
        f"got {len(scales)} scales for {weight.shape[1]} output channels; §6.1 pins one "
        f"per output channel, and a single scale is the per-tensor fold"
    )


def test_per_channel_quantization_preserves_a_small_channel_beside_an_outlier():
    """**The cell that catches the per-tensor fold**, grounded in why §6.1 says
    per-channel rather than in how it is implemented.

    Column 0 is 1270x the others. Per-channel, every column gets its own scale and
    round-trips to within half a step of its own grid. Per-tensor, the shared scale is
    1270/127 = 10, so the 1.0 columns quantize to round(1.0/10) = 0 and dequantize to
    0.0 — a 100% error on 5 of 6 channels.

    A per-tensor implementation misses this bound by ~50x. A weight matrix with uniform
    channel ranges cannot tell the two apart, which is why the fixture is built with an
    outlier.
    """
    quantize, dequantize = api(
        MODULE, "quantize_weight_per_channel", "dequantize_weight_per_channel")
    weight = heterogeneous_weight()
    q, scales = quantize(weight, output_axis=1)
    recovered = np.asarray(dequantize(q, scales, output_axis=1), dtype=np.float64)
    for channel in range(weight.shape[1]):
        column = weight[:, channel]
        error = np.abs(recovered[:, channel] - column).max() / np.abs(column).max()
        assert error <= 0.02, (
            f"channel {channel} round-tripped with {error:.3f} relative error; a "
            f"per-output-channel scale bounds it at ~1/254. A single per-tensor scale set "
            f"by the outlier column crushes this channel to zero (error 1.0)"
        )


def test_weight_quantization_is_symmetric():
    """§6.1: "**symmetric** int8". Symmetric means the grid is symmetric about zero — no
    zero point, and the negative end is -127, not -128. -128 has no positive counterpart,
    so admitting it is an asymmetric scheme wearing a symmetric name.

    Dequantization is therefore a bare multiply: `w ~= q * S`, with no offset to subtract.
    """
    quantize, dequantize = api(
        MODULE, "quantize_weight_per_channel", "dequantize_weight_per_channel")
    weight = heterogeneous_weight()
    q = np.asarray(quantize(weight, output_axis=1)[0])
    assert np.issubdtype(q.dtype, np.integer)
    assert q.min() >= -127, f"quantized weight reached {q.min()}; symmetric int8 is [-127, 127]"
    assert q.max() <= 127
    # zero maps to zero exactly: the defining property of a zero-point-free grid
    zeros = np.zeros((4, 2), dtype=np.float64)
    qz, sz = quantize(zeros, output_axis=1)
    assert np.all(np.asarray(qz) == 0)
    assert np.all(np.asarray(dequantize(qz, sz, output_axis=1)) == 0.0)


def test_a_projection_s_activation_scales_are_per_tensor():
    """§6.2's baseline: "static **per-tensor** activation scales". One number per activation
    tensor — not per channel.

    **Scoped to the projections deliberately, and the scope is the claim.** The property is
    about *where the channel index lives*: at a projection the weight carries it (§6.1) and
    the activations must not (§6.2). That is a real thing to get wrong, and this cell is
    what catches getting it wrong.

    At a weightless rescale there is no channel index at all — `rescale_for` returns a
    scalar pair by construction — so a cell asserting per-tensor-ness there could not fail
    on any real defect, and a test that cannot fail is worse than no test. The reach was
    reviewed when the site list split (2026-07-15) and this is the scope the claim supports;
    the integer-form property, which *does* apply path-wide, is covered over both lists by
    `test_scales_are_integer_multiplier_shift_pairs`.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    for name in model.scales.requant_sites():
        for role, scale in (("input", model.scales.input_scale(name)),
                            ("output", model.scales.output_scale(name))):
            assert isinstance(scale, float), (
                f"{name}: the {role} activation scale is {type(scale).__name__}; §6.2 pins "
                f"one number per tensor, not a per-channel vector"
            )
        assert len(model.scales.weight_scales(name)) > 1, (
            f"{name}: the channel index must live on the weight, not the activations"
        )


def test_requant_multipliers_are_per_output_channel():
    """The composed multiplier inherits the weight scale's granularity: one
    (multiplier, shift) per output channel, not one per tensor."""
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    for name in model.scales.requant_sites():
        multipliers, shifts = model.scales.requant_for(name)
        assert len(multipliers) == len(shifts)
        assert len(multipliers) == len(model.scales.weight_scales(name)), (
            f"{name}: {len(multipliers)} requant multipliers for "
            f"{len(model.scales.weight_scales(name))} output channels"
        )
        assert len(multipliers) > 1, (
            f"{name}: a single multiplier for the whole tensor is the per-tensor fold §6.1 "
            f"rules out"
        )


def test_per_channel_scales_differ_when_the_channels_differ():
    """The detector for a scalar broadcast into a vector of the right length.

    Every shape cell above is satisfied by folding one per-tensor scale and repeating it
    `out_channels` times. This cell is what makes that fail: given columns whose ranges
    differ by 1270x, the per-channel scales must differ by 1270x too. That difference IS the
    resolution §6.1 buys, so a pipeline where it is absent has not implemented §6.1
    whatever its shapes look like.
    """
    quantize = api(MODULE, "quantize_weight_per_channel")
    _, scales = quantize(heterogeneous_weight(), output_axis=1)
    assert len(set(scales)) > 1, (
        "every output channel got the same scale for a weight whose columns differ by "
        "1270x; the scale was folded per-tensor and broadcast"
    )
    assert scales[0] == pytest.approx(scales[1] * 1270.0, rel=1e-6), (
        "the outlier column's scale must be 1270x its neighbours' — that ratio is exactly "
        "the resolution a shared scale would destroy"
    )


def test_requant_composes_the_per_tensor_activation_and_per_channel_weight_scales():
    """M[j] = S_x * S_w[j] / S_out — the arithmetic §6.1 and §6.2 state together.

    Pins the split rather than the granularity: a pipeline that made the *activation* scale
    per-channel would also produce a varying multiplier vector and would pass the cells
    above. This is the cell that says which factor carries the channel index.
    """
    module = require(MODULE)
    fixture_model, quantize_multiplier = api(MODULE, "fixture_model", "quantize_multiplier")
    model = fixture_model(fixture_config(module))
    for name in model.scales.requant_sites():
        multipliers, shifts = model.scales.requant_for(name)
        s_in = model.scales.input_scale(name)
        s_out = model.scales.output_scale(name)
        weight_scales = model.scales.weight_scales(name)
        for channel, weight_scale in enumerate(weight_scales):
            expected = quantize_multiplier(s_in * weight_scale / s_out)
            assert (multipliers[channel], shifts[channel]) == expected, (
                f"{name}[{channel}]: requant does not compose S_x * S_w[{channel}] / S_out"
            )


def test_vec_requant_accepts_per_channel_arrays_and_is_bit_equal_to_scalar():
    """§6.2's SIMD discipline, extended to the per-channel form: the kernel now broadcasts a
    multiplier/shift **per output channel** along the last axis, and each lane must still be
    bit-equal to the scalar reference for its own channel's pair."""
    vec = api(MODULE, "vec_multiply_by_quantized_multiplier")
    scalar = api(INTMATH, "multiply_by_quantized_multiplier")
    acc = np.array([[7, -9, 2 ** 30, -(2 ** 30)],
                    [0, 1, -1, 12345]], dtype=np.int64)
    multipliers = np.array([1073741824, 2147483647, 1, 12345678], dtype=np.int64)
    shifts = np.array([0, 31, 7, 3], dtype=np.int64)
    got = np.asarray(vec(acc, multipliers, shifts))
    for row in range(acc.shape[0]):
        for channel in range(acc.shape[1]):
            expected = scalar(int(acc[row, channel]),
                              int(multipliers[channel]), int(shifts[channel]))
            assert int(got[row, channel]) == expected, f"[{row}][{channel}]"


# ==============================================================================
# 6b. Accumulator width — §6.8 C17
# ==============================================================================

def test_accumulator_width_is_int32_for_every_real_shape():
    """§6.8 C17 — **PINNED**: "int32, or int64 where dot-length x magnitude requires it —
    decided per layer at conversion from worst-case bounds and recorded in the artifact".

    The sim accumulates exactly (§3's carrier), which models an accumulator that is always
    wide enough. That is *equivalent* to the runtime only while C17's per-layer choice
    actually covers the bound — so the derivation is asserted rather than assumed. For every
    §5 shape the worst case `D * 127 * 127` fits int32 with >=4x headroom, so C17 selects
    int32 everywhere and the sim's exactness is faithful.
    """
    accumulator_width = api(MODULE, "accumulator_width")
    for dot in (128, 960, 1536, 2048, 8960, 11008, 32768):
        assert accumulator_width(dot_length=dot, max_abs_lhs=127, max_abs_rhs=127) == 32, (
            f"dot_length={dot}: worst case {dot * 127 * 127} fits int32"
        )


def test_accumulator_width_selects_int64_when_the_bound_requires_it():
    """The other half of C17, and the negative control for the cell above: a width function
    that returned 32 unconditionally would pass it."""
    accumulator_width = api(MODULE, "accumulator_width")
    int32_break = (2 ** 31 - 1) // (127 * 127)
    assert accumulator_width(dot_length=int32_break, max_abs_lhs=127, max_abs_rhs=127) == 32
    assert accumulator_width(dot_length=int32_break + 1, max_abs_lhs=127, max_abs_rhs=127) == 64
    assert accumulator_width(dot_length=4096, max_abs_lhs=2 ** 20, max_abs_rhs=127) == 64


# ==============================================================================
# 7. Greedy selection (§6.5 / C16) and the constrained decode (§15)
# ==============================================================================

def test_forward_returns_int32_logits_not_probabilities():
    """§6.5: "Final logits stay int32... No final softmax exists in the runtime.\""""
    module = require(MODULE)
    forward, fixture_model = api(MODULE, "forward", "fixture_model")
    cfg = fixture_config(module)
    logits = np.asarray(forward(fixture_model(cfg), [1, 2, 3]))
    assert logits.shape[-1] == cfg.vocab_size
    assert np.issubdtype(logits.dtype, np.integer), f"logits are {logits.dtype}, not integer"
    assert logits.min() >= INT32_MIN and logits.max() <= INT32_MAX


def test_greedy_selection_breaks_ties_on_the_lowest_token_index():
    """§6.8 C16. Argmax over int32 logits with the tie-break pinned. numpy's argmax happens
    to agree, but the pinning is a contract, not an inherited default — a torch path or a
    reordered reduction can disagree and nothing else in the suite would see it."""
    select = api(MODULE, "select_greedy")
    assert select(np.array([5, 9, 9, 2, 9], dtype=np.int32)) == 1
    assert select(np.array([7, 7, 7, 7], dtype=np.int32)) == 0
    assert select(np.array([INT32_MIN, INT32_MIN], dtype=np.int32)) == 0
    assert select(np.array([-1, INT32_MAX, INT32_MAX], dtype=np.int32)) == 1


def test_constrained_decode_selects_only_from_the_mask():
    """§15 scores "under schema-constrained greedy decoding", so the pipeline's decode is
    the constrained one — `constrain.py`'s masks applied to int32 logits before argmax
    (§9). A decode that ignores the mask measures a different task.
    """
    select_masked = api(MODULE, "select_greedy_masked")
    logits = np.array([100, 90, 80, 70], dtype=np.int32)
    assert select_masked(logits, [True, True, True, True]) == 0
    assert select_masked(logits, [False, True, True, True]) == 1
    assert select_masked(logits, [False, False, False, True]) == 3


def test_constrained_decode_masking_is_the_shared_primitive():
    """The pipeline masks through `constrain.argmax_masked` rather than a second
    implementation of the same rule. Two maskers is two tie-breaks."""
    select_masked = api(MODULE, "select_greedy_masked")
    argmax_masked = api("reference_pipeline.constrain", "argmax_masked")
    cases = [([5, 9, 9, 2], [True] * 4), ([5, 9, 9, 2], [False, True, True, False]),
             ([7, 7, 7, 7], [False, True, True, True]), ([1, 2, 3, 4], [True, False, False, False])]
    for logits, mask in cases:
        assert select_masked(np.array(logits, dtype=np.int32), mask) == argmax_masked(logits, mask)


def schema_vocab(vocab_size):
    """A character vocabulary that can spell the enum's JSON serialization."""
    base = ['"', "a", "b"]
    filler = [chr(ord("A") + i) for i in range(vocab_size - len(base))]
    return base + filler


def dfa_accepts(dfa, tokens):
    """Walk the DFA — §17 dim 10's "parse the produced tokens against the registered
    schema, assert validity". `constrain.TokenDFA` exposes the walk, not a verdict."""
    state = dfa.start
    for token_id in tokens:
        if token_id not in dfa.valid_tokens(state):
            return False
        state = dfa.step(state, token_id)
    return dfa.is_accepting(state)


def test_a_decode_under_a_schema_yields_schema_valid_output():
    """§17 dimension 10's feature oracle: "schema-constrained decode yields schema-valid
    output (parse the produced tokens against the registered schema, assert validity — not
    'same tokens twice')".

    The fixture model is arithmetic, not a trained model, so what is asserted is that the
    *decode* respects the schema — the mask holds whatever the logits say. Task quality is
    the spike's oracle (§15), never this suite's (J-N1).
    """
    module = require(MODULE)
    decode_constrained, fixture_model = api(MODULE, "decode_constrained", "fixture_model")
    compile_schema = api("reference_pipeline.constrain", "compile_schema")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    dfa = compile_schema({"enum": ["a", "b"]}, schema_vocab(cfg.vocab_size))
    tokens = decode_constrained(model, [1, 2], dfa, max_new_tokens=8)
    assert dfa_accepts(dfa, tokens), "the constrained decode produced tokens the schema rejects"


def test_decode_constrained_delegates_to_the_shared_decode_loop():
    """The pipeline supplies the logits; `constrain.greedy_decode` owns the loop.

    A pipeline that reimplements the mask/step/accept walk is a second decoder, and two
    decoders is two stopping rules and two tie-breaks. The pipeline's job at this seam is
    to be a `logits_fn`, not a decoder.
    """
    module = require(MODULE)
    decode_constrained, forward, fixture_model = api(
        MODULE, "decode_constrained", "forward", "fixture_model")
    compile_schema, greedy_decode = api(
        "reference_pipeline.constrain", "compile_schema", "greedy_decode")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    dfa = compile_schema({"enum": ["a", "b"]}, schema_vocab(cfg.vocab_size))
    prompt = [1, 2]
    expected = greedy_decode(
        dfa,
        lambda produced: [int(v) for v in np.asarray(forward(model, prompt + list(produced))[-1])],
        max_tokens=8,
    )
    assert decode_constrained(model, prompt, dfa, max_new_tokens=8) == expected


# ==============================================================================
# 8. Determinism — the product's own claim (§2), measured in the sim
# ==============================================================================

def test_the_same_inputs_produce_bit_identical_logits():
    """§2's value-determinism claim, in the sim. Same (weights, tokens, config) →
    bit-identical logits, run to run."""
    module = require(MODULE)
    forward, fixture_model = api(MODULE, "forward", "fixture_model")
    model = fixture_model(fixture_config(module))
    tokens = [1, 7, 3, 3, 11, 0, 31, 5]
    first = np.asarray(forward(model, tokens))
    for _ in range(3):
        assert np.array_equal(np.asarray(forward(model, tokens)), first)


def test_the_fixture_model_is_reproducible_from_its_config():
    """§11's fixture discipline: "a few-layer toy transformer with **pinned** weights,
    committed as text-form source". Pinned means two constructions agree — a float RNG or
    an unseeded shuffle would make the gate's own reference nondeterministic.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    cfg = fixture_config(module)
    a, b = fixture_model(cfg), fixture_model(cfg)
    for name in a.weight_names():
        assert np.array_equal(np.asarray(a.weight(name)), np.asarray(b.weight(name))), name


def test_the_fixture_model_weights_are_int8():
    """§6.1: W8 symmetric int8. A fixture whose weights are float is not exercising the
    pipeline under test."""
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    for name in model.weight_names():
        w = np.asarray(model.weight(name))
        assert np.issubdtype(w.dtype, np.integer), f"{name} is {w.dtype}"
        assert w.min() >= -128 and w.max() <= 127, f"{name} outside int8"


def test_decode_is_deterministic_run_to_run():
    """The product claim at the decode level, not just the forward."""
    module = require(MODULE)
    decode_greedy, fixture_model = api(MODULE, "decode_greedy", "fixture_model")
    model = fixture_model(fixture_config(module))
    first = decode_greedy(model, [1, 2, 3], max_new_tokens=6)
    for _ in range(3):
        assert decode_greedy(model, [1, 2, 3], max_new_tokens=6) == first


def test_batching_does_not_change_a_sequence_s_bits():
    """§8.2's batch-invariance property, asserted in the sim rather than assumed. A
    sequence's logits in a batch must equal its logits run alone — the failure mode is a
    reduction that crosses the batch dimension, which a vectorized rewrite introduces
    naturally and silently.
    """
    module = require(MODULE)
    forward, forward_batch, fixture_model = api(
        MODULE, "forward", "forward_batch", "fixture_model")
    model = fixture_model(fixture_config(module))
    sequences = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    batched = forward_batch(model, sequences)
    for i, seq in enumerate(sequences):
        alone = np.asarray(forward(model, seq))
        assert np.array_equal(np.asarray(batched[i]), alone), f"sequence {i} moved in a batch"


# ==============================================================================
# 9. Parity vs the float reference (§13 item 2)
# ==============================================================================

def test_integer_pipeline_tracks_the_float_reference_per_layer():
    """§13 item 2's converter parity gate, at the sim's scale: the integer pipeline's
    per-layer output against the float reference on probe inputs, within a stated
    tolerance.

    `red-uncalibrated` (test-design record §6): §13.2 sets the tolerances "from spike
    data", so the threshold is an OUTPUT of T-066, not an input to it. The bound below is
    a placeholder wide enough to catch a structurally wrong layer (a transposed head, a
    missing residual) and too wide to certify quantization damage — which is the spike's
    measurement, not this cell's.
    """
    module = require(MODULE)
    forward_layers, forward_layers_float, fixture_model = api(
        MODULE, "forward_layers", "forward_layers_float_reference", "fixture_model")
    model = fixture_model(fixture_config(module))
    tokens = [1, 7, 3, 11, 0]
    for index, (integer_out, float_out) in enumerate(
            zip(forward_layers(model, tokens), forward_layers_float(model, tokens))):
        a = np.asarray(integer_out, dtype=np.float64)
        b = np.asarray(float_out, dtype=np.float64)
        assert a.shape == b.shape, f"layer {index} shape"
        denominator = max(float(np.abs(b).max()), 1.0)
        assert float(np.abs(a - b).max()) / denominator <= 0.25, f"layer {index} diverged"


# ==============================================================================
# 9. G-2 — the float path is faithful to the UPSTREAM model (§13 item 2a)
# ==============================================================================
#
# §13 item 2a: the pipeline's greedy token sequence "also agrees, >= threshold, with the
# **upstream model's own framework** (HF `transformers`, fp/bf16) on the reference prompts
# — proving the integer pipeline is faithful to the *source model*, not merely
# self-consistent with our own float reference… it catches a faithful-looking but wrong
# reimplementation (wrong RoPE θ, attention scale, QK-norm order) that self-parity ships
# green."
#
# We are hand-implementing Qwen2.5's forward in numpy. Every one of θ, the attention scale,
# QK-norm order, the RoPE pair layout, the SwiGLU gate/up order, and the tied-embedding
# transpose is a place to be wrong in a way that runs, is deterministic, and is not Qwen.
# §13 item 2's parity gate cannot see any of it: it compares our integer path to OUR float
# path, and both would be wrong together.
#
# **These cells gate the integer path.** Until the float path reproduces upstream, no number
# the integer path produces can be attributed to quantization — which is the whole of T-066.
#
# ---- The tolerance, measured rather than chosen (Qwen2.5-1.5B, 2026-07-15) ----
#
#   noise floor, HF fp32 vs HF fp64   logits 2.07e-6   hidden 8.25e-6
#   the D-SLM38 defect (θ 1e6 -> 1e4) logits 1.27e-1   hidden 2.18e-2 (already at layer 1)
#   separation                        ~61,000x on logits
#
# UPSTREAM_REL_TOL = 1e-3 sits ~480x above the float noise and ~130x below the defect. Our
# float64 path against an HF fp32 reference should land at the noise floor; a wrong
# architecture misses by two orders of magnitude. This is NOT §13.2's spike-calibrated
# tolerance — that one measures quantization damage and is an OUTPUT of T-066. This is
# float-vs-float, where the bound is a numerical fact and derivable today.
#
# ---- Why the LOGIT cell leads and the TOKEN cell cannot stand alone ----
#
# Measured: on a single 8-token forward the wrong θ moved the logits 13% and **did not flip
# the argmax at all**. Over 32-token greedy generations it agreed 33.3% — and on one prompt
# the first divergence was at **token 23**. So §13 2a's "token sequence agrees >= threshold"
# is a real oracle only with enough generated tokens, and a short probe ships the defect.
# The continuous oracle leads; the token cell is scoped to a length the measurement supports.

UPSTREAM_REL_TOL = 1e-3
UPSTREAM_GREEDY_TOKENS = 32
UPSTREAM_PROMPTS = (
    "Book me with Rin, Thursday afternoon.",
    "Anyone but Mox.",
    "What's the wifi password?",
)


def qwen_checkpoint():
    """The cached Qwen2.5-1.5B snapshot, or None. §16's spike nominee."""
    # T-2152 (outside strike item 6): no private filesystem path fallback -- HF_HOME is the
    # standard HuggingFace cache location env var; an unset one means this opt-in test skips.
    for root in (os.environ.get("HF_HOME"),):
        if not root:
            continue
        found = sorted(pathlib.Path(root).glob(
            "hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/config.json"))
        if found:
            return found[0].parent
    return None


def upstream_required():
    """G-2 needs a real checkpoint and the upstream library — that is the point of it.

    Skips cleanly when either is absent so the suite runs on a bare box, and the spike run
    is what must not skip. A `skip` here is a statement that G-2 was not exercised, which is
    exactly what it is.
    """
    checkpoint = qwen_checkpoint()
    if checkpoint is None:
        pytest.skip("Qwen2.5-1.5B not cached; G-2 needs the real checkpoint")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return checkpoint


def upstream_reference(checkpoint, token_ids):
    """HF's own fp32 forward — the upstream §13 2a names. Returns (logits, hidden_states)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(torch.tensor([list(token_ids)]), output_hidden_states=True)
    logits = out.logits[0].double().numpy().copy()
    hidden = [h[0].double().numpy().copy() for h in out.hidden_states]
    del model
    return logits, hidden


@pytest.mark.upstream
def test_upstream_theta_is_absent_from_the_library_config():
    """D-SLM38, re-confirmed against the installed library rather than cited.

    Measured 2026-07-15 at this transformers version: `cfg.rope_theta` is **ABSENT** on the
    Qwen2.5 config object, while `config.json` carries `rope_theta: 1e6`. A converter on the
    4.x idiom reads nothing and defaults — and the defect is worth exactly what §11 says:
    the model loads, runs, generates fluent text, is deterministic, and is not Qwen.

    Measured cost of that default: logits move 13%, greedy tokens agree 33%.
    """
    checkpoint = upstream_required()
    from transformers import AutoConfig

    body = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert body["rope_theta"] == 1000000.0, "config.json is the normative source (C15)"

    cfg = AutoConfig.from_pretrained(str(checkpoint))
    assert getattr(cfg, "rope_theta", None) != body["rope_theta"], (
        "cfg.rope_theta now agrees with config.json at this library version. D-SLM38's rule "
        "stands regardless — the Config surface is version-unstable across majors, which is "
        "why §11 forbids it as a source — but this cell's premise has moved and the record "
        "should say so."
    )


@pytest.mark.upstream
def test_float_path_matches_upstream_logits():
    """**The G-2 cell.** Our float path, on real Qwen weights, against HF's own fp32 forward.

    Bound 1e-3: ~480x above the measured float noise floor (2.07e-6) and ~130x below the
    measured cost of a wrong θ (1.27e-1). A correct reimplementation lands at the noise
    floor; every architecture defect §13 2a names misses by orders of magnitude.
    """
    checkpoint = upstream_required()
    load_model, forward_float = api(MODULE, "load_model", "forward_float_reference")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    token_ids = tokenizer(UPSTREAM_PROMPTS[0], return_tensors=None)["input_ids"]

    expected, _ = upstream_reference(checkpoint, token_ids)
    got = np.asarray(forward_float(load_model(checkpoint), token_ids), dtype=np.float64)

    assert got.shape == expected.shape, f"{got.shape} != upstream {expected.shape}"
    relative = np.abs(got - expected).max() / np.abs(expected).max()
    assert relative <= UPSTREAM_REL_TOL, (
        f"the float path diverges from upstream by {relative:.3e} (bound {UPSTREAM_REL_TOL:.0e}). "
        f"The float noise floor is 2.07e-6, so this is an architecture defect, not precision — "
        f"check θ, the attention scale, QK-norm order, the RoPE pair layout, the SwiGLU "
        f"gate/up order, and the tied-embedding transpose"
    )


@pytest.mark.upstream
def test_float_path_matches_upstream_hidden_states_per_layer():
    """§13 item 2's "per-layer error within stated tolerances", against upstream.

    The cell that turns "the model is wrong" into "layer 7 is wrong" — which is the whole
    debugging value, and the reason to ask HF for hidden states rather than logits alone.
    Measured: the wrong θ is already visible at layer 1 (2.18e-2 vs a 8.25e-6 noise floor),
    so the first diverging layer names the defect.
    """
    checkpoint = upstream_required()
    load_model, forward_float_layers = api(MODULE, "load_model", "forward_float_layers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    token_ids = tokenizer(UPSTREAM_PROMPTS[0], return_tensors=None)["input_ids"]

    _, expected = upstream_reference(checkpoint, token_ids)
    got = list(forward_float_layers(load_model(checkpoint), token_ids))

    assert len(got) == len(expected), (
        f"{len(got)} hidden states vs upstream's {len(expected)}; upstream reports the "
        f"embedding output plus one per layer"
    )
    for index, (ours, theirs) in enumerate(zip(got, expected)):
        ours = np.asarray(ours, dtype=np.float64)
        relative = np.abs(ours - theirs).max() / max(np.abs(theirs).max(), 1e-12)
        assert relative <= UPSTREAM_REL_TOL, (
            f"hidden state {index} diverges from upstream by {relative:.3e}; the first "
            f"diverging layer is where the reimplementation is wrong"
        )


@pytest.mark.upstream
def test_float_path_matches_upstream_greedy_tokens():
    """§13 item 2a's stated form: the greedy token sequence agrees with upstream.

    **Scoped to a length the measurement supports, and it is the secondary oracle.** Measured
    on a single 8-token forward, a wrong θ moved the logits 13% and did not flip the argmax
    at all; over 32-token generations it agreed 33.3%, with one prompt's first divergence at
    token 23. A token cell on a short probe ships the defect, so this one generates 32 tokens
    across three prompts and the logit cell above carries the primary claim.

    Exact agreement is asserted rather than a threshold: our float64 path differs from an HF
    fp32 reference by ~2e-6, so a flip requires a genuine near-tie. **A disagreement here is
    a finding to investigate — a near-tie or a defect — not a bound to loosen.**

    **This cell called `decode_greedy` until 2026-07-15 and that was unsatisfiable by
    construction** (record P-16). §6.5 pins `decode_greedy` to the **integer** path, so the
    cell asserted that the integer path's greedy tokens match an fp32 reference *exactly* —
    that quantization does nothing. Its name, its section and its message all said float; its
    call said integer. A cell that can never pass is worse than a wrong bound, because the
    eventual "fix" is to loosen it. Repointed to `decode_greedy_float`.

    The integer half of §13 2a — the integer path within a *threshold* of upstream — is a
    separate claim whose threshold is an OUTPUT of T-066 (§13.2), and it stays owed exactly
    as the record's §4 ruled.
    """
    checkpoint = upstream_required()
    load_model, decode_greedy_float = api(MODULE, "load_model", "decode_greedy_float")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = load_model(checkpoint)

    for prompt in UPSTREAM_PROMPTS:
        token_ids = tokenizer(prompt, return_tensors=None)["input_ids"]
        expected = upstream_greedy(checkpoint, token_ids, UPSTREAM_GREEDY_TOKENS)
        got = list(decode_greedy_float(model, token_ids, max_new_tokens=UPSTREAM_GREEDY_TOKENS))
        assert got == expected, (
            f"{prompt!r}: the float path's greedy tokens diverge from upstream at position "
            f"{next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)}. A wrong θ "
            f"measures 33% agreement here; float precision alone measures 100%"
        )


def upstream_greedy(checkpoint, token_ids, count):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.float32).eval()
    ids = torch.tensor([list(token_ids)])
    out = []
    with torch.no_grad():
        for _ in range(count):
            nxt = int(model(ids).logits[0, -1].argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    del model
    return out


# ==============================================================================
# 10. Real-weight loading — §11's reject-over-degrade, applied to the checkpoint
# ==============================================================================

@pytest.mark.upstream
def test_load_model_derives_scales_from_each_real_tensor():
    """§6.1 on real weights: each projection's per-output-channel scales come from **that
    tensor's own** per-channel max-abs.

    The fixture model's weights are a bit-mix of each tensor's name, so its per-channel
    ranges are synthetic and near-uniform. A loader that carried a constant scale over from
    the fixture would pass every §3a shape cell on real weights and quantize Qwen with a
    number derived from a string.
    """
    checkpoint = upstream_required()
    load_model = api(MODULE, "load_model")
    model = load_model(checkpoint)
    for name in model.scales.requant_sites():
        scales = model.scales.weight_scales(name)
        assert len(set(scales)) > 1, (
            f"{name}: every output channel got the same scale from a real trained weight; "
            f"the scales are not derived from the tensor"
        )
        assert all(s > 0.0 for s in scales), f"{name}: a zero scale erases a channel"


def _write_synthetic_safetensors(path, tensors):
    """The minimal on-disk shape `pipeline._SafeTensors` reads: an 8-byte little-endian
    header length, the JSON header (name -> {dtype, shape, data_offsets}), then the raw
    bytes back to back. F32 so no BF16 reinterpretation is in play."""
    header = {}
    payload = bytearray()
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        data = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    header["__metadata__"] = {}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(bytes(payload))


def test_load_model_rejects_an_unmapped_tensor(tmp_path):
    """§11's reject-over-degrade discipline (N3): "an unrecognized … constant in the config
    is a hard rejection, never a silent drop". The same rule governs the weight map.

    A checkpoint tensor the loader does not recognize is either a model we do not implement
    or a mapping bug. **A silently-skipped projection is a model that runs and is not
    Qwen** — the same failure shape as the wrong θ, arriving through the name map.

    Needs a real `config.json` and a readable `.safetensors` file on disk: `load_model`
    reads `config.json` before it ever reaches the unmapped-tensor check, so a `tmp_path`
    with neither raises `ConfigError` (a `ValueError` subclass) three calls earlier and
    never exercises the check this test names — confirmed by execution, T-1915 finding F2.
    """
    load_model, unsupported = api(MODULE, "load_model", "UnsupportedOpSet")
    write_config(tmp_path)
    _write_synthetic_safetensors(tmp_path / "model.safetensors", {})
    with pytest.raises(unsupported, match="mystery_proj"):
        load_model(tmp_path, extra_tensors={"model.layers.0.self_attn.mystery_proj.weight": None})


@pytest.mark.upstream
def test_load_model_rejects_a_missing_tensor():
    """The other direction, and the more dangerous one: a projection the map names and the
    checkpoint lacks. Loading must fail, not substitute an identity or a zero."""
    checkpoint = upstream_required()
    load_model = api(MODULE, "load_model")
    with pytest.raises((KeyError, ValueError)):
        load_model(checkpoint, require_tensors=("model.layers.0.self_attn.absent_proj.weight",))


@pytest.mark.upstream
def test_load_model_shapes_match_config_json():
    """§11: the converter "verifies the op set" against `config.json`. A tensor whose shape
    contradicts the config is a different model wearing the config's name."""
    checkpoint = upstream_required()
    load_model, load_config = api(MODULE, "load_model", "load_config")
    cfg = load_config(checkpoint / "config.json")
    model = load_model(checkpoint)
    assert model.config.hidden_size == cfg.hidden_size == 1536
    assert model.config.num_hidden_layers == cfg.num_hidden_layers == 28
    assert model.config.vocab_size == cfg.vocab_size == 151936
    embedding = np.asarray(model.weight("embed"))
    assert embedding.shape == (cfg.vocab_size, cfg.hidden_size)


@pytest.mark.upstream
def test_load_model_ties_the_lm_head_to_the_embedding():
    """§5/D-SLM8's op set includes tied embeddings and Qwen2.5-1.5B ties
    (`tie_word_embeddings: true`). §6.5 keeps it at one tensor scale — measured, a
    per-channel lm_head scale picks a different token in 80% of draws."""
    checkpoint = upstream_required()
    load_model = api(MODULE, "load_model")
    lm_head_weight, embedding_weight = api(MODULE, "lm_head_weight", "embedding_weight")
    model = load_model(checkpoint)
    assert model.config.tie_word_embeddings is True
    assert np.array_equal(np.asarray(lm_head_weight(model)), np.asarray(embedding_weight(model)))


@pytest.mark.upstream
def test_load_model_keeps_the_original_float_weights_for_the_reference():
    """The float reference is the **unquantized** model, not a dequantized int8 one.

    §13 item 2 compares "integer pipeline outputs vs the float reference" — the reference is
    what quantization is measured *against*, so dequantizing the int8 weights back to float
    would compare the integer path to itself and report zero damage. It is also what makes
    G-2 possible: upstream has the original weights, so a float path built from dequantized
    int8 could never match it to 1e-3.

    This is the memory finding too (record P-12): dequantizing 1.5B weights to float64 at
    once is 11.5 GiB. The originals are already float and stream per-tensor.
    """
    checkpoint = upstream_required()
    load_model = api(MODULE, "load_model")
    model = load_model(checkpoint)
    name = model.scales.requant_sites()[0]
    tensor = name.split(".requant")[0]
    original = np.asarray(model.float_weight(tensor), dtype=np.float64)
    quantized = np.asarray(model.weight(tensor), dtype=np.int64)
    assert original.shape == quantized.shape
    assert not np.array_equal(original, quantized.astype(np.float64)), (
        "float_weight returned the int8 weights; the float reference must be the "
        "unquantized model or the parity gate measures nothing"
    )


# ==============================================================================
# 11. The KV cache — a determinism cell, not a perf cell (§13 item 4)
# ==============================================================================
#
# §13 item 4 makes invariance a first-class suite: chunk-size invariance,
# batch-composition invariance, restore-exactness. A KV cache is the same class of claim in
# miniature — it is a "pure optimization" that is trivially a *different computation* if any
# reduction order, any requant, or any position index moves. The cache exists because
# re-prefilling a 466-token prompt per generated token projected the run at ~150 hours; that
# is a real reason and it is not a reason to accept different bits.


def test_cached_decode_is_bit_identical_to_uncached():
    """**The cache's whole contract.** Same (weights, tokens, schema) -> the same bits, with
    and without the cache.

    A cache that changes one bit is not an optimization, it is a second pipeline — and the
    spike would be measuring whichever one happened to run.
    """
    module = require(MODULE)
    forward, fixture_model = api(MODULE, "forward", "fixture_model")
    model = fixture_model(fixture_config(module))
    tokens = [1, 7, 3, 3, 11, 0, 31, 5]
    uncached = np.asarray(forward(model, tokens, cache=None))
    cache = api(MODULE, "new_kv_cache")(model)
    cached = np.asarray(forward(model, tokens, cache=cache))
    assert np.array_equal(cached, uncached)


def test_cached_incremental_decode_matches_a_full_uncached_forward_at_every_step():
    """The property the cache actually has to hold: feeding tokens one at a time through a
    cache must equal re-running the whole prefix uncached, **at every prefix length**.

    One bit of divergence at step k propagates to every later token, so a cell that only
    checked the final logits would report one failure for a defect that started ten tokens
    earlier — and a cell that only checked step 1 would pass a cache that drifts.
    """
    module = require(MODULE)
    forward, fixture_model, new_cache = api(
        MODULE, "forward", "fixture_model", "new_kv_cache")
    model = fixture_model(fixture_config(module))
    tokens = [1, 7, 3, 3, 11, 0, 31, 5]
    cache = new_cache(model)
    for step in range(1, len(tokens) + 1):
        incremental = np.asarray(forward(model, tokens[step - 1:step], cache=cache))
        full = np.asarray(forward(model, tokens[:step], cache=None))
        assert np.array_equal(incremental[-1], full[-1]), (
            f"cached decode diverged from the uncached prefix at token {step - 1}"
        )


def test_a_fresh_cache_cannot_inherit_a_previous_sequence():
    """§17 dimension 1's poison property, in the sim: a reused cache must not leak the last
    sequence into the next. A cache that is merely *reset* rather than *cleared* produces a
    second sequence conditioned on a prefix that is not in its prompt — which reads as a
    quality result."""
    module = require(MODULE)
    forward, fixture_model, new_cache = api(
        MODULE, "forward", "fixture_model", "new_kv_cache")
    model = fixture_model(fixture_config(module))
    first, second = [1, 7, 3, 3], [11, 0, 31, 5]

    dirty = new_cache(model)
    forward(model, first, cache=dirty)
    dirty.reset()
    reused = np.asarray(forward(model, second, cache=dirty))

    clean = np.asarray(forward(model, second, cache=new_cache(model)))
    assert np.array_equal(reused, clean), "a reset cache carried the previous sequence"


def test_the_cache_does_not_change_the_constrained_decode():
    """§15 scores under schema-constrained greedy decoding, so the cache has to be
    invariant *there* — the decode that produces the spike's number, not just `forward`."""
    module = require(MODULE)
    decode_constrained, fixture_model = api(MODULE, "decode_constrained", "fixture_model")
    compile_schema = api("reference_pipeline.constrain", "compile_schema")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    dfa = compile_schema({"enum": ["a", "b"]}, schema_vocab(cfg.vocab_size))
    uncached = decode_constrained(model, [1, 2], dfa, max_new_tokens=8, use_cache=False)
    cached = decode_constrained(model, [1, 2], dfa, max_new_tokens=8, use_cache=True)
    assert cached == uncached


# ==============================================================================
# 11a. The primed-cache closure (`_logits_fn_dynamic`'s `primed_cache`/`primed_fed`,
# C9 Unit 8a, T-522) — closing T-742/S3
# ==============================================================================
#
# Neither this suite nor the harness suite
# (`Claude/Laplace/harness/softmax_iexp/test_campaign.py`) had ANY test that drives
# `_logits_fn_dynamic`'s primed branch (`Claude/Poirot/584ff3e-c9-unit8-primed-driver-cvgate-
# review-2026-07-25.md`, finding S3). Discarding `primed_fed` (`fed: list[int] = []` in place of
# `list(primed_fed) if primed_fed is not None else []`, pipeline.py:3237) left 1819/1819 in this
# tier and 103/103 in the harness green: with `fed` wrongly empty, the reset check
# `full[:len(fed)] != fed` reduces to `full[:0] != []`, which is never true, so no reset fires and
# the closure's first call re-forwards the WHOLE prompt -- including the tokens the primed cache
# already holds -- against a cache that already carries them. Every logit on every record of the
# funded slice would be wrong, silently. This is the sole place the primed cache's identity claim
# ("a caller who has already forwarded the shared prefix into `primed_cache` pays only for the
# divergent tail, and gets the SAME logits a full uncached forward would give") is proven by
# execution rather than assumed.


def test_logits_fn_dynamic_with_a_primed_cache_matches_the_uncached_forward_of_the_full_prompt():
    """The primed-cache closure's whole contract (T-522, C9 Unit 8a): priming a cache with a
    shared prefix and then calling `_logits_fn_dynamic` with `primed_cache`/`primed_fed` set to
    that prefix must reproduce EXACTLY the logits a full uncached forward of the complete prompt
    (prefix + divergent tail) would produce -- a feature oracle on the real forward, not a
    plumbing check on call arguments. `primed_fed` must seed the closure's `fed` tracker so its
    first call forwards only the tail; if it does not (S3), the whole prompt is re-fed into a
    cache that already holds the prefix and the two diverge.
    """
    module = require(MODULE)
    forward_dynamic, fixture_model, new_cache, logits_fn_dynamic = api(
        MODULE, "forward_dynamic", "fixture_model", "new_kv_cache", "_logits_fn_dynamic")
    model = fixture_model(fixture_config(module))
    prefix = [1, 7, 3, 3, 11]
    tail = [0, 31, 5]
    full_prompt = prefix + tail

    oracle = np.asarray(forward_dynamic(model, full_prompt, cache=None))

    primed_cache = new_cache(model)
    forward_dynamic(model, prefix, cache=primed_cache)

    fn = logits_fn_dynamic(model, full_prompt, use_cache=True,
                           primed_cache=primed_cache, primed_fed=prefix)
    got = np.asarray(fn([]))  # the closure's first call -- nothing generated yet

    assert np.array_equal(got, oracle[-1]), (
        "the primed-cache closure's first call did not reproduce the uncached forward of the "
        "full prompt -- primed_fed was not seeded into the closure's fed tracker, so either the "
        "prefix was re-forwarded into a cache that already holds it, or the cache was wrongly "
        "reset and the prefix lost"
    )


def test_logits_fn_dynamic_primed_cache_costs_only_the_divergent_tail_not_the_whole_prompt():
    """The efficiency half of the same contract, checked directly rather than inferred from
    bit-equality alone: a primed closure's first call must forward exactly the divergent TAIL
    against `forward_dynamic`, never the whole prompt (the S3 mutation feeds the whole prompt and
    still happens to be checked by the bit-equality test above, but this cell names the mechanism
    the review traced explicitly -- `full[len(fed):]` -- so a future regression that changes shape
    without changing final bits, e.g. on a degenerate model, is not invisible)."""
    module = require(MODULE)
    forward_dynamic, fixture_model, new_cache, logits_fn_dynamic = api(
        MODULE, "forward_dynamic", "fixture_model", "new_kv_cache", "_logits_fn_dynamic")
    model = fixture_model(fixture_config(module))
    prefix = [1, 7, 3, 3, 11]
    tail = [0, 31, 5]
    full_prompt = prefix + tail

    primed_cache = new_cache(model)
    forward_dynamic(model, prefix, cache=primed_cache)

    calls = []
    real_forward_dynamic = module.forward_dynamic
    def recording_forward_dynamic(model, tokens, cache=None, trace=None):
        calls.append(list(tokens))
        return real_forward_dynamic(model, tokens, cache=cache, trace=trace)
    module.forward_dynamic = recording_forward_dynamic
    try:
        fn = logits_fn_dynamic(model, full_prompt, use_cache=True,
                               primed_cache=primed_cache, primed_fed=prefix)
        fn([])
    finally:
        module.forward_dynamic = real_forward_dynamic

    assert len(calls) == 1
    assert calls[0] == tail, (
        f"the primed closure's first call forwarded {calls[0]!r}, expected only the divergent "
        f"tail {tail!r}; forwarding the prefix again means primed_fed did not seed the closure's "
        f"fed tracker"
    )


# ==============================================================================
# 12. The RoPE hoist (§6.4 / C12)
# ==============================================================================

def test_the_rope_tables_are_built_once_and_carried_by_the_model():
    """`rope_tables` was rebuilt inside every `forward()` — a pure-Python 32768x64
    `math.cos`/`math.sin` loop, ~1.3 s per token. Hoisting is required.

    The cell is not about speed: it pins that the hoisted tables are **the tables §6.4
    specifies**, so the hoist cannot quietly become a different table. C12 still governs —
    `context_cap` rows, `[0, context_cap)`, exclusive.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    rope_tables = api(ROPE, "rope_tables")
    cfg = fixture_config(module)
    model = fixture_model(cfg)

    cos_table, sin_table = model.rope_tables
    assert len(cos_table) == len(sin_table) == cfg.context_cap, (
        f"{len(cos_table)} rows for context_cap={cfg.context_cap}; §6.8 C12 pins "
        f"context_cap entries, indices [0, context_cap), EXCLUSIVE"
    )
    expected_cos, expected_sin = rope_tables(cfg.head_dim, cfg.context_cap, cfg.rope_theta)
    assert [list(r) for r in cos_table] == [list(r) for r in expected_cos]
    assert [list(r) for r in sin_table] == [list(r) for r in expected_sin]


def test_the_hoisted_tables_are_the_ones_the_forward_uses():
    """The hoist's real risk: the model carries correct tables and the forward builds its
    own anyway, or reads them at the wrong offset. Perturbing the carried tables must change
    the forward's output — if it does not, the forward is not reading them.
    """
    module = require(MODULE)
    forward, fixture_model = api(MODULE, "forward", "fixture_model")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    baseline = np.asarray(forward(model, [1, 7, 3, 11]))

    cos_table, sin_table = model.rope_tables
    poisoned = ([[0] * len(row) for row in cos_table], [list(row) for row in sin_table])
    perturbed = api(MODULE, "with_rope_tables")(model, poisoned)
    assert not np.array_equal(np.asarray(forward(perturbed, [1, 7, 3, 11])), baseline), (
        "zeroing the model's cosine table did not change the forward; the tables it uses "
        "are not the ones it carries"
    )


# ==============================================================================
# 13. Calibration is on the frozen corpus (§6.2 / D-SLM5, §11)
# ==============================================================================

# §4's six adversarial classes and their frozen-corpus counts. Calibration consumes the
# WHOLE corpus (Dan, 2026-07-15 — the full pass measured 446 s, so the subsample's saving
# did not buy anything worth a term in §11's reproducibility formula). The classes are still
# named because "every class at its corpus share" is what the stratification property became:
# it now forbids any subsample rather than describing one.
CALIBRATION_CLASSES = ("plain", "multi_slot", "negation", "correction",
                       "underspecified", "out_of_domain")
CALIBRATION_CORPUS_COUNTS = {"plain": 180, "multi_slot": 120, "negation": 90,
                             "correction": 90, "underspecified": 60, "out_of_domain": 60}


def synthetic_corpus(per_class=40):
    """A class-labelled record set. Deliberately unbalanced in *input* order so an
    order-dependent reduction has something to trip on."""
    records = []
    for index in range(per_class):
        for cls in CALIBRATION_CLASSES:
            records.append({"id": f"{cls}-{index:03d}", "class": cls,
                            "utterance": f"{cls} utterance {index}"})
    return records


def test_calibration_reads_the_frozen_corpus():
    """§6.2's baseline: "static per-tensor activation scales, calibrated offline **on a
    task-representative corpus**."

    The harness calibrated on four synthetic sequences of eight arithmetic token ids. Those
    are not task-representative and their activation ranges are not the corpus's, so every
    per-tensor activation scale in the run was set by data the spike does not measure — and
    §6.2's known cost is "quality on outlier-heavy activations", which is exactly what a
    synthetic calibration set cannot contain.
    """
    calibrate = api(MODULE, "calibrate")
    corpus_path = api(MODULE, "CALIBRATION_CORPUS_PATH")
    assert corpus_path.name == "shopkeeper_corpus_v1.jsonl", (
        f"calibration reads {corpus_path}; §6.2 requires the task-representative corpus"
    )
    assert corpus_path.exists()


# ==============================================================================
# 13a. The calibration INPUT — the same prompt the run decodes (§6.2)
# ==============================================================================
#
# **The urgent one, and it may be manufacturing T-066's answer.** The integer path produces
# zero signal on real Qwen: logits identically zero, all 151,936 tied, argmax 0 by tie-break.
# Per-layer, layer 0 has ~90% of channels alive and **layer 1 cliffs to 5.8%** — a ~400x
# massive-activation outlier (|max| 3121) sets the per-tensor scale at 3121/127 ~= 24.6 and
# ~95% of channels quantize to zero and stay there. That is §6.2's own predicted cost
# ("quality on outlier-heavy activations") arriving in full — **and it is not yet an answer**,
# because the scales may be set by data the run does not resemble.
#
# **Measured 2026-07-15**, calibration input vs run input, per §4 class:
#
#   class            bare   run    ratio
#   plain               8   466    58.2x
#   multi_slot         17   475    27.9x
#   negation            5   463    92.6x
#   correction         15   475    31.7x
#   underspecified      5   463    92.6x
#   out_of_domain       6   464    77.3x
#
# Calibration encodes `_record_text` — the bare utterance, or the joined `turns` — at **5-17
# tokens**. The run decodes `apply_chat_template(SYSTEM_PROMPT + build_prompt(record))` at
# **463-475**. So the scales are set on ~1-2% of the token content the run processes, and the
# system prompt — 450 tokens the calibration never sees — is where a massive activation would
# most plausibly live.
#
# §6.2 requires calibration "on a **task-representative** corpus". Text the run never sees is
# not task-representative. `load_model(..., tokenize=)` is parameterized for exactly this and
# **nothing pinned which it should be** — a reproducible-path constant set by an unstated
# implementation detail, which is C17's shape and this session's recurring one.


def corpus_record_per_class():
    """One frozen-corpus record of each §4 class — including `correction`, which carries
    `turns` rather than `utterance` and is the shape a text reader drops or crashes on."""
    text = CALIBRATION_CORPUS_FILE.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    seen = {}
    for record in records:
        seen.setdefault(record["class"], record)
    return [seen[cls] for cls in CALIBRATION_CLASSES]


def _calibration_corpus_file():
    # Read from reference_pipeline.pipeline's own CALIBRATION_CORPUS_PATH rather than
    # re-deriving a second, independent path here (T-2123/T-2137 B3: a second
    # `Path(__file__).resolve().parents[N]`-style derivation is exactly the silent-relocation
    # fragility the vendoring design names, §2.2/§3.4).
    return require(MODULE).CALIBRATION_CORPUS_PATH


CALIBRATION_CORPUS_FILE = _calibration_corpus_file()


def test_calibration_and_the_run_encode_the_same_prompt():
    """**The cell that matters right now.** §6.2: calibration is "on a task-representative
    corpus" — so the tokens calibration observes must be the tokens the run decodes.

    Measured, they are not: 5-17 tokens against 463-475, a 28x-93x gap. The activation
    ranges that set every per-tensor scale were measured on ~1-2% of what the run processes,
    and the 450-token system prompt the run decodes is invisible to them. That is enough to
    manufacture the layer-1 cliff, and enough to make it un-attributable either way — which
    is the failure this whole harness exists to prevent.

    Asserted on the **token ids**, not the text: the tokenizer is part of the shape, and two
    encoders producing the same string through different tokenizers are still two inputs.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    calibration_token_ids, run_token_ids = api(
        MODULE, "calibration_token_ids", "run_token_ids")
    model = fixture_model(fixture_config(module))
    for record in corpus_record_per_class():
        calibration_ids = list(calibration_token_ids(model, record))
        run_ids = list(run_token_ids(model, record))
        assert calibration_ids == run_ids, (
            f"class {record['class']!r}: calibration observes {len(calibration_ids)} tokens "
            f"and the run decodes {len(run_ids)}. A scale calibrated on text the run never "
            f"sees is not task-representative (§6.2), and the difference is where the "
            f"massive activation lives"
        )


def test_the_run_decodes_the_chat_template_not_the_bare_utterance():
    """The guard against satisfying the equality cell by collapsing **both** paths to the
    bare utterance.

    The checkpoint is Qwen2.5-1.5B-**Instruct** and §15's task is schema-constrained intent
    extraction, which the run drives through a system prompt and the chat template.

    **The bound's margins, restated for the encoder this cell actually drives (P-18,
    2026-07-15).** The 5-17 / 463-475 figures are the *checkpoint tokenizer's*, and this cell
    runs on the **fixture**, whose encoder is one token per byte of the rendered prompt.
    Measured there: the run's prompt is **1786-1885** tokens and a bare-utterance collapse
    would render 8-60, so the bound sits ~1.7x above the widest collapsed record and ~18x
    below the narrowest run prompt. The assertion is the same and the guard still bites; the
    original margins justified it against a tokenizer this cell never calls.

    **This bound is not what makes the fixture's calibration content-blind** — the truncation
    is `_calibrate`'s cap slice, not this cell's floor (P-18/F-1). Raising the fixture's
    `context_cap` to admit the prompt is not the escape: measured, `fixture_model` at
    `context_cap=2048` takes **540 s** to build because it calibrates all 600 records at
    construction, and ~40 cells build one.
    """
    module = require(MODULE)
    fixture_model, run_token_ids = api(MODULE, "fixture_model", "run_token_ids")
    model = fixture_model(fixture_config(module))
    for record in corpus_record_per_class():
        length = len(list(run_token_ids(model, record)))
        assert length > 100, (
            f"class {record['class']!r}: the run's prompt is {length} tokens. The bare "
            f"utterance measures 5-17 and the chat-template prompt 463-475; both paths "
            f"agreeing on the bare shape is the equality cell satisfied and the run wrong"
        )


def test_the_calibration_tokenization_is_recorded_in_the_artifact():
    """A reproducible-path constant set by an implementation detail is C17's shape, so it is
    recorded rather than left in the converter's source. `load_model(..., tokenize=)` is a
    parameter; which parameter produced the scales is a term reproducibility must be able to
    read."""
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    tokenization = model.calibration.tokenization
    assert isinstance(tokenization, str) and tokenization, (
        "the artifact does not record which tokenization produced its scales; the same "
        "checkpoint and the same hashed corpus under two encoders give two artifacts"
    )


# ==============================================================================
# 13b. Calibration consumes the WHOLE frozen corpus (Dan, 2026-07-15)
# ==============================================================================
#
# The full pass measured **446 s (7.4 min)**, so the stratified subsample's saving did not
# buy anything worth a term in §11's reproducibility formula. **The rule, N, and the §6.8 row
# P-15 called for all dissolve**: the corpus hash alone determines the scales and §11 is true
# as written, with no new term.
#
# The three properties stay and now guard the property rather than a rule. A whole-corpus
# max-abs satisfies all three trivially — which is the point. They forbid a subsample from
# returning, and "trivially satisfied by the right design" is what a good invariant looks
# like, not a reason to delete it.


def test_calibration_consumes_every_class_at_its_full_corpus_share():
    """**What the stratification cell became.** Calibration reads the whole frozen corpus, so
    every §4 class appears at its full count — 180/120/90/90/60/60, 600 records.

    Stronger than the 20-per-class cell it replaces, and it forbids *any* subsample rather
    than describing one: a draw that ignores class fails, a stratified draw fails, and the
    rule that was P-15's whole problem cannot come back without failing here.

    §4's classes are why this matters: negation and correction are where the hard activations
    live, and §6.2's known cost is precisely quality on outlier-heavy activations.
    """
    calibration_records = api(MODULE, "calibration_records")
    records = list(calibration_records())
    counts = {cls: sum(1 for r in records if r["class"] == cls) for cls in CALIBRATION_CLASSES}
    assert counts == CALIBRATION_CORPUS_COUNTS, (
        f"calibration consumed {counts}, not the whole corpus {CALIBRATION_CORPUS_COUNTS}; a "
        f"subsample re-introduces a rule §11's formula cannot carry (P-15)"
    )
    assert len(records) == 600


def test_no_subsample_rule_or_count_is_recorded():
    """P-15 dissolved, so the terms it required must be **gone**, not retained and inert.

    §6.6: HEAD holds current truth. A recorded rule that no longer selects anything is the
    exact shape P-15 warned about — a term in the artifact that reproducibility would read
    and that nothing depends on — and it is how a subsample creeps back.
    """
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    for gone in ("records_per_class", "rule"):
        assert not hasattr(model.calibration, gone), (
            f"the artifact still records {gone!r}; the full-corpus ruling removed the "
            f"subsample, so the term is inert and §11 needs no slot for it"
        )
    for gone in ("CALIBRATION_RECORDS_PER_CLASS", "CALIBRATION_RULE", "calibration_subsample"):
        assert not hasattr(module, gone), f"{gone} survives the full-corpus ruling"


def test_the_calibration_corpus_hash_is_recorded_in_the_artifact():
    """§11's formula carries the calibration-corpus hash, and with the whole corpus consumed
    that hash is now **sufficient**: it determines the scales on its own. That is what P-15's
    dissolution bought — a term removed from the formula rather than added to it."""
    module = require(MODULE)
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    assert model.calibration.corpus_sha256 == CORPUS_SHA256
    assert tuple(model.calibration.classes) == CALIBRATION_CLASSES


def test_calibration_is_deterministic_across_processes():
    """"Deterministic and stated", not reached through iteration order or a seed that lives
    nowhere.

    In-process repetition cannot see either: a set-iteration-order reduction is stable within
    a process, and a seeded RNG is stable while its seed does not move. A fresh interpreter
    with a different `PYTHONHASHSEED` sees both — Python randomises `str` hashing per process.

    A whole-corpus max-abs passes trivially, which is the design being right rather than the
    cell being weak: it is what fails a running mean, an early stop, or a top-k over a stream.

    **Driven through `in_cap_tokenize` (P-18, 2026-07-15)** — under the fixture's default
    encoder every record truncates to the same 16-byte prefix, so the reduction sees one
    sequence and no selection-rule defect is reachable. The subprocess imports the same
    helper from `conftest` rather than restating it: two constructions that must agree are a
    construction that will eventually not.
    """
    import subprocess
    import sys

    module = require(MODULE)
    calibrate, fixture_model = api(MODULE, "calibrate", "fixture_model")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    records = synthetic_corpus(per_class=4)
    here = repr(calibrate(model, records, tokenize=in_cap_tokenize(cfg)))

    script = (
        "import sys;"
        f"sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parents[2])!r});"
        f"sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parent)!r});"
        "import reference_pipeline.pipeline as p;"
        "from conftest import in_cap_tokenize;"
        f"cs={CALIBRATION_CLASSES!r};"
        "rs=[{'id':f'{c}-{i:03d}','class':c,'utterance':f'{c} utterance {i}'}"
        " for i in range(4) for c in cs];"
        f"cfg=p.ModelConfig(**{fixture_config_kwargs()!r});"
        "print(repr(p.calibrate(p.fixture_model(cfg), rs, tokenize=in_cap_tokenize(cfg))))"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert out.returncode == 0, f"the subprocess failed: {out.stderr[-400:]}"
    assert out.stdout.strip() == here, (
        "the calibrated scales changed under a different PYTHONHASHSEED; the reduction is "
        "reached through set/dict iteration order or an unstated seed"
    )


def test_calibration_is_pinned_to_the_frozen_corpus_bytes():
    """§11's reproducibility formula: "converter version + checkpoint hash + config +
    **calibration-corpus hash** -> identical bytes". The corpus hash is a term in it, so the
    calibration input is pinned by hash exactly as the scorer's is (harness record §5)."""
    import hashlib
    corpus_path = api(MODULE, "CALIBRATION_CORPUS_PATH")
    digest = hashlib.sha256(corpus_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == CORPUS_SHA256, (
        f"the calibration corpus is {digest}, not the frozen {CORPUS_SHA256}; §11 puts this "
        f"hash in the reproducibility formula"
    )


def test_calibration_observes_every_record_not_only_the_first():
    """**The feature oracle §6.2's calibration claim did not have** (P-18, 2026-07-15).

    §6.2's achievement claim is that the scales are *calibrated on* the corpus — the max-abs
    of the activations **the records drive**. Every other calibration cell is a consistency
    or a shape claim: `..._reads_the_frozen_corpus` checks a path, `..._consumes_every_class_
    at_its_full_corpus_share` checks `calibration_records()` before a forward runs,
    `..._is_pinned_to_the_frozen_corpus_bytes` checks a hash, and the two reduction cells
    assert the fold does the *same* thing twice. **None of them executes the feature and
    asserts that the corpus's content reached the scales** — so a `_calibrate` that observed
    one record and ignored 599 satisfied all of them. Measured under the fixture's default
    encoder, it does: `calibrate(one record) == calibrate(all 600)`.

    Grounded in max-abs's **definition**, not in a recode of `_calibrate`'s steps: a max-abs
    over a record set distributes over union, so

        scales(A + B) == elementwise_max(scales(A), scales(B))

    An early stop, a subsample, a running mean and a top-k over a stream each break that
    identity. Measured: the union identity holds exactly against a correct max-abs (24 of 28
    site/role pairs strictly exceed one input, so the assertion has something to bite on) and
    catches both seeded mutants.
    """
    module = require(MODULE)
    calibrate, fixture_model = api(MODULE, "calibrate", "fixture_model")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    tokenize = in_cap_tokenize(cfg)

    first = [{"id": f"a{i}", "utterance": "x"} for i in range(4)]
    second = [{"id": f"b{i}", "utterance": "x"} for i in range(4)]
    scales_first = calibrate(model, first, tokenize=tokenize)
    scales_second = calibrate(model, second, tokenize=tokenize)
    scales_both = calibrate(model, first + second, tokenize=tokenize)

    assert scales_first != scales_second, (
        "two disjoint record sets calibrated to identical scales; the calibration is not "
        "observing the records at all and every cell below is vacuous"
    )

    strict = 0
    for name in scales_both.requant_sites():
        for role in ("input_scale", "output_scale"):
            a = getattr(scales_first, role)(name)
            b = getattr(scales_second, role)(name)
            both = getattr(scales_both, role)(name)
            assert both == max(a, b), (
                f"{name}.{role}: calibrating on A+B gave {both}, not max({a}, {b}). §6.2's "
                f"scales are a max-abs over the corpus, and a max-abs distributes over "
                f"union — an early stop, a subsample or a running mean does not"
            )
            if both > min(a, b):
                strict += 1
    assert strict > 0, (
        "no site's union scale strictly exceeded either input, so the identity above is "
        "satisfied by every reduction and this cell cannot fail"
    )


def test_calibration_is_order_independent():
    """§11 again: the formula admits a corpus *hash*, not a corpus *order*, so two runs over
    the same frozen corpus must produce identical scales however the records are visited.

    This is the base converter's sibling of Japp's GAP-7. A max-abs reduction is
    order-independent by construction — which is the point: the cell proves the calibration
    IS that reduction and has not acquired an order-dependent step (a running mean, an
    early-stop, a top-k over a stream).

    **Driven through `in_cap_tokenize`, and the isolation is the cell (P-18, 2026-07-15).**
    Under the fixture's default encoder every record truncates to the same 16-byte
    system-prompt prefix, so *any* fold over them agrees with itself and this cell passed an
    early stop, a `first_n` draw and a running mean — 0 of the 3 defects it names. The
    reduction is this cell's subject; the encoder is not. See `conftest.in_cap_tokenize`.
    """
    module = require(MODULE)
    calibrate, fixture_model = api(MODULE, "calibrate", "fixture_model")
    cfg = fixture_config(module)
    model = fixture_model(cfg)
    tokenize = in_cap_tokenize(cfg)
    records = [{"id": f"r{i}", "utterance": f"utterance {i}"} for i in range(16)]
    forward_order = calibrate(model, records, tokenize=tokenize)
    reverse_order = calibrate(model, list(reversed(records)), tokenize=tokenize)
    assert forward_order == reverse_order, (
        "the calibrated scales depend on the order the corpus was visited; §11's "
        "reproducibility formula carries a corpus hash, not a corpus order"
    )


def test_calibration_does_not_materialise_every_weight_at_once():
    """Record P-12. `_calibrate` -> `_float_forward` -> `_float_weights` dequantized every
    weight to float64 simultaneously: **11.50 GiB for 1.5B params**, which does not run.

    Pinned as a contract rather than a memory assertion (a byte bound would be brittle and
    machine-specific): the float path takes weights **one tensor at a time** from
    `float_weight`, so a whole-model dequant is not the shape it can be built in. The
    sentinel is that requesting all weights at once is not part of the surface.
    """
    module = require(MODULE)
    assert not hasattr(module, "_float_weights") and not hasattr(module, "float_weights"), (
        "a whole-model float weight materialiser exists; 1.5B params at float64 is 11.5 GiB "
        "and the float path must stream per tensor (P-12)"
    )
    fixture_model = api(MODULE, "fixture_model")
    model = fixture_model(fixture_config(module))
    assert callable(model.float_weight), "the float path takes one tensor at a time"


def test_the_float_reference_is_not_the_integer_pipeline():
    """The negative control for the parity cell. A "float reference" that secretly calls
    the integer path makes the parity gate a tautology — it would agree perfectly and prove
    nothing. They must differ somewhere, because quantization is lossy.
    """
    module = require(MODULE)
    forward, forward_float, fixture_model = api(
        MODULE, "forward", "forward_float_reference", "fixture_model")
    model = fixture_model(fixture_config(module))
    tokens = [1, 7, 3, 11, 0, 31, 5, 2]
    integer_logits = np.asarray(forward(model, tokens), dtype=np.float64)
    float_logits = np.asarray(forward_float(model, tokens), dtype=np.float64)
    assert not np.array_equal(integer_logits, float_logits), (
        "the float reference is bit-identical to the integer pipeline; it is not an "
        "independent oracle and the parity gate proves nothing"
    )
