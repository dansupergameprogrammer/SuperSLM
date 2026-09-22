"""T-2909 (Curie) -- builds a real, hermetic .sslm model fixture carrying T-2853's own
free-text string leaf (a single-string-field schema, `Prompt_Result`, plan
`Claude/Plans/te266-gpu-path.md` Sec3.9.1) compiled by the production schema compiler
(`tools/sslm_convert_schema.py`), for TE-365's C2 finding (`Claude/Poirot/
te365-stage1-code-review-2026-09-21.md` Sec2/Sec4): plan Sec3.10.2 Cell 2 needs "a real
schema with a string field," and no fixture in this repo carries one -- T-2853's own leaf
landed in the compiler only, and the C39 synthetic's own schema (`g5_minimal_one_field`,
`tools/_t2199_s8_synthetic_full_model_fixture.py`) is a single boolean field compiled to
one accepting token, with no interior state at all (that file's own docstring: "2 states
and 1 transition").

REUSED, NOT RE-DERIVED. `build_config`/`P.fixture_model` are T-2199 Phase D's own hermetic
"Sec11 fixture model" (`tools/_t2199_s8_synthetic_full_model_fixture.py`), imported
unchanged -- this script's only departure from that file is the schema and its vocabulary
(index-identical to the model's own vocabulary, the same convention that file establishes
for `g5_minimal_one_field`). No production or tools-tree file is edited to build this
fixture: everything test-specific lives here, in this suite's own directory, matching
`tests/t2791-gpu-prefill-read-red-suite/make_g5an_fixture.py`'s own precedent of a
test-tree generator built from tools-tree primitives.

THE SCHEMA. {"type": "object", "additionalProperties": False, "properties":
{"Prompt_Result": {"type": "string"}}, "required": ["Prompt_Result"]} -- T-2853's own
worked example (Sec3.9.1/Sec3.10.2), the same schema shape `t2899_string_leaf_red.py`
compiles in isolation. Compiled here against a REAL vocabulary this script constructs,
index-identical to the model's own `vocab_size` tokens (the same relationship
`g5_minimal_one_field`'s vocab bears to its model).

T-2915 (porting the plan's final Sec3.9 design -- T-2908's byte alphabet, T-2910's open-side
boundary discipline, T-2912's value-level closure): the six meaningful pieces, in walk
order. T-2910's own boundary discipline refuses a token that opens the string's content
sub-automaton past its own first byte, so the literal object skeleton and the value's
opening quote can no longer share one vocabulary piece the way the pre-fold, character-
level compiler allowed -- token 0 now stops at the key's own closing colon, and the opening
quote is its own, separate token. Token 2 ('a') is an ordinary content byte, meaningful
under T-2912's own U/M value-level product (Sec3.9.1): a value that has contributed no
meaningful byte cannot close, so this fixture's own SETUP self-check below exercises token 2
before closing, though the C2 test cells this fixture serves (`cell_cpu_deadend_retry_
reset.cpp`, `cell_gpu_cell2_degenerate.cpp`) only need the escape state reachable and do not
walk this far. Token 0 spells the literal prefix up to the key's own colon
(`{"Prompt_Result":`); token 1 is the bare opening quote (`"`, state -> U, T-2910's
depth-0-only opening); token 2 is an ordinary content byte ('a', U -> M); token 3 is the
backslash that reaches the escape state (M -> M's own escape state, or, driven directly from
U as the C2 cells do, U -> U's own escape state); token 4 is one of the eight short escapes
RFC 8259 admits back to content ('n'); token 5 closes the string and the object ('"}'), from
a meaningful content state to the schema's one accepting state. Every remaining vocabulary
id (6..vocab_size-1) is an inert filler piece, unreachable from any state this schema's own
DFA defines -- deliberately, so a cell driving the real vocabulary's OTHER ids (a prompt, a
caller-supplied token on retry) can never accidentally advance the schema walk.

Deterministic, hermetic, regenerated fresh at test time (matching
`_t2199_s8_synthetic_full_model_fixture.py`'s own S-HARDEN-5 discipline); never committed
as a binary blob.

Usage: python make_t2909_string_schema_fixture.py <out.sslm>
"""
import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENG_TOOLS = os.path.normpath(os.path.join(_HERE, "..", "..", "tools"))
sys.path.insert(0, _ENG_TOOLS)
sys.path.insert(0, os.path.join(_ENG_TOOLS, "reference_pipeline"))

import convert_model as C  # noqa: E402
import pipeline as P  # noqa: E402
import sslm_convert_schema as SC  # noqa: E402
import sslm_format as F  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402
from _t2199_s8_synthetic_full_model_fixture import build_config  # noqa: E402

SCHEMA_NAME = "t2909_prompt_result_string"
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"Prompt_Result": {"type": "string"}},
    "required": ["Prompt_Result"],
}
# The seven meaningful pieces, in the order the header comment above walks them. Kept as a
# module-level tuple so the C2 test cells can cite the SAME ids by name instead of
# re-deriving or hardcoding them independently of this generator.
TOK_OPEN = 0        # '{"Prompt_Result":'  -- state 0 -> the pre-value literal state
TOK_OPEN_QUOTE = 1  # '"'                   -- pre-value -> U (T-2910: opens content, depth 0 only)
TOK_CONTENT = 2     # 'a'                   -- U -> M (a meaningful byte)
TOK_BACKSLASH = 3   # '\\'                  -- U -> U's own escape state, or M -> M's own escape state
TOK_ESCAPE_N = 4    # 'n'                   -- escape -> the state it branched from
TOK_CLOSE = 5       # '"}'                  -- M -> accept, in one compound token
TOK_BRACE = 6       # '}'                   -- the object's own literal close, reached after a
                     #                          BARE closing quote (id 1, walked from M) lands
                     #                          the walk at the literal state right after the
                     #                          string, which G-7a requires a covering token for
                     #                          -- a real vocabulary always has one


def _vocab(vocab_size: int):
    pieces = [b'{"Prompt_Result":', b'"', b"a", b"\\", b"n", b'"}', b"}"]
    if vocab_size < len(pieces):
        raise ValueError("vocab_size %d too small for the %d meaningful pieces" % (vocab_size, len(pieces)))
    pieces += [f"<unused-{i}>".encode("ascii") for i in range(len(pieces), vocab_size)]
    return pieces


def build_artifact_bytes():
    """Returns (data, fingerprint, fold_approximation_error) -- a complete, real .sslm
    model artifact byte string, identical in every dimension to the T-2199 Phase D "Sec11
    fixture model" except for the schema it carries."""
    cfg = build_config()
    model = P.fixture_model(cfg)
    model = replace(
        model,
        weights={k: v for k, v in model.weights.items()
                 if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))},
        weight_scales={k: v for k, v in model.weight_scales.items()
                       if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))},
    )
    sections, fold_approximation_error = C.build_sections(model, enable_damped_greedy=True)
    vocab = _vocab(cfg.vocab_size)
    # T-2917 (folding TE-370 C1, D-SLM7600): every id in this synthetic vocabulary is either
    # one of the six meaningful pieces or an inert filler piece -- no tokenizer special ids
    # exist here to exclude, so special_ids=frozenset() is the correct, deliberate call.
    masks = SC.compile_schema_to_mask_pages(SCHEMA, vocab, special_ids=frozenset())
    # SETUP self-check: the compiled DFA has the interior escape state (S_e) this fixture
    # exists to reach, and the six hand-picked ids above actually drive it -- fail loudly
    # at generation time, not inside a cell's own red run, if the compiler's own output
    # shape ever changes under these six tokens.
    #
    # T-2915 (porting the plan's final Sec3.9 design -- T-2910's boundary discipline, T-2912's
    # value-level closure). Two changes from the pre-fold, character-level compiler this
    # fixture was originally built against:
    #
    # 1. T-2910 refuses a token that opens the string's content sub-automaton past its own
    #    first byte, so the literal object skeleton and the value's own opening quote can no
    #    longer share one vocabulary piece -- TOK_OPEN now stops at the key's own colon, and
    #    TOK_OPEN_QUOTE is the separate, bare-quote token that actually opens content.
    # 2. The state directly after the opening quote is now `U` ("uncommitted" -- the decoded
    #    value has contributed no meaningful byte yet) rather than a single content state
    #    that can always close. `U` has no closing-quote edge at all, by design (Sec3.9.1: "a
    #    value that has contributed no meaningful byte cannot close"), so TOK_CLOSE is no
    #    longer reachable directly from the open state -- confirmed structurally below rather
    #    than assumed. Escaping the insignificant byte `\n` (TOK_ESCAPE_N) keeps the walk in
    #    `U`, which is what this SETUP check verifies; TOK_CONTENT ('a', a meaningful byte) is
    #    what makes the value closable, exercised on a separate walk from the same `U` state.
    #
    # The C++ Cell 2 cells this fixture serves only need S_e to exist and admit exactly
    # {TOK_BACKSLASH, TOK_ESCAPE_N, TOK_CLOSE} as valid next tokens (`cell_cpu_deadend_
    # retry_reset.cpp`, `cell_gpu_cell2_degenerate.cpp`) -- their own prefill sequence now
    # drives TOK_OPEN then TOK_OPEN_QUOTE (was TOK_OPEN alone) before TOK_BACKSLASH, matching
    # the split above; otherwise unaffected by this revision.
    pre_value = masks.step(masks.start, TOK_OPEN)
    if masks.is_accepting(pre_value):
        raise AssertionError("SETUP: TOK_OPEN already reaches an accepting state -- no interior state to test")
    u = masks.step(pre_value, TOK_OPEN_QUOTE)
    if masks.is_accepting(u) or u == pre_value:
        raise AssertionError("SETUP: TOK_OPEN_QUOTE does not reach a distinct, non-accepting U")
    s_e = masks.step(u, TOK_BACKSLASH)
    if masks.is_accepting(s_e) or s_e == u:
        raise AssertionError("SETUP: the backslash token does not reach a distinct, non-accepting S_e")
    back = masks.step(s_e, TOK_ESCAPE_N)
    if back != u:
        raise AssertionError("SETUP: the escape token (an insignificant byte, \\n) does not return S_e to U")
    if TOK_CLOSE in masks.valid_token_ids(u):
        raise AssertionError(
            "SETUP: TOK_CLOSE is wrongly admitted directly from U -- the value-level closure "
            "(T-2912) regressed: a value with no meaningful byte must not be closable"
        )
    m = masks.step(u, TOK_CONTENT)
    if m == u:
        raise AssertionError("SETUP: TOK_CONTENT (a meaningful byte) did not move the walk out of U")
    acc = masks.step(m, TOK_CLOSE)
    if not masks.is_accepting(acc):
        raise AssertionError("SETUP: the closing token does not reach an accepting state from a meaningful state")

    sections.append(F.Section(
        F.SectionType.SCHEMA_MASKS,
        _serialize_scm1([(SCHEMA_NAME, masks)], cfg.vocab_size)))
    data, fingerprint = F.build_artifact(
        sections, flags=C.artifact_flags_for_model(model) | F.DAMPED_GREEDY_CONSTANTS_FLAG)
    return data, fingerprint, fold_approximation_error


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "t2909_string_schema_fixture.sslm"
    data, fingerprint, fold_err = build_artifact_bytes()
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"wrote {out_path}: {len(data)} bytes, fingerprint={fingerprint}, "
          f"fold_approximation_error={fold_err}, schema={SCHEMA_NAME}")
