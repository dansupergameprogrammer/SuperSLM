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
`g5_minimal_one_field`'s vocab bears to its model): token 0 spells the literal prefix
through the string's opening quote (`{"Prompt_Result":"`), landing the walk at S_c (the
content state) directly from state 0; token 1 is an ordinary content character ('a',
admitted at S_c, self-loop); token 2 is the backslash that reaches S_e (the escape state)
from S_c; token 3 is one of the eight short escapes RFC 8259 admits from S_e back to S_c
('n'); token 4 closes the string and the object ('"}'), from S_c to the schema's one
accepting state. Every remaining vocabulary id (5..vocab_size-1) is an inert filler piece,
unreachable from any state this schema's own DFA defines -- deliberately, so a cell
driving the real vocabulary's OTHER ids (a prompt, a caller-supplied token on retry) can
never accidentally advance the schema walk.

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
# The five meaningful pieces, in the order the header comment above walks them. Kept as a
# module-level tuple so the C2 test cells can cite the SAME ids by name instead of
# re-deriving or hardcoding them independently of this generator.
TOK_OPEN = 0    # '{"Prompt_Result":"'  -- state 0 -> S_c
TOK_CONTENT = 1  # 'a'                   -- S_c -> S_c
TOK_BACKSLASH = 2  # '\\'                -- S_c -> S_e
TOK_ESCAPE_N = 3  # 'n'                  -- S_e -> S_c
TOK_CLOSE = 4    # '"}'                  -- S_c -> accept


def _vocab(vocab_size: int):
    pieces = ['{"Prompt_Result":"', "a", "\\", "n", '"}']
    if vocab_size < len(pieces):
        raise ValueError("vocab_size %d too small for the %d meaningful pieces" % (vocab_size, len(pieces)))
    pieces += [f"<unused-{i}>" for i in range(len(pieces), vocab_size)]
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
    masks = SC.compile_schema_to_mask_pages(SCHEMA, vocab)
    # SETUP self-check: the compiled DFA has the interior escape state (S_e) this fixture
    # exists to reach, and the four hand-picked ids above actually drive it -- fail loudly
    # at generation time, not inside a cell's own red run, if the compiler's own output
    # shape ever changes under these five tokens.
    s_c = masks.step(masks.start, TOK_OPEN)
    if masks.is_accepting(s_c):
        raise AssertionError("SETUP: token 0 already reaches an accepting state -- no interior state to test")
    s_e = masks.step(s_c, TOK_BACKSLASH)
    if masks.is_accepting(s_e) or s_e == s_c:
        raise AssertionError("SETUP: the backslash token does not reach a distinct, non-accepting S_e")
    back = masks.step(s_e, TOK_ESCAPE_N)
    if back != s_c:
        raise AssertionError("SETUP: the escape token does not return S_e to S_c")
    acc = masks.step(s_c, TOK_CLOSE)
    if not masks.is_accepting(acc):
        raise AssertionError("SETUP: the closing token does not reach an accepting state")

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
