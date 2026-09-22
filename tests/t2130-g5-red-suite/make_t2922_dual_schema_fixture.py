"""Hermetic A/B SCM1 fixture: same reachable q, opposite acceptance membership."""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
sys.path[:0] = [TOOLS, os.path.join(TOOLS, "reference_pipeline")]
import convert_model as C  # noqa: E402
import pipeline as P  # noqa: E402
import sslm_convert_schema as SC  # noqa: E402
import sslm_format as F  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402
from _t2199_s8_synthetic_full_model_fixture import build_config  # noqa: E402

SCHEMA = {"type":"object", "additionalProperties":False,
          "properties":{"Prompt_Result":{"type":"string"}},
          "required":["Prompt_Result"]}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_t2922_dual_schema_fixture.py <out.sslm>")
    cfg = build_config()
    model = P.fixture_model(cfg)
    model = replace(model,
        weights={k:v for k,v in model.weights.items() if not k.endswith((".q_norm.gain", ".k_norm.gain"))},
        weight_scales={k:v for k,v in model.weight_scales.items() if not k.endswith((".q_norm.gain", ".k_norm.gain"))})
    sections, fold_error = C.build_sections(model, enable_damped_greedy=True)
    vocab = [b'{"Prompt_Result":', b'"', b'a', b'\\', b'n', b'"}', b'}']
    vocab += [f"<unused-{i}>".encode() for i in range(len(vocab), cfg.vocab_size)]
    base = SC.compile_schema_to_mask_pages(SCHEMA, vocab, special_ids=frozenset())
    q = base.step(base.start, 0)
    if q == base.start or base.is_accepting(q):
        raise AssertionError(f"q={q} is not the intended reachable nonaccepting state")
    accepts_q = SC.MaskPages(base.vocab, base.transitions, base.start, base.accepting | {q})
    rejects_q = SC.MaskPages(base.vocab, base.transitions, base.start, base.accepting - {q})
    sections.append(F.Section(F.SectionType.SCHEMA_MASKS,
        _serialize_scm1([("t2922_accepts_q", accepts_q),
                         ("t2922_rejects_q", rejects_q)], cfg.vocab_size)))
    data, fingerprint = F.build_artifact(sections,
        flags=C.artifact_flags_for_model(model) | F.DAMPED_GREEDY_CONSTANTS_FLAG)
    with open(sys.argv[1], "wb") as output:
        output.write(data)
    print(f"wrote {sys.argv[1]} bytes={len(data)} fingerprint={fingerprint} "
          f"q={q} A=1 B=0 fold_error={fold_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
