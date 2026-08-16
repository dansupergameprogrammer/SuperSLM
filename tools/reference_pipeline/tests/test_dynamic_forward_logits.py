"""The bit-exact-logits cell: `forward_dynamic` against the independent full-forward oracle.

Charter: §15's harness pin ("two faithful implementations of the eval forward can no longer differ in
bits") over the pinned composition §6.8 C23–C30 (D-SLM56). Authored before the oracle body
existed (Curie-owned, paired session — design record §12.4); the oracle has since landed and
this cell executes green against it.

Test-design record: Claude/Curie/packets/eval/packet-09.md (amendment A-3)
"""


from pathlib import Path

import numpy as np

from conftest import api, require

MODULE = "reference_pipeline.pipeline"


def fixture_config(pipeline):
    """The §11 fixture model's shape, mirrored from test_dynamic_forward.py."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def test_forward_dynamic_logits_are_bit_exact_against_the_composition_oracle():
    """Bit-exact logits against the composition's full-forward oracle (C23–C30).

    The oracle gate is EXPLICIT and fails as red-unimplemented, not ImportError: the cell
    cannot green until the independent oracle exists (Curie-owned, paired session). This
    ordering is deliberate and is the packet's core discipline.

    §15 harness pin: the quantized logits produced by forward_dynamic match the oracle's
    exact big-int recomputation of the FULL C23–C30 composition over the fixture model's
    weights and constants, bit-for-bit, no tolerance. Two sequences: the pinned
    `[0, 1, 3, 5]` (executed pairwise-distinct D') and `[7, 2, 30, 11]` (site-constancy
    partner — different data, same length).
    """
    forward_dynamic, fixture_model = api(MODULE, "forward_dynamic", "fixture_model")

    # The oracle gate is EXPLICIT and fails as red-unimplemented, not ImportError:
    import composition_ref
    assert hasattr(composition_ref, "forward_dynamic_logits_oracle"), (
        "red-unimplemented: composition_ref.forward_dynamic_logits_oracle is not authored "
        "yet (Curie-owned, paired session — design record §12.4). This cell must not green "
        "until the independent oracle exists.")
    oracle = composition_ref.forward_dynamic_logits_oracle

    cfg = fixture_config(require(MODULE))
    model = fixture_model(cfg)

    # Two token sequences: pinned [0,1,3,5] and site-constancy partner [7,2,30,11]
    for tokens in ([0, 1, 3, 5], [7, 2, 30, 11]):
        logits = np.asarray(forward_dynamic(model, tokens))
        expected = oracle(model, tokens)  # exact big-int recomputation, C23-C30

        assert logits.dtype == np.int32
        assert logits.tolist() == [[int(v) for v in row] for row in expected]


def test_the_composition_oracle_is_independent_of_the_implementation():
    """Sanity check: the oracle module has no imports of intmath or pipeline.

    Green by design (composition_ref exists and is clean at charter). This is the one
    deliberately-green cell in the eval red suite, guarding the oracle's independence from
    the day the oracle module landed. The whole point of an oracle is two independent
    derivations of the same pinned formulas: the oracle recomputes from §6.8's definitions,
    never by re-running the implementation under test.
    """
    import composition_ref

    # Read the module source (robust to re-imports and re-executions)
    source = Path(composition_ref.__file__).read_text()

    # Scan for forbidden module-level imports. The patterns are the broad prefixes the
    # brief pinned — `import <impl>` / `from <impl>` — not the dotted spellings only:
    # `from reference_pipeline import intmath` slips a dotted-only list (executed,
    # Curie A-4), and the oracle must not reach ANY implementation module. Both the
    # pre-vendoring package name (`superslm_spike`, still cited in historical records)
    # and the vendored name (`reference_pipeline`, T-2123/T-2137) are checked, so this
    # cell keeps discriminating regardless of which name the implementation currently
    # carries.
    forbidden_patterns = [
        "import superslm_spike",
        "from superslm_spike",
        "import reference_pipeline",
        "from reference_pipeline",
    ]

    for line in source.splitlines():
        # Skip comments and docstrings
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        for pattern in forbidden_patterns:
            assert not stripped.startswith(pattern), (
                f"composition_ref violates the oracle independence rule: found '{pattern}' "
                f"in the module source. The oracle must be an independent derivation from "
                f"§6.8's pinned formulas, not a re-implementation of intmath or pipeline."
            )
