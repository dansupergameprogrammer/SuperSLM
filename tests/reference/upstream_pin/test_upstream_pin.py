"""Red suite for §13 item 2a's committed upstream pin.

Two halves, split by what they need:

- **CI-reachable** — self-consistency, structure, and the artifact tie. These run on a bare
  checkout with no `torch`, no `transformers` and no cached checkpoint, which is the whole
  reason the oracle is committed rather than generated (§13 item 6, S-HARDEN-5).
- **Recorded pre-tag** — the regeneration leg, which reproduces the sequences from the pin
  and byte-compares. It needs the upstream model and is skipped, visibly, where it is
  absent. A skip here is a statement that the leg was not exercised, which is exactly what
  it is (the same discipline `test_pipeline.py`'s `upstream_required` applies, and the same
  visible-skip shape T-1320 settled for the cross-repo `converter-validate` case).

Every guard below is proven able to FAIL on the fault it exists to catch, per
`StandardsDocument` §4 — a check shown only to pass on unchanged input is not shown to
check anything. `check_provenance.py`'s own suite is the pattern followed here.
"""

import copy
import hashlib
import json
import os
import pathlib

import pytest

import check_upstream_pin as cup

PIN_PATH = pathlib.Path(__file__).resolve().parent / "upstream_pin.json"


@pytest.fixture
def pin():
    if not PIN_PATH.exists():
        pytest.fail(
            f"{PIN_PATH.name} is absent. It is a committed oracle, not a generated one — "
            f"if it needs (re)making, run gen_upstream_pin.py as the recorded pre-tag step"
        )
    return cup.load_pin(PIN_PATH)


def _repin(body: dict) -> dict:
    """Re-seal a mutated body so its content hash is valid again.

    Used by the structure cells: without it every mutation would trip the self-consistency
    guard first, and the structure guard would never be the thing under test.
    """
    sealed = copy.deepcopy(body)
    sealed.pop("content_sha256", None)
    digest = hashlib.sha256(cup.canonical_bytes(sealed)).hexdigest()
    sealed["content_sha256"] = digest
    return sealed


# --- CI-reachable ------------------------------------------------------------------


def test_pin_is_self_consistent(pin):
    """The committed bytes hash to the digest they carry."""
    cup.check_self_consistent(pin)


def test_self_consistency_catches_a_hand_edit(pin):
    """Vitality: a single edited token id is caught.

    This is the fault a committed oracle is most exposed to and the one no amount of
    upstream availability would catch — the sequences would simply be believed.
    """
    tampered = copy.deepcopy(pin)
    tampered["pack"]["members"][0]["upstream_tokens"][0] += 1
    with pytest.raises(cup.PinFailure, match="content_sha256 mismatch"):
        cup.check_self_consistent(tampered)


def test_pin_structure_is_complete(pin):
    cup.check_structure(pin)


@pytest.mark.parametrize("field", cup.REQUIRED_PIN_FIELDS)
def test_structure_catches_a_hollow_pin_field(pin, field):
    """Vitality, one cell per pinned field: a null anywhere in the address is caught.

    Parametrized rather than sampled, because a content-addressed pin is only as strong as
    its weakest field and 'we checked three of eleven' is not a statement about the pin.
    """
    tampered = _repin(pin)
    tampered["pin"][field] = None
    with pytest.raises(cup.PinFailure, match="absent or null"):
        cup.check_structure(_repin(tampered))


def test_structure_catches_a_dropped_pack_member(pin):
    """Vitality: the pack is closed at five (D-SLM350); four is a different pack."""
    tampered = _repin(pin)
    del tampered["pack"]["members"][2]
    with pytest.raises(cup.PinFailure, match="closed pack"):
        cup.check_structure(_repin(tampered))


def test_structure_catches_a_truncated_sequence(pin):
    """Vitality: a short continuation is the exact shape that ships the defect.

    Measured at transformers 5.13.1, one prompt's first divergence under a wrong RoPE theta
    is at token 23 — a pin silently truncated to 16 would agree with a wrong model.
    """
    tampered = _repin(pin)
    tampered["pack"]["members"][0]["upstream_tokens"] = (
        tampered["pack"]["members"][0]["upstream_tokens"][:4])
    with pytest.raises(cup.PinFailure, match="upstream tokens against a declared"):
        cup.check_structure(_repin(tampered))


def test_structure_catches_an_out_of_vocabulary_token(pin):
    tampered = _repin(pin)
    tampered["pack"]["members"][1]["upstream_tokens"][0] = cup.VOCAB_SIZE
    with pytest.raises(cup.PinFailure, match="outside"):
        cup.check_structure(_repin(tampered))


def test_pin_matches_the_converted_artifact(pin):
    """The plan-of-record's own cell: the pin against the artifact S3a converts from.

    Reports rather than passes when the artifact is absent — the artifact directory is not
    in this repository, and a provenance check that goes green when it cannot see its
    subject is worse than no check.
    """
    result = cup.check_artifact_tie(pin, cup.DEFAULT_ARTIFACT)
    if result.startswith("UNAVAILABLE"):
        pytest.skip(result)
    assert result == "OK"


def test_artifact_tie_catches_a_different_artifact(pin, tmp_path):
    """Vitality: a manifest naming another calibrated model is caught.

    Built from a synthetic manifest rather than by moving the real artifact — this cell
    must not depend on the artifact being present, or the guard's proof would be as
    unavailable as the guard.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps({"source_fingerprint": "0" * 64}), encoding="utf-8")
    with pytest.raises(cup.PinFailure, match="different calibrated model"):
        cup.check_artifact_tie(pin, tmp_path)


def test_artifact_tie_reports_absence_rather_than_passing(pin, tmp_path):
    """Vitality of the reporting path itself: no manifest is UNAVAILABLE, never OK."""
    assert cup.check_artifact_tie(pin, tmp_path).startswith("UNAVAILABLE")


# --- recorded pre-tag --------------------------------------------------------------


def _upstream_available():
    try:
        import gen_upstream_pin as gen
    except Exception:
        return None
    try:
        return gen.checkpoint_dir()
    except SystemExit:
        return None


def test_pinned_checkpoint_hashes_still_hold(pin):
    """The pin's file hashes against the live cached checkpoint.

    This is what makes `model_revision` more than a string: if the snapshot on this machine
    resolved to different bytes, the pin is about a checkpoint this box does not have.
    """
    pytest.importorskip("huggingface_hub")
    checkpoint = _upstream_available()
    if checkpoint is None:
        pytest.skip("the upstream checkpoint is not cached; the pin's file hashes are "
                    "unverified on this machine")
    import gen_upstream_pin as gen

    assert checkpoint.name == pin["pin"]["model_revision"], (
        f"the cached snapshot is {checkpoint.name} and the pin names "
        f"{pin['pin']['model_revision']}"
    )
    for field, filename in gen.PINNED_FILES.items():
        path = checkpoint / filename
        if not path.exists():
            assert pin["pin"][field] is None, f"{filename} is absent but the pin hashes it"
            continue
        assert gen.sha256_file(path) == pin["pin"][field], (
            f"{filename} on this machine does not hash to the pinned value; the pinned "
            f"oracle is about different checkpoint bytes"
        )


REGEN_ENV = "SUPERSLM_RUN_UPSTREAM_PIN_REGEN"


@pytest.mark.slow
def test_pin_regenerates_byte_identically(pin):
    """The regeneration leg: reproduce the sequences from the pin and byte-compare.

    This is the leg §11 S3.8 requires and the one that cannot run in CI. It re-runs the
    upstream model over every pinned `input_ids` and asserts the continuation is the
    committed one, token for token.

    **Opt-in by environment variable, deliberately.** This repository has no pytest config,
    so a bare `pytest tests/` collects everything — and this leg drives a 1.5B model over
    five prompts of up to 571 tokens for 32 generated tokens each. The sibling spike tree
    has already paid that cost three times by accident (`Tools/superslm_spike/pytest.ini`
    records a killed background job and an ~18-minute burn), and it solved it with a
    default-deselect. A repo-wide `pytest.ini` here would change collection for every other
    suite in this tree, so the narrower fix is used: this one cell gates itself, and says so
    when it skips. Set `SUPERSLM_RUN_UPSTREAM_PIN_REGEN=1` to run it as the pre-tag step.
    """
    if not os.environ.get(REGEN_ENV):
        pytest.skip(
            f"the regeneration leg is opt-in: set {REGEN_ENV}=1. It drives the upstream "
            f"1.5B model over all five pack members and is the recorded pre-tag step, not "
            f"a per-commit check. This skip means §11 S3.8's regeneration leg did not run"
        )
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    checkpoint = _upstream_available()
    if checkpoint is None:
        pytest.skip("the upstream checkpoint is not cached; the regeneration leg did not run")
    import gen_upstream_pin as gen

    count = pin["decode"]["max_new_tokens"]
    for member in pin["pack"]["members"]:
        produced = gen.upstream_greedy(checkpoint, member["input_ids"], count)
        assert produced == member["upstream_tokens"], (
            f"member {member['index']} ({member['axis']}) regenerated a different upstream "
            f"continuation; first divergence at position "
            f"{next(i for i, (a, b) in enumerate(zip(produced, member['upstream_tokens'])) if a != b)}"
        )
