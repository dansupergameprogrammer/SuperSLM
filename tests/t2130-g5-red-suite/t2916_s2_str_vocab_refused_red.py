"""T-2916 (Curie) -- TE-370 S2's red cell: a `str` vocabulary is silently MISREAD, not refused
(`Claude/Poirot/te370-final-code-review-2026-09-21.md` Sec2 S2, Wizard repo; probe
`Claude/Poirot/te370-probe/strvocab.py`/`strvocab_output.txt`).

`compile_schema_to_mask_pages` moved from `Sequence[str]` to `Sequence[bytes]` at T-2908 (the
byte-level port), an undocumented public-tool API break. Given `str` pieces instead of `bytes`,
the vocabulary trie is keyed by characters while the DFA construction is keyed by ints (raw byte
values), so nothing matches and the compiler reports G-7a's own "no token in the vocabulary can
spell the required continuation" -- a diagnosis that is TRUE of the symptom and WRONG about the
cause: the vocabulary is not insufficient, it is the wrong element type. One known out-of-repo
caller (`t2804_append_potion_schema.py`, plugin repo, per the builder's own record) is still on
`str` and will hit exactly this misleading diagnostic once it upgrades.

RED at `0062c99`: the exact schema and vocabulary from Poirot's own probe
(`te370-probe/strvocab.py`) is executed here, reproduced verbatim
(`strvocab_output.txt`: `SchemaCompileError state 0 has an empty valid-token set...`). The fix
(Poirot's remedy) is a type check on vocabulary entries that raises `TypeError` naming `bytes` --
this cell pins THAT diagnostic, not merely "some exception happens": a caller upgrading from
1.5.0 needs the message to say what changed, not report a phantom vocabulary gap.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common
from tools.sslm_convert_schema import SchemaCompileError, compile_schema_to_mask_pages

_SC_PATH = "tools/sslm_convert_schema.py"
_BAD_COMMIT = "0062c99"

# Poirot's own probe schema and vocabulary (te370-probe/strvocab.py), reproduced verbatim.
_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}
_STR_VOCAB = list('{}":,okrtuefals') + ["true", "false", '"ok":']
_BYTES_VOCAB = [piece.encode("ascii") for piece in _STR_VOCAB]


class T2916_S2_StrVocabularyRefusedNamingBytes(unittest.TestCase):

    def test_str_vocabulary_raises_typeerror_naming_bytes(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            compile_schema_to_mask_pages(_SCHEMA, _STR_VOCAB)
        self.assertIn(
            "bytes", str(ctx.exception).lower(),
            "the refusal must name the actual requirement (a bytes vocabulary), not report a "
            "phantom 'no token can spell the continuation' vocabulary-coverage defect -- a "
            "caller who upgrades a str-vocabulary tool from 1.5.0 needs to be told what changed",
        )

    def test_str_vocabulary_does_not_raise_the_misleading_coverage_diagnostic(self) -> None:
        """The regression this pins against reappearing under a different guise: even if the
        message wording changes, it must not be `SchemaCompileError`'s own G-7a coverage
        diagnostic, which is true of the symptom and wrong about the cause."""
        try:
            compile_schema_to_mask_pages(_SCHEMA, _STR_VOCAB)
        except SchemaCompileError as exc:
            self.fail(
                f"still raising the misleading vocabulary-coverage diagnostic ({exc}) instead "
                "of refusing the str vocabulary by type"
            )
        except TypeError:
            pass  # the fixed behaviour

    def test_bytes_vocabulary_is_unaffected(self) -> None:
        """Positive control: the identical schema, a `bytes` vocabulary, compiles cleanly both
        before and after S2's fix -- proving the refusal above is about the vocabulary's
        element type, not this schema shape."""
        compile_schema_to_mask_pages(_SCHEMA, _BYTES_VOCAB)  # must not raise


class T2916_S2_MutationProof(unittest.TestCase):
    """The wrong version is `0062c99` itself -- confirmed to reproduce the exact misleading
    diagnostic Poirot's probe captured, then required to still reproduce it once a fix lands."""

    def test_frozen_0062c99_reproduces_the_misleading_diagnostic(self) -> None:
        sc = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        with self.assertRaises(sc.SchemaCompileError) as ctx:
            sc.compile_schema_to_mask_pages(_SCHEMA, _STR_VOCAB)
        self.assertIn("empty valid-token set", str(ctx.exception))

    def test_mutant_discrimination_pending_or_passing(self) -> None:
        if not common.fix_has_landed(_BAD_COMMIT, _SC_PATH):
            self.skipTest(
                f"PENDING T-2917: HEAD still equals {_BAD_COMMIT} for {_SC_PATH} -- no fix has "
                "landed yet. Re-run after T-2917 lands."
            )
        with self.assertRaises(TypeError) as ctx:
            compile_schema_to_mask_pages(_SCHEMA, _STR_VOCAB)  # shipped: TypeError naming bytes
        self.assertIn("bytes", str(ctx.exception).lower())

        sc_frozen = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        with self.assertRaises(sc_frozen.SchemaCompileError):
            sc_frozen.compile_schema_to_mask_pages(_SCHEMA, _STR_VOCAB)  # frozen: still misleads


if __name__ == "__main__":
    unittest.main()
