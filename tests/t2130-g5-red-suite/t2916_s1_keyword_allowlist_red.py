"""T-2916 (Curie) -- TE-370 S1's red cells: the per-branch keyword allowlist (Sec3.9.6, T-2908
S3) over-rejects schemas `v1.5.0` always compiled
(`Claude/Poirot/te370-final-code-review-2026-09-21.md` Sec2 S1, Wizard repo; probe
`Claude/Poirot/te370-probe/enum_type.py`/`enum_type_output.txt`).

Two shapes of over-rejection, both entered through the remedy for TE-365 S3 (`T2908_KeywordAllowlistGroup`,
`t2908_byte_level_red.py`), which generalized keyword rejection to every branch without noticing
either:

  1. **`{"type": "string"/"boolean", "enum": [...]}`** -- the usual JSON-Schema enum form, and
     what pydantic emits for a `Literal`/`Enum` field -- fails, because the `enum` branch's own
     allowlist is `frozenset({"enum"})` only (`_ALLOWED_KEYWORDS["enum"]`,
     `tools/sslm_convert_schema.py:159-164`). `type` alongside `enum` is redundant, not
     constraining (every enum value already carries its own JSON type), so this compiler must
     accept it.
  2. **Non-constraining annotation keywords** (`description`, `title`, `$schema`, `examples`,
     `default`) are rejected on any branch, though they constrain nothing the compiler enforces
     or fails to enforce -- the stated rationale for rejecting an unimplemented keyword ("the
     constraint would silently go unenforced") does not apply to a keyword that is not a
     constraint at all.

Four cells reproduce Poirot's own probe schemas exactly (`enum_type.py`'s `cases` dict); a fifth
is genuinely pydantic-generated (`pydantic.BaseModel.model_json_schema()`, executed here, not a
hand-typed imitation of pydantic's shape) using a `Literal` field, which is the JSON-Schema-legal,
`$ref`-free form D-SLM45's compilable subset can reach (pydantic's `Enum` form emits `$defs`/`$ref`,
which this compiler's subset never supported and is not what S1 is about).

ALL FIVE ARE RED at `0062c99`: each is rejected today (`enum_type_output.txt` confirms four of
five directly; the pydantic-generated one is newly executed here and rejected for the same two
reasons). Each must COMPILE post-fix and produce mask pages structurally IDENTICAL to `v1.5.0`'s
own compile of the same schema -- verified representable directly: an index-aligned `str` vocab
(1.5.0, char-level) and `bytes` vocab (1.6.0+, byte-level) compile to token ids that are pure
vocabulary-list positions in both compilers, so `mp15.transitions == mp16.transitions` is a valid,
exact structural-equality check once both accept the schema (confirmed by direct execution on a
schema both versions already accept, `D:/_t2916/probe/check_alignment.py`, same_transition_rows).

The must-reject side (Sec3.9.6's own point) is NOT weakened: a genuinely constraining, unimplemented
keyword (`maxLength`, `minLength`, `pattern`, `format`, `const`, `multipleOf`) must still be
refused, by name, on every branch -- `t2908_byte_level_red.py`'s `_EXTRA_KEYWORDS` list is
corrected in the same commit as this file to hold only the constraining keywords (the
non-constraining ones it wrongly listed -- `title`, `description`, `default`, `examples` -- move
here, to `T2916_S1_AnnotationKeywordsAcceptedEverywhere`, which pins ACCEPTANCE across every
branch shape (enum, boolean, string, object) instead of only the four narrow schemas the review
happened to probe -- closing the recurrence Poirot's own prediction names: "the remedy for a
prior finding introduced the next defect, and neither remedy carried a cell for the behaviour it
changed." Fixing the allowlist for four schemas and leaving the string branch's own annotation
handling untested would be exactly that pattern one more time.

CALLING CONVENTION: at authoring time every cell below called `compile_schema_to_mask_pages(schema,
vocab)` -- the then-true two-positional-argument signature -- deliberately, not C1's pinned
`special_ids=`-required contract (`t2916_c1_special_exclusion_red.py`), so S1 would fail for its
own reason rather than C1's the moment C1 landed first. **T-2917 (landing C1): every live call
below now carries `special_ids=frozenset()`** -- the mechanical, signature-forced consequence this
docstring pre-authorized ("whoever changes `compile_schema_to_mask_pages`'s signature owns
updating every call site that signature change touches, across this whole suite, in the same
round"), and none of this file's own schemas carry a real tokenizer special id, so an empty set
changes nothing S1 is testing.

MUTATION PROOF: as in `t2916_c1_special_exclusion_red.py`, the wrong version is `0062c99` itself,
loaded via `t2913_common.frozen_module`, no invented reference.
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

# Poirot's own probe charset (te370-probe/enum_type.py), reproduced verbatim so the vocabulary
# this file compiles against is the same one the finding was executed on.
_CHARS = sorted(set('{}[]",:abcdefghijklmnopqrstuvwxyz0123456789 .!?/-_ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
_VOCAB_STR = list(_CHARS)
_VOCAB_BYTES = [c.encode("ascii") for c in _CHARS]

# The four review schemas, reproduced verbatim from `Claude/Poirot/te370-probe/enum_type.py`.
_REVIEW_SCHEMAS: dict[str, dict] = {
    "type_and_enum_string": {
        "type": "object", "additionalProperties": False, "required": ["mood"],
        "properties": {"mood": {"type": "string", "enum": ["happy", "sad"]}},
    },
    "type_and_enum_boolean_ish": {
        "type": "object", "additionalProperties": False, "required": ["ok"],
        "properties": {"ok": {"type": "boolean", "enum": [True, False]}},
    },
    "description_on_field": {
        "type": "object", "additionalProperties": False, "required": ["mood"],
        "properties": {"mood": {"enum": ["happy", "sad"], "description": "x"}},
    },
    "top_level_schema_and_title": {
        "$schema": "http://json-schema.org/draft-07/schema#", "title": "T",
        "type": "object", "additionalProperties": False, "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    },
}


def _pydantic_generated_schema() -> dict:
    """Genuinely produced by pydantic v2 (`model_json_schema()`), not a hand-typed imitation.
    Uses `Literal` rather than `Enum` for the enum-shaped field: pydantic v2 inlines a `Literal`
    as `{"type": "string", "enum": [...], "title": ...}` directly on the property (the S1 shape),
    whereas `Enum` emits a `$defs`/`$ref` indirection this compiler's D-SLM45 subset does not
    reach at all -- a different, unrelated gap, not what S1 is about."""
    from typing import Literal

    from pydantic import BaseModel, ConfigDict

    class Mood(BaseModel):
        model_config = ConfigDict(extra="forbid")
        mood: Literal["happy", "sad"]
        ok: bool

    return Mood.model_json_schema()


class T2916_S1_ReviewSchemasCompileAndMatch1p5p0(unittest.TestCase):
    """The four schemas TE-370 executed directly, plus a genuinely pydantic-generated fifth.
    Each must compile at `0062c99`+fix and match `v1.5.0`'s own mask pages for the identical
    schema, exactly -- structural transition-table equality, not merely 'both accept'."""

    def _assert_compiles_and_matches_1p5p0(self, schema: dict) -> None:
        ref150 = common.reference_v150()
        mp15 = ref150.compile_schema_to_mask_pages(schema, _VOCAB_STR)
        try:
            mp16 = compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())
        except SchemaCompileError as exc:
            self.fail(
                f"rejected at {getattr(exc, 'state_id', None)} / {exc}: v1.5.0 compiled this "
                "schema cleanly (see mp15 above), so 1.6.0+ must too (TE-370 S1)"
            )
        self.assertEqual(
            mp15.transitions, mp16.transitions,
            "1.6.0's mask pages diverge structurally from v1.5.0's for the identical schema "
            "(index-aligned str/bytes vocabularies compile to identical token-id transition "
            "tables when both accept -- see check_alignment.py -- so any divergence here is a "
            "real behavioural difference, not a vocabulary-encoding artifact)",
        )

    def test_type_and_enum_string(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_REVIEW_SCHEMAS["type_and_enum_string"])

    def test_type_and_enum_boolean_ish(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_REVIEW_SCHEMAS["type_and_enum_boolean_ish"])

    def test_description_on_field(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_REVIEW_SCHEMAS["description_on_field"])

    def test_top_level_schema_and_title(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_REVIEW_SCHEMAS["top_level_schema_and_title"])

    def test_pydantic_generated_schema(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_pydantic_generated_schema())


class T2916_S1_AnnotationKeywordsAcceptedEverywhere(unittest.TestCase):
    """Closes the recurrence Poirot's own prediction names: fixing the allowlist only for the
    four schemas the review happened to probe (all enum/object branches) would leave the STRING
    branch's own handling of the same annotation keywords untested and free to diverge -- exactly
    the shape S1 and S2 both took (a remedy that fixed what was measured and regressed what
    wasn't). Each non-constraining annotation keyword must be accepted and ignored on every
    branch shape the compiler has: enum, boolean, string, object."""

    _ANNOTATION_KEYWORDS = ["description", "title", "default", "examples"]

    def _leaf_schemas(self) -> dict[str, dict]:
        return {
            "enum": {"enum": ["a", "b"]},
            "boolean": {"type": "boolean"},
            "string": {"type": "string"},
        }

    def test_annotation_keyword_accepted_on_every_leaf_branch(self) -> None:
        for keyword in self._ANNOTATION_KEYWORDS:
            for branch_name, leaf in self._leaf_schemas().items():
                with self.subTest(keyword=keyword, branch=branch_name):
                    field = dict(leaf, **{keyword: "irrelevant-value"})
                    schema = {
                        "type": "object", "additionalProperties": False, "required": ["f"],
                        "properties": {"f": field},
                    }
                    # T-2917 RECONCILIATION: a bare `"` token alongside the whole `"f":` literal
                    # gives the trie a spurious partial spelling of the key's own literal (`"`
                    # landing mid-literal, then no token can spell the rest, `f":`) -- a reachable
                    # G-7a dead end this cell's own allowlist fix now reaches for the first time
                    # (0062c99 always rejected these schemas earlier, on the keyword itself, so
                    # this vocab gap was never exercised). `t2908_byte_level_red.py`'s own
                    # `_schema_with` vocab already carries the same `f":`-without-quote covering
                    # token for exactly this reason; added here too, mechanically, not a product
                    # fix (`tools/sslm_convert_schema.py` is unchanged by this line).
                    vocab = [b"{", b"}", b'"f":', b'f":', b'"', b"a", b"b", b"true", b"false"]
                    try:
                        compile_schema_to_mask_pages(schema, vocab, special_ids=frozenset())
                    except SchemaCompileError as exc:
                        self.fail(
                            f"{keyword!r} on the {branch_name} branch was rejected ({exc}) -- "
                            "annotation keywords constrain nothing and must be accepted and "
                            "ignored on every branch, not only the branches TE-370's probe "
                            "happened to exercise"
                        )

    def test_annotation_keyword_accepted_at_the_object_root(self) -> None:
        for keyword in ["$schema", *self._ANNOTATION_KEYWORDS]:
            with self.subTest(keyword=keyword):
                schema = {
                    keyword: "irrelevant-value",
                    "type": "object", "additionalProperties": False, "required": ["ok"],
                    "properties": {"ok": {"type": "boolean"}},
                }
                vocab = [b"{", b"}", b'"ok":', b"true", b"false"]
                try:
                    compile_schema_to_mask_pages(schema, vocab, special_ids=frozenset())
                except SchemaCompileError as exc:
                    self.fail(f"{keyword!r} at the object root was rejected ({exc})")


class T2916_S1_ConstrainingKeywordsStillRejected(unittest.TestCase):
    """The must-reject side of Sec3.9.6 is not weakened by S1's fix: a genuinely constraining,
    unimplemented keyword is refused, by name, on every branch -- including the branches this
    file's own acceptance cells now exercise. Regression control, held green before AND after
    T-2917's fix (constraining keywords were never the bug)."""

    _CONSTRAINING_KEYWORDS = ["minLength", "pattern", "format", "const", "multipleOf"]

    def test_constraining_keyword_still_rejected_naming_itself(self) -> None:
        for keyword in self._CONSTRAINING_KEYWORDS:
            with self.subTest(keyword=keyword):
                schema = {
                    "type": "object", "additionalProperties": False, "required": ["f"],
                    "properties": {"f": {"type": "string", keyword: "irrelevant-value"}},
                }
                vocab = [b"{", b"}", b'"f":', b'"', b"a", b"b"]
                with self.assertRaises(SchemaCompileError) as ctx:
                    compile_schema_to_mask_pages(schema, vocab, special_ids=frozenset())
                self.assertIn(keyword, str(ctx.exception))


class T2916_S1_MutationProof(unittest.TestCase):
    """The wrong version is `0062c99` itself -- confirmed to reproduce the rejection, per the
    review's own probe output, then required to still reproduce it once a fix has landed."""

    def test_frozen_0062c99_rejects_all_five(self) -> None:
        sc = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        for name, schema in _REVIEW_SCHEMAS.items():
            with self.subTest(schema=name):
                with self.assertRaises(sc.SchemaCompileError):
                    sc.compile_schema_to_mask_pages(schema, _VOCAB_BYTES)
        with self.assertRaises(sc.SchemaCompileError):
            sc.compile_schema_to_mask_pages(_pydantic_generated_schema(), _VOCAB_BYTES)

    def test_mutant_discrimination_pending_or_passing(self) -> None:
        if not common.fix_has_landed(_BAD_COMMIT, _SC_PATH):
            self.skipTest(
                f"PENDING T-2917: HEAD still equals {_BAD_COMMIT} for {_SC_PATH} -- no fix has "
                "landed yet. Re-run after T-2917 lands."
            )
        sc_frozen = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        for name, schema in _REVIEW_SCHEMAS.items():
            with self.subTest(schema=name):
                compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())  # shipped: must not raise
                with self.assertRaises(sc_frozen.SchemaCompileError):
                    sc_frozen.compile_schema_to_mask_pages(schema, _VOCAB_BYTES)  # frozen: must still raise


if __name__ == "__main__":
    unittest.main()
