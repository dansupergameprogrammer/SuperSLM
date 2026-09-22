"""T-2919 (Curie) -- TE-372 S2's red cell: a nullable-enum schema crashes the compiler with an
uncaught `TypeError` (`Claude/Poirot/te372-final-code-review-2026-09-21.md` Sec2 S2, Wizard
repo; probe `Claude/Poirot/te372-probe/probe_compiler.py` P2, `probe_compiler_output.txt`).

`_groups()`'s enum branch (`tools/sslm_convert_schema.py`) reads `node_type = schema.get("type")`
and, when `type` is present, looks it up in `_JSON_SCHEMA_TYPE_CHECKS` via
`_JSON_SCHEMA_TYPE_CHECKS.get(node_type)`. JSON Schema (and OpenAPI 3.1) allow `type` to be a
LIST of type names, e.g. `["string", "null"]` for a nullable string -- `dict.get` on a list key
raises `TypeError: unhashable type: 'list'`, uncaught, naming no field. `v1.5.0` compiles the
same schema: its own `_groups()` returns from the `"enum" in schema` branch before `type` is
ever read at all (`git show v1.5.0:tools/sslm_convert_schema.py`, confirmed by inspection), so
v1.5.0 accepts every `enum`+`type` combination, valid or not, purely from the enum's own values.

RED at `60eb357` (this ticket's own artifact commit, which TE-372 reviewed and ruled DO NOT SHIP
over this exact finding, among others): every schema below raises the uncaught `TypeError`
today. Each must COMPILE post-fix (T-2920, the builder) and produce mask pages structurally
IDENTICAL to `v1.5.0`'s own compile of the same schema -- the same index-aligned `str`/`bytes`
structural-equality technique `t2916_s1_keyword_allowlist_red.py` already established and
verified by direct execution (`check_alignment.py`, `same_transition_rows`).

SIBLINGS FROM THE SAME CLASS (per the brief: "every `type` form JSON-Schema allows beside
`enum` -- a list, and `null` members"): the crash is triggered by `type` being a LIST, not by
`null` specifically -- a list `type` with no `null` member crashes identically. Both dimensions
are varied independently below: `type` as a scalar string vs. a list, and the enum's own values
with vs. without a `None` (JSON `null`) member. Two of the five schemas are regression controls
(must already pass at `60eb357`, unaffected by the crash), bounding the claim to the list-`type`
form specifically.

NOT PINNED (routed, not invented): whether a list-`type` enum whose values do NOT all match one
of the listed types is accepted (checked per-value, generalizing the existing scalar-`type`
mismatch check) or refused outright is undetermined by the finding or this ticket's brief --
only the ACCEPT-and-match-v1.5.0 contract for valid forms is pinned here. That is the builder's
(T-2920) design choice to make when landing the fix; a mismatch cell is not authored to avoid
coupling this suite to a guess of the mechanism (Curie's "realize the model; a gap in it is a
finding, not an invention").

MUTATION PROOF (StandardsDocument.md Sec5.4): the wrong version is `60eb357` itself -- not
invented -- loaded via `t2913_common.frozen_module` (git history, no out-of-repo input), the
same pattern `t2916_c1_special_exclusion_red.py`/`t2916_s1_keyword_allowlist_red.py`/
`t2916_s2_str_vocab_refused_red.py` already established for TE-370's cells.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common
from tools.sslm_convert_schema import SchemaCompileError, compile_schema_to_mask_pages

_SC_PATH = "tools/sslm_convert_schema.py"
_BAD_COMMIT = "60eb357"

# A small fixed charset covering every schema below, reused (str for v1.5.0, bytes for the
# branch) exactly as `t2916_s1_keyword_allowlist_red.py::_VOCAB_STR`/`_VOCAB_BYTES` does, so
# `mp15.transitions == mp16.transitions` is a valid, exact structural-equality check once both
# accept (index-aligned vocabularies -> token ids are pure vocabulary-list positions in both
# compilers).
_CHARS = sorted(set('{}[]",:xy1.-truefalsnAOKPCS+ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
_VOCAB_STR = list(_CHARS)
_VOCAB_BYTES = [c.encode("ascii") for c in _CHARS]

# The review's own exact reproduction (Poirot's probe_compiler.py P2), reproduced verbatim.
_NULLABLE_ENUM_LIST_TYPE_STRING_AND_NULL: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": ["x", None], "type": ["string", "null"]}},
}
# Sibling: a list `type` with exactly one member, no `null` -- confirms the crash is about
# `type` being a list at all, not about nullability specifically.
_LIST_TYPE_SINGLE_ELEMENT_NO_NULL: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": ["x", "y"], "type": ["string"]}},
}
# Sibling: a list `type` with two non-null members, matching enum values of both types.
_LIST_TYPE_MULTIPLE_NON_NULL_MEMBERS: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": ["x", 1], "type": ["string", "integer"]}},
}
# Regression control: an enum with a `null` VALUE but no `type` key at all -- `node_type` is
# `None`, so the `if node_type is not None:` branch (where the crash lives) is never entered.
# Must already compile at `60eb357`; isolates the defect to `type`'s own hashability, not to
# `null` appearing anywhere in the schema.
_ENUM_WITH_NULL_VALUE_NO_TYPE_DECLARED: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": ["x", None]}},
}
# Regression control: a SCALAR `"null"` type (not a list) alongside an enum whose only value is
# `None` -- exercises the existing (T-2917) scalar-type-checker path, which already has a "null"
# entry in `_JSON_SCHEMA_TYPE_CHECKS` and is hashable. Must already compile at `60eb357`;
# bounds the claim to LIST types, not to the word "null" appearing as a type name.
_ENUM_NULL_ONLY_SCALAR_NULL_TYPE: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": [None], "type": "null"}},
}

_CRASHING_SCHEMAS: dict[str, dict] = {
    "nullable_enum_list_type_string_and_null": _NULLABLE_ENUM_LIST_TYPE_STRING_AND_NULL,
    "list_type_single_element_no_null": _LIST_TYPE_SINGLE_ELEMENT_NO_NULL,
    "list_type_multiple_non_null_members": _LIST_TYPE_MULTIPLE_NON_NULL_MEMBERS,
}
_CONTROL_SCHEMAS: dict[str, dict] = {
    "enum_with_null_value_no_type_declared": _ENUM_WITH_NULL_VALUE_NO_TYPE_DECLARED,
    "enum_null_only_scalar_null_type": _ENUM_NULL_ONLY_SCALAR_NULL_TYPE,
}


class T2919_S2_ListTypeEnumSchemasCompileAndMatch1p5p0(unittest.TestCase):
    """Every list-`type`+`enum` schema must compile post-fix and match v1.5.0's own mask pages
    for the identical schema exactly -- structural transition-table equality, not merely 'both
    accept' (the same bar `t2916_s1_keyword_allowlist_red.py` sets for TE-370 S1)."""

    def _assert_compiles_and_matches_1p5p0(self, schema: dict) -> None:
        ref150 = common.reference_v150()
        mp15 = ref150.compile_schema_to_mask_pages(schema, _VOCAB_STR)
        try:
            mp16 = compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())
        except SchemaCompileError as exc:
            self.fail(
                f"rejected at {getattr(exc, 'state_id', None)} / {exc}: v1.5.0 compiled this "
                "schema cleanly (see mp15 above), so 1.6.0+ must too (TE-372 S2)"
            )
        self.assertEqual(
            mp15.transitions, mp16.transitions,
            "1.6.0's mask pages diverge structurally from v1.5.0's for the identical schema",
        )

    def test_nullable_enum_list_type_string_and_null(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_NULLABLE_ENUM_LIST_TYPE_STRING_AND_NULL)

    def test_list_type_single_element_no_null(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_LIST_TYPE_SINGLE_ELEMENT_NO_NULL)

    def test_list_type_multiple_non_null_members(self) -> None:
        self._assert_compiles_and_matches_1p5p0(_LIST_TYPE_MULTIPLE_NON_NULL_MEMBERS)


class T2919_S2_RegressionControlsUnaffectedByTheCrash(unittest.TestCase):
    """Bounds the S2 claim: these two schemas already compile at `60eb357` -- one because no
    `type` key is present at all (the crash's own `if node_type is not None:` guard is never
    entered), one because `type` is a hashable SCALAR string (`"null"`), not a list. Held green
    before AND after the fix (the defect was never these schemas' own path)."""

    def test_enum_with_null_value_no_type_declared_already_compiles(self) -> None:
        mp16 = compile_schema_to_mask_pages(
            _ENUM_WITH_NULL_VALUE_NO_TYPE_DECLARED, _VOCAB_BYTES, special_ids=frozenset()
        )
        mp15 = common.reference_v150().compile_schema_to_mask_pages(
            _ENUM_WITH_NULL_VALUE_NO_TYPE_DECLARED, _VOCAB_STR
        )
        self.assertEqual(mp15.transitions, mp16.transitions)

    def test_enum_null_only_scalar_null_type_already_compiles(self) -> None:
        mp16 = compile_schema_to_mask_pages(
            _ENUM_NULL_ONLY_SCALAR_NULL_TYPE, _VOCAB_BYTES, special_ids=frozenset()
        )
        mp15 = common.reference_v150().compile_schema_to_mask_pages(
            _ENUM_NULL_ONLY_SCALAR_NULL_TYPE, _VOCAB_STR
        )
        self.assertEqual(mp15.transitions, mp16.transitions)


class T2919_S2_MutationProof(unittest.TestCase):
    """The wrong version is `60eb357` itself -- confirmed to reproduce the uncaught `TypeError`
    on every crashing schema, then required to still reproduce it once T-2920 lands a fix."""

    def test_frozen_60eb357_raises_uncaught_typeerror_on_every_crashing_schema(self) -> None:
        sc = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        for name, schema in _CRASHING_SCHEMAS.items():
            with self.subTest(schema=name):
                with self.assertRaises(TypeError) as ctx:
                    sc.compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())
                self.assertIn(
                    "unhashable", str(ctx.exception).lower(),
                    f"the frozen {_BAD_COMMIT} copy must reproduce S2's own uncaught TypeError "
                    f"for {name!r}, not some other failure, for this to be a real 'wrong "
                    f"version' control",
                )

    def test_frozen_60eb357_control_schemas_already_compile(self) -> None:
        """Sanity: the two regression controls do NOT crash even at the known-bad commit --
        confirms they are genuinely unaffected by S2, not merely untested."""
        sc = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        for name, schema in _CONTROL_SCHEMAS.items():
            with self.subTest(schema=name):
                sc.compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())

    def test_mutant_discrimination_pending_or_passing(self) -> None:
        if not common.fix_has_landed(_BAD_COMMIT, _SC_PATH):
            self.skipTest(
                f"PENDING T-2920: HEAD still equals {_BAD_COMMIT} for {_SC_PATH} -- no fix has "
                "landed yet. Re-run after T-2920 lands."
            )
        sc_frozen = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        for name, schema in _CRASHING_SCHEMAS.items():
            with self.subTest(schema=name):
                compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())  # shipped: must not raise
                with self.assertRaises(TypeError):
                    sc_frozen.compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())  # frozen: must still raise


if __name__ == "__main__":
    unittest.main()
