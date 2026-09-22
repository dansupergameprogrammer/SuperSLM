"""T-2920 (Brunel) -- pin for the enum/list-type disagreement rule landed while fixing TE-372 S2
(`Claude/Poirot/te372-final-code-review-2026-09-21.md` Sec2 S2; `Claude/Curie/
t2919-te372-test-fixes-2026-09-21.md` Sec2.3, which routed this exact question to T-2920 rather
than inventing an answer: "whether a list-`type` enum whose values do NOT all match one of the
listed types is accepted (checked per-value) or refused outright is undetermined by the finding
or this ticket's brief").

THE RULE (settled at source, `tools/sslm_convert_schema.py::_groups()`'s enum branch): a `type`
list is JSON Schema's union reading of the keyword -- a value agrees with the declared type if it
matches AT LEAST ONE listed type. A value matching NONE of the listed types is refused with
`SchemaCompileError`, naming the mismatched value(s) -- the direct generalization of the
pre-existing SCALAR `type`+`enum` mismatch rule (T-2917, TE-370 S1) to a union of types, not a
new mechanism: a scalar type is normalized to a one-element list and checked exactly the same way,
so the two rules are one rule with one code path.

WHY THIS IS THE CONSISTENT READING, not an arbitrary pick: JSON Schema / OpenAPI 3.1 define a
`type` array as "valid if the instance matches ANY ONE of the given types" -- the identical
semantics `anyOf` gives a set of sub-schemas. Requiring a value to match every listed type would
make `["string", "null"]` (the review's own reproduction) reject every value which is exactly the
form's whole point to admit, and would contradict `t2919_te372_s2_nullable_enum_red.py`'s own
already-pinned ACCEPT cases (`list_type_multiple_non_null_members`: `"x"` matches only `string`,
`1` matches only `integer`, both accepted).

THE MUTANT THIS PIN KILLS: reverting T-2920's fix (`git checkout 60eb357 --
tools/sslm_convert_schema.py`) makes every cell below raise the SAME uncaught, un-field-naming
`TypeError: unhashable type: 'list'` that TE-372 S2 found -- proven live below via
`t2913_common.frozen_module`, the identical "wrong version is the actual reviewed commit, not an
invented one" technique `t2919_te372_s2_nullable_enum_red.py` already established. A mutant that
deletes or weakens the mismatch check specifically (e.g. `any(...)` -> a no-op, or the `if
mismatched:` guard removed) would make `test_value_matching_no_listed_type_is_refused` below
silently accept a schema the JSON-Schema-union reading forbids -- the cell this pin exists for.
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

_CHARS = sorted(set('{}[]",:xy1.-truefalsnAOKPCS+ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
_VOCAB_BYTES = [c.encode("ascii") for c in _CHARS]

# A value (5.5) that matches neither listed type (`string`, `null`) -- the case T-2919 explicitly
# routed rather than invented. Note 5.5 is not `bool` (which would spuriously satisfy neither
# `string` nor `null` either way but risks confusion with the int/bool carve-out the scalar
# checkers already apply) and not `int` (which some readers might mistake for "numeric enough").
_ENUM_VALUE_MATCHES_NEITHER_LISTED_TYPE: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": ["x", 5.5], "type": ["string", "null"]}},
}
# A value matching NONE of THREE listed types (bounds the claim beyond the two-type case above).
_ENUM_VALUE_MATCHES_NONE_OF_THREE_LISTED_TYPES: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": [True], "type": ["string", "integer", "null"]}},
}
# An empty `type` list alongside `enum` -- names no candidate type, so no value could ever agree
# with it. Not the crash's own reproduction (a non-empty list), but the same code path's own
# boundary: a list with zero elements is still a list, still routed through the new normalization.
_EMPTY_TYPE_LIST: dict = {
    "type": "object", "additionalProperties": False, "required": ["a"],
    "properties": {"a": {"enum": ["x"], "type": []}},
}

_MISMATCH_SCHEMAS: dict[str, dict] = {
    "enum_value_matches_neither_listed_type": _ENUM_VALUE_MATCHES_NEITHER_LISTED_TYPE,
    "enum_value_matches_none_of_three_listed_types": _ENUM_VALUE_MATCHES_NONE_OF_THREE_LISTED_TYPES,
    "empty_type_list": _EMPTY_TYPE_LIST,
}


class T2920_ListTypeEnumDisagreementIsRefused(unittest.TestCase):
    """Every schema above must be refused with `SchemaCompileError` -- never silently accepted,
    never an uncaught `TypeError` -- once T-2920's fix lands."""

    def test_value_matching_no_listed_type_is_refused(self) -> None:
        with self.assertRaises(SchemaCompileError) as ctx:
            compile_schema_to_mask_pages(
                _ENUM_VALUE_MATCHES_NEITHER_LISTED_TYPE, _VOCAB_BYTES, special_ids=frozenset()
            )
        self.assertIn("5.5", str(ctx.exception))

    def test_value_matching_none_of_three_listed_types_is_refused(self) -> None:
        with self.assertRaises(SchemaCompileError) as ctx:
            compile_schema_to_mask_pages(
                _ENUM_VALUE_MATCHES_NONE_OF_THREE_LISTED_TYPES, _VOCAB_BYTES, special_ids=frozenset()
            )
        self.assertIn("True", str(ctx.exception))

    def test_empty_type_list_alongside_enum_is_refused(self) -> None:
        with self.assertRaises(SchemaCompileError):
            compile_schema_to_mask_pages(_EMPTY_TYPE_LIST, _VOCAB_BYTES, special_ids=frozenset())


class T2920_MutationProof(unittest.TestCase):
    """The wrong version is `60eb357` itself, exactly as `t2919_te372_s2_nullable_enum_red.py`
    pins for S2's own crashing schemas -- every mismatch schema above ALSO crashes there (the
    frozen bug swallows the whole list-`type` class, mismatches included), so this proof shows
    T-2920's fix turned a crash into the correct, specific refusal rather than merely avoiding the
    crash by accepting everything."""

    def test_frozen_60eb357_raises_uncaught_typeerror_on_every_mismatch_schema(self) -> None:
        sc = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        for name, schema in _MISMATCH_SCHEMAS.items():
            with self.subTest(schema=name):
                with self.assertRaises(TypeError) as ctx:
                    sc.compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())
                self.assertIn("unhashable", str(ctx.exception).lower())

    def test_mutant_discrimination_pending_or_passing(self) -> None:
        if not common.fix_has_landed(_BAD_COMMIT, _SC_PATH):
            self.skipTest(
                f"PENDING T-2920: HEAD still equals {_BAD_COMMIT} for {_SC_PATH} -- no fix has "
                "landed yet. Re-run after T-2920 lands."
            )
        sc_frozen = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        for name, schema in _MISMATCH_SCHEMAS.items():
            with self.subTest(schema=name):
                with self.assertRaises(SchemaCompileError):
                    compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())  # shipped: refuses cleanly
                with self.assertRaises(TypeError):
                    sc_frozen.compile_schema_to_mask_pages(schema, _VOCAB_BYTES, special_ids=frozenset())  # frozen: still crashes


if __name__ == "__main__":
    unittest.main()
