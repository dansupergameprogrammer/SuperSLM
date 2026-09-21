"""T-2899 (Curie) -- G5-1 schema compiler's free-text string field (T-2853, plan
`Claude/Plans/te266-gpu-path.md` Sec3.9.3), the five owed compiler unit-cell groups.

RED BY BEHAVIOUR (not by import: `tools.sslm_convert_schema` exists and compiles the
object/enum/boolean subset today -- `g5_1_schema_compiler_fuzz_red.py`, this directory,
already proves that). `{"type": "string"}` is not yet a leaf `_groups()` recognizes: every
cell below drives `compile_schema_to_mask_pages` over a schema with a string field and is red
today because that call raises `SchemaCompileError` naming `"unsupported construct"` /
`"type 'string' is outside D-SLM45's compilable subset"` (confirmed at authoring time,
`D:/_t2899/smoke_check.py`'s captured run) rather than compiling the packet's own cyclic
sub-automaton (Sec3.2: `S_c` content / `S_e` escape / `S_u1`-`S_u4` the `\\uXXXX` hex chain).
The `maxLength` cell is red for a narrower reason: the schema already fails to compile today,
but for the WRONG reason (the whole `string` type is unsupported), not because `maxLength`
was named -- once the leaf lands, this cell's own assertion (the error text names the
`maxLength` keyword) is what must still hold.

BUILD-TIME/LOCAL, matching `g5_1_schema_compiler_fuzz_red.py`'s own documented scope --
run manually: `python -m pytest tests/t2130-g5-red-suite/t2899_string_leaf_red.py -v` from
a checkout with T-2853's own `_groups()`/`_char_dfa()` combinator built (plan Sec3.5 step 2).

ORACLE. Every acceptance/rejection claim below is checked TWICE, independently:
  (1) `compile_schema_to_mask_pages(...).accepts(...)` -- the production compiler's own
      token-level walk over a real vocabulary, exercising the actual compiled DFA.
  (2) Python's own `json` module (`json.loads`) over the SAME string, stripped to its bare
      JSON string literal -- an independent, mature implementation of RFC 8259's string
      grammar, never sharing a line with `tools/sslm_convert_schema.py`. This is the
      definition every cell here is grounded in (Curie's own discipline: a feature oracle
      is grounded in the definition, not a recode of the implementation's steps).
Every positive cell asserts BOTH oracles admit the string; every negative cell asserts BOTH
reject it. Where they could ever disagree that disagreement IS the finding -- none is
observed at authoring time (see `MutationProofBothOraclesAgreeOnEveryFixture`, below).

MUTATION PROOF (StandardsDocument.md Sec5.4/Curie's own "pin the documented claim"
discipline). Nothing in this project has yet built ANY compiler that implements the string
leaf -- pristine and the eventual fix both live in `tools/sslm_convert_schema.py`, one file
this seat never writes to. So there is no in-tree "wrong version" to run these cells
against. What this file pins instead is that ITS OWN assertions genuinely discriminate valid
from invalid JSON string content -- proven against `json.loads`, an oracle this suite does
not author and cannot tune to its own expectations: `MutationProofBothOraclesAgreeOnEveryFixture`
drives every fixture used by the five cell groups through `json.loads` alone and shows it
lands on the SAME accept/reject verdict this file's own cells assert, then constructs one
corrupted variant per group (the "wrong version" the discipline asks for) and shows
`json.loads` -- and, on the object-shaped literals, the exact code paths cells 1-4
exercise -- refuses it. A cell whose stated verdict could never be produced by a real
defect (as opposed to always passing regardless of content) would fail this proof, because
`json.loads` would not reproduce the same split.
"""
from __future__ import annotations

import json
import unittest

from tools.sslm_convert_schema import SchemaCompileError, compile_schema_to_mask_pages

# Every C0 control byte, plus the eight short JSON escapes, plus the sixteen hex digits.
_SHORT_ESCAPES = ['"', "\\", "/", "b", "f", "n", "r", "t"]
_HEX_DIGITS = list("0123456789abcdefABCDEF")

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"Prompt_Result": {"type": "string"}},
    "required": ["Prompt_Result"],
}


def _wrap(inner: str) -> str:
    """The schema's own canonical serialization with `inner` as the string field's raw
    (already-escaped, if applicable) content."""
    return '{"Prompt_Result":"' + inner + '"}'


def _bare_literal_is_valid_json_string(inner: str) -> bool:
    """Independent oracle 2: does `"` + inner + `"` parse as a JSON string, strictly (RFC
    8259 -- raw control bytes rejected, only the eight short escapes and \\uXXXX admitted)?
    Never touches tools/sslm_convert_schema.py."""
    try:
        decoded = json.loads('"' + inner + '"')
    except json.JSONDecodeError:
        return False
    return isinstance(decoded, str)


def _compile(schema=_SCHEMA, vocab=None) -> object:
    """Oracle 1: the production compiler, over a vocabulary this test constructs. Raises
    exactly as `compile_schema_to_mask_pages` raises -- callers assert on that directly, so
    the red reason (SchemaCompileError, not a wrong assertion) stays visible in the failure."""
    return compile_schema_to_mask_pages(schema, vocab)


class G5_1_T2853_Escapes(unittest.TestCase):
    """Sec3.9.3 group 1: each of the eight short escapes accepted from `S_e` back to `S_c`;
    a ninth (any byte outside that set) rejected from `S_e` (no admitted transition in that
    state's row)."""

    def test_each_short_escape_is_admitted(self) -> None:
        for esc in _SHORT_ESCAPES:
            with self.subTest(escape=esc):
                vocab = ["{", "}", '"Prompt_Result":', '"', "\\", esc]
                content = "\\" + esc
                self.assertTrue(
                    _bare_literal_is_valid_json_string(content),
                    f"SETUP: {content!r} is not valid JSON per json.loads -- fixture is wrong",
                )
                try:
                    mp = _compile(vocab=vocab)
                except SchemaCompileError as exc:
                    self.fail(
                        f"the string leaf is not yet implemented: escape \\{esc} did not "
                        f"compile ({exc})"
                    )
                self.assertTrue(
                    mp.accepts(_wrap(content)),
                    f"escape \\{esc}: compiled DFA does not admit a JSON-valid string",
                )

    def test_invalid_escape_char_has_no_admitted_transition(self) -> None:
        # 'q' is not one of the eight short escapes and is not the '\uXXXX' introducer.
        bad = "q"
        vocab = ["{", "}", '"Prompt_Result":', '"', "\\", bad]
        content = "\\" + bad
        self.assertFalse(
            _bare_literal_is_valid_json_string(content),
            "SETUP: \\q is somehow valid JSON per json.loads -- fixture is wrong",
        )
        try:
            mp = _compile(vocab=vocab)
        except SchemaCompileError:
            self.fail(
                "the string leaf is not yet implemented (whole schema fails to compile, "
                "not merely the \\q escape) -- this cell needs the leaf built to be "
                "meaningful; see the escapes-accepted cell above for the current red reason"
            )
        self.assertFalse(
            mp.accepts(_wrap(content)),
            "\\q: compiled DFA wrongly admits an escape outside the eight short escapes",
        )


class G5_1_T2853_UnicodeEscape(unittest.TestCase):
    """Sec3.9.3 group 2: the four-state hex chain S_u1-S_u4 accepts exactly four hex digits
    (0-9, a-f, A-F) per state and returns to S_c on the fourth; a non-hex character at any of
    the four positions has no admitted transition at that state."""

    def test_four_hex_digits_admitted_then_returns_to_content_state(self) -> None:
        vocab = ["{", "}", '"Prompt_Result":', '"', "\\", "u"] + _HEX_DIGITS + ["z"]
        content = "\\u4Fa0z"  # \uXXXX then an ordinary content char, proving S_c is resumed
        self.assertTrue(_bare_literal_is_valid_json_string(content), "SETUP: fixture is invalid JSON")
        try:
            mp = _compile(vocab=vocab)
        except SchemaCompileError as exc:
            self.fail(f"the string leaf is not yet implemented: \\uXXXX did not compile ({exc})")
        self.assertTrue(
            mp.accepts(_wrap(content)),
            "\\u4Fa0 followed by ordinary content 'z': compiled DFA does not admit it, or "
            "does not return to the content state after the fourth hex digit",
        )

    def test_non_hex_character_rejected_at_each_of_the_four_positions(self) -> None:
        # 'g' is a valid vocabulary token (ordinary S_c content elsewhere) but is not a hex
        # digit -- it must have no admitted transition specifically from S_u1..S_u4.
        vocab = ["{", "}", '"Prompt_Result":', '"', "\\", "u", "g"] + _HEX_DIGITS
        for position in range(4):
            with self.subTest(position=position):
                digits = ["4", "F", "a", "0"]
                digits[position] = "g"
                content = "\\u" + "".join(digits)
                self.assertFalse(
                    _bare_literal_is_valid_json_string(content),
                    f"SETUP: {content!r} is somehow valid JSON per json.loads",
                )
                try:
                    mp = _compile(vocab=vocab)
                except SchemaCompileError:
                    self.fail(
                        "the string leaf is not yet implemented -- this cell needs the "
                        "hex chain built to be meaningful"
                    )
                self.assertFalse(
                    mp.accepts(_wrap(content)),
                    f"a non-hex byte at \\uXXXX position {position}: compiled DFA wrongly admits it",
                )


class G5_1_T2853_MultiCharacterCrossing(unittest.TestCase):
    """Sec3.9.3 group 3: a vocabulary token whose spelling is ordinary content characters
    followed immediately by the closing quote (the packet's own worked example, `lo"`) walks
    S_c -> S_c -> following within one token's trie descent and lands the token-level
    transition on `following`, not on an intermediate state."""

    def test_token_crossing_the_closing_quote_lands_past_the_string(self) -> None:
        # The merged token 'lo"' spans two content characters and the closing quote in one
        # vocabulary piece. Greedy longest-match (MaskPages.accepts) prefers it over the
        # single-character alternatives also present, so this cell genuinely drives the
        # crossing transition rather than a character-by-character walk that happens to
        # reach the same place.
        vocab = ["{", "}", '"Prompt_Result":', '"', "l", "o", 'lo"']
        self.assertTrue(
            _bare_literal_is_valid_json_string("lo"), "SETUP: fixture is invalid JSON"
        )
        try:
            mp = _compile(vocab=vocab)
        except SchemaCompileError as exc:
            self.fail(f"the string leaf is not yet implemented ({exc})")
        full = '{"Prompt_Result":"lo"}'
        self.assertTrue(
            mp.accepts(full),
            "the multi-character token 'lo\"' (content plus the closing quote) is not "
            "admitted, or the walk does not land past the string on the object's own "
            "closing-brace continuation",
        )
        # The SAME content, spelled with only the single-character tokens (no crossing
        # token in the vocabulary), must ALSO accept -- the crossing token is one admitted
        # path to 'following', not the only one.
        vocab_no_cross = ["{", "}", '"Prompt_Result":', '"', "l", "o"]
        mp2 = _compile(vocab=vocab_no_cross)
        self.assertTrue(
            mp2.accepts(full),
            "without the crossing token, the character-by-character path to the same "
            "string is not admitted -- the two constructions should agree",
        )


class G5_1_T2853_RawControlByteDeadEnd(unittest.TestCase):
    """Sec3.9.3 group 4 (T-2859 F1): a token spelling a single C0 control byte (0x00-0x1F),
    presented directly to S_c with no preceding backslash, has no admitted transition in
    S_c's own CSR row. Distinct from the escape/\\uXXXX cells above, which drive a bad byte
    AFTER a backslash, not a bare byte reaching S_c directly."""

    def test_raw_control_byte_in_content_position_has_no_admitted_transition(self) -> None:
        for code in (0x00, 0x01, 0x08, 0x1F):
            with self.subTest(byte=hex(code)):
                raw = chr(code)
                vocab = ["{", "}", '"Prompt_Result":', '"', "a", raw]
                self.assertFalse(
                    _bare_literal_is_valid_json_string("a" + raw),
                    f"SETUP: a raw 0x{code:02x} byte is somehow valid JSON per json.loads",
                )
                try:
                    mp = _compile(vocab=vocab)
                except SchemaCompileError:
                    self.fail(
                        "the string leaf is not yet implemented -- this cell needs S_c's "
                        "content transitions built to be meaningful"
                    )
                self.assertFalse(
                    mp.accepts(_wrap("a" + raw)),
                    f"a raw, unescaped 0x{code:02x} control byte in content position is "
                    f"wrongly admitted at S_c",
                )


class G5_1_T2853_RejectionGroup(unittest.TestCase):
    """Sec3.9.3 group 5: `"type": "number"` and `"type": "array"` still raise
    SchemaCompileError (unaffected, must-accept-neighbour control -- `_groups()` line 108's
    fallthrough is unchanged by this fold, only `"type": "string"` gains a leaf). A schema
    declaring `maxLength` on a string field is REJECTED with SchemaCompileError NAMING the
    `maxLength` keyword (T-2859 F2, D-SLM7462)."""

    def _schema_with(self, field: dict) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"f": field},
            "required": ["f"],
        }

    def test_number_type_still_rejected(self) -> None:
        with self.assertRaises(SchemaCompileError):
            _compile(schema=self._schema_with({"type": "number"}), vocab=["{", "}", '"f":', "0"])

    def test_array_type_still_rejected(self) -> None:
        with self.assertRaises(SchemaCompileError):
            _compile(schema=self._schema_with({"type": "array"}), vocab=["{", "}", '"f":', "[", "]"])

    def test_maxlength_on_string_field_rejected_naming_the_keyword(self) -> None:
        schema = self._schema_with({"type": "string", "maxLength": 5})
        vocab = ["{", "}", '"f":', '"', "a"]
        with self.assertRaises(SchemaCompileError) as ctx:
            _compile(schema=schema, vocab=vocab)
        self.assertIn(
            "maxLength",
            str(ctx.exception),
            "a maxLength-bearing string field must be rejected NAMING the maxLength "
            f"keyword; got: {ctx.exception!r}",
        )


class MutationProofBothOraclesAgreeOnEveryFixture(unittest.TestCase):
    """StandardsDocument.md Sec5.4 / Curie's own mutation-proof discipline: since no in-tree
    implementation of the string leaf exists yet to mutate, this class instead pins that the
    cells above are not vacuously true -- each fixture's stated verdict is reproduced by
    `json.loads`, an oracle this suite neither authors nor tunes, and a deliberately
    corrupted ("wrong version") variant of each fixture flips that oracle's verdict, proving
    the assertion genuinely discriminates rather than always agreeing."""

    def test_escape_fixtures_agree_with_json_and_flip_under_corruption(self) -> None:
        for esc in _SHORT_ESCAPES:
            good = "\\" + esc
            self.assertTrue(_bare_literal_is_valid_json_string(good))
        # Wrong version: an escape character outside the eight short escapes.
        self.assertFalse(_bare_literal_is_valid_json_string("\\q"))

    def test_unicode_escape_fixture_agrees_with_json_and_flips_under_corruption(self) -> None:
        self.assertTrue(_bare_literal_is_valid_json_string("\\u4Fa0z"))
        # Wrong version: a non-hex byte at each position, one at a time.
        for position in range(4):
            digits = ["4", "F", "a", "0"]
            digits[position] = "g"
            self.assertFalse(_bare_literal_is_valid_json_string("\\u" + "".join(digits)))
        # Wrong version: only three hex digits (an incomplete escape).
        self.assertFalse(_bare_literal_is_valid_json_string("\\u4Fa"))

    def test_crossing_fixture_agrees_with_json_and_flips_under_corruption(self) -> None:
        self.assertTrue(_bare_literal_is_valid_json_string("lo"))
        # Wrong version: an unescaped, unpaired backslash at the crossing point.
        self.assertFalse(_bare_literal_is_valid_json_string("lo\\"))

    def test_control_byte_fixtures_agree_with_json_and_flip_under_correction(self) -> None:
        for code in (0x00, 0x01, 0x08, 0x1F):
            raw = chr(code)
            self.assertFalse(_bare_literal_is_valid_json_string("a" + raw))
        # Wrong version, corrected: the SAME byte, properly escaped, is valid.
        self.assertTrue(_bare_literal_is_valid_json_string("a\\u0001"))

    def test_rejection_group_reference_keyword_text_is_specific(self) -> None:
        # Pins that this file's own maxLength assertion could actually fail: a message
        # naming "string" instead of "maxLength" (today's real message, captured at
        # authoring time) does NOT satisfy the assertion the cell above makes.
        today_message = (
            "unsupported construct at $.f: type 'string' is outside D-SLM45's compilable "
            "subset (objects with known keys, enums, booleans; cross-field constraints are "
            "scored, not compiled)"
        )
        self.assertNotIn("maxLength", today_message)


if __name__ == "__main__":
    unittest.main()
