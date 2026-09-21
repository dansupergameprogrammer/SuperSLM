"""T-2899 (Curie) -- G5-1 schema compiler's free-text string field (T-2853, plan
`Claude/Plans/te266-gpu-path.md` Sec3.9.3), the five owed base compiler unit-cell groups plus the
literal-to-literal crossing control -- REVISED (T-2913) to byte-level and value-level semantics
now that Sec3.9.3's design is final across T-2908 (byte alphabet), T-2910 (open/close boundary,
special-token exclusion) and T-2912 (value-level close, superseding T-2911).

STATUS (T-2915: the compiler port landed). This branch's own `tools/sslm_convert_schema.py` now
IS T-2908/T-2910/T-2912's shipped byte-level, value-level design (`compile_schema_to_mask_pages`
takes `Sequence[bytes]`; `_token_targets` carries the `content_states` open/close boundary
discipline). Every cell below is authored against a BYTE-piece vocabulary (`list[bytes]`), passed
as raw bytes to both the branch (oracle 1) and the T-2912 reference chain (oracle 3) -- the two
now run the identical algorithm, which is what this file's own regression-control groups (below)
exist to confirm stays true.

ORACLE. Every acceptance/rejection claim is checked THREE ways:
  (1) `_compile_red(...).accepts(...)` -- this branch's own in-tree compiler, over the raw
      byte-piece vocabulary.
  (2) Python's own `json.loads` over the bare JSON string literal -- an independent, mature
      RFC 8259 implementation sharing no line with either compiler. This is the definition every
      cell is grounded in.
  (3) `_compile_green(...).accepts(...)` -- the T-2912 reference compiler
      (`Claude/Vitruvius/t2912-probe/`), over the same content as real bytes.
Every positive cell asserts all three oracles admit the string; every negative cell asserts all
three reject it.

REGRESSION CONTROLS, NOT RED. The escape/hex/control-byte/rejection groups compile under the
in-tree module both before and after T-2915's port (the string leaf's escape/hex/rejection
mechanism is unchanged by the byte-level/value-level redesign); they are true regression
controls, held green and required to stay green. The literal-to-literal crossing cell is the
same: a must-accept control the boundary-discipline fix does not touch (Sec3.9.3: "T-2910 does
not touch this"). T-2908/T-2910/T-2912's own capabilities (the byte alphabet, the open/close
asymmetry, the value-level closure) are exercised by the sibling files `t2908_byte_level_red.py`,
`t2910_boundary_red.py` and `t2912_value_level_red.py`, whose own branch-vs-reference
characterizations of the pre-port defect are, correspondingly, now branch-vs-reference
confirmations that the port closed each one (named per file, at each site T-2915 touched).

MUTATION PROOF (StandardsDocument.md Sec5.4 / Curie's "pin the documented claim"). Each fixture's
stated verdict is reproduced by `json.loads`, an oracle this suite neither authors nor tunes, and
a deliberately corrupted ("wrong version") variant of each fixture flips that oracle's verdict --
`MutationProofBothOraclesAgreeOnEveryFixture`, unchanged in method from the pre-revision file.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common
from tools.sslm_convert_schema import SchemaCompileError, compile_schema_to_mask_pages

_SHORT_ESCAPES = ['"', "\\", "/", "b", "f", "n", "r", "t"]
_HEX_DIGITS = list("0123456789abcdefABCDEF")

_SCHEMA = common.PROMPT_RESULT_SCHEMA


def _wrap(inner: bytes) -> bytes:
    return b'{"Prompt_Result":"' + inner + b'"}'


def _bare_literal_is_valid_json_string(inner: bytes) -> bool:
    """Independent oracle 2: does `"` + inner + `"` parse as a JSON string, strictly (RFC
    8259)? Never touches either compiler."""
    try:
        decoded = json.loads(b'"' + inner + b'"')
    except json.JSONDecodeError:
        return False
    return isinstance(decoded, str)


def _compile_red(pieces: list[bytes], schema=_SCHEMA):
    """Oracle 1: this branch's own in-tree compiler, over the raw byte pieces (T-2915: ported
    to T-2908/T-2910/T-2912's byte-level/value-level design -- the branch no longer decodes a
    vocabulary piece to `str` before compiling; it is byte-level throughout, exactly like
    oracle 3 below). Kept as its own named oracle, rather than folded into `_compile_green`,
    because the two remain independently callable checks on the SAME production module the
    branch ships versus the records-tree reference chain."""
    return compile_schema_to_mask_pages(schema, list(pieces))


def _compile_green(pieces: list[bytes], schema=_SCHEMA):
    """Oracle 3: the T-2912 reference compiler, over the raw byte pieces -- the alphabet T-2908
    ships. No special ids appear in these small synthetic fixtures, so `zero_special_ids` is not
    needed here."""
    ref = common.reference_t2912()
    return ref.compile_schema_to_mask_pages(schema, list(pieces))


class G5_1_T2853_Escapes(unittest.TestCase):
    """Sec3.9.3 group 1: each of the eight short escapes accepted from `S_e` back to content; a
    ninth (any byte outside that set) rejected from `S_e`. Byte-level revision (T-2913): pieces
    are `bytes`, decoded for RED, raw for GREEN."""

    def test_each_short_escape_is_admitted(self) -> None:
        # Content is prefixed with a plain 'a' so the decoded value always reaches at least one
        # meaningful (M) byte regardless of which escape is under test -- three of the eight
        # short escapes (\n, \r, \t) decode to bytes in T-2912's own ten-byte insignificant set,
        # and a value consisting SOLELY of those bytes has no closing edge under the value-level
        # design (Sec3.9.3 V2/V4). That value-level distinction is this base group's own escape
        # MECHANISM cell's concern only insofar as it must not corrupt an otherwise-meaningful
        # value; the insignificant-alone case is covered by the sibling `t2912_value_level_red.py`.
        for esc in _SHORT_ESCAPES:
            with self.subTest(escape=esc):
                esc_b = esc.encode("ascii")
                vocab = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"\\", esc_b, b"a"]
                content = b"a\\" + esc_b
                self.assertTrue(
                    _bare_literal_is_valid_json_string(content),
                    f"SETUP: {content!r} is not valid JSON per json.loads -- fixture is wrong",
                )
                try:
                    mp_red = _compile_red(vocab)
                except SchemaCompileError as exc:
                    self.fail(f"regression: escape \\{esc} no longer compiles on the branch ({exc})")
                self.assertTrue(
                    mp_red.accepts(_wrap(content)),
                    f"escape \\{esc}: branch DFA does not admit a JSON-valid string",
                )
                mp_green = _compile_green(vocab)
                self.assertTrue(
                    mp_green.accepts(_wrap(content)),
                    f"escape \\{esc}: T-2912 reference DFA does not admit a JSON-valid string",
                )

    def test_invalid_escape_char_has_no_admitted_transition(self) -> None:
        bad = b"q"  # not one of the eight short escapes, not the '\uXXXX' introducer
        vocab = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"\\", bad]
        content = b"\\" + bad
        self.assertFalse(
            _bare_literal_is_valid_json_string(content),
            "SETUP: \\q is somehow valid JSON per json.loads -- fixture is wrong",
        )
        mp_red = _compile_red(vocab)
        self.assertFalse(
            mp_red.accepts(_wrap(content)),
            "\\q: branch DFA wrongly admits an escape outside the eight short escapes",
        )
        mp_green = _compile_green(vocab)
        self.assertFalse(
            mp_green.accepts(_wrap(content)),
            "\\q: T-2912 reference DFA wrongly admits an escape outside the eight short escapes",
        )


class G5_1_T2853_UnicodeEscape(unittest.TestCase):
    """Sec3.9.3 group 2: the four-state hex chain accepts exactly four hex digits and returns to
    content on the fourth; a non-hex character at any of the four positions has no admitted
    transition at that state."""

    def test_four_hex_digits_admitted_then_returns_to_content_state(self) -> None:
        vocab = ([b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"\\", b"u", b"z"]
                 + [d.encode("ascii") for d in _HEX_DIGITS])
        content = b"\\u4Fa0z"  # \uXXXX then ordinary content, proving content is resumed
        self.assertTrue(_bare_literal_is_valid_json_string(content), "SETUP: fixture is invalid JSON")
        mp_red = _compile_red(vocab)
        self.assertTrue(
            mp_red.accepts(_wrap(content)),
            "\\u4Fa0 followed by 'z': branch DFA does not admit it or does not resume content",
        )
        mp_green = _compile_green(vocab)
        self.assertTrue(
            mp_green.accepts(_wrap(content)),
            "\\u4Fa0 followed by 'z': T-2912 reference DFA does not admit it",
        )

    def test_non_hex_character_rejected_at_each_of_the_four_positions(self) -> None:
        vocab = ([b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"\\", b"u", b"g"]
                 + [d.encode("ascii") for d in _HEX_DIGITS])
        for position in range(4):
            with self.subTest(position=position):
                digits = ["4", "F", "a", "0"]
                digits[position] = "g"
                content = ("\\u" + "".join(digits)).encode("ascii")
                self.assertFalse(
                    _bare_literal_is_valid_json_string(content),
                    f"SETUP: {content!r} is somehow valid JSON per json.loads",
                )
                mp_red = _compile_red(vocab)
                self.assertFalse(
                    mp_red.accepts(_wrap(content)),
                    f"non-hex byte at \\uXXXX position {position}: branch DFA wrongly admits it",
                )
                mp_green = _compile_green(vocab)
                self.assertFalse(
                    mp_green.accepts(_wrap(content)),
                    f"non-hex byte at \\uXXXX position {position}: reference DFA wrongly admits it",
                )


class G5_1_T2853_MultiCharacterCrossing(unittest.TestCase):
    """Sec3.9.3 group 3, content-then-close crossing: a vocabulary token spanning ordinary
    content characters and the closing quote in one piece walks content -> content -> following
    in one token's trie descent. This is the packet's own `lo"` worked example -- still admitted
    under T-2910's asymmetric rule (close-side crossings are unrestricted) and under T-2912's
    M-mode automaton (the same close behaviour, unaffected by the U/M split)."""

    def test_token_crossing_the_closing_quote_lands_past_the_string(self) -> None:
        vocab = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"l", b"o", b'lo"']
        self.assertTrue(_bare_literal_is_valid_json_string(b"lo"), "SETUP: fixture is invalid JSON")
        full = _wrap(b"lo")
        mp_red = _compile_red(vocab)
        self.assertTrue(
            mp_red.accepts(full),
            'the multi-character token \'lo"\' is not admitted on the branch, or the walk does '
            "not land past the string on the object's own closing-brace continuation",
        )
        mp_green = _compile_green(vocab)
        self.assertTrue(
            mp_green.accepts(full),
            'the multi-character token \'lo"\' is not admitted by the T-2912 reference (close-'
            "side crossings must remain admitted)",
        )
        vocab_no_cross = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"l", b"o"]
        self.assertTrue(
            _compile_red(vocab_no_cross).accepts(full),
            "without the crossing token, the character-by-character path is not admitted on the "
            "branch -- the two constructions should agree",
        )
        self.assertTrue(
            _compile_green(vocab_no_cross).accepts(full),
            "without the crossing token, the character-by-character path is not admitted by the "
            "reference -- the two constructions should agree",
        )


class G5_1_T2910_LiteralToLiteralCrossing(unittest.TestCase):
    """Sec3.9.3: 'Multi-character tokens crossing between two LITERAL states ... remain admitted
    -- T-2910 does not touch this.' A must-accept regression control, distinct from the
    content-then-close crossing above: this token spans a key's closing quote and the following
    colon -- two literal/skeleton states, never touching the string leaf's own content
    sub-automaton at all. Green on the branch today (the object/enum/boolean subset already
    exercises this crossing shape) and required to stay green under every fold."""

    def test_key_quote_and_colon_crossing_is_unaffected_by_the_boundary_fix(self) -> None:
        # {"Prompt_Result":"x"} -- the token '":' spans the key's closing quote and the
        # following colon, both literal-skeleton bytes, with no content-state endpoint. The
        # whole-literal and missing-leading-quote tokens stay in the vocabulary (matching the
        # base escape/hex cells' own established fixture shape) so every OTHER reachable state
        # in the key literal's own chain still has a covering token -- only the crossing token's
        # own path is the thing under test.
        vocab = [
            b"{", b"}",
            b'"Prompt_Result":',   # whole key literal, one token
            b'Prompt_Result":',    # covers the state reached after a standalone leading quote
            b'"Prompt_Result',     # covers everything up to (not including) the key's own close
            b'":',                 # THE crossing token: key-closing-quote + colon, one token
            b'"',                  # standalone quote (key open, value open, value close)
            b":",                  # standalone colon, covering the closing-quote-then-colon path
            b"x",
        ]
        full = _wrap(b"x")
        self.assertTrue(_bare_literal_is_valid_json_string(b"x"), "SETUP: fixture is invalid JSON")
        mp_red = _compile_red(vocab)
        self.assertTrue(
            mp_red.accepts(full),
            "a literal-to-literal crossing token (key-close-quote + colon) regressed on the branch",
        )
        mp_green = _compile_green(vocab)
        self.assertTrue(
            mp_green.accepts(full),
            "a literal-to-literal crossing token regressed under the T-2912 reference -- the "
            "boundary discipline must never restrict a crossing with no content endpoint",
        )


class G5_1_T2853_RawControlByteDeadEnd(unittest.TestCase):
    """Sec3.9.3 group 4 (T-2859 F1): a token spelling a single C0 control byte (0x00-0x1F),
    presented directly to content with no preceding backslash, has no admitted transition."""

    def test_raw_control_byte_in_content_position_has_no_admitted_transition(self) -> None:
        for code in (0x00, 0x01, 0x08, 0x1F):
            with self.subTest(byte=hex(code)):
                raw = bytes([code])
                vocab = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"a", raw]
                self.assertFalse(
                    _bare_literal_is_valid_json_string(b"a" + raw),
                    f"SETUP: a raw 0x{code:02x} byte is somehow valid JSON per json.loads",
                )
                mp_red = _compile_red(vocab)
                self.assertFalse(
                    mp_red.accepts(_wrap(b"a" + raw)),
                    f"a raw, unescaped 0x{code:02x} control byte is wrongly admitted on the branch",
                )
                mp_green = _compile_green(vocab)
                self.assertFalse(
                    mp_green.accepts(_wrap(b"a" + raw)),
                    f"a raw, unescaped 0x{code:02x} control byte is wrongly admitted by the reference",
                )


class G5_1_T2853_RejectionGroup(unittest.TestCase):
    """Sec3.9.3 group 5: `"type": "number"` and `"type": "array"` still raise
    SchemaCompileError. A schema declaring `maxLength` on a string field is REJECTED naming the
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
            _compile_red([b"{", b"}", b'"f":', b"0"], schema=self._schema_with({"type": "number"}))
        with self.assertRaises(common.reference_t2912().SchemaCompileError):
            _compile_green([b"{", b"}", b'"f":', b"0"], schema=self._schema_with({"type": "number"}))

    def test_array_type_still_rejected(self) -> None:
        with self.assertRaises(SchemaCompileError):
            _compile_red([b"{", b"}", b'"f":', b"[", b"]"], schema=self._schema_with({"type": "array"}))
        with self.assertRaises(common.reference_t2912().SchemaCompileError):
            _compile_green([b"{", b"}", b'"f":', b"[", b"]"], schema=self._schema_with({"type": "array"}))

    def test_maxlength_on_string_field_rejected_naming_the_keyword(self) -> None:
        schema = self._schema_with({"type": "string", "maxLength": 5})
        vocab = [b"{", b"}", b'"f":', b'"', b"a"]
        with self.assertRaises(SchemaCompileError) as ctx:
            _compile_red(vocab, schema=schema)
        self.assertIn("maxLength", str(ctx.exception))
        with self.assertRaises(common.reference_t2912().SchemaCompileError) as ctx_green:
            _compile_green(vocab, schema=schema)
        self.assertIn("maxLength", str(ctx_green.exception))


class MutationProofBothOraclesAgreeOnEveryFixture(unittest.TestCase):
    """StandardsDocument.md Sec5.4 / Curie's "pin the documented claim": pins that this file's
    assertions genuinely discriminate valid from invalid JSON string content, proven against
    `json.loads`, an oracle this suite neither authors nor tunes."""

    def test_escape_fixtures_agree_with_json_and_flip_under_corruption(self) -> None:
        for esc in _SHORT_ESCAPES:
            self.assertTrue(_bare_literal_is_valid_json_string(b"\\" + esc.encode("ascii")))
        self.assertFalse(_bare_literal_is_valid_json_string(b"\\q"))

    def test_unicode_escape_fixture_agrees_with_json_and_flips_under_corruption(self) -> None:
        self.assertTrue(_bare_literal_is_valid_json_string(b"\\u4Fa0z"))
        for position in range(4):
            digits = ["4", "F", "a", "0"]
            digits[position] = "g"
            self.assertFalse(_bare_literal_is_valid_json_string(("\\u" + "".join(digits)).encode("ascii")))
        self.assertFalse(_bare_literal_is_valid_json_string(b"\\u4Fa"))

    def test_crossing_fixture_agrees_with_json_and_flips_under_corruption(self) -> None:
        self.assertTrue(_bare_literal_is_valid_json_string(b"lo"))
        self.assertFalse(_bare_literal_is_valid_json_string(b"lo\\"))

    def test_literal_crossing_fixture_agrees_with_json_and_flips_under_corruption(self) -> None:
        self.assertTrue(_bare_literal_is_valid_json_string(b"x"))
        # Wrong version: an unescaped bare quote inside the content, breaking the string early.
        self.assertFalse(_bare_literal_is_valid_json_string(b'x"y'))

    def test_control_byte_fixtures_agree_with_json_and_flip_under_correction(self) -> None:
        for code in (0x00, 0x01, 0x08, 0x1F):
            raw = bytes([code])
            self.assertFalse(_bare_literal_is_valid_json_string(b"a" + raw))
        self.assertTrue(_bare_literal_is_valid_json_string(b"a\\u0001"))

    def test_rejection_group_reference_keyword_text_is_specific(self) -> None:
        today_message = (
            "unsupported construct at $.f: type 'string' is outside D-SLM45's compilable "
            "subset (objects with known keys, enums, booleans, unbounded free-text strings; "
            "cross-field constraints are scored, not compiled)"
        )
        self.assertNotIn("maxLength", today_message)


if __name__ == "__main__":
    unittest.main()
