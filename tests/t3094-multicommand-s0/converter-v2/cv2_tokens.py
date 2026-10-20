"""Token-boundary fixtures (plan §6 rows 7, 8 and 11; §4.1 `_token_targets` retained): documents
tokenized one byte per token except at one named boundary, where a compound token spans it.

Two directions are pinned at both array levels (the intent item boundary and the condition atom boundary):
- structure-to-content (opening): a token carrying structure into a free-string value, or the opening quote
  plus content, must be refused at that state while the one-byte quote token is admitted;
- content-or-literal-to-structure (closing and literal crossings): a token that closes a value and continues
  into structure, or spans `},{`, `"},{"`, `}],`, `],"`, `}]`, `]}`, `[]` or `":[` without opening a
  value, is admitted and lands where the one-byte walk lands.
Each case is (id, document, boundary offset, compound piece, Oracle B verdict on that tokenization).
"""
from __future__ import annotations

import cv2_cases as C

ONE, ONE_ATOM, TWO = C.ONE, C.ONE_ATOM, C.TWO


def _at(raw, marker, shift=0):
    index = raw.index(marker)
    return index + shift


CASES = [
    # literal crossings at the intent item boundary and the atom boundary
    ("intent_boundary_close_open_brace_quote", TWO, _at(TWO, b'},{"action"'), b'},{"', True),
    ("intent_boundary_close_open_brace", TWO, _at(TWO, b'},{"action"'), b"},{", True),
    ("atom_boundary_quote_close_open_quote", TWO, _at(TWO, b'"},{"role"', 0), b'"},{"', True),
    ("conditions_close_then_comma", TWO, _at(TWO, b'}],"depends'), b"}],", True),
    ("conditions_close_comma_quote", TWO, _at(TWO, b'],"depends'), b'],"', True),
    ("root_array_close_brace", ONE, len(ONE) - 3, b"}]", True),
    ("root_array_close_root", ONE, len(ONE) - 2, b"]}", True),
    ("empty_conditions_pair", ONE, _at(ONE, b"[]"), b"[]", True),
    ("key_close_colon_open_array", ONE, _at(ONE, b'":[]', 0), b'":[', True),
    ("key_quote_and_first_letter_is_literal", ONE, _at(ONE, b'"action"'), b'"a', True),
    ("boolean_literal_whole", ONE, _at(ONE, b"false}"), b"false", True),
    # closing direction out of content
    ("subject_close_comma_key_quote", ONE, _at(ONE, b'","target"'), b'","', True),
    ("subject_close_comma", ONE, _at(ONE, b'","target"'), b'",', True),
    ("atom_value_close_from_content", ONE_ATOM, _at(ONE_ATOM, b'-","negated"'), b'-"', True),
    # opening direction into content: refused
    ("subject_key_close_colon_open_quote", ONE, _at(ONE, b'":"alpha'), b'":"', False),
    ("subject_open_quote_plus_content", ONE, _at(ONE, b'"alpha'), b'"a', False),
    ("atom_value_open_quote_plus_content", ONE_ATOM, _at(ONE_ATOM, b'"-"'), b'"-', False),
    ("entity_key_close_colon_open_quote", ONE_ATOM, _at(ONE_ATOM, b'":"beacon'), b'":"', False),
    ("second_intent_subject_opening", TWO, TWO.rindex(b'"subject":"') + len(b'"subject'), b'":"', False),
    ("second_atom_entity_opening", TWO, TWO.rindex(b'"entity":"') + len(b'"entity'), b'":"', False),
]


def pieces(raw, offset, piece):
    assert raw[offset:offset + len(piece)] == piece, (raw[offset:offset + len(piece)], piece)
    return ([bytes([b]) for b in raw[:offset]] + [piece] +
            [bytes([b]) for b in raw[offset + len(piece):]])


# Byte-vocabulary variant: the 256 single bytes plus every compound piece above, appended in order.
COMPOUNDS = tuple(dict.fromkeys(case[3] for case in CASES))
