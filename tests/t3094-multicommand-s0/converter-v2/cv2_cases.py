"""Named IntentDoc v2 documents for the accepted-language cells (plan §4.1 converter paragraph, §6 row 7).
Each case is (id, raw bytes, Oracle B verdict). The verdicts are written by hand from the contract and are
checked against Oracle B by the reference controls; the product cells compare the converter against them."""
from __future__ import annotations

import cv2_fixtures as F
import cv2_oracle as B

S = F.INTENTDOC


def ser(value):
    return B.serialize(S, value)


ONE = ser(F.doc(F.intent()))
ONE_ATOM = ser(F.doc(F.intent(conditions=[F.atom()])))
TWO = ser(F.doc(F.intent(), F.intent(action="move", conditions=F.atoms(2))))
NESTED = ser(F.doc(F.intent(conditions=F.atoms(1)), F.intent(action="scan"),
                   F.intent(action="hold", conditions=F.atoms(3, 1))))
EIGHT_BY_EIGHT = ser(F.doc(*F.intents(8, conditions_each=8)))

ACCEPT = [
    ("one_intent_empty_conditions", ONE),
    ("one_intent_one_atom", ONE_ATOM),
    ("two_intents_0_and_2_atoms", TWO),
    ("three_intents_1_0_3_atoms_loop_in_loop", NESTED),
    ("eight_intents_eight_atoms_each", EIGHT_BY_EIGHT),
    ("utf8_two_three_four_byte", ser(F.doc(F.intent(subject="café crew", target="東門",
                                                    conditions=[F.atom(entity="🛡 shield")])))),
    ("short_escapes", ONE.replace(b'"alpha wing"', b'"al\\"pha\\\\wing\\/\\b\\f"')),
    ("unicode_escapes_both_cases", ONE.replace(b'"north gate"', b'"n\\u00e9\\u00C9 \\uD83D\\uDE00"')),
    ("structural_bytes_after_meaningful", ONE.replace(b'"north gate"', b'"x,]}{[:"')),
    ("insignificant_prefix_then_meaningful", ONE.replace(b'"alpha wing"', b'" \\n\\t\\u0020go"')),
    ("sentinel_value_dash", ONE_ATOM),
    ("raw_del_byte", ONE.replace(b'"alpha wing"', b'"a\x7fb"')),
    ("sentence_close", ONE.replace(b'"alpha wing"', b'"go now."')),
]

REJECT = [
    ("root_empty_array", b'{"intents":[]}'),
    ("trailing_comma_intents", TWO[:-2] + b",]}"),
    ("trailing_comma_conditions", ONE_ATOM.replace(b'"auto"}]', b'"auto"},]')),
    ("comma_only_conditions", ONE.replace(b'"conditions":[]', b'"conditions":[,]')),
    ("leading_comma_intents", ONE.replace(b'{"intents":[', b'{"intents":[,')),
    ("leading_comma_conditions", ONE_ATOM.replace(b'"conditions":[', b'"conditions":[,')),
    ("double_comma_intents", TWO.replace(b"},{\"action\"", b"},,{\"action\"", 1)),
    ("double_comma_conditions", TWO.replace(b'"},{"role"', b'"},,{"role"', 1)),
    ("whitespace_after_colon", ONE.replace(b'"action":', b'"action": ', 1)),
    ("whitespace_between_items", TWO.replace(b"},{\"action\"", b"}, {\"action\"", 1)),
    ("whitespace_in_empty_conditions", ONE.replace(b'"conditions":[]', b'"conditions":[ ]')),
    ("trailing_newline", ONE + b"\n"),
    ("leading_space", b" " + ONE),
    ("reversed_intent_keys", ONE.replace(b'"action":"fire","subject":"alpha wing"',
                                         b'"subject":"alpha wing","action":"fire"')),
    ("reversed_atom_keys", ONE_ATOM.replace(b'"role":"start","entity":"beacon"', b'"entity":"beacon","role":"start"')),
    ("extra_intent_property", ONE.replace(b'"depends_on_previous":false}', b'"depends_on_previous":false,"x":"y"}')),
    ("extra_atom_property", ONE_ATOM.replace(b'"watch":"auto"}', b'"watch":"auto","x":"y"}')),
    ("missing_conditions_property", ONE.replace(b'"conditions":[],', b"")),
    ("premature_termination", ONE[:-1]),
    ("premature_inside_conditions", ONE_ATOM[:ONE_ATOM.index(b'"auto"}') + 7]),
    ("second_document", ONE + ONE),
    ("empty_leaf", ONE.replace(b'"alpha wing"', b'""')),
    ("insignificant_leaf_space", ONE.replace(b'"alpha wing"', b'" "')),
    ("insignificant_leaf_structural", ONE_ATOM.replace(b'"beacon"', b'"[],{}:"')),
    ("insignificant_leaf_escapes", ONE_ATOM.replace(b'"lit"', b'"\\n\\u0020\\u007b"')),
    ("enum_escaped_spelling", ONE.replace(b'"fire"', b'"f\\u0069re"')),
    ("enum_unknown", ONE.replace(b'"fire"', b'"jump"')),
    ("atom_enum_unknown", ONE_ATOM.replace(b'"auto"', b'"never"')),
    ("boolean_capitalized", ONE.replace(b"false}", b"False}")),
    ("boolean_as_string", ONE_ATOM.replace(b'"negated":false', b'"negated":"false"')),
    ("conditions_as_object", ONE.replace(b'"conditions":[]', b'"conditions":{}')),
    ("nested_array_wrapper", ONE.replace(b'"conditions":[]', b'"conditions":[[]]')),
    ("intents_as_object", b'{"intents":' + ONE[len(b'{"intents":['):-2] + b"}"),
    ("raw_control_byte", ONE.replace(b'"alpha wing"', b'"alpha\twing"')),
    ("utf8_overlong", ONE.replace(b'"alpha wing"', b'"a\xc0\xafb"')),
    ("utf8_encoded_surrogate", ONE.replace(b'"alpha wing"', b'"a\xed\xa0\x80b"')),
    ("utf8_past_max", ONE.replace(b'"alpha wing"', b'"a\xf4\x90\x80\x80b"')),
    ("utf8_bad_lead", ONE.replace(b'"alpha wing"', b'"a\xf5b"')),
    ("bad_short_escape", ONE.replace(b'"alpha wing"', b'"a\\xb"')),
    ("bad_unicode_escape", ONE.replace(b'"alpha wing"', b'"a\\u12G4"')),
]

# Documents for the mutation sweep: each single-byte deletion, insertion and substitution.
MUTATION_BASES = [("one_intent", ONE), ("one_atom", ONE_ATOM), ("two_intents", TWO)]
MUTATION_ALPHABET = b',[]{}":\\ a-'
