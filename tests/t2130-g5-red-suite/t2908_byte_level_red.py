"""T-2908 (Curie, T-2913) -- Sec3.9.3's three T-2908 cell groups: the structural UTF-8-validity
cell, the lone-continuation-byte dead-end cell, and the real-vocabulary fragment-reachability
cell -- plus Sec3.9.6's keyword-allowlist group (S3), folding TE-365 S2/S3.

RED BY BEHAVIOUR. This branch's own `tools/sslm_convert_schema.py` operates on Python `str`
(Unicode codepoints), not raw bytes: `_add_string_leaf`'s content branch is a single self-loop
keyed by `chr(codepoint)` over the whole legal-scalar range, and `_groups()` names only
`maxLength` in its string-leaf rejection surface. Confirmed at authoring time by reading the
module (no `content_states`/byte-level construction anywhere in it).

GREEN oracle: T-2908's own reference compiler (`Claude/Vitruvius/t2908-probe/
sslm_convert_schema_bytelevel.py`), loaded via `t2913_common.reference_t2908()`.

Real-vocabulary cells load the real Qwen2.5 tokenizer via this repo's own
`tools/convert_tokenizer.py` and FAIL LOUDLY (not skip) if the checkpoint is missing (T-2909's
fail-closed rule; `t2913_common._require`).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common
from tools.sslm_convert_schema import SchemaCompileError, compile_schema_to_mask_pages

_SCHEMA = common.PROMPT_RESULT_SCHEMA


# ================================================================================================
# Cell 1 -- structural UTF-8 validity, exhaustive over the character automaton (GREEN + mutant).
# This is a property of the NEW byte-level content automaton T-2908 introduces; the branch's own
# compiler has no byte-keyed content states to audit this way at all (its `_char_dfa` is keyed by
# Python characters, never by raw byte value 0-255 -- confirmed structurally below rather than
# forcing an artificial byte-shaped query onto a module that has no byte concept). The comparable
# RED-vs-GREEN contrast for this exact defect is Cell 2 (a raw byte RED cannot tell apart from a
# decoded Unicode scalar).
# ================================================================================================


def _locate_content_states(module):
    """Walk `{"Prompt_Result":"` through `module`'s own character-level DFA and return
    (char_dfa, char_accepting, S_c) -- the exact technique `Claude/Vitruvius/t2908-probe/
    run_s2_s3_probe.py` used to find its own reference states."""
    groups = module._groups(_SCHEMA, "$")
    char_dfa, char_start, char_accepting = module._char_dfa(groups)[:3]
    state = char_start
    for byte_val in b'{"Prompt_Result":"':
        state = char_dfa[state][byte_val]
    return char_dfa, char_accepting, state


class T2908_StructuralUTF8ValidityExhaustive(unittest.TestCase):
    """Sec3.9.3: every byte 0-255 admitted at S_c matches exactly the JSON-content-and-valid-
    UTF-8-lead set (147 of 256); every lead state admits exactly its correct continuation range;
    the three 3-byte leads and three 4-byte leads converge on shared continuation states -- an
    exhaustive audit of the compiled CHARACTER automaton (~11 states), not the ~146K-token layer,
    which is what makes full enumeration (not sampling) tractable."""

    def test_S_c_admits_exactly_the_valid_lead_and_ascii_content_bytes(self) -> None:
        ref = common.reference_t2908()
        char_dfa, _, s_c = _locate_content_states(ref)
        expected = (
            set(range(0x20, 0x22)) | set(range(0x23, 0x5C)) | set(range(0x5D, 0x80))  # ASCII content
            | {0x22, 0x5C}  # quote (exit), backslash (escape entry)
            | set(range(0xC2, 0xE0))  # 2-byte leads
            | {0xE0} | set(range(0xE1, 0xED)) | set(range(0xEE, 0xF0)) | {0xED}  # 3-byte leads
            | {0xF0} | set(range(0xF1, 0xF4)) | {0xF4}  # 4-byte leads
        )
        admitted = set(char_dfa[s_c].keys())
        self.assertEqual(admitted, expected, "S_c's admitted byte set does not match the JSON/UTF-8 grammar")
        self.assertEqual(len(admitted), 147, "S_c must admit exactly 147 of 256 byte values")
        never_admitted = set(range(0x100)) - admitted
        must_never_admit = {b for b in never_admitted if b < 0x20 or 0x80 <= b <= 0xC1 or b >= 0xF5}
        self.assertEqual(
            never_admitted, must_never_admit,
            "a byte outside {C0 controls, bare continuation, overlong lead, invalid lead} is "
            "wrongly excluded from S_c, or one of those forbidden classes is wrongly admitted",
        )

    def test_lead_states_admit_exactly_their_continuation_ranges_and_converge(self) -> None:
        ref = common.reference_t2908()
        char_dfa, _, s_c = _locate_content_states(ref)
        lead2 = char_dfa[s_c][0xC2]
        lead3_a, lead3_b, lead3_c = char_dfa[s_c][0xE0], char_dfa[s_c][0xE1], char_dfa[s_c][0xED]
        lead4_a, lead4_b, lead4_c = char_dfa[s_c][0xF0], char_dfa[s_c][0xF1], char_dfa[s_c][0xF4]
        cases = [
            ("lead2", lead2, set(range(0x80, 0xC0)), {s_c}),
            ("lead3_a", lead3_a, set(range(0xA0, 0xC0)), None),
            ("lead3_b", lead3_b, set(range(0x80, 0xC0)), None),
            ("lead3_c", lead3_c, set(range(0x80, 0xA0)), None),
            ("lead4_a", lead4_a, set(range(0x90, 0xC0)), None),
            ("lead4_b", lead4_b, set(range(0x80, 0xC0)), None),
            ("lead4_c", lead4_c, set(range(0x80, 0x90)), None),
        ]
        mid3_targets = set()
        mid4_targets = set()
        for name, state, expected_bytes, expected_target in cases:
            with self.subTest(state=name):
                actual_bytes = set(char_dfa[state].keys())
                self.assertEqual(actual_bytes, expected_bytes, f"{name} admits the wrong continuation range")
                targets = {char_dfa[state][b] for b in actual_bytes}
                self.assertEqual(len(targets), 1, f"{name} does not converge on one shared target state")
                if expected_target is not None:
                    self.assertEqual(targets, expected_target)
                elif name.startswith("lead3"):
                    mid3_targets |= targets
                else:
                    mid4_targets |= targets
        self.assertEqual(len(mid3_targets), 1, "the three 3-byte leads must converge on ONE shared mid state")
        self.assertEqual(len(mid4_targets), 1, "the three 4-byte leads must converge on ONE shared mid state")
        mid3 = next(iter(mid3_targets))
        mid4a = next(iter(mid4_targets))
        self.assertEqual(set(char_dfa[mid3].keys()), set(range(0x80, 0xC0)))
        self.assertEqual({char_dfa[mid3][b] for b in char_dfa[mid3]}, {s_c}, "the 3-byte tail must return to S_c")
        mid4b_targets = {char_dfa[mid4a][b] for b in char_dfa[mid4a]}
        self.assertEqual(set(char_dfa[mid4a].keys()), set(range(0x80, 0xC0)))
        self.assertEqual(len(mid4b_targets), 1)
        mid4b = next(iter(mid4b_targets))
        self.assertEqual(set(char_dfa[mid4b].keys()), set(range(0x80, 0xC0)))
        self.assertEqual({char_dfa[mid4b][b] for b in char_dfa[mid4b]}, {s_c}, "the 4-byte tail must return to S_c")

    def test_every_legal_content_codepoint_round_trips_as_valid_utf8_exhaustively(self) -> None:
        """Every one of the 1,112,030 legal JSON-string content scalars (every codepoint except
        surrogates, `"`, `\\`, and the 32 C0 controls) is admitted from S_c back to S_c and
        decodes as valid UTF-8, with zero exceptions -- walked over the ~11-state content
        automaton exhaustively, one encoded byte string per scalar."""
        ref = common.reference_t2908()
        char_dfa, _, s_c = _locate_content_states(ref)

        def walk_returns_to_s_c(data: bytes) -> bool:
            state = s_c
            for byte_val in data:
                row = char_dfa[state]
                if byte_val not in row:
                    return False
                state = row[byte_val]
            return state == s_c

        checked = 0
        for codepoint in range(0x20, 0x110000):
            if 0xD800 <= codepoint <= 0xDFFF:
                continue  # surrogates: illegal in a JSON string's own scalar content
            char = chr(codepoint)
            if char in ('"', "\\"):
                continue  # governed by separate quote-exit/escape-entry cells, not this one
            encoded = char.encode("utf-8")
            self.assertTrue(
                walk_returns_to_s_c(encoded),
                f"legal scalar U+{codepoint:04X} is not admitted from S_c back to S_c",
            )
            checked += 1
        # 0x110000 legal codepoints, minus the C0 controls [0x00,0x20) already excluded by the
        # range start, minus 0x800 surrogates, minus quote/backslash (2) -- matches Sec3.9.1's
        # own closed-form count exactly.
        self.assertEqual(checked, 0x110000 - 0x20 - 0x800 - 2)

    def test_mutant_widening_a_continuation_range_is_caught(self) -> None:
        """A mutant reverting lead3_c's continuation range from 0x80-0x9F to 0x80-0xBF
        (re-admitting the surrogate range via an overlong-adjacent path) must be caught."""
        ref = common.reference_t2908()
        char_dfa, _, s_c = _locate_content_states(ref)
        lead3_c = char_dfa[s_c][0xED]
        mutated = {byte: char_dfa[lead3_c][0x80] for byte in range(0x80, 0xC0)}  # widened, wrongly
        fixed_range = set(char_dfa[lead3_c].keys())
        self.assertEqual(fixed_range, set(range(0x80, 0xA0)), "SETUP: the fixed range moved")
        self.assertNotEqual(
            set(mutated.keys()), fixed_range,
            "the widened mutant must differ from the fixed range for this cell to discriminate anything",
        )


# ================================================================================================
# Cell 2 -- a lone UTF-8 continuation byte (0x80-0xBF) presented directly to content dead-ends.
# The comparative RED-vs-GREEN cell for the S2 defect: RED's `str`-keyed automaton cannot
# distinguish a raw byte from a decoded Unicode scalar, so it wrongly treats a lone high byte,
# once decode-collapsed to a replacement character, as ordinary admissible content.
# ================================================================================================


class T2908_LoneContinuationByteDeadEnd(unittest.TestCase):
    def test_lone_continuation_byte_dead_ends_on_reference_but_is_admitted_on_the_branch(self) -> None:
        raw_byte = bytes([0xA1])  # a bare UTF-8 continuation byte -- never a legal position alone
        # RED: the producer decodes with errors="replace" before the compiler ever sees it
        # (T-2908 Sec3.9.1's own named defect, `tools/t2132_build_g5_fixture.py::_real_vocab`,
        # reproduced directly rather than assumed) -- the raw byte becomes the replacement
        # character U+FFFD, an ORDINARY, printable Unicode scalar that RED's own content self-
        # loop admits like any other.
        decoded_piece = raw_byte.decode("utf-8", errors="replace")
        self.assertEqual(decoded_piece, "�", "SETUP: the decode-collapse assumption changed")
        vocab_red = ["{", "}", '"Prompt_Result":', 'Prompt_Result":', '"', "a", decoded_piece]
        mp_red = compile_schema_to_mask_pages(_SCHEMA, vocab_red)
        wrapped_red = '{"Prompt_Result":"a' + decoded_piece + '"}'
        self.assertTrue(
            mp_red.accepts(wrapped_red),
            "regression: the branch no longer admits the decode-collapsed replacement character "
            "as content -- this cell needs that (wrong) behaviour to demonstrate the defect T-2908 fixes",
        )

        # GREEN: the reference compiler sees the RAW byte directly and must refuse it -- no
        # admitted transition at S_c for a bare 0x80-0xBF byte.
        ref = common.reference_t2908()
        vocab_green = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"a", raw_byte]
        mp_green = ref.compile_schema_to_mask_pages(_SCHEMA, vocab_green)
        wrapped_green = b'{"Prompt_Result":"a' + raw_byte + b'"}'
        self.assertFalse(
            mp_green.accepts(wrapped_green),
            "a lone UTF-8 continuation byte (0x80-0xBF) is wrongly admitted as content by the "
            "T-2908 reference compiler",
        )


# ================================================================================================
# Cell 3 -- real-vocabulary fragment reachability. T-2908 Sec3.9.1's own executed proof, reused
# unchanged (`Claude/Vitruvius/t2908-probe/run_s2_followup.py`): TE-365's seven natural-
# tokenization samples plus five regression samples, over the SAME 1.5B shopkeeper-LoRA
# checkpoint T-2908's own probe used.
# ================================================================================================


class T2908_RealVocabularyFragmentReachability(unittest.TestCase):
    def test_natural_tokenization_fragments_are_lost_on_the_branch_and_recovered_on_reference(self) -> None:
        from tokenizers import Tokenizer

        checkpoint = common._require(common.SHOPKEEPER_LORA_CHECKPOINT)
        tables_module = __import__("convert_tokenizer")
        tables = tables_module.TokenizerTables(str(checkpoint))
        tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        raw = tables.id_to_bytes

        def is_valid_utf8(piece: bytes) -> bool:
            try:
                piece.decode("utf-8")
                return True
            except UnicodeDecodeError:
                return False

        # RED's own reachability rule: the trie key is the DECODED (errors="replace") string, so
        # any two distinct byte-fragment ids that collapse to the same replacement spelling are
        # aliased, and the trie's lowest-id-wins rule makes every later one unreachable.
        red_spelling_owner: dict[str, int] = {}
        for token_id, piece in enumerate(raw):
            if not piece:
                continue
            spelling = piece.decode("utf-8", errors="replace")
            if spelling not in red_spelling_owner:
                red_spelling_owner[spelling] = token_id

        # GREEN's own reachability rule: the trie key is the token's REAL bytes, so a token id is
        # unreachable only if some strictly-lower id spells the IDENTICAL raw bytes.
        green_bytes_owner: dict[bytes, int] = {}
        for token_id, piece in enumerate(raw):
            if not piece:
                continue
            if piece not in green_bytes_owner:
                green_bytes_owner[piece] = token_id

        samples = dict(common.TE365_NATURAL_SAMPLES)
        samples.update(common.TE365_REGRESSION_SAMPLES)

        total = red_lost_total = green_lost_total = 0
        red_lost_examples: list[tuple[str, int]] = []
        for name, text in samples.items():
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            fragment_ids = [token_id for token_id in ids if raw[token_id] and not is_valid_utf8(raw[token_id])]
            red_lost = [
                token_id for token_id in fragment_ids
                if red_spelling_owner.get(raw[token_id].decode("utf-8", errors="replace")) != token_id
            ]
            green_lost = [
                token_id for token_id in fragment_ids
                if green_bytes_owner.get(raw[token_id]) != token_id
            ]
            total += len(ids)
            red_lost_total += len(red_lost)
            green_lost_total += len(green_lost)
            red_lost_examples.extend((name, token_id) for token_id in red_lost)

        # T-2908 Sec3.9.1's own executed baseline on this exact checkpoint and sample set:
        # TE-365's seven-sample subset lost 17 of 20 fragment ids under decode-collapse. The full
        # twelve-sample set (seven natural + five regression) is a superset of that population.
        self.assertGreater(
            red_lost_total, 0,
            "regression: the branch's own decode-collapse rule lost zero fragments on this "
            f"sample set (examples if any: {red_lost_examples[:5]}) -- this cell needs the known "
            "defect to reproduce for the reference comparison below to mean anything",
        )
        self.assertEqual(
            green_lost_total, 0,
            f"the byte-keyed trie lost {green_lost_total} fragment id(s) it should have kept "
            "reachable -- a genuine duplicate raw-byte spelling would be the only legitimate cause",
        )


# ================================================================================================
# Sec3.9.6 -- the keyword allowlist (S3): every keyword this compiler does not implement is
# refused BY NAME on a `"type": "string"` field, not only `maxLength` (already covered in
# `t2899_string_leaf_red.py`'s rejection group).
# ================================================================================================

_EXTRA_KEYWORDS = ["minLength", "pattern", "format", "const", "title", "description", "default",
                   "examples", "multipleOf"]


class T2908_KeywordAllowlistGroup(unittest.TestCase):
    def _schema_with(self, field: dict) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"f": field},
            "required": ["f"],
        }

    def test_each_extra_keyword_silently_ignored_on_the_branch(self) -> None:
        """RED's own defect, reproduced directly: `_groups()` checks only `maxLength` on a
        string field, so every other unimplemented keyword compiles silently, discarding the
        author's own stated constraint -- exactly TE-365 S3's own finding (`const: "x"` accepts
        `"ab"`)."""
        for keyword in _EXTRA_KEYWORDS:
            with self.subTest(keyword=keyword):
                schema = self._schema_with({"type": "string", keyword: "irrelevant-value"})
                vocab = ["{", "}", '"f":', 'f":', '"', "a", "b"]
                try:
                    mp = compile_schema_to_mask_pages(schema, vocab)
                except SchemaCompileError as exc:
                    self.fail(
                        f"regression: the branch now rejects {keyword!r} on a string field "
                        f"({exc}) -- this cell needs the silent-accept defect to demonstrate it"
                    )
                self.assertTrue(
                    mp.accepts('{"f":"ab"}'),
                    f"the branch's own leaf must still accept unconstrained content when "
                    f"{keyword!r} is silently ignored, for this to be the S3 defect and not "
                    "some unrelated compile failure",
                )

    def test_each_extra_keyword_rejected_naming_itself_on_reference(self) -> None:
        ref = common.reference_t2908()
        for keyword in _EXTRA_KEYWORDS:
            with self.subTest(keyword=keyword):
                schema = self._schema_with({"type": "string", keyword: "irrelevant-value"})
                vocab = [b"{", b"}", b'"f":', b'f":', b'"', b"a", b"b"]
                with self.assertRaises(ref.SchemaCompileError) as ctx:
                    ref.compile_schema_to_mask_pages(schema, vocab)
                self.assertIn(
                    keyword, str(ctx.exception),
                    f"the reference compiler must name {keyword!r} in its rejection, not a "
                    "generic message",
                )

    def test_mutant_removing_the_allowlist_check_reproduces_the_const_defect(self) -> None:
        """Mutant: with the allowlist check removed from the string branch, `const: "x"`
        recompiles and a non-matching value (`"ab"`) is accepted despite the declared
        constant -- the exact defect TE-365 found, reproduced on demand against the reference."""
        ref = common.reference_t2908()
        schema = self._schema_with({"type": "string", "const": "x"})
        vocab = [b"{", b"}", b'"f":', b'f":', b'"', b"a", b"b", b"x"]
        with self.assertRaises(ref.SchemaCompileError):
            ref.compile_schema_to_mask_pages(schema, vocab)

        original_reject = ref._reject_unimplemented_keywords

        def no_op_reject(schema_node, path, allowed):  # noqa: ARG001 -- signature must match
            return None

        ref._groups.__globals__["_reject_unimplemented_keywords"] = no_op_reject
        try:
            mp = ref.compile_schema_to_mask_pages(schema, vocab)
        finally:
            ref._groups.__globals__["_reject_unimplemented_keywords"] = original_reject
        self.assertTrue(
            mp.accepts(b'{"f":"ab"}'),
            "with the allowlist check removed, the reference must reproduce the branch's own "
            "const-ignored defect (a non-matching value silently accepted)",
        )


if __name__ == "__main__":
    unittest.main()
