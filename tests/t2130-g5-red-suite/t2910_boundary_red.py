"""T-2910 (Curie, T-2913) -- Sec3.9.3's three T-2910 cell groups: the value-open cell, the
asymmetric value-close cell, and the special-token-exclusion cell, folding TE-366's FRACTURE
(D-SLM7586) -- "tokens that cross a structural boundary into or out of free-text content."

RED BY BEHAVIOUR. This branch's own `tools/sslm_convert_schema.py`'s `_token_targets` takes no
`content_states` parameter at all (confirmed at authoring time: `def _token_targets(dfa, state,
root)`), so it applies NO region discipline whatsoever -- a vocabulary token may cross into or
out of the string leaf's content sub-automaton from any depth, in either direction, and nothing
excludes the tokenizer's special/control ids from content. TE-366 demonstrated this is live on
the real A-EX artifact: 16 of 40 real prompts returned a value that was not the model's answer.

GREEN oracle: T-2910's own reference compiler (`Claude/Vitruvius/t2910-probe/
sslm_convert_schema_bytelevel_boundary.py`), loaded via `t2913_common.reference_t2910()`. Its own
"no boundary discipline at all" mutant is T-2908's reference compiler
(`common.reference_t2908()`) directly, reused as the required mutant rather than re-derived by
hand -- T-2910 layers only the region check and special-id zeroing on top of T-2908's own
byte-level module, so "T-2910 with the fix removed" and "T-2908 unmodified" are the same
construction (mirroring the plan's own T-2912-reuses-T-2910/T-2911-modules-as-mutants pattern).

Real-vocabulary counts reproduce Sec3.9.1/Sec3.9.3's own executed figures (1 of 268 at value-
open, 162 true closes at S_c, 22 of 22 special ids reachable under the mutant) using this repo's
own `tools/convert_tokenizer.py` against the real A-EX checkpoint -- fails loudly, never skips
silently, if the checkpoint is missing (T-2909's fail-closed rule).
"""
from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common
from tools.sslm_convert_schema import compile_schema_to_mask_pages

_SCHEMA = common.PROMPT_RESULT_SCHEMA
_PREFIX = b'{"Prompt_Result":'


def _locate_pre_value_and_content(module, vocab: list[bytes]):
    """Compile `_SCHEMA` under a T-2910-shaped reference `module` (one that returns a 4-tuple
    `(char_dfa, char_start, char_accepting, content_states)` from `_char_dfa` and takes
    `content_states` as `_token_targets`'s fourth argument) and return
    `(mask_pages, pre_value_state, s_c_state, content_states_renumbered)`.

    `content_states_renumbered` is computed by REPLICATING `compile_schema_to_mask_pages`'s own
    BFS exactly (same start state, same `sorted(targets)` discovery order) so its numbering is
    identical to the real, returned `mask_pages` -- an independent BFS in a different iteration
    order assigns different ids to the same states, which silently makes a content-membership
    check meaningless (the bug this docstring exists to warn a future editor away from)."""
    char_dfa, char_start, _char_accepting, content_states = module._char_dfa(module._groups(_SCHEMA, "$"))
    trie = module._vocab_trie(tuple(vocab))
    ids: dict[int, int] = {char_start: 0}
    queue: deque[int] = deque([char_start])
    while queue:
        current = queue.popleft()
        targets = module._token_targets(char_dfa, current, trie, content_states)
        for token_id in sorted(targets):
            target = targets[token_id]
            if target not in ids:
                ids[target] = len(ids)
                queue.append(target)
    content_states_new = {ids[state] for state in content_states if state in ids}

    mp = module.compile_schema_to_mask_pages(_SCHEMA, vocab)

    def _walk_literal(state: int, remaining: bytes) -> int:
        while remaining:
            row = mp.transitions[state]
            token_id = max(
                (tid for tid in row if remaining.startswith(vocab[tid])),
                key=lambda tid: len(vocab[tid]),
            )
            state = row[token_id]
            remaining = remaining[len(vocab[token_id]):]
        return state

    pre_value = _walk_literal(mp.start, _PREFIX)
    open_tid = next(tid for tid in mp.transitions[pre_value] if vocab[tid] == b'"')
    s_c = mp.transitions[pre_value][open_tid]
    return mp, pre_value, s_c, content_states_new


# ================================================================================================
# Cell 1 -- value-open: the literal state expecting the value's opening quote admits exactly one
# token, the bare quote. Unconditional -- TE-366 attributed every one of its 14 boundary-leakage
# instances to an open-side compound token.
# ================================================================================================


class T2910_ValueOpen(unittest.TestCase):
    def test_compound_open_token_wrongly_admitted_on_the_branch(self) -> None:
        # A single vocabulary token spelling the opening quote PLUS a content byte in one piece
        # (the packet's own `lo"`-shape reused at the open boundary) must be refused once the
        # boundary discipline lands; the branch has no such discipline and admits it today.
        vocab = ["{", "}", '"Prompt_Result":', 'Prompt_Result":', '"', "a", '"a']
        mp = compile_schema_to_mask_pages(_SCHEMA, vocab)
        full = '{"Prompt_Result":"a"}'
        self.assertTrue(
            mp.accepts(full),
            "regression: the branch no longer admits a compound open+content token -- this cell "
            "needs that (wrong) admission to demonstrate the defect T-2910 fixes",
        )

    def test_compound_open_token_refused_on_reference(self) -> None:
        ref = common.reference_t2910()
        vocab = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"a", b'"a']
        _, pre_value, _s_c, _content = _locate_pre_value_and_content(ref, vocab)
        mp = ref.compile_schema_to_mask_pages(_SCHEMA, vocab)
        admitted = mp.transitions[pre_value]
        admitted_pieces = {vocab[tid] for tid in admitted}
        self.assertEqual(
            admitted_pieces, {b'"'},
            "the value-open state must admit exactly the bare boundary byte, and nothing else",
        )

    def test_real_vocabulary_value_open_row_is_exactly_one_token(self) -> None:
        """Sec3.9.3's own executed figure: 1 of 268 (the compound-token count differs by
        vocabulary composition; the invariant checked here is the count itself, not the 268)."""
        ref = common.reference_t2910()
        vocab = common.real_byte_vocab_zeroed()
        _, pre_value, _s_c, _content = _locate_pre_value_and_content(ref, vocab)
        mp = ref.compile_schema_to_mask_pages(_SCHEMA, vocab)
        admitted = mp.transitions[pre_value]
        self.assertEqual(len(admitted), 1, "the real-vocabulary value-open row must admit exactly one token")
        (only_id,) = admitted.keys()
        self.assertEqual(vocab[only_id], b'"')

    def test_mutant_no_boundary_discipline_restores_spanning_token_admission(self) -> None:
        """T-2908's own reference compiler IS 'T-2910 with the open-side and special-token
        fixes removed' -- reused directly as the mutant rather than re-derived. Real token 3252
        (b'":\"') must be admitted at the value-open position under the mutant and refused under
        the fix; forcing it in the fixed build would select value `:` instead of the model's
        real answer (Sec3.9.1's own executed specimen)."""
        fixed_ref = common.reference_t2910()
        mutant_ref = common.reference_t2908()
        vocab = common.real_byte_vocab_zeroed()

        _, fixed_pre, _, _ = _locate_pre_value_and_content(fixed_ref, vocab)
        fixed_mp = fixed_ref.compile_schema_to_mask_pages(_SCHEMA, vocab)
        fixed_admitted = fixed_mp.transitions[fixed_pre]

        mutant_mp = mutant_ref.compile_schema_to_mask_pages(_SCHEMA, vocab)
        mutant_state = mutant_mp.start
        for byte_val in _PREFIX:
            row = mutant_mp.transitions[mutant_state]
            mutant_state = next(nxt for tid, nxt in row.items() if vocab[tid] == bytes([byte_val]))
        mutant_pre = mutant_state
        mutant_admitted = mutant_mp.transitions[mutant_pre]

        self.assertEqual(vocab[3252], b'":"', "SETUP: the real vocabulary's own token 3252 changed")
        self.assertNotIn(3252, fixed_admitted, "the fix must refuse the spanning token 3252")
        self.assertIn(3252, mutant_admitted, "the mutant must restore admission of the spanning token 3252")
        self.assertEqual(len(fixed_admitted), 1)
        self.assertGreater(len(mutant_admitted), 1, "the mutant must admit more than the bare quote")


# ================================================================================================
# Cell 2 -- value-close: unrestricted at any depth, admitting every token that truly lands
# outside content, whether the bare quote or genuine content bytes followed by the close
# (optionally followed by more literal-matching bytes).
# ================================================================================================


class T2910_ValueClose(unittest.TestCase):
    def test_content_then_close_token_admitted_on_branch_and_reference(self) -> None:
        # 'a."' -- content 'a.', then the closing quote, in one token: a genuine close-side
        # crossing, admitted on both the branch (no discipline at all) and the reference (the
        # close side is deliberately unrestricted).
        vocab_red = ["{", "}", '"Prompt_Result":', 'Prompt_Result":', '"', "a", ".", 'a."']
        full = '{"Prompt_Result":"a."}'
        self.assertTrue(compile_schema_to_mask_pages(_SCHEMA, vocab_red).accepts(full))

        ref = common.reference_t2910()
        vocab_green = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"a", b".", b'a."']
        self.assertTrue(ref.compile_schema_to_mask_pages(_SCHEMA, vocab_green).accepts(_wrap(b"a.")))

    def test_real_vocabulary_true_closes_count_is_162(self) -> None:
        """Sec3.9.3's own executed figure: S_c admits 162 tokens that truly land outside content
        (down from the raw 1,492 tokens that leave S_c at all -- the other 1,330 are UTF-8
        lead-byte transitions into S_c's own interior lead2/lead3/lead4/hex states, still
        content, correctly excluded from this count)."""
        ref = common.reference_t2910()
        vocab = common.real_byte_vocab_zeroed()
        _, _pre_value, s_c, content_states = _locate_pre_value_and_content(ref, vocab)
        mp = ref.compile_schema_to_mask_pages(_SCHEMA, vocab)
        row = mp.transitions[s_c]
        leaves_s_c = [token_id for token_id, target in row.items() if target != s_c]
        true_closes = [token_id for token_id in leaves_s_c if row[token_id] not in content_states]
        self.assertEqual(len(true_closes), 162)
        self.assertEqual(len(leaves_s_c), 1_492)

    def test_mutant_symmetric_close_refusal_reproduces_the_regression(self) -> None:
        """The earlier, fully symmetric attempt (a crossing refused at any depth other than 0,
        BOTH directions) collapses the true-close count from 162 to 1 (the bare quote alone) --
        the product regression the conductor identified, reproduced structurally by restricting
        the close side to depth-zero-only exactly as the open side already is."""
        ref = common.reference_t2910()
        vocab = common.real_byte_vocab_zeroed()

        original_token_targets = ref._token_targets

        def symmetric_token_targets(dfa, state, root, content_states):
            """Both directions restricted to depth zero (the superseded, over-restrictive
            attempt)."""
            targets: dict[int, int] = {}
            stack = [(root, state, 0, False)]
            while stack:
                trie_node, current, depth, stop_after = stack.pop()
                if trie_node.token_id is not None:
                    targets[trie_node.token_id] = current
                if stop_after:
                    continue
                row = dfa[current]
                cur_is_content = current in content_states
                for byte_val, child in trie_node.children.items():
                    target = row.get(byte_val)
                    if target is None:
                        continue
                    crosses = (cur_is_content) != (target in content_states)
                    if crosses and depth != 0:
                        continue
                    stack.append((child, target, depth + 1, crosses))
            return targets

        ref._token_targets = symmetric_token_targets
        try:
            _, _pre_value, s_c, content_states = _locate_pre_value_and_content(ref, vocab)
            mp = ref.compile_schema_to_mask_pages(_SCHEMA, vocab)
        finally:
            ref._token_targets = original_token_targets

        row = mp.transitions[s_c]
        true_closes = [tid for tid, target in row.items() if target != s_c and target not in content_states]
        self.assertEqual(
            len(true_closes), 1,
            "the symmetric mutant must collapse the true-close count to 1 (the bare quote alone)",
        )
        self.assertEqual(vocab[true_closes[0]], b'"')


def _wrap(inner: bytes) -> bytes:
    return b'{"Prompt_Result":"' + inner + b'"}'


# ================================================================================================
# Cell 3 -- no special or control token is ever admitted, in any state, of any schema.
# ================================================================================================


class T2910_SpecialTokenExclusion(unittest.TestCase):
    def test_special_token_reaches_content_on_the_branch(self) -> None:
        # The branch has no producer-side special-id zeroing concept at the compiler level at
        # all (`tools/t2132_build_g5_fixture.py::_real_vocab` performs no such zeroing today) --
        # a "special" piece is just another vocabulary string and is admitted as ordinary
        # content wherever its spelling is legal JSON content, exactly like any other piece.
        special_piece = "<|im_end|>"
        vocab = ["{", "}", '"Prompt_Result":', 'Prompt_Result":', '"', "a", special_piece]
        full = '{"Prompt_Result":"a' + special_piece + '"}'
        mp = compile_schema_to_mask_pages(_SCHEMA, vocab)
        self.assertTrue(
            mp.accepts(full),
            "regression: the branch no longer admits an unzeroed special-shaped piece as "
            "content -- this cell needs that admission to demonstrate the defect T-2910 fixes",
        )

    def test_special_token_excluded_everywhere_on_reference(self) -> None:
        ref = common.reference_t2910()
        special_piece = b"<|im_end|>"
        raw_vocab = [b"{", b"}", b'"Prompt_Result":', b'Prompt_Result":', b'"', b"a", special_piece]
        special_id = raw_vocab.index(special_piece)
        zeroed_vocab = ref.zero_special_ids(raw_vocab, frozenset({special_id}))
        self.assertEqual(zeroed_vocab[special_id], b"")
        mp = ref.compile_schema_to_mask_pages(_SCHEMA, zeroed_vocab)
        admitted_anywhere = {tid for row in mp.transitions.values() for tid in row}
        self.assertNotIn(special_id, admitted_anywhere, "the zeroed special id must be admitted nowhere")

    def test_real_vocabulary_zero_of_22_specials_admitted_anywhere(self) -> None:
        ref = common.reference_t2910()
        vocab = common.real_byte_vocab_zeroed()
        mp = ref.compile_schema_to_mask_pages(_SCHEMA, vocab)
        admitted_anywhere = {tid for row in mp.transitions.values() for tid in row}
        specials = common.real_special_ids()
        self.assertEqual(len(specials), 22)
        self.assertEqual(len(specials & admitted_anywhere), 0)

    def test_mutant_unzeroed_vocabulary_admits_all_22_specials(self) -> None:
        """Sec3.9.3's own executed figure: 22 of 22 admitted somewhere once special-id zeroing
        is omitted (feeding the RAW, unzeroed real vocabulary to the reference compiler)."""
        ref = common.reference_t2910()
        raw_vocab = list(common.real_raw_vocab())  # NOT zeroed
        mp = ref.compile_schema_to_mask_pages(_SCHEMA, raw_vocab)
        admitted_anywhere = {tid for row in mp.transitions.values() for tid in row}
        specials = common.real_special_ids()
        self.assertEqual(len(specials & admitted_anywhere), 22)


if __name__ == "__main__":
    unittest.main()
