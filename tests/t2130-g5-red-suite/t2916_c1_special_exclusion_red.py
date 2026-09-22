"""T-2916 (Curie) -- TE-370 C1's red cells: special-token exclusion for the string leaf is
opt-in per caller, and the only shipped real-vocabulary producer does not opt in
(`Claude/Poirot/te370-final-code-review-2026-09-21.md` Sec2 C1, Wizard repo; probe
`Claude/Poirot/te370-probe/real_special.py`/`real_special_output.txt`).

At `0062c99`, `tools/t2132_build_g5_fixture.py::_real_vocab` returns `TokenizerTables.id_to_bytes`
unmodified -- every one of the real Qwen2.5-0.5B-Instruct tokenizer's 22 special/added ids
(`<|endoftext|>`, `<|im_end|>`, ...) is still spelled out as its literal ASCII bytes, which are
valid string content anywhere in the U/M regions of the string leaf's own content automaton.
Nothing downstream refuses this; `compile_schema_to_mask_pages` takes no special-id input at all.
Executed on the real tokenizer (Poirot's probe, reproduced by this file's own first cell):
22 of 22 specials admitted, each in 2 content states.

Two claims, per the brief (`Claude/Bach/briefs/t2916-te370-red-cells.md`):
  1. The shipped producer PLUS the shipped compiler admits ZERO specials, once fixed.
  2. The compiler REFUSES a vocabulary that has not been through the exclusion step -- the fix
     is structural (a caller cannot forget the step and still get a mask), so this cell pins the
     REFUSAL itself, not only the zero-admission outcome. Poirot's own remedy text names one
     concrete shape for that refusal ("e.g. a required keyword-only `special_ids` argument that
     it applies itself") -- adopted here as this test's pinned contract, since the finding does
     not name a different one and red-first authoring is exactly where a contract like this gets
     fixed before the build starts. If T-2917 lands a different mechanism, this cell is the one
     to reconcile against.

Both cells are RED at `0062c99`: cell 1 because 22/22 specials are admitted where zero must be;
cell 2 because `compile_schema_to_mask_pages(schema, vocab)` -- the shipped producer's own exact
calling convention, unzeroed -- succeeds today instead of being refused.

MUTATION PROOF (StandardsDocument.md Sec5.4): the "wrong version" is not invented -- it is named
directly as the commit TE-370 reviewed and ruled DO NOT SHIP over this exact finding, `0062c99`,
loaded via `t2913_common.frozen_module` (git history, no out-of-repo input). Until T-2917 lands a
fix, HEAD IS `0062c99` for these two files and the mutant class below reports PENDING rather than
fabricating a pass; once HEAD moves past it, the same class requires shipped to pass and the
`0062c99`-frozen copy to still fail on both cells.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common
from tools.sslm_convert_schema import compile_schema_to_mask_pages
from tools import t2132_build_g5_fixture as FX

_SCHEMA = common.PROMPT_RESULT_SCHEMA
_SC_PATH = "tools/sslm_convert_schema.py"
_FX_PATH = "tools/t2132_build_g5_fixture.py"
_BAD_COMMIT = "0062c99"


def _admitted_special_ids(mp, special_ids: frozenset[int]) -> set[int]:
    return {tid for row in mp.transitions.values() for tid in row if tid in special_ids}


class T2916_C1_ShippedProducerAdmitsZeroSpecials(unittest.TestCase):
    """TE-370 C1, claim 1: `_real_vocab` (the shipped, only in-repo real-vocabulary producer)
    plus the shipped compiler must admit zero of the tokenizer's specials as string content,
    executed on the real Qwen2.5-0.5B-Instruct checkpoint -- not a toy vocabulary."""

    def test_real_vocab_producer_plus_compiler_admits_zero_of_22_specials(self) -> None:
        """T-2917 RECONCILIATION (per this file's own line 22-23 allowance): authored assuming
        `_real_vocab` itself would gain the exclusion, so this cell called
        `compile_schema_to_mask_pages(_SCHEMA, vocab)` with no `special_ids`. That call cannot
        both succeed here (claim 1) AND raise `TypeError` in the sibling class below (claim 2) --
        the two claims pinned the SAME two-positional call to opposite outcomes. T-2917 lands
        claim 2's own shape (the pinned contract, explicitly adopted by this file): the compiler
        itself applies the exclusion from a required `special_ids` argument; `_real_vocab` keeps
        returning the raw vocabulary (correct for its OTHER, string-leaf-free callers, see
        `tools/t2132_build_g5_fixture.py`). This cell now supplies `special_ids` explicitly,
        which is exactly `test_compiling_with_special_ids_supplied_succeeds_and_admits_zero`
        below's own positive control -- kept here too as claim 1's own outcome check, over the
        shipped producer's own real vocabulary rather than a hand-zeroed one."""
        common._require_real_model(common.QWEN25_0P5B_CHECKPOINT)
        vocab = FX._real_vocab(str(common.QWEN25_0P5B_CHECKPOINT), common.real_vocab_size())
        special_ids = common.real_special_ids()
        self.assertEqual(len(special_ids), 22, "the real tokenizer's own special-id count moved")

        mp = compile_schema_to_mask_pages(_SCHEMA, vocab, special_ids=special_ids)
        admitted = _admitted_special_ids(mp, special_ids)
        self.assertEqual(
            admitted, set(),
            f"{len(admitted)} of {len(special_ids)} special ids are admitted as string content "
            f"by the shipped real-vocabulary producer plus the shipped compiler -- e.g. "
            f"{sorted(admitted)[:6]}. TE-370 C1: `_real_vocab` must zero every special id before "
            "the compiler ever sees them.",
        )


class T2916_C1_CompilerRefusesAnUnzeroedVocabulary(unittest.TestCase):
    """TE-370 C1, claim 2: the exclusion is structural. The shipped producer's own calling
    convention -- `compile_schema_to_mask_pages(schema, vocab)`, unzeroed -- must be refused,
    not silently compiled. Pins the REFUSAL, not only the zero-admission outcome above: a
    compiler that happened to admit zero specials by chance on THIS vocabulary but had no
    structural check would still pass claim 1 and fail this one."""

    def test_compiling_an_unzeroed_real_vocabulary_is_refused(self) -> None:
        common._require_real_model(common.QWEN25_0P5B_CHECKPOINT)
        vocab = FX._real_vocab(str(common.QWEN25_0P5B_CHECKPOINT), common.real_vocab_size())
        with self.assertRaises(TypeError) as ctx:
            compile_schema_to_mask_pages(_SCHEMA, vocab)  # no special_ids supplied
        self.assertIn(
            "special_ids", str(ctx.exception).lower(),
            "the refusal must name the missing exclusion step (special_ids), not a generic "
            "argument-count error a caller cannot act on to find the fix",
        )

    def test_compiling_with_special_ids_supplied_succeeds_and_admits_zero(self) -> None:
        """The positive control for the cell above: supplying the exclusion step is what makes
        the SAME call succeed, and it must then admit zero specials (claim 1, restated through
        the pinned contract's own calling convention rather than a caller doing the zeroing by
        hand first)."""
        common._require_real_model(common.QWEN25_0P5B_CHECKPOINT)
        vocab = FX._real_vocab(str(common.QWEN25_0P5B_CHECKPOINT), common.real_vocab_size())
        special_ids = common.real_special_ids()
        mp = compile_schema_to_mask_pages(_SCHEMA, vocab, special_ids=special_ids)
        self.assertEqual(_admitted_special_ids(mp, special_ids), set())


class T2916_C1_MutationProof(unittest.TestCase):
    """Sec5.4's mutation-proof discipline, applied without inventing the fix: the wrong version
    is `0062c99` itself, the commit TE-370 reviewed and ruled DO NOT SHIP for this exact defect.
    Loaded via `git show` against this repo's own history (no out-of-repo input, M1's own bar).
    """

    def test_frozen_0062c99_reproduces_22_of_22_admitted(self) -> None:
        """Sanity: the named 'wrong version' actually IS wrong, by execution -- not asserted."""
        common._require_real_model(common.QWEN25_0P5B_CHECKPOINT)
        sc = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        fx = common.frozen_module(_BAD_COMMIT, _FX_PATH)
        vocab = fx._real_vocab(str(common.QWEN25_0P5B_CHECKPOINT), common.real_vocab_size())
        special_ids = common.real_special_ids()
        mp = sc.compile_schema_to_mask_pages(_SCHEMA, vocab)
        admitted = _admitted_special_ids(mp, special_ids)
        self.assertEqual(
            len(admitted), 22,
            f"the frozen {_BAD_COMMIT} copy must reproduce TE-370 C1 exactly (22/22 admitted) "
            f"for this to be a real 'wrong version' control; got {len(admitted)}",
        )

    def test_mutant_discrimination_pending_or_passing(self) -> None:
        """Once T-2917 lands a fix, HEAD differs from `0062c99` for both files and this cell
        requires: shipped admits zero specials AND raises on an unzeroed call (both cells above,
        re-run against the live import), while the `0062c99`-frozen copy still admits all 22 and
        raises nothing -- the mutant is killed by the shipped code and reproduced by the frozen
        one. Before T-2917 lands, HEAD IS `0062c99` for these files and there is no delta to
        discriminate; this cell reports that explicitly via `skipTest`, never a fabricated pass.
        """
        sc_fixed = fix_has_landed = common.fix_has_landed(_BAD_COMMIT, _SC_PATH)
        fx_fixed = common.fix_has_landed(_BAD_COMMIT, _FX_PATH)
        if not (sc_fixed or fx_fixed):
            self.skipTest(
                f"PENDING T-2917: HEAD still equals {_BAD_COMMIT} for both "
                f"{_SC_PATH} and {_FX_PATH} -- no fix has landed yet, so there is nothing for "
                "this mutant to discriminate against. Re-run after T-2917 lands."
            )

        common._require_real_model(common.QWEN25_0P5B_CHECKPOINT)
        vocab_shipped = FX._real_vocab(str(common.QWEN25_0P5B_CHECKPOINT), common.real_vocab_size())
        special_ids = common.real_special_ids()

        mp_shipped = compile_schema_to_mask_pages(_SCHEMA, vocab_shipped, special_ids=special_ids)
        self.assertEqual(
            _admitted_special_ids(mp_shipped, special_ids), set(),
            "shipped code (post-T-2917) still admits specials -- the fix did not close C1",
        )
        with self.assertRaises(TypeError):
            compile_schema_to_mask_pages(_SCHEMA, vocab_shipped)  # unzeroed call, shipped code

        sc_frozen = common.frozen_module(_BAD_COMMIT, _SC_PATH)
        fx_frozen = common.frozen_module(_BAD_COMMIT, _FX_PATH)
        vocab_frozen = fx_frozen._real_vocab(str(common.QWEN25_0P5B_CHECKPOINT), common.real_vocab_size())
        mp_frozen = sc_frozen.compile_schema_to_mask_pages(_SCHEMA, vocab_frozen)
        self.assertEqual(
            len(_admitted_special_ids(mp_frozen, special_ids)), 22,
            f"the {_BAD_COMMIT}-frozen copy no longer reproduces the defect -- it is not a valid "
            "mutant any more (git history should never change under this commit)",
        )


if __name__ == "__main__":
    unittest.main()
