"""T-2912 (Curie, T-2913) -- Sec3.9.3's five value-level test-author cells (V1-V5), the three
named mutants, and the commissioned answer-value oracle's own re-confirmation. Supersedes
T-2911's token-local structural-close blacklist (deleted, not retained as a second guard --
V3/V4 below prove it contributes no accepted-language closure).

STATUS (T-2915: the compiler port landed). This branch's own `tools/sslm_convert_schema.py` now
carries T-2912's own U/M value-level closure: a decoded value made SOLELY of JSON punctuation and
whitespace cannot close, on the branch exactly as on the reference. V2/V3 below confirm this
directly, over the branch's own real vocabulary (`t2913_common.real_red_mask_pages`, now compiled
byte-level like the reference). The three required mutants (dropping U's own close guard,
admitting specials, relaxing opening strictness) are checked against the reference chain, per the
plan's own Sec3.9.3 -- unaffected by which module "branch" resolves to, since a mutant
characterizes the DESIGN's own defect class, not a live divergence from the branch.

GREEN oracle: the T-2912 reference compiler (`Claude/Vitruvius/t2912-probe/
sslm_convert_schema_value_level.py`), loaded via `t2913_common.reference_t2912()`.

V1 and V5 run the REAL A-EX artifact through the real, unmodified TE-366 C++ harness
(`D:/_te368/probe/te368_schema_run.exe`, the CPU decode path both R-T2853a/R-T2853b already
adopt) -- genuine live decode, not a proxy. V5 additionally reuses T-2912's own pre-registered,
hashed heldout prompts and its commissioned `T2912-answer-value-oracle` UNALTERED (the brief's
hard rule). Both cells build the branch's own artifact locally (`_build_red_artifact`, now the
SAME production compiler and vocabulary producer the reference chain's own construction uses)
and verify the pinned GREEN artifact's own SHA-256 before using it, so a drifted or missing
artifact fails loudly rather than silently; both are now expected to agree, which V1/V5 confirm
by execution rather than by assuming the port is complete.

Filtered/capped reads only in every helper below -- no raw token lists, byte/hex dumps, or bulk
model-output prints; only counts, short verdicts, and small named examples.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common

_SCHEMA = common.PROMPT_RESULT_SCHEMA
_PREFIX = b'{"Prompt_Result":'
_ENGINE = Path("D:/SuperSLM/.worktrees/t2809-stage1-build")
_HARNESS = Path("D:/_te368/probe/te368_schema_run.exe")
_GREEN_ARTIFACT = Path("D:/_t2912/probe/aex-prompt-result-t2912.sslm")
_GREEN_ARTIFACT_SHA256 = "6a41f87d3a48c91751b41fe716c4f8289b7ce97761850c3f1f2c53d01201205c"
_CANONICAL_CONTROL_IDS = common._VITRUVIUS / "t2912-probe" / "t2912_canonical_control.ids"
_SCRATCH = Path("D:/_t2913/probe")


def _require_engine_artifacts() -> None:
    for path in (_HARNESS, _GREEN_ARTIFACT, common.QWEN25_0P5B_CHECKPOINT, common.A_EX_ARTIFACT):
        if not path.exists():
            raise FileNotFoundError(f"real-model cell needs {path}, which is missing on this box")
    actual = hashlib.sha256(_GREEN_ARTIFACT.read_bytes()).hexdigest()
    if actual != _GREEN_ARTIFACT_SHA256:
        raise AssertionError(
            f"the pinned T-2912 artifact {_GREEN_ARTIFACT} has drifted: sha256={actual}, "
            f"expected {_GREEN_ARTIFACT_SHA256} -- refusing to grade against an unverified artifact"
        )


# ================================================================================================
# Shared value-language machinery: locate pre_value/U/M for GREEN (T-2912) and pre_value/S_c for
# RED, then walk single-byte/single-char tokens directly (fast -- avoids MaskPages.accepts'
# O(row size) generic string search at real-vocabulary scale).
# ================================================================================================


@lru_cache(maxsize=1)
def _green_states():
    mp = common.real_green_mask_pages()
    vocab = common.real_byte_vocab_zeroed()
    single = common.single_byte_token_ids(vocab)
    pre_value = common.walk_literal(mp, vocab, mp.start, _PREFIX)
    u_state = mp.transitions[pre_value][single[0x22]]
    m_state = mp.transitions[u_state][single[ord("A")]]
    return mp, vocab, single, pre_value, u_state, m_state


def _walk_single_bytes(mp, single: dict[int, int], state: int, data: bytes) -> int | None:
    for byte_val in data:
        token_id = single.get(byte_val)
        if token_id is None or token_id not in mp.transitions.get(state, {}):
            return None
        state = mp.transitions[state][token_id]
    return state


def _green_completes(content: bytes, tail: bytes = b"") -> bool:
    """Does `content` (walked from U), the closing quote, then `tail` (walked as ordinary
    literal/skeleton bytes -- NOT content) reach the object's own accepting close? `tail` is the
    bytes a compound close token would carry AFTER its own closing quote (Sec3.9.3's split-path
    and multi-character-close shapes); it defaults to empty for a plain "content then close"
    check."""
    mp, _vocab, single, _pre_value, u_state, _m_state = _green_states()
    state = _walk_single_bytes(mp, single, u_state, content)
    if state is None:
        return False
    quote_id = single[0x22]
    if quote_id not in mp.transitions.get(state, {}):
        return False
    state = mp.transitions[state][quote_id]
    state = _walk_single_bytes(mp, single, state, tail)
    if state is None:
        return False
    close_id = single[0x7D]
    if close_id not in mp.transitions.get(state, {}):
        return False
    state = mp.transitions[state][close_id]
    return state in mp.accepting


@lru_cache(maxsize=1)
def _red_states():
    """T-2915 (the compiler port landed): the branch now carries the identical U/M value-level
    closure the reference does (`_green_states`, above) -- mirrored here over the branch's own
    unzeroed real vocabulary (`common.real_red_mask_pages`) rather than duplicated by a separate
    single-content-state walk, since the branch is no longer a single-content-state design."""
    mp = common.real_red_mask_pages()
    vocab = common.real_raw_vocab()
    single = common.single_byte_token_ids(vocab)
    pre_value = common.walk_literal(mp, vocab, mp.start, _PREFIX)
    u_state = mp.transitions[pre_value][single[0x22]]
    m_state = mp.transitions[u_state][single[ord("A")]]
    return mp, vocab, single, pre_value, u_state, m_state


def _red_completes(content: bytes, tail: bytes = b"") -> bool:
    """Mirrors `_green_completes`'s own content/close/tail/close shape exactly, over the
    branch's own real vocabulary (unzeroed) -- included so V2/V4's branch-vs-reference
    agreement is measured the same way on both sides, not asserted from the design description
    alone."""
    mp, _vocab, single, _pre_value, u_state, _m_state = _red_states()
    state = _walk_single_bytes(mp, single, u_state, content)
    if state is None:
        return False
    quote_id = single[0x22]
    if quote_id not in mp.transitions.get(state, {}):
        return False
    state = mp.transitions[state][quote_id]
    state = _walk_single_bytes(mp, single, state, tail)
    if state is None:
        return False
    close_id = single[0x7D]
    if close_id not in mp.transitions.get(state, {}):
        return False
    state = mp.transitions[state][close_id]
    return state in mp.accepting


# ================================================================================================
# V2 -- TE-368 complete class census, generalized from token spellings to the decoded value.
# ================================================================================================


class T2912_V2_CompleteClassCensus(unittest.TestCase):
    def test_punctuation_only_value_refused_on_branch_and_reference(self) -> None:
        """T-2915 (the compiler port landed): a bare '{' is refused as a complete string value
        on both the branch and the reference -- the class T-2912 closes."""
        self.assertFalse(
            _red_completes(b"{"),
            "a punctuation-only value ('{') is wrongly closable on the branch",
        )
        self.assertFalse(
            _green_completes(b"{"),
            "a punctuation-only value ('{') is wrongly closable on the T-2912 reference",
        )

    def test_te368_structural_only_values_length_one_to_three(self) -> None:
        """Sec3.9.3's own executed figure: 0 of 84 accepted, now on both branch and reference."""
        population = list(itertools.chain.from_iterable(
            itertools.product(b"{},:", repeat=length) for length in (1, 2, 3)
        ))
        self.assertEqual(len(population), 84)
        green_accepted = sum(_green_completes(bytes(value)) for value in population)
        red_accepted = sum(_red_completes(bytes(value)) for value in population)
        self.assertEqual(green_accepted, 0, f"{green_accepted} of 84 structural-only values wrongly closable on reference")
        self.assertEqual(red_accepted, 0, f"{red_accepted} of 84 structural-only values wrongly closable on the branch")


# ================================================================================================
# V3 -- TE-368's unchanged split-path walk: refused-value split paths (a punctuation prefix in
# one token, the closing quote in a separate token) must not complete either.
# ================================================================================================


class T2912_V3_SplitPaths(unittest.TestCase):
    def test_split_paths_derived_at_source_are_all_refused_on_reference(self) -> None:
        """The eight refused ids are DERIVED (T-2910 admits, T-2911 refuses), not copied from
        the record, matching Sec3.9's own "derived at source" discipline."""
        t2910 = common.reference_t2910()
        t2911 = common.reference_t2911()
        vocab = common.real_byte_vocab_zeroed()

        def locate_s_c(module):
            mp = module.compile_schema_to_mask_pages(_SCHEMA, vocab)
            single = common.single_byte_token_ids(vocab)
            pre_value = common.walk_literal(mp, vocab, mp.start, _PREFIX)
            s_c = mp.transitions[pre_value][single[0x22]]
            return mp, s_c

        t2910_mp, t2910_sc = locate_s_c(t2910)
        t2911_mp, t2911_sc = locate_s_c(t2911)
        refused_ids = sorted(set(t2910_mp.transitions[t2910_sc]) - set(t2911_mp.transitions[t2911_sc]))
        self.assertEqual(len(refused_ids), 8)

        green_completed = 0
        red_completed = 0
        for token_id in refused_ids:
            piece = vocab[token_id]
            quote_at = piece.index(0x22)
            content, tail = piece[:quote_at], piece[quote_at + 1:]
            green_completed += int(_green_completes(content, tail))
            red_completed += int(_red_completes(content, tail))
        self.assertEqual(green_completed, 0, f"{green_completed} of 8 split paths wrongly complete on reference")
        self.assertEqual(
            red_completed, 0,
            f"{red_completed} of 8 split paths wrongly complete on the branch (T-2915: the "
            "compiler port landed, so the branch now carries the identical value-level "
            "closure the reference does)",
        )


# ================================================================================================
# V4 -- exhaustive accepted values through four bytes, plus the all-length monotone-state proof.
# ================================================================================================


class T2912_V4_ExhaustiveAcceptedValues(unittest.TestCase):
    def test_raw_and_canonical_banned_values_length_1_to_4(self) -> None:
        """Sec3.9.3's own executed figures: 0/11110 (raw) and 0/11110 (canonically spelled)."""
        banned = tuple(sorted(b"{}[],: \t\r\n"))
        raw_total = raw_accepted = 0
        semantic_total = semantic_accepted = 0
        for length in range(1, 5):
            for value in itertools.product(banned, repeat=length):
                raw = bytes(value)
                raw_total += 1
                raw_accepted += int(_green_completes(raw))
                semantic = b"".join(common.canonical_escape_spelling(b) for b in value)
                semantic_total += 1
                semantic_accepted += int(_green_completes(semantic))
        self.assertEqual((raw_total, semantic_total), (11_110, 11_110))
        self.assertEqual(raw_accepted, 0)
        self.assertEqual(semantic_accepted, 0)

    def test_alternate_escape_spellings_all_refused(self) -> None:
        """Sec3.9.3's own executed figure: 0/30 -- raw, short-escape and \\uXXXX/\\uXXXX spellings
        of each of the ten insignificant bytes."""
        banned = tuple(sorted(b"{}[],: \t\r\n"))
        alternates: list[bytes] = []
        for byte_val in banned:
            if byte_val >= 0x20:
                alternates.append(bytes([byte_val]))
            if byte_val in (0x09, 0x0A, 0x0D):
                alternates.append(common.canonical_escape_spelling(byte_val))
            alternates.append(f"\\u{byte_val:04x}".encode("ascii"))
            alternates.append(f"\\u{byte_val:04X}".encode("ascii"))
        self.assertEqual(len(alternates), 30)
        accepted = sum(_green_completes(spelling) for spelling in alternates)
        self.assertEqual(accepted, 0)

    def test_all_length_invariant_from_the_monotone_state_product(self) -> None:
        """The all-length proof is structural (U has no close edge, M is absorbing and owns the
        close edge), not an extrapolation from the four-byte enumeration above -- checked
        directly against the compiled table's own U/M states."""
        mp, _vocab, single, _pre_value, u_state, m_state = _green_states()
        quote_id = single[0x22]
        self.assertNotIn(quote_id, mp.transitions.get(u_state, {}), "U must have no closing-quote edge")
        self.assertIn(quote_id, mp.transitions.get(m_state, {}), "M must have a closing-quote edge")

    def test_complete_semantic_utf8_language_counts_by_length(self) -> None:
        """Sec3.9.3's own executed closed-form counts: every Unicode scalar the JSON string
        grammar admits, at lengths 1-4, minus exactly the 10^n all-insignificant strings."""
        scalars_by_utf8_length = {1: 128, 2: 1_920, 3: 61_440, 4: 1_048_576}
        all_valid = {0: 1}
        expected_accepted = {1: 118, 2: 18_204, 3: 2_649_112, 4: 383_260_912}
        for length in range(1, 5):
            all_valid[length] = sum(
                scalars_by_utf8_length[width] * all_valid[length - width]
                for width in range(1, min(4, length) + 1)
            )
            accepted = all_valid[length] - 10 ** length
            self.assertEqual(accepted, expected_accepted[length], f"length {length} accepted count mismatch")


# ================================================================================================
# V1 -- TE-368 prompt 01, unchanged real prompt/harness.
# ================================================================================================


def _run_harness(artifact: Path, prompt_ids: Path, forced_ids: Path | None = None) -> dict:
    command = [str(_HARNESS), str(artifact), str(prompt_ids), "300", "1", "prompt_result"]
    if forced_ids is not None:
        command.append(str(forced_ids))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"harness exit={result.returncode}: {result.stdout[-300:]}")
    lines = result.stdout.splitlines()
    stop = next(line for line in lines if line.startswith("STOP=")).split()[0][5:]
    ids = [int(v) for v in next(line for line in lines if line.startswith("IDS="))[4:].split()]
    return {"stop": stop, "ids": ids}


def _decode_value(ids: list[int], forced_prefix: bool) -> tuple[str, bool]:
    tables = common.real_tokenizer_tables()
    specials = common.real_special_ids()
    raw = (_PREFIX + b'"' if forced_prefix else b"") + b"".join(tables.id_to_bytes[i] for i in ids)
    text = raw.decode("utf-8", errors="replace")
    try:
        obj = json.loads(text)
        value = obj["Prompt_Result"]
    except (json.JSONDecodeError, KeyError, TypeError):
        tail = text.split('"Prompt_Result":', 1)[-1]
        value = "<unparsed> " + (tail[1:] if tail.startswith('"') else tail)
    selected_special = any(i in specials for i in ids)
    return value, selected_special


def _build_red_artifact(out_path: Path) -> None:
    import struct
    import time

    sys.path.insert(0, str(_ENGINE / "tools"))
    import sslm_format as fmt
    from t2132_build_g5_fixture import _serialize_scm1, _real_vocab
    from sslm_convert_schema import compile_schema_to_mask_pages, zero_special_ids

    config = fmt.read_section_bytes(str(common.A_EX_ARTIFACT), fmt.SectionType.CONFIG)
    (vocab_size,) = struct.unpack_from("<I", config, 32)
    vocab = _real_vocab(str(common.QWEN25_0P5B_CHECKPOINT), vocab_size)
    # T-2910 (Sec3.9.1): a caller compiling a string-leaf schema must zero every tokenizer
    # special/added id before the vocabulary reaches the compiler -- `_real_vocab` itself does
    # not (correctly: `t2132_build_g5_fixture.py`'s own two schemas have no string leaf, so
    # zeroing there would be a no-op; this schema, `_SCHEMA` (Prompt_Result), DOES have one).
    # Found live: without this step, `_build_red_artifact`'s own output could select a special
    # id as string content on the real engine, diverging from the properly-zeroed GREEN
    # reference on some heldout prompts (V5, below) for a reason unrelated to the compiler port.
    vocab = zero_special_ids(vocab, common.real_special_ids())
    mask_pages = compile_schema_to_mask_pages(_SCHEMA, vocab)
    scm1 = _serialize_scm1([("prompt_result", mask_pages)], vocab_size)
    header = fmt.read_header(str(common.A_EX_ARTIFACT))
    sections = []
    for section in header["sections"]:
        if section["type"] == fmt.SectionType.SCHEMA_MASKS:
            continue
        data = header["data"][section["offset"]:section["offset"] + section["byte_size"]]
        sections.append(fmt.Section(
            type=section["type"], data=bytes(data), dtype=section["dtype"],
            elem_count=section["elem_count"], alignment=section["alignment"],
        ))
    sections.append(fmt.Section(type=fmt.SectionType.SCHEMA_MASKS, data=scm1))
    fmt.write_artifact(str(out_path), sections, flags=header["flags"])


class T2912_V1_TE368Prompt01(unittest.TestCase):
    def test_branch_artifact_answers_te368_prompt_01_on_the_real_engine(self) -> None:
        """T-2915 (the compiler port landed): executed fresh against real hardware and the real
        A-EX model, the branch's own artifact (built by the SAME production compiler and
        vocabulary producer T-2915 ported) now answers TE-368 prompt 01, matching the T-2912
        reference exactly -- confirmed by execution, not assumed from the port's own static
        agreement with the reference (V2-V4, above)."""
        _require_engine_artifacts()
        te368_prompts_path = common.RECORDS_ROOT / "Claude" / "Loki" / "te368-probe" / "te368_prompts.json"
        prompts = json.loads(te368_prompts_path.read_bytes().decode("utf-8-sig"))
        prompt_text = prompts[1]
        self.assertEqual(prompt_text, "Return the player's inventory as a JSON object.")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(common.QWEN25_0P5B_CHECKPOINT), local_files_only=True)
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids_path = _SCRATCH / "v1_prompt_01.ids"
        _SCRATCH.mkdir(parents=True, exist_ok=True)
        prompt_ids_path.write_text(" ".join(map(str, tokenizer(chat_text).input_ids)), encoding="ascii")

        red_artifact = _SCRATCH / "v1_red.sslm"
        _build_red_artifact(red_artifact)

        red_run = _run_harness(red_artifact, prompt_ids_path)
        red_value, red_special = _decode_value(red_run["ids"], forced_prefix=False)
        oracle = common.reference_oracle()
        self.assertTrue(
            oracle.is_answer(red_value, red_special),
            f"the branch's own artifact must answer TE-368 prompt 01; got {red_value!r}",
        )

        green_run = _run_harness(_GREEN_ARTIFACT, prompt_ids_path)
        green_value, green_special = _decode_value(green_run["ids"], forced_prefix=False)
        self.assertTrue(
            oracle.is_answer(green_value, green_special),
            f"the T-2912 reference artifact must answer TE-368 prompt 01; got {green_value!r}",
        )

    def test_dropped_value_history_mutant_reproduces_the_bracket_defect_via_real_token_1183(self) -> None:
        """Sec3.9.3's own named mutant: dropping the value-history close guard admits real
        token 1183 (b'[\"') at U, and completes the two-token split path b'[' then b'\"' -- both
        refused by the fixed compiler."""
        fixed = common.reference_t2912()
        vocab = common.real_byte_vocab_zeroed()
        self.assertEqual(vocab[1183], b'["', "SETUP: the real vocabulary's own token 1183 changed")

        fixed_mp, _fixed_vocab, fixed_single, _pre, fixed_u, _m = _green_states()
        self.assertNotIn(1183, fixed_mp.transitions.get(fixed_u, {}), "the fix must refuse real token 1183 at U")
        left_bracket_id = fixed_single[ord("[")]
        state = fixed_mp.transitions.get(fixed_u, {}).get(left_bracket_id)
        state = fixed_mp.transitions.get(state, {}).get(fixed_single[0x22]) if state is not None else None
        self.assertIsNone(state, "the fix must refuse the split '[' then '\"' path from U")

        original_leaf = fixed._BASE._add_string_leaf

        def leaf_without_value_guard(edges, add_node, cursor, content_nodes):
            following = original_leaf(edges, add_node, cursor, content_nodes)
            u_state = next(iter(edges[cursor][0x22]))
            meaningful = next(iter(edges[u_state][ord("A")]))
            edges[u_state].setdefault(0x22, set()).add(following)
            return following

        fixed._BASE._add_string_leaf = leaf_without_value_guard
        try:
            mutant_mp = fixed.compile_schema_to_mask_pages(_SCHEMA, vocab)
        finally:
            fixed._BASE._add_string_leaf = original_leaf

        mutant_pre = common.walk_literal(mutant_mp, vocab, mutant_mp.start, _PREFIX)
        mutant_u = mutant_mp.transitions[mutant_pre][fixed_single[0x22]]
        self.assertIn(1183, mutant_mp.transitions[mutant_u], "the mutant must admit real token 1183 at U")
        state = mutant_mp.transitions[mutant_u][left_bracket_id]
        state = mutant_mp.transitions[state][fixed_single[0x22]]
        state = mutant_mp.transitions[state][fixed_single[0x7D]]
        self.assertIn(state, mutant_mp.accepting, "the mutant must complete the split '[' then '\"' path")


# ================================================================================================
# V5 -- the held-out real census. T-2912's own pre-registered, hashed prompts and its
# commissioned oracle, unaltered.
# ================================================================================================


def _run_population(artifact: Path, prompt_ids: list[Path], forced_ids: Path | None, workers: int = 4):
    def one(prompt_path: Path):
        return _run_harness(artifact, prompt_path, forced_ids)

    results: list[dict] = [None] * len(prompt_ids)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, path): index for index, path in enumerate(prompt_ids)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


class T2912_V5_HeldoutCensus(unittest.TestCase):
    def test_heldout_census_free_vs_forced_control(self) -> None:
        _require_engine_artifacts()
        source = common.heldout_prompts_path()
        source_bytes = source.read_bytes()
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        expected_sha256 = "44ffc54de4b62d5ff5f9ff84044b93947dc38bf316489b5e140ed8002e27fa76"
        self.assertEqual(actual_sha256, expected_sha256, "the heldout prompts file must not be altered")
        rows = json.loads(source_bytes)
        self.assertEqual(len(rows), 40)
        self.assertGreaterEqual(sum(row["stratum"] == "structured" for row in rows), 20)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(common.QWEN25_0P5B_CHECKPOINT), local_files_only=True)
        ids_dir = _SCRATCH / "v5_heldout"
        ids_dir.mkdir(parents=True, exist_ok=True)
        prompt_paths = []
        for index, row in enumerate(rows):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            path = ids_dir / f"prompt_{index:02d}.ids"
            path.write_text(" ".join(map(str, tokenizer(text).input_ids)), encoding="ascii")
            prompt_paths.append(path)

        red_artifact = _SCRATCH / "v5_red.sslm"
        _build_red_artifact(red_artifact)

        oracle = common.reference_oracle()

        def grade(run: dict, forced: bool) -> bool:
            value, special = _decode_value(run["ids"], forced_prefix=forced)
            return oracle.is_answer(value, special)

        red_free = _run_population(red_artifact, prompt_paths, None)
        green_free = _run_population(_GREEN_ARTIFACT, prompt_paths, None)
        green_control = _run_population(_GREEN_ARTIFACT, prompt_paths, _CANONICAL_CONTROL_IDS)

        red_answers = sum(grade(run, False) for run in red_free)
        green_free_answers = sum(grade(run, False) for run in green_free)
        green_control_answers = sum(grade(run, True) for run in green_control)

        self.assertEqual(
            red_answers, 40,
            "the branch's own artifact must answer all 40 heldout prompts, matching the "
            "T-2912 reference (T-2915: the compiler port landed)",
        )
        self.assertEqual(green_free_answers, 40, "the T-2912 reference must answer all 40 heldout prompts")
        self.assertEqual(green_control_answers, 40, "the forced-canonical control must answer all 40 prompts")

        free_bad = {i for i, run in enumerate(green_free) if not grade(run, False)}
        control_bad = {i for i, run in enumerate(green_control) if not grade(run, True)}
        self.assertEqual(len(free_bad - control_bad), 0, "grammar-induced nonanswers must be 0 of 40")
        self.assertEqual(control_bad - free_bad, set())


# ================================================================================================
# Oracle re-confirmation -- must read COMMISSIONED and RECONFIRMED before its heldout reading (V5)
# is load-bearing.
# ================================================================================================


def _pinned_json(relative_path: str) -> object:
    """Read a records-tree JSON file AS IT STOOD at this plan's own pinned commit
    (`2fe6a7299b`, the brief's own "at records commit" reference) rather than the live working
    tree, which a concurrent session is actively editing (confirmed: this exact records worktree
    shows uncommitted edits to `t2912_oracle_must_reject.json` from an in-flight T-2914 fold
    responding to a later adversary strike, TE-369, on this same design -- out of this ticket's
    own scope, per the brief's pinned commit and its "stay out of concurrent work" instruction)."""
    result = subprocess.run(
        ["git", "show", f"2fe6a7299b:Claude/Vitruvius/t2912-probe/{relative_path}"],
        cwd=common.RECORDS_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    return json.loads(result.stdout)


class T2912_AnswerOracleReconfirmation(unittest.TestCase):
    def test_must_accept_and_must_reject_fixtures(self) -> None:
        oracle = common.reference_oracle()
        must_accept = _pinned_json("t2912_oracle_must_accept.json")
        must_reject = _pinned_json("t2912_oracle_must_reject.json")
        for row in must_accept:
            self.assertTrue(
                oracle.is_answer(row["value"], bool(row.get("special_selected", False))),
                f"must-accept fixture wrongly rejected: {row!r}",
            )
        for row in must_reject:
            self.assertFalse(
                oracle.is_answer(row["value"], bool(row.get("special_selected", False))),
                f"must-reject fixture wrongly accepted: {row!r}",
            )


if __name__ == "__main__":
    unittest.main()
