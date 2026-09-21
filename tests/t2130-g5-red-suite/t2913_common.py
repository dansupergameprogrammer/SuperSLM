"""T-2913 shared helpers -- the Sec3.9.3 string-leaf red suite (T-2908/T-2910/T-2912's final
design, plan `Claude/Plans/te266-gpu-path.md`, Wizard repo).

Every cell in this suite is checked against TWO compilers:
  - RED: `tools.sslm_convert_schema`, this branch's own in-tree module -- T-2853's reviewed,
    pre-fold, character-level design (confirmed at authoring time: no byte-level alphabet, no
    value-open/value-close boundary discipline, no special-token exclusion, no U/M value-level
    closure, and only `maxLength` named in its rejection surface).
  - GREEN: the T-2912 reference compiler chain, loaded directly from its own filed location in
    the records tree (`Claude/Vitruvius/t2908-probe/` .. `t2912-probe/`) rather than copied --
    the records tree is the artifact this suite is graded against, and importing it by path
    means there is exactly one copy of the reference to drift from. This module never writes to
    that tree.

Real vocabulary: the real Qwen2.5-0.5B-Instruct checkpoint (the A-EX artifact's own tokenizer),
loaded through this repo's own `tools/convert_tokenizer.py` (read, not modified) -- the same
`TokenizerTables` class T-2908/T-2910/T-2912's own probes used. Real-vocabulary cells fail loudly
(FileNotFoundError) rather than skip when the checkpoint or artifact is not present on this box.
"""
from __future__ import annotations

import functools
import importlib.util
import struct
import sys
from pathlib import Path
from typing import Any

# --- this repo's own tools/, read-only imports -------------------------------------------------

_ENGINE_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _ENGINE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# --- the records tree, where the reference compiler chain and T-2912's pinned instruments live --

RECORDS_ROOT = Path("D:/Wizard/.claude/worktrees/superslm-super-embedder-fixes-c20ddf")
_VITRUVIUS = RECORDS_ROOT / "Claude" / "Vitruvius"

T2908_PROBE_PATH = _VITRUVIUS / "t2908-probe" / "sslm_convert_schema_bytelevel.py"
T2910_PROBE_PATH = _VITRUVIUS / "t2910-probe" / "sslm_convert_schema_bytelevel_boundary.py"
T2911_PROBE_PATH = _VITRUVIUS / "t2911-probe" / "sslm_convert_schema_close_structural.py"
T2912_PROBE_PATH = _VITRUVIUS / "t2912-probe" / "sslm_convert_schema_value_level.py"
T2912_ORACLE_PATH = _VITRUVIUS / "t2912-probe" / "t2912_answer_oracle.py"
T2912_HELDOUT_PATH = _VITRUVIUS / "t2912-probe" / "t2912_heldout_prompts.json"

PROMPT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"Prompt_Result": {"type": "string"}},
    "required": ["Prompt_Result"],
}

# A-EX's real checkpoint and shipped artifact (Qwen2.5-0.5B-Instruct, context_cap 4096).
QWEN25_0P5B_CHECKPOINT = Path(
    "D:/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
A_EX_ARTIFACT = Path("D:/hf_cache/superslm_artifacts/example/qwen2.5-0.5b-instruct-cap4096-aex.sslm")

# T-2908's own reachability probe used this 1.5B checkpoint (`run_s2_followup.py`); reused here
# unchanged so the reachability cell is grounded in the plan's own executed construction rather
# than a fresh, undiscussed choice of samples/tokenizer.
SHOPKEEPER_LORA_CHECKPOINT = Path(
    "D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-shopkeeper-lora-v2-merged"
)

# TE-365's own seven natural-tokenization samples (T-2908 Sec3.9.1's own executed proof; reused
# verbatim from `Claude/Vitruvius/t2908-probe/run_s2_followup.py`, itself reusing
# `Claude/Poirot/te365-probe/p4_scripts.py`'s dictionary unchanged).
TE365_NATURAL_SAMPLES: dict[str, str] = {
    "hindi": "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "thai": "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a",
    "flag": "\U0001F1EF\U0001F1F5",
    "korean": "\uc548\ub155\ud558\uc138\uc694",
    "rare_cjk": "\u9f98\u9f49\u9ea4",
    "emoji_zwj": "\U0001F468\u200d\U0001F469\u200d\U0001F467",
    "math": "\u2200x \u2203y \u2261 \u221e",
}
# T-2908's own five regression samples (Sec3.9.1: "curly quotes, an apostrophe, an em-dash, a
# simple emoji, an accented Latin word").
TE365_REGRESSION_SAMPLES: dict[str, str] = {
    "curly_quotes": "\u201cquoted\u201d",
    "apostrophe": "don\u2019t",
    "em_dash": "wait\u2014really",
    "simple_emoji": "\U0001F600",
    "accented_latin": "caf\u00e9",
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"real-model cell needs {path}, which is not present on this box -- this cell "
            "fails loudly rather than skipping silently (T-2909's fail-closed rule)"
        )
    return path


def _load_module(name: str, path: Path):
    _require(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def reference_t2908():
    """T-2908's byte-level redesign alone (UTF-8-valid content automaton, no boundary/value
    discipline yet) -- the GREEN oracle for the structural UTF-8-validity and lone-continuation-
    byte cells, which are about the alphabet, not the boundary or value-level rules."""
    return _load_module("_t2913_ref_t2908", T2908_PROBE_PATH)


@functools.lru_cache(maxsize=1)
def reference_t2910():
    """T-2908 + the open/close boundary discipline + special-token exclusion. The GREEN oracle
    for the value-open, value-close and special-token-exclusion cells."""
    return _load_module("_t2913_ref_t2910", T2910_PROBE_PATH)


@functools.lru_cache(maxsize=1)
def reference_t2911():
    """Historical only (T-2912 deletes this module's own token-local close blacklist as
    redundant -- Sec3.9.3's own V3/V4 proof). Loaded only by the one cell that documents the
    supersession by construction, never used as a GREEN oracle for a live cell."""
    return _load_module("_t2913_ref_t2911", T2911_PROBE_PATH)


@functools.lru_cache(maxsize=1)
def reference_t2912():
    """The shipping design: T-2910's open side and special-token exclusion, plus the U/M
    value-level close product. THE reference compiler this suite's Proof section names. Its own
    `_BASE_PATH` resolves relative to its real, filed location in the records tree, so no path
    patching is needed after loading it from there."""
    return _load_module("_t2913_ref_t2912", T2912_PROBE_PATH)


@functools.lru_cache(maxsize=1)
def reference_oracle():
    """`T2912-answer-value-oracle`, COMMISSIONED and RECONFIRMED (Claude/Vitruvius/t2912-probe/
    attestations/T2912-answer-value-oracle.json). Loaded unaltered from its own filed path --
    the brief's hard rule ("do not alter either")."""
    return _load_module("_t2913_ref_oracle", T2912_ORACLE_PATH)


def heldout_prompts_path() -> Path:
    return _require(T2912_HELDOUT_PATH)


@functools.lru_cache(maxsize=1)
def real_vocab_size() -> int:
    import sslm_format as fmt

    config = fmt.read_section_bytes(str(_require(A_EX_ARTIFACT)), fmt.SectionType.CONFIG)
    (vocab_size,) = struct.unpack_from("<I", config, 32)
    return vocab_size


@functools.lru_cache(maxsize=1)
def real_tokenizer_tables():
    from convert_tokenizer import TokenizerTables

    return TokenizerTables(str(_require(QWEN25_0P5B_CHECKPOINT)))


@functools.lru_cache(maxsize=1)
def real_special_ids() -> frozenset[int]:
    tables = real_tokenizer_tables()
    return frozenset(token_id for _, token_id in tables.specials)


@functools.lru_cache(maxsize=1)
def real_raw_vocab() -> tuple[bytes, ...]:
    """The real vocabulary as `Sequence[bytes]` -- undecoded, the alphabet T-2908+ compile
    against. Special ids are NOT zeroed here; callers that need the shipping vocabulary call
    `real_byte_vocab_zeroed()`."""
    tables = real_tokenizer_tables()
    size = real_vocab_size()
    return tuple(tables.id_to_bytes[i] if i < len(tables.id_to_bytes) else b"" for i in range(size))


@functools.lru_cache(maxsize=1)
def real_byte_vocab_zeroed() -> list[bytes]:
    """The real vocabulary as the T-2910+ compilers actually consume it: raw bytes, all 22
    tokenizer special ids zeroed to an empty piece before the trie is built."""
    ref = reference_t2910()
    return ref.zero_special_ids(list(real_raw_vocab()), real_special_ids())


def single_byte_token_ids(vocab: list[bytes]) -> dict[int, int]:
    """Map raw byte value -> lowest token id spelling that single byte, for a bytes-vocab."""
    table: dict[int, int] = {}
    for token_id, piece in enumerate(vocab):
        if len(piece) == 1 and piece[0] not in table:
            table[piece[0]] = token_id
    return table


@functools.lru_cache(maxsize=1)
def real_red_mask_pages():
    """T-2915 (the compiler port landed): this branch's own compiler over the real vocabulary,
    RAW bytes, unzeroed -- exactly what `tools/t2132_build_g5_fixture.py::_real_vocab` (also
    ported alongside the compiler) actually returns when a caller does not additionally call
    `zero_special_ids`. Cached: real-vocabulary compiles are re-used across every cell in this
    suite that needs the SAME compiled table rather than recompiled per cell."""
    from tools.sslm_convert_schema import compile_schema_to_mask_pages

    return compile_schema_to_mask_pages(PROMPT_RESULT_SCHEMA, real_raw_vocab())


@functools.lru_cache(maxsize=1)
def real_green_mask_pages():
    """The T-2912 reference compiler over the real vocabulary, raw bytes, specials zeroed --
    the exact construction `Claude/Vitruvius/t2912-probe/t2912_build_artifact.py` used."""
    ref = reference_t2912()
    return ref.compile_schema_to_mask_pages(PROMPT_RESULT_SCHEMA, real_byte_vocab_zeroed())




def canonical_escape_spelling(byte_val: int) -> bytes:
    """The canonical JSON spelling of one of the three whitespace bytes JSON requires an escape
    or a raw byte to spell (a raw literal C0 control byte is never legal JSON content in any
    design); every other byte is spelled as its own raw literal."""
    if byte_val == 0x09:
        return rb"\t"
    if byte_val == 0x0A:
        return rb"\n"
    if byte_val == 0x0D:
        return rb"\r"
    return bytes([byte_val])


def walk_literal(mp, vocab: Any, start_state: int, remaining: Any):
    """Greedy longest-match walk of `remaining` (bytes or str) through `mp.transitions`,
    starting at `start_state`. `vocab` is the same sequence passed to the compiler (so
    `vocab[token_id]` and `remaining` are the same type). Returns the reached state."""
    state = start_state
    while remaining:
        row = mp.transitions[state]
        token_id = max(
            (tid for tid in row if remaining[: len(vocab[tid])] == vocab[tid] and vocab[tid]),
            key=lambda tid: len(vocab[tid]),
        )
        state = row[token_id]
        remaining = remaining[len(vocab[token_id]):]
    return state
