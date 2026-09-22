"""T-2913 shared helpers -- the Sec3.9.3 string-leaf red suite (T-2908/T-2910/T-2912's final
design, plan `Claude/Plans/te266-gpu-path.md`, Wizard repo).

Every cell in this suite is checked against TWO compilers:
  - RED: `tools.sslm_convert_schema`, this branch's own in-tree module -- T-2853's reviewed,
    pre-fold, character-level design (confirmed at authoring time: no byte-level alphabet, no
    value-open/value-close boundary discipline, no special-token exclusion, no U/M value-level
    closure, and only `maxLength` named in its rejection surface). T-2915 folded T-2908/T-2910/
    T-2912's design into this module, so RED now implements the same design GREEN does; see
    `reference/PROVENANCE.md` and `t2899_string_leaf_red.py`'s own docstring for what that
    means for a test that used to compare the two as though GREEN were independent ground
    truth.
  - GREEN: the T-2912 reference compiler chain, loaded from this repo's own vendored copy
    (`reference/t2912-probe/`, `reference/PROVENANCE.md`) rather than an out-of-repo records
    worktree (T-2919, TE-372 S3: the suite must be self-contained and reproducible from a
    fresh clone). This module never writes to the vendored copy.

Real vocabulary: the real Qwen2.5-0.5B-Instruct checkpoint (the A-EX artifact's own tokenizer),
loaded through this repo's own `tools/convert_tokenizer.py` (read, not modified) -- the same
`TokenizerTables` class T-2908/T-2910/T-2912's own probes used. Real-vocabulary cells are gated
on `SUPERSLM_G5_REAL_MODEL_TESTS` (T-2919, TE-372 S3): unset, the cell is skipped (a bare CI
runner never has these checkpoints); set but the checkpoint or artifact is not present on this
box, the cell fails loudly rather than skipping silently (T-2909's fail-closed rule, preserved
for the opted-in case).
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import struct
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# --- this repo's own tools/, read-only imports -------------------------------------------------

_ENGINE_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _ENGINE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# --- vendored reference material, in this repo (T-2919, TE-372 S3) -----------------------------
#
# Every reference compiler, the commissioned answer oracle, and the pre-registered prompt sets
# this suite grades against are copied into `reference/` alongside this file, with their
# provenance recorded in `reference/PROVENANCE.md` -- resolved relative to this file's own
# location, never against a private, per-session records worktree.

_REFERENCE_ROOT = Path(__file__).resolve().parent / "reference"
_T2912_REF_DIR = _REFERENCE_ROOT / "t2912-probe"

T2908_PROBE_PATH = _REFERENCE_ROOT / "t2908-probe" / "sslm_convert_schema_bytelevel.py"
T2910_PROBE_PATH = _REFERENCE_ROOT / "t2910-probe" / "sslm_convert_schema_bytelevel_boundary.py"
T2911_PROBE_PATH = _REFERENCE_ROOT / "t2911-probe" / "sslm_convert_schema_close_structural.py"
T2912_PROBE_PATH = _T2912_REF_DIR / "sslm_convert_schema_value_level.py"
T2912_ORACLE_PATH = _T2912_REF_DIR / "t2912_answer_oracle.py"
T2912_HELDOUT_PATH = _T2912_REF_DIR / "t2912_heldout_prompts.json"
T2912_MUST_ACCEPT_PATH = _T2912_REF_DIR / "t2912_oracle_must_accept.json"
T2912_MUST_REJECT_PATH = _T2912_REF_DIR / "t2912_oracle_must_reject.json"
T2912_CANONICAL_CONTROL_PATH = _T2912_REF_DIR / "t2912_canonical_control.ids"
TE368_PROMPTS_PATH = _REFERENCE_ROOT / "te368-probe" / "te368_prompts.json"
TE368_HARNESS_SOURCE = _REFERENCE_ROOT / "te368-probe" / "te366_schema_run.cpp"
TE368_HARNESS_BUILD_SCRIPT = _REFERENCE_ROOT / "te368-probe" / "build_te368_harness.bat"

# Historical mutation controls are frozen files, not Git-object lookups.  The required CTest
# target must run unchanged from a depth-one clone and from a Git-free source archive.
_HISTORICAL_ROOT = _REFERENCE_ROOT / "historical"
_FROZEN_FIXTURES: dict[tuple[str, str], Path] = {
    ("v1.5.0", "tools/sslm_convert_schema.py"): _HISTORICAL_ROOT / "v1_5_0_sslm_convert_schema.py",
    ("0062c99", "tools/sslm_convert_schema.py"): _HISTORICAL_ROOT / "0062c99_sslm_convert_schema.py",
    ("0062c99", "tools/t2132_build_g5_fixture.py"): _HISTORICAL_ROOT / "0062c99_t2132_build_g5_fixture.py",
    ("60eb357", "tools/sslm_convert_schema.py"): _HISTORICAL_ROOT / "60eb357_sslm_convert_schema.py",
}
# These pin the complete vendored fixture (provenance header plus exact historical source).
_FROZEN_FIXTURE_SHA256: dict[tuple[str, str], str] = {
    ("v1.5.0", "tools/sslm_convert_schema.py"): "819c44d325b6fe67cda636d175d59b736702e1c1fab00f02913df65b1f885214",
    ("0062c99", "tools/sslm_convert_schema.py"): "f498aa7a5a55c3e84a1a724531328c2e732aad296400a4e0c52248e966930cf5",
    ("0062c99", "tools/t2132_build_g5_fixture.py"): "a4d0044e81c58c5e431de4a9a2ab8a22fbe58334b66fbc2d8ed7df9550c68b37",
    ("60eb357", "tools/sslm_convert_schema.py"): "72a4294e3bbf805dcdf1ae9339df776e6ad0bdb00212f4804e770c5aa487cc18",
}
# Raw historical source hashes, recorded in each fixture header, identify whether HEAD still
# equals the named bad version without asking Git for history.
_FROZEN_SOURCE_SHA256: dict[tuple[str, str], str] = {
    ("v1.5.0", "tools/sslm_convert_schema.py"): "a8e9f9940dce2c9668aa05db6ea3c0ce63dec0c22e579f214cc70bba17252f2f",
    ("0062c99", "tools/sslm_convert_schema.py"): "4433aaa39ae1758476899d312bb51a109c37fdf759761a567061f6ecf313fad2",
    ("0062c99", "tools/t2132_build_g5_fixture.py"): "423790e2ec1a0eb81729d963b34cb982888789157d3634a2b80cf9d82098631b",
    ("60eb357", "tools/sslm_convert_schema.py"): "d6a14c9c70e2356ef0395a5a437f1f6499f3d8d2cff255ac0d57712776a48b5b",
}

# The commissioned oracle and the heldout prompts are never altered (the brief's hard rule,
# T-2912's own design); these pins guard the VENDORED copy exactly as they guarded the
# records-tree original, so a drift is caught the same way whichever copy is live.
T2912_ORACLE_SHA256 = "5dcdbebc212cba5c11cd00fa27edfd72f150d8ca69b2932044e260cc2e1fb97e"
T2912_HELDOUT_SHA256 = "44ffc54de4b62d5ff5f9ff84044b93947dc38bf316489b5e140ed8002e27fa76"

PROMPT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"Prompt_Result": {"type": "string"}},
    "required": ["Prompt_Result"],
}

# --- real-model artifact gate (T-2919, TE-372 S3) -----------------------------------------------

REAL_MODEL_ENV = "SUPERSLM_G5_REAL_MODEL_TESTS"

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
    """Fails loudly if `path` (in-repo, vendored content this suite always ships) is missing.
    Never used for a real-model artifact -- see `_require_real_model` below for those."""
    if not path.exists():
        raise FileNotFoundError(
            f"this suite needs {path}, which is not present -- it should be vendored in-repo "
            "(reference/PROVENANCE.md); this is not a real-model artifact and is never gated "
            "on SUPERSLM_G5_REAL_MODEL_TESTS"
        )
    return path


def _require_real_model(path: Path) -> Path:
    """Gates a real-model artifact (a cached HF checkpoint, the shipped A-EX .sslm artifact, or
    the TE-368 harness binary) on `SUPERSLM_G5_REAL_MODEL_TESTS` (T-2919, TE-372 S3).

    Unset: the calling cell is skipped -- none of these artifacts is provisioned on a bare CI
    runner, so real-model coverage is opt-in there. Set (any non-empty value) but `path` is
    missing: fails loudly, never skips -- a box that opted in and then can't find the artifact
    has a real setup gap, not an absent feature (T-2909's fail-closed rule, preserved for the
    opted-in case)."""
    if not os.environ.get(REAL_MODEL_ENV):
        pytest.skip(
            f"{REAL_MODEL_ENV} is not set -- real-model cell opted out (would need {path})"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"{REAL_MODEL_ENV} is set but {path} is missing -- opting into real-model tests "
            "requires the artifact actually be provisioned on this box"
        )
    return path


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise AssertionError(
            f"{label} at {path} has drifted: sha256={actual}, expected {expected} -- refusing "
            "to grade against an unverified copy"
        )


def frozen_fixture_inventory() -> dict[tuple[str, str], tuple[Path, str]]:
    """The integrity-test inventory for all historical inputs this suite loads."""
    return {
        key: (path, _FROZEN_FIXTURE_SHA256[key]) for key, path in _FROZEN_FIXTURES.items()
    }


def _frozen_source(commit_ish: str, repo_relative_path: str) -> tuple[Path, str]:
    key = (commit_ish, repo_relative_path)
    try:
        path = _FROZEN_FIXTURES[key]
    except KeyError as exc:
        raise ValueError(
            f"no frozen fixture registered for {commit_ish}:{repo_relative_path}; "
            "add and hash-pin it before making it a suite input"
        ) from exc
    _require(path)
    _verify_sha256(path, _FROZEN_FIXTURE_SHA256[key], f"frozen fixture {commit_ish}:{repo_relative_path}")
    return path, _FROZEN_SOURCE_SHA256[key]


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
    value-level close product. Loaded from its own vendored, filed location
    (`reference/t2912-probe/`) so no path patching is needed after loading it -- it chain-loads
    `../t2911-probe/sslm_convert_schema_close_structural.py` as its own base module. T-2915
    ported this same design into `tools/sslm_convert_schema.py`; a cell that treats agreement
    with this module as its ONLY correctness evidence is comparing the shipped compiler to a
    copy of its own design, not to an independent oracle -- see `reference/PROVENANCE.md`."""
    return _load_module("_t2913_ref_t2912", T2912_PROBE_PATH)


@functools.lru_cache(maxsize=1)
def reference_oracle():
    """`T2912-answer-value-oracle`, COMMISSIONED and RECONFIRMED (Wizard repo,
    `Claude/Vitruvius/t2912-probe/attestations/T2912-answer-value-oracle.json`). Loaded
    unaltered from its own vendored, hash-pinned copy (`reference/t2912-probe/`) -- the brief's
    hard rule ("do not alter either")."""
    _verify_sha256(T2912_ORACLE_PATH, T2912_ORACLE_SHA256, "the commissioned T2912-answer-value-oracle")
    return _load_module("_t2913_ref_oracle", T2912_ORACLE_PATH)


def heldout_prompts_path() -> Path:
    _require(T2912_HELDOUT_PATH)
    _verify_sha256(T2912_HELDOUT_PATH, T2912_HELDOUT_SHA256, "the pre-registered heldout prompts")
    return T2912_HELDOUT_PATH


def oracle_must_accept_path() -> Path:
    """T-2912's must-accept fixture, vendored byte-identical (`reference/PROVENANCE.md`)."""
    return _require(T2912_MUST_ACCEPT_PATH)


def oracle_must_reject_path() -> Path:
    """T-2912's must-reject fixture, vendored at its pinned commit `2fe6a7299b` (5 rows) --
    deliberately NOT the live records-tree file, which carries 4 additional rows from an
    in-flight, out-of-scope fold (`reference/PROVENANCE.md`)."""
    return _require(T2912_MUST_REJECT_PATH)


@functools.lru_cache(maxsize=1)
def real_vocab_size() -> int:
    import sslm_format as fmt

    config = fmt.read_section_bytes(str(_require_real_model(A_EX_ARTIFACT)), fmt.SectionType.CONFIG)
    (vocab_size,) = struct.unpack_from("<I", config, 32)
    return vocab_size


@functools.lru_cache(maxsize=1)
def real_tokenizer_tables():
    from convert_tokenizer import TokenizerTables

    return TokenizerTables(str(_require_real_model(QWEN25_0P5B_CHECKPOINT)))


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


@functools.lru_cache(maxsize=1)
def reference_v150():
    """The `v1.5.0` `tools/sslm_convert_schema.py` frozen fixture. It is a committed historical
    input with provenance and a content pin in-tree, so this suite has no repository dependency.

    v1.5.0 predates T-2908's byte-level port (`compile_schema_to_mask_pages` there takes
    `Sequence[str]`, not `Sequence[bytes]`) and predates the S1 regression entirely -- it is the
    GREEN oracle for T-2916's S1 cells (`{"type":"string","enum":[...]}`, annotation keywords),
    which 1.5.0 always compiled and 0062c99 wrongly rejects."""
    path, _ = _frozen_source("v1.5.0", "tools/sslm_convert_schema.py")
    source = path.read_text(encoding="utf-8")
    if not source.strip():
        raise RuntimeError("frozen v1.5.0 tools/sslm_convert_schema.py fixture is empty")
    module = types.ModuleType("_t2916_ref_v150")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


# --- T-2916's generic pre-fix/post-fix mutant harness -------------------------------------------
#
# C1/S1/S2 pin a fix that T-2917 (the builder) lands AFTER this suite is authored (red-first).
# Rather than inventing a candidate implementation of that fix to mutate (which would let this
# suite's own guess of the mechanism substitute for the design), the "wrong version" required by
# the mutation-proof discipline (StandardsDocument.md Sec5.4, Curie's "pin the documented claim")
# is simply named directly: the shipped module AS IT STANDS AT 0062c99, the commit TE-370 reviewed
# and ruled DO NOT SHIP over exactly these findings. `frozen_module` loads any tracked file from
# its named frozen fixture. `fix_has_landed` tells a cell whether HEAD still equals that known-bad
# commit for the file in question, so a cell run before T-2917 lands reports "pending", never a
# fabricated pass or a silent skip. T-2919 (TE-372 S2) reuses the identical pattern, pinned to
# `60eb357` instead (the commit TE-372 reviewed and ruled DO NOT SHIP over the nullable-enum
# TypeError, among others).


@functools.lru_cache(maxsize=None)
def frozen_module(commit_ish: str, repo_relative_path: str):
    """Load the named historical module from its hash-pinned in-tree fixture."""
    path, _ = _frozen_source(commit_ish, repo_relative_path)
    source = path.read_text(encoding="utf-8")
    if not source.strip():
        raise RuntimeError(f"frozen fixture {commit_ish}:{repo_relative_path} is empty")
    module = types.ModuleType(f"_t2916_frozen_{commit_ish}_{repo_relative_path.replace('/', '_')}")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def fix_has_landed(commit_ish: str, repo_relative_path: str) -> bool:
    """True once the shipped file at HEAD differs from its content at `commit_ish` -- i.e. once a
    fix has actually landed on top of the known-bad commit. False means the mutant proof below is
    not yet meaningful (there is no delta to discriminate) and must report PENDING, not PASS."""
    _, frozen_source_hash = _frozen_source(commit_ish, repo_relative_path)
    shipped = (_ENGINE_ROOT / repo_relative_path).read_bytes()
    return hashlib.sha256(shipped).hexdigest() != frozen_source_hash


def single_byte_token_ids(vocab: list[bytes]) -> dict[int, int]:
    """Map raw byte value -> lowest token id spelling that single byte, for a bytes-vocab."""
    table: dict[int, int] = {}
    for token_id, piece in enumerate(vocab):
        if len(piece) == 1 and piece[0] not in table:
            table[piece[0]] = token_id
    return table


@functools.lru_cache(maxsize=1)
def real_red_mask_pages():
    """T-2917 (folding TE-370 C1, D-SLM7600): this branch's own compiler over the real
    vocabulary, RAW bytes, with `special_ids=frozenset()` -- explicitly asking for no
    exclusion, the only way left to reach that state now that `special_ids` is a required
    keyword-only argument the compiler applies itself (a caller can no longer reach an
    admitting compile by omitting the argument silently). Cached: real-vocabulary compiles are
    re-used across every cell in this suite that needs the SAME compiled table rather than
    recompiled per cell."""
    from tools.sslm_convert_schema import compile_schema_to_mask_pages

    return compile_schema_to_mask_pages(PROMPT_RESULT_SCHEMA, real_raw_vocab(), special_ids=frozenset())


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
