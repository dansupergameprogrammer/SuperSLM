"""Shared harness for `tools/reference_pipeline/`'s ported test suite (T-2123/T-2137).

Ported, with its module-name strings rewritten from `superslm_spike.*` to
`reference_pipeline.*`, from the T-066 spike test suite this design's own §6 B0
build item vendors (`Claude/Vitruvius/t2123-converter-vendoring-design-2026-08-16.md`
§2.1). `require` and `api` turn a missing module or a missing attribute into an
ordinary test FAILURE carrying a `red-unimplemented` message, so a not-yet-built
primitive reports as F rather than as a collection error or a skip. A skipped test
reads as swept on every future audit; a failing test does not.

Test-design record: Claude/Curie/superslm-t066-harness-test-design-2026-07-14.md
(original, Wizard repo)
"""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
CORPUS_PATH = REPO_ROOT / "tools" / "reference_pipeline" / "data" / "shopkeeper_corpus_v1.jsonl"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def require(module_name):
    """Import an implementation module, or fail the calling test as red-unimplemented."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(
            f"red-unimplemented: cannot import {module_name} ({exc}). "
            f"The implementation does not exist yet; see the API contract in "
            f"Claude/Curie/superslm-t066-harness-test-design-2026-07-14.md"
        ) from None


def api(module_name, *attrs):
    """Return the named attributes of an implementation module.

    Fails the calling test as red-unimplemented if the module or any attribute is
    absent. Returns a single value for one attr, a tuple for several.
    """
    module = require(module_name)
    missing = [a for a in attrs if not hasattr(module, a)]
    if missing:
        raise AssertionError(
            f"red-unimplemented: {module_name} lacks {', '.join(missing)}. "
            f"See the API contract in "
            f"Claude/Curie/superslm-t066-harness-test-design-2026-07-14.md"
        )
    values = tuple(getattr(module, a) for a in attrs)
    return values[0] if len(values) == 1 else values


# --- isolating the calibration reduction from the encoder ----------------------


def in_cap_tokenize(cfg):
    """A per-record encoding that DIFFERS per record and fits `cfg.context_cap`.

    **Why this exists (finding P-18, 2026-07-15).** The §11 fixture's `context_cap` is 16
    and its run prompt renders to ~1813 bytes, so `_calibrate` truncates every record to the
    same 16-byte system-prompt prefix — `'<|system|>\\nYou a'`. Measured: the fixture's 600
    run-prompt encodings are **600 distinct at full length and 1 distinct after the slice**.
    Under the default encoder the calibration reduction therefore sees one sequence, 600
    times, and `calibrate(one record) == calibrate(all 600)`.

    That silently disarms the cells that measure the *reduction*. Measured against the
    mutants the test-design record claims they catch: `test_calibration_is_order_independent`
    passed an early stop, a `first_n` draw and a running mean — **0 of 3**. Supplied with
    distinct in-cap sequences it catches **3 of 3**.

    **Why the encoding is supplied rather than the fixture's encoder changed.** These cells'
    subject is `_calibrate`'s fold over the record set — order-independence, cross-process
    determinism, whole-corpus observation. The encoder is not their subject, and an encoder
    contrived to survive a 16-token slice would be an encoder built to satisfy a bound. The
    run's prompt shape is pinned separately, by
    `test_calibration_and_the_run_encode_the_same_prompt` and
    `test_the_run_decodes_the_chat_template_not_the_bare_utterance`. `calibrate(..., tokenize=)`
    exists for exactly this isolation.

    Derived by SHA-256 of the record id, not `hash()`: Python randomises `str` hashing per
    process, and `test_calibration_is_deterministic_across_processes` compares across
    processes — a process-varying encoder would fail it for the encoder's reason rather than
    the reduction's.
    """
    def tokenize(record):
        digest = hashlib.sha256(str(record["id"]).encode("utf-8")).digest()
        return [byte % cfg.vocab_size for byte in digest[:8]]

    return tokenize


# --- the frozen §4 corpus ------------------------------------------------------

# Pinned facts about Claude/Docs/spike/shopkeeper_corpus_v1.jsonl. Schema §4 freezes
# and hashes the corpus, so these are constants, not measurements. Every corpus-level
# expected score in test_scoring.py is derived from them by hand in the test-design
# record and hard-coded there, so no test re-computes the scorer it is checking.
#
# CORPUS_SHA256 pins the exact bytes those derivations belong to. It is asserted, not
# documented: if the corpus moves, the derived scores are stale and the guard cell
# must fail loudly rather than let a stale expectation pass against new data. It has
# already earned its keep once — the corpus was regenerated for schema §8 F-5
# (`negated_slot`), which moved four derived scores, and this constant is what made
# that a loud failure instead of a silent wrong number.
#
# The hash is over the LF-normalized bytes — the canonical form Claude/Docs/spike/CORPUS.md
# publishes. `.gitattributes` now pins `Claude/Docs/spike/*.jsonl text eol=lf`, so a
# worktree checkout already equals the canonical bytes on every platform; the
# normalization in `corpus_canonical_bytes` is belt-and-braces against that pin
# regressing, and costs nothing while it holds.
CORPUS_SHA256 = "640a5770da9e093446e2aece8b3b7e2869476963e54039af1e897ec750e4d9ae"
CORPUS_RECORDS = 600
CORPUS_SPECIFIED_GOLD_SLOTS = 1375  # gold slots whose value != "unspecified"
CORPUS_EMPTY_GOLD_SLOTS = 3425      # 600 * 8 - 1375
CORPUS_INTENT_BOOK = 348
CORPUS_POLARITY_AFFIRM = 510
CORPUS_OOD_RECORDS = 60
CORPUS_CORRECTION_RECORDS = 90
CORPUS_NEGATION_RECORDS = 90        # class == "negation"; identical to polarity == "negate"


def corpus_canonical_bytes():
    """The corpus in its canonical LF form, whatever the checkout did to line endings."""
    return CORPUS_PATH.read_bytes().replace(b"\r\n", b"\n")


@pytest.fixture(scope="session")
def corpus():
    """The frozen 600-record shopkeeper eval corpus, as a list of records."""
    if not CORPUS_PATH.exists():
        pytest.fail(f"corpus fixture missing: {CORPUS_PATH}", pytrace=False)
    text = corpus_canonical_bytes().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- sharded safetensors checkpoint fixtures (T-1912/T-1921) -------------------
#
# Shared low-level writers for a checkpoint's on-disk shape: one safetensors file
# (`write_safetensors_shard`), an index (`write_index_json`), and the two ways a Curie
# cell needs to assemble them -- a self-consistent multi-shard checkpoint built directly
# from `{shard_filename: {tensor_name: array}}` (`self_consistent_sharded_checkpoint`),
# and a portable symlink farm mirroring the real HuggingFace hub cache's own
# `snapshots/`/`blobs/` layout (`symlinked_shard_farm`, D-SLM2209). Kept here rather than
# in one test file because both `test_pipeline.py`-facing and `test_artifact_cache.py`-
# facing cells need them (T-1912 design §6 Slot 1).
#
# Test-design record: Claude/Curie/t1921-shard-support-red-suite-2026-08-11.md


def write_safetensors_shard(path, tensors, *, dtypes=None):
    """One safetensors file at `path`, from `{tensor_name: array}`.

    Mirrors `pipeline._SafeTensors`'s own on-disk shape exactly: an 8-byte little-endian
    header-length prefix, the JSON header (name -> {dtype, shape, data_offsets}), then the
    raw bytes back to back -- the same minimal shape `test_pipeline.py` and
    `test_artifact_cache.py` already each write independently for the single-file case.

    `dtypes` (optional): `{tensor_name: "F32" | "BF16"}`, default `"F32"` for every tensor
    not named. A `"BF16"` entry stores the array's **top 16 bits only** -- the same
    reinterpretation `_SafeTensors.tensor()` widens back from
    (`raw.view(uint16).astype(uint32) << 16`) -- so a value written through
    `bf16_clean` below round-trips exactly; a value that is not already bf16-clean loses
    its low mantissa bits on write, same as any real bf16-stored checkpoint would have.
    """
    import numpy as np

    dtypes = dtypes or {}
    header = {}
    payload = bytearray()
    offset = 0
    for name, arr in tensors.items():
        dtype = dtypes.get(name, "F32")
        arr32 = np.ascontiguousarray(arr, dtype=np.float32)
        if dtype == "BF16":
            words = arr32.view(np.uint32)
            data = (words >> np.uint32(16)).astype(np.uint16).tobytes()
        elif dtype == "F32":
            data = arr32.tobytes()
        else:
            raise ValueError(f"write_safetensors_shard: unsupported dtype {dtype!r}")
        header[name] = {"dtype": dtype, "shape": list(arr32.shape),
                        "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    header["__metadata__"] = {}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(bytes(payload))


def bf16_clean(arr):
    """Zero an array's low 16 mantissa bits so it is already exactly bf16-representable.

    Writing a value through this first makes the write->widen round trip exact by
    construction -- independent of which code performs the truncation -- so a numeric-
    fidelity cell can assert `np.array_equal` rather than an approximate bound.
    """
    import numpy as np

    arr32 = np.ascontiguousarray(arr, dtype=np.float32)
    words = arr32.view(np.uint32) & np.uint32(0xFFFF0000)
    return words.copy().view(np.float32)


def write_index_json(path, weight_map, *, total_size=None):
    """`model.safetensors.index.json`'s on-disk shape: `{"metadata": {...}, "weight_map":
    {tensor_name: shard_filename}}` -- the HF sharding convention `_open_checkpoint_tensors`
    (T-1912 design §4.2) tests for by this file's presence, not by counting shards."""
    if total_size is None:
        total_size = 0
    body = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    Path(path).write_text(json.dumps(body), encoding="utf-8")


def self_consistent_sharded_checkpoint(checkpoint_dir, shard_contents, *, dtypes=None):
    """A checkpoint directory whose N shards and index are mutually self-consistent.

    `shard_contents`: `{shard_filename: {tensor_name: array}}`. Every tensor named by a
    shard's own on-disk header is also named by `weight_map` for that exact shard, and
    vice versa -- every one of the T-1912 design's six §4.4 modes passes (vacuously where
    a mode's condition never arises) on this construction. `dtypes` (optional):
    `{tensor_name: "F32" | "BF16"}`, forwarded per-tensor to `write_safetensors_shard`.
    Returns `weight_map` (`{tensor_name: shard_filename}`) so a test can mutate the index,
    a shard's header, or the on-disk file set from a known-good starting point.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard_name, tensors in shard_contents.items():
        write_safetensors_shard(checkpoint_dir / shard_name, tensors, dtypes=dtypes)
        for name in tensors:
            weight_map[name] = shard_name
    write_index_json(checkpoint_dir / "model.safetensors.index.json", weight_map)
    return weight_map


def symlinked_shard_farm(tmp_path, shard_contents):
    """A synthetic checkpoint whose shard files are REAL `os.symlink`s into a sibling
    directory outside the checkpoint directory -- mirroring the HuggingFace hub cache's own
    `snapshots/<hash>/` (the checkpoint dir) / `blobs/<hash>` (the symlink targets) layout,
    without a real download (D-SLM2209; the T-1917 strike found the real Qwen2.5-3B-Instruct
    checkpoint stored exactly this way, and the design's own mode-0 fix -- judging a shard
    name by its own declared text, never by where it resolves -- exists because of it).

    `index.json` is written directly into the checkpoint directory (not symlinked); only
    the two shard files are, matching the design's own required Coverage Model cell
    (T-1912 design §7 dimension 2) -- the checkpoint dir's declared contents include the
    symlink itself, which is the property mode 0 must accept.

    Returns `(checkpoint_dir, weight_map)`. Skips with a named reason -- never fails and
    never silently passes as green -- when this host cannot create a symlink (Windows
    without Developer Mode or elevation; `huggingface_hub` itself falls back to copying
    real files in that case, per T-1912 design §3), mirroring `@pytest.mark.upstream`'s
    skip-if-absent convention for a real-checkpoint dependency this fixture does not need
    but a host permission might still deny.
    """
    tmp_path = Path(tmp_path)
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = tmp_path / "snapshot"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard_name, tensors in shard_contents.items():
        blob_path = blobs_dir / f"blob-{shard_name}"
        write_safetensors_shard(blob_path, tensors)
        link_path = checkpoint_dir / shard_name
        try:
            os.symlink(blob_path, link_path)
        except OSError as exc:
            pytest.skip(
                f"host cannot create a symlink ({exc}); the mode-0 symlink-farm cell needs "
                f"real os.symlink privilege (Windows: Developer Mode or elevation) to "
                f"reproduce the HuggingFace hub cache's own snapshots/blobs/ layout "
                f"(D-SLM2209) -- this is a host-permission gap, not a design or fixture "
                f"defect, and this cell is not exercised on this host"
            )
        for name in tensors:
            weight_map[name] = shard_name
    write_index_json(checkpoint_dir / "model.safetensors.index.json", weight_map)
    return checkpoint_dir, weight_map
