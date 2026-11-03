"""Engine 1.19: a tied checkpoint that also ships a duplicate `lm_head.weight` (red suite).

Contract (release scope, 2026-11-03): when `config.json` has `tie_word_embeddings: true` and
the checkpoint ALSO carries `lm_head.weight`,

- accept it when the tensor is byte-identical to `model.embed_tokens.weight` -- same dtype,
  same shape, same raw bytes -- and convert exactly as if it were absent: the output is
  byte-identical to converting a copy with `lm_head.weight` removed;
- refuse with a clear, named error when it differs in any byte, in shape, or in dtype;
- untied checkpoints are unchanged.

The real case is Qwen/Qwen3-0.6B (bfloat16, `model.`-prefixed backbone, QK-norm, no q/k/v
bias, head_dim 128, `lm_head.weight` stored after the backbone). The cloud has no Hugging Face
access, so every checkpoint here is a tiny synthetic one written to disk in that shape.
head_dim is 128 because the fused-QK converter refuses any other head_dim for a QK-norm model
(`UnsupportedFusedKHeadDim`), and the acceptance cells must reach the written `.sslm`.

Red today: `pipeline.load_model` maps `lm_head.weight` only for untied configs, so on a tied
config the duplicate is an unmapped tensor and the load raises the generic
`UnsupportedOpSet("... checkpoint tensors the map does not name ...")`. That rejection is
indiscriminate -- it refuses the byte-identical duplicate the contract accepts, and its
message never names `embed_tokens.weight` or the property that differs, which the refusal
cells require.

Test-design record: /mnt/project-files/plans/superslm-array-converter/records/
curie-qwen3-red-suite.md
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import require

MODULE = "reference_pipeline.pipeline"

_TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

HIDDEN = 8
LAYERS = 1
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 128
INTERMEDIATE = 16

EMBED = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"


def _fixture_mod():
    import _calibrate_checkpoint_fixture as fixture_mod
    return fixture_mod


# ==============================================================================
# Checkpoint construction -- dtype-exact, so dtype/shape/bytes can each be varied alone
# ==============================================================================


def _bf16(values):
    """bfloat16 raw bytes: float32's top 16 bits (truncation -- the stored value is whatever
    those bits say, and every comparison below is on the stored bits)."""
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return (words >> np.uint32(16)).astype("<u2").tobytes()


def _bf16_widened(raw):
    """The exact float32 values a bfloat16 byte string stores."""
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)


def _entry(dtype, shape, raw):
    return {"dtype": dtype, "shape": tuple(shape), "raw": bytes(raw)}


def _f32(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    return _entry("F32", values.shape, values.astype("<f4").tobytes())


def _backbone(*, qk_norm, biased, embed_dtype, seed=0):
    """`{name: entry}` for one checkpoint, `lm_head.weight` excluded.

    `qk_norm=True, biased=False` is Qwen3's shape; `qk_norm=False, biased=True` is Qwen2.5's.
    The embedding is stored in `embed_dtype` ("BF16" is what Qwen3-0.6B ships); everything
    else is F32, which the loader reads in the same file without complaint."""
    rng = np.random.default_rng(seed)
    vocab = _fixture_mod().VOCAB_SIZE

    def small(*shape):
        return rng.normal(scale=0.02, size=shape).astype(np.float32)

    embed = small(vocab, HIDDEN)
    tensors = {EMBED: (_entry("BF16", embed.shape, _bf16(embed)) if embed_dtype == "BF16"
                       else _f32(embed))}
    tensors["model.norm.weight"] = _f32(np.ones(HIDDEN))
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = _f32(np.ones(HIDDEN))
        tensors[f"{p}.post_attention_layernorm.weight"] = _f32(np.ones(HIDDEN))
        tensors[f"{p}.self_attn.q_proj.weight"] = _f32(small(HEADS * HEAD_DIM, HIDDEN))
        tensors[f"{p}.self_attn.k_proj.weight"] = _f32(small(KV_HEADS * HEAD_DIM, HIDDEN))
        tensors[f"{p}.self_attn.v_proj.weight"] = _f32(small(KV_HEADS * HEAD_DIM, HIDDEN))
        tensors[f"{p}.self_attn.o_proj.weight"] = _f32(small(HIDDEN, HEADS * HEAD_DIM))
        if biased:
            tensors[f"{p}.self_attn.q_proj.bias"] = _f32(small(HEADS * HEAD_DIM))
            tensors[f"{p}.self_attn.k_proj.bias"] = _f32(small(KV_HEADS * HEAD_DIM))
            tensors[f"{p}.self_attn.v_proj.bias"] = _f32(small(KV_HEADS * HEAD_DIM))
        if qk_norm:
            tensors[f"{p}.self_attn.q_norm.weight"] = _f32(0.5 + rng.random(HEAD_DIM))
            tensors[f"{p}.self_attn.k_norm.weight"] = _f32(0.5 + rng.random(HEAD_DIM))
        tensors[f"{p}.mlp.gate_proj.weight"] = _f32(small(INTERMEDIATE, HIDDEN))
        tensors[f"{p}.mlp.up_proj.weight"] = _f32(small(INTERMEDIATE, HIDDEN))
        tensors[f"{p}.mlp.down_proj.weight"] = _f32(small(HIDDEN, INTERMEDIATE))
    return tensors


def _write_checkpoint(checkpoint_dir, tensors, *, tie_word_embeddings):
    """config.json + one safetensors file (entries in dict order, so `lm_head.weight` added
    last is stored last, as in the real checkpoint) + the shared fixture tokenizer."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    header, payload, offset = {}, bytearray(), 0
    for name, entry in tensors.items():
        header[name] = {"dtype": entry["dtype"], "shape": list(entry["shape"]),
                        "data_offsets": [offset, offset + len(entry["raw"])]}
        payload += entry["raw"]
        offset += len(entry["raw"])
    header["__metadata__"] = {"format": "pt"}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(checkpoint_dir / "model.safetensors", "wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(bytes(payload))
    fixture_mod = _fixture_mod()
    config = {
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE, "vocab_size": fixture_mod.VOCAB_SIZE,
        "rope_theta": 1000000.0, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": tie_word_embeddings,
        "max_position_embeddings": fixture_mod.MAX_POSITION_EMBEDDINGS,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    fixture_mod._write_tokenizer(checkpoint_dir)
    return checkpoint_dir


def _with_lm_head(tensors, lm_head):
    with_head = dict(tensors)
    with_head[LM_HEAD] = lm_head
    return with_head


# ==============================================================================
# The product path -- the two documented CLI steps, run as a caller runs them
# ==============================================================================


def _convert(checkpoint, work):
    """`calibrate_checkpoint.py` then `convert_model.py --skip-verify` (quickstart steps 1a
    and 3; the QK-norm flow's own first conversion is the same `load_model` + cache +
    `convert_model.py` sequence). Returns the cache directory and the `.sslm` bytes."""
    work = Path(work)
    work.mkdir(parents=True)
    cache = work / "cache"
    sslm = work / "model.sslm"
    for command in (
        [sys.executable, str(_TOOLS_DIR / "calibrate_checkpoint.py"),
         "--checkpoint", str(checkpoint), "--out", str(cache)],
        [sys.executable, str(_TOOLS_DIR / "convert_model.py"),
         "--artifact", str(cache), "--out", str(sslm), "--skip-verify"],
    ):
        run = subprocess.run(command, capture_output=True, text=True)
        assert run.returncode == 0, (
            f"{Path(command[1]).name} refused {checkpoint.name}:\n{run.stderr[-3000:]}")
    return cache, sslm.read_bytes()


def _cache_contents(cache):
    """Every file of the calibrated cache, with `metadata.json`'s `checkpoint_path` (the one
    field that records WHERE the checkpoint was, not what it held) set aside."""
    contents = {}
    for path in sorted(Path(cache).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(cache).as_posix()
        if rel == "metadata.json":
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata.pop("checkpoint_path", None)
            contents[rel] = json.dumps(metadata, sort_keys=True)
        else:
            contents[rel] = path.read_bytes()
    return contents


_VARIANTS = {
    # The product case: Qwen3-0.6B ships its tied pair in bfloat16.
    "qwen3-bf16": dict(qk_norm=True, biased=False, embed_dtype="BF16"),
    "qwen3-f32": dict(qk_norm=True, biased=False, embed_dtype="F32"),
    # The contract is about tied configs, not about Qwen3: a Qwen2.5-shaped tied checkpoint
    # carrying the same duplicate takes the same accept path.
    "qwen25-f32": dict(qk_norm=False, biased=True, embed_dtype="F32"),
}

_ABSENT_TWINS = {}


def _absent_twin(variant, tmp_path_factory):
    """The oracle: the same checkpoint with `lm_head.weight` removed, converted once per
    variant. This side is green on today's code -- it is the shape every tied incumbent has."""
    if variant not in _ABSENT_TWINS:
        root = tmp_path_factory.mktemp(f"absent-{variant}")
        checkpoint = _write_checkpoint(root / "checkpoint", _backbone(**_VARIANTS[variant]),
                                       tie_word_embeddings=True)
        _ABSENT_TWINS[variant] = (checkpoint, *_convert(checkpoint, root / "out"))
    return _ABSENT_TWINS[variant]


# ==============================================================================
# Cells
# ==============================================================================


def test_oracle_control_the_conversion_is_byte_stable_across_checkpoint_location(tmp_path,
                                                                                  tmp_path_factory):
    """Calibrates the comparison the acceptance cells rely on (green today, by design): the
    same lm_head-free checkpoint at two different paths converts to the same `.sslm` bytes
    and the same cache content. Without this, a byte difference in an acceptance cell could
    be the checkpoint's location rather than the duplicate head."""
    _checkpoint, absent_cache, absent_sslm = _absent_twin("qwen3-bf16", tmp_path_factory)
    elsewhere = _write_checkpoint(tmp_path / "elsewhere" / "checkpoint",
                                  _backbone(**_VARIANTS["qwen3-bf16"]), tie_word_embeddings=True)
    cache, sslm = _convert(elsewhere, tmp_path / "out")
    assert sslm == absent_sslm
    assert _cache_contents(cache) == _cache_contents(absent_cache)


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_tied_checkpoint_with_byte_identical_lm_head_converts_exactly_as_if_it_were_absent(
        variant, tmp_path, tmp_path_factory):
    """FEAT. A tied config whose checkpoint also carries `lm_head.weight` as an exact byte
    copy of `model.embed_tokens.weight` (same dtype, shape, bytes; stored last, as Qwen3-0.6B
    stores it) is accepted, and the `.sslm` it converts to is byte-identical to the one its
    lm_head-free twin converts to -- as is every file of the calibrated cache, location
    aside. Red today: `calibrate_checkpoint.py` exits non-zero on the unmapped
    `lm_head.weight`."""
    _checkpoint, absent_cache, absent_sslm = _absent_twin(variant, tmp_path_factory)
    tensors = _backbone(**_VARIANTS[variant])
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", _with_lm_head(tensors, tensors[EMBED]),
                                   tie_word_embeddings=True)
    cache, sslm = _convert(checkpoint, tmp_path / "out")
    assert sslm == absent_sslm, "the duplicate head changed the converted .sslm"
    assert _cache_contents(cache) == _cache_contents(absent_cache), (
        "the duplicate head changed the calibrated cache")


def _differing_heads():
    """Each case differs from the embedding in exactly ONE of {bytes, shape, dtype}, so an
    implementation that checks fewer than all three is caught by the case it skips. The
    embedding is bfloat16, as in the product checkpoint."""
    vocab = _fixture_mod().VOCAB_SIZE

    def one_bit_flipped(embed):
        raw = bytearray(embed["raw"])
        raw[0] ^= 0x01                 # the low mantissa bit of element [0, 0]: one ULP
        return _entry(embed["dtype"], embed["shape"], raw), "byte"

    def signed_zero(embed):
        # Value-equal, byte-different: element [0, 0] is +0.0 in the embedding and -0.0 in
        # the head. Catches a comparison by value (np.array_equal) instead of by bytes.
        return _entry(embed["dtype"], embed["shape"], b"\x00\x80" + embed["raw"][2:]), "byte"

    def transposed_shape(embed):
        # The identical byte string declared [hidden, vocab]: bytes and dtype agree.
        return _entry(embed["dtype"], (HIDDEN, vocab), embed["raw"]), "shape"

    def f16_same_bytes(embed):
        # The identical byte string and shape, declared float16: only the dtype differs.
        return _entry("F16", embed["shape"], embed["raw"]), "dtype"

    def f32_same_values(embed):
        # The exact values the bfloat16 embedding stores, re-stored as float32. Equal as read
        # (the loader widens both to the same float64), different dtype on disk. Catches a
        # comparison of the loader's widened arrays instead of the stored tensors.
        values = _bf16_widened(embed["raw"]).reshape(embed["shape"])
        return _f32(values), "dtype"

    return {"one-bit-flipped": one_bit_flipped, "signed-zero": signed_zero,
            "transposed-shape": transposed_shape, "f16-same-bytes": f16_same_bytes,
            "f32-same-values": f32_same_values}


def _setup_zero(tensors):
    """`signed_zero` needs element [0, 0] of the embedding to be +0.0 (bf16 bits 0x0000)."""
    embed = tensors[EMBED]
    tensors[EMBED] = _entry(embed["dtype"], embed["shape"], b"\x00\x00" + embed["raw"][2:])
    return tensors


@pytest.mark.parametrize("case", sorted(_differing_heads()))
def test_tied_checkpoint_with_a_differing_lm_head_is_refused_by_a_named_error(case, tmp_path):
    """A tied config whose `lm_head.weight` differs from `model.embed_tokens.weight` in any
    byte, in shape, or in dtype is refused at load, by an error class the pipeline defines,
    whose message names both tensors and the property that differs. Red today: the load is
    refused, but by the generic unmapped-tensor rejection, whose message names neither
    `embed_tokens.weight` nor the differing property."""
    pipeline = require(MODULE)
    tensors = _setup_zero(_backbone(**_VARIANTS["qwen3-bf16"]))
    lm_head, differing_property = _differing_heads()[case](tensors[EMBED])
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", _with_lm_head(tensors, lm_head),
                                   tie_word_embeddings=True)
    with pytest.raises(Exception) as caught:
        pipeline.load_model(checkpoint)
    error = caught.value
    message = str(error)
    assert type(error).__module__ == pipeline.__name__, (
        f"refused by {type(error).__module__}.{type(error).__name__}, not a named pipeline "
        f"error: {message}")
    assert "the map does not name" not in message, (
        f"refused by the generic unmapped-tensor rejection, not by the tied-duplicate "
        f"comparison: {message}")
    assert LM_HEAD in message and "embed_tokens.weight" in message, (
        f"the refusal does not name both tensors: {message}")
    assert differing_property in message.lower(), (
        f"the refusal does not say the {differing_property} differs: {message}")


def test_the_differing_head_cases_each_differ_in_exactly_the_property_they_name():
    """Guards the refusal cells' own construction (green by design): each case differs from
    the embedding in exactly the one property it is named for, and agrees on the other two."""
    tensors = _setup_zero(_backbone(**_VARIANTS["qwen3-bf16"]))
    embed = tensors[EMBED]
    for case, build in _differing_heads().items():
        head, prop = build(embed)
        differs = {"byte": head["raw"] != embed["raw"], "shape": head["shape"] != embed["shape"],
                   "dtype": head["dtype"] != embed["dtype"]}
        if prop == "dtype" and case == "f32-same-values":
            # Re-storing as float32 necessarily changes the byte string too; what this case
            # isolates is that the VALUES as read agree.
            assert not differs["shape"] and differs["dtype"], case
            assert np.array_equal(_bf16_widened(embed["raw"]),
                                  np.frombuffer(head["raw"], dtype="<f4")), case
            continue
        assert differs == {key: key == prop for key in differs}, (case, differs)


def test_untied_checkpoint_keeps_its_own_lm_head_even_when_byte_identical_to_the_embedding(
        tmp_path):
    """Untied checkpoints are unchanged (green today; a regression guard for the fix): an
    untied config's `lm_head.weight` is its own weight and is loaded as `lm_head` even when
    its bytes happen to equal the embedding's. Catches a fix that drops the duplicate without
    reading `tie_word_embeddings`."""
    pipeline = require(MODULE)
    tensors = _backbone(**_VARIANTS["qwen3-bf16"])
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", _with_lm_head(tensors, tensors[EMBED]),
                                   tie_word_embeddings=False)
    model = pipeline.load_model(checkpoint)
    assert "lm_head" in model.weights
    assert np.array_equal(np.asarray(model.weights["lm_head"]), np.asarray(model.weights["embed"]))


# ==============================================================================
# Engine 1.19 fold (coverage audit gaps 1 and 2): where the difference sits, and how the
# checkpoint is stored
# ==============================================================================

# The compare chunk the cells below inject in place of the production 16 MiB, so a tensor of
# a few hundred bytes spans many chunks. The guard cell proves the injection reaches the
# comparison; without it a chunk cell would pass vacuously on a one-chunk tensor.
_SMALL_CHUNK = 64


class _RecordingBytes:
    """A byte sequence that records each slice the comparison takes of it."""

    def __init__(self, raw):
        self._raw = np.frombuffer(bytes(raw), dtype=np.uint8)
        self.slices = []

    def __len__(self):
        return len(self._raw)

    def __getitem__(self, key):
        self.slices.append((key.start, key.stop))
        return self._raw[key]


def test_the_injected_compare_chunk_is_the_one_the_comparison_uses(monkeypatch):
    """Guard for the chunk cells: with `_STORED_COMPARE_CHUNK` patched, the comparison reads
    the tensor in slices no wider than the patched chunk and reads every slice up to the end
    when the bytes agree."""
    pipeline = require(MODULE)
    monkeypatch.setattr(pipeline, "_STORED_COMPARE_CHUNK", _SMALL_CHUNK)
    size = _SMALL_CHUNK * 5 + 7
    left, right = _RecordingBytes(bytes(size)), _RecordingBytes(bytes(size))
    assert pipeline._first_differing_byte(left, right) is None
    assert len(left.slices) == 6
    assert all(min(stop, size) - start <= _SMALL_CHUNK for start, stop in left.slices)
    assert left.slices[-1][1] >= size


def _byte_flipped_at(offset):
    def build(embed):
        raw = bytearray(embed["raw"])
        raw[offset] ^= 0x01
        return _entry(embed["dtype"], embed["shape"], raw)
    return build


def _planted_offsets(embed_size):
    """`{case: offset}`: the last byte (a ragged final chunk, since the embedding is not a
    multiple of the chunk), the first byte past chunk 1, and the last byte of chunk 1."""
    assert embed_size % _SMALL_CHUNK != 0 and embed_size > 3 * _SMALL_CHUNK, embed_size
    return {"last-byte": embed_size - 1, "first-byte-past-chunk-1": _SMALL_CHUNK,
            "last-byte-of-chunk-1": _SMALL_CHUNK - 1}


@pytest.mark.parametrize("case", ["last-byte", "first-byte-past-chunk-1",
                                  "last-byte-of-chunk-1"])
def test_a_head_differing_only_deep_in_the_tensor_is_refused_at_the_planted_byte(
        case, tmp_path, monkeypatch):
    """Audit gap 1. A tied head equal to the embedding everywhere except one byte, placed
    where a prefix or first-chunk comparison would not look, is refused by
    `TiedLmHeadMismatch`; the message names both tensors and reports the planted offset."""
    pipeline = require(MODULE)
    monkeypatch.setattr(pipeline, "_STORED_COMPARE_CHUNK", _SMALL_CHUNK)
    tensors = _backbone(**_VARIANTS["qwen3-bf16"])
    embed = tensors[EMBED]
    offset = _planted_offsets(len(embed["raw"]))[case]
    checkpoint = _write_checkpoint(
        tmp_path / "checkpoint", _with_lm_head(tensors, _byte_flipped_at(offset)(embed)),
        tie_word_embeddings=True)
    with pytest.raises(pipeline.TiedLmHeadMismatch) as caught:
        pipeline.load_model(checkpoint)
    message = str(caught.value)
    assert LM_HEAD in message and EMBED in message, message
    assert f"first at byte {offset})" in message, message


def _shard_split(tensors, lm_head_shard_name, rest_shard_name):
    """`{shard filename: {name: entry}}`: `lm_head.weight` alone in its own shard, every
    other tensor (the embedding included) in the other."""
    shards = {rest_shard_name: {}, lm_head_shard_name: {}}
    for name, entry in tensors.items():
        shards[lm_head_shard_name if name == LM_HEAD else rest_shard_name][name] = entry
    return {name: entries for name, entries in shards.items() if entries}


def _write_sharded_checkpoint(checkpoint_dir, tensors, *, tie_word_embeddings):
    """The same checkpoint `_write_checkpoint` writes, stored as an index plus two shards,
    with `lm_head.weight` in a different shard from the embedding (as a sharded checkpoint
    that stores the head last does)."""
    checkpoint_dir = _write_checkpoint(checkpoint_dir, tensors,
                                       tie_word_embeddings=tie_word_embeddings)
    (checkpoint_dir / "model.safetensors").unlink()
    first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    weight_map = {}
    for shard_name, entries in _shard_split(tensors, second, first).items():
        header, payload, offset = {}, bytearray(), 0
        for name, entry in entries.items():
            header[name] = {"dtype": entry["dtype"], "shape": list(entry["shape"]),
                            "data_offsets": [offset, offset + len(entry["raw"])]}
            payload += entry["raw"]
            offset += len(entry["raw"])
            weight_map[name] = shard_name
        header["__metadata__"] = {"format": "pt"}
        header_bytes = json.dumps(header).encode("utf-8")
        with open(checkpoint_dir / shard_name, "wb") as handle:
            handle.write(len(header_bytes).to_bytes(8, "little"))
            handle.write(header_bytes)
            handle.write(bytes(payload))
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(len(e["raw"]) for e in tensors.values())},
                    "weight_map": weight_map}), encoding="utf-8")
    return checkpoint_dir


def _bare(tensors):
    """The same tensors under the bare namespace (`embed_tokens.weight`, no `model.`)."""
    return {(name[len("model."):] if name.startswith("model.") else name): entry
            for name, entry in tensors.items()}


def test_the_sharded_writer_puts_the_head_and_the_embedding_in_different_shards(tmp_path):
    """Guard for the sharded cells: the checkpoint is read through the sharded reader, and
    the head and the embedding come from different shard files."""
    pipeline = require(MODULE)
    tensors = _backbone(**_VARIANTS["qwen3-bf16"])
    checkpoint = _write_sharded_checkpoint(tmp_path / "checkpoint",
                                           _with_lm_head(tensors, tensors[EMBED]),
                                           tie_word_embeddings=True)
    assert not (checkpoint / "model.safetensors").exists()
    reader = pipeline._open_checkpoint_tensors(checkpoint)
    assert type(reader).__name__ == "_ShardedSafeTensors"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    assert index["weight_map"][LM_HEAD] != index["weight_map"][EMBED]
    assert reader.stored(LM_HEAD)[2].tobytes() == reader.stored(EMBED)[2].tobytes()


def test_sharded_tied_checkpoint_with_byte_identical_lm_head_converts_exactly_as_if_it_were_absent(
        tmp_path, tmp_path_factory):
    """Audit gap 2. The identical duplicate in its own shard is accepted, and the `.sslm` and
    cache equal those of the single-file, head-free twin."""
    _checkpoint, absent_cache, absent_sslm = _absent_twin("qwen3-bf16", tmp_path_factory)
    tensors = _backbone(**_VARIANTS["qwen3-bf16"])
    checkpoint = _write_sharded_checkpoint(tmp_path / "checkpoint",
                                           _with_lm_head(tensors, tensors[EMBED]),
                                           tie_word_embeddings=True)
    cache, sslm = _convert(checkpoint, tmp_path / "out")
    assert sslm == absent_sslm, "the sharded duplicate head changed the converted .sslm"
    assert _cache_contents(cache) == _cache_contents(absent_cache)


@pytest.mark.parametrize("case", ["one-bit-flipped", "signed-zero", "last-byte"])
def test_sharded_tied_checkpoint_with_a_differing_lm_head_is_refused_by_the_named_error(
        case, tmp_path):
    """Audit gap 2. A differing head in its own shard is refused by `TiedLmHeadMismatch`,
    naming both tensors and the planted offset. `signed-zero` is value-equal and
    byte-different."""
    pipeline = require(MODULE)
    tensors = _setup_zero(_backbone(**_VARIANTS["qwen3-bf16"]))
    embed = tensors[EMBED]
    if case == "last-byte":
        lm_head, offset = _byte_flipped_at(len(embed["raw"]) - 1)(embed), len(embed["raw"]) - 1
    else:
        lm_head, offset = _differing_heads()[case](embed)[0], 0 if case == "one-bit-flipped" else 1
    checkpoint = _write_sharded_checkpoint(tmp_path / "checkpoint",
                                           _with_lm_head(tensors, lm_head),
                                           tie_word_embeddings=True)
    with pytest.raises(pipeline.TiedLmHeadMismatch) as caught:
        pipeline.load_model(checkpoint)
    message = str(caught.value)
    assert LM_HEAD in message and EMBED in message, message
    assert f"first at byte {offset})" in message, message


_BARE_EMBED = "embed_tokens.weight"


def test_bare_namespace_tied_checkpoint_with_byte_identical_lm_head_converts_as_if_absent(
        tmp_path):
    """Audit gap 2, bare namespace. A tied checkpoint whose backbone carries no `model.`
    prefix and whose `lm_head.weight` is a byte copy of `embed_tokens.weight` converts to the
    same `.sslm` and cache as its own head-free bare twin."""
    tensors = _bare(_backbone(**_VARIANTS["qwen3-bf16"]))
    absent = _write_checkpoint(tmp_path / "absent", tensors, tie_word_embeddings=True)
    absent_cache, absent_sslm = _convert(absent, tmp_path / "out-absent")
    duplicate = _write_checkpoint(tmp_path / "duplicate",
                                  _with_lm_head(tensors, tensors[_BARE_EMBED]),
                                  tie_word_embeddings=True)
    cache, sslm = _convert(duplicate, tmp_path / "out-duplicate")
    assert sslm == absent_sslm
    assert _cache_contents(cache) == _cache_contents(absent_cache)


@pytest.mark.parametrize("case", ["one-bit-flipped", "signed-zero"])
def test_bare_namespace_tied_checkpoint_with_a_differing_lm_head_is_refused(case, tmp_path):
    """Audit gap 2, bare namespace. The comparison finds the bare embedding: a differing head
    is refused by `TiedLmHeadMismatch` naming `embed_tokens.weight` without a prefix."""
    pipeline = require(MODULE)
    tensors = _bare(_setup_zero(_backbone(**_VARIANTS["qwen3-bf16"])))
    lm_head, _prop = _differing_heads()[case](tensors[_BARE_EMBED])
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", _with_lm_head(tensors, lm_head),
                                   tie_word_embeddings=True)
    with pytest.raises(pipeline.TiedLmHeadMismatch) as caught:
        pipeline.load_model(checkpoint)
    message = str(caught.value)
    assert f"{LM_HEAD}, which is not a stored copy of {_BARE_EMBED}:" in message, message
    assert "byte" in message, message
