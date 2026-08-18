"""T-1912 — multi-shard safetensors checkpoint support: the red suite (T-1921).

Realizes `Claude/Vitruvius/t1912-sharded-checkpoint-support-design-2026-08-11.md` §7's
Coverage Model, as folded by T-1916 and T-1920 (design §11/§12) -- the design's most
recent state, not a transcription of an earlier draft. Authored red, before
`_open_checkpoint_tensors`, `_ShardedSafeTensors`, and `_require_safe_shard_filename` exist
in `reference_pipeline.pipeline`: every cell that targets one of those three symbols directly
fails via `conftest.api`'s `red-unimplemented` AssertionError, this suite's existing
precedent for a cell that cannot go red any other way (`conftest.py`'s own module
docstring). Cells that drive the new code through the existing `load_model` and
`artifact_cache.load_artifact` entry points instead fail today for those entry points' OWN
current reason -- `pipeline._checkpoint_tensor_file`'s pre-existing "N weight shards"
rejection, or `artifact_cache._reopen_or_stub_float_source`'s current message shape -- named
at each cell as `red-uncalibrated`.

Coverage Model dimension -> cell map (design §7; dimension 3, concurrency, is
not-applicable and carries no cell; dimension 5's construction-time-firing property is
discharged structurally by dimension 2's six mode cells, none of which ever calls
`.tensor()` -- see this file's dimension-5 note below rather than a duplicate cell):

    1  -> TestDimension1Lifetime
    2  -> TestDimension2TrustBoundaries (+ TestDimension2Mode0)
    4  -> TestDimension4ShapeAndPlatform
    5  -> note only, in TestDimension2TrustBoundaries's module docstring
    6  -> TestDimension6NumericalEdges
    7  -> TestDimension7ContractClaims
    8  -> TestDimension8Composition
    9  -> TestDimension9Persistence
    10 -> TestDimension10FunctionalAchievement
    11 -> discharged by TestDimension2TrustBoundaries's mode 1-5 cells (each is a single
         mutation off a known-valid baseline) -- see that class's docstring

Full cell-by-cell rationale: `Claude/Curie/t1921-shard-support-red-suite-2026-08-11.md`.

**T-1928 addendum (2026-08-11).** The T-1922 build landed this feature and every cell above
now passes. A confirmation pass (`Claude/Poirot/d5dd7ce507-shard-fix-confirmation.md`, C1)
then reverted each of three production fixes to its literal pre-fix text and found that NO
cell above -- in this file, the pre-existing regression tier, or the real-checkpoint
upstream tier -- moved. The three cells at the end of this file (`TestT1928MutationPins`)
close that gap: each is a mutation-verified tripwire for one specific, previously-invisible
regression, proven red against the reverted text and green against the fix, both directions
executed and reported. Test-design record: `Claude/Curie/t1928-shard-mutation-pins-2026-08-11.md`.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    TOOLS_DIR,
    api,
    bf16_clean,
    require,
    self_consistent_sharded_checkpoint,
    symlinked_shard_farm,
    write_index_json,
    write_safetensors_shard,
)

np = pytest.importorskip("numpy")

MODULE = "reference_pipeline.pipeline"
ARTIFACT_MODULE = "reference_pipeline.artifact_cache"


# ==============================================================================
# Shared fixtures local to this file -- a fully populated tiny checkpoint, and the
# real Qwen2.5-3B-Instruct checkpoint's discovery (mirrors `test_pipeline.py`'s
# `qwen_checkpoint()`/`upstream_required()` for the 1.5B checkpoint, at the 3B tier).
# ==============================================================================


def _tiny_cfg(pipeline, **overrides):
    """A config small enough that `load_model`'s calibration pass (600 corpus records,
    the float reference recomputed per record) completes in well under a second -- the
    same sizing idiom `test_artifact_cache.py`'s `_tiny_checkpoint_cfg` uses, restated
    here so this file has no import dependency on that file's local helper."""
    body = dict(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=4, intermediate_size=8, vocab_size=4,
                rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
                context_cap=4)
    body.update(overrides)
    return pipeline.ModelConfig(**body)


def _write_tiny_config_json(tmp_path, cfg):
    body = dict(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        rope_theta=cfg.rope_theta, rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=cfg.tie_word_embeddings,
        max_position_embeddings=cfg.context_cap,
    )
    path = Path(tmp_path) / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _upstream_tensor_shapes(pipeline, cfg):
    """`{upstream_name: shape}` for every tensor `_upstream_names(cfg)` demands, including
    the q/k/v biases (1-D, output width) `_weight_shapes` does not carry."""
    names = pipeline._upstream_names(cfg)
    own_shapes = dict(pipeline._weight_shapes(cfg))
    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    bias_width = {"q_proj": q_width, "k_proj": kv_width, "v_proj": kv_width}
    shapes = {}
    for upstream, ours in names.items():
        if ours.endswith(".bias"):
            leaf = ours.rsplit(".", 2)[-2]
            shapes[upstream] = (bias_width[leaf],)
        else:
            shapes[upstream] = own_shapes[ours]
    return shapes


def _deterministic_tensor(name, shape):
    """A reproducible array for `name`, derived by SHA-256 rather than `hash()` -- process-
    stable, mirroring `conftest.in_cap_tokenize`'s own reason for the same choice."""
    seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def _write_full_sharded_checkpoint(checkpoint_dir, pipeline, cfg, *, shard_of):
    """Every tensor `_upstream_names(cfg)` demands, deterministically valued, split across
    shards by `shard_of(upstream_name) -> shard_filename`, plus a real `config.json` -- a
    checkpoint `load_model` can load end to end with no rejection. Returns `values`,
    `{upstream_name: float32 array}`, the exact bytes written, for direct comparison."""
    _write_tiny_config_json(checkpoint_dir, cfg)
    shapes = _upstream_tensor_shapes(pipeline, cfg)
    shard_contents = {}
    values = {}
    for upstream, shape in shapes.items():
        arr = _deterministic_tensor(upstream, shape)
        values[upstream] = arr
        shard_contents.setdefault(shard_of(upstream), {})[upstream] = arr
    self_consistent_sharded_checkpoint(checkpoint_dir, shard_contents)
    return values


def deterministic_tokenize_prompt(cfg):
    """A `tokenize_prompt` matching `load_model`'s OWN parameter contract: `load_model`
    calls it as `tokenize_prompt(run_prompt_messages(record))` (`pipeline.py:1394-1395`) --
    with the rendered chat MESSAGES list, never the raw corpus record. This is a different
    shape than `conftest.in_cap_tokenize`, which is built for the standalone `calibrate()`
    function's `tokenize` parameter (`pipeline.calibrate`, called as `tokenize(record)`
    directly) -- passing that helper to `load_model(..., tokenize_prompt=...)` instead
    raises `TypeError` (`record["id"]` against a list of message dicts), confirmed by
    executing this fixture during authoring. Derives token ids from the messages' own
    content by SHA-256, deterministic and process-stable, fitting `cfg.context_cap` and
    needing no real tokenizer."""
    def tokenize_prompt(messages):
        blob = json.dumps(list(messages), sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(blob).digest()
        return [byte % cfg.vocab_size for byte in digest[:8]]

    return tokenize_prompt


def qwen3b_checkpoint():
    """The cached Qwen2.5-3B-Instruct snapshot, or None -- the sharded checkpoint this
    design exists for (T-1912 design §1/§3), mirroring `test_pipeline.py`'s
    `qwen_checkpoint()` at the 3B tier."""
    import os
    import pathlib

    # T-2152 (outside strike item 6): no private filesystem path fallback -- HF_HOME is the
    # standard HuggingFace cache location env var; an unset one means this opt-in test skips.
    for root in (os.environ.get("HF_HOME"),):
        if not root:
            continue
        found = sorted(pathlib.Path(root).glob(
            "hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*/config.json"))
        if found:
            return found[0].parent
    return None


def upstream_3b_required():
    """Skip cleanly when the real 3B checkpoint is not cached -- a `skip` here states this
    cell was not exercised, exactly as `test_pipeline.py`'s `upstream_required()` does."""
    checkpoint = qwen3b_checkpoint()
    if checkpoint is None:
        pytest.skip("Qwen2.5-3B-Instruct not cached; the real-checkpoint sharded cells need it")
    return checkpoint


# ==============================================================================
# Dimension 1 -- Lifetime and reuse
# ==============================================================================


def test_a_sharded_reader_parses_shard_headers_once_and_reuses_them_correctly(tmp_path):
    """The reader is constructed once and `.tensor()` is called on it repeatedly, in an
    order that touches the SECOND shard before the first and revisits a name already read
    -- ruling out an "only the first shard opened actually works" bug, and a "each call
    re-parses the header" bug that a small fixture would not expose as a performance
    problem but would still expose as a correctness one if the re-parse ever disagreed
    with the first.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    weight_map = self_consistent_sharded_checkpoint(checkpoint_dir, {
        "shard_a.safetensors": {"t.first": np.array([1.0, 2.0], dtype=np.float32)},
        "shard_b.safetensors": {"t.second": np.array([3.0, 4.0, 5.0], dtype=np.float32)},
    })
    assert weight_map == {"t.first": "shard_a.safetensors", "t.second": "shard_b.safetensors"}

    open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
    reader = open_checkpoint_tensors(checkpoint_dir)
    order = ["t.second", "t.second", "t.first", "t.first", "t.second"]
    expected = {"t.first": np.array([1.0, 2.0]), "t.second": np.array([3.0, 4.0, 5.0])}
    for name in order:
        got = np.asarray(reader.tensor(name), dtype=np.float64)
        assert np.array_equal(got, expected[name]), (
            f"{name}: read after touching the other shard first disagreed with a fresh read"
        )


# ==============================================================================
# Dimension 2 -- Trust boundaries and hostile inputs (§4.4's six modes)
#
# Modes 1-5 share one baseline: a self-consistent, valid two-shard checkpoint. Each mode's
# test applies exactly ONE mutation to that baseline and confirms exactly that mode's
# rejection fires, by message content, not merely exception type -- and never by calling
# `.tensor()`, which is this suite's discharge of dimension 5 (every failure fires from
# `_ShardedSafeTensors.__init__` itself: a lazily-checking implementation would let
# construction SUCCEED on these fixtures, and `pytest.raises` would report "DID NOT RAISE"
# rather than passing). The same one-mutation-from-valid shape is dimension 11's guard-
# vitality requirement, so each mode test below discharges both dimensions at its own
# citation rather than being duplicated.
# ==============================================================================


def _valid_two_shard_baseline(tmp_path):
    """Two shards, mutually self-consistent: shard A carries `a.weight`/`b.weight`, shard B
    carries `c.weight`. Constructing `_ShardedSafeTensors` over this must succeed --
    baseline for every mode 1-5 single-mutation test below."""
    checkpoint_dir = tmp_path / "checkpoint"
    values = {
        "a.weight": np.array([1.0, 2.0], dtype=np.float32),
        "b.weight": np.array([3.0], dtype=np.float32),
        "c.weight": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }
    weight_map = self_consistent_sharded_checkpoint(checkpoint_dir, {
        "shard_a.safetensors": {"a.weight": values["a.weight"], "b.weight": values["b.weight"]},
        "shard_b.safetensors": {"c.weight": values["c.weight"]},
    })
    return checkpoint_dir, weight_map, values


def test_the_valid_two_shard_baseline_itself_constructs_cleanly(tmp_path):
    """Not a mode cell: proves the baseline every mode 1-5 test mutates is itself accepted,
    so a mode test's failure is attributable to its own single mutation, never to a
    baseline that was already broken."""
    checkpoint_dir, weight_map, values = _valid_two_shard_baseline(tmp_path)
    sharded_safe_tensors = api(MODULE, "_ShardedSafeTensors")
    reader = sharded_safe_tensors(checkpoint_dir, checkpoint_dir / "model.safetensors.index.json")
    assert reader.keys() == set(weight_map)
    for name, expected in values.items():
        assert np.array_equal(np.asarray(reader.tensor(name), dtype=np.float64),
                              expected.astype(np.float64))


def test_mode_1_a_shard_the_index_names_is_absent_from_disk(tmp_path):
    """§4.4 mode 1: `weight_map` names a shard file the checkpoint directory does not
    contain. Single mutation: delete `shard_b.safetensors` after the baseline writes it;
    the index still claims it."""
    checkpoint_dir, weight_map, _values = _valid_two_shard_baseline(tmp_path)
    (checkpoint_dir / "shard_b.safetensors").unlink()

    sharded_safe_tensors, config_error = api(MODULE, "_ShardedSafeTensors", "ConfigError")
    with pytest.raises(config_error, match="shard_b.safetensors") as excinfo:
        sharded_safe_tensors(checkpoint_dir, checkpoint_dir / "model.safetensors.index.json")
    assert str(checkpoint_dir) in str(excinfo.value)


def test_mode_2_a_shard_header_carries_a_tensor_the_index_does_not_name(tmp_path):
    """§4.4 mode 2: a shard's own on-disk header lists a tensor `weight_map` never names.
    Single mutation: shard A gains an extra header entry, `unmapped.weight`, whose declared
    payload range extends past the file's actual bytes -- so a correct implementation must
    detect this from the header comparison alone, never by trying to read the (nonexistent)
    payload first. `write_safetensors_shard` appends bytes for every tensor passed to it, so
    to get a header entry with NO matching payload the file is hand-assembled below rather
    than through that helper.
    """
    checkpoint_dir, weight_map, values = _valid_two_shard_baseline(tmp_path)

    # Rewrite shard A's header to add "unmapped.weight" with a payload range past EOF,
    # without appending any bytes for it -- the shard's real payload is untouched.
    shard_a_path = checkpoint_dir / "shard_a.safetensors"
    with open(shard_a_path, "rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
        payload = handle.read()
    real_end = header["b.weight"]["data_offsets"][1]
    header["unmapped.weight"] = {"dtype": "F32", "shape": [1],
                                 "data_offsets": [real_end, real_end + 4]}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(shard_a_path, "wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(payload)          # unchanged -- "unmapped.weight"'s 4 bytes never written

    sharded_safe_tensors, config_error = api(MODULE, "_ShardedSafeTensors", "ConfigError")
    with pytest.raises(config_error, match="unmapped.weight") as excinfo:
        sharded_safe_tensors(checkpoint_dir, checkpoint_dir / "model.safetensors.index.json")
    assert "shard_a.safetensors" in str(excinfo.value)


def test_mode_3_the_index_names_a_tensor_no_shard_header_contains(tmp_path):
    """§4.4 mode 3: `weight_map` names a tensor absent from every shard's own header --
    the opposite direction from mode 2. Single mutation: add `phantom.weight -> shard_a`
    to the index, written to a shard file that never gained that header entry."""
    checkpoint_dir, weight_map, _values = _valid_two_shard_baseline(tmp_path)
    weight_map["phantom.weight"] = "shard_a.safetensors"
    write_index_json(checkpoint_dir / "model.safetensors.index.json", weight_map)

    sharded_safe_tensors, config_error = api(MODULE, "_ShardedSafeTensors", "ConfigError")
    with pytest.raises(config_error, match="phantom.weight"):
        sharded_safe_tensors(checkpoint_dir, checkpoint_dir / "model.safetensors.index.json")


def test_mode_4_the_index_attributes_a_tensor_to_the_wrong_shard(tmp_path):
    """§4.4 mode 4: `weight_map` claims `b.weight` lives in shard B; it actually lives only
    in shard A's own header (the baseline never moved it). A naive implementation that
    trusts `weight_map` for routing without cross-checking each shard's own header would
    pass this silently -- the whole reason `owner` (§4.3) is built from the shards'
    headers directly."""
    checkpoint_dir, weight_map, _values = _valid_two_shard_baseline(tmp_path)
    weight_map["b.weight"] = "shard_b.safetensors"          # falsely claimed; really in A
    write_index_json(checkpoint_dir / "model.safetensors.index.json", weight_map)

    sharded_safe_tensors, config_error = api(MODULE, "_ShardedSafeTensors", "ConfigError")
    with pytest.raises(config_error, match="b.weight") as excinfo:
        sharded_safe_tensors(checkpoint_dir, checkpoint_dir / "model.safetensors.index.json")
    message = str(excinfo.value)
    assert "shard_b.safetensors" in message           # the claimed shard
    assert "shard_a.safetensors" in message           # the shard it was actually found in


def test_mode_5_the_same_tensor_name_appears_in_two_shard_headers(tmp_path):
    """§4.4 mode 5, and dimension 7's contract claim in the same fixture: a duplicate
    tensor across shards is rejected, never silently resolved by dict-overwrite ordering.
    Single mutation: `b.weight` is added to shard B's own header TOO, with DIFFERENT bytes
    than shard A's copy -- a naive implementation merging `{**header_a, **header_b}` (last-
    write-wins) would silently accept this and read shard B's copy, which is exactly the
    "plausible-looking pass" this fixture is built to catch: the test fails if the reader
    accepts it under EITHER copy, not just if it crashes.

    The index itself is untouched and still names only ONE shard for `b.weight` -- so this
    fires even when `weight_map` is perfectly self-consistent with itself, proving the
    duplicate check is not merely inherited from mode 2/3/4's index-vs-header cross-checks.
    """
    checkpoint_dir, weight_map, values = _valid_two_shard_baseline(tmp_path)

    shard_b_path = checkpoint_dir / "shard_b.safetensors"
    different_bytes = values["b.weight"] + 100.0          # decisively different, not a near-tie
    with open(shard_b_path, "rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
        payload = bytearray(handle.read())
    tail = header["c.weight"]["data_offsets"][1]
    data = different_bytes.tobytes()
    header["b.weight"] = {"dtype": "F32", "shape": [1], "data_offsets": [tail, tail + len(data)]}
    payload += data
    header_bytes = json.dumps(header).encode("utf-8")
    with open(shard_b_path, "wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(bytes(payload))

    sharded_safe_tensors, config_error = api(MODULE, "_ShardedSafeTensors", "ConfigError")
    with pytest.raises(config_error, match="b.weight") as excinfo:
        sharded_safe_tensors(checkpoint_dir, checkpoint_dir / "model.safetensors.index.json")
    message = str(excinfo.value)
    assert "shard_a.safetensors" in message
    assert "shard_b.safetensors" in message


class TestDimension2Mode0:
    """§4.4 mode 0: a `weight_map` shard-name value is unsafe -- not a `str`, or not a
    single bare path component. Tested directly against `_require_safe_shard_filename`
    (cheap, precise message assertions) and, separately, against the full
    `_ShardedSafeTensors` constructor over the real checkpoint and a portable symlink farm
    (the T-1917 strike's own fracture shape, §4.4/§7 dimension 2, D-SLM2552/2565).
    """

    HOSTILE_SHAPES = [
        "../evil.safetensors", "sub/evil.safetensors", "D:evil.safetensors",
        "C:\\Windows\\evil.safetensors", "\\\\server\\share\\evil.safetensors",
        ".", "..", "",
    ]

    def test_every_hostile_shape_is_rejected_by_its_own_declared_text(self, tmp_path):
        require_safe_shard_filename, config_error = api(
            MODULE, "_require_safe_shard_filename", "ConfigError")
        index_path = tmp_path / "model.safetensors.index.json"
        for shard_name in self.HOSTILE_SHAPES:
            with pytest.raises(config_error) as excinfo:
                require_safe_shard_filename(shard_name, tmp_path, index_path)
            message = str(excinfo.value)
            assert repr(shard_name) in message or shard_name in message, (
                f"{shard_name!r}: rejection message does not name the offending value: "
                f"{message!r}"
            )
            assert str(index_path) in message

    def test_a_non_string_weight_map_value_is_rejected_not_a_typeerror(self, tmp_path):
        """The type-confusion half of the original T-1914 finding: an un-typed JSON value
        (a number, `None`) must raise `ConfigError`, never bubble up as an unguarded
        `TypeError` from an `in`/comparison check."""
        require_safe_shard_filename, config_error = api(
            MODULE, "_require_safe_shard_filename", "ConfigError")
        index_path = tmp_path / "model.safetensors.index.json"
        for shard_name in (7, None):
            with pytest.raises(config_error):
                require_safe_shard_filename(shard_name, tmp_path, index_path)

    def test_real_shard_names_from_the_sharding_convention_are_accepted(self, tmp_path):
        require_safe_shard_filename = api(MODULE, "_require_safe_shard_filename")
        index_path = tmp_path / "model.safetensors.index.json"
        for shard_name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            require_safe_shard_filename(shard_name, tmp_path, index_path)   # must not raise

    def test_a_drive_relative_name_is_rejected_unconditionally_even_on_its_own_drive(self, tmp_path):
        """The deliberate T-1920 tightening (§4.4): `"D:evil.safetensors"` is rejected
        regardless of which drive the checkpoint happens to be mounted on -- the T-1916
        check's context-dependent same-drive exception is gone. `tmp_path` is used as the
        checkpoint directory whatever drive this host places it on, so this cell does not
        assume `D:` is the test-runner's own drive."""
        require_safe_shard_filename, config_error = api(
            MODULE, "_require_safe_shard_filename", "ConfigError")
        checkpoint_drive = str(tmp_path).split(":")[0] if ":" in str(tmp_path) else "D"
        drive_relative_name = f"{checkpoint_drive}:evil.safetensors"
        with pytest.raises(config_error):
            require_safe_shard_filename(drive_relative_name, tmp_path,
                                        tmp_path / "model.safetensors.index.json")

    @pytest.mark.upstream
    def test_the_real_qwen_3b_checkpoints_shard_names_construct_a_working_reader(self):
        """The cell that would have caught the T-1917 fracture directly: mode 0, enabled,
        against the real, symlinked Qwen2.5-3B-Instruct checkpoint. 434 tensors reachable,
        an exact bijection with `_upstream_names(cfg)`, and a cross-shard read (the second
        shard touched first) `np.array_equal` to a direct read of the same tensor from its
        named shard (T-1912 design §4.4 "Executed this fold")."""
        checkpoint = upstream_3b_required()
        pipeline = require(MODULE)

        cfg = pipeline.load_config(checkpoint / "config.json")
        names = pipeline._upstream_names(cfg)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]

        assert set(weight_map) == set(names), (
            "the real checkpoint's weight_map is not an exact bijection with "
            "_upstream_names(cfg); this cell's own premise (434 names, confirmed at design "
            "time) no longer holds"
        )

        open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
        reader = open_checkpoint_tensors(checkpoint)
        assert reader.keys() == set(weight_map)

        # Touch the SECOND shard's tensor first, then cross-check against a direct,
        # independent read of the same tensor from its own named shard file.
        by_shard = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        shard_names_sorted = sorted(by_shard)
        second_shard_tensor = by_shard[shard_names_sorted[-1]][0]
        first_shard_tensor = by_shard[shard_names_sorted[0]][0]
        for name in (second_shard_tensor, first_shard_tensor):
            direct = pipeline._SafeTensors(checkpoint / weight_map[name]).tensor(name)
            got = reader.tensor(name)
            assert np.array_equal(np.asarray(got), np.asarray(direct)), (
                f"{name}: sharded read disagreed with a direct read of its own named shard"
            )

    def test_a_synthetic_portable_symlink_farm_is_accepted_regardless_of_where_it_resolves(
        self, tmp_path
    ):
        """The Coverage Model gap the T-1917 strike found structural (§7 dimension 2,
        D-SLM2565): every prior mode-0 fixture was written into `tmp_path`, which contains
        no symlinks, so no cell could exhibit the real checkpoint's actual failure shape.
        This cell reproduces the HF hub cache's own `snapshots/`/`blobs/` layout with real
        `os.symlink`s and requires BOTH bare shard names to be accepted regardless of where
        they resolve on disk -- a resolve-then-compare check (the T-1916 fold's own,
        superseded mechanism) rejects both; the required, current mechanism must not."""
        values = {"only.weight": np.array([9.0, 8.0, 7.0], dtype=np.float32)}
        checkpoint_dir, weight_map = symlinked_shard_farm(
            tmp_path, {"model-00001-of-00001.safetensors": values})
        assert (checkpoint_dir / "model-00001-of-00001.safetensors").is_symlink(), (
            "fixture defect: the shard is not actually a symlink, so this cell cannot "
            "exhibit the fracture shape it exists to reproduce"
        )

        open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
        reader = open_checkpoint_tensors(checkpoint_dir)
        got = np.asarray(reader.tensor("only.weight"), dtype=np.float64)
        assert np.array_equal(got, values["only.weight"].astype(np.float64))


# ==============================================================================
# Dimension 4 -- Shape and platform matrices
# ==============================================================================


def _generic_n_shard_checkpoint(tmp_path, n):
    """N shards, one tensor each, self-consistent -- proves the resolver is generic over N
    rather than hardcoded to 2, per design §7 dimension 4's own framing: "a hardcoded-to-2
    regression is exactly the class of bug this dimension exists to catch"."""
    checkpoint_dir = tmp_path / f"checkpoint_n{n}"
    shard_contents = {
        f"shard{i}.safetensors": {f"t{i}.weight": np.array([float(i), float(i) + 0.5],
                                                            dtype=np.float32)}
        for i in range(n)
    }
    weight_map = self_consistent_sharded_checkpoint(checkpoint_dir, shard_contents)
    return checkpoint_dir, weight_map, shard_contents


@pytest.mark.parametrize("n", [1, 2, 4])
def test_the_resolver_is_generic_over_shard_count(tmp_path, n):
    """N=1 (legal, unusual -- an index-carrying checkpoint with exactly one shard file),
    the real N=2 shape, and a synthetic N=4+ case all resolve through the SAME sharded
    code path and read back correctly -- proving genericity over N, not merely that N=2
    happens to work."""
    checkpoint_dir, weight_map, shard_contents = _generic_n_shard_checkpoint(tmp_path, n)

    open_checkpoint_tensors, sharded_safe_tensors = api(
        MODULE, "_open_checkpoint_tensors", "_ShardedSafeTensors")
    reader = open_checkpoint_tensors(checkpoint_dir)
    assert isinstance(reader, sharded_safe_tensors), (
        f"N={n}: an index-carrying checkpoint did not resolve through the sharded reader"
    )
    for shard_name, tensors in shard_contents.items():
        for name, expected in tensors.items():
            got = np.asarray(reader.tensor(name), dtype=np.float64)
            assert np.array_equal(got, expected.astype(np.float64))


def test_dtype_dispatch_is_per_tensor_and_unaffected_by_which_shard_it_lives_in(tmp_path):
    """A two-shard checkpoint with a BF16 tensor in one shard and an F32 tensor in the
    other -- per-tensor dtype dispatch must be correct regardless of shard placement, not
    a property of "the first shard opened" or a dtype assumed uniform across the
    checkpoint."""
    checkpoint_dir = tmp_path / "checkpoint"
    bf16_values = bf16_clean(np.array([1.5, -2.25, 0.0], dtype=np.float32))
    f32_values = np.array([10.0, -20.5, 30.25], dtype=np.float32)
    self_consistent_sharded_checkpoint(
        checkpoint_dir,
        {"shard_bf16.safetensors": {"bf16.weight": bf16_values},
         "shard_f32.safetensors": {"f32.weight": f32_values}},
        dtypes={"bf16.weight": "BF16"},
    )

    open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
    reader = open_checkpoint_tensors(checkpoint_dir)
    got_bf16 = np.asarray(reader.tensor("bf16.weight"), dtype=np.float64)
    got_f32 = np.asarray(reader.tensor("f32.weight"), dtype=np.float64)
    assert np.array_equal(got_bf16, bf16_values.astype(np.float64)), (
        "BF16 widening through the sharded path is not exact"
    )
    assert np.array_equal(got_f32, f32_values.astype(np.float64))


# ==============================================================================
# Dimension 6 -- Numerical edges and determinism
# ==============================================================================


def test_single_file_and_sharded_storage_of_the_same_bytes_produce_array_equal_reads(tmp_path):
    """The core guarantee: the SAME tensor bytes, stored once as a single-file checkpoint
    and once split across two shards -- including a BF16 tensor specifically, the exact
    no-rounding widening being the property most worth breaking silently -- must produce
    `np.array_equal` (not `np.allclose`) results through both paths."""
    tensors = {
        "embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "norm.weight": bf16_clean(np.array([1.0, -1.0, 0.5, 2.5], dtype=np.float32)),
        "proj.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }
    dtypes = {"norm.weight": "BF16"}

    single_dir = tmp_path / "single"
    single_dir.mkdir()
    write_safetensors_shard(single_dir / "model.safetensors", tensors, dtypes=dtypes)

    sharded_dir = tmp_path / "sharded"
    self_consistent_sharded_checkpoint(
        sharded_dir,
        {"shard_a.safetensors": {"embed.weight": tensors["embed.weight"]},
         "shard_b.safetensors": {"norm.weight": tensors["norm.weight"],
                                 "proj.weight": tensors["proj.weight"]}},
        dtypes=dtypes,
    )

    open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
    single_reader = open_checkpoint_tensors(single_dir)
    sharded_reader = open_checkpoint_tensors(sharded_dir)
    for name in tensors:
        single_value = np.asarray(single_reader.tensor(name), dtype=np.float64)
        sharded_value = np.asarray(sharded_reader.tensor(name), dtype=np.float64)
        assert np.array_equal(single_value, sharded_value), (
            f"{name}: single-file and sharded storage of identical bytes disagreed"
        )


@pytest.mark.upstream
def test_real_3b_checkpoint_routing_is_array_equal_to_a_direct_shard_read():
    """Dimension 6's real-checkpoint instance (T-1916 fold; NOT §5's gate, which is scoped
    to the unsharded 1.5B path only -- design §5 "Scope note"). For a representative
    sample of `_upstream_names(cfg)`, `_open_checkpoint_tensors(checkpoint).tensor(name)`
    is `np.array_equal` to an independent direct read of the specific shard the real
    `weight_map` names for that tensor, read WITHOUT going through `_ShardedSafeTensors`'s
    own routing logic -- proving routing and dtype dispatch are correct on the real
    checkpoint's real shard boundaries. Raw header-parse-plus-memmap only: no calibration,
    no GPU (T-1912 design §8, checked tree-wide by the T-1916 fold).

    **T-1932 note.** This cell samples roughly 20 of the checkpoint's 434 tensors by
    stride -- cheap enough to run in every CI pass, but a routing defect on any of the
    other ~414 names is invisible to it. `test_all_434_real_3b_tensors_route_to_an_array_
    equal_direct_shard_read` below makes the identical comparison over every name; this
    cell is kept alongside it as the fast, always-on spot check, not superseded by it.
    """
    checkpoint = upstream_3b_required()
    pipeline = require(MODULE)

    cfg = pipeline.load_config(checkpoint / "config.json")
    names = pipeline._upstream_names(cfg)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    sample = sorted(names)[::max(1, len(names) // 20)]           # ~20 names, deterministic

    open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
    reader = open_checkpoint_tensors(checkpoint)
    for name in sample:
        shard_file = weight_map[name]
        direct = pipeline._SafeTensors(checkpoint / shard_file).tensor(name)
        routed = reader.tensor(name)
        assert np.array_equal(np.asarray(routed), np.asarray(direct)), (
            f"{name}: routed through _ShardedSafeTensors disagrees with a direct, "
            f"independent read of its own named shard {shard_file}"
        )


@pytest.mark.upstream
def test_all_434_real_3b_tensors_route_to_an_array_equal_direct_shard_read():
    """T-1932: closes the sampling gap in the cell above. That cell checks a stride sample
    of ~20 of the checkpoint's 434 tensors; a cross-shard routing defect on any of the
    other ~414 names is invisible to it, and to every other tier -- no synthetic cell
    reaches the real checkpoint's real shard boundaries (this project's synthetic
    numeric-edges cell, dimension 6, is exhaustive over its own small fixture, not this
    checkpoint's real one). This cell makes the identical comparison the sampled cell
    makes -- `_open_checkpoint_tensors(checkpoint).tensor(name)` `np.array_equal` to an
    independent direct read of the shard `weight_map` names for that tensor -- over every
    one of `_upstream_names(cfg)`'s names, not a stride.

    **Deliberately does NOT call `load_model`.** `load_model` runs a full calibration pass
    over the 600-record corpus after resolving tensors -- ~213 minutes end to end for this
    checkpoint (D-SLM2591) -- which is exactly the cost that made a full-coverage cell too
    expensive to write in the first place. This cell only ever calls
    `_open_checkpoint_tensors`/`_SafeTensors.tensor()`, both pure CPU/numpy memmap reads
    with no framework cast and no forward pass (T-1912 design §8) -- the same
    header-parse-plus-memmap shape T-1911's disposable merge tool already proved cheap
    across this exact checkpoint's all 434 tensors in one pass, with no GPU and no
    calibration (D-SLM2526).

    Each shard's `_SafeTensors` reader is constructed once and reused across every tensor
    it owns (`direct_readers`, keyed by shard filename) rather than once per tensor, so the
    two real shard headers are parsed twice total rather than 434 times -- a construction-
    cost optimization only; each tensor's actual comparison is still an independent
    `.tensor(name)` memmap read, exactly as the sampled cell above performs it.
    """
    checkpoint = upstream_3b_required()
    pipeline = require(MODULE)

    cfg = pipeline.load_config(checkpoint / "config.json")
    names = pipeline._upstream_names(cfg)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert set(weight_map) == set(names), (
        "the real checkpoint's weight_map is not an exact bijection with "
        "_upstream_names(cfg); this cell's own premise (434 names) no longer holds"
    )

    open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
    reader = open_checkpoint_tensors(checkpoint)

    direct_readers = {}
    mismatched = []
    compared = 0
    for name in sorted(names):
        shard_file = weight_map[name]
        if shard_file not in direct_readers:
            direct_readers[shard_file] = pipeline._SafeTensors(checkpoint / shard_file)
        direct = direct_readers[shard_file].tensor(name)
        routed = reader.tensor(name)
        if not np.array_equal(np.asarray(routed), np.asarray(direct)):
            mismatched.append(name)
        compared += 1

    assert compared == len(names) == 434, (
        f"compared {compared} names against {len(names)} predicted by _upstream_names(cfg) "
        f"(434 expected for Qwen2.5-3B-Instruct) -- the loop did not cover the full set"
    )
    assert not mismatched, (
        f"{len(mismatched)} of {compared} names disagreed between the routed read and a "
        f"direct read of their own named shard: {mismatched[:10]}"
        f"{'...' if len(mismatched) > 10 else ''}"
    )


# ==============================================================================
# Dimension 7 -- Contract claims
# ==============================================================================


def test_sharding_signal_is_the_indexs_presence_not_the_file_count(tmp_path):
    """A synthetic checkpoint with exactly ONE `*.safetensors` file AND an
    `index.json` must still resolve through the SHARDED path -- proving the design does
    not silently special-case N=1 back to the old single-file branch (which would bypass
    every one of §4.4's six consistency checks)."""
    checkpoint_dir = tmp_path / "checkpoint"
    self_consistent_sharded_checkpoint(checkpoint_dir, {
        "solo.safetensors": {"only.weight": np.array([1.0], dtype=np.float32)},
    })
    assert len(list(checkpoint_dir.glob("*.safetensors"))) == 1

    open_checkpoint_tensors, sharded_safe_tensors = api(
        MODULE, "_open_checkpoint_tensors", "_ShardedSafeTensors")
    reader = open_checkpoint_tensors(checkpoint_dir)
    assert isinstance(reader, sharded_safe_tensors), (
        "a one-shard-plus-index checkpoint fell back to the single-file reader instead of "
        "the sharded path"
    )


# (The duplicate-tensor contract claim -- "a duplicate across shards is always rejected,
# never silently resolved by dict-overwrite ordering" -- is discharged by
# test_mode_5_the_same_tensor_name_appears_in_two_shard_headers above, whose fixture
# already uses DIFFERENT bytes per copy specifically so a last-write-wins merge cannot
# pass by accident; see that test's docstring.)
#
# ("_SafeTensors and _checkpoint_tensor_file are not modified -- zero lines" is a source-
# diff obligation stated in T-1912 design §5, not a pytest cell -- design §7 dimension 7's
# own text confirms this citation "gains no new obligation from being listed here a second
# time." No cell is authored for it; a build-time diff check discharges it.)


# ==============================================================================
# Dimension 8 -- Composition
# ==============================================================================


def test_load_model_rejects_an_unmapped_tensor_on_a_sharded_checkpoint(tmp_path):
    """The sharded sibling T-1918 (D-SLM2543) did not build (design §12.3, D-SLM2567):
    T-1918 fixed the single-file shape of this same pre-existing defect --
    `test_load_model_rejects_an_unmapped_tensor` passing on `load_config`'s missing-
    `config.json` rejection three calls before the check it is named for. The sharded
    equivalent is a checkpoint recognized as sharded (an `index.json` present, even with
    an empty `weight_map` -- vacuously satisfying every one of §4.4's six modes, exactly
    mirroring T-1918's own minimal empty-`{}`-safetensors-file fixture for the single-file
    case) plus a real `config.json`, so `load_model` reaches the unmapped-tensor check via
    the SHARDED resolution path specifically."""
    load_model, unsupported = api(MODULE, "load_model", "UnsupportedOpSet")
    _write_tiny_config_json(tmp_path, _tiny_cfg(require(MODULE)))
    write_index_json(tmp_path / "model.safetensors.index.json", {})   # sharded, vacuously valid

    with pytest.raises(unsupported, match="mystery_proj"):
        load_model(tmp_path, extra_tensors={"model.layers.0.self_attn.mystery_proj.weight": None})


def test_load_model_rejects_a_missing_tensor_on_a_sharded_checkpoint(tmp_path):
    """The `require_tensors` mirror of the cell above (design §12.3's second owed cell):
    every real tensor `_upstream_names(cfg)` demands is declared present via
    `extra_tensors` (so `unmapped` cannot fire and mask this check), and exactly ONE
    fictional `require_tensors` entry is left absent -- proving the missing-tensor
    rejection fires correctly when the checkpoint underlying `tensors.keys()` is the
    sharded reader, and that its message names the specific absent tensor, not some other
    coincidentally-absent one."""
    load_model = api(MODULE, "load_model")
    pipeline = require(MODULE)
    cfg = _tiny_cfg(pipeline)
    _write_tiny_config_json(tmp_path, cfg)
    write_index_json(tmp_path / "model.safetensors.index.json", {})   # sharded, vacuously valid
    names = pipeline._upstream_names(cfg)

    with pytest.raises(KeyError, match="phantom_proj"):
        load_model(
            tmp_path,
            extra_tensors={n: None for n in names},
            require_tensors=("model.layers.0.self_attn.phantom_proj.weight",),
        )


def test_qk_projection_placed_in_different_shards_load_to_the_same_model_as_together(tmp_path):
    """Cross with the RoPE pair permutation (`_permuted_if_rope`, design §7 dimension 8):
    the SAME tensor values, loaded once with `q_proj`/`k_proj` for layer 0 in the SAME
    shard as everything else and once with them split into their OWN separate shards, must
    produce IDENTICAL loaded `QuantizedModel.weights` (post-permutation, post-quantization)
    -- proving per-tensor shard dispatch and the permutation logic compose correctly
    regardless of shard boundary placement. Runs `load_model` fully end to end on a tiny
    config with a supplied `tokenize_prompt` (`deterministic_tokenize_prompt`), so no real
    tokenizer or checkpoint download is needed and calibration completes over the full
    600-record corpus in well under a second.
    """
    load_model = api(MODULE, "load_model")
    pipeline = require(MODULE)
    cfg = _tiny_cfg(pipeline)
    tokenize_prompt = deterministic_tokenize_prompt(cfg)

    together_dir = tmp_path / "together"
    together_dir.mkdir()
    _write_full_sharded_checkpoint(together_dir, pipeline, cfg, shard_of=lambda _n: "all.safetensors")

    split_dir = tmp_path / "split"
    split_dir.mkdir()

    def shard_of(upstream_name):
        if upstream_name.endswith("self_attn.q_proj.weight"):
            return "q_shard.safetensors"
        if upstream_name.endswith("self_attn.k_proj.weight"):
            return "k_shard.safetensors"
        return "rest.safetensors"

    _write_full_sharded_checkpoint(split_dir, pipeline, cfg, shard_of=shard_of)

    model_together = load_model(together_dir, tokenize_prompt=tokenize_prompt)
    model_split = load_model(split_dir, tokenize_prompt=tokenize_prompt)

    assert set(model_together.weights) == set(model_split.weights)
    for name in model_together.weights:
        together_codes = np.asarray(model_together.weights[name])
        split_codes = np.asarray(model_split.weights[name])
        assert np.array_equal(together_codes, split_codes), (
            f"{name}: loaded weight codes differ depending on whether q_proj/k_proj shared "
            f"a shard with everything else or were split into their own shards"
        )
    assert model_together.weight_scales == model_split.weight_scales


# ==============================================================================
# Dimension 9 -- Persistence round-trip and version evolution
# ==============================================================================


def _model_pointed_at_a_sharded_checkpoint(pipeline, cfg, checkpoint_dir):
    """`fixture_model(cfg)` with its float_source swapped for a real reader over a
    SHARDED checkpoint -- the sharded sibling of `test_artifact_cache.py`'s own
    `_model_pointed_at_checkpoint`, which only ever built the single-file case."""
    import dataclasses

    model = pipeline.fixture_model(cfg)
    names = pipeline._upstream_names(cfg)
    open_checkpoint_tensors = api(MODULE, "_open_checkpoint_tensors")
    tensors = open_checkpoint_tensors(checkpoint_dir)
    float_source = pipeline._CheckpointFloatSource(tensors, names, cfg)
    return dataclasses.replace(model, float_source=float_source)


def test_reopening_a_saved_artifacts_sharded_checkpoint_is_array_equal_to_the_original(tmp_path):
    """`artifact_cache.save_artifact`/`load_artifact` round-trips a SHARDED checkpoint's
    float source: the reopened reader's tensor reads are `np.array_equal` to the same
    tensors read directly at original `load_model` time -- the comparison oracle design
    §7 dimension 9 names explicitly (mirroring §5's gate), applied to the sharded case
    `_reopen_checkpoint_float_source` does not yet handle (it still calls
    `pipeline._checkpoint_tensor_file` directly, unconditionally -- design §4.5 wires it
    to `_open_checkpoint_tensors` instead)."""
    save_artifact, load_artifact = api(ARTIFACT_MODULE, "save_artifact", "load_artifact")
    pipeline = require(MODULE)
    cfg = _tiny_cfg(pipeline)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    embed_shape = (cfg.vocab_size, cfg.hidden_size)
    embed_values = np.arange(np.prod(embed_shape), dtype=np.float32).reshape(embed_shape)
    names = pipeline._upstream_names(cfg)
    embed_upstream = next(u for u, o in names.items() if o == "embed")
    self_consistent_sharded_checkpoint(checkpoint_dir, {
        "shard_a.safetensors": {embed_upstream: embed_values},
        "shard_b.safetensors": {},
    })

    model = _model_pointed_at_a_sharded_checkpoint(pipeline, cfg, checkpoint_dir)
    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path, checkpoint_path=str(checkpoint_dir))

    loaded = load_artifact(artifact_path)
    reopened = np.asarray(loaded.float_weight("embed"), dtype=np.float64)
    assert np.array_equal(reopened, embed_values.astype(np.float64))


def test_a_partially_missing_sharded_checkpoint_degrades_to_the_stub_on_reload(tmp_path):
    """The existing moved/deleted-checkpoint fork (`artifact_cache.py:400-416`) must catch
    a PARTIALLY-missing sharded checkpoint (one shard file deleted since save, not the
    whole directory) and degrade to the informative stub rather than propagating unhandled
    -- design §7 dimension 9's second obligation."""
    save_artifact, load_artifact = api(ARTIFACT_MODULE, "save_artifact", "load_artifact")
    pipeline = require(MODULE)
    cfg = _tiny_cfg(pipeline)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    names = pipeline._upstream_names(cfg)
    embed_upstream = next(u for u, o in names.items() if o == "embed")
    embed_values = np.zeros((cfg.vocab_size, cfg.hidden_size), dtype=np.float32)
    self_consistent_sharded_checkpoint(checkpoint_dir, {
        "shard_a.safetensors": {embed_upstream: embed_values},
        "shard_b.safetensors": {},
    })
    model = _model_pointed_at_a_sharded_checkpoint(pipeline, cfg, checkpoint_dir)

    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path, checkpoint_path=str(checkpoint_dir))
    (checkpoint_dir / "shard_a.safetensors").unlink()          # partial deletion, not rmtree

    loaded = load_artifact(artifact_path)                       # must not raise on load
    with pytest.raises(RuntimeError):
        loaded.float_weight("embed")


def test_the_stub_message_names_the_real_cause_for_a_genuine_absence(tmp_path):
    """Design §7 dimension 9, tightened by fold T-1920 (D-SLM2566): the degrade-to-stub
    obligation is not satisfied by confirming a stub installs -- the stub's OWN message
    must contain the causing exception's text. Genuine absence: the whole checkpoint
    directory is gone by reload time."""
    save_artifact, load_artifact = api(ARTIFACT_MODULE, "save_artifact", "load_artifact")
    import shutil

    pipeline = require(MODULE)
    cfg = _tiny_cfg(pipeline)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    names = pipeline._upstream_names(cfg)
    embed_upstream = next(u for u, o in names.items() if o == "embed")
    self_consistent_sharded_checkpoint(checkpoint_dir, {
        "shard_a.safetensors": {embed_upstream: np.zeros((cfg.vocab_size, cfg.hidden_size),
                                                          dtype=np.float32)},
        "shard_b.safetensors": {},
    })
    model = _model_pointed_at_a_sharded_checkpoint(pipeline, cfg, checkpoint_dir)
    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path, checkpoint_path=str(checkpoint_dir))
    shutil.rmtree(checkpoint_dir)

    loaded = load_artifact(artifact_path)
    with pytest.raises(RuntimeError) as excinfo:
        loaded.float_weight("embed")
    message = str(excinfo.value)
    assert "safetensors" in message.lower(), (
        f"the stub's message does not appear to carry the causing exception's own text -- "
        f"a genuinely absent checkpoint's real cause names its missing weights file, not "
        f"just the generic degrade wrapper: {message!r}"
    )


def test_the_stub_message_names_the_real_cause_for_a_false_positive_rejection(tmp_path):
    """The exact D-SLM2553 shape (design §12.2, D-SLM2564): the checkpoint is PRESENT,
    valid, and readable, but this design's own construction-time check rejects it (here,
    simulated by pointing `checkpoint_path` at a directory whose index.json declares mode
    0's own kind of violation -- a shard name that is not a bare path component). The
    generic "moved, deleted, or no longer a valid checkpoint" message is dishonest for
    this case specifically -- there was nothing moved or deleted -- and the fold's fix
    folds the causing `ConfigError`'s own text into the stub's message."""
    save_artifact, load_artifact = api(ARTIFACT_MODULE, "save_artifact", "load_artifact")
    pipeline = require(MODULE)
    cfg = _tiny_cfg(pipeline)

    # A model NOT pointed at a real checkpoint (fixture_model's own dict-backed source) --
    # the artifact just needs to carry SOME checkpoint_path string; the real reopen at
    # load time is what must fail with a ConfigError-shaped, false-positive cause.
    model = pipeline.fixture_model(cfg)
    hostile_checkpoint_dir = tmp_path / "hostile_checkpoint"
    hostile_checkpoint_dir.mkdir()
    write_index_json(hostile_checkpoint_dir / "model.safetensors.index.json",
                     {"../evil.safetensors": "../evil.safetensors"})

    artifact_path = tmp_path / "artifact"
    save_artifact(model, artifact_path, checkpoint_path=str(hostile_checkpoint_dir))

    loaded = load_artifact(artifact_path)
    with pytest.raises(RuntimeError) as excinfo:
        loaded.float_weight("embed")
    message = str(excinfo.value)
    assert "evil.safetensors" in message or "bare" in message.lower(), (
        f"the stub reports the generic moved/deleted text with no trace of the real, "
        f"false-positive cause -- the fix (design §4.6) folds the causing ConfigError's "
        f"own text (mode 0's rejection names the offending shard value and the word "
        f"'bare') into the reason string: {message!r}"
    )


# ==============================================================================
# Dimension 10 -- Functional achievement
# ==============================================================================


@pytest.mark.upstream
def test_the_real_3b_checkpoint_converts_to_a_working_quantized_model():
    """The feature's achievement claim, end to end: a real, multi-shard HF checkpoint
    converts to a working `QuantizedModel`. `load_model` executed against the real
    Qwen2.5-3B-Instruct checkpoint, asserting `config.num_hidden_layers == 36` and every
    projection's weight array non-vacuously populated -- shape matches
    `_upstream_names(cfg)`'s prediction, neither all-zero nor all-NaN (design §7 dimension
    10, F4) -- not merely that no exception was raised. This IS the "real-checkpoint
    calibration" run T-1909 stopped short of (design §2), so it is gated `upstream` and
    left to run at full cost when this suite is exercised with `-m upstream`.

    **Measured cost (D-SLM2591, 2026-08-11, superseding D-SLM2582's wrong 62-minute figure
    and D-SLM2584's "pending" state): ~213 minutes wall-clock (~3.6 hours)** for a full 3B
    conversion including the 600-record calibration pass, confirmed reaching and passing this
    cell's own assertions -- executed as PID 30512, roughly 12.5 cores busy throughout,
    ~11.5 GB peak resident, GPU idle end to end (D-SLM2591's own provenance: a blocking
    process wait from the dispatching context, sampled at 92/182.6/190 minutes before exit).
    Pytest's own internal session timer for that run reports `12437.10s` (`3:27:17`, 207.3
    minutes) for the 3-cell upstream session as a whole -- this cell is the overwhelming
    majority of that time; the other two `@pytest.mark.upstream` cells in this file are raw
    header-parse-plus-memmap reads costing seconds. The two readings (207.3 min pytest-
    internal, ~213 min external process wall-clock) differ by interpreter startup and import
    time outside pytest's own timer, not by disagreement about the work. This sits close to,
    without beating, T-1909's ~5.83-hour single-core projection for the same operation
    (D-SLM2550) -- unlike D-SLM2582's 62-minute figure, which measured a run that died early
    against a still-landing implementation and never reached this cell's own work at all
    (D-SLM2584). Annotated in place so the next runner learns the real price from the test
    rather than from a stopwatch -- not a re-tiering or a re-marking."""
    checkpoint = upstream_3b_required()
    load_model = api(MODULE, "load_model")
    pipeline = require(MODULE)

    model = load_model(checkpoint)

    assert model.config.num_hidden_layers == 36
    names = pipeline._upstream_names(model.config)
    own_shapes = dict(pipeline._weight_shapes(model.config))
    for upstream, ours in names.items():
        if ours.endswith(".bias"):
            continue
        expected_shape = own_shapes[ours]
        weight = np.asarray(model.weights[ours])
        assert tuple(weight.shape) == tuple(expected_shape), (
            f"{ours}: loaded shape {weight.shape} != predicted {expected_shape}"
        )
        assert weight.any(), f"{ours}: loaded weight is all-zero"
        assert not np.isnan(weight.astype(np.float64)).any(), f"{ours}: loaded weight has NaN"


# ==============================================================================
# T-1928 -- mutation pins for three previously-invisible regressions
#
# The T-1922 build made every cell above pass. A confirmation pass
# (`Claude/Poirot/d5dd7ce507-shard-fix-confirmation.md`, finding C1) then reverted each of
# three production fixes to its literal pre-fix text -- one mutation at a time -- and ran
# every tier: this file's CI cells, the pre-existing regression suite, and the two cheap
# real-checkpoint upstream cells. Not one cell moved, in any tier, under any of the three
# reversions. The three cells below are that pass's own specification (its casebook §6.1),
# each proven a genuine tripwire during authoring by executing both directions -- reverted
# (red) and restored (green) -- against a validated scratch copy of this package, never
# against the live source tree (this suite is read-only on `pipeline.py`; the same
# discipline the confirmation pass itself used, per its own §9 "Ceiling"). The executed
# both-directions proof for each cell, with true counts, is in the test-design record:
# `Claude/Curie/t1928-shard-mutation-pins-2026-08-11.md`.
# ==============================================================================


_MODE1_DEVICE_PROBE_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, {tools_dir!r})
from reference_pipeline import pipeline

try:
    pipeline._ShardedSafeTensors(Path({checkpoint_dir!r}), Path({index_path!r}))
except pipeline.ConfigError as exc:
    print("MODE1_REJECTED:" + str(exc))
except Exception as exc:
    print("MODE1_WRONG_EXCEPTION:" + type(exc).__name__ + ":" + str(exc))
else:
    print("MODE1_ACCEPTED_NO_ERROR")
"""


class TestT1928MutationPins:
    """Three cells, one per finding in the confirmation casebook's §6.1: validation
    ordering (item 1, D-SLM2598-adjacent), `is_file()` vs `exists()` (item 2), and the
    non-UTF-8 index (item 3). Each targets the exact reverted text the casebook names."""

    def test_mode_0_via_constructor_catches_a_poisoned_value_mixed_with_valid_shard_names(
        self, tmp_path
    ):
        """Casebook §6.1 item 1 (T-1924 S1's own defect class, its third appearance --
        T-1914's Structural 1 / D-SLM2532, then T-1924's `sorted(set(...))` recurrence,
        now pinned). Drives a HETEROGENEOUS `weight_map` through `_ShardedSafeTensors`'s
        own constructor -- not the bare `_require_safe_shard_filename` unit the suite
        already exercises elsewhere in this file, which the pre-fix code called correctly
        once reached and so cannot see this class of regression at all.

        The realistic hostile shape, per the casebook's own framing: several VALID string
        shard names, one poisoned non-string value placed LAST in insertion order -- not an
        all-non-string map. An all-non-string `weight_map` is homogeneous and comparable, so
        `sorted(set(weight_map.values()))` never raises on it and the loop still reaches
        `_require_safe_shard_filename`'s own `isinstance` guard correctly EVEN under the
        pre-fix ordering -- which is exactly why the casebook's own table calls that shape
        already-passing and names the MIXTURE, not the monoculture, as the case the remedy
        actually changed. A mixture of `str` and a non-`str` hashable (here, an `int`) is
        what `sorted()` cannot compare, so if the fix's own line order (validate every raw
        value, THEN deduplicate and sort) is ever reverted to the pre-fix order (deduplicate
        and sort, THEN validate), this poisoned entry crashes `sorted()` with an unguarded
        `TypeError` before `_require_safe_shard_filename` is ever called on anything -- for
        the valid entries too, not only the poisoned one.

        Mutation-verified during authoring (test-design record, both directions).
        """
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        index_path = checkpoint_dir / "model.safetensors.index.json"
        weight_map = {
            "layer0.weight": "shard_a.safetensors",
            "layer1.weight": "shard_a.safetensors",
            "layer2.weight": "shard_b.safetensors",
            "layer3.weight": 7,          # poisoned: non-string, LAST in insertion order
        }
        write_index_json(index_path, weight_map)

        sharded_safe_tensors, config_error = api(MODULE, "_ShardedSafeTensors", "ConfigError")
        with pytest.raises(config_error) as excinfo:
            sharded_safe_tensors(checkpoint_dir, index_path)
        message = str(excinfo.value)
        assert "7" in message, (
            f"expected the poisoned value (7) named in the rejection message: {message!r}"
        )

    def test_mode_1_rejects_a_reserved_device_name_shard_without_hanging(self, tmp_path):
        """Casebook §6.1 item 2 (T-1924 S2). `is_file()` must reject a shard name that
        resolves but is not a regular file BEFORE `_SafeTensors.__init__` ever opens it --
        and the pre-fix `exists()` check is not merely wrong, it is a HANG. Built around
        what actually reproduces on THIS host, not the general claim: of the classic
        reserved device names, only `CON` and `NUL` resolve as devices here (`COM1`/`PRN`/
        `AUX`/`LPT1` do not), and of those two only `CON` hangs -- `NUL` returns EOF and
        raises a distinct, non-hanging error (casebook C5). Confirmed directly during
        authoring, outside pytest, under a bounded probe: `open("CON", "rb").read()` did
        not return inside a 10 s bound.

        **T-1931 round-2 confirmation, D6 (Observation, disposed rather than expanded into
        a new cell).** The hostile-device set on this host is wider than either this
        docstring or `pipeline.py`'s own enumeration names: `CONIN$` and `CONOUT$` (not
        members of the classic reserved-name family either docstring scopes to) also
        return `exists()=True`/`is_file()=False` here, and `CONIN$` also hangs on read
        (confirmed under an 8 s bounded subprocess probe during this round, same shape as
        `CON`). This is an enumeration-accuracy gap, not a coverage hole: `is_file()`
        rejects both by the same predicate `CON` exercises, not by name-matching a list, so
        the remedy already covers them and no additional cell is needed to close the gap --
        recording it here is the disposition.

        **T-1931 round-2 confirmation, D5 (Observation, disposed by an explicit skip).**
        This cell's discriminating power rests on `CON` resolving as a device on the host
        running it; where it does not, `exists()` and `is_file()` agree (both `False`) and
        BOTH the fixed and the reverted predicate raise `ConfigError` identically, so the
        cell would pass under a live regression rather than catch it -- a vacuous pass, not
        a swept one, but indistinguishable from either without saying so. The precondition
        is checked explicitly below and the cell skips, naming the reason, rather than
        passing silently when it cannot discriminate -- mirroring `symlinked_shard_farm`'s
        own skip-if-the-host-lacks-the-precondition convention elsewhere in this file. This
        keeps a "skip reads as swept" failure mode (`conftest.py`'s own module docstring)
        from applying here: what is skipped is not the check, it is the one host-shape this
        cell cannot exercise, and the skip message says exactly that.

        This is the ticket's own named hazard -- "one of these defects is a hang that
        consumes the machine with no diagnostic" -- so the construction runs in a SEPARATE
        PROCESS under an explicit timeout, not in-process. Today (fixed code) the
        subprocess returns in well under a second with a clean `ConfigError`. If this cell
        ever regresses, `subprocess.run`'s own timeout converts what would otherwise be an
        indefinite, undiagnosable wedge of the whole pytest run into a single bounded test
        FAILURE naming exactly what happened -- verified during authoring by reverting
        `is_file()` to `exists()` in a scratch copy and observing `TimeoutExpired` fire
        cleanly at the bound, with no orphaned process surviving it (test-design record).

        Mutation-verified during authoring (test-design record, both directions).
        """
        if not (Path("CON").exists() and not Path("CON").is_file()):
            pytest.skip(
                "this host's 'CON' does not resolve as exists()=True/is_file()=False, so "
                "the fixed and reverted predicates cannot be told apart here -- both would "
                "raise identically and this cell would pass vacuously under a live "
                "regression rather than catch it (T-1931 D5); skipping states that "
                "explicitly instead of passing silently"
            )
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        index_path = checkpoint_dir / "model.safetensors.index.json"
        write_index_json(index_path, {"only.weight": "CON"})

        script = _MODE1_DEVICE_PROBE_SCRIPT.format(
            tools_dir=str(TOOLS_DIR), checkpoint_dir=str(checkpoint_dir),
            index_path=str(index_path),
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                "mode 1 accepted the reserved device name 'CON' as a shard and the "
                "subsequent read blocked past a 20 s bound -- exists() lets a device name "
                "through where is_file() must reject it before _SafeTensors ever opens it",
                pytrace=False,
            )
        assert "MODE1_REJECTED" in proc.stdout, (
            f"expected a clean ConfigError rejection of the device name; "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r} returncode={proc.returncode}"
        )

    def test_load_index_json_rejects_a_non_utf8_index_body(self, tmp_path):
        """Casebook §6.1 item 3 (T-1924 M1). A `model.safetensors.index.json` whose bytes
        are not valid UTF-8 must raise `ConfigError` naming the encoding failure -- never a
        raw `UnicodeDecodeError`.

        **Corrected by T-1931 (round-2 confirmation, D2): the prior text here was false.**
        `UnicodeDecodeError` IS a `ValueError` (`issubclass(UnicodeDecodeError, ValueError)
        is True`, executed) and so it IS caught by
        `artifact_cache._reopen_or_stub_float_source`'s degrade net
        (`except (OSError, ValueError):`, `artifact_cache.py:411`) -- it does not propagate
        out of `load_artifact`. The true reason this cell matters is stronger than the false
        one: a raw `UnicodeDecodeError` is **silently absorbed** into the generic degrade
        stub and the artifact loads with no live float source and no visible sign that
        anything is wrong until a caller happens to touch `float_weight` -- the exact
        degrade-over-reject outcome `_ShardedSafeTensors`'s own class docstring cites §11
        to forbid. Confirmed end to end during authoring, against a scratch copy carrying
        this exact mutation with a real `ModelConfig` and a checkpoint directory holding a
        non-UTF-8 index: `_load_index_json` raised the raw `UnicodeDecodeError`,
        `_reopen_or_stub_float_source` caught it and returned the stub (did not propagate),
        and the stub's own message folded in the causing exception's text.

        Mutation-verified during authoring (test-design record, both directions).
        """
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        index_path = checkpoint_dir / "model.safetensors.index.json"
        index_path.write_bytes(b'{"weight_map": {"a": "\xff\xfe bad utf8"}}')

        load_index_json, config_error = api(MODULE, "_load_index_json", "ConfigError")
        with pytest.raises(config_error, match="(?i)utf-8"):
            load_index_json(index_path)
