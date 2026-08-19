"""Curie's red-then-green pin for T-2194 (D-SLM3697): `read_peft_lora_pair` dies on bf16 LoRAs.

bf16 is the prevailing PEFT/LoRA training default. Before this ticket's fix,
`read_peft_lora_pair` called `.astype(np.float64)` on the array `safetensors`' own
`framework="numpy"` binding returned for a bf16-dtyped tensor -- numpy has no bfloat16 dtype, and
that raised `TypeError: data type bfloat16 not understood`, reproduced against the reverted fix
(`git stash` the fix, run this module, restore) and recorded in the T-2194 casebook rather than
re-created as a permanent second code path in this suite. Green under the fix is what every cell
below proves.

The equivalence oracle (StandardsDocument.md §5.4 "a test exercises the real workload"): bf16 ->
float32 widening is LOSSLESS -- bf16 IS float32's top 16 bits, so the widening shift zero-fills the
dropped mantissa bits and no rounding occurs. A value already bf16-clean (its low 16 mantissa bits
zeroed) therefore reads back BIT FOR BIT identical whether it is stored BF16 (exercising the fixed
reader) or F32 (the path that already worked before this fix) -- `np.array_equal`, never
`np.allclose`, per the fixture's own `bf16_clean` construction (`_t2194_bf16_lora_fixture.py`).
"""

import json

import numpy as np

import sslm_convert_adapter as A
from _t2194_bf16_lora_fixture import (
    KEY_A,
    KEY_B,
    LORA_ALPHA,
    Q_WIDTH,
    RANK,
    TARGET_MODULES,
    build_bf16_lora_fixture,
    build_fp32_reference_fixture,
    write_adapter_config,
    write_safetensors_file,
)


def _meta():
    return A.AdapterMeta(rank=RANK, lora_alpha=LORA_ALPHA, use_rslora=False,
                         target_modules=tuple(TARGET_MODULES), scaling=1.0)


def test_read_peft_lora_pair_reads_genuine_bf16_without_raising(tmp_path):
    """The defect itself: a genuinely bf16-stored adapter must not raise
    `TypeError: data type bfloat16 not understood` -- it must return float64 arrays."""
    adapter_dir = build_bf16_lora_fixture(tmp_path / "adapter_bf16",
                                          base_model_name_or_path="unused", seed=7)
    a_f, b_f = A.read_peft_lora_pair(adapter_dir, layer=0, proj_short="q_proj", meta=_meta())
    assert a_f.dtype == np.float64
    assert b_f.dtype == np.float64
    assert a_f.shape == (RANK, Q_WIDTH)
    assert b_f.shape == (Q_WIDTH, RANK)


def test_bf16_read_matches_the_previously_working_f32_path_bit_for_bit(tmp_path):
    """Equivalence oracle: the SAME already-bf16-clean values, read once through a genuinely bf16
    file and once through an F32 file of the identical numbers (the path that already worked
    pre-fix), must produce identical float64 arrays."""
    bf16_dir = build_bf16_lora_fixture(tmp_path / "adapter_bf16",
                                       base_model_name_or_path="unused", seed=11)
    f32_dir = build_fp32_reference_fixture(tmp_path / "adapter_f32",
                                           base_model_name_or_path="unused", seed=11)
    meta = _meta()
    a_bf16, b_bf16 = A.read_peft_lora_pair(bf16_dir, layer=0, proj_short="q_proj", meta=meta)
    a_f32, b_f32 = A.read_peft_lora_pair(f32_dir, layer=0, proj_short="q_proj", meta=meta)

    assert np.array_equal(a_bf16, a_f32), (
        "bf16 widening diverged from the F32 reference read of the identical numbers (A)"
    )
    assert np.array_equal(b_bf16, b_f32), (
        "bf16 widening diverged from the F32 reference read of the identical numbers (B)"
    )
    # Fixture sanity: the arrays are genuinely nonzero and non-degenerate, so the comparison
    # above is not vacuously true against two all-zero arrays.
    assert np.any(a_bf16 != 0.0)
    assert np.any(b_bf16 != 0.0)


def test_bf16_read_matches_hand_widened_reference(tmp_path):
    """Independent of the F32 comparison above: the bf16 read also matches the widening formula
    computed directly from the fixture's own raw bf16 bytes (`view(uint16).astype(uint32) << 16`,
    viewed as float32) -- the SAME arithmetic `pipeline._SafeTensors.tensor()` performs, verified
    here against the fixture's own known input rather than by re-reading the code under test."""
    adapter_dir = build_bf16_lora_fixture(tmp_path / "adapter_bf16",
                                          base_model_name_or_path="unused", seed=3)
    a_f, b_f = A.read_peft_lora_pair(adapter_dir, layer=0, proj_short="q_proj", meta=_meta())

    raw_bytes = (adapter_dir / "adapter_model.safetensors").read_bytes()
    header_len = int.from_bytes(raw_bytes[:8], "little")
    header = json.loads(raw_bytes[8:8 + header_len])
    payload_offset = 8 + header_len

    def _hand_widen(name, shape):
        spec = header[name]
        start, end = spec["data_offsets"]
        raw = np.frombuffer(raw_bytes, dtype=np.uint16,
                            count=(end - start) // 2, offset=payload_offset + start)
        words = raw.astype(np.uint32) << np.uint32(16)
        return words.view(np.float32).astype(np.float64).reshape(shape)

    assert np.array_equal(a_f, _hand_widen(KEY_A, a_f.shape))
    assert np.array_equal(b_f, _hand_widen(KEY_B, b_f.shape))


def test_read_peft_lora_pair_still_handles_fp16_and_fp32_after_the_fix(tmp_path):
    """T-2194's own sweep instruction: the fix routes every dtype through `_SafeTensors`, not
    only BF16 -- confirm F16 and F32 LoRA tensors (the shapes this reader already handled before
    the fix) still convert correctly, so the repair did not narrow correctness to bf16 alone."""
    rng = np.random.default_rng(5)
    a32 = rng.normal(size=(RANK, Q_WIDTH)).astype(np.float32)
    b32 = rng.normal(size=(Q_WIDTH, RANK)).astype(np.float32)

    for label, arrs, sf_dtype in (("f32", (a32, b32), "F32"),
                                  ("f16", (a32.astype(np.float16), b32.astype(np.float16)), "F16")):
        adapter_dir = tmp_path / f"adapter_{label}"
        adapter_dir.mkdir()
        a_arr, b_arr = arrs
        write_safetensors_file(adapter_dir / "adapter_model.safetensors",
                               {KEY_A: a_arr, KEY_B: b_arr},
                               dtypes={KEY_A: sf_dtype, KEY_B: sf_dtype})
        write_adapter_config(adapter_dir, base_model_name_or_path="unused")

        a_f, b_f = A.read_peft_lora_pair(adapter_dir, layer=0, proj_short="q_proj", meta=_meta())
        assert np.array_equal(a_f, a_arr.astype(np.float64)), f"{label}: A tensor mismatch"
        assert np.array_equal(b_f, b_arr.astype(np.float64)), f"{label}: B tensor mismatch"


def test_read_peft_lora_pair_rejects_an_unwidenable_dtype_explicitly(tmp_path):
    """The other half of the sweep: a dtype `_SafeTensors` does not widen exactly (e.g. an int8
    tensor, never a legal LoRA weight dtype) must raise, not silently produce a wrong float --
    `UnsupportedOpSet`, the same explicit-rejection contract `pipeline._SafeTensors.tensor()`
    already carries for the base checkpoint reader."""
    adapter_dir = tmp_path / "adapter_bad_dtype"
    adapter_dir.mkdir()
    a_arr = np.zeros((RANK, Q_WIDTH), dtype=np.int8)
    b_arr = np.zeros((Q_WIDTH, RANK), dtype=np.int8)
    write_safetensors_file(adapter_dir / "adapter_model.safetensors",
                           {KEY_A: a_arr, KEY_B: b_arr}, dtypes={})
    # write_safetensors_file only knows F32/F16/BF16; hand-patch the header to claim I8 directly,
    # matching the raw bytes an int8 tensor would actually occupy.
    raw = (adapter_dir / "adapter_model.safetensors").read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + header_len])
    payload = bytearray()
    offset = 0
    for name, arr in ((KEY_A, a_arr), (KEY_B, b_arr)):
        data = arr.tobytes()
        header[name] = {"dtype": "I8", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(adapter_dir / "adapter_model.safetensors", "wb") as fh:
        fh.write(len(header_bytes).to_bytes(8, "little"))
        fh.write(header_bytes)
        fh.write(bytes(payload))
    write_adapter_config(adapter_dir, base_model_name_or_path="unused")

    pipeline = A._load_spike()
    try:
        A.read_peft_lora_pair(adapter_dir, layer=0, proj_short="q_proj", meta=_meta())
        assert False, "an I8-dtyped LoRA tensor must be rejected explicitly, not silently mis-read"
    except pipeline.UnsupportedOpSet:
        pass
