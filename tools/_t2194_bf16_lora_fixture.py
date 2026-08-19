"""A tiny, real, on-disk PEFT LoRA adapter directory whose weights are genuinely stored bf16
(T-2194, D-SLM3697): `read_peft_lora_pair`'s own defect shape, before this ticket's fix, was
`.astype(np.float64)` on a bf16-dtyped safetensors tensor -- numpy has no bfloat16 dtype, so that
raised `TypeError: data type bfloat16 not understood`.

Adapter shapes bind against `tools/_calibrate_checkpoint_fixture.py`'s own base-checkpoint shape
(`HIDDEN_SIZE=8`, `NUM_ATTENTION_HEADS=2`, `HEAD_DIM=4` -> `q_proj` is `[8, 8]`), so the same
fixture checkpoint that already exercises `pipeline.load_model` also supplies a real base this
adapter can bind against for an end-to-end run. Only `q_proj` is targeted -- the smallest shape
that exercises the real reader path (one projection, rank 2), not a stand-in for a real multi-
projection adapter.

`lora_alpha == r` (`LORA_ALPHA == RANK`) so `scaling == lora_alpha / r == 1.0` exactly
(`use_rslora=False`, `AdapterMeta.scaling`, `SuperSLM_Plan.md` §11(b)(ii)) -- the fixture's own
equivalence check (see `test_sslm_convert_adapter_bf16.py`) is then a property of the BF16 WIDENING
step alone, not diluted by a scaling multiply.

`write_safetensors_file` mirrors `pipeline._SafeTensors.tensor()`'s own on-disk shape exactly (an
8-byte little-endian header-length prefix, the JSON header, then raw bytes back to back) -- the
same minimal construction `tools/reference_pipeline/tests/conftest.py`'s own
`write_safetensors_shard`/`bf16_clean` already use for the base-checkpoint reader's own tests,
restated locally per this tree's own convention of a small, file-local fixture writer
(`_calibrate_checkpoint_fixture.py`'s own `_write_safetensors`) rather than a cross-suite import
(`tools/reference_pipeline/tests/` carries no `__init__.py` and is not a package).
"""

import json
from pathlib import Path

import numpy as np

from _calibrate_checkpoint_fixture import HEAD_DIM, HIDDEN_SIZE, NUM_ATTENTION_HEADS
from sslm_convert_adapter import _PROJECTION_FULL_PATH

RANK = 2
LORA_ALPHA = float(RANK)  # scaling = lora_alpha / r == 1.0 exactly (use_rslora=False)
TARGET_MODULES = ["q_proj"]
LAYER = 0
Q_WIDTH = NUM_ATTENTION_HEADS * HEAD_DIM  # 8, matches the fixture checkpoint's own q_proj width

KEY_A = f"base_model.model.model.layers.{LAYER}.{_PROJECTION_FULL_PATH['q_proj']}.lora_A.weight"
KEY_B = f"base_model.model.model.layers.{LAYER}.{_PROJECTION_FULL_PATH['q_proj']}.lora_B.weight"


def bf16_clean(arr) -> np.ndarray:
    """Zero an array's low 16 mantissa bits so it is already exactly bf16-representable --
    writing it through the BF16 path below and widening it back is then exact by construction,
    independent of which code performs the truncation (same construction as
    `tools/reference_pipeline/tests/conftest.py`'s own `bf16_clean`)."""
    arr32 = np.ascontiguousarray(arr, dtype=np.float32)
    words = arr32.view(np.uint32) & np.uint32(0xFFFF0000)
    return words.copy().view(np.float32)


def lora_values(seed: int):
    """`(A, B)` for the single `q_proj` pair this fixture targets, already bf16-clean -- so
    writing them BF16 or F32 and reading either back is exact and directly comparable."""
    rng = np.random.default_rng(seed)
    a = bf16_clean(rng.normal(scale=0.5, size=(RANK, Q_WIDTH)))
    b = bf16_clean(rng.normal(scale=0.5, size=(Q_WIDTH, RANK)))
    return a, b


def write_safetensors_file(path, tensors, *, dtypes=None) -> None:
    """One safetensors file at `path`, from `{tensor_name: array}`. `dtypes` (optional):
    `{tensor_name: "F32" | "F16" | "BF16"}`, default `"F32"` for a name not listed. A `"BF16"`
    entry stores the array's top 16 bits only -- the same reinterpretation
    `pipeline._SafeTensors.tensor()` widens back from
    (`raw.view(uint16).astype(uint32) << 16`)."""
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
        elif dtype == "F16":
            data = arr32.astype(np.float16).tobytes()
        else:
            raise ValueError(f"write_safetensors_file: unsupported dtype {dtype!r}")
        header[name] = {"dtype": dtype, "shape": list(arr32.shape),
                        "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    header["__metadata__"] = {}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(len(header_bytes).to_bytes(8, "little"))
        fh.write(header_bytes)
        fh.write(bytes(payload))


def write_adapter_config(adapter_dir: Path, *, base_model_name_or_path: str) -> None:
    config = {
        "r": RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "use_rslora": False,
        "init_lora_weights": True,
        "use_dora": False,
        "bias": "none",
        "fan_in_fan_out": False,
        "base_model_name_or_path": base_model_name_or_path,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def build_bf16_lora_fixture(adapter_dir: Path, *, base_model_name_or_path: str,
                            seed: int = 0) -> Path:
    """A real, on-disk PEFT adapter directory (`adapter_config.json` +
    `adapter_model.safetensors`) whose `layer 0` `q_proj` lora_A/lora_B tensors are stored
    genuinely BF16 -- the T-2194 defect shape."""
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    a, b = lora_values(seed)
    write_safetensors_file(adapter_dir / "adapter_model.safetensors", {KEY_A: a, KEY_B: b},
                           dtypes={KEY_A: "BF16", KEY_B: "BF16"})
    write_adapter_config(adapter_dir, base_model_name_or_path=base_model_name_or_path)
    return adapter_dir


def build_fp32_reference_fixture(adapter_dir: Path, *, base_model_name_or_path: str,
                                 seed: int = 0) -> Path:
    """The SAME (already bf16-clean) lora_A/lora_B values as `build_bf16_lora_fixture` for the
    identical `seed`, written F32 instead of BF16 -- the equivalence oracle's reference side
    (T-2194): F32 adapters already read correctly through the PRE-FIX code
    (`.astype(np.float64)` on a real numpy dtype never raised), so a fixed reader's BF16 read must
    produce byte-identical values to this F32 read of the identical numbers."""
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    a, b = lora_values(seed)
    write_safetensors_file(adapter_dir / "adapter_model.safetensors", {KEY_A: a, KEY_B: b},
                           dtypes={KEY_A: "F32", KEY_B: "F32"})
    write_adapter_config(adapter_dir, base_model_name_or_path=base_model_name_or_path)
    return adapter_dir
