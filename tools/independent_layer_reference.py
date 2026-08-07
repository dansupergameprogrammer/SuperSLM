#!/usr/bin/env python3
"""Independent third-witness float composition (T-1690) -- design
`Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md`
S6.

WHAT THIS IS AND WHY IT IS INDEPENDENT, READ BEFORE TRUSTING ANY OUTPUT.

T-1683's campaign has two witnesses that disagree at layers 21-28 on four
"mechanism-2" prompts: the int8 engine (`src/forward/forward_sites.cpp`) and
the float reference (`tools/float_reference_layer_dump.py`, T-1686). Both
existing float-side captures call `Qwen2DecoderLayer.forward` -- the
identical `transformers` library method, sharing every line of that
library's own attention/RoPE/MLP arithmetic. This module is a THIRD witness:
a from-scratch composition of the same checkpoint's forward arithmetic,
built to its own textbook definition, that calls no `Qwen2*`-named class or
method anywhere, and shares no code with either existing arm.

Independence, stated field by field (StandardsDocument S5.4 -- "name the
reference's inputs explicitly and show that none is produced by the engine
or the existing float reference"):

  - Weight tensors (`q_proj.weight`, `k_proj.weight`, `v_proj.weight`,
    `o_proj.weight`, `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`,
    `input_layernorm.weight`, `post_attention_layernorm.weight`,
    `embed_tokens.weight`) -- read DIRECTLY from the checkpoint's
    `model.safetensors` file via `safetensors.safe_open`, never through
    `transformers.AutoModelForCausalLM` or any `Qwen2*` class. This is
    STRONGER independence than the design's own minimum bar (reading
    `model.state_dict()` off a loaded model object): this module never
    instantiates a Qwen2 model object of any kind, so a defect in the
    library's OWN state_dict assembly (parameter renaming, sharing, or
    aliasing) cannot leak into this witness either.
  - Architecture hyperparameters (`hidden_size`, `num_hidden_layers`,
    `num_attention_heads`, `num_key_value_heads`, `rms_norm_eps`,
    `rope_theta`) -- read DIRECTLY from the checkpoint's own `config.json`
    by field name. These are published model hyperparameters, not a value
    either implementation under test computes or derives -- the one
    necessarily-shared input the design itself names ("a model
    hyperparameter read from the checkpoint") and it cannot be otherwise:
    there is no other source for "how many attention heads does this
    checkpoint have" than the checkpoint's own declared architecture.
  - The rotary embedding's `inv_freq` table -- NOT read from any library
    buffer. Computed directly from `rope_theta` and `head_dim` by this
    module's own formula (`rope_inv_freq` below), the standard published
    RoPE construction, independent of `transformers.Qwen2RotaryEmbedding`.
  - Tokenization and chat templating -- uses `transformers.AutoTokenizer`
    (the checkpoint's own tokenizer, a lossless, deterministic string-to-
    ID mapping neither the int8 engine's own tokenizer.sslm nor this
    witness's own forward arithmetic is under test on; T-1686 also uses it,
    and T-1685's int8 side uses its own independent tokenizer artifact --
    an identical shared-utility dependency this campaign already accepts
    for exactly this class of input, not part of "the composition" any
    witness is being graded on).
  - NOT read from, and NOT derived from, the int8 engine's own dump format,
    `CarriedScale` values, `(mult, shift)` fold triples, or ANY quantized
    representation -- this witness never opens a `.sslm` artifact.
  - NOT read from, and NOT derived from,
    `tools/float_reference_layer_dump.py`'s own captured dump, hook
    mechanism, or any of its helper functions -- this witness never
    imports that module. The two are compared, never composed together.

Seven composed steps (design S6), each independently formula-derived and
individually red-first proven in `tools/test_independent_layer_reference.py`
before the composed whole is trusted:

  1. RMSNorm from its own defining formula.
  2. Q/K/V projection as a plain `x @ W.T (+ bias)`.
  3. RoPE from the standard rotate-half formula, `inv_freq` computed
     directly from `rope_theta`.
  4. Grouped-query attention via explicit `repeat_interleave`.
  5. Causal masking from a plain lower-triangular construction.
  6. SwiGLU MLP from its own formula.
  7. Token-at-a-time incremental composition with this module's own
     from-scratch K/V cache (never `transformers.DynamicCache`).

Position 0 only (this campaign's own Phase-1 scope): the prompt's own last
token's forward pass, after every earlier prompt token has been walked
through the incremental cache one at a time -- matching the int8 side's own
composition (design S10's "never a batched forward alone" constraint).

Dump format (matching T-1685/T-1686's own shared shape, so the SAME
comparison tooling reads any of the three dumps): uint64 rows (29), uint64
hidden_size, uint64 prompt_fingerprint (FNV-1a 64-bit, over the raw
chat-templated prompt text, UTF-8 -- identical formula to
`tools/float_reference_layer_dump.py`'s own `fnv1a64`, reproduced here
rather than imported, since importing it would share code with the arm
being ruled in or out), uint64 capture_mode (always 1), then rows *
hidden_size float32 values, row-major.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_MODEL = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

REQUIRED_TENSOR_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
REQUIRED_GLOBAL_KEYS = ("model.embed_tokens.weight", "model.norm.weight")
REQUIRED_CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_theta",
)


def fnv1a64(s: str) -> int:
    """FNV-1a, 64-bit, over UTF-8 bytes. Reproduced (not imported) from
    `tools/float_reference_layer_dump.py`'s own formula -- the standard
    constants, independently transcribed, so this module shares no code
    with the arm it is being compared against."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


class StateDictShapeError(Exception):
    """Raised by `load_checkpoint_weights` on a missing tensor or a shape
    that does not match the accompanying config -- a loud, named rejection,
    never a silent misread (design S6 dimension-2 correction, S11
    dimension 2)."""


# --------------------------------------------------------------------------
# Step 1 -- RMSNorm, its own defining formula.
# --------------------------------------------------------------------------
def rms_norm(x, weight, eps: float):
    """x / sqrt(mean(x^2, dim=-1) + eps) * weight -- computed entirely in
    x's own working dtype (no internal upcast to a different precision than
    the caller is already running at); this is a genuine, stated
    implementation choice, not a defect -- the design's own S6 boundary
    names float64/float32 vs. bfloat16-native operation-ordering
    differences as expected, not disqualifying."""
    import torch

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    return normed * weight


def rms_norm_WRONG_no_eps(x, weight, eps: float):
    """Mutation for the red-first guard-vitality proof: drops the epsilon
    term entirely. Used only by the test module, never by the real
    composition."""
    import torch

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance)
    return normed * weight


# --------------------------------------------------------------------------
# Step 3 -- RoPE, standard rotate-half formula.
# --------------------------------------------------------------------------
def rope_inv_freq(head_dim: int, rope_theta: float, device, dtype):
    import torch

    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float64)
    return (1.0 / (rope_theta ** (idx / head_dim))).to(dtype)


def rope_cos_sin(position: int, inv_freq, dtype):
    """cos/sin tables for one position, shape [head_dim]."""
    import torch

    freqs = inv_freq.to(torch.float64) * float(position)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    """x: [..., head_dim]. Standard rotate-half: (-x2, x1)."""
    import torch

    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: [heads, head_dim]; cos/sin: [head_dim]. Broadcasts over heads."""
    return x * cos + rotate_half(x) * sin


def apply_rope_WRONG_no_rotation(x, cos, sin):
    """Mutation for guard-vitality: returns x unrotated (cos/sin ignored).
    Test-module use only."""
    return x


# --------------------------------------------------------------------------
# Step 4 -- grouped-query attention, explicit repeat_interleave.
# --------------------------------------------------------------------------
def repeat_kv_heads(x, n_rep: int):
    """x: [num_kv_heads, seq, head_dim] -> [num_kv_heads * n_rep, seq, head_dim],
    via explicit `torch.repeat_interleave` along the head axis -- never
    `transformers.repeat_kv`'s own expand/reshape construction (design S6
    step 4's own stated boundary)."""
    import torch

    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, n_rep, dim=0)


# --------------------------------------------------------------------------
# Step 5 -- causal mask, plain lower-triangular construction.
# --------------------------------------------------------------------------
def causal_additive_mask(q_len: int, kv_len: int, device, dtype):
    """Returns an additive mask, shape [q_len, kv_len]: 0.0 where key
    position <= (kv_len - q_len) + query offset (i.e. causally visible),
    -inf otherwise. For this module's own incremental decode path q_len is
    always 1 and kv_len == the number of cached positions (0..t inclusive
    after this step's own K/V write) -- every cached position is causally
    visible to the single current query by construction (nothing later than
    the current position is ever written to the cache), so the mask is a
    row of zeros; this function exists, and is separately red-first proven,
    for the general q_len > 1 case the small synthetic fixtures exercise."""
    import torch

    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    kv_pos = torch.arange(kv_len, device=device)
    visible = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [q_len, kv_len]
    mask = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
    mask = mask.masked_fill(~visible, float("-inf"))
    return mask


# --------------------------------------------------------------------------
# Step 6 -- SwiGLU MLP, its own formula.
# --------------------------------------------------------------------------
def silu(x):
    import torch

    return x * torch.sigmoid(x)


def swiglu_mlp(x, w_gate, w_up, w_down):
    """down(silu(gate(x)) * up(x)) -- no bias on any of the three Qwen2 MLP
    projections."""
    gate = x @ w_gate.T
    up = x @ w_up.T
    return (silu(gate) * up) @ w_down.T


def swiglu_mlp_WRONG_relu(x, w_gate, w_up, w_down):
    """Mutation for guard-vitality: ReLU instead of SiLU. Test-module use
    only."""
    import torch

    gate = x @ w_gate.T
    up = x @ w_up.T
    return (torch.relu(gate) * up) @ w_down.T


# --------------------------------------------------------------------------
# Step 7 -- from-scratch, incrementally-grown K/V cache.
# --------------------------------------------------------------------------
class IndependentKVCache:
    """A warm object grown one position at a time within one invocation --
    never `transformers.DynamicCache`. Two tensors per layer
    (`[num_kv_heads, max_len, head_dim]`), pre-allocated to `max_len` and
    written slot-by-slot; `length` tracks how many leading slots are valid.
    This is the object design S6's own dimension-1/dimension-4 correction
    names: its own red-first proof (in the test module) grows it across
    >=8 synthetic positions and reads an early position back after the
    cache has grown past it, asserting bit-for-bit (float64-exact)
    agreement with the value first written."""

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, max_len: int, dtype, device):
        import torch

        self.k = torch.zeros((num_layers, num_kv_heads, max_len, head_dim), dtype=dtype, device=device)
        self.v = torch.zeros((num_layers, num_kv_heads, max_len, head_dim), dtype=dtype, device=device)
        self.length = 0

    def write(self, layer: int, pos: int, k_val, v_val):
        self.k[layer, :, pos, :] = k_val
        self.v[layer, :, pos, :] = v_val

    def write_WRONG_off_by_one(self, layer: int, pos: int, k_val, v_val):
        """Mutation for guard-vitality: writes to slot pos-1 instead of pos
        (clamped to 0). Test-module use only."""
        target = max(pos - 1, 0)
        self.k[layer, :, target, :] = k_val
        self.v[layer, :, target, :] = v_val

    def read(self, layer: int, upto: int):
        """Returns (K, V) for cached positions [0, upto)."""
        return self.k[layer, :, :upto, :], self.v[layer, :, :upto, :]


# --------------------------------------------------------------------------
# Checkpoint loading -- direct safetensors + config.json, no model object.
# --------------------------------------------------------------------------
@dataclass
class CheckpointConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int


def load_config(model_path: Path) -> CheckpointConfig:
    with open(model_path / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise StateDictShapeError(f"config.json missing required key(s): {missing}")
    hidden_size = int(cfg["hidden_size"])
    num_heads = int(cfg["num_attention_heads"])
    if hidden_size % num_heads != 0:
        raise StateDictShapeError(
            f"config.json hidden_size ({hidden_size}) not divisible by num_attention_heads ({num_heads})"
        )
    return CheckpointConfig(
        hidden_size=hidden_size,
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=num_heads,
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        rms_norm_eps=float(cfg["rms_norm_eps"]),
        rope_theta=float(cfg["rope_theta"]),
        head_dim=hidden_size // num_heads,
    )


class CheckpointWeights:
    """Reads `model.safetensors` DIRECTLY via `safetensors.safe_open` --
    never `transformers.AutoModelForCausalLM` -- validated against `config`
    before any arithmetic runs (design S6 dimension-2's own owed cell: a
    wrong key or a shape that does not match the accompanying config is a
    loud, named rejection, never a silent misread)."""

    def __init__(self, tensors: dict, config: CheckpointConfig):
        self.tensors = tensors
        self.config = config

    @staticmethod
    def from_safetensors_dict(tensors: dict, config: CheckpointConfig) -> "CheckpointWeights":
        missing = [k for k in REQUIRED_GLOBAL_KEYS if k not in tensors]
        for layer in range(config.num_hidden_layers):
            for suffix in REQUIRED_TENSOR_SUFFIXES:
                key = f"model.layers.{layer}.{suffix}"
                if key not in tensors:
                    missing.append(key)
        if missing:
            raise StateDictShapeError(f"checkpoint tensors missing {len(missing)} required key(s), e.g. {missing[:5]}")

        embed = tensors["model.embed_tokens.weight"]
        if embed.shape[1] != config.hidden_size:
            raise StateDictShapeError(
                f"model.embed_tokens.weight hidden dim {embed.shape[1]} != config.hidden_size {config.hidden_size}"
            )
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        for layer in range(config.num_hidden_layers):
            qw = tensors[f"model.layers.{layer}.self_attn.q_proj.weight"]
            if tuple(qw.shape) != (q_dim, config.hidden_size):
                raise StateDictShapeError(
                    f"layer {layer} q_proj.weight shape {tuple(qw.shape)} != expected "
                    f"({q_dim}, {config.hidden_size}) given num_attention_heads={config.num_attention_heads}"
                )
            kw = tensors[f"model.layers.{layer}.self_attn.k_proj.weight"]
            if tuple(kw.shape) != (kv_dim, config.hidden_size):
                raise StateDictShapeError(
                    f"layer {layer} k_proj.weight shape {tuple(kw.shape)} != expected "
                    f"({kv_dim}, {config.hidden_size}) given num_key_value_heads={config.num_key_value_heads}"
                )
        return CheckpointWeights(tensors, config)

    def get(self, layer: int, suffix: str):
        return self.tensors[f"model.layers.{layer}.{suffix}"]


def load_checkpoint_weights(model_path: Path, dtype, device) -> CheckpointWeights:
    from safetensors import safe_open

    config = load_config(model_path)
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise StateDictShapeError(f"no .safetensors file found under {model_path}")
    tensors = {}
    for st_path in st_files:
        with safe_open(str(st_path), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    return CheckpointWeights.from_safetensors_dict(tensors, config)


# --------------------------------------------------------------------------
# The composed decoder layer and the incremental forward.
# --------------------------------------------------------------------------
def attention_step(x, weights: CheckpointWeights, layer: int, cache: IndependentKVCache, pos: int,
                    cos, sin, cfg: CheckpointConfig):
    """x: [hidden_size], the normed residual-stream row for this position.
    Returns o_proj(context): [hidden_size]."""
    import torch

    wq = weights.get(layer, "self_attn.q_proj.weight")
    bq = weights.get(layer, "self_attn.q_proj.bias")
    wk = weights.get(layer, "self_attn.k_proj.weight")
    bk = weights.get(layer, "self_attn.k_proj.bias")
    wv = weights.get(layer, "self_attn.v_proj.weight")
    bv = weights.get(layer, "self_attn.v_proj.bias")
    wo = weights.get(layer, "self_attn.o_proj.weight")

    q = (x @ wq.T + bq).view(cfg.num_attention_heads, cfg.head_dim)
    k = (x @ wk.T + bk).view(cfg.num_key_value_heads, cfg.head_dim)
    v = (x @ wv.T + bv).view(cfg.num_key_value_heads, cfg.head_dim)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    cache.write(layer, pos, k, v)
    k_all, v_all = cache.read(layer, pos + 1)  # [num_kv_heads, pos+1, head_dim]

    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    k_rep = repeat_kv_heads(k_all, n_rep)  # [num_heads, pos+1, head_dim]
    v_rep = repeat_kv_heads(v_all, n_rep)

    scale = 1.0 / (cfg.head_dim ** 0.5)
    # q: [num_heads, head_dim] ; k_rep: [num_heads, pos+1, head_dim]
    scores = torch.einsum("hd,hpd->hp", q, k_rep) * scale  # [num_heads, pos+1]
    mask = causal_additive_mask(1, pos + 1, x.device, scores.dtype)  # [1, pos+1], all zeros here
    scores = scores + mask
    weights_sm = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)
    context = torch.einsum("hp,hpd->hd", weights_sm, v_rep)  # [num_heads, head_dim]
    context = context.reshape(-1)  # [hidden_size]
    return context @ wo.T


def decoder_layer_step(hidden, weights: CheckpointWeights, layer: int, cache: IndependentKVCache, pos: int,
                        cos, sin, cfg: CheckpointConfig):
    residual = hidden
    normed = rms_norm(hidden, weights.get(layer, "input_layernorm.weight"), cfg.rms_norm_eps)
    attn_out = attention_step(normed, weights, layer, cache, pos, cos, sin, cfg)
    hidden = residual + attn_out

    residual = hidden
    normed = rms_norm(hidden, weights.get(layer, "post_attention_layernorm.weight"), cfg.rms_norm_eps)
    mlp_out = swiglu_mlp(
        normed,
        weights.get(layer, "mlp.gate_proj.weight"),
        weights.get(layer, "mlp.up_proj.weight"),
        weights.get(layer, "mlp.down_proj.weight"),
    )
    hidden = residual + mlp_out
    return hidden


def run_incremental_forward(weights: CheckpointWeights, input_ids, dtype, device):
    """input_ids: 1-D python list/tensor of token ids (the full
    chat-templated prompt). Runs one token per call, growing the K/V cache
    (design S10's own "never a batched forward alone" constraint). Returns
    {row_index: torch.Tensor[hidden_size] float64} for row 0 (embedding)
    and rows 1..num_hidden_layers (each layer's raw post-residual output),
    taken from the LAST token's own forward only."""
    import torch

    cfg = weights.config
    n = len(input_ids)
    inv_freq = rope_inv_freq(cfg.head_dim, cfg.rope_theta, device, dtype)
    cache = IndependentKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, n, dtype, device)
    embed_table = weights.tensors["model.embed_tokens.weight"]

    captured = {}
    with torch.no_grad():
        for t in range(n):
            tok = int(input_ids[t])
            hidden = embed_table[tok].clone()
            cos, sin = rope_cos_sin(t, inv_freq, dtype)
            if t == n - 1:
                captured[0] = hidden.detach().to(torch.float64).clone()
            for layer in range(cfg.num_hidden_layers):
                hidden = decoder_layer_step(hidden, weights, layer, cache, t, cos, sin, cfg)
                if t == n - 1:
                    captured[layer + 1] = hidden.detach().to(torch.float64).clone()
    return captured


# --------------------------------------------------------------------------
# Dump I/O -- shared shape with T-1685/T-1686's own dumps.
# --------------------------------------------------------------------------
def write_dump(path: Path, captured: dict, hidden_size: int, fingerprint: int, n_layers: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<QQQQ", n_layers + 1, hidden_size, fingerprint, 1))
        for idx in range(n_layers + 1):
            row = captured[idx].to("cpu").numpy().astype("float32")
            f.write(row.tobytes())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="system prompt",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a local HF checkpoint directory")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                         help="compute precision this witness composes in (design S6 dimension-6)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dump", required=True, help="path to write the per-layer float32 dump")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    model_path = Path(args.model)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    weights = load_checkpoint_weights(model_path, dtype, device)
    captured = run_incremental_forward(weights, input_ids, dtype, device)

    fingerprint = fnv1a64(prompt_text)
    write_dump(Path(args.dump), captured, weights.config.hidden_size, fingerprint, weights.config.num_hidden_layers)
    print(
        f"independent_layer_reference: dtype={args.dtype} device={device} "
        f"{weights.config.num_hidden_layers + 1} rows x {weights.config.hidden_size} hidden_size, "
        f"prompt_fingerprint=0x{fingerprint:016X} -> {args.dump}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
