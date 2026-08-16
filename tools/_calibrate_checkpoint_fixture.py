"""A tiny, real, on-disk checkpoint directory for exercising `calibrate_checkpoint.py`'s CLI
(B1, T-2123/T-2137 design §6). No existing construction in either repository builds this:
`sslm_pinned_calibration_fixture.py` and `pipeline.fixture_model` both build a synthetic,
in-memory model fed directly to `build_sections`/`_calibrate` and never touch a checkpoint
directory on disk (checked at source, design §6 B1's own coverage note) -- neither ever calls
`pipeline.load_model`, which is the function `calibrate_checkpoint.py`'s CLI actually calls
and which requires a real `checkpoint / "config.json"` plus safetensors tensors on disk, and
a real tokenizer directory with a chat template (`_checkpoint_tokenize_prompt`).

Sized only to exercise `load_model`'s own rejection paths and a minimal successful load --
not a stand-in for the real Qwen2.5-1.5B-Instruct checkpoint §5's release gate uses.
"""

import json
from pathlib import Path

import numpy as np

# A tiny fixed vocabulary covering the words this repo's own calibration-prompt text and the
# corpus's turns are built from, plus [UNK]/[PAD]/[BOS]/[EOS]/two chat-role tokens. Every word
# not in this list maps to [UNK] -- the tokenizer never produces an id outside [0, VOCAB_SIZE).
_SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
_WORDS = [
    "the", "a", "an", "you", "are", "is", "for", "and", "or", "not", "this", "that",
    "system", "user", "assistant", "customer", "book", "cancel", "price", "intent",
    "slot", "day", "time", "size", "style", "budget", "deposit", "artist", "tattoo",
    "shop", "front", "desk", "json", "object", "key", "value", "true", "false", "none",
    "unspecified", "monday", "tuesday", "wednesday", "arm", "leg", "back",
]
VOCAB = _SPECIAL + _WORDS
VOCAB_SIZE = len(VOCAB)  # 48

HIDDEN_SIZE = 8
NUM_HIDDEN_LAYERS = 1
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 1
HEAD_DIM = 4
INTERMEDIATE_SIZE = 16
MAX_POSITION_EMBEDDINGS = 256

_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant:\n{% endif %}"
)


def _write_tokenizer(checkpoint_dir: Path) -> None:
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import WordLevel

    vocab = {tok: i for i, tok in enumerate(VOCAB)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace()]
    )
    tok.decoder = decoders.WordPiece()
    tok.save(str(checkpoint_dir / "tokenizer.json"))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": _CHAT_TEMPLATE,
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "model_max_length": MAX_POSITION_EMBEDDINGS,
    }
    (checkpoint_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )
    (checkpoint_dir / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "[UNK]", "pad_token": "[PAD]",
                    "bos_token": "[BOS]", "eos_token": "[EOS]"}),
        encoding="utf-8",
    )


def _write_config(checkpoint_dir: Path) -> None:
    config = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "model_type": "qwen2",
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_safetensors(checkpoint_dir: Path, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    q_width = NUM_ATTENTION_HEADS * HEAD_DIM
    kv_width = NUM_KEY_VALUE_HEADS * HEAD_DIM

    def small(*shape):
        return rng.normal(scale=0.02, size=shape).astype(np.float32)

    tensors = {
        "model.embed_tokens.weight": small(VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": np.ones(HIDDEN_SIZE, dtype=np.float32),
    }
    for layer in range(NUM_HIDDEN_LAYERS):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = small(q_width, HIDDEN_SIZE)
        tensors[f"{p}.self_attn.k_proj.weight"] = small(kv_width, HIDDEN_SIZE)
        tensors[f"{p}.self_attn.v_proj.weight"] = small(kv_width, HIDDEN_SIZE)
        tensors[f"{p}.self_attn.o_proj.weight"] = small(HIDDEN_SIZE, q_width)
        tensors[f"{p}.self_attn.q_proj.bias"] = small(q_width)
        tensors[f"{p}.self_attn.k_proj.bias"] = small(kv_width)
        tensors[f"{p}.self_attn.v_proj.bias"] = small(kv_width)
        tensors[f"{p}.mlp.gate_proj.weight"] = small(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        tensors[f"{p}.mlp.up_proj.weight"] = small(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        tensors[f"{p}.mlp.down_proj.weight"] = small(HIDDEN_SIZE, INTERMEDIATE_SIZE)

    header = {}
    payload = bytearray()
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        data = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    header["__metadata__"] = {}
    header_bytes = json.dumps(header).encode("utf-8")

    path = checkpoint_dir / "model.safetensors"
    with open(path, "wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(bytes(payload))


def build_fixture_checkpoint(checkpoint_dir: Path, *, seed: int = 0) -> Path:
    """Build a complete, real, on-disk checkpoint directory at `checkpoint_dir`:
    `config.json`, `model.safetensors`, and a working fast tokenizer with a chat template.
    Every field `pipeline.load_config`/`pipeline.load_model` requires is present; every
    tensor `pipeline._upstream_names` demands is on disk. Idempotent -- safe to call once
    per test session and reuse.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_config(checkpoint_dir)
    _write_safetensors(checkpoint_dir, seed=seed)
    _write_tokenizer(checkpoint_dir)
    return checkpoint_dir
