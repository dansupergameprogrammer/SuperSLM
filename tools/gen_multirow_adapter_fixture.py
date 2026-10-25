"""Multi-row plan §7 (Fixtures): the cloud adapter fixture `pkv_def_r4.adp`, a rank-4 runtime LoRA
over `pkv_def.sslm` (tools/gen_paged_kv_fixture.py).

The multi-row cells cross every shape with an adapter (cell 4.1's cloud matrix, mutant MX6), and the
paged-KV generator writes none, so this writer makes one. It is written field by field to the
standalone adapter-artifact format (design §24.2, as tools/sslm_convert_adapter.py's
`build_runtime_additive_sections` assembles it): the base's own Config section copied byte for
byte, the sigmoid LUT, an ADP1 provenance carrying the base's integrity hash, a WGT1 manifest of
`layer<L>.<proj>.lora_A` [4, in] and `.lora_B` [out, 4] int8 tensors, and the DFS1/UFS1 manifests
of per-channel and per-rank (identity, mult, exponent) fold triples.

What it covers, and why:
  - every adaptable projection (q, k, v, o, gate, up, down) on every layer, so each of the five
    adapter-bearing tail sites, and both of kv_proj's slots, run a nonzero delta;
  - lora_A and lora_B are nonzero and vary along every axis (input channel, rank index, output
    channel, projection and layer), so a tail that read another row's activations, another
    projection's tensor or another rank's column produces different bytes (mutant MX6);
  - the fold triples are exact powers of two (u: 2^-6, delta: 2^-2): the rank-4 intermediate lands
    in the int8 code domain for typical inputs and the delta moves the base accumulator without
    dominating it. They are derived through the converter's own `derive_amplifying_triple`.

Every byte is integer arithmetic or an exact power-of-two conversion, so the file is identical on
every host. The writer prints its sha256; `--expect-sha256 HEX` exits non-zero on any other value.

Usage: python tools/gen_multirow_adapter_fixture.py BASE_PKV_DEF_SSLM OUT_ADP [--expect-sha256 HEX]
"""

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sslm_convert_adapter as sca  # noqa: E402
import sslm_format as sf  # noqa: E402
import sslm_model_writer as smw  # noqa: E402

RANK = 4
LORA_ALPHA = 8.0
U_RHO = 2.0 ** -6
DELTA_RHO = 2.0 ** -2
NAME = "pkv-def-r4-multirow"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_cfg1(blob: bytes) -> dict:
    if blob[:4] != b"CFG1":
        raise SystemExit("base Config section is not CFG1")
    fields = struct.unpack_from("<16I", blob, 4)
    names = ("version", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
             "head_dim", "intermediate_size", "vocab_size", "context_cap")
    return dict(zip(names, fields))


def dims(cfg: dict, proj: str):
    """(in_channels, out_channels) of one projection, as adapter_marshal.h's
    AdapterInChannelsFor/AdapterOutChannelsFor define them."""
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    q_width = cfg["num_attention_heads"] * cfg["head_dim"]
    kv = cfg["num_key_value_heads"] * cfg["head_dim"]
    return {
        "q_proj": (hidden, q_width), "k_proj": (hidden, kv), "v_proj": (hidden, kv),
        "o_proj": (q_width, hidden), "gate_proj": (hidden, inter), "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }[proj]


def pattern(shape, salt: int) -> np.ndarray:
    """Deterministic int8 values in [-4, 4] that vary along every index: an integer hash of the
    flat index and a per-tensor salt, never a float RNG."""
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (idx + np.uint64(salt) * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    vals = (z % np.uint64(9)).astype(np.int64) - 4
    return vals.astype(np.int8).reshape(shape)


def triple(rho: float):
    t = sca.derive_amplifying_triple(rho)
    if t is None:
        raise SystemExit(f"no fold triple for ratio {rho}")
    return [t.identity, t.mult, t.exponent]


def build(base_path: Path) -> bytes:
    cfg_bytes = sf.read_section_bytes(str(base_path), sf.SectionType.CONFIG)
    if cfg_bytes is None:
        raise SystemExit(f"{base_path}: no Config section")
    cfg = parse_cfg1(cfg_bytes)
    base_hash = sf.raw_integrity_hash(str(base_path))
    weights, dfs1, ufs1 = {}, {}, {}
    salt = 1
    for layer in range(cfg["num_hidden_layers"]):
        for proj in PROJECTIONS:
            d_in, d_out = dims(cfg, proj)
            name = f"layer{layer}.{proj}"
            weights[f"{name}.lora_A"] = pattern((RANK, d_in), salt)
            weights[f"{name}.lora_B"] = pattern((d_out, RANK), salt + 1)
            salt += 2
            dfs1[name] = np.asarray([triple(DELTA_RHO)] * d_out, dtype=np.int32)
            ufs1[name] = np.asarray([triple(U_RHO)] * RANK, dtype=np.int32)
    adp1 = sca.write_adp1(rank=RANK, target_modules=PROJECTIONS, base_artifact_hash=base_hash,
                          lora_alpha=LORA_ALPHA, use_rslora=False, source_adapter_name=NAME)
    sections = [
        sf.Section(sf.SectionType.CONFIG, cfg_bytes),
        sf.Section(sf.SectionType.SIGMOID_LUT, smw.write_sil1()),
        sf.Section(sf.SectionType.PROVENANCE, adp1),
        sf.Section(sf.SectionType.WEIGHTS, smw.write_tensor_manifest(smw.WGT1, np.int8, weights)),
        sf.Section(sf.SectionType.DELTA_FOLD_SCALES, smw.write_tensor_manifest(smw.DFS1, np.int32, dfs1)),
        sf.Section(sf.SectionType.U_FOLD_SCALES, smw.write_tensor_manifest(smw.UFS1, np.int32, ufs1)),
    ]
    data, _fingerprint = sf.build_artifact(sections)
    return data


def main(argv) -> int:
    args = argv[1:]
    expect = None
    if "--expect-sha256" in args:
        i = args.index("--expect-sha256")
        if i + 1 >= len(args):
            print("--expect-sha256 needs a value", file=sys.stderr)
            return 2
        expect = args[i + 1].lower()
        del args[i:i + 2]
    if len(args) != 2:
        print(__doc__.strip().split("\n")[-1], file=sys.stderr)
        return 2
    base, out = Path(args[0]), Path(args[1])
    data = build(base)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"wrote {out}: {len(data)} bytes, rank {RANK}, sha256={digest}")
    if expect is not None and digest != expect:
        print(f"sha256 mismatch: expected {expect}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
