"""Build a real V=70 artifact with an all-zero untied head for an integrated tie.

Every head row ties for the maximum, including rows 0 and 64 on opposite sides
of the T=3 partition. The embedding remains independent of the head.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out" / "tie"
EXPECTED_SHA256 = "7a5b3e08da856b88453829886440f27e1a2568dcd999fd27e9106ccf9419d960"
sys.path.insert(0, str(ROOT / "tools"))
import _calibrate_checkpoint_fixture as writer

writer.VOCAB = writer.VOCAB[:48] + [f"[TIE_{i}]" for i in range(22)]
writer.VOCAB_SIZE = len(writer.VOCAB)


def head_location(data: bytes) -> tuple[int, int, tuple[int, int]]:
    sections = struct.unpack_from("<I", data, 12)[0]
    for i in range(sections):
        kind, _, offset, _, _, _, _ = struct.unpack_from("<IIQQQII", data, 64 + 40 * i)
        if kind != 2:
            continue
        if data[offset:offset + 4] != b"WGT1":
            raise ValueError("weights section is not WGT1")
        count = struct.unpack_from("<I", data, offset + 8)[0]
        names = offset + 16 + count * 48
        for j in range(count):
            desc = offset + 16 + j * 48
            name_offset, name_length, rank = struct.unpack_from("<III", data, desc)
            name = data[names + name_offset:names + name_offset + name_length]
            if name != b"lm_head":
                continue
            if rank != 2:
                raise ValueError("head rank is not 2")
            shape = struct.unpack_from("<II", data, desc + 12)
            data_offset, elements = struct.unpack_from("<QQ", data, desc + 28)
            return offset + data_offset, elements, shape
    raise ValueError("untied lm_head was not found")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "checkpoint"
    calibrated = OUT / "calibrated"
    artifact = OUT / "tie.sslm"
    writer.build_parameterized_fixture_checkpoint(
        checkpoint, prefix="model.", biased=True, tie_word_embeddings=False,
        lm_head_present=True, qk_norm=False, seed=1,
    )
    subprocess.run([sys.executable, str(ROOT / "tools" / "calibrate_checkpoint.py"),
                    "--checkpoint", str(checkpoint), "--out", str(calibrated)],
                   cwd=ROOT, check=True, timeout=120)
    subprocess.run([sys.executable, str(ROOT / "tools" / "convert_model.py"),
                    "--artifact", str(calibrated), "--out", str(artifact), "--skip-verify"],
                   cwd=ROOT, check=True, timeout=120)
    data = bytearray(artifact.read_bytes())
    offset, elements, shape = head_location(data)
    if shape != (70, 8) or elements != 70 * 8:
        raise ValueError(f"unexpected head {shape}, elements={elements}")
    data[offset:offset + elements] = bytes(elements)
    data[32:64] = bytes(32)
    data[32:64] = hashlib.sha256(data).digest()
    artifact.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"TIE fixture digest drift: {digest} != {EXPECTED_SHA256}")
    print(f"TIE artifact={artifact} V=70 H=8 sha256={digest}")


if __name__ == "__main__":
    main()
