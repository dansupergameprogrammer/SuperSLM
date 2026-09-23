"""Build T-2851's untied RU fixture with the repository's checkpoint writer.

The chosen seed is the first whose real CPU decode differs from a copy with
lm_head replaced by embed. Both artifacts are generated through the converter;
the latter differs only at the WGT1 lm_head bytes and its integrity digest.
All generated files live below this suite's ignored out/ directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "ru"
EXPECTED_SEED = 1
EXPECTED_SHA256 = "687dd34628493ddd5bd43eadd8aff4c9d316e57b33c4a43809ba43ce9626142a"
sys.path.insert(0, str(ROOT / "tools"))
import _calibrate_checkpoint_fixture as checkpoint_writer

# The writer's current vocabulary has 50 words. RU's design fixes V=48; trim
# two ordinary words before invoking its parameterized writer.
checkpoint_writer.VOCAB = checkpoint_writer.VOCAB[:48]
checkpoint_writer.VOCAB_SIZE = len(checkpoint_writer.VOCAB)


def sections(buf: bytes):
    if buf[:4] != b"SSLM":
        raise ValueError("not an SSLM artifact")
    count = struct.unpack_from("<I", buf, 12)[0]
    for i in range(count):
        kind, dtype, off, size, elements, align, reserved = struct.unpack_from(
            "<IIQQQII", buf, 64 + 40 * i
        )
        yield kind, off, size


def tensors(buf: bytes):
    kind, off, size = next(row for row in sections(buf) if row[0] == 2)
    if buf[off : off + 4] != b"WGT1":
        raise ValueError("weights are not WGT1")
    _, count, name_bytes = struct.unpack_from("<III", buf, off + 4)
    names = off + 16 + count * 48
    found = {}
    for i in range(count):
        desc = off + 16 + i * 48
        name_off, name_len, rank = struct.unpack_from("<III", buf, desc)
        shape = struct.unpack_from("<IIII", buf, desc + 12)
        data_off, elements = struct.unpack_from("<QQ", buf, desc + 28)
        name = bytes(buf[names + name_off : names + name_off + name_len]).decode()
        found[name] = (off + data_off, elements, shape[:rank])
    return found


def substituted(src: Path, dst: Path):
    b = bytearray(src.read_bytes())
    weights = tensors(b)
    embed_off, embed_n, embed_shape = weights["embed"]
    head_off, head_n, head_shape = weights["lm_head"]
    if embed_n != head_n or embed_shape != head_shape or embed_n != 48 * 8:
        raise ValueError("RU head geometry is wrong")
    if b[embed_off : embed_off + embed_n] == b[head_off : head_off + head_n]:
        raise ValueError("RU is tied in bytes")
    b[head_off : head_off + head_n] = b[embed_off : embed_off + embed_n]
    b[32:64] = bytes(32)
    b[32:64] = hashlib.sha256(b).digest()
    dst.write_bytes(b)


def decode(backend: str, path: Path, steps=16):
    exe = HERE / "out" / "cell_real_decode.exe"
    result = subprocess.run(
        [str(exe), backend, str(path), "-", str(steps), "0", "0", "0,1,2"],
        cwd=exe.parent, text=True, capture_output=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"{backend} decode failed ({result.returncode}): {result.stdout} {result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.startswith("TOKENS")]
    if len(lines) != 1:
        raise RuntimeError(f"{backend} emitted {len(lines)} token lines")
    return [int(x) for x in lines[0].split()[1:]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in range(32):
        checkpoint = OUT / "checkpoint"
        calibrated = OUT / "calibrated"
        artifact = OUT / "ru.sslm"
        shadow = OUT / "ru_embed_shadow.sslm"
        checkpoint_writer.build_parameterized_fixture_checkpoint(
            checkpoint, prefix="model.", biased=True,
            tie_word_embeddings=False, lm_head_present=True,
            qk_norm=False, seed=seed,
        )
        if json.loads((checkpoint / "config.json").read_text())["tie_word_embeddings"]:
            raise RuntimeError("SETUP: RU checkpoint is tied")
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "calibrate_checkpoint.py"),
             "--checkpoint", str(checkpoint), "--out", str(calibrated)],
            cwd=ROOT, check=True, timeout=120,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "convert_model.py"),
             "--artifact", str(calibrated), "--out", str(artifact), "--skip-verify"],
            cwd=ROOT, check=True, timeout=120,
        )
        substituted(artifact, shadow)
        real = decode("cpu", artifact)
        fake = decode("cpu", shadow)
        if real == fake:
            continue
        gpu = decode("gpu", artifact)
        if gpu != real:
            raise RuntimeError(f"SETUP: GPU tokens differ from CPU at seed {seed}: {gpu} vs {real}")
        result = {
            "seed": seed,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "embed_shadow_sha256": hashlib.sha256(shadow.read_bytes()).hexdigest(),
            "real_tokens": real,
            "embed_shadow_tokens": fake,
            "gpu_tokens": gpu,
        }
        if seed != EXPECTED_SEED or result["sha256"] != EXPECTED_SHA256:
            raise RuntimeError(f"RU pin changed: seed={seed}, sha256={result['sha256']}")
        (OUT / "preconditions.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    raise RuntimeError("SETUP: no seed 0..31 discriminated lm_head from embed")


if __name__ == "__main__":
    raise SystemExit(main())
