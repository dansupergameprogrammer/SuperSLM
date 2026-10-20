"""T-3103 S0 reference: encoder E1 and ResolvedVectorBatch v1 (plan rev 7 section 4.2).

This module is the executable specification the Wizard/SuperSLM-Unreal bridge (S3b) must
reproduce bit for bit. It imports no SuperSLM, SuperFAISS or plugin code.

Encoder E1, frozen at S0 (T-3103):
  input   one resolved command: a JSON object with nonempty string keys. Values may be
          strings, booleans, null, integers, arrays or objects. Non-integer numbers are
          refused (EncoderUnsupportedValue), because their text form is not portable.
  step 1  every string (keys and values, recursively) is Unicode NFC-normalized.
  step 2  for each top-level key, in Unicode code-point order, the value is serialized as
          json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
          allow_nan=False), then UTF-8 encoded.
  step 3  h = SHA-256(b"SSLM-E1" + 0x1F + key_utf8 + 0x1F + value_utf8).
          For j in 0..3: u = little-endian uint32 of h[4j:4j+4]; lane = u % 64;
          sign = +1 when (h[16+j] & 1) == 0, else -1; acc[lane] += sign (integer).
  step 4  n2 = sum(acc[i]^2) as an integer. n2 == 0 refuses (EncoderZeroRow).
          row[i] = float32(acc[i] / sqrt(n2)), computed in IEEE binary64 (one correctly
          rounded sqrt, one correctly rounded divide), then rounded once to binary32.
  output  64 finite float32 values, unit L2 norm up to binary32 rounding.

The trigger is encoded like every other field, from its canonical JSON. E1 therefore does
not depend on whether the trigger is a string or a structured condition (D-SLM8075 revision);
a change to the resolved command's field set changes the vectors, and a deliberate change of
the algorithm takes a new encoder version, never an edit of E1.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import unicodedata

ENCODER_VERSION = "E1"
DIMS = 64
METRIC = "dot"
QUANT = "float32"
MIN_DIMS, MAX_DIMS = 3, 131072
_DOMAIN = b"SSLM-E1\x1f"


class EncoderError(ValueError):
    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}" if detail else status)
        self.status = status


def _nfc(value):
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise EncoderError("EncoderUnsupportedValue", "non-integer number")
    if isinstance(value, list):
        return [_nfc(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise EncoderError("EncoderUnsupportedValue", "empty or non-string key")
            nk = unicodedata.normalize("NFC", k)
            if nk in out:
                raise EncoderError("EncoderUnsupportedValue", "keys collide after NFC")
            out[nk] = _nfc(v)
        return out
    raise EncoderError("EncoderUnsupportedValue", type(value).__name__)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def encode_bits(command: dict) -> tuple[int, ...]:
    """E1 row as 64 binary32 bit patterns."""
    if not isinstance(command, dict) or not command:
        raise EncoderError("EncoderUnsupportedValue", "command must be a nonempty object")
    normalized = _nfc(command)
    acc = [0] * DIMS
    for key in sorted(normalized):
        digest = hashlib.sha256(_DOMAIN + key.encode("utf-8") + b"\x1f" + canonical(normalized[key])).digest()
        for j in range(4):
            lane = int.from_bytes(digest[4 * j:4 * j + 4], "little") % DIMS
            acc[lane] += 1 if (digest[16 + j] & 1) == 0 else -1
    n2 = sum(a * a for a in acc)
    if n2 == 0:
        raise EncoderError("EncoderZeroRow")
    root = math.sqrt(n2)
    return tuple(struct.unpack("<I", struct.pack("<f", a / root))[0] for a in acc)


def bits_to_floats(bits) -> tuple[float, ...]:
    return tuple(struct.unpack("<f", struct.pack("<I", b))[0] for b in bits)


def float_to_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# ---------------------------------------------------------------- ResolvedVectorBatch v1

STATUSES = ("Published", "OverCap", "EncoderZeroRow", "EncoderUnsupportedValue", "NonFiniteRow",
            "WrongDimension", "MapMismatch", "StaleContext", "PartialAppend", "InvalidBatch")


def build_batch(batch_id: str, catalogue_version: str, state_version: str, commands: list[dict]) -> dict:
    """Reference bridge projection after S3 authorization: one E1 row per final command."""
    rows = [list(encode_bits(c)) for c in commands]
    return {
        "format": "ResolvedVectorBatch v1", "batch_id": batch_id,
        "catalogue_version": catalogue_version, "state_version": state_version,
        "encoder_version": ENCODER_VERSION, "dims": DIMS, "metric": METRIC, "quant": QUANT,
        "channels": 0, "rows": rows,
        "host_map": [{"batch_id": batch_id, "ordinal": i, "command": c,
                      "catalogue_version": catalogue_version, "state_version": state_version}
                     for i, c in enumerate(commands)],
    }


def publish_status(batch: dict, authorized_commands: list[dict], *, cap: int,
                   live_catalogue_version: str, live_state_version: str,
                   append_statuses: list[str] | None = None) -> str:
    """Reference verdict for whether a staged batch may publish. Any non-Published status
    means: staged bank destroyed, zero rows visible, zero action attempts.

    Order of checks is part of the contract: shape, cap, context, rows, map, appends."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    try:
        if batch.get("format") != "ResolvedVectorBatch v1" or batch.get("encoder_version") != ENCODER_VERSION \
                or batch.get("metric") != METRIC or batch.get("quant") != QUANT or batch.get("channels") != 0:
            return "InvalidBatch"
        n = len(authorized_commands)
        if n == 0:
            return "InvalidBatch"
        if n > cap:
            return "OverCap"
        if (batch["catalogue_version"], batch["state_version"]) != (live_catalogue_version, live_state_version):
            return "StaleContext"
        dims = batch["dims"]
        if not (MIN_DIMS <= dims <= MAX_DIMS) or dims != DIMS:
            return "WrongDimension"
        rows = batch["rows"]
        if len(rows) != n:
            return "MapMismatch"
        for row in rows:
            if len(row) != dims:
                return "WrongDimension"
            if any(not math.isfinite(v) for v in bits_to_floats(row)):
                return "NonFiniteRow"
        for i, (row, command) in enumerate(zip(rows, authorized_commands)):
            if tuple(row) != encode_bits(command):
                return "MapMismatch"
        host_map = batch["host_map"]
        if len(host_map) != n or [m.get("ordinal") for m in host_map] != list(range(n)):
            return "MapMismatch"
        for i, entry in enumerate(host_map):
            if entry.get("batch_id") != batch["batch_id"] or entry.get("command") != authorized_commands[i] \
                    or (entry.get("catalogue_version"), entry.get("state_version")) != \
                    (batch["catalogue_version"], batch["state_version"]):
                return "MapMismatch"
        if append_statuses is not None and (len(append_statuses) != n or any(s != "Ok" for s in append_statuses)):
            return "PartialAppend"
    except EncoderError as error:
        return error.status
    except (KeyError, TypeError):
        return "InvalidBatch"
    return "Published"
