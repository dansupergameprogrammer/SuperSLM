"""Mutate one KVC1 attn_norm row to force an Idle strict-partial refusal."""
import hashlib
import struct
import sys

if len(sys.argv) != 4:
    raise SystemExit("usage: make_t2922_reset_guard_fixture.py <base.sslm> <layer> <out.sslm>")
source, layer_text, destination = sys.argv[1:]
data = bytearray(open(source, "rb").read())
section = None
for index in range(struct.unpack_from("<I", data, 12)[0]):
    kind, _, offset, size, _, _, _ = struct.unpack_from("<IIQQQII", data, 64 + 40 * index)
    if kind == 7:
        section = (offset, size)
if section is None:
    raise SystemExit("missing CompositionConstants")
offset, _ = section
magic, _, count, width, _, _ = struct.unpack_from("<4sIIIII", data, offset)
if magic != b"KVC1" or width != 2:
    raise SystemExit("unexpected CompositionConstants")
descriptors = offset + 24; values = descriptors + 8 * count; names = values + 16 * count
target = f"layer{int(layer_text)}.attn_norm"; hits = 0
for index in range(count):
    name_offset, name_length = struct.unpack_from("<II", data, descriptors + 8 * index)
    name = bytes(data[names + name_offset:names + name_offset + name_length]).decode()
    if name == target:
        struct.pack_into("<q", data, values + 16 * index, -(1 << 30)); hits += 1
if hits != 1:
    raise SystemExit(f"patched {hits}, want 1")
data[32:64] = bytes(32); data[32:64] = hashlib.sha256(data).digest()
with open(destination, "wb") as output:
    output.write(data)
print(f"patched={target} sha256={hashlib.sha256(data).hexdigest()} bytes={len(data)}")
