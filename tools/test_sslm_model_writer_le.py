"""Curie's red suite for sslm_model_writer.py's explicit-little-endian fix
(S-HARDEN-3, F13): "the writer additionally relies on host-native NumPy byte
order while the format documents explicit little-endian."

Honest scope, checked by experiment before this suite was written: NumPy
fully normalizes a dtype's byte-order flag on construction, so on THIS
project's own CI matrix (Windows/Linux/macOS-ARM, all little-endian per
.github/workflows/tests.yml) `numpy.dtype('<i4')` and `numpy.dtype(np.int32)`
are indistinguishable by every introspection this module tried (`==`,
`.byteorder`, `.str` all agree). There is no little-endian-host test that can
observe the class of defect the fix addresses (a big-endian BUILD host
casting to its own native representation) -- confirmed by red-checking a
reverted (native-dtype) version of the writer against a "distinguish by
dtype introspection" test, which stayed green under the reverted code. That
test is not included here; asserting something the suite cannot actually
falsify is the mutation-pinning failure `StandardsDocument` and Curie's own
disciplines name ("a cell that passes under both the right and the wrong
claim pins nothing"). The fix itself rests on NumPy's own documented
guarantee that an explicit `'<i*'` dtype string denotes little-endian on any
host (verified at source, in NumPy's own dtype semantics, not by
construction over this project's code) -- see sslm_model_writer.py's
_LE_DTYPE comment for the full reasoning.

What IS directly testable, and is below: the tensor-manifest and SIL1 data
payloads are byte-for-byte the little-endian struct.pack encoding of their
source values -- pinning the exact layout the fix touches, even though it
cannot distinguish "correct because explicit" from "correct because this
host happens to be little-endian."
"""

import struct

import numpy as np

import sslm_model_writer as W


def test_write_tensor_manifest_tensor_payload_matches_independent_struct_pack():
    # A single rank-1 int32 tensor: the whole tensor-manifest byte layout up
    # to the data region is exercised by the existing C++ hostile-fixture
    # suite (mirrors SslmTensorManifest::Parse); this test isolates ONLY the
    # data payload's byte order against an independently-built struct.pack
    # reference, which is what the LE fix actually changes the source of.
    values = [1, -2, 2147483647, -2147483648]
    blob = W.write_tensor_manifest(W.WGT1, np.int32, {"t0": np.array(values, dtype=np.int64)})
    expected_payload = struct.pack("<4i", *values)
    assert blob.endswith(expected_payload), (
        "the tensor data payload must be the exact little-endian struct.pack encoding of the "
        "source values"
    )


def test_write_sil1_table_matches_independent_struct_pack():
    # write_sil1 requires exactly SIGMOID_LUT_ENTRIES nodes; a full-size table
    # with distinctive planted values at the first four positions and the
    # generated sigmoid values elsewhere (build_sigmoid_lut is deterministic
    # and does not need re-deriving here) exercises the real code path at
    # its real size, not a synthetic short table write_sil1 would reject.
    table = W.build_sigmoid_lut().copy()
    table[:4] = [100, -100, 0, 2147483647]
    blob = W.write_sil1(table=table)
    header, payload = blob[:16], blob[16:]
    expected_payload = struct.pack(f"<{W.SIGMOID_LUT_ENTRIES}i", *[int(v) for v in table])
    assert header[:4] == W.SIL1
    assert payload == expected_payload
