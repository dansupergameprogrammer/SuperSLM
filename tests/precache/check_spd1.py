"""Independent struct/hashlib reader for the 1.18.0 delta prefix blob 'SPD1' (cell D15/E6).

Given a parent SPX1, its child's full SPX1 and the child's SPD1 (as cell D15/E6 writes them), it checks the
delta's layout against the plan's table (superslm-prefix-delta-1p18 §3) and rebuilds the child's SPX1 from the
parent and the delta by its own assemble, which must equal the engine's full save. --self-test also feeds it
mutated copies, each of which it must reject for the named reason.
"""

import argparse
import hashlib
from pathlib import Path
import struct
import sys


def u32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def u64(b: bytes, off: int) -> int:
    return struct.unpack_from("<Q", b, off)[0]


def sealed(b: bytes) -> bool:
    return hashlib.sha256(b[:28] + bytes(32) + b[60:]).digest() == b[28:60]


def reseal(b: bytearray) -> bytes:
    b[28:60] = hashlib.sha256(bytes(b[:28]) + bytes(32) + bytes(b[60:])).digest()
    return bytes(b)


def check(parent: bytes, full: bytes, delta: bytes) -> None:
    assert parent[:4] == b"SPX1" and full[:4] == b"SPX1", "parent and child are SPX1"
    assert delta[:4] == b"SPD1", "delta magic"
    assert sealed(parent) and sealed(full), "SPX1 digests"
    assert sealed(delta), "delta digest"
    runs, dim, hidden, reserved = (u32(delta, off) for off in (216, 220, 224, 228))
    assert reserved == 0, "reserved word"
    length, origin = u64(delta, 112), u64(delta, 120)
    assert 1 <= origin <= length, "origin within [1, L]"
    assert len(delta) == 232 + hidden + (length - origin) * runs * dim, "delta size"
    assert len(parent) == 184 + hidden + origin * runs * dim, "parent size at the origin"
    assert delta[4:28] == full[4:28] and delta[60:184] == full[60:184], "delta header fields equal the full save's"
    parent_rows = parent[184 + hidden:]
    assert hashlib.sha256(parent_rows).digest() == delta[184:216], "parent rows digest"
    assert parent[4:28] == delta[4:28] and parent[60:96] == delta[60:96], "parent identity"
    own = delta[232 + hidden:]
    rows = b"".join(parent_rows[r * origin * dim:(r + 1) * origin * dim] +
                    own[r * (length - origin) * dim:(r + 1) * (length - origin) * dim] for r in range(runs))
    assembled = bytearray(b"SPX1" + delta[4:184] + delta[232:232 + hidden] + rows)
    assert reseal(assembled) == full, "independent assemble equals the full save"


def self_test(parent: bytes, full: bytes, delta: bytes) -> int:
    check(parent, full, delta)
    hidden = u32(delta, 224)

    def flip(b: bytes, off: int, keep_seal: bool = True) -> bytes:
        m = bytearray(b)
        m[off] ^= 1
        return reseal(m) if keep_seal else bytes(m)

    rejects = [
        ("delta digest", parent, full, flip(delta, 232 + hidden + 5, keep_seal=False)),
        ("reserved word", parent, full, flip(delta, 228)),
        ("parent rows digest", flip(parent, len(parent) - 1), full, delta),
        ("independent assemble equals the full save", parent, full, flip(delta, len(delta) - 1)),
        ("delta header fields equal the full save's", parent, full, flip(delta, 144)),
        ("delta size", parent, full, reseal(bytearray(delta + b"\0"))),
        ("delta magic", parent, full, b"SPD9" + delta[4:]),
    ]
    for reason, p, f, d in rejects:
        try:
            check(p, f, d)
        except AssertionError as exc:
            assert str(exc) == reason, f"must-reject {reason!r} refused for {exc!r}"
            continue
        raise AssertionError(f"must-reject {reason!r} was accepted")
    print(f"check_spd1: accepted the engine's blobs and rejected {len(rejects)} mutants")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", type=Path, help="the directory cell D15/E6 wrote parent.spx1, child.spx1, child.spd1 to")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    blobs = [(args.dir / name).read_bytes() for name in ("parent.spx1", "child.spx1", "child.spd1")]
    try:
        if args.self_test:
            return self_test(*blobs)
        check(*blobs)
    except AssertionError as exc:
        print(f"check_spd1: {exc}", file=sys.stderr)
        return 1
    print("check_spd1: layout, digests, parent rows digest and the independent assemble match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
