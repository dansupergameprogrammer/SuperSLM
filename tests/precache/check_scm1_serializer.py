"""Converter-side PC-12 hash refusal, with the anchored two-argument signature."""

import inspect
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools"))
    import sslm_convert_schema as schema
    import t2132_build_g5_fixture as converter

    serialize = converter._serialize_scm1
    signature = inspect.signature(serialize)
    assert list(signature.parameters) == ["named_schemas", "vocab_size"], (
        "_serialize_scm1 signature changed; a bypass parameter is forbidden")
    assert all(p.default is inspect.Parameter.empty for p in signature.parameters.values())
    mp = schema.MaskPages([b"a", b"b"], {0: {0: 1}, 1: {}}, 0, frozenset({1}))

    def rejected(names: tuple[str, ...]) -> None:
        try:
            serialize([(name, mp) for name in names], 2)
        except (ValueError, RuntimeError, SystemExit) as exc:
            message = str(exc)
            assert all(name in message for name in names), (
                f"converter refusal did not name every affected schema: {message}")
            return
        raise AssertionError(f"_serialize_scm1 accepted colliding or zero-hash schemas: {names}")

    rejected(("P1z16@?^_s(Q", "abcdefgh"))
    rejected((r"VD4a7{kfW@@vq_OZTumf]HHH\LHQ=",))
    healthy = serialize([("P1z16@?^_s(Q", mp), ("abcdefgi", mp)], 2)
    assert healthy[:4] == b"SCM1" and len(healthy) > 24 + 2 * 56
    print("_serialize_scm1: collision and zero hash refused; renamed pair accepted")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ImportError) as exc:
        print(f"check_scm1_serializer: {exc}", file=sys.stderr)
        raise SystemExit(1)
