"""Build the T-3030 commissioning artifacts for the certified-artifact schema-hash audit.

Each output is a complete `.sslm`: every section of a real base artifact carried unchanged
(as tools/t2132_build_g5_fixture.py carries them), plus one SCM1 section compiled against the
base's real tokenizer vocabulary.

  must-accept-renamed-pair.sslm   names [P1z16@?^_s(Q, abcdefgi]  -- production _serialize_scm1
  must-reject-two-name.sslm       names [P1z16@?^_s(Q, abcdefgh]  -- scm1_writer (equal hashes)
  must-reject-zero-hash.sslm      names [<zero>, t3030_companion]  -- scm1_writer (hash 0), when
                                  --zero-name is given

Usage: python build_constructions.py --base <no-SCM1 .sslm> --ckpt <dir> --out-dir <dir>
       [--zero-name <name>]
"""
import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import scm1_writer  # noqa: E402
import sslm_format as F  # noqa: E402
from t2132_build_g5_fixture import _serialize_scm1  # noqa: E402  (must-accept only)


def write_with(base, out, scm1_bytes):
    header = F.read_header(base)
    if any(s["type"] == F.SectionType.SCHEMA_MASKS for s in header["sections"]):
        raise SystemExit(f"{base}: already carries an SCM1 section")
    sections = []
    for s in header["sections"]:
        data = header["data"][s["offset"]:s["offset"] + s["byte_size"]]
        sections.append(F.Section(type=s["type"], data=bytes(data), dtype=s["dtype"],
                                  elem_count=s["elem_count"], alignment=s["alignment"]))
    sections.append(F.Section(type=F.SectionType.SCHEMA_MASKS, data=scm1_bytes))
    fp = F.write_artifact(out, sections, flags=header["flags"])
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zero-name", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = F.read_section_bytes(a.base, F.SectionType.CONFIG)
    (vocab_size,) = struct.unpack_from("<I", cfg, 32)
    vocab, specials = common.real_vocab(a.ckpt, vocab_size)
    mp_a, mp_b = common.compile_all([common.SCHEMA_A, common.SCHEMA_B], vocab, specials)

    plan = {
        "must-accept-renamed-pair.sslm": ([(common.COLLIDING_A, mp_a), (common.RENAMED_B, mp_b)],
                                          _serialize_scm1),
        "must-reject-two-name.sslm": ([(common.COLLIDING_A, mp_a), (common.COLLIDING_B, mp_b)],
                                      scm1_writer.write_scm1),
    }
    if a.zero_name is not None:
        zb = a.zero_name.encode("utf-8")
        if common.fnv1a64(zb) != 0:
            raise SystemExit("--zero-name does not hash to 0")
        plan["must-reject-zero-hash.sslm"] = ([(a.zero_name, mp_a), ("t3030_companion", mp_b)],
                                              scm1_writer.write_scm1)

    manifest = {"base": a.base, "base_sha256": common.sha256_hex(Path(a.base).read_bytes()),
                "ckpt": a.ckpt, "vocab_size": vocab_size, "artifacts": {}}
    for fname, (named, serializer) in plan.items():
        scm1 = serializer(named, vocab_size)
        out = out_dir / fname
        fp = write_with(a.base, str(out), scm1)
        manifest["artifacts"][fname] = {
            "serializer": serializer.__module__ + "." + serializer.__name__,
            "names": [n for n, _ in named],
            "fnv1a64": [f"{common.fnv1a64(n.encode()):016x}" for n, _ in named],
            "state_counts": [scm1_writer._dense_state_count(mp) for _, mp in named],
            "scm1_bytes": len(scm1), "scm1_sha256": common.sha256_hex(scm1),
            "fingerprint": fp, "sha256": common.sha256_hex(out.read_bytes()),
        }
        print(fname, json.dumps(manifest["artifacts"][fname]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
