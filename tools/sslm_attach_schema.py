"""sslm_attach_schema.py -- the offline schema attachment command (T-3094 S1, plan of record
`Claude/Plans/superslm-multicommand-array/plan.md` Sec4.1, Wizard repo).

Takes a completed `.sslm` artifact, the exact tokenizer checkpoint it was built from, one or more
`(schema name, JSON Schema file)` inputs and an output path. Each schema is compiled by
`sslm_convert_schema.compile_schema_to_mask_pages` against the checkpoint's undecoded token bytes
with every special/added id excluded, the named `MaskPages` are serialized into one SCM1 section,
and that section is appended to the base artifact with every other section and every header flag
carried over unchanged. The independent artifact verifier (`sslm_verify`, the C++ loader) runs on
the new artifact before success is reported.

    python tools/sslm_attach_schema.py --base model.sslm --ckpt <checkpoint dir>
        --schema intents:intentdoc.schema.json [--schema NAME:PATH ...] --out model-with-schema.sslm
        [--limits-json limits.json] [--verifier build/sslm_verify]

Refusals (non-zero exit, the reason on stderr, and no file at `--out`):
- the base is not an intact `.sslm` (magic, length or integrity hash);
- the base already carries an SCM1 (schema-mask) section: replacing it needs a deterministic merge
  policy, and none is defined;
- the base has no TOKENIZER section (a model-only artifact): equal vocabulary size alone does not
  establish token-ID identity;
- the checkpoint's serialized tokenizer tables differ from the base's TOKENIZER section;
- a schema name that is empty, repeated, or whose FNV-1a-64 hash is zero (the runtime resolves a
  saved schema by that hash);
- a schema the compiler refuses: an unsupported keyword, a resource limit (`--limits-json` replaces
  named limits, as `compile_schema_to_mask_pages(limits=...)` does), or a table that cannot finish;
- the verifier is missing or rejects the new artifact.

The output is written to a temporary file beside `--out`, verified there, and only then renamed
into place, so no refusal leaves a partial artifact at the output path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sslm_format as F  # noqa: E402
import sslm_convert_schema as SC  # noqa: E402


class AttachRefused(Exception):
    """A refusal: the message is printed on stderr and nothing is written at the output path."""


def _schema_arg(raw: str) -> tuple[bytes, Path]:
    """`NAME:PATH`, split at the first colon. The name is taken as the raw argument bytes
    (`os.fsencode`), so a name that is not valid UTF-8 reaches the name checks unchanged."""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(f"--schema expects NAME:PATH, got {raw!r}")
    name, _, path = raw.partition(":")
    return os.fsencode(name), Path(path)


def _read_base(path: Path) -> dict:
    header = F.read_header(str(path))
    data = header["data"]
    if header["file_bytes"] != len(data):
        raise AttachRefused(f"base artifact {path}: header file_bytes {header['file_bytes']} != actual size {len(data)}")
    zeroed = bytearray(data)
    zeroed[F.INTEGRITY_HASH_OFFSET:F.INTEGRITY_HASH_OFFSET + F.INTEGRITY_HASH_BYTES] = bytes(F.INTEGRITY_HASH_BYTES)
    if hashlib.sha256(zeroed).digest() != header["integrity_hash"]:
        raise AttachRefused(f"base artifact {path}: integrity hash mismatch (not an intact .sslm)")
    sections = []
    for s in header["sections"]:
        sections.append(F.Section(type=s["type"], data=bytes(data[s["offset"]:s["offset"] + s["byte_size"]]),
                                  dtype=s["dtype"], elem_count=s["elem_count"], alignment=s["alignment"]))
    return {"flags": header["flags"], "sections": sections}


def _config_vocab_size(sections) -> int:
    config = [s.data for s in sections if s.type == F.SectionType.CONFIG]
    if len(config) != 1:
        raise AttachRefused(f"base artifact carries {len(config)} CONFIG sections, expected 1")
    # CFG1: vocab_size at payload offset 32 (docs/sslm_format.md).
    return struct.unpack_from("<I", config[0], 32)[0]


def _check_names(names: list[bytes]) -> None:
    seen = set()
    for name in names:
        if not name:
            raise AttachRefused("schema name is empty: every attached schema needs a non-empty name")
        if SC.fnv1a64(name) == 0:
            raise AttachRefused(f"schema name {name!r} has an FNV-1a-64 hash of zero, which the runtime "
                                f"reserves; choose another name")
        if name in seen:
            raise AttachRefused(f"schema name {name!r} is given more than once: schema names must be unique")
        seen.add(name)


def _load_limits(path: Path | None) -> dict | None:
    if path is None:
        return None
    limits = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(limits, dict):
        raise AttachRefused(f"--limits-json {path}: expected a JSON object of limit name to integer")
    unknown = sorted(set(limits) - set(SC.RELEASE_LIMITS))
    if unknown:
        raise AttachRefused(f"--limits-json {path}: unknown limit(s) {unknown}; the limits are {sorted(SC.RELEASE_LIMITS)}")
    return limits


def _verifier(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in (REPO_ROOT / "build" / "Release" / "sslm_verify.exe", REPO_ROOT / "build" / "Release" / "sslm_verify",
                      REPO_ROOT / "build" / "sslm_verify.exe", REPO_ROOT / "build" / "sslm_verify"):
        if candidate.is_file():
            return str(candidate)
    raise AttachRefused("artifact verifier sslm_verify not found (build it with "
                        "`cmake --build build --target sslm_verify`, or pass --verifier); "
                        "the attached artifact cannot be verified, so it is not written")


def attach(base: Path, ckpt: Path, schemas: list[tuple[bytes, Path]], out: Path,
           limits_json: Path | None = None, verifier: str | None = None) -> dict:
    """Builds, verifies and writes the attached artifact; raises `AttachRefused` on any refusal
    (nothing exists at `out` afterwards unless it existed before and the call succeeded)."""
    from convert_tokenizer import TokenizerTables

    names = [name for name, _ in schemas]
    _check_names(names)
    limits = _load_limits(limits_json)

    artifact = _read_base(base)
    sections = artifact["sections"]
    if any(s.type == F.SectionType.SCHEMA_MASKS for s in sections):
        raise AttachRefused("base artifact already carries an SCM1 (schema_masks) section; replacing or "
                            "merging an existing SCM1 needs a deterministic merge policy, and none is defined")
    tokenizer_sections = [s.data for s in sections if s.type == F.SectionType.TOKENIZER]
    if not tokenizer_sections:
        raise AttachRefused("base artifact has no TOKENIZER section (a model-only artifact): equal vocabulary "
                            "size alone does not establish token-ID identity, so no schema can be compiled for it")
    if len(tokenizer_sections) != 1:
        raise AttachRefused(f"base artifact carries {len(tokenizer_sections)} TOKENIZER sections, expected 1")
    tables = TokenizerTables(str(ckpt))
    if bytes(tables.serialize_tokenizer()) != tokenizer_sections[0]:
        raise AttachRefused(f"tokenizer mismatch: the serialized tokenizer tables of {ckpt} differ from the base "
                            f"artifact's TOKENIZER section, so its token ids are not the artifact's")

    verifier_path = _verifier(verifier)
    vocab_size = _config_vocab_size(sections)
    vocab = [tables.id_to_bytes[i] if i < len(tables.id_to_bytes) else b"" for i in range(vocab_size)]
    special = frozenset(token_id for _, token_id in tables.specials)

    compiled = []
    for name, path in schemas:
        schema = json.loads(path.read_text(encoding="utf-8"))
        try:
            mp = SC.compile_schema_to_mask_pages(schema, vocab, special_ids=special, limits=limits, name=name)
        except SC.SchemaCompileError as error:
            raise AttachRefused(f"schema {name!r} ({path}) refused by the compiler: {error}") from error
        compiled.append((name, mp))
    scm1 = SC.serialize_scm1(compiled, vocab_size)

    data, fingerprint = F.build_artifact(sections + [F.Section(F.SectionType.SCHEMA_MASKS, scm1)],
                                         flags=artifact["flags"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".partial", dir=str(out.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        manifest = tmp.with_suffix(".verify.json")
        try:
            proc = subprocess.run([verifier_path, str(tmp), str(manifest)], capture_output=True, text=True)
        finally:
            if manifest.exists():
                manifest.unlink()
        if proc.returncode != 0:
            raise AttachRefused(f"verification failed: the independent artifact verifier rejected the attached "
                                f"artifact (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[-600:]}")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"fingerprint": fingerprint, "scm1_bytes": len(scm1),
            "schemas": {name.decode("utf-8", "backslashreplace"): mp.resource_counts for name, mp in compiled}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--base", required=True, type=Path, help="the completed .sslm artifact")
    ap.add_argument("--ckpt", required=True, type=Path, help="the exact tokenizer checkpoint the artifact was built from")
    ap.add_argument("--schema", required=True, action="append", type=_schema_arg, metavar="NAME:PATH",
                    help="a schema name and its JSON Schema file (repeatable)")
    ap.add_argument("--out", required=True, type=Path, help="the output artifact path")
    ap.add_argument("--limits-json", type=Path, default=None,
                    help="a JSON object replacing named compiler resource limits for this run")
    ap.add_argument("--verifier", default=None, help="path to the sslm_verify binary (default: searched under build/)")
    args = ap.parse_args(argv)
    try:
        report = attach(args.base, args.ckpt, args.schema, args.out, args.limits_json, args.verifier)
    except AttachRefused as refusal:
        print(f"sslm_attach_schema: refused: {refusal}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    print(f"fingerprint {report['fingerprint']}")
    print(f"SCM1 bytes {report['scm1_bytes']}")
    for name, counts in report["schemas"].items():
        print(f"schema {name}: {json.dumps(counts, sort_keys=True)}")
    print("verified: independent loader accepted the artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
