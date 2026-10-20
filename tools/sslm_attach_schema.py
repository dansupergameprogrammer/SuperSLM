"""sslm_attach_schema.py -- the offline schema attachment command (T-3094 S1, plan of record
`Claude/Plans/superslm-multicommand-array/plan.md` Sec4.1, Wizard repo).

Takes a completed `.sslm` artifact, the exact tokenizer checkpoint it was built from, one or more
`(schema name, JSON Schema file)` inputs and an output path. Each schema is compiled by
`sslm_convert_schema.compile_schema_to_mask_pages` against the checkpoint's undecoded token bytes
with every special/added id excluded, the named `MaskPages` are serialized into one SCM1 section,
and that section is appended to the base artifact with every other section and every header flag
carried over unchanged. The independent artifact verifier (`sslm_verify`, the C++ loader) runs on
the base before anything is compiled, and on the new artifact before success is reported.

    python tools/sslm_attach_schema.py --base model.sslm --ckpt <checkpoint dir>
        --schema intents:intentdoc.schema.json [--schema NAME:PATH ...] --out model-with-schema.sslm
        [--limits-json limits.json] [--verifier build/sslm_verify] [--force]

Resource limits are per schema: each `--schema` is compiled under its own limits (the release limits,
or `--limits-json`), matching the runtime loader's per-schema bounds; the command adds no cap on the
combined SCM1 section of several schemas.

Refusals (non-zero exit, the reason on stderr, and no new file at `--out`):
- `--out` names the base itself (always), or a file that already exists (unless `--force`);
- the base is missing, not an intact `.sslm` (magic, length or integrity hash), or rejected by the
  verifier on its own (checked before any schema is compiled);
- the base already carries an SCM1 (schema-mask) section: replacing it needs a deterministic merge
  policy, and none is defined;
- the base has no TOKENIZER section (a model-only artifact): equal vocabulary size alone does not
  establish token-ID identity;
- the checkpoint's serialized tokenizer tables differ from the base's TOKENIZER section;
- a schema name that is empty, repeated, or whose FNV-1a-64 hash is zero (the runtime resolves a
  saved schema by that hash);
- a schema file that is missing or unreadable, or is not valid JSON;
- a `--limits-json` that is unreadable, not JSON, names an unknown limit, or gives a value that is not
  a non-negative integer;
- a schema the compiler refuses: an unsupported keyword, a resource limit (`--limits-json` replaces
  named limits, as `compile_schema_to_mask_pages(limits=...)` does), or a table that cannot finish;
- the verifier is missing or rejects the new artifact.

The output is written to a temporary file beside `--out`, flushed and synced to disk, verified there,
given the ordinary file mode (0666 less the umask), and only then renamed into place (the directory is
synced after the rename), so no refusal leaves a partial artifact at the output path.
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


def _check_out(base: Path, out: Path, force: bool) -> None:
    same = out.resolve() == base.resolve()
    if not same and out.exists() and base.exists():
        same = os.path.samefile(out, base)
    if same:
        raise AttachRefused(f"--out {out} is the base artifact itself; the base is never replaced")
    if out.exists() and not force:
        raise AttachRefused(f"--out {out} already exists; pass --force to replace it")


def _read_base(path: Path) -> dict:
    try:
        header = F.read_header(str(path))
    except OSError as error:
        raise AttachRefused(f"base artifact {path}: cannot be read ({error})") from error
    except (ValueError, struct.error) as error:
        raise AttachRefused(f"base artifact {path}: not an intact .sslm ({error})") from error
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
    try:
        limits = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AttachRefused(f"--limits-json {path}: cannot be read ({error})") from error
    except ValueError as error:
        raise AttachRefused(f"--limits-json {path}: not valid JSON ({error})") from error
    if not isinstance(limits, dict):
        raise AttachRefused(f"--limits-json {path}: expected a JSON object of limit name to integer")
    unknown = sorted(set(limits) - set(SC.RELEASE_LIMITS))
    if unknown:
        raise AttachRefused(f"--limits-json {path}: unknown limit(s) {unknown}; the limits are {sorted(SC.RELEASE_LIMITS)}")
    for key, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AttachRefused(f"--limits-json {path}: limit {key!r} must be a non-negative integer, not {value!r}")
    return limits


def _read_schema(name: bytes, path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AttachRefused(f"schema {name!r}: file {path} cannot be read ({error})") from error
    except ValueError as error:
        raise AttachRefused(f"schema {name!r}: file {path} is not valid JSON ({error})") from error


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


def _run_verifier(verifier_path: str, artifact: Path, scratch_dir: Path) -> subprocess.CompletedProcess:
    """Runs the verifier on `artifact`; its manifest goes to a scratch file that is always removed."""
    fd, manifest_name = tempfile.mkstemp(prefix=".sslm_attach.", suffix=".verify.json", dir=str(scratch_dir))
    os.close(fd)
    manifest = Path(manifest_name)
    try:
        return subprocess.run([verifier_path, str(artifact), str(manifest)], capture_output=True, text=True)
    except OSError as error:
        raise AttachRefused(f"artifact verifier {verifier_path} cannot be run ({error})") from error
    finally:
        if manifest.exists():
            manifest.unlink()


def _rejection(proc: subprocess.CompletedProcess) -> str:
    return f"exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-600:]}"


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return  # a platform without directory handles (Windows) has nothing to sync here
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def attach(base: Path, ckpt: Path, schemas: list[tuple[bytes, Path]], out: Path,
           limits_json: Path | None = None, verifier: str | None = None, force: bool = False) -> dict:
    """Builds, verifies and writes the attached artifact; raises `AttachRefused` on any refusal
    (nothing new exists at `out` afterwards unless the call succeeded)."""
    from convert_tokenizer import TokenizerTables

    _check_out(base, out, force)
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
    try:
        tables = TokenizerTables(str(ckpt))
    except (OSError, ValueError, KeyError) as error:
        raise AttachRefused(f"tokenizer checkpoint {ckpt}: cannot be loaded ({error})") from error
    if bytes(tables.serialize_tokenizer()) != tokenizer_sections[0]:
        raise AttachRefused(f"tokenizer mismatch: the serialized tokenizer tables of {ckpt} differ from the base "
                            f"artifact's TOKENIZER section, so its token ids are not the artifact's")

    verifier_path = _verifier(verifier)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = _run_verifier(verifier_path, base, out.parent)
    if proc.returncode != 0:
        raise AttachRefused(f"base artifact {base} is rejected by the independent artifact verifier on its own, "
                            f"before any schema is attached ({_rejection(proc)})")

    vocab_size = _config_vocab_size(sections)
    vocab = [tables.id_to_bytes[i] if i < len(tables.id_to_bytes) else b"" for i in range(vocab_size)]
    special = frozenset(token_id for _, token_id in tables.specials)

    compiled = []
    for name, path in schemas:
        schema = _read_schema(name, path)
        try:
            mp = SC.compile_schema_to_mask_pages(schema, vocab, special_ids=special, limits=limits, name=name)
        except SC.SchemaCompileError as error:
            raise AttachRefused(f"schema {name!r} ({path}) refused by the compiler: {error}") from error
        except (TypeError, ValueError) as error:
            raise AttachRefused(f"schema {name!r} ({path}) cannot be compiled: {error}") from error
        compiled.append((name, mp))
    scm1 = SC.serialize_scm1(compiled, vocab_size)

    data, fingerprint = F.build_artifact(sections + [F.Section(F.SectionType.SCHEMA_MASKS, scm1)],
                                         flags=artifact["flags"])
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".partial", dir=str(out.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        proc = _run_verifier(verifier_path, tmp, out.parent)
        if proc.returncode != 0:
            raise AttachRefused(f"verification failed: the independent artifact verifier rejected the attached "
                                f"artifact ({_rejection(proc)})")
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, out)
        _fsync_dir(out.parent)
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
    ap.add_argument("--out", required=True, type=Path, help="the output artifact path (never the base)")
    ap.add_argument("--limits-json", type=Path, default=None,
                    help="a JSON object replacing named compiler resource limits for this run (per schema)")
    ap.add_argument("--verifier", default=None, help="path to the sslm_verify binary (default: searched under build/)")
    ap.add_argument("--force", action="store_true", help="replace an existing --out (never the base)")
    args = ap.parse_args(argv)
    try:
        report = attach(args.base, args.ckpt, args.schema, args.out, args.limits_json, args.verifier, args.force)
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
