"""sslm_verify must refuse a SchemaMasks section the runtime loader refuses, and must still accept a
healthy one (array S2 finding, 2026-10-25; hardened in the S2 fix round).

Attaches IntentDoc v2 (`intents`, an array schema) to the converter-v2 synthetic base with
tools/sslm_attach_schema.py, passing the verifier under test as --verifier so the attach's own
verification gate runs that binary, and checks the attach's exit status. Then runs the verifier
under test on four artifacts, each the attached artifact with its SCM1 bytes rewritten:

    clean         SCM1 unchanged                                 must accept
    accept_shift  an accepting-state id moved by one (valid)     must accept
    offset_swap   two CSR state offsets swapped (non-monotonic)  must reject: SchemaMasksRejected, BadSchemaStateOffsets
    header_word   SCM1 header word at offset 16 raised by 4      must reject: SchemaMasksRejected, SchemaMasksOutOfBounds

A refusal counts only with the manifest's `reject_status` equal to SchemaMasksRejected and the
diagnostic naming the expected reason, so a crash or an unrelated rejection fails the check.
Collected by pytest as tests/t3094-multicommand-s0/converter-v2/test_cv2_verify_schema_masks.py.

    python tests/array-s2/verify_schema_masks_check.py [path/to/sslm_verify]
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
C = R / "tests" / "t3094-multicommand-s0" / "converter-v2"
for _p in (HERE, C, R / "tools", R / "tools" / "reference_pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = R / "tests" / "t3094-multicommand-s0" / "v2" / "intentdoc-v2.schema.json"

# (case, must accept, expected diagnostic substring when refused)
CASES = [
    ("clean", True, None),
    ("accept_shift", True, None),
    ("offset_swap", False, "BadSchemaStateOffsets"),
    ("header_word", False, "SchemaMasksOutOfBounds"),
]


def default_verifier() -> Path:
    for c in (R / "build" / "Release" / "sslm_verify.exe", R / "build" / "Release" / "sslm_verify",
              R / "build" / "sslm_verify.exe", R / "build" / "sslm_verify", R / "out" / "sslm_verify.exe"):
        if c.is_file():
            return c
    return R / "build" / "sslm_verify"


def run_verifier(verifier: Path, artifact: Path, work: Path) -> dict:
    manifest = work / (artifact.stem + ".json")
    if manifest.exists():
        manifest.unlink()
    p = subprocess.run([str(verifier), str(artifact), str(manifest)], capture_output=True, text=True)
    doc = {}
    if manifest.is_file():
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
    return {"returncode": p.returncode, "reject_status": doc.get("reject_status"),
            "diagnostic": doc.get("diagnostic", ""), "output": (p.stdout + p.stderr).strip()[-300:]}


def header_word(src: Path, out: Path) -> str:
    import sslm_format as SF
    from dataclasses import replace
    header = SF.read_header(str(src))
    sections = []
    for s in header["sections"]:
        data = bytes(header["data"][s["offset"]:s["offset"] + s["byte_size"]])
        sec = SF.Section(type=s["type"], data=data, dtype=s["dtype"], elem_count=s["elem_count"],
                         alignment=s["alignment"])
        if s["type"] == SF.SectionType.SCHEMA_MASKS:
            d = bytearray(data)
            struct.pack_into("<I", d, 16, struct.unpack_from("<I", d, 16)[0] + 4)
            sec = replace(sec, data=bytes(d))
        sections.append(sec)
    SF.write_artifact(str(out), sections, header["flags"])
    return "header_word: SCM1 u32 at offset 16 raised by 4"


def run(verifier: Path, work: Path | None = None) -> list[dict]:
    """Every case's result; each carries `ok`. The attach is the first entry."""
    import cv2_attach_fixtures as A
    import scm1_mutants as M
    verifier = Path(verifier).resolve()
    if not verifier.is_file():
        raise FileNotFoundError(f"verifier under test not found: {verifier} "
                                "(build it with `cmake --build build --target sslm_verify`)")
    work = Path(work or tempfile.mkdtemp(prefix="verify_scm1_"))
    fx = A.build(work / "fx")
    attached = work / "attached.sslm"
    p = subprocess.run([sys.executable, str(R / "tools" / "sslm_attach_schema.py"), "--base", str(fx["base"]),
                        "--ckpt", str(fx["ckpt"]), "--schema", f"intents:{SCHEMA}", "--out", str(attached),
                        "--verifier", str(verifier), "--force"], capture_output=True, text=True)
    results = [{"case": "attach", "ok": p.returncode == 0 and attached.is_file(), "returncode": p.returncode,
                "what": "sslm_attach_schema.py --verifier <binary under test>",
                "output": (p.stdout + p.stderr).strip()[-300:]}]
    if not results[0]["ok"]:
        return results
    for case, must_accept, reason in CASES:
        bad = work / f"{case}.sslm"
        what = header_word(attached, bad) if case == "header_word" else M.write_mutant_artifact(attached, case, bad)
        res = run_verifier(verifier, bad, work)
        if must_accept:
            ok = res["returncode"] == 0
        else:
            ok = (res["returncode"] == 1 and res["reject_status"] == "SchemaMasksRejected"
                  and reason in res["diagnostic"])
        results.append({"case": case, "ok": ok, "must": "accept" if must_accept else f"reject ({reason})",
                        "what": what, **res})
    return results


def main(argv: list[str]) -> int:
    verifier = Path(argv[0]) if argv else default_verifier()
    results = run(verifier)
    for r in results:
        print(f"{'PASS' if r['ok'] else 'FAIL'} {r['case']}: must {r.get('must', 'exit 0')}; exit {r['returncode']}"
              f" reject_status={r.get('reject_status')} diagnostic={r.get('diagnostic', '')!r}  [{r['what']}]")
        if not r["ok"]:
            print("   ", r["output"])
    failed = [r["case"] for r in results if not r["ok"]]
    if len(results) != len(CASES) + 1:
        failed.append("missing cases")
    print("FAIL: " + ", ".join(failed) if failed else f"PASS: {len(results)} of {len(results)} with {verifier}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
