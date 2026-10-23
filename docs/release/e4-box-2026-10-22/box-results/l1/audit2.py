"""L1 re-run (debunker): audit of the v2-bound population at 1948e1f2 and its manifest (E4, rev 1.18.13a).

Independent of the generator. For each of the 20 stems:
- run1 == run2;
- every header field is recomputed from the real inputs: model, prompt ids and source, world and persona, anchor,
  harness 47bdcc9e, B=128, cap, tokenizer;
- discrimination_record_sha256 == SHA-256 of discrimination-v2/<group>/<stem>.stream.json, never a v1 hash;
- the sidecar: instrument T-3035-precache-stream-v2, model_sha256 == the box artifact, length == the stem's prompt
  id count, target_ids_sha256 == the prompt ids. Its reference_sha256 names the 5d5ca58 file, and that file's token
  rows equal the new reference's rows (A0's two suggested checks);
- the byte-v1 sidecar: present, same model, length and target; recorded as provenance;
- the body: 5 adopters, mode parity, 64 in-vocabulary tokens, and the tail screen.
Writes the manifest (deterministic JSON) when there are zero findings.

usage: python -I audit2.py <population> <old-population> <prompts> <model> <v2-dir> <v1-dir> <registry> <b5-log>
                           <out-audit-json> <out-manifest>
"""
import hashlib
import json
import sys
from pathlib import Path

ANCHOR = "1948e1f2ea4528790dc6701e26c47bda5739b003"
HARNESS = "47bdcc9e24917f91e5f9bf67e8a0141662ca5114"
TOKENIZER = "TokenizerView::Encode(rel10,4da4c4c)"
VOCAB = 151936
GROUPS = {"P1-real0p5-flat": "flat", "P2-real0p5-nested": "nested"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fields(line):
    out = {}
    for tok in line.split():
        k, _, v = tok.partition("=")
        if k in out:
            raise ValueError(f"duplicate field {k}")
        out[k] = v
    return out


def rows_of(path):
    lines = path.read_text(encoding="ascii").splitlines()
    return lines[0], [fields(l) for l in lines[1:] if l.strip()]


def main():
    pop, old, prompts, model, v2, v1, registry, b5, out_audit, out_manifest = map(Path, sys.argv[1:11])
    findings = []
    bad = findings.append
    model_sha = sha(model)
    reg = json.loads(registry.read_text(encoding="utf-8"))
    reg = reg.get("instruments", reg)
    standing = {i: reg.get(i, {}).get("status") for i in ("T-3035-precache-stream-v2", "T-3035-precache-byte-v1")}
    for i, s in standing.items():
        if s != "COMMISSIONED":
            bad(f"registry: {i} is {s}")
    v1_hashes = {sha(p) for p in v1.rglob("*.discrimination.json")}
    b5_text = b5.read_text(encoding="utf-8")
    if b5_text.count("pair-equal") != 20 or "pairs differing: 0" not in b5_text or b5_text.count("binds-v2") != 20 \
            or "headers not bound to their v2 sidecar: 0" not in b5_text:
        bad("B5 log does not read 20 pair-equal, 0 differing, 20 binds-v2, 0 unbound")
    files, streams, bytes_ = [], [], []
    for group, kind in GROUPS.items():
        for i in range(10):
            stem = f"{kind}{i:02d}"
            r1, r2 = pop / group / f"{stem}-run1.ref", pop / group / f"{stem}-run2.ref"
            if r1.read_bytes() != r2.read_bytes():
                bad(f"{stem}: run1 != run2")
            header, rows = rows_of(r1)
            h = fields(header[2:]) if header.startswith("# ") else {}
            ids_path = prompts / f"{stem}.ids"
            n_ids = len(ids_path.read_text(encoding="ascii").split())
            expect = {"anchor": ANCHOR, "harness": HARNESS, "model_sha256": model_sha, "B": "128", "cap": "32768",
                      "decode_modes_available": "1", "tokenizer": TOKENIZER,
                      "prompt_ids_sha256": sha(ids_path), "prompt_source_sha256": sha(prompts / f"{stem}.txt")}
            if kind == "nested":
                p = f"persona{i:02d}"
                expect.update({"world_ids_sha256": sha(prompts / "world.ids"),
                               "world_source_sha256": sha(prompts / "world.txt"),
                               "persona_ids_sha256": sha(prompts / f"{p}.ids"),
                               "persona_source_sha256": sha(prompts / f"{p}.txt")})
            for k, v in expect.items():
                if h.get(k) != v:
                    bad(f"{stem}: header {k}={h.get(k)} expected {v}")
            extra = set(h) - set(expect) - {"discrimination_record_sha256"}
            if extra:
                bad(f"{stem}: unexpected header fields {sorted(extra)}")
            sp, bp = v2 / group / f"{stem}.stream.json", v2 / group / f"{stem}.byte.json"
            s_sha, b_sha = sha(sp), sha(bp)
            rec = h.get("discrimination_record_sha256")
            if rec != s_sha:
                bad(f"{stem}: header record {rec} is not {sp.name} ({s_sha})")
            if rec in v1_hashes:
                bad(f"{stem}: header binds a v1 record")
            s, b = json.loads(sp.read_text(encoding="utf-8")), json.loads(bp.read_text(encoding="utf-8"))
            if s.get("instrument") != "T-3035-precache-stream-v2" or b.get("instrument") != "T-3035-precache-byte-v1":
                bad(f"{stem}: sidecar instruments {s.get('instrument')}, {b.get('instrument')}")
            for name, side in (("stream", s), ("byte", b)):
                if side.get("model_sha256") != model_sha:
                    bad(f"{stem}: {name} sidecar model {side.get('model_sha256')}")
                if side.get("length") != n_ids:
                    bad(f"{stem}: {name} sidecar length {side.get('length')} vs {n_ids} prompt ids")
                if side.get("target_ids_sha256") != expect["prompt_ids_sha256"]:
                    bad(f"{stem}: {name} sidecar target ids differ")
            old_ref = old / group / f"{stem}-run1.ref"
            if s.get("reference_sha256") != sha(old_ref):
                bad(f"{stem}: stream sidecar reference_sha256 does not name {old_ref}")
            _, old_rows = rows_of(old_ref)
            key = lambda r: (r["adopter"], r["decode_mode"], r["adopter_mode"], r["tokens"])
            if sorted(map(key, old_rows)) != sorted(map(key, rows)):
                bad(f"{stem}: token rows differ from the 5d5ca58 reference the sidecar names")
            seen = set()
            tail_min = 64
            for row in rows:
                a = int(row["adopter"])
                if (a, row["decode_mode"]) in seen or row["prefix_id"] != stem:
                    bad(f"{stem}: row identity {a} {row['decode_mode']} {row['prefix_id']}")
                seen.add((a, row["decode_mode"]))
                if row["adopter_mode"] != ("budget" if a % 2 == 0 else "legacy"):
                    bad(f"{stem}: adopter {a} mode {row['adopter_mode']}")
                toks = [int(t) for t in row["tokens"].split(",")]
                if len(toks) != 64 or any(t < 0 or t >= VOCAB for t in toks):
                    bad(f"{stem}: adopter {a} token count or range")
                tail_min = min(tail_min, len(set(toks[-16:])))
            if len(rows) != 5:
                bad(f"{stem}: {len(rows)} rows")
            for run, path in (("run1", r1), ("run2", r2)):
                files.append({"path": f"{group}/{path.name}", "sha256": sha(path), "stem": stem, "run": run,
                              "discrimination_record_sha256": rec})
            streams.append({"stem": stem, "path": f"discrimination-v2/{group}/{sp.name}", "sha256": s_sha,
                            "instrument": s["instrument"], "page_positions": s.get("page_positions"),
                            "adopter": f"{s.get('format')} {s.get('adopter_mode')} {s.get('decode_mode')}",
                            "reference_sha256_5d5ca58": s.get("reference_sha256"), "tail16_distinct_min": tail_min})
            bytes_.append({"stem": stem, "path": f"discrimination-v2/{group}/{bp.name}", "sha256": b_sha,
                           "instrument": b["instrument"], "role": "provenance (exact-state operand), not in record_hashes"})
    audit = {"findings": findings, "registry_standing": standing, "model_sha256": model_sha,
             "records": streams, "byte_provenance": bytes_}
    out_audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"stems {len(streams)}; findings {len(findings)}; registry {standing}")
    for f in findings:
        print("FINDING", f)
    if findings:
        return 1
    manifest = {
        "format": "PC-POPULATION-MANIFEST-1",
        "population": "t3035-precache-refs-post-m1 (P1-real0p5-flat, P2-real0p5-nested)",
        "anchor": ANCHOR, "generator_commit": HARNESS, "model_sha256": model_sha,
        "sidecar_source": "origin/curie/t3035-precache-refs 972d5b7b tests/t3035-precache-refs/discrimination-v2",
        "record_instrument": "T-3035-precache-stream-v2 (registry: COMMISSIONED, D-SLM8030)",
        "provenance_instrument": "T-3035-precache-byte-v1 (registry: COMMISSIONED, D-SLM8030)",
        "discrimination_note": "discrimination measured at B = 16 (page_positions 16), one legacy greedy SSB5 adopter per "
                               "stem; the population is at B = 128 (plan rev 1.18.13 §15.33 (4)); B = 128 exact-state "
                               "kills are carried by other cells (A0 §3)",
        "b5_log_sha256": sha(b5), "files": files, "records": streams, "byte_provenance": bytes_,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("manifest", out_manifest, sha(out_manifest))
    print("record_hashes", ",".join(r["sha256"] for r in streams))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
