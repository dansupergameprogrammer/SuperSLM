"""L1 debunker audit of the regenerated precache reference population (E4, 2026-10-22).

Independent of the generator: every header identity is recomputed from the real input files, the bodies are parsed
and screened, the pairs compared, each named discrimination record is opened and its standing read, and the streams
are compared with the 5d5ca58 population (same model, same prompts, B=16 then).

usage: python -I audit.py <population> <old-population> <prompts> <model> <v1-records> <out-json>
"""
import hashlib
import json
import sys
from pathlib import Path

ANCHOR = "1948e1f2ea4528790dc6701e26c47bda5739b003"
HARNESS = "e8c029ed68f6d3fb0a728673c9a41d16d433916d"
TOKENIZER = "TokenizerView::Encode(rel10,4da4c4c)"
VOCAB = 151936


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


def main():
    pop, old, prompts, model, v1, out = map(Path, sys.argv[1:7])
    model_sha = sha(model)
    findings, files = [], []
    def bad(msg):
        findings.append(msg)

    groups = {"P1-real0p5-flat": ("flat", 10), "P2-real0p5-nested": ("nested", 10)}
    for group, (kind, n) in groups.items():
        for i in range(n):
            stem = f"{kind}{i:02d}"
            r1, r2 = pop / group / f"{stem}-run1.ref", pop / group / f"{stem}-run2.ref"
            rec = {"stem": stem, "group": group}
            b1, b2 = r1.read_bytes(), r2.read_bytes()
            rec["pair_equal"] = b1 == b2
            if b1 != b2:
                bad(f"{stem}: run1 != run2")
            lines = b1.decode("ascii").splitlines()
            if not lines[0].startswith("# "):
                bad(f"{stem}: no header")
            h = fields(lines[0][2:])
            rec["header"] = h
            expect = {
                "anchor": ANCHOR, "harness": HARNESS, "model_sha256": model_sha, "B": "128", "cap": "32768",
                "decode_modes_available": "1", "tokenizer": TOKENIZER,
                "prompt_ids_sha256": sha(prompts / f"{stem}.ids"),
                "prompt_source_sha256": sha(prompts / f"{stem}.txt"),
            }
            if kind == "nested":
                expect["world_ids_sha256"] = sha(prompts / "world.ids")
                expect["world_source_sha256"] = sha(prompts / "world.txt")
                p = f"persona{i:02d}"
                expect["persona_ids_sha256"] = sha(prompts / f"{p}.ids")
                expect["persona_source_sha256"] = sha(prompts / f"{p}.txt")
            for k, v in expect.items():
                if h.get(k) != v:
                    bad(f"{stem}: header {k}={h.get(k)} expected {v}")
            extra = set(h) - set(expect) - {"discrimination_record_sha256"}
            if extra:
                bad(f"{stem}: unexpected header fields {sorted(extra)}")
            record = v1 / ("real0p5-" + kind) / f"{stem}.discrimination.json"
            rsha = sha(record)
            rec["record"] = str(record)
            rec["record_sha256_matches_header"] = rsha == h.get("discrimination_record_sha256")
            if not rec["record_sha256_matches_header"]:
                bad(f"{stem}: discrimination_record_sha256 does not match {record}")
            r = json.loads(record.read_text(encoding="utf-8"))
            summ = r.get("population_summary", {})
            rec["record_standing"] = {
                "instrument": r.get("instrument"), "status": r.get("status"),
                "thresholds_met": summ.get("thresholds_met"),
                "population_adequate": r.get("population_adequate"),
                "page_positions": r.get("page_positions"),
                "target_ids_sha256_matches": r.get("target_ids_sha256") == expect["prompt_ids_sha256"],
            }
            # body
            rows = [fields(l) for l in lines[1:] if l.strip()]
            rec["rows"] = len(rows)
            seen = set()
            screens = []
            for row in rows:
                a = int(row["adopter"])
                key = (a, row["decode_mode"])
                if key in seen:
                    bad(f"{stem}: duplicate row {key}")
                seen.add(key)
                if row["prefix_id"] != stem:
                    bad(f"{stem}: row prefix_id {row['prefix_id']}")
                if row["adopter_mode"] != ("budget" if a % 2 == 0 else "legacy"):
                    bad(f"{stem}: adopter {a} mode {row['adopter_mode']}")
                toks = [int(t) for t in row["tokens"].split(",")]
                if len(toks) != 64:
                    bad(f"{stem}: adopter {a} has {len(toks)} tokens")
                if any(t < 0 or t >= VOCAB for t in toks):
                    bad(f"{stem}: adopter {a} token out of range")
                screens.append(len(set(toks[-16:])))
            rec["adopters"] = sorted(seen)
            rec["last16_distinct_min"] = min(screens) if screens else None
            # old population comparison (token streams only; headers differ by design)
            o = old / group / f"{stem}-run1.ref"
            if o.is_file():
                orows = {(int(x["adopter"]), x["decode_mode"]): x["tokens"]
                         for x in (fields(l) for l in o.read_text(encoding="ascii").splitlines()[1:] if l.strip())}
                same = sum(1 for row in rows if orows.get((int(row["adopter"]), row["decode_mode"])) == row["tokens"])
                rec["streams_equal_to_5d5ca58"] = f"{same}/{len(rows)}"
            files.append(rec)

    standing = sorted({json.dumps(f["record_standing"], sort_keys=True) for f in files})
    result = {"findings": findings, "files": files, "model_sha256": model_sha,
              "distinct_record_standings": [json.loads(s) for s in standing]}
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"files {len(files)}; mechanical findings {len(findings)}")
    for f in findings:
        print("FINDING", f)
    for s in result["distinct_record_standings"]:
        print("record standing", json.dumps(s, sort_keys=True))
    for f in files:
        print(f["stem"], "rows", f["rows"], "adopters", len(f["adopters"]), "last16min", f["last16_distinct_min"],
              "vs5d5ca58", f.get("streams_equal_to_5d5ca58"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
