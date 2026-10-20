"""T-3103 S0: attach the frozen G1 gate schema to a stock int8 artifact (plan rev 15 s0.1, s8 S0).

Test tooling derived from the T-2132 fixture route (tools/t2132_build_g5_fixture.py); not the S1 product
attachment command. For one base artifact it:
1. proves tokenizer identity: the checkpoint's serialized tokenizer tables must equal the artifact's own
   TOKENIZER section byte for byte (equal vocabulary size alone is not identity, plan s4.1);
2. compiles G1 at the real vocabulary with every special ID excluded, through the current production
   converter. G1g, the detection arm's schema (D-SLM8098 pending), is compiled and counted only, never
   attached, so the arm stays removable;
3. writes base sections plus one SCM1 section named intent_gate, preserving the base feature flags;
4. re-reads the written artifact and exports, for g1_attach_check.exe, the SCM1 section bytes, the real
   vocabulary's raw token bytes, and one G1 document the schema must accept and one it must reject.

Usage (from the SuperSLM tools directory on sys.path):
  py -3 attach_g1.py --base <stock.sslm> --ckpt <hf snapshot> --out <out.sslm> --export <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import sslm_convert_schema as SC  # noqa: E402
import sslm_format as F  # noqa: E402
import t2132_build_g5_fixture as T  # noqa: E402
from convert_tokenizer import TokenizerTables  # noqa: E402

V2 = HERE.parent
ACCEPT = ('{"intent":{"action":"fire","subject":"alpha wing","target":"north gate","condition":{"role":"start",'
          '"entity":"beacon","state":"lit","cmp":"none","value":"-","negated":false,"watch":"auto"},'
          '"depends_on_previous":false}}')
REJECT = ACCEPT.replace('"value":"-"', '"value":""')


def count(mp):
    states = set(mp.transitions) | {mp.start} | {t for row in mp.transitions.values() for t in row.values()}
    return {"states": len(states), "transitions": sum(len(r) for r in mp.transitions.values()),
            "accepting": len(mp.accepting)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    report = {"base": args.base, "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest()}
    header = F.read_header(args.base)
    if any(s["type"] == F.SectionType.SCHEMA_MASKS for s in header["sections"]):
        raise SystemExit("base already carries SCM1; no merge policy in test tooling")
    tok_section = F.read_section_bytes(args.base, F.SectionType.TOKENIZER)
    tables = TokenizerTables(args.ckpt)
    serialized = tables.serialize_tokenizer()
    report["tokenizer_identity"] = tok_section is not None and bytes(serialized) == bytes(tok_section)
    if not report["tokenizer_identity"]:
        print(json.dumps(report))
        raise SystemExit("TOKENIZER IDENTITY FAILED: checkpoint tables differ from the artifact's TOKENIZER section")
    vocab_size = T._read_config_vocab_size(args.base)
    vocab = T._real_vocab(args.ckpt, vocab_size)
    special = T._real_special_ids(args.ckpt)
    g1 = json.loads((V2 / "g1.schema.json").read_text(encoding="utf-8"))
    g1g = json.loads((V2 / "g1g.schema.json").read_text(encoding="utf-8"))
    mp = SC.compile_schema_to_mask_pages(g1, vocab, special_ids=special)
    report["g1"] = count(mp)
    report["g1g_compile_only"] = count(SC.compile_schema_to_mask_pages(g1g, vocab, special_ids=special))
    scm1 = T._serialize_scm1([("intent_gate", mp)], vocab_size)
    report["scm1_bytes"] = len(scm1)
    sections = []
    for s in header["sections"]:
        data = header["data"][s["offset"]:s["offset"] + s["byte_size"]]
        sections.append(F.Section(type=s["type"], data=bytes(data), dtype=s["dtype"], elem_count=s["elem_count"],
                                  alignment=s["alignment"]))
    sections.append(F.Section(type=F.SectionType.SCHEMA_MASKS, data=scm1))
    fp = F.write_artifact(args.out, sections, flags=header["flags"])
    report.update(out=args.out, fingerprint=fp, flags=hex(header["flags"]),
                  out_sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    # export from the WRITTEN artifact for the C++ loader check
    exp = Path(args.export)
    exp.mkdir(parents=True, exist_ok=True)
    written = F.read_section_bytes(args.out, F.SectionType.SCHEMA_MASKS)
    (exp / "scm1.bin").write_bytes(bytes(written))
    (exp / "accept.json").write_bytes(ACCEPT.encode("utf-8"))
    (exp / "reject.json").write_bytes(REJECT.encode("utf-8"))
    with open(exp / "vocab.hex", "w", encoding="ascii", newline="\n") as fh:
        for piece in vocab:
            fh.write(piece.hex() + "\n")
    report["vocab_size"] = T._read_config_vocab_size(args.out)
    report["elapsed_s"] = round(time.time() - t0, 1)
    (exp / "attach-report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
