"""Attachment-command fixtures, derived from the T-2132 fixture mechanism (tools/t2132_build_g5_fixture.py:
read or build sections, `sslm_format.build_artifact`, `TokenizerTables.serialize_tokenizer`,
`_serialize_scm1`). Test tooling only; the command under test is tools/sslm_attach_schema.py.

Base artifact: when T3094_COMPLETE_ARTIFACT and T3094_MATCHING_CHECKPOINT name a real complete `.sslm` and its
checkpoint, every fixture derives from it. Otherwise the base is the T-2199 S8 synthetic complete model
(every tensor the engine requires) converted at the pinned tokenizer's padded vocabulary size, plus that
tokenizer's TOKENIZER section, so the cloud can run the contract without a model download.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import cv2_fixtures as F

_ENV_BASE, _ENV_CKPT = "T3094_COMPLETE_ARTIFACT", "T3094_MATCHING_CHECKPOINT"


def _tools():
    import sys
    for p in (F.TOOLS, F.TOOLS / "reference_pipeline"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import sslm_format as SF
    from convert_tokenizer import TokenizerTables
    return SF, TokenizerTables


def _sections_of(SF, raw_path):
    header = SF.read_header(str(raw_path))
    out = []
    for s in header["sections"]:
        data = header["data"][s["offset"]:s["offset"] + s["byte_size"]]
        out.append(SF.Section(type=s["type"], data=bytes(data), dtype=s["dtype"], elem_count=s["elem_count"],
                              alignment=s["alignment"]))
    return out, header["flags"]


def _synthetic_base(SF, TokenizerTables, ckpt):
    import convert_model as CM
    import pipeline as PL
    import _t2199_s8_synthetic_full_model_fixture as S8
    cfg = replace(S8.build_config(), vocab_size=F.REAL_VOCAB_SIZE)
    model = PL.fixture_model(cfg)
    model = replace(
        model,
        weights={k: v for k, v in model.weights.items() if not k.endswith((".q_norm.gain", ".k_norm.gain"))},
        weight_scales={k: v for k, v in model.weight_scales.items() if not k.endswith((".q_norm.gain", ".k_norm.gain"))},
    )
    sections, _ = CM.build_sections(model)
    tables = TokenizerTables(str(ckpt))
    # The format pairs TOKENIZER with UNICODE_TABLES: the verifier rejects an artifact carrying one without the
    # other (src/model.cpp, "TokenizerRejected"), so a base with TOKENIZER alone is invalid on its own.
    sections.append(SF.Section(SF.SectionType.TOKENIZER, bytes(tables.serialize_tokenizer())))
    sections.append(SF.Section(SF.SectionType.UNICODE_TABLES, bytes(tables.u.serialize())))
    return sections, CM.artifact_flags_for_model(model)


def TOKENIZER_PAIR(SF):
    return (SF.SectionType.TOKENIZER, SF.SectionType.UNICODE_TABLES)


def build(workdir: Path) -> dict:
    """Write every fixture under `workdir` and return their paths."""
    SF, TokenizerTables = _tools()
    workdir.mkdir(parents=True, exist_ok=True)
    if os.environ.get(_ENV_BASE) and os.environ.get(_ENV_CKPT):
        ckpt = Path(os.environ[_ENV_CKPT])
        sections, flags = _sections_of(SF, Path(os.environ[_ENV_BASE]))
        sections = [s for s in sections if s.type != SF.SectionType.SCHEMA_MASKS]
    else:
        ckpt = F.TOKENIZER_DIR
        sections, flags = _synthetic_base(SF, TokenizerTables, ckpt)

    def write(name, secs, fl=flags):
        path = workdir / name
        path.write_bytes(SF.build_artifact(secs, flags=fl)[0])
        return path

    out = {"ckpt": ckpt, "base": write("base.sslm", sections)}
    out["model_only"] = write("model-only.sslm", [s for s in sections if s.type not in TOKENIZER_PAIR(SF)])

    import sslm_convert_schema as SC
    from t2132_build_g5_fixture import _serialize_scm1
    vocab_size = F.REAL_VOCAB_SIZE
    tables = TokenizerTables(str(ckpt))
    vocab = [tables.id_to_bytes[i] if i < len(tables.id_to_bytes) else b"" for i in range(vocab_size)]
    special = frozenset(i for _, i in tables.specials)
    minimal = SC.compile_schema_to_mask_pages({"type": "object", "properties": {"ok": {"type": "boolean"}}},
                                              vocab, special_ids=special)
    out["with_scm1"] = write("with-scm1.sslm", sections + [
        SF.Section(SF.SectionType.SCHEMA_MASKS, _serialize_scm1([("g5_minimal_one_field", minimal)], vocab_size))])

    damaged = []
    for s in sections:
        if s.type == SF.SectionType.SIGMOID_LUT:
            data = bytearray(s.data)
            data[len(data) // 2] ^= 0x01
            s = SF.Section(type=s.type, data=bytes(data), dtype=s.dtype, elem_count=s.elem_count, alignment=s.alignment)
        damaged.append(s)
    out["bad_sil1"] = write("bad-sil1.sslm", damaged)

    # Same vocabulary size, different token-ID assignment: swap the ids of two ordinary multi-byte tokens.
    wrong = workdir / "wrong-ckpt" / ckpt.name
    wrong.mkdir(parents=True, exist_ok=True)
    for f in ckpt.iterdir():
        if f.is_file():
            shutil.copy2(f, wrong / f.name)
    tj = json.loads((wrong / "tokenizer.json").read_text(encoding="utf-8"))
    by_id = {v: k for k, v in tj["model"]["vocab"].items()}
    a, b = by_id[1000], by_id[1001]
    tj["model"]["vocab"][a], tj["model"]["vocab"][b] = 1001, 1000
    (wrong / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    out["wrong_ckpt"] = wrong

    for name, schema in (("intentdoc", F.INTENTDOC), ("g1", F.G1)):
        path = workdir / f"{name}.schema.json"
        path.write_text(json.dumps(schema), encoding="utf-8")
        out[name] = path
    return out
