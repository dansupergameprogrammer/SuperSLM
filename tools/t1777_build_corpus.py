#!/usr/bin/env python3
"""T-1777 -- assembles the retrieval-agreement measurement's corpus and
writes the three input files the engine batch tool, the float batch tool,
and the report script each need.

In-distribution (domain "id"): a stratified, evenly-spaced sample of the
shopkeeper spike corpus (Claude/Docs/spike/shopkeeper_corpus_v1.jsonl, 600
records, frozen 2026-07-14, D-SLM447's own calibration corpus family),
proportional to each of its 6 classes so the sample's class mix matches the
full corpus's.

Out-of-distribution (domain "ood"): tools/t1777_ood_corpus.py's own 99
original sentences, written for this ticket, outside the shopkeeper booking
domain -- see that file's own docstring.

Outputs (all under --out-dir):
  prompts.tsv   -- <label>\t<chat-templated prompt, newlines as "<NL>">,
                   for tools/t1777_pooled_trace_batch.cpp / build_t1740_pooled_trace's exe
  docs.jsonl    -- {"label":..., "text":...} per line, for
                   tools/t1777_pooled_float_dump_batch.py
  manifest.jsonl -- {"label":..., "domain": "id"|"ood", "text":..., "class":...},
                   for tools/t1777_retrieval_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1777_ood_corpus import OOD_DOCUMENTS  # noqa: E402

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SHOPKEEPER_CORPUS = Path(r"D:\Wizard\Claude\Docs\spike\shopkeeper_corpus_v1.jsonl")


def build_prompt(user_text: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    """k evenly-spaced indices into range(n) (k <= n), spanning the full
    range rather than clustering at the start -- avoids a first-N sample
    picking up whatever stylistic drift exists early in a class's id
    numbering."""
    if k >= n:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) if k > 1 else 0 for i in range(k)]


def sample_id_corpus(n_target: int) -> list[dict]:
    by_class: dict[str, list[dict]] = {}
    with open(SHOPKEEPER_CORPUS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_class.setdefault(rec["class"], []).append(rec)

    total = sum(len(v) for v in by_class.values())
    out = []
    for cls, records in by_class.items():
        k = round(n_target * len(records) / total)
        idxs = evenly_spaced_indices(len(records), k)
        for i in idxs:
            rec = records[i]
            text = rec["utterance"] if "utterance" in rec else " ".join(rec["turns"])
            out.append({"id": rec["id"], "class": cls, "text": text})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-id", type=int, default=140)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_docs = sample_id_corpus(args.n_id)
    ood_docs = [{"id": f"ood-{i:03d}", "class": "ood", "text": t} for i, t in enumerate(OOD_DOCUMENTS)]

    all_docs = []
    for d in id_docs:
        all_docs.append({"label": f"id_{d['id']}", "domain": "id", "class": d["class"], "text": d["text"]})
    for d in ood_docs:
        all_docs.append({"label": f"ood_{d['id']}", "domain": "ood", "class": d["class"], "text": d["text"]})

    labels = set()
    for d in all_docs:
        if d["label"] in labels:
            raise SystemExit(f"duplicate label {d['label']}")
        labels.add(d["label"])

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(out_dir / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps({"label": d["label"], "text": d["text"]}, ensure_ascii=False) + "\n")

    with open(out_dir / "prompts.tsv", "w", encoding="utf-8") as f:
        for d in all_docs:
            p = build_prompt(d["text"]).replace("\n", "<NL>")
            f.write(f"{d['label']}\t{p}\n")

    n_id = sum(1 for d in all_docs if d["domain"] == "id")
    n_ood = sum(1 for d in all_docs if d["domain"] == "ood")
    print(f"corpus assembled: {len(all_docs)} documents ({n_id} id, {n_ood} ood) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
