"""T-2700 capture-input preparation and pass-A table union.

The compiled driver owns integer capture.  This file only materializes the
already-pinned calibration token corpus and rebuilds QKC1 through the ordinary
artifact-cache and converter writer; it never reimplements a forward kernel.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from reference_pipeline import artifact_cache, pipeline


def _tokens(path: Path):
    return [int(value) for value in path.read_text(encoding="utf-8").split()]


def _write_tokens(path: Path, tokens):
    path.write_text(" ".join(str(token) for token in tokens) + "\n", encoding="utf-8")


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = pipeline._checkpoint_tokenize_prompt(args.checkpoint)
    rows = [tokenizer(pipeline.run_prompt_messages(record))
            for record in pipeline.calibration_records()]
    prefix = list(rows[0])
    for row in rows[1:]:
        limit = min(len(prefix), len(row))
        prefix = prefix[:next((i for i in range(limit) if prefix[i] != row[i]), limit)]
    if len(rows) != 600 or len(prefix) != 453 or any(not row[453:] for row in rows):
        raise RuntimeError(f"unexpected calibration token shape: rows={len(rows)} prefix={len(prefix)}")
    _write_tokens(out / "prefix-453.txt", prefix)
    suffixes = []
    for index, row in enumerate(rows):
        path = out / f"suffix-{index:03d}.txt"
        _write_tokens(path, row[453:])
        suffixes.append(path.name)
    manifest = {
        "records": len(rows), "prefix_tokens": len(prefix),
        "suffix_tokens": sum(len(row) - len(prefix) for row in rows),
        "prefix_sha256": hashlib.sha256((out / "prefix-453.txt").read_bytes()).hexdigest(),
        "suffixes": suffixes,
    }
    (out / "capture-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


def _parse_capture(path: Path, expected):
    real = np.zeros(expected, dtype=np.float64)
    raw = np.zeros(expected, dtype=np.uint64)
    scales = {}
    saturation = None
    rows = 0
    with path.open("r", encoding="utf-8") as report:
        if report.readline().strip() != "T2700_FUSED_K_CAPTURE_V1":
            raise RuntimeError("unrecognized capture report")
        for line in report:
            fields = line.rstrip("\n").split("\t")
            if not fields or fields[0] == "layer":
                continue
            if fields[0] == "summary":
                if fields[1] == "landing_saturation_count":
                    saturation = int(fields[2])
                continue
            layer, head, channel = (int(fields[0]), int(fields[1]), int(fields[2]))
            raw[layer, head, channel] = int(fields[3])
            scale = (int(fields[4]), int(fields[5]))
            if layer in scales and scales[layer] != scale:
                raise RuntimeError(f"inconsistent KWideSourceScale for layer {layer}")
            scales[layer] = scale
            value = float.fromhex(fields[6])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(f"non-finite peak at {layer}/{head}/{channel}")
            real[layer, head, channel] = value
            rows += 1
    if rows != int(np.prod(expected)) or saturation is None or len(scales) != expected[0]:
        raise RuntimeError(f"incomplete capture report rows={rows} scales={len(scales)} saturation={saturation}")
    return raw, real, saturation, scales


def merge(args):
    source = Path(args.cache)
    out_cache = Path(args.out_cache)
    out_sslm = Path(args.out_sslm)
    if out_cache.exists() or out_sslm.exists():
        raise RuntimeError("pass-A outputs must not already exist")
    model = artifact_cache.load_artifact(source)
    expected = (model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.head_dim)
    captures = [_parse_capture(Path(report), expected) for report in args.capture_report]
    raw = np.maximum.reduce([capture[0] for capture in captures])
    integer_peak = np.maximum.reduce([capture[1] for capture in captures])
    saturation = sum(capture[2] for capture in captures)
    scales = captures[0][3]
    if any(capture[3] != scales for capture in captures[1:]):
        raise RuntimeError("KWideSourceScale differs between capture shards")
    float_peaks = getattr(model, "qk_channel_peaks", {})
    float_peak = np.stack([np.asarray(float_peaks[f"layer{layer}"], dtype=np.float64)
                           for layer in range(expected[0])])
    if float_peak.shape != expected:
        raise RuntimeError(f"float peak geometry {float_peak.shape}, expected {expected}")
    final_peak = np.maximum(float_peak, integer_peak)
    final_model = pipeline.with_provisional_qk_channel_table(
        model, {f"layer{layer}": final_peak[layer] for layer in range(expected[0])})
    artifact_cache.save_artifact(final_model, out_cache, args.checkpoint)
    out_sslm.parent.mkdir(parents=True, exist_ok=True)
    converter = Path(__file__).with_name("convert_model.py")
    convert_command = [sys.executable, str(converter), "--artifact", str(out_cache), "--out", str(out_sslm),
                       "--verifier", args.verifier]
    if args.skip_verify:
        convert_command.append("--skip-verify")
    subprocess.run(convert_command, check=True)
    summary = {
        "capture_report_sha256": [hashlib.sha256(Path(report).read_bytes()).hexdigest()
                                  for report in args.capture_report],
        "float_peak_sha256": hashlib.sha256(float_peak.tobytes()).hexdigest(),
        "integer_peak_sha256": hashlib.sha256(integer_peak.tobytes()).hexdigest(),
        "final_peak_sha256": hashlib.sha256(final_peak.tobytes()).hexdigest(),
        "raw_peak_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "landing_saturation_count": saturation,
        "wide_source_scales": {str(layer): list(scales[layer]) for layer in sorted(scales)},
        "artifact_sha256": hashlib.sha256(out_sslm.read_bytes()).hexdigest(),
    }
    (out_sslm.parent / "pass-a-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", required=True)
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.set_defaults(fn=prepare)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--cache", required=True)
    merge_parser.add_argument("--checkpoint", required=True,
                              help="checkpoint directory to persist with the merged cache")
    merge_parser.add_argument("--capture-report", required=True, nargs="+")
    merge_parser.add_argument("--out-cache", required=True)
    merge_parser.add_argument("--out-sslm", required=True)
    merge_parser.add_argument("--verifier", required=True)
    merge_parser.add_argument("--skip-verify", action="store_true",
                              help="fixture-only writer path; production merges verify independently")
    merge_parser.set_defaults(fn=merge)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
