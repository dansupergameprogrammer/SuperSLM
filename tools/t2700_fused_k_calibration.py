"""T-2700 fused-K calibration, capture, convergence, and artifact conversion.

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


class ChannelScaleDidNotConverge(RuntimeError):
    """Pass C exceeded the accepted clipped-landing rate."""


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


def _capture_metadata(path: Path):
    """Read the observation count and per-channel clipped-landings from a capture TSV."""
    callbacks = None
    saturation = None
    overshoots = []
    with path.open("r", encoding="utf-8") as report:
        if report.readline().strip() != "T2700_FUSED_K_CAPTURE_V1":
            raise RuntimeError("unrecognized capture report")
        for line in report:
            fields = line.rstrip("\n").split("\t")
            if not fields or fields[0] == "layer":
                continue
            if fields[0] == "summary":
                if fields[1] == "callback_count":
                    callbacks = int(fields[2])
                elif fields[1] == "landing_saturation_count":
                    saturation = int(fields[2])
                continue
            if len(fields) >= 8 and int(fields[7]) > 0:
                overshoots.append({"layer": int(fields[0]), "head": int(fields[1]),
                                   "channel": int(fields[2]), "count": int(fields[7])})
    if callbacks is None or callbacks <= 0 or saturation is None:
        raise RuntimeError(f"capture metadata incomplete: {path}")
    if saturation and not overshoots:
        raise RuntimeError(
            f"capture report has {saturation} clipped landings but no per-channel saturation column: {path}")
    if overshoots and sum(row["count"] for row in overshoots) != saturation:
        raise RuntimeError(f"capture per-channel saturation total disagrees: {path}")
    return callbacks, saturation, overshoots


def _convert(cache: Path, out_sslm: Path, verifier: str, skip_verify: bool):
    if out_sslm.exists():
        raise RuntimeError(f"conversion output already exists: {out_sslm}")
    out_sslm.parent.mkdir(parents=True, exist_ok=True)
    converter = Path(__file__).with_name("convert_model.py")
    command = [sys.executable, str(converter), "--artifact", str(cache), "--out", str(out_sslm),
               "--verifier", verifier]
    if skip_verify:
        command.append("--skip-verify")
    subprocess.run(command, check=True)


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
    _convert(out_cache, out_sslm, args.verifier, args.skip_verify)
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
    summary_path = Path(getattr(args, "summary", out_sslm.parent / "pass-a-summary.json"))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def _run_capture(capture: Path, artifact: Path, inputs: Path, report: Path):
    prefix = inputs / "prefix-453.txt"
    suffixes = sorted(inputs.glob("suffix-*.txt"))
    if not prefix.is_file() or len(suffixes) != 600:
        raise RuntimeError(f"capture inputs must contain one prefix and 600 suffixes: {inputs}")
    command = [str(capture)]
    if capture.suffix.lower() == ".py":
        command.insert(0, sys.executable)
    suffix_list = report.with_suffix(".suffixes.txt")
    suffix_list.write_text("\n".join(str(path) for path in suffixes) + "\n", encoding="utf-8")
    subprocess.run([*command, str(artifact), str(prefix), str(report), "--suffix-list", str(suffix_list)], check=True)


def _peak_table(model, capture: Path):
    expected = (model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.head_dim)
    return _parse_capture(capture, expected)[1]


def _flow_from_cache(args, source_cache: Path, inputs: Path, work: Path):
    """Run A/B/C from an already-calibrated cache; shared by the checkpoint CLI and fixture cell."""
    model = artifact_cache.load_artifact(source_cache)
    provisional = work / "provisional.sslm"
    report_a, report_b, report_c = (work / "pass-a.tsv", work / "pass-b.tsv", work / "pass-c.tsv")
    after_a_cache, after_a_sslm = work / "after-a-cache", work / "after-a.sslm"
    final_cache, final_sslm = work / "final-cache", Path(args.out)
    staged_sslm = work / "final-pending.sslm"
    _convert(source_cache, provisional, args.verifier, args.skip_verify)
    _run_capture(Path(args.capture), provisional, inputs, report_a)
    merge(argparse.Namespace(cache=str(source_cache), checkpoint=args.checkpoint,
                             capture_report=[str(report_a)], out_cache=str(after_a_cache),
                             out_sslm=str(after_a_sslm), verifier=args.verifier, skip_verify=args.skip_verify,
                             summary=str(work / "after-a-summary.json")))
    _run_capture(Path(args.capture), after_a_sslm, inputs, report_b)
    merge(argparse.Namespace(cache=str(source_cache), checkpoint=args.checkpoint,
                             capture_report=[str(report_a), str(report_b)], out_cache=str(final_cache),
                             out_sslm=str(staged_sslm), verifier=args.verifier, skip_verify=args.skip_verify,
                             summary=str(work / "final-summary.json")))
    _run_capture(Path(args.capture), staged_sslm, inputs, report_c)

    callback_count, clipped, overshoots = _capture_metadata(report_c)
    if clipped / callback_count > args.pass_c_clipped_per_callback:
        raise ChannelScaleDidNotConverge(
            f"ChannelScaleDidNotConverge: pass C clipped {clipped}/{callback_count} "
            f"({clipped * 1_000_000 / callback_count:.6g} per million); channels={overshoots}")

    # A candidate artifact remains inside the new work directory until its
    # convergence gate succeeds.  A failure therefore cannot leave args.out
    # looking like a completed conversion to a later step.
    staged_sslm.replace(final_sslm)

    float_peak = np.stack([np.asarray(model.qk_channel_peaks[f"layer{layer}"], dtype=np.float64)
                           for layer in range(model.config.num_hidden_layers)])
    a_peak, b_peak, c_peak = (_peak_table(model, report) for report in (report_a, report_b, report_c))
    final_peak = np.maximum.reduce([float_peak, a_peak, b_peak])
    tables_path = work / "peak-tables.npz"
    np.savez(tables_path, float_peak=float_peak, pass_a_peak=a_peak, pass_b_peak=b_peak,
             pass_c_peak=c_peak, final_peak=final_peak)
    report = {
        "artifact_sha256": hashlib.sha256(final_sslm.read_bytes()).hexdigest(),
        "capture_report_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for name, path in (("pass_a", report_a), ("pass_b", report_b), ("pass_c", report_c))},
        "pass_c_callback_count": callback_count,
        "pass_c_landing_saturation_count": clipped,
        "pass_c_clipped_per_million": clipped * 1_000_000 / callback_count,
        "pass_c_overshooting_channels": overshoots,
        "peak_tables": str(tables_path),
        "peak_tables_sha256": hashlib.sha256(tables_path.read_bytes()).hexdigest(),
        "final_peak_sha256": hashlib.sha256(final_peak.tobytes()).hexdigest(),
    }
    (work / "flow-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


def flow(args):
    work = Path(args.work)
    out = Path(args.out)
    if work.exists() or out.exists():
        raise RuntimeError("flow --work and --out must not already exist")
    if not Path(args.checkpoint).is_dir():
        raise RuntimeError(f"checkpoint is not a directory: {args.checkpoint}")
    work.mkdir(parents=True)
    source_cache = work / "float-cache"
    model = pipeline.load_model(args.checkpoint)
    artifact_cache.save_artifact(model, source_cache, args.checkpoint)
    prepare(argparse.Namespace(checkpoint=args.checkpoint, out=str(work / "inputs")))
    _flow_from_cache(args, source_cache, work / "inputs", work)


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
    flow_parser = sub.add_parser("flow", help="checkpoint through float calibration, A/B/C capture, and final artifact")
    flow_parser.add_argument("--checkpoint", required=True)
    flow_parser.add_argument("--out", required=True, help="final verified .sslm output; must not exist")
    flow_parser.add_argument("--work", required=True, help="new work directory for caches, captures, and report")
    flow_parser.add_argument("--capture", required=True, help="compiled t2700_fused_k_capture executable")
    flow_parser.add_argument("--verifier", required=True, help="compiled sslm_verify executable")
    flow_parser.add_argument("--skip-verify", action="store_true",
                             help="fixture-only writer path; production flow verifies independently")
    flow_parser.add_argument("--pass-c-clipped-per-callback", type=float, default=1 / 1_000_000,
                             help="maximum accepted pass-C clipped/callback rate (default: 1e-6)")
    flow_parser.set_defaults(fn=flow)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
