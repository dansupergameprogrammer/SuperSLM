"""Estimate a conservative final-channel headroom from a failed flow work directory.

This is deliberately an estimator, not a replacement for flow's pass-C gate.
Widening a channel table changes the later integer landing path, so a conversion
rerun must capture and gate its own pass C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


_COLUMNS = [
    "layer", "head", "channel", "raw_abs_peak", "wide_scale_m", "wide_scale_e",
    "real_peak", "landing_saturation_count",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_pass_c(path: Path):
    callbacks = None
    saturation = None
    observations = []
    with path.open("r", encoding="utf-8") as report:
        if report.readline().strip() != "T2700_FUSED_K_CAPTURE_V1":
            raise ValueError(f"unrecognized pass-C report: {path}")
        for line in report:
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            if fields[0] == "summary":
                if len(fields) != 3:
                    raise ValueError(f"malformed summary row: {line!r}")
                if fields[1] == "callback_count":
                    callbacks = int(fields[2])
                elif fields[1] == "landing_saturation_count":
                    saturation = int(fields[2])
                continue
            if fields == _COLUMNS:
                continue
            if len(fields) != len(_COLUMNS):
                raise ValueError(f"malformed channel row: {line!r}")
            layer, head, channel = (int(fields[index]) for index in range(3))
            observed = float.fromhex(fields[6])
            clipped = int(fields[7])
            if layer < 0 or head < 0 or channel < 0 or not math.isfinite(observed) or observed < 0 or clipped < 0:
                raise ValueError(f"invalid channel row: {line!r}")
            observations.append((layer, head, channel, observed, clipped))
    if callbacks is None or callbacks <= 0 or saturation is None or saturation < 0:
        raise ValueError(f"pass-C report missing a valid callback/saturation summary: {path}")
    if sum(row[4] for row in observations) != saturation:
        raise ValueError("pass-C per-channel saturation total disagrees with its summary")
    return callbacks, saturation, observations


def _read_final_table(work: Path):
    metadata_path = work / "final-cache" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    peaks = metadata.get("qk_channel_peaks")
    if not isinstance(peaks, dict) or not peaks:
        raise ValueError(f"final cache has no qk_channel_peaks table: {metadata_path}")
    arrays = {}
    for key, value in peaks.items():
        if not key.startswith("layer") or not key[5:].isdigit():
            raise ValueError(f"unexpected final table key: {key!r}")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"invalid final table for {key}")
        arrays[int(key[5:])] = array
    ordered = [arrays[index] for index in range(len(arrays))]
    if len(ordered) != len(arrays):
        raise ValueError("final table layer numbers are not contiguous from zero")
    return metadata_path, arrays, np.stack(ordered)


def _curve(observations, table, callbacks: int, limit_per_million: float):
    ratios = []
    seen = set()
    for layer, head, channel, observed, clipped in observations:
        coordinate = (layer, head, channel)
        if coordinate in seen:
            raise ValueError(f"duplicate pass-C channel: {coordinate}")
        seen.add(coordinate)
        try:
            peak = float(table[layer][head, channel])
        except (KeyError, IndexError) as error:
            raise ValueError(f"pass-C channel absent from final table: {coordinate}") from error
        ratio = observed / peak
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError(f"invalid observed/final ratio at {coordinate}")
        if clipped:
            ratios.append((ratio, clipped))

    # At h, a channel retains all observed clips only while ratio > h.  Group
    # equal ratios so every point is a true strict-inequality breakpoint.
    grouped = {}
    for ratio, clipped in ratios:
        if ratio > 1.0:
            grouped[ratio] = grouped.get(ratio, 0) + clipped
    remaining = sum(grouped.values())
    points = [(1.0, remaining)]
    for ratio in sorted(grouped):
        remaining -= grouped[ratio]
        points.append((ratio, remaining))

    allowed_clips = math.floor(callbacks * limit_per_million / 1_000_000)
    candidate = next((headroom for headroom, clips in points if clips <= allowed_clips), None)
    if candidate is None:
        raise ValueError("no finite headroom breakpoint reaches the requested clip limit")
    return points, candidate, allowed_clips


def _sample_curve(points, samples: int):
    if samples >= len(points):
        return points
    indices = {round(index * (len(points) - 1) / (samples - 1)) for index in range(samples)}
    return [point for index, point in enumerate(points) if index in indices]


def estimate(work: Path, limit_per_million: float, curve_points: int, full_curve: bool):
    if not math.isfinite(limit_per_million) or limit_per_million < 0:
        raise ValueError("limit-per-million must be finite and non-negative")
    pass_c_path = work / "pass-c.tsv"
    callbacks, saturation, observations = _read_pass_c(pass_c_path)
    metadata_path, table, stacked_table = _read_final_table(work)
    points, candidate, allowed_clips = _curve(observations, table, callbacks, limit_per_million)
    shown = points if full_curve else _sample_curve(points, curve_points)
    summary_path = work / "final-summary.json"
    final_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    final_peak_sha256 = hashlib.sha256(stacked_table.tobytes()).hexdigest()
    recorded_peak_sha256 = final_summary.get("final_peak_sha256")
    if recorded_peak_sha256 is not None and recorded_peak_sha256 != final_peak_sha256:
        raise ValueError("final cache table digest disagrees with final-summary.json")

    def point(headroom, clips):
        return {
            "headroom": headroom,
            "estimated_clips": clips,
            "estimated_clips_per_million": clips * 1_000_000 / callbacks,
        }

    return {
        "work": str(work),
        "input_digests": {
            "pass_c_sha256": _sha256(pass_c_path),
            "final_cache_metadata_sha256": _sha256(metadata_path),
            "final_peak_sha256": final_peak_sha256,
            "final_summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
            "final_pending_artifact_sha256_recorded": final_summary.get("artifact_sha256"),
        },
        "pass_c_callback_count": callbacks,
        "pass_c_landing_saturation_count": saturation,
        "limit_per_million": limit_per_million,
        "allowed_estimated_clips": allowed_clips,
        "curve_breakpoint_count": len(points),
        "curve_is_complete": full_curve,
        "curve": [point(headroom, clips) for headroom, clips in shown],
        "smallest_headroom_at_or_under_limit": candidate,
        "candidate_estimated_clips_per_million": next(
            point(headroom, clips)["estimated_clips_per_million"]
            for headroom, clips in points if headroom == candidate),
        "k_landing_resolution_cost_bits": math.log2(candidate),
        "rerun_check": (
            "A widened conversion must capture and gate its own pass C: widening the table "
            "changes the downstream integer path, so this estimate is conservative guidance only."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="failed flow work directory (read-only)")
    parser.add_argument("--limit-per-million", type=float, default=1.0,
                        help="target estimated clips per million (default: 1.0)")
    parser.add_argument("--curve-points", type=int, default=15,
                        help="number of evenly spaced exact breakpoints to print (default: 15)")
    parser.add_argument("--full-curve", action="store_true",
                        help="print every exact breakpoint rather than a compact sampled curve")
    args = parser.parse_args()
    if args.curve_points < 2:
        parser.error("--curve-points must be at least 2")
    try:
        report = estimate(args.work, args.limit_per_million, args.curve_points, args.full_curve)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
