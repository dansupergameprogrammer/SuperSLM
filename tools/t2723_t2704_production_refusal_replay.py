#!/usr/bin/env python3
"""Replay T-2704's one-grid residual construction on pinned production traces.

The default invocation measures both shipped models. It reconstructs the captured
current-engine selected-grid residual rows and checks every recorded D',
normalization, reciprocal, and C26 output scale before reporting refusal counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
FUNNEL_D_PRIME_MAX = 1 << 31

SSLM_HEADER_BYTES = 64
SSLM_SECTION_DESC_BYTES = 40
SSLM_INTEGRITY_OFFSET = 32
SSLM_INTEGRITY_BYTES = 32
COMPOSITION_CONSTANTS_SECTION = 7

REQUIRED_RECORDS = {
    "stream",
    "o_proj.requant",
    "attn_residual",
    "down_proj.requant",
    "mlp_residual",
}

REFUSAL_REASONS = (
    "landing_overflow",
    "int64_min_inversion",
    "checked_sum_overflow",
    "d_prime_over_2_31",
    "c26_scale_fold_rejection",
    "zero_operand_scale",
    "operand_mantissa_out_of_int32",
)

TRACE_ITEM_COUNTS = {"qwen3": 28, "qwen2p5": 6}


@dataclass(frozen=True)
class Scale:
    m: int
    e: int


@dataclass(frozen=True)
class ModelInput:
    name: str
    trace_dir: Path
    artifact: Path
    expected_whole_file_sha256: str
    expected_trace_population_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(path: Path) -> tuple[str, str, bytes]:
    whole = hashlib.sha256()
    integrity = hashlib.sha256()
    offset = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            whole.update(chunk)
            zeroed = bytearray(chunk)
            begin = max(SSLM_INTEGRITY_OFFSET - offset, 0)
            end = min(SSLM_INTEGRITY_OFFSET + SSLM_INTEGRITY_BYTES - offset, len(zeroed))
            if begin < end:
                zeroed[begin:end] = b"\x00" * (end - begin)
            integrity.update(zeroed)
            offset += len(chunk)
    with path.open("rb") as handle:
        header = handle.read(SSLM_HEADER_BYTES)
    if len(header) != SSLM_HEADER_BYTES:
        raise ValueError(f"{path}: truncated SSLM header")
    return whole.hexdigest(), integrity.hexdigest(), header


def read_composition_constants(path: Path) -> tuple[dict[str, Scale], dict[str, Any]]:
    whole_sha, integrity_sha, header = artifact_hashes(path)
    if header[:4] != b"SSLM":
        raise ValueError(f"{path}: bad SSLM magic")
    version, header_bytes, section_count, flags, reserved = struct.unpack_from("<IIIII", header, 4)
    file_bytes = struct.unpack_from("<Q", header, 24)[0]
    stored_integrity = header[SSLM_INTEGRITY_OFFSET:SSLM_INTEGRITY_OFFSET + SSLM_INTEGRITY_BYTES].hex()
    if header_bytes != SSLM_HEADER_BYTES or reserved != 0:
        raise ValueError(f"{path}: unsupported SSLM header geometry")
    if file_bytes != path.stat().st_size:
        raise ValueError(f"{path}: file-size field {file_bytes} != actual {path.stat().st_size}")
    if integrity_sha != stored_integrity:
        raise ValueError(
            f"{path}: embedded integrity {stored_integrity} != recomputed {integrity_sha}"
        )

    section = None
    with path.open("rb") as handle:
        handle.seek(header_bytes)
        for _ in range(section_count):
            row = handle.read(SSLM_SECTION_DESC_BYTES)
            if len(row) != SSLM_SECTION_DESC_BYTES:
                raise ValueError(f"{path}: truncated section table")
            section_type, dtype, offset, byte_size, elem_count, alignment, sec_reserved = struct.unpack(
                "<IIQQQII", row
            )
            if section_type == COMPOSITION_CONSTANTS_SECTION:
                section = (dtype, offset, byte_size, elem_count, alignment, sec_reserved)
    if section is None:
        raise ValueError(f"{path}: no CompositionConstants section")
    dtype, offset, byte_size, elem_count, alignment, sec_reserved = section
    if dtype != 0 or elem_count != byte_size or sec_reserved != 0:
        raise ValueError(f"{path}: malformed CompositionConstants descriptor")
    with path.open("rb") as handle:
        handle.seek(offset)
        blob = handle.read(byte_size)
    if len(blob) != byte_size:
        raise ValueError(f"{path}: truncated CompositionConstants section")
    section_sha = hashlib.sha256(blob).hexdigest()

    if len(blob) < 24:
        raise ValueError(f"{path}: truncated KVC1 header")
    magic, kvc_version, entry_count, value_words, name_bytes, kvc_reserved = struct.unpack_from(
        "<4sIIIII", blob, 0
    )
    if magic != b"KVC1" or kvc_version != 1 or value_words != 2 or kvc_reserved != 0:
        raise ValueError(f"{path}: unsupported CompositionConstants KVC1 header")
    descriptors_offset = 24
    values_offset = descriptors_offset + 8 * entry_count
    names_offset = values_offset + 8 * value_words * entry_count
    if names_offset + name_bytes != len(blob):
        raise ValueError(f"{path}: KVC1 geometry does not cover the section exactly")

    entries: dict[str, Scale] = {}
    for index in range(entry_count):
        name_offset, name_size = struct.unpack_from("<II", blob, descriptors_offset + 8 * index)
        if name_offset + name_size > name_bytes:
            raise ValueError(f"{path}: KVC1 name {index} exceeds the name blob")
        name = blob[names_offset + name_offset:names_offset + name_offset + name_size].decode("utf-8")
        m, e = struct.unpack_from("<qq", blob, values_offset + 16 * index)
        if name in entries:
            raise ValueError(f"{path}: duplicate KVC1 key {name!r}")
        entries[name] = Scale(m, e)

    return entries, {
        "path": str(path.resolve()),
        "format_version": version,
        "flags": flags,
        "file_bytes": file_bytes,
        "whole_file_sha256": whole_sha,
        "embedded_integrity_sha256": stored_integrity,
        "recomputed_zeroed_hash_field_sha256": integrity_sha,
        "composition_constants_section_sha256": section_sha,
        "composition_constants_entry_count": entry_count,
        "composition_constants_alignment": alignment,
    }


def normalize_scale(d_prime: int) -> Scale:
    if d_prime < 1 or d_prime > I64_MAX:
        raise ValueError(f"NormalizeScale input outside [1, INT64_MAX]: {d_prime}")
    p = d_prime.bit_length() - 1
    s = 30 - p
    dn = d_prime << s if s >= 0 else d_prime >> 1
    return Scale(dn, s)


def dynamic_scale_reciprocal(dn: int) -> int:
    """The exact positive result C19's Newton/correction implementation returns."""
    if not (1 << 30) <= dn < (1 << 31):
        raise ValueError(f"canonical reciprocal input outside [2^30, 2^31): {dn}")
    return ((1 << 63) + dn) // (2 * dn)


def landing_rescale(code: int, source: Scale, target: Scale, target_shift: int) -> tuple[int, bool]:
    target_norm = normalize_scale(abs(target.m))
    reciprocal = dynamic_scale_reciprocal(target_norm.m)
    if target_norm.e != target_shift:
        raise AssertionError("target shift does not match normalized target magnitude")
    magnitude = abs(code) * abs(source.m) * reciprocal
    k = 62 - (source.e - target.e + target_shift)
    if k >= 0:
        rounded_magnitude = (2 * magnitude + (1 << k)) >> (k + 1)
    else:
        rounded_magnitude = magnitude << (-k)
    negative = (code < 0) != (source.m < 0)
    value = -rounded_magnitude if negative else rounded_magnitude
    return value, rounded_magnitude > I64_MAX


def saturating_add_i64(a: int, b: int) -> int:
    value = a + b
    return min(max(value, I64_MIN), I64_MAX)


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    value = (a * b + (1 << 30)) >> 31
    return min(value, I32_MAX)


def combine_carried_scale(a: Scale, b: Scale) -> Scale:
    if not (I32_MIN <= a.m <= I32_MAX and I32_MIN <= b.m <= I32_MAX):
        raise ValueError("CombineCarriedScale precondition violated")
    e = saturating_add_i64(saturating_add_i64(a.e, b.e), 31)
    m = saturating_rounding_doubling_high_mul(a.m, b.m)
    if m < (1 << 30):
        m <<= 1
        e -= 1
    return Scale(m, e)


def c26_left_fold(incoming: Scale, site_constant: Scale, d_prime: int) -> tuple[Scale | None, bool]:
    ns = normalize_scale(d_prime)
    factors = (incoming, site_constant, Scale(ns.m, -ns.e))
    if any(not I32_MIN <= factor.m <= I32_MAX for factor in factors):
        return None, False
    running = factors[0]
    for factor in factors[1:]:
        running = combine_carried_scale(running, factor)
        if not I32_MIN <= running.m <= I32_MAX:
            return None, False
    return running, True


def select_branch_grid(branch: Scale, stream: Scale) -> bool:
    branch_magnitude = abs(branch.m)
    stream_magnitude = abs(stream.m)
    if branch.e > stream.e and branch.e - stream.e > 31:
        return False
    if stream.e > branch.e and stream.e - branch.e > 31:
        return True
    difference = branch.e - stream.e
    if difference >= 0:
        return (branch_magnitude << difference) < stream_magnitude
    return branch_magnitude < (stream_magnitude << -difference)


def scale_from_record(record: dict[str, Any]) -> Scale:
    return Scale(int(record["m"]), int(record["e"]))


def record_site_suffix(record: dict[str, Any]) -> tuple[int, str]:
    site = str(record["site"])
    prefix, suffix = site.split(".", 1)
    if not prefix.startswith("layer"):
        raise ValueError(f"malformed chain site {site!r}")
    return int(prefix[5:]), suffix


def trace_files(model: str, trace_dir: Path, expected_population_sha256: str) -> tuple[list[tuple[Path, str]], str]:
    expected_names = [f"item-{index:02d}.residual.jsonl" for index in range(TRACE_ITEM_COUNTS[model])]
    actual_paths = sorted(trace_dir.glob("*.residual.jsonl"), key=lambda path: path.name)
    actual_names = [path.name for path in actual_paths]
    if actual_names != expected_names:
        raise ValueError(
            f"{model}: trace file set mismatch; got {actual_names}, expected {expected_names}"
        )
    result = []
    for path in actual_paths:
        digest = file_sha256(path)
        result.append((path, digest))
    population = hashlib.sha256()
    for path, digest in result:
        population.update(path.name.encode("utf-8"))
        population.update(b"\x00")
        population.update(bytes.fromhex(digest))
    actual_population_sha256 = population.hexdigest()
    if actual_population_sha256 != expected_population_sha256:
        raise ValueError(
            f"{model}: trace population SHA-256 {actual_population_sha256} != "
            f"pinned {expected_population_sha256}"
        )
    return result, actual_population_sha256


def read_trace_groups(path: Path) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "stream":
                layer = int(record["layer"])
                suffix = "stream"
            elif record_type == "chain":
                layer, suffix = record_site_suffix(record)
            else:
                raise ValueError(f"{path}:{line_number}: unknown record type {record_type!r}")
            key = (int(record["token"]), layer)
            group = groups.setdefault(key, {})
            if suffix in group:
                raise ValueError(f"{path}:{line_number}: duplicate {suffix} record for {key}")
            group[suffix] = record
    for key, group in groups.items():
        if set(group) != REQUIRED_RECORDS:
            raise ValueError(
                f"{path}: record set for token/layer {key} is {sorted(group)}, "
                f"expected {sorted(REQUIRED_RECORDS)}"
            )
    return groups


def validate_captured_residual(
    *,
    stream: dict[str, Any],
    branch: dict[str, Any],
    residual: dict[str, Any],
    site_constant: Scale,
    location: dict[str, Any],
) -> int:
    stream_scale = scale_from_record(stream)
    branch_scale = scale_from_record(branch)
    stream_codes = stream["codes"]
    branch_codes = branch["codes"]
    captured_wide = residual["x_int"]
    if not len(stream_codes) == len(branch_codes) == len(captured_wide):
        raise ValueError(f"{location}: residual operand widths differ")

    def reconstruct(select_branch: bool) -> tuple[list[int], Scale] | None:
        selected_scale = branch_scale if select_branch else stream_scale
        other_scale = stream_scale if select_branch else branch_scale
        direct_codes = branch_codes if select_branch else stream_codes
        other_codes = stream_codes if select_branch else branch_codes
        target_shift = normalize_scale(abs(selected_scale.m)).e
        wide: list[int] = []
        for index, (direct, other) in enumerate(zip(direct_codes, other_codes)):
            landed, exceeded = landing_rescale(int(other), other_scale, selected_scale, target_shift)
            if exceeded or (selected_scale.m < 0 and landed == I64_MIN):
                return None
            if selected_scale.m < 0:
                landed = -landed
            value = int(direct) + landed
            if not I64_MIN <= value <= I64_MAX:
                return None
            wide.append(value)
        d_prime = max(1, max(abs(value) for value in wide))
        running, in_domain = c26_left_fold(selected_scale, site_constant, d_prime)
        if not in_domain or running is None:
            return None
        return wide, selected_scale

    branch_selected = select_branch_grid(branch_scale, stream_scale)
    reconstructed = reconstruct(branch_selected)
    if reconstructed is None:
        reconstructed = reconstruct(not branch_selected)
    if reconstructed is None:
        raise ValueError(f"{location}: captured residual has no viable selected-grid preflight")
    wide, selected_scale = reconstructed
    if wide != [int(value) for value in captured_wide]:
        for index, (actual, expected) in enumerate(zip(wide, captured_wide)):
            if actual != int(expected):
                raise ValueError(
                    f"{location}: selected-grid reconstruction differs at element {index}: "
                    f"computed {actual}, captured {expected}"
                )
        raise AssertionError("selected-grid reconstruction width changed after validation")

    d_prime = max(1, max(abs(value) for value in wide))
    ns = normalize_scale(d_prime)
    reciprocal = dynamic_scale_reciprocal(ns.m)
    captured_primitives = (
        int(residual["d_prime"]),
        int(residual["dn"]),
        int(residual["s"]),
        int(residual["r"]),
    )
    computed_primitives = (d_prime, ns.m, ns.e, reciprocal)
    if computed_primitives != captured_primitives:
        raise ValueError(
            f"{location}: funnel primitive tuple {computed_primitives} != captured "
            f"{captured_primitives}"
        )
    running, in_domain = c26_left_fold(selected_scale, site_constant, d_prime)
    if not in_domain or running is None:
        raise ValueError(f"{location}: captured selected-grid row unexpectedly fails C26")
    captured_scale = scale_from_record(residual)
    if running != captured_scale:
        raise ValueError(
            f"{location}: C26 scale {running} != captured {captured_scale}"
        )
    return len(captured_wide)


def refusal_witness(
    *,
    reason: str,
    location: dict[str, Any],
    branch: dict[str, Any],
    stream: dict[str, Any],
    site_constant: Scale,
    selected: str | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reason": reason,
        "location": location,
        "selected_grid": selected,
        "branch": {
            "scale": {"m": int(branch["m"]), "e": int(branch["e"])},
            "codes": [int(value) for value in branch["codes"]],
        },
        "stream": {
            "scale": {"m": int(stream["m"]), "e": int(stream["e"])},
            "codes": [int(value) for value in stream["codes"]],
        },
        "site_constant": {"m": site_constant.m, "e": site_constant.e},
        "details": details,
    }


def replay_residual(
    *,
    branch: dict[str, Any],
    stream: dict[str, Any],
    site_constant: Scale,
    location: dict[str, Any],
) -> dict[str, Any]:
    branch_scale = scale_from_record(branch)
    stream_scale = scale_from_record(stream)
    if branch_scale.m == 0 or stream_scale.m == 0:
        return {
            "status": "zero_operand_scale",
            "witness": refusal_witness(
                reason="zero_operand_scale", location=location, branch=branch, stream=stream,
                site_constant=site_constant, selected=None, details={},
            ),
        }
    if not (
        I32_MIN <= branch_scale.m <= I32_MAX
        and I32_MIN <= stream_scale.m <= I32_MAX
    ):
        return {
            "status": "operand_mantissa_out_of_int32",
            "witness": refusal_witness(
                reason="operand_mantissa_out_of_int32", location=location, branch=branch,
                stream=stream, site_constant=site_constant, selected=None, details={},
            ),
        }

    branch_selected = select_branch_grid(branch_scale, stream_scale)
    selected_name = "branch" if branch_selected else "stream"
    selected_record = branch if branch_selected else stream
    nonselected_record = stream if branch_selected else branch
    selected_scale = branch_scale if branch_selected else stream_scale
    nonselected_scale = stream_scale if branch_selected else branch_scale
    selected_codes = selected_record["codes"]
    nonselected_codes = nonselected_record["codes"]
    if len(selected_codes) != len(nonselected_codes):
        raise ValueError(f"{location}: residual operand widths differ")
    target_ns = normalize_scale(abs(selected_scale.m))
    wide: list[int] = []
    for index, (direct_code, nonselected_code) in enumerate(
        zip(selected_codes, nonselected_codes)
    ):
        landed, exceeded = landing_rescale(
            int(nonselected_code), nonselected_scale, selected_scale, target_ns.e
        )
        if exceeded:
            details = {
                "element_index": index,
                "direct_selected_code": int(direct_code),
                "nonselected_landing": landed,
                "constructed_prefix": wide,
            }
            return {
                "status": "landing_overflow",
                "witness": refusal_witness(
                    reason="landing_overflow", location=location, branch=branch, stream=stream,
                    site_constant=site_constant, selected=selected_name, details=details,
                ),
            }
        if selected_scale.m < 0 and landed == I64_MIN:
            details = {
                "element_index": index,
                "direct_selected_code": int(direct_code),
                "nonselected_landing": landed,
                "constructed_prefix": wide,
            }
            return {
                "status": "int64_min_inversion",
                "witness": refusal_witness(
                    reason="int64_min_inversion", location=location, branch=branch, stream=stream,
                    site_constant=site_constant, selected=selected_name, details=details,
                ),
            }
        if selected_scale.m < 0:
            landed = -landed
        wide_sum = int(direct_code) + landed
        if not I64_MIN <= wide_sum <= I64_MAX:
            details = {
                "element_index": index,
                "direct_selected_code": int(direct_code),
                "oriented_nonselected_landing": landed,
                "infinite_precision_sum": wide_sum,
                "constructed_prefix": wide,
            }
            return {
                "status": "checked_sum_overflow",
                "witness": refusal_witness(
                    reason="checked_sum_overflow", location=location, branch=branch, stream=stream,
                    site_constant=site_constant, selected=selected_name, details=details,
                ),
            }
        wide.append(wide_sum)

    d_prime = max(1, max(abs(value) for value in wide))
    if d_prime > FUNNEL_D_PRIME_MAX:
        return {
            "status": "d_prime_over_2_31",
            "witness": refusal_witness(
                reason="d_prime_over_2_31", location=location, branch=branch, stream=stream,
                site_constant=site_constant, selected=selected_name,
                details={"d_prime": d_prime, "wide_row": wide},
            ),
        }
    running, in_domain = c26_left_fold(selected_scale, site_constant, d_prime)
    if not in_domain or running is None:
        return {
            "status": "c26_scale_fold_rejection",
            "witness": refusal_witness(
                reason="c26_scale_fold_rejection", location=location, branch=branch, stream=stream,
                site_constant=site_constant, selected=selected_name,
                details={"d_prime": d_prime, "wide_row": wide},
            ),
        }
    element_index = max(range(len(wide)), key=lambda index: abs(wide[index]))
    return {
        "status": "accepted",
        "selected_grid": selected_name,
        "d_prime": d_prime,
        "peak_element_index": element_index,
        "peak_element_value": wide[element_index],
        "selected_scale": {"m": selected_scale.m, "e": selected_scale.e},
        "c26_output_scale": {"m": running.m, "e": running.e},
    }


def measure_model(model_input: ModelInput) -> dict[str, Any]:
    constants, artifact = read_composition_constants(model_input.artifact)
    if artifact["whole_file_sha256"] != model_input.expected_whole_file_sha256:
        raise ValueError(
            f"{model_input.name}: artifact SHA-256 {artifact['whole_file_sha256']} != "
            f"pinned {model_input.expected_whole_file_sha256}"
        )
    pinned_files, trace_population_sha256 = trace_files(
        model_input.name, model_input.trace_dir, model_input.expected_trace_population_sha256
    )
    counts = {reason: 0 for reason in REFUSAL_REASONS}
    witnesses: list[dict[str, Any]] = []
    total_calls = 0
    accepted = 0
    selected_counts = {"branch": 0, "stream": 0}
    validated_elements = 0
    peak: dict[str, Any] | None = None
    used_constant_keys: set[str] = set()

    for path, _digest in pinned_files:
        groups = read_trace_groups(path)
        for (token, layer), records in sorted(groups.items()):
            calls = (
                ("attn_residual", records["o_proj.requant"], records["stream"], records["attn_residual"]),
                ("mlp_residual", records["down_proj.requant"], records["attn_residual"], records["mlp_residual"]),
            )
            for site_name, branch, stream, residual in calls:
                constant_key = f"layer{layer}.{site_name}"
                if constant_key not in constants:
                    raise ValueError(
                        f"{model_input.name}: artifact lacks required constant {constant_key!r}"
                    )
                site_constant = constants[constant_key]
                used_constant_keys.add(constant_key)
                location = {
                    "trace_file": path.name,
                    "token": token,
                    "layer": layer,
                    "site": site_name,
                }
                validated_elements += validate_captured_residual(
                    stream=stream,
                    branch=branch,
                    residual=residual,
                    site_constant=site_constant,
                    location=location,
                )
                result = replay_residual(
                    branch=branch,
                    stream=stream,
                    site_constant=site_constant,
                    location=location,
                )
                total_calls += 1
                if result["status"] == "accepted":
                    accepted += 1
                    selected_counts[result["selected_grid"]] += 1
                    if peak is None or result["d_prime"] > peak["magnitude"]:
                        peak = {
                            "magnitude": result["d_prime"],
                            "location": {
                                **location,
                                "element_index": result["peak_element_index"],
                            },
                            "signed_element_value": result["peak_element_value"],
                            "selected_grid": result["selected_grid"],
                            "selected_scale": result["selected_scale"],
                            "site_constant": {"m": site_constant.m, "e": site_constant.e},
                            "c26_output_scale": result["c26_output_scale"],
                        }
                else:
                    counts[result["status"]] += 1
                    witnesses.append(result["witness"])

    residual_constants = {
        key: {"m": constants[key].m, "e": constants[key].e}
        for key in sorted(used_constant_keys)
    }
    expected_residual_key_count = 56 if model_input.name == "qwen3" else 48
    if len(residual_constants) != expected_residual_key_count:
        raise ValueError(
            f"{model_input.name}: used {len(residual_constants)} residual constants, "
            f"expected {expected_residual_key_count}"
        )
    refusal_total = sum(counts.values())
    if total_calls != accepted + refusal_total:
        raise AssertionError("call accounting does not close")
    if peak is None and accepted:
        raise AssertionError("accepted population has no peak")

    return {
        "model": model_input.name,
        "status": "MEASURED",
        "artifact": artifact,
        "trace_population": {
            "path": str(model_input.trace_dir.resolve()),
            "file_count": len(pinned_files),
            "files": {path.name: digest for path, digest in pinned_files},
            "manifest_sha256": trace_population_sha256,
        },
        "residual_site_constant_count": len(residual_constants),
        "residual_site_constants": residual_constants,
        "total_residual_calls": total_calls,
        "accepted_calls": accepted,
        "refusal_total": refusal_total,
        "refusal_rate": {
            "numerator": refusal_total,
            "denominator": total_calls,
            "decimal": refusal_total / total_calls if total_calls else None,
        },
        "refusals": counts,
        "refusal_witnesses": witnesses,
        "selected_grid_calls": selected_counts,
        "peak_accepted_row": peak,
        "apparatus_validation": {
            "captured_residual_calls_reconstructed_exactly": total_calls,
            "captured_residual_elements_reconstructed_exactly": validated_elements,
            "checks_per_call": [
                "captured selected-grid wide row",
                "D'",
                "normalized denominator and shift",
                "C19 reciprocal",
                "C26 left-associated output scale",
            ],
            "status": "PASS",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("both", "qwen3", "qwen2p5"), default="both")
    parser.add_argument(
        "--qwen3-traces",
        type=Path,
        default=Path(r"D:\_artifacts\superslm\_t2703fid\resadd\qwen3\engine"),
    )
    parser.add_argument("--qwen3-artifact-sha256", required=True)
    parser.add_argument("--qwen3-trace-population-sha256", required=True)
    parser.add_argument(
        "--qwen3-artifact",
        type=Path,
        default=Path(r"D:\_artifacts\superslm\_t2703conv\flow-final\qwen3-embedding-0.6b-1p5.sslm"),
    )
    parser.add_argument("--qwen2p5-artifact-sha256", required=True)
    parser.add_argument("--qwen2p5-trace-population-sha256", required=True)
    parser.add_argument(
        "--qwen2p5-traces",
        type=Path,
        default=Path(r"D:\_artifacts\superslm\_t2703fid\resadd\qwen2p5\engine"),
    )
    parser.add_argument(
        "--qwen2p5-artifact",
        type=Path,
        default=Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-0.5b-instruct.sslm"),
    )
    parser.add_argument("--output", type=Path, help="write the JSON result here as well as stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "qwen3": ModelInput(
            "qwen3",
            args.qwen3_traces,
            args.qwen3_artifact,
            args.qwen3_artifact_sha256,
            args.qwen3_trace_population_sha256,
        ),
        "qwen2p5": ModelInput(
            "qwen2p5", args.qwen2p5_traces, args.qwen2p5_artifact,
            args.qwen2p5_artifact_sha256, args.qwen2p5_trace_population_sha256,
        ),
    }
    selected = ("qwen3", "qwen2p5") if args.model == "both" else (args.model,)
    result = {
        "measurement": "T-2723 T-2704 production refusal replay",
        "construction": "T-2704 design section 2 one selected finer grid and real C26 left fold",
        "models": {name: measure_model(inputs[name]) for name in selected},
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
