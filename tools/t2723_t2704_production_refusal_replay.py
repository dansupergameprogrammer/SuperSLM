#!/usr/bin/env python3
"""Replay T-2704's one-grid residual construction on the frozen production traces.

The default invocation measures both shipped models.  It also reconstructs the
captured, pre-T-2704 residual rows from their original operands and checks every
recorded D', normalization, reciprocal, and C26 output scale before reporting the
new-construction refusal counts.
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

FROZEN_TRACE_SHA256 = {
    "qwen3": {
        "item-00.residual.jsonl": "89d042146f56d30d044b2f2c31d23203085f038241c2db14dad71bf52f3151f5",
        "item-01.residual.jsonl": "73b04b3e9e182dbe275373e9a53b83be89a5b7ca46afafccd69538835a3f53d9",
        "item-02.residual.jsonl": "7a40d7677586272218ec0595938328b2e90f39a57cf369fc7f9b77875b4eba9d",
        "item-03.residual.jsonl": "1c373293034723e7b03a5407ef7338836dc1f3d685dc3cd38fac272daf9e8eeb",
        "item-04.residual.jsonl": "f7c224a38fbed7e39527f61c622199c6244c0601811c9faf7149fca39026e50f",
        "item-05.residual.jsonl": "9ad83f4bbc6cee7de5721d88a5b7d1335f06ab00c321db5df90efbbdb71e37c8",
        "item-06.residual.jsonl": "7b329c6d6d09b0b12ce9c8ce39ba96e96536ba1a1ab5e16efe61c0dea9c4f921",
        "item-07.residual.jsonl": "1a51e67214ef54436df27d5322e1599532c461077eab67befc26e4a944c05907",
        "item-08.residual.jsonl": "1515431d00fc156eb35fd14756e3027786f38879cf60ade5927a487da527d6eb",
        "item-09.residual.jsonl": "b04c18ab60b6cda5b56f6210b512ce64de78b794964d04f1ba8dcada109d517c",
        "item-10.residual.jsonl": "cd94ec8354a45e1c300c5c8731b8a2d55c59495d6342cfeb89df764b2c326995",
        "item-11.residual.jsonl": "bc4ee3a8518efd31e2d05caaf2a60ecd380339b17d98378923bae91ceb700907",
        "item-12.residual.jsonl": "d8d0404f641394ff2f87535aa23c211747f3d359066cf5fcf6fea18da8a694a3",
        "item-13.residual.jsonl": "dc8afd4f54b7f13125c32091a99907cae14561c180695802c9411505f649c3b6",
        "item-14.residual.jsonl": "610c56451db3edded97fecf3f9e755d74870d99711d616305807f884e6a31238",
        "item-15.residual.jsonl": "215da74473ff1afb39eb0ee4751e1c3650efd6ce9fbb26ef1727b70405bab68a",
        "item-16.residual.jsonl": "41f5ae3f07975684153571d20dfa342dc3dcff215cabbb59b2b1bfba6102aea0",
        "item-17.residual.jsonl": "3ab91c25374908dde4ca663a666451b66ab94f704742169b5753007db9c8f4e4",
        "item-18.residual.jsonl": "f9d26e712ad9b754ec650b84fb3b1b0b1b26a267b8b565bed75aa55d4e535bda",
        "item-19.residual.jsonl": "228dd59a59f6326bdf4da43ab8c228116e1f898075a9723339a30bcfd24cb3f6",
        "item-20.residual.jsonl": "e366ee49f2f333ea9e3c2f0c069d7beb2ede87981405431761306968b05dc639",
        "item-21.residual.jsonl": "b12e486f88e51bdbf772bc3d0d8d9090d5688b42613b3f9082f657b7f4786bca",
        "item-22.residual.jsonl": "97d24106104f09a0a28b2f62ec493f257f6668147efd5260218341a67c53dbcf",
        "item-23.residual.jsonl": "64b63c78b2216480630499eca57fa50dd1a70ee650f7250a77ae656b891e2456",
        "item-24.residual.jsonl": "d523188445c5fa40c50cae0d4b470f05d42d1911a5d759732dd715455738caae",
        "item-25.residual.jsonl": "b52ead2e430c8e6d19eb330c0a0fc444fa0764a506cfa0393a33603f117feca7",
        "item-26.residual.jsonl": "be5953161c8a9f6e9fad67b382cfb9454922e4124e528d63f243deee17849654",
        "item-27.residual.jsonl": "4377dd9f5b6c12a7988deec3fcc03e082c4ab4e8f84c4a6fccec461d079cb1da",
    },
    "qwen2p5": {
        "item-00.residual.jsonl": "2e2323aa88a0449c28b0a3b06c72d1a937073ab253af4dd813850cf89ba7ceea",
        "item-01.residual.jsonl": "836c1e6e709ba2e168a054ac1864b3a12735b0f27591f9acd3dc5a788555e0f5",
        "item-02.residual.jsonl": "d95a605222c1d3ee6615ecd9f798a6455c82ae4d1574e41173d15e0ca17e72e9",
        "item-03.residual.jsonl": "3ad496e547c447668b6590f0ba7571a0725d6e2ce9fc6cdce5637d9581cc3cb9",
        "item-04.residual.jsonl": "e5e5213ff1e490a489173f37e14b84d9c14ded7283742ed3c017b71a7fa007da",
        "item-05.residual.jsonl": "dd15d9667a5f835b7b754815e7b0d0837ff806842f4d52be45d92c4d6bdaa80f",
    },
}


@dataclass(frozen=True)
class Scale:
    m: int
    e: int


@dataclass(frozen=True)
class ModelInput:
    name: str
    trace_dir: Path
    artifact: Path
    expected_integrity_sha256: str | None = None


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


def trace_files(model: str, trace_dir: Path) -> list[tuple[Path, str]]:
    expected = FROZEN_TRACE_SHA256[model]
    actual_paths = sorted(trace_dir.glob("*.residual.jsonl"), key=lambda path: path.name)
    actual_names = [path.name for path in actual_paths]
    if actual_names != list(expected):
        raise ValueError(
            f"{model}: trace file set mismatch; got {actual_names}, expected {list(expected)}"
        )
    result = []
    for path in actual_paths:
        digest = file_sha256(path)
        if digest != expected[path.name]:
            raise ValueError(
                f"{model}: {path.name} SHA-256 {digest} != frozen {expected[path.name]}"
            )
        result.append((path, digest))
    return result


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
    if stream_scale.m <= 0:
        raise ValueError(f"{location}: captured stream mantissa is not positive")
    ns_target = normalize_scale(stream_scale.m)
    if ns_target.e != 0 or ns_target.m != stream_scale.m:
        raise ValueError(f"{location}: captured stream scale is not canonical")
    stream_codes = stream["codes"]
    branch_codes = branch["codes"]
    captured_wide = residual["x_int"]
    if not len(stream_codes) == len(branch_codes) == len(captured_wide):
        raise ValueError(f"{location}: residual operand widths differ")
    for index, (stream_code, branch_code, expected) in enumerate(
        zip(stream_codes, branch_codes, captured_wide)
    ):
        landed, exceeded = landing_rescale(branch_code, branch_scale, stream_scale, 0)
        if exceeded:
            raise ValueError(f"{location}: captured baseline landing overflows at element {index}")
        actual = int(stream_code) + landed
        if actual != int(expected):
            raise ValueError(
                f"{location}: baseline reconstruction differs at element {index}: "
                f"computed {actual}, captured {expected}"
            )
    d_prime = max(1, max(abs(int(value)) for value in captured_wide))
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
    running, in_domain = c26_left_fold(stream_scale, site_constant, d_prime)
    if not in_domain or running is None:
        raise ValueError(f"{location}: captured baseline unexpectedly fails C26")
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
    if (
        model_input.expected_integrity_sha256 is not None
        and artifact["embedded_integrity_sha256"] != model_input.expected_integrity_sha256
    ):
        raise ValueError(
            f"{model_input.name}: artifact integrity {artifact['embedded_integrity_sha256']} != "
            f"required {model_input.expected_integrity_sha256}"
        )
    pinned_files = trace_files(model_input.name, model_input.trace_dir)
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

    trace_population_digest = hashlib.sha256()
    for path, digest in pinned_files:
        trace_population_digest.update(path.name.encode("utf-8"))
        trace_population_digest.update(b"\x00")
        trace_population_digest.update(bytes.fromhex(digest))

    return {
        "model": model_input.name,
        "status": "MEASURED",
        "artifact": artifact,
        "trace_population": {
            "path": str(model_input.trace_dir.resolve()),
            "file_count": len(pinned_files),
            "files": {path.name: digest for path, digest in pinned_files},
            "manifest_sha256": trace_population_digest.hexdigest(),
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
                "captured pre-T-2704 wide row",
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
        default=Path(r"D:\_t2703fid\resadd\qwen3\engine"),
    )
    parser.add_argument(
        "--qwen3-artifact",
        type=Path,
        default=Path(r"D:\_t2703conv\flow-final\qwen3-embedding-0.6b-1p5.sslm"),
    )
    parser.add_argument(
        "--qwen2p5-traces",
        type=Path,
        default=Path(r"D:\_t2703fid\resadd\qwen2p5\engine"),
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
            "f0248cf757d94808ec146cf07379e9d5dfb2de3ef99e876b99f7a1cfca9b3497",
        ),
        "qwen2p5": ModelInput("qwen2p5", args.qwen2p5_traces, args.qwen2p5_artifact),
    }
    selected = ("qwen3", "qwen2p5") if args.model == "both" else (args.model,)
    result = {
        "measurement": "T-2723 T-2704 production refusal replay",
        "construction": "T-2704 design section 2 one selected finer grid and real C26 left fold",
        "models": {name: measure_model(inputs[name]) for name in selected},
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
