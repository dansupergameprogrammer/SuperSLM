"""Independent T-2693 calibration-vector arithmetic oracle.

This file deliberately imports neither the converter nor reference_pipeline.  It consumes
only a JSON calibration vector and evaluates the numeric contract with Python integers and
the standard library's binary64 bit operations.  It is an oracle for slice-2 vectors, not a
second forward implementation.
"""

import argparse
import json
import math
import struct
from fractions import Fraction
from pathlib import Path


def floor_div(numerator, denominator):
    if denominator <= 0:
        raise ValueError("floor_div denominator must be positive")
    return numerator // denominator


def round_nearest_away(numerator, denominator):
    if denominator <= 0:
        raise ValueError("round denominator must be positive")
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    return sign * ((magnitude + denominator // 2) // denominator)


def round_half_even_shift(value, shift):
    if shift < 0:
        return value << -shift
    if shift == 0:
        return value
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        return quotient + 1
    return quotient


def canonical_scale_from_positive_fraction(value):
    """Return `(m,e)` with value == m * 2**e and 2**30 <= m < 2**31."""
    if value <= 0:
        raise ValueError("canonical scale must be positive")
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    # Establish 1 <= value / 2**exponent < 2 exactly, correcting the bit-length estimate.
    while value < Fraction(1 << exponent, 1) if exponent >= 0 else value < Fraction(1, 1 << -exponent):
        exponent -= 1
    while value >= (Fraction(1 << (exponent + 1), 1) if exponent + 1 >= 0 else Fraction(1, 1 << -(exponent + 1))):
        exponent += 1
    scale_exponent = exponent - 30
    scaled = value / (Fraction(1 << scale_exponent, 1) if scale_exponent >= 0 else Fraction(1, 1 << -scale_exponent))
    mantissa = round_half_even_fraction(scaled)
    if mantissa == (1 << 31):
        mantissa = 1 << 30
        scale_exponent += 1
    if not (1 << 30) <= mantissa < (1 << 31):
        raise ValueError("canonical mantissa outside [2^30,2^31)")
    return mantissa, scale_exponent


def round_half_even_fraction(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    twice = remainder * 2
    if twice > value.denominator or (twice == value.denominator and quotient & 1):
        return quotient + 1
    return quotient


def binary64_bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def binary64_fraction(bits):
    sign = -1 if bits >> 63 else 1
    exponent = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    if exponent == 0x7FF:
        raise ValueError("binary64 scale must be finite")
    if exponent == 0:
        significand, power = fraction, -1074
    else:
        significand, power = (1 << 52) | fraction, exponent - 1075
    if power >= 0:
        return Fraction(sign * (significand << power), 1)
    return Fraction(sign * significand, 1 << -power)


def evaluate(vector):
    gain = int(vector["gain_code"])
    if not -128 <= gain <= 127:
        raise ValueError("gain_code outside loader-admitted Int8 domain")
    source_m = int(vector["k_norm_m"])
    source_e = int(vector["k_norm_e"])
    if not (1 << 30) <= source_m < (1 << 31):
        raise ValueError("k_norm_m outside canonical domain")
    wide_m, wide_e = canonical_scale_from_positive_fraction(
        Fraction(127 * source_m, 1) * (Fraction(1 << source_e, 1) if source_e >= 0 else Fraction(1, 1 << -source_e)))
    values = [int(x) for x in vector["k_codes"]]
    if not values:
        raise ValueError("k_codes is empty")
    if any(not -128 <= x <= 127 for x in values):
        raise ValueError("k_codes outside loader-admitted Int8 domain")
    sumsq = sum(x * x for x in values)
    if sumsq == 0:
        raise ValueError("RMSNorm zero row")
    norm_numerator = int(vector.get("norm_numerator", 1 << 30))
    # The vector supplies the already-domain-checked positive denominator so this oracle
    # can pin signed flooring without importing the converter's reciprocal implementation.
    norm_denominator = int(vector.get("norm_denominator", math.isqrt(sumsq << 32)))
    if norm_denominator <= 0:
        raise ValueError("norm_denominator must be positive")
    wide = [floor_div(code * gain * norm_numerator, norm_denominator) for code in values]
    cos_q30 = int(vector["cos_q30"])
    sin_q30 = int(vector["sin_q30"])
    if len(wide) % 2:
        raise ValueError("k_codes must contain RoPE pairs")
    rotated = []
    for left, right in zip(wide[0::2], wide[1::2]):
        rotated.extend((
            round_nearest_away(left * cos_q30 - right * sin_q30, 1 << 30),
            round_nearest_away(left * sin_q30 + right * cos_q30, 1 << 30),
        ))
    landing_bits = int(vector["landing_scale_bits"])
    landing_scale = binary64_fraction(landing_bits)
    if landing_scale <= 0:
        raise ValueError("landing scale must be finite and positive")
    source_scale = Fraction(wide_m, 1) * (Fraction(1 << wide_e, 1) if wide_e >= 0 else Fraction(1, 1 << -wide_e))
    ratio_q31 = round_nearest_away(source_scale.numerator * landing_scale.denominator * (1 << 31),
                                   source_scale.denominator * landing_scale.numerator)
    if not 1 <= ratio_q31 <= (1 << 31):
        raise ValueError("channel ratio outside [1,2^31]")
    q_codes = [int(x) for x in vector["q_codes"]]
    if len(q_codes) != len(rotated):
        raise ValueError("q_codes and k_codes differ in length")
    score_accumulator = sum(q * k * ratio_q31 for q, k in zip(q_codes, rotated))
    return {
        "binary64_bits": f"{landing_bits:016x}",
        "k_wide_source_scale": {"m": wide_m, "e": wide_e},
        "rms_sumsq": sumsq,
        "wide": wide,
        "rotated": rotated,
        "ratio_q31": ratio_q31,
        "score_accumulator": score_accumulator,
        "score_q31": round_nearest_away(score_accumulator, 1 << 31),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("vector", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(json.loads(args.vector.read_text(encoding="utf-8")))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
