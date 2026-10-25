"""MR4 red suite: fsfd_refuse.sslm for cell 7.9 (the jump verb rests at entry on a guard refusal at every row j).

F-SFD (tools/gen_schema_fast_decode_fixture.py) with mr2_fixtures.py's patch: the CompositionConstants entry of the last
layer's attn_residual gets an extreme mantissa, so its guard refuses some rows and not others. The value is chosen with
a probe of longkey's 26-token start chain (record: red-suite-mr4.md): at -(7 << 27) prompts refuse first at chain rows
up to 14, so one prompt whose first refusal is at row 7 or later gives every pass row j in 0..7 (7.9's sliding window);
mr2's default -(1 << 30) trips no row past 5, and larger magnitudes trip none. The cell finds its prompt at run time
and pins nothing else.

Usage: python3 mr4_fixtures.py FIXTURE_DIR   (reads FIXTURE_DIR/fsfd.sslm, writes FIXTURE_DIR/fsfd_refuse.sslm)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mr2_fixtures  # noqa: E402

MANTISSA = -(7 << 27)


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    d = argv[1].rstrip("/")
    src = "%s/fsfd.sslm" % d
    last = mr2_fixtures.layers(src)
    h = mr2_fixtures.patch(src, "%s/fsfd_refuse.sslm" % d, "layer%d.attn_residual" % last, MANTISSA)
    print("fsfd_refuse.sslm: layer%d.attn_residual mantissa %d, sha256 %s" % (last, MANTISSA, h))


if __name__ == "__main__":
    main(sys.argv)
