"""calibrate_checkpoint.py -- a general-purpose "calibrate this checkpoint" CLI (T-2123/T-2137,
design §1/§6 B1).

`convert_model.py`'s CLI takes `--artifact <calibrated artifact directory>` -- it never takes a
raw checkpoint. Before this file, exactly one code path anywhere in this repository or the
source spike turns a raw Hugging Face checkpoint into that artifact directory
(`pipeline.load_model(checkpoint)` followed by `artifact_cache.save_artifact(model,
artifact_dir, checkpoint)`), and the only place those two calls were wired together was
`superslm_spike/smoke_eval_run.py` -- a demo/eval driver with hardcoded paths, bundled with an
unrelated smoke-decode, DFA-schema compile, and per-record ledger pipeline. This CLI is the
missing general-purpose command: `--checkpoint <dir> --out <dir>`, calibration and nothing
else -- no DFA compile, no smoke decode, no ledger.

No hardcoded paths: `--checkpoint` and `--out` are required arguments, never defaults.
"""

import argparse
import sys
from pathlib import Path

from reference_pipeline import artifact_cache, pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_checkpoint.py",
        description=(
            "Calibrate a raw Hugging Face checkpoint into a .sslm-ready artifact directory. "
            "This is calibration only -- convert_model.py --artifact <out> --out <model.sslm> "
            "is the next step."
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a local Hugging Face checkpoint directory (config.json + safetensors + "
             "tokenizer files).",
    )
    parser.add_argument(
        "--out", required=True,
        help="Path to write the calibrated artifact directory to (created if absent).",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        print(
            f"calibrate_checkpoint.py: --checkpoint {args.checkpoint!r} is not a directory",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out)
    if not out_dir.parent.is_dir():
        print(
            f"calibrate_checkpoint.py: --out {args.out!r}'s parent directory does not exist",
            file=sys.stderr,
        )
        return 2

    model = pipeline.load_model(str(checkpoint_dir))
    artifact_cache.save_artifact(model, str(out_dir), str(checkpoint_dir))
    fingerprint = artifact_cache._compute_fingerprint(model)
    print(f"written fingerprint: {fingerprint}")
    print(f"artifact written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
