"""Build a zero-target adapter bound to a supplied base artifact.

The adapter has no projection tensors.  It is still a real, loadable adapter handle,
which is sufficient for the ADAPTER_SWAP_RESET lifecycle cell: the cell grades the
sequence-state guard, not adapter arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import sslm_convert_adapter as adapter_writer  # noqa: E402
import sslm_format as fmt  # noqa: E402
import sslm_model_writer as model_writer  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_t2941_zero_adapter_fixture.py <base.sslm> <out.sslm>")
    base = Path(sys.argv[1])
    output = Path(sys.argv[2])
    cfg = fmt.read_section_bytes(base, fmt.SectionType.CONFIG)
    if cfg is None:
        raise SystemExit(f"base artifact has no CONFIG section: {base}")
    provenance = adapter_writer.write_adp1(
        rank=1,
        target_modules=[],
        base_artifact_hash=fmt.raw_integrity_hash(base),
        lora_alpha=1.0,
        use_rslora=False,
        source_adapter_name="t2941-zero-target",
    )
    sections = [
        fmt.Section(fmt.SectionType.CONFIG, cfg),
        fmt.Section(fmt.SectionType.SIGMOID_LUT, model_writer.write_sil1()),
        fmt.Section(fmt.SectionType.PROVENANCE, provenance),
        fmt.Section(
            fmt.SectionType.WEIGHTS,
            model_writer.write_tensor_manifest(model_writer.WGT1, np.int8, {}),
        ),
        fmt.Section(
            fmt.SectionType.DELTA_FOLD_SCALES,
            model_writer.write_tensor_manifest(model_writer.DFS1, np.int32, {}),
        ),
        fmt.Section(
            fmt.SectionType.U_FOLD_SCALES,
            model_writer.write_tensor_manifest(model_writer.UFS1, np.int32, {}),
        ),
    ]
    data, fingerprint = fmt.build_artifact(sections)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f"wrote {output} bytes={len(data)} fingerprint={fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
