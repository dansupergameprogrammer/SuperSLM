"""Stage 2 -- Q2: does bf16 work, natively or at all?

Two checks, reported separately: (1) does torch report native bf16 hardware
support for this device, and (2) does an actual bf16 matmul on the device run
and produce finite, non-degenerate output (a device can report support and
still emulate it, or report no support and still run it in software --
this stage measures both rather than trusting either alone).
"""
import sys
import traceback

from common import emit, fail, ok

STAGE = "bf16"


def main() -> int:
    try:
        import torch
    except Exception as e:
        emit(fail(STAGE, "torch did not import (see stage_import's own failure)",
                  {"exception": repr(e)}))
        return 1

    if not torch.cuda.is_available():
        emit(fail(STAGE, "no device available -- see stage_import"))
        return 1

    detail = {}
    try:
        native_support = torch.cuda.is_bf16_supported()
        detail["torch_reports_native_bf16"] = native_support
        print(f"[{STAGE}] torch.cuda.is_bf16_supported() = {native_support}")
    except Exception as e:
        detail["native_support_query_exception"] = repr(e)
        print(f"[{STAGE}] torch.cuda.is_bf16_supported() raised: {e!r}")

    try:
        dev = torch.device("cuda:0")
        torch.manual_seed(0)
        a = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
        b = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
        c = a @ b
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(c).all().item())
        nonzero = bool((c.abs().sum() > 0).item())
        sample = float(c[0, 0].item())
        detail.update({
            "matmul_ran": True,
            "result_dtype": str(c.dtype),
            "result_finite": finite,
            "result_nonzero": nonzero,
            "sample_value": sample,
        })
        print(f"[{STAGE}] 512x512 bf16 matmul on device: finite={finite} "
              f"nonzero={nonzero} sample={sample}")
        passed = finite and nonzero
        if passed:
            emit(ok(STAGE, detail))
            return 0
        else:
            emit(fail(STAGE, "bf16 matmul ran without raising but produced "
                              "non-finite or all-zero output", detail))
            return 1
    except Exception as e:
        print(f"[{STAGE}] bf16 matmul raised: {e!r}")
        traceback.print_exc()
        detail["exception"] = repr(e)
        detail["traceback"] = traceback.format_exc()
        emit(fail(STAGE, "a bf16 tensor operation on the device raised an "
                          "exception -- bf16 does not work on this device/build, "
                          "even if is_bf16_supported() reported True above",
                  detail))
        return 1


if __name__ == "__main__":
    sys.exit(main())
