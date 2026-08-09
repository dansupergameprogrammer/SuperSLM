"""Stage 1 -- Q1: does PyTorch import, and does it see the device as gfx1100?

Run standalone: python stage_import.py
Prints progress to stdout, then a final RESULT_JSON: line the orchestrator parses.
Exit code 0 = pass, 1 = fail. Never raises past this file -- every exception is
caught and reported in the result, because an uncaught traceback here is exactly
the kind of thing Dan should not have to read and interpret.
"""
import platform
import sys
import traceback

from common import emit, fail, ok

STAGE = "import"


def main() -> int:
    print(f"[{STAGE}] python {sys.version.split()[0]} on {platform.platform()}")
    try:
        import torch
    except Exception as e:
        print(f"[{STAGE}] FAILED to import torch: {e!r}")
        traceback.print_exc()
        emit(fail(STAGE, "import torch raised an exception -- the ROCm/torch wheel "
                          "install did not produce a working package on this machine",
                  {"exception": repr(e), "traceback": traceback.format_exc()}))
        return 1

    detail = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    print(f"[{STAGE}] torch imported OK: {torch.__version__}")

    try:
        device_available = torch.cuda.is_available()
    except Exception as e:
        print(f"[{STAGE}] torch imported but torch.cuda.is_available() raised: {e!r}")
        traceback.print_exc()
        detail["exception"] = repr(e)
        detail["traceback"] = traceback.format_exc()
        emit(fail(STAGE, "torch imported but querying device availability raised "
                          "an exception -- the HIP runtime did not initialize cleanly",
                  detail))
        return 1

    detail["device_available"] = device_available
    if not device_available:
        print(f"[{STAGE}] torch.cuda.is_available() == False -- no HIP/ROCm device visible")
        emit(fail(STAGE, "torch.cuda.is_available() returned False -- the runtime "
                          "did not detect any GPU (driver, HIP runtime, or "
                          "AMD_VARIANT_PROVIDER env vars may need attention; see README)",
                  detail))
        return 1

    try:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", None)
        detail.update({
            "device_count": device_count,
            "device_name": device_name,
            "device_properties": str(props),
            "gcn_arch_name": gcn_arch,
        })
        print(f"[{STAGE}] device_count={device_count} device_name={device_name!r} "
              f"gcnArchName={gcn_arch!r}")
    except Exception as e:
        print(f"[{STAGE}] device available but property query raised: {e!r}")
        traceback.print_exc()
        detail["exception"] = repr(e)
        detail["traceback"] = traceback.format_exc()
        emit(fail(STAGE, "a device was detected but querying its name/properties "
                          "raised an exception", detail))
        return 1

    haystack = f"{device_name} {gcn_arch}".lower()
    is_gfx1100 = "gfx1100" in haystack or "7900 xtx" in haystack or "7900xtx" in haystack
    detail["gfx1100_confirmed"] = is_gfx1100
    if is_gfx1100:
        print(f"[{STAGE}] PASS -- device confirmed as gfx1100 / 7900 XTX")
        emit(ok(STAGE, detail))
        return 0
    else:
        print(f"[{STAGE}] device detected but did NOT read as gfx1100/7900 XTX -- "
              f"reporting as a soft pass with a flag, not a hard failure")
        detail["note"] = ("a device was found and is usable, but its name/arch string "
                           "did not match gfx1100 or 7900 XTX -- check device_name and "
                           "gcn_arch_name above by hand")
        emit(ok(STAGE, detail))
        return 0


if __name__ == "__main__":
    sys.exit(main())
