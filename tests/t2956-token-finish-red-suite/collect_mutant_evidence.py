"""Freeze the scratch production-mutant readings into a tracked evidence manifest."""
from __future__ import annotations

import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
LOGS = Path(r"D:\_t2956-mutants\logs")
IDS = ("a b b2 c d e f g h h2 i j k l1 l2 l3 l4 l5 l6 m n "
       "n2_upload n2_restore n3_upload n3_restore o o2 p").split()
LANDING = {
    "a": "forward_sites.cpp:2838",
    "b": "forward_sites.cpp:2848-2851",
    "b2": "forward_sites.cpp:2848-2851",
    "c": "forward_sites.cpp:2792-2802 (semantic proxy; exact task-local mutant unexecuted)",
    "d": "logits_site.hlsl:75",
    "e": "gpu_1p0.cpp:985-988",
    "f": "gpu_1p0.cpp:985-988",
    "g": "sslm_abi.cpp prefill path",
    "h": "gpu_1p0.cpp:1180",
    "h2": "gpu_1p0.cpp:1200-1207",
    "i": "gpu_1p0.cpp:2944-2951",
    "j": "gpu_1p0.cpp:851-858",
    "k": "gpu_1p0.cpp:851-858",
    "m": "gpu_1p0.cpp:1200-1207",
    "n": "gpu_1p0.cpp:890",
    "o": "gpu_1p0.cpp:922",
    "o2": "gpu_1p0.cpp:4142",
    "p": "d3d12_harness.h:412-418",
}


def read_log(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith(b"\xff\xfe") else "utf-8",
                       errors="replace")


def main() -> None:
    results = []
    for ident in IDS:
        folder = LOGS / ident
        mutation = read_log(folder / "mutation.txt")
        matches = re.findall(r"^MUTANT \S+ (\S+) sha256=([0-9a-f]{64})\r?$",
                             mutation, re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f"{ident}: missing single production-source digest")
        lines = read_log(folder / "cell.txt").splitlines()
        failed = [line.strip().split(" : ", 1)[-1] for line in lines
                  if re.search(r"(?:^|: )FAIL (?!HR )", line)]
        if not failed:
            raise RuntimeError(f"{ident}: no failing cell assertion")
        landing = LANDING.get(ident)
        if landing is None:
            if ident.startswith("l"):
                landing = "gpu_1p0.cpp:851, bundle allocation " + ident[1]
            elif ident.startswith("n2_") or ident.startswith("n3_"):
                landing = ("gpu_1p0.cpp:618-633" if ident.endswith("upload") else
                           "superslm_gpu.cpp:4587-4596")
            else:
                raise RuntimeError(f"{ident}: missing landing")
        results.append({"id": ident, "landing": landing, "source": matches[0][0],
                        "source_sha256": matches[0][1], "reading": failed[0],
                        "classification": "semantic proxy only" if ident == "c" else
                                          "executed production mutant killed"})
    target = HERE / "mutant_evidence.json"
    target.write_text(json.dumps({"builder_commit":
        "c9081e0fd33fccabf52e3fb247ad2f51efb74e6e", "mutants": results},
        indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {target} mutants={len(results)}")


if __name__ == "__main__":
    main()
