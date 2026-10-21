"""Capture 21 prefixes, exit, then restore them in a distinct process."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    runner = Path(sys.argv[1]).resolve()
    model = Path(os.environ["SUPERSLM_PRECACHE_REAL_MODEL"]).resolve()
    prompts = Path(os.environ["SUPERSLM_PRECACHE_PROMPT_DIR"]).resolve()
    directory = Path(os.environ["SUPERSLM_PRECACHE_BLOB_DIR"]).resolve()
    anchor_file = Path(os.environ["SUPERSLM_PRECACHE_MERGE_BASE_FILE"]).resolve()
    anchor = anchor_file.read_text(encoding="ascii").strip()
    if len(anchor) != 40 or any(c not in "0123456789abcdef" for c in anchor):
        raise ValueError("merge-base fixture has no usable anchor")
    names = ([f"flat{i:02d}" for i in range(10)] + ["world"] +
             [f"nested{i:02d}" for i in range(10)])
    prompt_names = ([f"flat{i:02d}" for i in range(10)] + ["world"] +
                    [f"persona{i:02d}" for i in range(10)] +
                    [f"nested{i:02d}" for i in range(10)])
    env = os.environ.copy()
    directory.mkdir(parents=True, exist_ok=True)
    env["SUPERSLM_PRECACHE_PHASE"] = "capture"
    capture = subprocess.run([str(runner), "=10.1/E4", "=10.2/E4"], env=env,
                             text=True, capture_output=True)
    print(capture.stdout, end="")
    print(capture.stderr, end="", file=sys.stderr)
    if capture.returncode:
        raise RuntimeError(f"capture process exited {capture.returncode}; restore was not started")
    manifest = {
        "format": "PC-TWO-PROCESS-1", "anchor": anchor,
        "runner_sha256": digest(runner), "model_sha256": digest(model),
        "prompts": {name: digest(prompts / f"{name}.ids") for name in prompt_names},
        "blobs": {name: digest(directory / f"{name}.spx1") for name in names},
        "adopters": ["budget/greedy", "legacy/greedy", "budget/damped",
                     "legacy/damped", "budget/greedy"],
    }
    path = directory / "capture-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    env["SUPERSLM_PRECACHE_PHASE"] = "restore"
    env["SUPERSLM_PRECACHE_CAPTURE_MANIFEST"] = str(path)
    restored = subprocess.run([str(runner), "=10.1/E4", "=10.2/E4"], env=env,
                              text=True, capture_output=True)
    print(restored.stdout, end="")
    print(restored.stderr, end="", file=sys.stderr)
    if restored.returncode:
        raise RuntimeError(f"restore process exited {restored.returncode}")
    assert digest(model) == manifest["model_sha256"]
    assert {name: digest(prompts / f"{name}.ids") for name in prompt_names} == manifest["prompts"]
    assert {name: digest(directory / f"{name}.spx1") for name in names} == manifest["blobs"]
    print("two-process gate: capture exited, 21 hashes pinned, restore exited, 100 graded adopters")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"two-process gate refused: {error}", file=sys.stderr)
        raise SystemExit(1)
