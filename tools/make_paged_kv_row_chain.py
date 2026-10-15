"""Paged-KV plan rev 16.21 cell 7.9 (§23.8 step T4): the row blob chain and its manifests.

One chain of the 7.9 harness stream (seed 79) per target, made by the production pair executable's `prefill` mode
and cut at every row's restore width: e.g. 1,200, 2,044 and 3,988 on the 0.5B; 4,096, 8,188, 12,284, 15,872 and
18,678 on the 1.5B, where 4,096 and 12,284 are piece limits rather than rows. A piece is at most 4,096 tokens below
position 15,872 and at most 512 at or above it; a piece that overruns --cap-seconds (25 minutes by default) is
discarded and rerun at half its length. The chain resumes from its last verified piece.

For each restore width N the tool writes manifest_r<N>.json, the form tools/run_paged_kv_rows.py's check_row_blob
(79-6) verifies before every process: {"complete": true, "model_sha256", "stream": {"seed": 79, "restore": N},
"pieces": [{"file", "start", "count", "input_sha256", "sha256"}, ...] up to N, "final_blob", "final_sha256"}, plus
the geometry and the executable's SHA-256 for the record. chain.json holds the whole chain.

    make_paged_kv_row_chain.py --exe superslm_pkv_pair.exe --model F --cap C --layers L --heads H --dim D
                               --restores 1200,2044,3988 --directory DIR [--cap-seconds 1500]

Box-owed for real artifacts: making a real chain loads the model. On the bench it runs only on generated fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FAST_LIMIT = 4096   # tokens per piece below SLOW_FROM
SLOW_LIMIT = 512    # tokens per piece at or above SLOW_FROM, where prefill runs at decode speed
SLOW_FROM = 15872
CAP_SECONDS = 1500  # 25 minutes


class Refusal(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def piece_end(start, restores, fast_limit=FAST_LIMIT, slow_limit=SLOW_LIMIT, slow_from=SLOW_FROM):
    """Where the piece starting at `start` ends: the next restore width, the piece limit, or slow_from."""
    limit = fast_limit if start < slow_from else slow_limit
    end = start + limit
    if start < slow_from:
        end = min(end, slow_from)
    return min([end] + [r for r in restores if r > start])


def verified_prefix(chain, directory, exe_sha):
    """The pieces of a previous chain.json whose chain still verifies, in order. A piece is reused only when the
    executable resuming the chain made it: each piece records its maker's SHA-256 (exe_sha256), and a reusable piece
    made by another executable, or recorded without one, is refused rather than relabelled (T-3202 F-6)."""
    out, at, previous = [], 0, "-"
    for piece in chain.get("pieces", []):
        path = directory / piece.get("file", "")
        if (piece.get("start") != at or piece.get("input_sha256") != previous or not path.is_file()
                or sha(path) != piece.get("sha256")):
            break
        if piece.get("exe_sha256") != exe_sha:
            raise Refusal(f"piece {piece.get('file')} was made by executable {piece.get('exe_sha256')}, not by "
                          f"{exe_sha}, the one resuming the chain; remove {directory} or resume with that executable")
        out.append(piece)
        at += piece["count"]
        previous = piece["sha256"]
    return out


def run_piece(command, log, cap_seconds, env):
    with log.open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        start = time.monotonic()
        while True:
            try:
                return process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                print(f"heartbeat pid={process.pid} elapsed={elapsed:.0f}s log={log}", flush=True)
                if elapsed >= cap_seconds:
                    process.kill()
                    process.wait()
                    return None


def make_chain(exe, model, cap, layers, heads, dim, restores, directory, cap_seconds=CAP_SECONDS,
               fast_limit=FAST_LIMIT, slow_limit=SLOW_LIMIT):
    exe, model, directory = Path(exe).resolve(), Path(model).resolve(), Path(directory).resolve()
    restores = sorted(set(int(r) for r in restores))
    if not restores or restores[0] <= 0 or not exe.is_file() or not model.is_file():
        raise Refusal("an executable, a model and positive restore widths are owed")
    directory.mkdir(parents=True, exist_ok=True)
    model_sha, exe_sha = sha(model), sha(exe)
    width = restores[-1] + 512  # the prefill mode's stream is Stream(79, width - 512); a prefix of it is every row's
    geometry = [cap, layers, heads, dim]
    chain_path = directory / "chain.json"
    try:
        old = json.loads(chain_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    same = old.get("model_sha256") == model_sha and old.get("geometry") == geometry
    pieces = verified_prefix(old, directory, exe_sha) if same else []
    chain = {"model_sha256": model_sha, "exe_sha256": exe_sha, "geometry": geometry,
             "stream": {"seed": 79, "tokens": restores[-1]}, "restores": restores, "pieces": pieces}
    env = os.environ.copy()
    env["SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR"] = str(model.parent)
    at = sum(p["count"] for p in pieces)
    halve = {}
    while at < restores[-1]:
        end = piece_end(at, restores, fast_limit, slow_limit)
        count = min(end - at, halve.get(at, end - at))
        prior = directory / pieces[-1]["file"] if pieces else None
        output = directory / f"piece_{at:05d}_{count:04d}.blob"
        partial = output.with_name(output.name + ".partial")
        partial.unlink(missing_ok=True)
        command = [str(exe), "prefill", model.name, str(cap), str(layers), str(heads), str(dim), str(width), str(at),
                   str(count), str(prior) if prior else "-", str(partial)]
        result = run_piece(command, directory / f"piece_{at:05d}.log", cap_seconds, env)
        if result is None:
            partial.unlink(missing_ok=True)
            if count == 1:
                raise Refusal(f"a one-token piece at {at} overran {cap_seconds} s")
            halve[at] = max(1, count // 2)
            print(f"piece {at}+{count} overran {cap_seconds} s; rerun at {halve[at]}", flush=True)
            continue
        if result != 0 or not partial.is_file():
            raise Refusal(f"prefill piece {at}+{count} failed (exit {result}); see piece_{at:05d}.log")
        os.replace(partial, output)
        digest = sha(output)
        pieces.append({"file": output.name, "start": at, "count": count,
                       "input_sha256": pieces[-1]["sha256"] if pieces else "-", "sha256": digest, "exe_sha256": exe_sha})
        at += count
        atomic_json(chain_path, chain)
        print(f"piece {output.name} sha256={digest}", flush=True)
    atomic_json(chain_path, chain)
    manifests = {}
    for restore in restores:
        upto = []
        for piece in pieces:
            upto.append(piece)
            if piece["start"] + piece["count"] == restore:
                break
        if sum(p["count"] for p in upto) != restore:
            raise Refusal(f"no piece ends at restore {restore}")
        manifest = {"complete": True, "model_sha256": model_sha, "stream": {"seed": 79, "restore": restore},
                    "pieces": upto, "final_blob": upto[-1]["file"], "final_sha256": upto[-1]["sha256"],
                    "geometry": geometry, "exe_sha256": exe_sha}
        path = directory / f"manifest_r{restore}.json"
        atomic_json(path, manifest)
        manifests[restore] = path
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cap", required=True, type=int)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--heads", required=True, type=int)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--restores", required=True, help="comma-separated restore widths")
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--cap-seconds", type=int, default=CAP_SECONDS)
    parser.add_argument("--fast-limit", type=int, default=FAST_LIMIT)
    parser.add_argument("--slow-limit", type=int, default=SLOW_LIMIT)
    args = parser.parse_args(argv)
    try:
        manifests = make_chain(args.exe, args.model, args.cap, args.layers, args.heads, args.dim,
                               [int(x) for x in args.restores.split(",") if x], args.directory, args.cap_seconds,
                               args.fast_limit, args.slow_limit)
    except Refusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 1
    for restore, path in manifests.items():
        print(f"restore {restore}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
