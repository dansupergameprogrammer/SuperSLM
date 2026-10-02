"""Capture one real 4.3 producer job and its independently hashed artifacts.

Run on each compiler host, with a fresh build directory per compiler. The
resulting MSVC/GNU/Clang bundles are copied together into CROSS_COMPILER_DIR.
"""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import subprocess


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def cmake_field(text: str, key: str) -> str:
    match = re.search(r"^set\(" + re.escape(key) + r' "([^"]+)"\)', text, re.M)
    if not match:
        raise ValueError(f"CMake compiler record lacks {key}")
    return match.group(1)


def run(args: list[str], **kwargs) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, **kwargs)
    if result.returncode:
        raise RuntimeError(f"job exited {result.returncode}: {' '.join(args)}\n{result.stdout}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    build = args.build_dir.resolve()
    bundle = args.bundle_dir.resolve()
    if build.exists() and any(build.iterdir()):
        raise ValueError("compiler job requires a fresh empty build directory")
    build.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(parents=True, exist_ok=True)
    revision = run(["git", "-C", str(source), "rev-parse", "HEAD"]).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source revision is not a Git commit")
    if run(["git", "-C", str(source), "status", "--porcelain"]).strip():
        raise ValueError("compiler job requires a clean source revision")
    configure = run([args.cmake, "-S", str(source), "-B", str(build),
                     "-G", args.generator])
    records = list(build.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if len(records) != 1:
        raise ValueError(f"expected one compiler record, found {len(records)}")
    cmake_text = records[0].read_text(encoding="utf-8")
    compiler = cmake_field(cmake_text, "CMAKE_CXX_COMPILER_ID")
    version = cmake_field(cmake_text, "CMAKE_CXX_COMPILER_VERSION")
    compiler_path = Path(cmake_field(cmake_text, "CMAKE_CXX_COMPILER"))
    if compiler not in {"MSVC", "GNU", "Clang"} or not compiler_path.is_file():
        raise ValueError("unsupported or missing CMake compiler binary")
    build_log = configure + run([args.cmake, "--build", str(build), "--config",
                                 args.config, "--target", "superslm_precache_e1"])
    runner_name = "superslm_precache_e1.exe" if os.name == "nt" else "superslm_precache_e1"
    candidates = [build / args.config / runner_name, build / runner_name]
    runner = next((p for p in candidates if p.is_file()), None)
    if runner is None:
        raise FileNotFoundError(f"built runner missing: {candidates}")
    env = os.environ.copy()
    env.update(SUPERSLM_PRECACHE_CROSS_COMPILER_DIR=str(bundle),
               SUPERSLM_PRECACHE_COMPILER_PHASE="capture",
               SUPERSLM_PRECACHE_RUNNER_EXE=str(runner))
    capture_log = run([str(runner), "=4.3/E4"], env=env)
    if "4.3/E4 green" not in capture_log or "1 cells, 0 red" not in capture_log:
        raise RuntimeError("selected 4.3 compiler capture did not pass")
    stem = bundle / compiler
    shutil.copyfile(runner, stem.with_suffix(".runner"))
    shutil.copyfile(compiler_path, stem.with_suffix(".compiler"))
    shutil.copyfile(records[0], stem.with_suffix(".cmake"))
    (bundle / f"{compiler}.build.log").write_text(build_log, encoding="utf-8")
    (bundle / f"{compiler}.capture.log").write_text(capture_log, encoding="utf-8")
    receipt = dict(line.split("=", 1) for line in
                   (bundle / f"{compiler}.receipt").read_text(encoding="utf-8").splitlines())
    if (receipt.get("compiler") != compiler or
            receipt.get("compiler_version") != version or
            receipt.get("runner_sha256") != digest(runner) or
            receipt.get("blob_sha256") != digest(bundle / f"{compiler}.spx1")):
        raise ValueError("capture receipt does not match the built producer and its blob")
    job = dict(format="PC-COMPILER-JOB-1", compiler=compiler,
               compiler_version=version, runner_sha256=digest(stem.with_suffix(".runner")),
               compiler_tool_sha256=digest(stem.with_suffix(".compiler")),
               cmake_record_sha256=digest(stem.with_suffix(".cmake")),
               build_log_sha256=digest(bundle / f"{compiler}.build.log"),
               capture_log_sha256=digest(bundle / f"{compiler}.capture.log"),
               blob_sha256=digest(bundle / f"{compiler}.spx1"),
               source_revision=revision)
    (bundle / f"{compiler}.job").write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(job.items())), encoding="ascii")
    print(f"captured {compiler} {version} job {revision}: {job['runner_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"compiler job refused: {error}")
