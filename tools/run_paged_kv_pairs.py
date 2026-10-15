"""Production-library C6 prompt checkpoints and in-process 7.9/7.9b pairs.

`prepare` checkpoints at most 4096 prompt tokens per production-library process.
`run` grades exactly 508 pairs; a shorter invocation is an ungraded mechanism run.
All results remain quarantined until independent instrument commissioning.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import struct
import subprocess
import sys
import time


class Refusal(RuntimeError):
    pass


BOX_LOCK = Path("D:/_scratch/SuperSLM/pkv-c6-timing.lock")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_record(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    required = {"kind", "complete", "model_hashes", "token_sha256", "pairs_sha256", "pair_count",
                "exe_sha256", "library_sha256", "seed", "status", "hook"}
    return record if record.get("complete") is True and required <= record.keys() else None


def pid_live(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True)
        return result.returncode == 0 and re.search(rf',"{pid}"(?:,|$)', result.stdout) is not None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def live_load_state(record: Path) -> dict:
    try:
        state = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Refusal("load generator record torn") from None
    if (not pid_live(int(state.get("pid", 0))) or state.get("phase") not in ("on", "off") or
        state.get("complete") is not False or not state.get("worker_pids") or
        not all(pid_live(int(pid)) for pid in state["worker_pids"])):
        raise Refusal("load generator stopped or worker dead")
    return state


_LOCK_BYTE = 1 << 20
LOCK_PID_MAX = 2**31 - 1  # a pid is 1 to this (T-3206 N-4); other content is torn
LOCK_TRIM = " \t\r\n"  # the only characters stripped around the pid, as run_paged_kv_rows.py's (T-3210 N-1)
LOCK_MAX_BYTES = 4096  # a longer lock file is torn, as run_paged_kv_rows.py's (T-3211 O-1)


def _read_capped(fd: int) -> bytes:
    """At most LOCK_MAX_BYTES + 1 bytes of `fd` from its start; more than LOCK_MAX_BYTES means the file is too long."""
    os.lseek(fd, 0, os.SEEK_SET)
    data = b""
    while len(data) <= LOCK_MAX_BYTES:
        chunk = os.read(fd, LOCK_MAX_BYTES + 1 - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _os_lock(fd: int) -> bool:
    """The exclusive, non-blocking OS byte lock on `fd`, one byte far past any content; False when another process
    holds it. The OS releases it when the holder exits, however it exits."""
    os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, _LOCK_BYTE)
        return True
    except OSError:
        return False


def _os_unlock(fd: int) -> None:
    os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.lockf(fd, fcntl.LOCK_UN, 1, _LOCK_BYTE)
    except OSError:
        pass


def boot_epoch() -> float | None:
    """The system's last boot, in seconds since the epoch: now less the uptime (GetTickCount64 on Windows, counting
    sleep and hibernation; CLOCK_BOOTTIME on Linux); None where neither exists. The same reading as
    run_paged_kv_rows.py's and paged_kv_c6_provenance.ps1's (D-SLM8466 (3))."""
    if os.name == "nt":
        import ctypes
        tick = ctypes.windll.kernel32.GetTickCount64
        tick.restype = ctypes.c_uint64
        return time.time() - tick() / 1000.0
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    return time.time() - time.clock_gettime(boottime) if boottime is not None else None


def _before_boot(fd: int) -> bool:
    """True when the lock file's modification time is earlier than the last boot (D-SLM8466 (3))."""
    boot = boot_epoch()
    return boot is not None and os.fstat(fd).st_mtime < boot


class PidLock:
    """The pid lock: the file holds the owner's bare decimal pid. D-SLM8464 gives the box timing lock one format and one
    protocol in every program that takes it (run_paged_kv_rows.py's timing_lock, paged_kv_c6_provenance.ps1's
    Enter-PkvTimingLock and Lock74 in tests/paged-kv/c6_dim7_reset_adopt_timing.cpp are the others): a fresh acquire moves a temporary file holding the pid into place without
    replacing, so the lock is never visible empty; an existing file is recovered only under an exclusive non-blocking OS
    lock on it; then a file modified before the system's last boot is stale whatever pid it holds (D-SLM8466 (3): that
    pid may now name an unrelated process); otherwise a pid liveness check (a live pid refuses); then an overwrite with
    this pid. An empty file is stale; content other than ASCII digits naming a pid from 1 to LOCK_PID_MAX, with only
    space, tab, CR and LF stripped around it (LOCK_TRIM), is torn (T-3206 N-4, T-3210 N-1), and so is a file longer
    than LOCK_MAX_BYTES, whatever its first LOCK_MAX_BYTES hold (T-3211 O-1). A release unlinks the file when it still
    holds this pid and never truncates it; a file longer than LOCK_MAX_BYTES holds no pid."""

    ATTEMPTS = 3

    def __init__(self, path: Path):
        self.path = Path(path)

    def _publish(self, pid: int) -> bool:
        temp = self.path.with_name(f"{self.path.name}.{pid}.new")
        temp.write_bytes(str(pid).encode("ascii"))
        try:
            if os.name == "nt":
                os.rename(temp, self.path)  # MoveFileEx without MOVEFILE_REPLACE_EXISTING: fails when the lock exists
            else:
                os.link(temp, self.path)  # fails when the lock exists
            return True
        except (FileExistsError, PermissionError):
            return False
        finally:
            temp.unlink(missing_ok=True)

    def _recover(self, pid: int) -> bool:
        try:
            fd = os.open(self.path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        except (FileNotFoundError, PermissionError):
            return False  # gone (or going): retry the fresh acquire
        try:
            if not _os_lock(fd):
                raise Refusal(f"second runner refused: {self.path} is being recovered by another runner")
            try:
                try:
                    if not os.path.samestat(os.fstat(fd), os.stat(self.path)):
                        return False  # released and replaced after this open
                except FileNotFoundError:
                    return False
                stale = _before_boot(fd)
                data = _read_capped(fd)
                if len(data) > LOCK_MAX_BYTES and not stale:
                    raise Refusal(f"torn pid lock: {self.path} is longer than {LOCK_MAX_BYTES} bytes")
                text = data.decode("ascii", "replace").strip(LOCK_TRIM)
                if text and not stale:
                    # T-3206 N-4: ASCII digits naming a pid from 1 to 2**31 - 1, as run_paged_kv_rows.py's _read_owner
                    owner = int(text) if text.isascii() and text.isdigit() else 0
                    if not 1 <= owner <= LOCK_PID_MAX:
                        raise Refusal(f"torn pid lock: {self.path}")
                    if pid_live(owner):
                        raise Refusal(f"second runner refused: pid {owner}")
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                os.write(fd, str(pid).encode("ascii"))
                return True
            finally:
                _os_unlock(fd)
        finally:
            os.close(fd)

    def __enter__(self):
        pid = os.getpid()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(self.ATTEMPTS):
            if self._publish(pid) or self._recover(pid):
                return self
        raise Refusal(f"second runner refused: {self.path} could not be acquired in {self.ATTEMPTS} attempts")

    def __exit__(self, *_):
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            try:
                data = _read_capped(fd)
            finally:
                os.close(fd)
            own = str(os.getpid()).encode("ascii")
            if len(data) <= LOCK_MAX_BYTES and data.strip(LOCK_TRIM.encode("ascii")) == own:
                self.path.unlink()
        except OSError:
            pass  # a reader holds it open: it keeps this pid, stale once this process exits


def production_library(exe: Path) -> tuple[str, str]:
    if exe.name.lower() != "superslm_pkv_pair.exe":
        raise Refusal("expected the production superslm_pkv_pair.exe target")
    lib = exe.with_name("superslm.lib")
    if not exe.is_file() or not lib.is_file():
        raise Refusal("production pair executable or superslm.lib missing")
    identity = subprocess.run([str(exe), "identity"], capture_output=True, text=True, timeout=30)
    if identity.returncode or identity.stdout.strip() != "PKV_PAIR_PRODUCTION_V2 defs=none":
        raise Refusal("production pair identity or build definitions invalid")
    try:
        manifest = json.loads(Path(str(exe) + ".source-set.json").read_text(encoding="utf-8"))
        commands = manifest["build_recipe"]["commands"]
        compiled = [line for block in commands.values() for line in block.splitlines()
                    if re.search(r"(?:cl\.exe|clang\+\+|g\+\+).*\.(?:cpp|cc|cxx)\b", line, re.I)]
        if (not compiled or any(re.search(r"(?:/Od\b|-O0\b)", line, re.I) or
                                not re.search(r"(?:/O2\b|-O[23]\b)", line, re.I) for line in compiled)):
            raise Refusal("production pair build is not optimized")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise Refusal("production pair build recipe missing") from None
    return sha(exe), sha(lib)


def source_bound_receipt(exe: Path, kind: str) -> tuple[bool, str | None]:
    """Check a verified successor receipt against independent source authority."""
    receipt_id = "PKV-TIMING-79-C6-R168" if kind == "79" else "PKV-TIMING-79B-C6"
    # Test fixtures may call verify_paged_kv_binding.bound directly. A graded
    # run must never redirect its registry or verifier through its environment.
    if "PKV_C6_REGISTRY" in os.environ or "PKV_C6_RECEIPT_VERIFIER" in os.environ:
        return False, None
    registry_path = Path("D:/Wizard/Claude/Tooling/instrument-registry.json")
    manifest_path = exe.with_suffix(exe.suffix + ".source-set.json")
    if not registry_path.is_file() or not manifest_path.is_file():
        return False, None
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entry = registry.get(receipt_id)
        if not entry:
            return False, sha(manifest_path)
        verifier = Path("D:/Wizard/Claude/Tooling/instrument-commission.ps1")
        verified = subprocess.run(["powershell", "-NoProfile", "-File", str(verifier), "-Verify"],
                                  capture_output=True, text=True, timeout=120)
        if f"COMMISSIONED  {receipt_id}" not in verified.stdout.splitlines():
            return False, sha(manifest_path)
        repo = Path(__file__).resolve().parents[1]
        from verify_paged_kv_binding import bound
        return bound(exe, repo, registry_path, receipt_id)
    except (OSError, ValueError, KeyError, ImportError, subprocess.TimeoutExpired):
        return False, sha(manifest_path)


def box_observation() -> dict:
    if os.name != "nt":
        try:
            return {"load_1m": os.getloadavg()[0], "offenders": []}
        except OSError:
            return {"load_1m": None, "offenders": []}
    command = ("$p=Get-Process -Name UnrealEditor,UnrealEditor-Cmd,cl,link,dotnet,UnrealBuildTool,smoke_adapter_driver,superslm_pkv_* "
               "-ErrorAction SilentlyContinue | ForEach-Object { \"$($_.ProcessName):$($_.Id)\" };"
               "$cpu=(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average;"
               "@{cpu_percent=$cpu;offenders=@($p);build_lock=(Test-Path 'D:\\_ssu_build_lock')} "
               "| ConvertTo-Json -Compress")
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True,
                                text=True, timeout=15)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return {"cpu_percent": None, "offenders": [], "observation_error": True}


def wait_for_smoke() -> None:
    if os.name == "nt":
        while True:
            listing = subprocess.run(["C:/Windows/System32/tasklist.exe", "/FI",
                                      "IMAGENAME eq smoke_adapter_driver.exe", "/NH"],
                                     capture_output=True, text=True, timeout=15)
            if listing.returncode:
                raise Refusal("cannot check smoke_adapter_driver before process start")
            if "smoke_adapter_driver.exe" not in listing.stdout.lower():
                break
            print("waiting for smoke_adapter_driver.exe before process start", flush=True)
            time.sleep(30)
def run_logged(command: list[str], log: Path, cap: int, env: dict[str, str]) -> int | None:
    wait_for_smoke()
    start = time.monotonic()
    with log.open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        while True:
            try:
                return process.wait(timeout=max(0.1, min(30, cap - (time.monotonic() - start))))
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                print(f"heartbeat pid={process.pid} elapsed={elapsed:.0f}s log={log}", flush=True)
                if elapsed >= cap:
                    process.kill()
                    process.wait()
                    return None


def validate_manifest(manifest: dict, model_sha: str, width: int, directory: Path,
                      geometry: list[int] | None = None) -> int:
    if (manifest.get("model_sha256") != model_sha or
        manifest.get("stream") != {"seed": 79, "width": width, "prompt_tokens": width - 512} or
        (geometry is not None and manifest.get("geometry") != geometry)):
        return 0
    at, previous = 0, "-"
    for piece in manifest.get("pieces", []):
        path = directory / piece.get("file", "")
        if (piece.get("start") != at or piece.get("input_sha256") != previous or not path.is_file() or sha(path) != piece.get("sha256")):
            return at
        at += piece.get("count", 0)
        previous = piece["sha256"]
    return at


def prepare(args) -> dict:
    exe, model, directory = args.exe.resolve(), args.model.resolve(), args.directory.resolve()
    production_library(exe)
    if not model.is_file() or args.width <= 512 or not 1 <= args.piece_limit <= 4096:
        raise Refusal("model, width or piece limit invalid")
    directory.mkdir(parents=True, exist_ok=True)
    model_sha = sha(model)
    manifest_path = directory / "manifest.json"
    with PidLock(directory / "prepare.lock"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            manifest = {}
        at = validate_manifest(manifest, model_sha, args.width, directory,
                               [args.cap, args.layers, args.heads, args.dim])
        if at == 0:
            manifest = {"model_sha256": model_sha, "stream": {"seed": 79, "width": args.width, "prompt_tokens": args.width - 512},
                        "geometry": [args.cap, args.layers, args.heads, args.dim], "pieces": [], "next_piece_limit": args.piece_limit}
        else:
            manifest["pieces"] = [p for p in manifest["pieces"] if p["start"] < at]
        limit = min(manifest.get("next_piece_limit", args.piece_limit), 4096)
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR"] = str(model.parent)
        while at < args.width - 512:
            count = min(limit, args.width - 512 - at)
            prior = directory / manifest["pieces"][-1]["file"] if manifest["pieces"] else None
            output = directory / f"piece_{at:05d}_{count:04d}.blob"
            partial = output.with_suffix(".partial")
            partial.unlink(missing_ok=True)
            command = [str(exe), "prefill", model.name, str(args.cap), str(args.layers), str(args.heads), str(args.dim),
                       str(args.width), str(at), str(count), str(prior) if prior else "-", str(partial)]
            result = run_logged(command, directory / f"piece_{at:05d}.log", args.cap_seconds, env)
            if result is None:
                partial.unlink(missing_ok=True)
                if count == 1:
                    raise Refusal(f"one-token piece at {at} exceeded {args.cap_seconds}s")
                limit = max(1, count // 2)
                manifest["next_piece_limit"] = limit
                atomic_json(manifest_path, manifest)
                continue
            if result or not partial.is_file():
                raise Refusal(f"prefill piece {at}+{count} failed; see piece_{at:05d}.log")
            os.replace(partial, output)
            digest = sha(output)
            manifest["pieces"].append({"start": at, "count": count, "file": output.name,
                                        "input_sha256": sha(prior) if prior else "-", "sha256": digest})
            at += count
            manifest["next_piece_limit"] = limit
            atomic_json(manifest_path, manifest)
            print(f"checkpoint prompt {at}/{args.width - 512} sha256={digest}", flush=True)
        manifest["complete"] = True
        manifest["final_blob"] = manifest["pieces"][-1]["file"]
        manifest["final_sha256"] = manifest["pieces"][-1]["sha256"]
        atomic_json(manifest_path, manifest)
        return manifest


def checked_blob(manifest_path: Path, model: Path, width: int,
                 geometry: list[int] | None = None) -> tuple[Path, str]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        raise Refusal(f"missing or torn blob manifest: {manifest_path}") from None
    if (not manifest.get("complete") or
        validate_manifest(manifest, sha(model), width, manifest_path.parent, geometry) != width - 512):
        raise Refusal(f"blob/manifest mismatch: {manifest_path}")
    blob = manifest_path.parent / manifest.get("final_blob", "")
    pieces = manifest.get("pieces") or []
    if (not pieces or manifest.get("final_blob") != pieces[-1].get("file") or
        manifest.get("final_sha256") != pieces[-1].get("sha256") or
        not blob.is_file() or sha(blob) != manifest.get("final_sha256")):
        raise Refusal(f"final blob mismatch: {blob}")
    return blob, sha(manifest_path)


def parse_pairs(path: Path, count: int) -> tuple[list[float], str]:
    slowdown, tokens = [], hashlib.sha256()
    with path.open(newline="") as source:
        rows = list(csv.reader(source))
    if len(rows) != count:
        raise Refusal(f"incomplete pair set: {len(rows)} of {count}")
    for index, row in enumerate(rows):
        try:
            if len(row) != 6 or int(row[0]) != index or row[1] not in ("A", "B"):
                raise Refusal(f"invalid pair row {index}")
            paged, one = float(row[2]), float(row[3])
            a, b = int(row[4]), int(row[5])
        except (ValueError, IndexError) as exc:
            raise Refusal(f"invalid pair row {index}") from exc
        if not math.isfinite(paged) or not math.isfinite(one) or paged <= 0 or one <= 0:
            raise Refusal(f"invalid duration at pair {index}")
        if a != b:
            raise Refusal(f"token-stream mismatch at pair {index}")
        tokens.update(struct.pack("<i", a))
        slowdown.append(1.0 - one / paged)
    return slowdown, tokens.hexdigest()


def grade(values: list[float]) -> dict:
    if len(values) != 508:
        return {"status": "MECHANISM_ONLY", "pair_count": len(values)}
    sorted_values = sorted(values)
    lo, hi = sorted_values[228], sorted_values[279]
    return {"status": "QUARANTINED", "pair_count": 508, "median_slowdown": statistics.median(values),
            "half_width": (hi - lo) / 2, "sign_interval": [lo, hi],
            "effect": "RESOLVED" if lo > 0 or hi < 0 else "NO RESULT",
            "bar": "PASS" if hi <= .05 else "FAIL" if lo > .05 else "NO RESULT"}


RUN_LIST_SLOTS = {"79b": "R3"}


def read_result(path: Path, expected: dict) -> dict | None:
    record = load_record(path)
    if record is None or any(record.get(key) != value for key, value in expected.items()):
        return None
    pairs_file = path.with_suffix(".pairs.csv")
    if not pairs_file.is_file() or sha(pairs_file) != record["pairs_sha256"]:
        return None
    try:
        values, token_sha = parse_pairs(pairs_file, record["pair_count"])
    except (OSError, Refusal):
        return None
    if token_sha != record["token_sha256"]:
        return None
    computed = grade(values)
    if any(record.get(key) != value for key, value in computed.items() if key != "status"):
        return None
    return record


def run(args) -> dict:
    if args.pairs != 508 and not getattr(args, "mechanism", False):
        raise Refusal("incomplete graded pair set: exactly 508 required")
    exe = args.exe.resolve()
    exe_sha, lib_sha = production_library(exe)
    receipt_bound, source_manifest_sha = source_bound_receipt(exe, args.kind)
    model_a = args.model_a.resolve()
    model_b = (args.model_b or args.model_a).resolve()
    if not model_a.is_file() or not model_b.is_file() or model_a.parent != model_b.parent:
        raise Refusal("both models must exist in one artifact directory")
    model_hashes = [sha(model_a), sha(model_b)]
    if args.kind == "79b" and args.model_b:
        raise Refusal("7.9b uses one model")
    manifest_hashes, blobs = [], []
    if args.kind == "79":
        if not args.manifest_a or not args.manifest_b:
            raise Refusal("7.9 requires a manifest for each arm")
        if args.cap_b is None:
            raise Refusal("7.9 requires the one-page arm's cap")
        for manifest, model, cap in ((args.manifest_a, model_a, args.cap_a),
                                     (args.manifest_b, model_b, args.cap_b)):
            blob, digest = checked_blob(manifest.resolve(), model, args.width,
                                        [cap, args.layers, args.heads, args.dim])
            blobs.append(blob)
            manifest_hashes.append(digest)
    result = args.result.resolve()
    result.parent.mkdir(parents=True, exist_ok=True)
    BOX_LOCK.parent.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    hook = os.environ.get("SUPERSLM_PAGED_KV_79_SLOW_ARM", "")
    load_record = getattr(args, "load_record", None)
    if load_record and not load_record.is_file():
        raise Refusal(f"load generator record missing: {load_record}")
    load_qualified = not load_record
    if load_record:
        load_state = live_load_state(load_record)
        calibration = load_state.get("calibration") or {}
        load_qualified = False
        if receipt_bound and calibration.get("qualified") and calibration.get("slowdown", 0) >= 2.8:
            try:
                rid = "PKV-TIMING-79-C6-R168" if args.kind == "79" else "PKV-TIMING-79B-C6"
                claim = json.loads(Path("D:/Wizard/Claude/Tooling/instrument-registry.json").read_text())[rid]["source_binding"]
                probe = Path(calibration["probe_path"]).resolve()
                load_qualified = (probe.is_file() and sha(probe) == calibration["probe_sha256"] ==
                                  claim["load_probe_sha256"])
            except (OSError, ValueError, KeyError, TypeError):
                load_qualified = False
    load_sha = sha(load_record) if load_record else None
    standing = receipt_bound and load_qualified
    # D-SLM8465: every pair-row record names its run-list slot; the 7.9b leg is row R3 (the 7.9 kind has no row of its
    # own: the five-process driver, run_paged_kv_rows.py, grades 7.9's rows). In `expected` too, so an earlier record
    # without its slot is rerun rather than reused.
    slot = RUN_LIST_SLOTS.get(args.kind)
    expected = {"kind": args.kind, "slot": slot, "exe_sha256": exe_sha, "library_sha256": lib_sha,
                "source_manifest_sha256": source_manifest_sha,
                "model_hashes": model_hashes, "manifest_hashes": manifest_hashes,
                "pair_count": args.pairs, "seed": seed, "hook": hook, "load_sha256": load_sha,
                "status": "MUTANT_CONSTRUCTION" if hook else
                          "MECHANISM_ONLY" if getattr(args, "mechanism", False) else
                          "STANDING" if standing else "QUARANTINED"}
    with PidLock(BOX_LOCK):
        wait_for_smoke()
        if load_record:
            live_load_state(load_record)
        prior = read_result(result, expected)
        if prior is not None:
            return prior
        before = box_observation()
        pairs_file = result.with_suffix(".pairs.partial")
        pairs_file.unlink(missing_ok=True)
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR"] = str(model_a.parent)
        if args.kind == "79":
            command = [str(exe), "pair79", model_a.name, model_b.name, str(args.cap_a), str(args.cap_b),
                       str(args.layers), str(args.heads), str(args.dim), str(blobs[0]), str(blobs[1]),
                       str(args.pairs), str(seed), str(pairs_file), str(args.width - 512)]
        else:
            command = [str(exe), "pair79b", model_a.name, str(args.cap_a), str(args.layers), str(args.heads),
                       str(args.dim), str(args.pairs), str(seed), str(pairs_file)]
        exit_code = run_logged(command, result.with_suffix(".log"), args.cap_seconds, env)
        if ([sha(model_a), sha(model_b)] != model_hashes or production_library(exe) != (exe_sha, lib_sha)):
            raise Refusal("model or production build changed during pair run")
        if args.kind == "79":
            for manifest, model, cap, prior_blob, prior_digest in zip(
                    (args.manifest_a, args.manifest_b), (model_a, model_b), (args.cap_a, args.cap_b),
                    blobs, manifest_hashes):
                blob, digest = checked_blob(manifest.resolve(), model, args.width,
                                            [cap, args.layers, args.heads, args.dim])
                if blob != prior_blob or digest != prior_digest:
                    raise Refusal("prompt inputs changed during pair run")
        if receipt_bound and source_bound_receipt(exe, args.kind) != (True, source_manifest_sha):
            raise Refusal("receipt source changed during pair run")
        load_after_sha = None
        if load_record:
            after_load = live_load_state(load_record)
            if (after_load.get("pid") != load_state.get("pid") or
                after_load.get("started_monotonic") != load_state.get("started_monotonic") or
                after_load.get("calibration") != load_state.get("calibration")):
                raise Refusal("load generator changed during pair run")
            load_after_sha = sha(load_record)
        after = box_observation()
        if exit_code is None or exit_code != 0 or not pairs_file.is_file():
            raise Refusal("production pair runner failed or capped; see log")
        values, token_sha = parse_pairs(pairs_file, args.pairs)
        stats = grade(values)
        if hook:
            stats["status"] = "MUTANT_CONSTRUCTION"
        elif getattr(args, "mechanism", False):
            stats["status"] = "MECHANISM_ONLY"
        elif standing:
            stats["status"] = "STANDING"
        record = {"kind": args.kind, "slot": slot, "complete": True, "exe_sha256": exe_sha, "library_sha256": lib_sha,
                  "source_manifest_sha256": source_manifest_sha,
                  "model_hashes": model_hashes, "manifest_hashes": manifest_hashes,
                  "token_sha256": token_sha, "pairs_sha256": sha(pairs_file), "seed": seed,
                  "hook": hook, "load_sha256": load_sha, "load_record_after_sha256": load_after_sha,
                  "load_before": before, "load_after": after, **stats}
        os.replace(pairs_file, result.with_suffix(".pairs.csv"))
        atomic_json(result, record)
        return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--exe", required=True, type=Path)
    prep.add_argument("--model", required=True, type=Path)
    prep.add_argument("--directory", required=True, type=Path)
    prep.add_argument("--width", required=True, type=int)
    for p in (prep,):
        p.add_argument("--cap", required=True, type=int)
        p.add_argument("--layers", required=True, type=int)
        p.add_argument("--heads", required=True, type=int)
        p.add_argument("--dim", required=True, type=int)
    prep.add_argument("--piece-limit", type=int, default=4096)
    prep.add_argument("--cap-seconds", type=int, default=1500)
    pair = sub.add_parser("run")
    pair.add_argument("--exe", required=True, type=Path)
    pair.add_argument("--kind", required=True, choices=("79", "79b"))
    pair.add_argument("--model-a", required=True, type=Path)
    pair.add_argument("--model-b", type=Path)
    pair.add_argument("--manifest-a", type=Path)
    pair.add_argument("--manifest-b", type=Path)
    pair.add_argument("--result", required=True, type=Path)
    pair.add_argument("--cap-a", required=True, type=int)
    pair.add_argument("--cap-b", type=int)
    pair.add_argument("--layers", required=True, type=int)
    pair.add_argument("--heads", required=True, type=int)
    pair.add_argument("--dim", required=True, type=int)
    pair.add_argument("--width", required=True, type=int)
    pair.add_argument("--pairs", type=int, default=508)
    pair.add_argument("--seed", type=int)
    pair.add_argument("--mechanism", action="store_true", help="explicitly allow an ungraded short construction")
    pair.add_argument("--cap-seconds", type=int, default=3600)
    pair.add_argument("--load-record", type=Path, help="the on/off generator record bound into this result")
    args = parser.parse_args(argv)
    try:
        record = prepare(args) if args.mode == "prepare" else run(args)
        print(json.dumps(record, sort_keys=True))
        return 0
    except Refusal as exc:
        print(f"INFRASTRUCTURE: {exc}", file=sys.stderr)
        return 40


if __name__ == "__main__":
    raise SystemExit(main())
