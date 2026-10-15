"""C-4: the direct 7.4 lock (Lock74 in tests/paged-kv/c6_dim7_reset_adopt_timing.cpp) under the box timing lock's one
format and protocol (D-SLM8464; tools/run_paged_kv_rows.py's comment above _LOCK_BYTE), as the other holders take it:

  live     a live pid written after the last boot refuses, and the file keeps that pid
  dead     a dead pid is recovered; the holder's release at exit removes the file
  empty    an empty lock file is stale: it is acquired (never refused as incomplete)
  torn     content other than ASCII digits naming a pid from 1 to 2**31 - 1 (space, tab, CR and LF stripped), or a
           file longer than 4096 bytes, is torn and refused, and the file is left as it was; '<dead pid>\\r\\n' and a
           dead pid with leading zeros still read as the pid
  boot     (D-SLM8466 (3)) a lock file modified before the last boot is stale whatever it holds: a live pid (the
           reused-pid case: a pid recorded before the reboot may name an unrelated process now), torn content, a
           4097-byte file. The same live pid written after the boot still refuses
  oslock   another process holding the OS byte (offset 1 MiB) over a dead pid: Lock74 refuses, so a recovery in progress
           in any holder excludes it

Environment (fails, never skips, without it): PKV_C6_BUILD, the build directory holding superslm_pkv_c6 (.exe on
Windows). The geometry cell also needs PKV_C6_FIXTURES, the generated fixture directory (tools/gen_paged_kv_fixture.py).
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


BUILD_DIR = os.environ.get("PKV_C6_BUILD", "")
FIXTURES_DIR = os.environ.get("PKV_C6_FIXTURES", "")
EXE_NAME = "superslm_pkv_c6.exe" if os.name == "nt" else "superslm_pkv_c6"


def last_boot():
    """The last boot in epoch seconds, this file's own reading: now less the uptime."""
    if os.name == "nt":
        import ctypes
        tick = ctypes.windll.kernel32.GetTickCount64
        tick.restype = ctypes.c_uint64
        return time.time() - tick() / 1000.0
    return time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)


class Lock74Cells(unittest.TestCase):
    def setUp(self):
        if not BUILD_DIR:
            self.fail("PKV_C6_BUILD is unset: name the build directory holding superslm_pkv_c6; there is no default")
        self.exe = Path(BUILD_DIR) / EXE_NAME
        if not self.exe.is_file():
            self.fail(f"PKV_C6_BUILD names {BUILD_DIR}, which holds no {EXE_NAME}")
        self.folder = tempfile.TemporaryDirectory(prefix="pkv-74-lock-")
        self.addCleanup(self.folder.cleanup)
        self.lock = Path(self.folder.name) / "timing.lock"
        self.procs = []
        self.addCleanup(self.stop_procs)

    def stop_procs(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.stdin.close()
                proc.wait(timeout=10)

    def run_lock(self):
        env = os.environ.copy()
        env["PKV_DEBUNK_74_TEST_LOCK_PATH"] = str(self.lock)
        return subprocess.run([str(self.exe), "=7.4t/C6:lock"], env=env, capture_output=True, text=True, timeout=15)

    def live_pid(self):
        proc = subprocess.Popen([sys.executable, "-I", "-c", "import sys; sys.stdin.readline()"],
                                stdin=subprocess.PIPE, text=True)
        self.procs.append(proc)
        return proc.pid

    @staticmethod
    def dead_pid():
        proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
        proc.wait(timeout=10)
        return proc.pid

    def write_lock(self, content, before_boot=False):
        self.lock.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        if before_boot:
            when = last_boot() - 3600
            os.utime(self.lock, (when, when))

    def assert_held(self, name):
        result = self.run_lock()
        self.assertEqual(result.returncode, 0, f"{name}: {result.stdout}{result.stderr}")
        self.assertFalse(self.lock.exists(), f"{name}: the holder did not release the lock at exit")

    def assert_refused(self, name, content, message):
        result = self.run_lock()
        self.assertEqual(result.returncode, 77, f"{name}: {result.stdout}{result.stderr}")
        self.assertIn(message, result.stderr, name)
        self.assertEqual(self.lock.read_bytes(), content, f"{name}: a refusal changed the lock file")

    # live
    def test_live_pid_refuses(self):
        for name, pid in (("this test's pid", os.getpid()), ("another live pid", self.live_pid())):
            with self.subTest(name):
                self.write_lock(str(pid))
                self.assert_refused(name, str(pid).encode("ascii"), "already held")

    # dead, and a fresh acquire with no file present
    def test_dead_pid_recovered_and_released(self):
        self.assert_held("no lock file")
        for name, content in (("dead pid", "{pid}"), ("dead pid, CRLF", "{pid}\r\n"), ("dead pid, leading zeros",
                                                                                       "000{pid}")):
            with self.subTest(name):
                self.write_lock(content.format(pid=self.dead_pid()))
                self.assert_held(name)

    # empty
    def test_empty_lock_is_stale(self):
        for name, content in (("empty", ""), ("whitespace only", " \r\n")):
            with self.subTest(name):
                self.write_lock(content)
                self.assert_held(name)

    # torn
    def test_torn_content_refused(self):
        dead = self.dead_pid()
        for name, content in (("json", '{"pid": 1'), ("sign", f"+{dead}"), ("zero", "0"),
                              ("past the largest pid", "2147483648"), ("vertical tab", f"{dead}\x0b"),
                              ("fullwidth digits", "１２"), ("4097 bytes", f"{dead}".ljust(4097))):
            with self.subTest(name):
                self.write_lock(content)
                self.assert_refused(name, content.encode("utf-8"), "torn")

    # boot: stale before the last boot whatever it holds; the same live pid after the boot refuses
    def test_lock_modified_before_the_last_boot_is_stale(self):
        live = self.live_pid()
        self.write_lock(str(live))
        self.assert_refused("live pid, after the boot", str(live).encode("ascii"), "already held")
        for name, content in (("live pid (reused), before the boot", str(live)), ("torn, before the boot", '{"pid": 1'),
                              ("4097 bytes, before the boot", f"{live}".ljust(4097))):
            with self.subTest(name):
                self.write_lock(content, before_boot=True)
                self.assert_held(name)

    # oslock: another process mid-recovery holds the OS byte; Lock74 refuses rather than racing it
    def test_os_byte_held_refuses(self):
        content = str(self.dead_pid()).encode("ascii")
        self.write_lock(content)
        holder = subprocess.Popen([sys.executable, "-I", "-c", (
            "import os, sys\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | getattr(os, 'O_BINARY', 0))\n"
            "os.lseek(fd, 1 << 20, os.SEEK_SET)\n"
            "if os.name == 'nt':\n"
            "    import msvcrt; msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)\n"
            "else:\n"
            "    import fcntl; fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 1 << 20)\n"
            "print('LOCKED', flush=True)\n"
            "sys.stdin.readline()\n"), str(self.lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.procs.append(holder)
        self.addCleanup(holder.stdout.close)
        self.assertEqual(holder.stdout.readline().strip(), "LOCKED")
        self.assert_refused("OS byte held", content, "being recovered by another runner")
        holder.stdin.close()
        holder.wait(timeout=10)
        self.assert_held("OS byte released")


class GeometryCell(unittest.TestCase):
    def test_geometry_refused_separately_from_the_lock(self):
        if not BUILD_DIR or not FIXTURES_DIR:
            self.fail("PKV_C6_BUILD and PKV_C6_FIXTURES must both be set; there is no default")
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_FIXTURE_DIR"] = FIXTURES_DIR
        env["SUPERSLM_PAGED_KV_74_FIXTURE_LARGE"] = "pkv_qk"
        geometry = subprocess.run([str(Path(BUILD_DIR) / EXE_NAME), "=7.4/C6:fixtures"], env=env,
                                  capture_output=True, text=True, timeout=15)
        self.assertEqual(geometry.returncode, 71, geometry.stdout + geometry.stderr)
        self.assertIn("geometry mismatch", geometry.stderr)


if __name__ == "__main__":
    unittest.main()
