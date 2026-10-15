"""P-4: the no-control 7.4 floor consumes measured corrected medians."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/paged-kv/c6_dim7_reset_adopt_timing.cpp"
TOOLCHAIN = ROOT / "tests/t2130-g5-red-suite/resolve_toolchain.ps1"


def box_free() -> None:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "if (Get-Process -Name smoke_adapter_driver -ErrorAction SilentlyContinue) { exit 2 }"],
        capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError("smoke_adapter_driver.exe owns the box")


def floor_source() -> tuple[str, str]:
    text = SOURCE.read_text(encoding="utf-8")
    median = re.search(r"^double Median\(std::vector<double> v\) \{[^\n]+\}", text, re.M)
    if not median:
        raise AssertionError("7.4 Median source not found")
    start = text.index('const char* control=std::getenv("SUPERSLM_PAGED_KV_74_FLOOR_CONTROL_TICKS")')
    end = text.index("for(auto& entry:", start)
    branch = text[start:end].strip()
    refusal = 'Refuse(74,"corrected median under 100 ticks");'
    if branch.count(refusal) != 1:
        raise AssertionError("7.4 measured floor refusal not found")
    return median.group(), branch.replace(refusal, "return 74;")


def probe_source(median: str, branch: str) -> str:
    return r'''
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <intrin.h>
#include <vector>
#pragma intrinsic(__rdtsc)
''' + median + r'''
__declspec(noinline) double Fast() {
    _ReadWriteBarrier(); auto a=__rdtsc(); _ReadWriteBarrier(); auto b=__rdtsc();
    return double(b-a);
}
__declspec(noinline) double Slow() {
    volatile unsigned long long sum=0;
    _ReadWriteBarrier(); auto a=__rdtsc();
    for(int i=0;i<1000;++i) sum+=static_cast<unsigned long long>(i);
    auto b=__rdtsc(); _ReadWriteBarrier();
    return double(b-a);
}
int main() {
    std::vector<double> empty, aa, ab, ra, rb;
    for(int i=0;i<205;++i) empty.push_back(Fast());
    const double overhead=Median(empty);
    for(int i=0;i<205;++i) {
        aa.push_back(Fast()-overhead);
        ab.push_back(Slow()-overhead);
        ra.push_back(Slow()-overhead);
        rb.push_back(Slow()-overhead);
    }
    std::printf("measured floor medians: aa=%.0f ab=%.0f ra=%.0f rb=%.0f\n",
                Median(aa),Median(ab),Median(ra),Median(rb));
    if(Median(aa)>=100||Median(ab)<100||Median(ra)<100||Median(rb)<100) return 71;
    ''' + branch + r'''
    return 0;
}
'''


class MeasuredFloor(unittest.TestCase):
    def test_p4_measured_median_refuses_without_control_hook(self):
        median, branch = floor_source()
        original = 'control?std::strtod(control,nullptr):Median(aa)'
        mutant = 'control?std::strtod(control,nullptr):1e9'
        self.assertEqual(branch.count(original), 1)
        with tempfile.TemporaryDirectory(prefix="pkv-74-floor-") as folder:
            temp = Path(folder)
            paths = {}
            for name, body in (("mutant", branch.replace(original, mutant)),
                               ("current", branch)):
                source = temp / f"{name}.cpp"
                exe = temp / f"{name}.exe"
                source.write_text(probe_source(median, body), encoding="utf-8")
                box_free()
                command = (f". '{TOOLCHAIN}'; Enter-SuperSlmVsDevShell | Out-Null; "
                           f"& cl /nologo /O2 /EHsc /std:c++20 '/Fe:{exe}' '{source}'; "
                           "exit $LASTEXITCODE")
                build = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                       capture_output=True, text=True, timeout=120)
                self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
                paths[name] = exe
            env = os.environ.copy()
            env.pop("SUPERSLM_PAGED_KV_74_FLOOR_CONTROL_TICKS", None)
            env.pop("SUPERSLM_PAGED_KV_74_FORCE_FLOOR", None)
            for name, expected in (("mutant", 0), ("current", 74)):
                box_free()
                run = subprocess.run([str(paths[name])], env=env, capture_output=True,
                                     text=True, timeout=30)
                print(f"P-4 {name}: exit={run.returncode} {run.stdout.strip()}", flush=True)
                self.assertEqual(run.returncode, expected, run.stdout + run.stderr)
                self.assertIn("measured floor medians:", run.stdout)


if __name__ == "__main__":
    unittest.main()
