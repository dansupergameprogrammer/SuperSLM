"""Print build.bat's superslm_tests compile command as three lines: sources, -D defines, -I includes.
Usage: python3 p41_parse_build_bat.py <build.bat | ->  (multi-row plan cell P4.1, tests/multirow-gpu/p41_cloud.sh)"""
import re
import sys

text = (sys.stdin.read() if sys.argv[1] == "-" else open(sys.argv[1], encoding="utf-8", errors="replace").read())
joined = re.sub(r"\^[ \t]*\r?\n[ \t]*", " ", text)
for line in joined.splitlines():
    s = line.strip()
    if s.startswith("cl ") and "superslm_tests.exe" in s:
        toks = s.split()
        print(" ".join(t.replace("\\", "/") for t in toks if t.endswith(".cpp")))
        print(" ".join("-D" + t[2:] for t in toks if t.startswith("/D")))
        print(" ".join("-I" + t[2:].replace("\\", "/") for t in toks if t.startswith("/I")))
        sys.exit(0)
sys.exit("build.bat: no cl command producing superslm_tests.exe")
