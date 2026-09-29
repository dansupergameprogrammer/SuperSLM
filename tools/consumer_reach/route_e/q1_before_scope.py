#!/usr/bin/env python3
"""q1_before_scope.py -- Q1's "before" diff scope (plan rev 9; coverage-mutants-rev8 Qd, Qe; the conductor's rev-9 item 4).

Q1's "before" is the shipped v1.8.0 base with the build-configuration record constant applied, and nothing else.
Rev 8.1 printed the diff and graded nothing, and took the base commit as an operator argument. This grades it:
  * the shipped commit is READ from the engine repository's own v1.8.0 tag, not given;
  * `git diff <shipped> <before>` changes exactly one file, src/matmul.cpp, and deletes no line;
  * the lines it adds, blank lines at either edge aside, are exactly the candidate's record block: from the line
    "// ---- Build-configuration record" through the line that closes the constant ("}";), in the candidate's
    src/matmul.cpp at its tag.
usage: q1_before_scope.py <shipped repo> <before clone> <before ref> <candidate clone> <candidate ref>
       -> prints the verdict; exit 0 in scope, 1 not.
"""
import subprocess, sys

def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout

def record_block(text):
    lines = text.splitlines()
    s = [i for i, l in enumerate(lines) if l.startswith("// ---- Build-configuration record")]
    if len(s) != 1:
        return None
    e = [i for i in range(s[0], len(lines)) if lines[i].rstrip().endswith('"}";')]
    return lines[s[0]:e[0] + 1] if e else None

def strip_edges(ls):
    while ls and not ls[0].strip(): ls = ls[1:]
    while ls and not ls[-1].strip(): ls = ls[:-1]
    return ls

def main():
    shipped_repo, before, bref, cand, cref = sys.argv[1:6]
    ship = git(shipped_repo, "rev-parse", "v1.8.0^{commit}").strip()
    bef = git(before, "rev-parse", f"{bref}^{{commit}}").strip()
    try:
        git(before, "cat-file", "-e", ship)
    except subprocess.CalledProcessError:
        git(before, "fetch", "-q", shipped_repo, "refs/tags/v1.8.0:refs/q1-shipped/v1.8.0")
    print(f"  shipped base (read from the engine repository's v1.8.0 tag): {ship}")
    print(f"  before: {bef}")
    num = [l.split("\t") for l in git(before, "diff", "--numstat", ship, bef).splitlines()]
    files = [f for _, _, f in num]
    bad = []
    if files != ["src/matmul.cpp"]:
        bad.append(f"the diff touches {len(files)} file(s): {', '.join(files[:6])}{' ...' if len(files) > 6 else ''}")
    if any(d != "0" for _, d, _ in num):
        bad.append("the diff deletes lines: " + ", ".join(f"{f} -{d}" for _, d, f in num if d != "0"))
    if "src/matmul.cpp" in files:
        body = git(before, "diff", "-U0", ship, bef, "--", "src/matmul.cpp").splitlines()
        added = strip_edges([l[1:] for l in body if l.startswith("+") and not l.startswith("+++")])
        blk = record_block(git(cand, "show", f"{cref}:src/matmul.cpp"))
        if blk is None:
            bad.append("no record block in the candidate's src/matmul.cpp")
        elif added != blk:
            extra = len(added) - len(blk)
            bad.append(f"the lines added to src/matmul.cpp are not the candidate's record block ({len(added)} lines added, block {len(blk)}; "
                       f"{'extra' if extra > 0 else 'missing'} {abs(extra)})" if extra else
                       "the lines added to src/matmul.cpp differ from the candidate's record block")
    if bad:
        for b in bad:
            print(f"  OUT OF SCOPE: {b}")
        return 1
    print("  in scope: one file, src/matmul.cpp, no deletions, the added lines are exactly the candidate's record block")
    return 0

if __name__ == "__main__":
    sys.exit(main())
