"""Strict must-reject runner for the schema-hash audit (T-3030 commissioning).

Runs the audit on each artifact separately. Exits 1 (REJECTED) only when EVERY run exited
non-zero AND printed a JSON record with status "FAIL" carrying a duplicate or zero finding.
Anything else -- a PASS, an ERROR, a crash, bad JSON -- exits 0, which the registry reads as
ACCEPTED and writes the instrument DEAD. A crash therefore cannot pass as a refusal.

Usage: python must_reject_wrapper.py <audit.py> <artifact.sslm> [...]
"""
import json
import subprocess
import sys


def main():
    audit, artifacts = sys.argv[1], sys.argv[2:]
    all_rejected = bool(artifacts)
    for path in artifacts:
        run = subprocess.run([sys.executable, audit, path], capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        try:
            rec = json.loads(run.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            rec = {}
        found = bool(rec.get("duplicate_hashes")) or bool(rec.get("zero_hashes"))
        rejected = run.returncode != 0 and rec.get("status") == "FAIL" and found
        print(f"wrapper: {path} exit={run.returncode} status={rec.get('status')} "
              f"finding={found} rejected={rejected}")
        all_rejected &= rejected
    print("wrapper verdict:", "REJECTED" if all_rejected else "ACCEPTED")
    return 1 if all_rejected else 0


if __name__ == "__main__":
    sys.exit(main())
