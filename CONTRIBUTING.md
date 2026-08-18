# Contributing

Issues and pull requests are welcome as patches and bug reports. Please read
this before opening one.

## How this project is maintained

This is a single-maintainer project. Every pull request is reviewed by the
maintainer personally, and none is merged without that review — there is no
auto-merge and no second committer. This is a deliberate security posture,
not a backlog problem: GitHub Actions workflows on this repository run with
elevated trust once an author has ever had a PR merged, so a project that
never merges an outside PR keeps every fork's CI run behind an approval gate
at every subsequent submission, not just the first.

Practically, this means:

- A pull request may sit unmerged even after it is accepted in spirit — the
  maintainer may instead apply the same change by hand (as a patch) so the
  commit lands under their own review rather than as a merge.
- Response time has no fixed SLA.
- A PR that changes generated/golden files, CI workflows, or anything in
  `.github/` gets extra scrutiny, for the reason above.

## Before opening a PR

- Run the local test matrix your change touches — see
  [README.md#building](README.md#building) for the `cmake`+`ctest` command
  and `build.bat` on Windows. `python -m pytest tests/ci` for anything under
  `tests/ci/` or `tools/ci/`.
- Keep the change scoped. A PR that mixes an unrelated refactor with the
  actual fix is harder to review and more likely to be reimplemented by hand
  instead of merged as-is.
- If your change touches a golden/generated file (anything a `gen_*.py`
  script or `tools/unicode_tables_golden.py` produces), regenerate it with
  the script rather than hand-editing it, and say so in the PR description.

## Reporting bugs

Open an issue with: what you expected, what happened, the platform, and
(if it's a determinism or loader issue) the smallest reproducing input you
have. See [SECURITY.md](SECURITY.md) instead if it's a security issue.

## License

By submitting a change, you agree it is licensed under this project's
[Apache 2.0 license](LICENSE).
