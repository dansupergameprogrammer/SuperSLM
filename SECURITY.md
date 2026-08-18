# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's security advisory
process rather than a public issue:

<https://github.com/dansupergameprogrammer/SuperSLM/security/advisories/new>

This opens a private draft advisory visible only to the maintainer until a
fix is ready. Please do not open a public issue for a suspected
vulnerability.

## What to include

Enough to reproduce: the affected version/commit, the platform, and the
smallest input or call sequence that demonstrates the problem. For a
memory-safety or loader-validation issue (the `.sslm` artifact loader is a
declared trust boundary — see [README.md](README.md#what-ships-in-an-artifact)),
a minimal artifact or byte sequence that triggers it is the most useful
attachment.

## Scope

This covers the SuperSLM Layer 1 runtime and conversion pipeline in this
repository: the loader, the inference engine (CPU and GPU paths), the
tokenizer, and the offline converter. Third-party dependencies (see
[requirements.txt](requirements.txt)) should be reported to their own
projects.

## Response

This is a single-maintainer project. There is no fixed SLA, but reports are
read and acknowledged as soon as practical, and a fix or mitigation is
prioritized once a report is confirmed.
