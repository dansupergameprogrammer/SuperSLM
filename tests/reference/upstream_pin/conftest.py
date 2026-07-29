"""Register this directory's own marker.

This repository carries no pytest config, and adding one at the root would change
collection for every other suite in the tree. Registering `slow` here keeps the scope to
the upstream-pin fixture and removes the PytestUnknownMarkWarning that would otherwise
appear on every run of it.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: drives the upstream 1.5B model; the recorded pre-tag step, not per-commit CI",
    )
