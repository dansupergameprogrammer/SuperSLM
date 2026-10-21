"""Commission the version-header gate with independent accept/reject inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory

from check_version_header import check


def main() -> None:
    with TemporaryDirectory() as directory:
        cmake = Path(directory) / "CMakeLists.txt"
        header = Path(directory) / "sslm_version.h"

        def project(version: str) -> None:
            cmake.write_text(f"cmake_minimum_required(VERSION 3.20)\n"
                             f"project(superslm VERSION {version} LANGUAGES CXX)\n",
                             encoding="utf-8")

        def macros(major: int, minor: int, patch: int) -> None:
            header.write_text("\n".join((
                f"#define SUPERSLM_VERSION_MAJOR {major}",
                f"#define SUPERSLM_VERSION_MINOR {minor}",
                f"#define SUPERSLM_VERSION_PATCH {patch}",
            )) + "\n", encoding="utf-8")

        def rejects(reason: str) -> None:
            try:
                check(cmake, header)
            except ValueError as error:
                assert reason in str(error), (reason, str(error))
            else:
                raise AssertionError(f"version checker accepted {reason} fixture")

        project("1.9.3")
        macros(1, 9, 3)
        check(cmake, header)
        macros(1, 9, 2)
        rejects("differs from CMake")
        macros(1, 9, 3)
        project("one.nine.three")
        rejects("parseable VERSION")
        project("1.9.3")
        macros(1, 9, 3)
        header.unlink()
        try:
            check(cmake, header)
        except OSError:
            pass
        else:
            raise AssertionError("version checker accepted a missing header")
    print("version checker: matching accepted; one-patch-behind, malformed project, missing header rejected")


if __name__ == "__main__":
    main()
