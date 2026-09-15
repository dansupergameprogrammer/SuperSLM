#!/usr/bin/env bash
# Conductor-only proof harness for T2701. Do not run normal mode as a unit test:
# it intentionally performs four full Windows builds in disposable worktrees.
set -euo pipefail

self_test=false
if [[ "${1:-}" == "--self-test" ]]; then
    self_test=true
    shift
fi
repo="${1:-$(git rev-parse --show-toplevel)}"
repo="$(cd "$repo" && pwd)"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/t2701-buildbat-exit-proof.XXXXXX")"
PYTHON_CMD=()

to_windows_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        (cd "$1" && pwd -W)
    fi
}

resolve_python() {
    local candidate resolved
    if [[ -n "${PYTHON:-}" ]]; then
        candidate="$PYTHON"
        resolved="$(command -v "$candidate" 2>/dev/null || true)"
        [[ -n "$resolved" && "$resolved" != *WindowsApps* ]] || {
            echo "FAIL: PYTHON is not a real interpreter: $PYTHON" >&2; return 1; }
        PYTHON_CMD=("$resolved")
    else
        resolved="$(command -v py 2>/dev/null || true)"
        if [[ -n "$resolved" && "$resolved" != *WindowsApps* ]] &&
           "$resolved" -3 -c "import sys" </dev/null >/dev/null 2>&1; then
            PYTHON_CMD=("$resolved" -3)
        else
            for candidate in python3 python; do
                resolved="$(command -v "$candidate" 2>/dev/null || true)"
                if [[ -n "$resolved" && "$resolved" != *WindowsApps* ]] &&
                   "$resolved" -c "import sys" </dev/null >/dev/null 2>&1; then
                    PYTHON_CMD=("$resolved")
                    break
                fi
            done
        fi
    fi
    [[ ${#PYTHON_CMD[@]} -gt 0 ]] || { echo "FAIL: no real Python 3 interpreter" >&2; return 1; }
    "${PYTHON_CMD[@]}" -c "import sys; assert sys.version_info >= (3, 0)" </dev/null
}

replace_once() {
    local path="$1"
    local old="$2"
    local new="$3"
    "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import sys

path, old, new = sys.argv[1:]
data = Path(path).read_bytes()
for encoding in ("\n", "\r\n"):
    before = old.replace("\n", encoding).encode()
    after = new.replace("\n", encoding).encode()
    if data.count(before) == 1:
        Path(path).write_bytes(data.replace(before, after))
        break
else:
    raise SystemExit(f"expected exactly one matching mutation site in {path}")
' "$path" "$old" "$new"
}

insert_error() {
    local position="$1"
    local path="$2"
    local message="$3"
    "${PYTHON_CMD[@]}" -c '
from pathlib import Path
import sys

position, path, message = sys.argv[1:]
data = Path(path).read_bytes()
newline = b"\r\n" if b"\r\n" in data else b"\n"
injected = b"#error " + message.encode("ascii") + newline
if position == "prepend":
    Path(path).write_bytes(injected + data)
elif position == "append":
    Path(path).write_bytes(data + newline + injected)
else:
    raise SystemExit(f"unknown insertion position: {position}")
' "$position" "$path" "$message"
}

mutate_s8_fixture_flag() {
    replace_once "$1/tools/_t2199_s8_synthetic_full_model_fixture.py" \
        'sections, flags=C.artifact_flags_for_model(model) | F.DAMPED_GREEDY_CONSTANTS_FLAG' \
        'sections, flags=F.DAMPED_GREEDY_CONSTANTS_FLAG'
}

mutate_early_shader() {
    insert_error prepend "$1/src/gpu/shaders/attention_score_site.hlsl" \
        'T2701 forced early shader failure'
}

mutate_late_per_tool_build() {
    # build.bat:1320 is the final direct per-tool compiler hard stop before suite/python gates.
    insert_error append "$1/tools/t2132_m4_forced_token_count_pin.cpp" \
        'T2701 forced late per-tool build failure'
}

run_case() {
    local name="$1"
    local expected="$2"
    local mutator="$3"
    local batch_override="${4:-}"
    local worktree="" batch status remove_ec=0

    if [[ -n "$batch_override" ]]; then
        batch="$batch_override"
    else
        worktree="$scratch/$name"
        if ! git -C "$repo" worktree add --detach "$worktree" HEAD >/dev/null; then
            printf 'FAIL %-30s could not create worktree\n' "$name" >&2
            return 1
        fi
        if [[ "$mutator" != "none" ]] && ! "$mutator" "$worktree"; then
            git -C "$repo" worktree remove --force "$worktree" >/dev/null || true
            printf 'FAIL %-30s could not apply mutation\n' "$name" >&2
            return 1
        fi
        batch="$(to_windows_path "$worktree")\\build.bat"
    fi

    # Git Bash rewrites bare /d and /c for native cmd.exe; doubled slashes preserve cmd switches.
    set +e
    cmd.exe //d //c "$batch"
    status=$?
    set -e
    if [[ -n "$worktree" ]]; then
        git -C "$repo" worktree remove --force "$worktree" >/dev/null || remove_ec=$?
    fi

    if [[ "$remove_ec" -ne 0 ]]; then
        printf 'FAIL %-30s worktree cleanup exit=%s\n' "$name" "$remove_ec" >&2
        return 1
    fi
    if [[ "$status" == "$expected" ]]; then
        printf 'PASS %-30s expected=%s actual=%s\n' "$name" "$expected" "$status"
        return 0
    fi
    printf 'FAIL %-30s expected=%s actual=%s\n' "$name" "$expected" "$status" >&2
    return 1
}

run_self_test() {
    local stub_dir="$scratch/self test" overall=0
    mkdir -p "$stub_dir"
    printf '@echo off\r\nexit /b 3\r\n' > "$stub_dir/exit_three.bat"
    printf '@echo off\r\nexit /b 0\r\n' > "$stub_dir/exit_zero.bat"
    printf '@echo off\r\nif 1==1 (\r\n  exit /b 1\r\n)\r\nexit /b 9\r\n' > "$stub_dir/nested_exit.bat"
    if ! run_case self_exit_three_space_path 3 none "$(to_windows_path "$stub_dir")\\exit_three.bat"; then overall=1; fi
    if ! run_case self_exit_zero_space_path 0 none "$(to_windows_path "$stub_dir")\\exit_zero.bat"; then overall=1; fi
    if ! run_case self_nested_exit_path 1 none "$(to_windows_path "$stub_dir")\\nested_exit.bat"; then overall=1; fi
    rm -f "$stub_dir/exit_three.bat" "$stub_dir/exit_zero.bat" "$stub_dir/nested_exit.bat"
    rmdir "$stub_dir"
    return "$overall"
}

cleanup() {
    rmdir "$scratch" 2>/dev/null || true
}
trap cleanup EXIT

if "$self_test"; then
    run_self_test
    exit $?
fi

if ! resolve_python; then exit 1; fi
overall=0
if ! run_case s8_fixture_flag_reverted 1 mutate_s8_fixture_flag; then overall=1; fi
if ! run_case early_broken_shader 1 mutate_early_shader; then overall=1; fi
if ! run_case late_per_tool_build 1 mutate_late_per_tool_build; then overall=1; fi
if ! run_case clean_tree 0 none; then overall=1; fi
exit "$overall"
