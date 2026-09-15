#!/usr/bin/env bash
# Conductor-only proof harness for T2701.  Do not run this as a normal unit test:
# it intentionally performs four full Windows builds in disposable worktrees.
set -euo pipefail

repo="${1:-$(git rev-parse --show-toplevel)}"
repo="$(cd "$repo" && pwd)"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/t2701-buildbat-exit-proof.XXXXXX")"

to_windows_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        (cd "$1" && pwd -W)
    fi
}

replace_once() {
    local path="$1"
    local old="$2"
    local new="$3"
    python - "$path" "$old" "$new" <<'PY'
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
PY
}

insert_error() {
    local position="$1"
    local path="$2"
    local message="$3"
    python - "$position" "$path" "$message" <<'PY'
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
PY
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
    insert_error append "$1/tools/t2139_dim9_current_token_pin.cpp" \
        'T2701 forced late per-tool build failure'
}

run_case() {
    local name="$1"
    local expected="$2"
    local mutator="$3"
    local worktree="$scratch/$name"
    local batch status

    git -C "$repo" worktree add --detach "$worktree" HEAD >/dev/null
    if [[ "$mutator" != "none" ]]; then
        "$mutator" "$worktree"
    fi
    batch="$(to_windows_path "$worktree")\\build.bat"
    set +e
    cmd.exe /d /c "\"$batch\""
    status=$?
    set -e
    git -C "$repo" worktree remove --force "$worktree" >/dev/null

    if [[ "$status" == "$expected" ]]; then
        printf 'PASS %-30s expected=%s actual=%s\n' "$name" "$expected" "$status"
    else
        printf 'FAIL %-30s expected=%s actual=%s\n' "$name" "$expected" "$status" >&2
        return 1
    fi
}

cleanup() {
    # Every worktree has already been removed by run_case; scratch is mktemp-owned.
    rmdir "$scratch" 2>/dev/null || true
}
trap cleanup EXIT

run_case s8_fixture_flag_reverted 1 mutate_s8_fixture_flag
run_case early_broken_shader 1 mutate_early_shader
run_case late_per_tool_build 1 mutate_late_per_tool_build
run_case clean_tree 0 none
