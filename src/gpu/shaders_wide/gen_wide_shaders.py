"""Generate the row-widened variant of every non-GEMM per-layer site shader (multi-row plan §3.1).

Each generated shader is the one-row shader in ../shaders with exactly four mechanical edits, so
its arithmetic is the one-row arithmetic, byte for byte in the source:

1. The three per-row buffers (SeqState u0, LayerScratch u1, WorkScratch u3) become arrays of
   SSLM_WIDE_H raw views, one per row (register spaces 1, 2, 3). KvCache (u2), the page table
   (t10) and the page constants (b1, both declared in site_common.hlsli) stay single bindings
   shared by every row: each row writes only its own position and reads positions 0..its own.
2. A thread-private `g_row` is declared, and `SeqState`/`LayerScratch`/`WorkScratch` become macros
   for the row's own view, so every existing Load/Store and every helper-call argument (the fused
   adapter tail's included) addresses the row's own window at the same offsets as the one-row
   shader.
3. `g_position` and `g_width` become the row's own values: the host passes row 0's position and
   causal width, and row r is r positions later, `(g_position + g_row)` and `(g_width + g_row)`.
   So row r's K/V address is `KvRowAddrGpu(..., position + r)` through the one page table, and its
   attention width is `width + r`; the host's widest row only sizes the dispatch.
4. `main` takes SV_GroupID (added when absent) and sets `g_row = gid.y` first. The host dispatches
   (groups_x, SSLM_WIDE_H, 1), so group row y is sequence row y, and SV_GroupID.x and
   SV_DispatchThreadID.x keep their one-row values.

Padded rows (a chunk of M rows on the next variant up) carry a nonzero sticky word, and every site
returns on it before anything else it does. The generator refuses a site whose `main` reads the
page table (t10, b1) before its sticky return, because t10 has no bounds check and a padded row's
position can lie past the real rows (plan §3.1, cell 4.6b).

The six projection GEMMs are not generated: proj_gemm_wide.hlsl is one hand-written source for
all six, because sharing the weight stream needs one weight load serving every row.

Usage:
  python gen_wide_shaders.py             write ./<site>_wide.hlsl from ../shaders/<site>.hlsl
  python gen_wide_shaders.py --check     regenerate in memory; exit 1 if a committed file differs
  python gen_wide_shaders.py --out DIR   write the generated files into DIR instead
Output is deterministic (no timestamps, LF line endings) and is never edited by hand: it is
regenerated from the base's own shaders at every rebase.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "shaders"

# Every non-GEMM site the one-row body records, in recording order (superslm_gpu.cpp,
# RecordOneTokenFullDepthDispatchBody). qk_norm_site runs only for a model with QK norm.
SITES = [
    "attn_norm_site", "q_proj_site", "kv_proj_site", "qk_norm_site", "rope_guard_site",
    "rope_commit_site", "attention_score_site", "softmax_site", "context_accumulate_site",
    "ctx_fold_site", "o_proj_site", "attn_residual_site", "mlp_norm_site", "gate_proj_site",
    "up_proj_site", "mlp_act_site", "down_proj_site", "mlp_residual_site", "commit_site",
]

SPACE = {"SeqState": 1, "LayerScratch": 2, "WorkScratch": 3}
DECL = re.compile(r"RWByteAddressBuffer\s+(SeqState|LayerScratch|WorkScratch)\s*:\s*register\(u[013]\)\s*;")
MAIN = re.compile(r"void\s+main\s*\(([^)]*)\)")
# The page-table helpers and bindings site_common.hlsli declares (paged-KV plan §4 item 1).
PAGE_USE = re.compile(r"\b(KvRowAddrGpu|KvPhysPageBaseGpu|KvPageTable|g_page_positions|g_page_bytes)\b")
STICKY_RETURN = re.compile(r"!=\s*kTagOk\s*\)\s*return\b")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _check_sticky_first(name: str, text: str) -> None:
    """Refuse a site whose main touches the page table before returning on a set sticky word."""
    code = _strip_comments(text)
    m = MAIN.search(code)
    if not m:
        raise SystemExit(f"{name}: no main")
    body = code[m.end():]
    sticky = STICKY_RETURN.search(body)
    page = PAGE_USE.search(body)
    if sticky is None:
        raise SystemExit(f"{name}: main has no sticky-word return; a padded row would not be inert")
    if page is not None and page.start() < sticky.start():
        raise SystemExit(f"{name}: main reads the page table before its sticky-word return; a padded "
                         f"row past the real rows would read t10 out of bounds")


def widen(name: str, text: str) -> str:
    _check_sticky_first(name, text)
    found = DECL.findall(text)
    if sorted(found) != sorted(SPACE):
        raise SystemExit(f"{name}: expected one declaration each of {sorted(SPACE)}, found {found}")
    text = DECL.sub(
        lambda m: f"RWByteAddressBuffer {m.group(1)}Rows[SSLM_WIDE_H] : register(u0, space{SPACE[m.group(1)]});",
        text)

    # The row machinery goes after the line that carries the last of the three declarations.
    lines = text.split("\n")
    last = max(i for i, ln in enumerate(lines) if re.search(r"(SeqState|LayerScratch|WorkScratch)Rows\[", ln))
    block = [
        "// Row machinery (generated, gen_wide_shaders.py): group row y is sequence row y.",
        "static uint g_row;",
        "#define SeqState SeqStateRows[g_row]",
        "#define LayerScratch LayerScratchRows[g_row]",
        "#define WorkScratch WorkScratchRows[g_row]",
        "#define g_position (g_position + g_row)",
        "#define g_width (g_width + g_row)",
    ]
    lines[last + 1:last + 1] = block
    text = "\n".join(lines)

    m = MAIN.search(text)
    params = m.group(1)
    gid = re.search(r"uint3\s+(\w+)\s*:\s*SV_GroupID\b", params)
    if gid:
        gid_name = gid.group(1)
        new_params = params
    else:
        gid_name = "sslm_wide_gid"
        new_params = params + ", uint3 sslm_wide_gid : SV_GroupID"
    text = text[:m.start(1)] + new_params + text[m.end(1):]
    brace = text.index("{", m.start())
    text = text[:brace + 1] + f"\n    g_row = {gid_name}.y;  // dispatch row y is sequence row y" + text[brace + 1:]

    header = (f"// GENERATED by gen_wide_shaders.py from ../shaders/{name}.hlsl -- do not edit by hand.\n"
              f"// The row-widened variant (multi-row plan §3.1); compile with -D SSLM_WIDE_H=<rows> -I ../shaders.\n")
    return header + text


def generate() -> dict:
    out = {}
    for name in SITES:
        src = (SRC / f"{name}.hlsl").read_text(encoding="utf-8")
        out[f"{name}_wide.hlsl"] = widen(name, src)
    return out


def main(argv) -> int:
    args = argv[1:]
    check = False
    out_dir = HERE
    while args:
        a = args.pop(0)
        if a == "--check":
            check = True
        elif a == "--out" and args:
            out_dir = Path(args.pop(0))
        else:
            print(f"unknown argument: {a}", file=sys.stderr)
            return 2
    files = generate()
    if check:
        stale = []
        for fname, text in files.items():
            path = out_dir / fname
            if not path.exists() or path.read_bytes() != text.encode("utf-8"):
                stale.append(fname)
        if stale:
            print("stale generated shaders (run gen_wide_shaders.py and commit): " + ", ".join(stale),
                  file=sys.stderr)
            return 1
        print(f"generated shaders current: {len(files)} files")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (out_dir / fname).write_bytes(text.encode("utf-8"))
    print(f"generated {len(files)} widened site shaders")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
