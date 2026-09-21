"""T-2916 (Curie) -- generates one of three MUT_SLM5 variants of `gpu_1p0.cpp` FROM THE LIVE TIP,
each deleting exactly one of TE-370 S3's three unpinned bounds checks in
`sslm_gpu_seq_restoreImpl` (`src/gpu/gpu_1p0.cpp`, around lines 2200-2224):

  INDEXCOUNT   -- `if (static_cast<size_t>(restored_schema_index) >= model->schemas.Count()) { ... }`
  WALKSTATE    -- `if (!resolved_entry || restored_walk_state >= resolved_entry->state_count) { ... }`
  UNBOUNDWALK  -- `else if (restored_walk_state != 0xFFFFFFFFu) { ... }` (the symmetric
                  unbound-blob-carries-a-real-walk malformation check)

Located by an EXACT literal match (tabs and all, extracted verbatim from the live tree at
authoring time), not a hand-approximated regex -- refuses (nonzero exit, no output written)
unless the literal block occurs exactly once in the source, so a source drift is caught here
rather than silently mutating the wrong text or, worse, matching zero times and writing an
UNCHANGED "mutant" that would then wrongly report as a surviving mutant.

Usage: python make_mut_slm5_bounds.py <INDEXCOUNT|WALKSTATE|UNBOUNDWALK> <path-to-gpu_1p0.cpp> <out-mutant.cpp>
"""
import sys

# Exact text, extracted verbatim from src/gpu/gpu_1p0.cpp at authoring time (tabs, not spaces).
_ORIGINAL = (
    "\tif (restored_schema_index >= 0) {\n"
    "\t\tif (static_cast<size_t>(restored_schema_index) >= model->schemas.Count()) {\n"
    "\t\t\tsslm_gpu_seq_release(ctx, fresh);\n"
    "\t\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
    "\t\t}\n"
    "\t\tconst superslm::SchemaEntry* resolved_entry =\n"
    "\t\t    model->schemas.ByIndex(static_cast<size_t>(restored_schema_index));\n"
    "\t\tif (!resolved_entry || restored_walk_state >= resolved_entry->state_count) {\n"
    "\t\t\tsslm_gpu_seq_release(ctx, fresh);\n"
    "\t\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
    "\t\t}\n"
    "\t} else if (restored_walk_state != 0xFFFFFFFFu) {\n"
    "\t\t// Symmetric malformation, mirroring sslm_seq_restore's own identical check: an unbound\n"
    "\t\t// blob (schema index < 0) must carry the unused-walk sentinel, never a real state id.\n"
    "\t\tsslm_gpu_seq_release(ctx, fresh);\n"
    "\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
    "\t}\n"
)

_REPLACEMENTS = {
    "INDEXCOUNT": (
        "\tif (restored_schema_index >= 0) {\n"
        "\t\t// T-2916 MUT_INDEXCOUNT: the index>=schemas.Count() check deleted.\n"
        "\t\tconst superslm::SchemaEntry* resolved_entry =\n"
        "\t\t    model->schemas.ByIndex(static_cast<size_t>(restored_schema_index));\n"
        "\t\tif (!resolved_entry || restored_walk_state >= resolved_entry->state_count) {\n"
        "\t\t\tsslm_gpu_seq_release(ctx, fresh);\n"
        "\t\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
        "\t\t}\n"
        "\t} else if (restored_walk_state != 0xFFFFFFFFu) {\n"
        "\t\tsslm_gpu_seq_release(ctx, fresh);\n"
        "\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
        "\t}\n"
    ),
    "WALKSTATE": (
        "\tif (restored_schema_index >= 0) {\n"
        "\t\tif (static_cast<size_t>(restored_schema_index) >= model->schemas.Count()) {\n"
        "\t\t\tsslm_gpu_seq_release(ctx, fresh);\n"
        "\t\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
        "\t\t}\n"
        "\t\t// T-2916 MUT_WALKSTATE: the walk>=state_count check deleted (resolved_entry is\n"
        "\t\t// still resolved, just never bounds-checked against).\n"
        "\t\tconst superslm::SchemaEntry* resolved_entry =\n"
        "\t\t    model->schemas.ByIndex(static_cast<size_t>(restored_schema_index));\n"
        "\t\t(void)resolved_entry;\n"
        "\t} else if (restored_walk_state != 0xFFFFFFFFu) {\n"
        "\t\tsslm_gpu_seq_release(ctx, fresh);\n"
        "\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
        "\t}\n"
    ),
    "UNBOUNDWALK": (
        "\tif (restored_schema_index >= 0) {\n"
        "\t\tif (static_cast<size_t>(restored_schema_index) >= model->schemas.Count()) {\n"
        "\t\t\tsslm_gpu_seq_release(ctx, fresh);\n"
        "\t\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
        "\t\t}\n"
        "\t\tconst superslm::SchemaEntry* resolved_entry =\n"
        "\t\t    model->schemas.ByIndex(static_cast<size_t>(restored_schema_index));\n"
        "\t\tif (!resolved_entry || restored_walk_state >= resolved_entry->state_count) {\n"
        "\t\t\tsslm_gpu_seq_release(ctx, fresh);\n"
        "\t\t\treturn SSLM_SEQUENCE_KV_BUFFER_MISMATCH;\n"
        "\t\t}\n"
        "\t}\n"
        "\t// T-2916 MUT_UNBOUNDWALK: the symmetric unbound-carries-a-real-walk check deleted.\n"
    ),
}


def main(argv: list[str]) -> None:
    if len(argv) != 4 or argv[1] not in _REPLACEMENTS:
        sys.exit("usage: make_mut_slm5_bounds.py <INDEXCOUNT|WALKSTATE|UNBOUNDWALK> "
                  "<path-to-gpu_1p0.cpp> <out-mutant.cpp>")
    variant, src, dst = argv[1], argv[2], argv[3]
    with open(src, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    # T-2916 (TE-370 M1): normalize CRLF -> LF before matching -- see
    # make_mut_gpu_checkedreturn_seam.py's identical comment for why (a fresh checkout of
    # gpu_1p0.cpp is CRLF; the worktree these anchors were authored against happened to be LF).
    text = text.replace("\r\n", "\n")
    count = text.count(_ORIGINAL)
    if count != 1:
        sys.exit(f"expected exactly 1 occurrence of the S3 bounds-check block, found {count} -- "
                  "source may have moved, refusing to guess")
    out = text.replace(_ORIGINAL, _REPLACEMENTS[variant], 1)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"wrote {dst} (variant={variant})")


if __name__ == "__main__":
    main(sys.argv)
