#!/usr/bin/env python3
"""Schema fast-decode plan rev 1.4.4 §12: the engine-side GPU source mutants of slices G1-F, G1-M and G1-J.

Each mutant is a list of exact (file, anchor, replacement) edits. Every anchor must occur exactly once in its file,
or the script refuses: a drifted anchor never mutates the wrong place. The mutated files are written under OUT with
their repository-relative paths, so a mutant build overlays them on a copy of the tree.

  make_sfdg_mutants.py --check [--repo=DIR]          every anchor of every mutant is unique (the cloud check)
  make_sfdg_mutants.py XGn [--repo=DIR] --out=DIR    write mutant XGn's files under DIR
  make_sfdg_mutants.py --list                        the mutants, their files and the cells that kill them

Killing runs are box-only (the cells need a D3D12 device), except XG22, which G0.1's host half can kill in the
cloud. The deferred mutants (G1-X's, rev 1.4.4) and the plugin's (P2) are not here.
"""
import argparse
import os
import sys

GPU = "src/gpu/gpu_1p0.cpp"
DBG = "src/gpu/debug_layer_mode.h"
HARNESS = "src/gpu/d3d12_harness.h"

MUTANTS = {
    "XG1": ("A tied GPU device branch keeps the full finish under a bound schema (tied heads only)", "G6.3 (S-GROWS)", [
        (GPU, "} else if (base->config.tie_word_embeddings) {\n\t\th->legal_row_head = h->embed_weights.data();",
              "} else if (false && base->config.tie_word_embeddings) {\n\t\th->legal_row_head = h->embed_weights.data();"),
    ]),
    "XG2": ("The GPU pending token is re-embedded (a second row of it, not walked)", "G11.1", [
        (GPU, "\t// The headroom, on the host, before any submission",
              "\tif (has_pending && take > 0) {\n\t\trows.insert(rows.begin(), pending_token);\n\t\trows.pop_back();\n\t}\n"
              "\t// The headroom, on the host, before any submission"),
        (GPU, "/*prewalked_prefix=*/static_cast<uint32_t>(pend)",
              "/*prewalked_prefix=*/static_cast<uint32_t>(pend * 2)"),
    ]),
    "XG3": ("The GPU table is read from the wrong schema index", "G7.1", [
        (GPU, "& table = model->forced.table[si];",
              "& table = model->forced.table[(si + 1) % model->forced.table.size()];"),
        (GPU, "model->forced.table[si], static_cast<uint32_t>(si),",
              "model->forced.table[(si + 1) % model->forced.table.size()], static_cast<uint32_t>(si),"),
    ]),
    "XG8": ("The fused verb walks the pending token again", "G11.1, G12.1", [
        (GPU, "/*prewalked_prefix=*/static_cast<uint32_t>(pend)", "/*prewalked_prefix=*/0u"),
    ]),
    "XG9": ("A bound finish on a tied device-head model still runs the device logits", "G6.3 (count seam)", [
        (GPU, "\t\tfinish_rows = re - rb;\n",
              "\t\tfinish_rows = re - rb;\n\t\tif (model->device_logits) {\n"
              "\t\t\tconst SslmGpuStatus xg9 = RunDeviceLogits(ctx, *model->device_logits, final_codes.data(), "
              "wide_logits.data());\n\t\t\tif (xg9 != SSLM_OK) return xg9;\n\t\t\tfinish_on_device = true;\n\t\t}\n"),
    ]),
    "XG15": ("At the cap, the verb returns SSLM_DEVICE_LOST", "G12.6", [
        (GPU, "SSLM_GPU_KV_BUDGET_EXCEEDED : SSLM_GPU_CONTEXT_CAP_EXCEEDED;",
              "SSLM_GPU_KV_BUDGET_EXCEEDED : SSLM_DEVICE_LOST;"),
    ]),
    "XG19": ("A canonical device failure keeps the bridge's derived prefix (no restore)", "G12.7", [
        (GPU, "return fail(chunk_fault == superslm::SslmForwardStatus::GpuAllocationFailed",
              "return (chunk_fault == superslm::SslmForwardStatus::GpuAllocationFailed"),
    ]),
    "XG22": ("The harness's value 2 still enables GPU-based validation", "G0.1 (host half: cloud)", [
        (DBG, "return mode == 1;", "return mode >= 1;"),
    ]),
    "XG26": ("Value 2 never creates the info queue", "G0.1 (device half)", [
        (HARNESS, "\t\tif (debug_layer_enabled) {\n\t\t\tif (FAILED(dev.As(&debug_info_queue))) {",
                  "\t\tif (debug_layer_enabled && debug_layer_mode != 2) {\n\t\t\tif (FAILED(dev.As(&debug_info_queue))) {"),
    ]),
    "XG30": ("A canonical failure leaves the prefill snapshot cleared", "G12.7", [
        (GPU, "\tseq->prefill_hidden_valid = snap.prefill_valid;\n", ""),
    ]),
    "XG31": ("The walked embed leaves the ready flag set", "G12.2", [
        (GPU, "\tseq->dfa_walk_state = next_state;\n\tseq->ready_for_logits = false;\n\treturn SSLM_OK;\n}\n\n"
              "SslmGpuStatus sslm_gpu_seq_ready_for_logitsImpl",
              "\tseq->dfa_walk_state = next_state;\n\treturn SSLM_OK;\n}\n\n"
              "SslmGpuStatus sslm_gpu_seq_ready_for_logitsImpl"),
    ]),
    "XG37": ("The GPU verb accepts SSLM_JUMP_EXACT (and runs it as canonical)", "G12.9", [
        (GPU, "if (opts->mode == 1) return SSLM_SEQUENCE_REJECTED;",
              "if (false && opts->mode == 1) return SSLM_SEQUENCE_REJECTED;"),
    ]),
    "XGB1": ("A host bad_alloc in the verb's pass skips the entry restore (engine code review B1)", "G12.10", [
        (GPU, "\t} catch (const std::bad_alloc&) {\n\t\treturn fail(SSLM_GPU_ALLOCATION_FAILED);",
              "\t} catch (const std::bad_alloc&) {\n\t\tthrow;"),
    ]),
}


def read(repo, rel):
    with open(os.path.join(repo, rel), "r", encoding="utf-8", newline="") as f:
        return f.read()


def apply(repo, name):
    """Returns {rel: mutated text}; exits on a drifted anchor."""
    texts = {}
    for rel, old, new in MUTANTS[name][2]:
        text = texts.get(rel) if rel in texts else read(repo, rel)
        # The tree may be checked out with CRLF; anchors are written with LF.
        crlf = "\r\n" in text
        o = old.replace("\n", "\r\n") if crlf else old
        n = new.replace("\n", "\r\n") if crlf else new
        count = text.count(o)
        if count != 1:
            sys.exit("%s: anchor in %s occurs %d times, refusing: %r" % (name, rel, count, old[:70]))
        texts[rel] = text.replace(o, n, 1)
    return texts


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("mutant", nargs="?")
    ap.add_argument("--repo", default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv[1:])
    if a.list:
        for k, (what, cells, edits) in MUTANTS.items():
            print("%-5s %-45s killed by %s -- %s" % (k, ", ".join(sorted({e[0] for e in edits})), cells, what))
        return 0
    if a.check:
        for k in MUTANTS:
            apply(a.repo, k)
        print("make_sfdg_mutants: %d mutants, every anchor unique" % len(MUTANTS))
        return 0
    if a.mutant not in MUTANTS or not a.out:
        ap.error("give a mutant (%s) and --out, or --check / --list" % ", ".join(MUTANTS))
    for rel, text in apply(a.repo, a.mutant).items():
        dst = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print("wrote %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
